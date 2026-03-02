//! GPU tile renderer using raster_tile.wgsl.
//!
//! Karpathy step: "One pipeline, one draw call per tile, correct on screen."
//!
//! This renderer:
//! - Creates the render pipeline from raster_tile.wgsl
//! - Manages viewport uniform buffer (group 0)
//! - Creates per-tile bind groups (group 1: uniforms + texture + sampler)
//! - Batches tile quads into a single vertex/index buffer
//!
//! What it does NOT do (yet):
//! - Texture atlas (each tile gets its own bind group)
//! - GPU-side projection (uses CPU view_proj matrix)
//! - LOD / placeholder tiles

use x_planets_gpu::GpuContext;
use x_planets_math::{TileCoord, ViewportUniforms};

use crate::pipeline::{build_tile_mesh, tile_uniforms_with_uv, RenderableTile};
use crate::render::{RenderLayerData, TileVertex};
use crate::viewport::Viewport;
use std::collections::HashMap;

const RASTER_TILE_SHADER: &str = include_str!("../../../shaders/rendering/raster_tile.wgsl");

/// A tile prepared for rendering (owns its GPU resources).
pub struct PreparedTile {
    _buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Renders raster tiles to the screen.
pub struct TileRenderer {
    pipeline: wgpu::RenderPipeline,
    _viewport_bgl: wgpu::BindGroupLayout,
    tile_bgl: wgpu::BindGroupLayout,
    viewport_buffer: wgpu::Buffer,
    viewport_bg: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    depth_view: wgpu::TextureView,
    depth_format: wgpu::TextureFormat,
    surface_width: u32,
    surface_height: u32,
}

impl TileRenderer {
    /// Create a new tile renderer.
    ///
    /// Requires a GpuContext with a surface (panics if headless).
    pub fn new(gpu: &GpuContext) -> Self {
        let format = gpu
            .surface_format()
            .expect("TileRenderer requires a surface");

        // ── Bind group layout 0: viewport uniforms ──
        let _viewport_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("viewport-bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        // ── Bind group layout 1: tile uniforms + texture + sampler ──
        let tile_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("tile-bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX
                                | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float {
                                    filterable: true,
                                },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(
                                wgpu::SamplerBindingType::Filtering,
                            ),
                            count: None,
                        },
                    ],
                });

        // ── Shader + Pipeline ──
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("raster-tile-shader"),
                source: wgpu::ShaderSource::Wgsl(RASTER_TILE_SHADER.into()),
            });

        let pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("raster-tile-layout"),
                    bind_group_layouts: &[&_viewport_bgl, &tile_bgl],
                    push_constant_ranges: &[],
                });

        let pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("raster-tile-pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[TileVertex::layout()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::LessEqual,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });

        // ── Viewport uniform buffer (updated per frame) ──
        let viewport_uniforms = ViewportUniforms {
            view_proj: [0.0; 16],
            resolution: [0.0; 4],
            camera: [0.0; 4],
        };
        let viewport_buffer =
            gpu.create_uniform_buffer("viewport-uniforms", &viewport_uniforms);

        let viewport_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport-bg"),
            layout: &_viewport_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: viewport_buffer.as_entire_binding(),
            }],
        });

        // ── Sampler ──
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tile-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ── Depth texture ──
        let (surface_width, surface_height) = gpu
            .surface
            .as_ref()
            .map(|s| (s.config.width, s.config.height))
            .unwrap_or((800, 600));
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let depth_view = Self::create_depth_texture(&gpu.device, surface_width, surface_height, depth_format);

        log::info!("TileRenderer created (format: {:?}, depth: {:?})", format, depth_format);

        Self {
            pipeline,
            _viewport_bgl,
            tile_bgl,
            viewport_buffer,
            viewport_bg,
            sampler,
            depth_view,
            depth_format,
            surface_width,
            surface_height,
        }
    }

    /// Create a depth texture and return its view.
    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth-texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Recreate depth texture after window resize.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width != self.surface_width || height != self.surface_height {
            self.surface_width = width;
            self.surface_height = height;
            self.depth_view = Self::create_depth_texture(device, width, height, self.depth_format);
        }
    }

    /// Prepare a tile for rendering: create uniform buffer + bind group.
    fn prepare_tile(
        &self,
        gpu: &GpuContext,
        coord: &TileCoord,
        texture_view: &wgpu::TextureView,
        opacity: f32,
        uv_rect: [f32; 4],
        vp_f64: &glam::DMat4,
    ) -> PreparedTile {
        let uniforms = tile_uniforms_with_uv(coord, opacity, uv_rect, vp_f64);
        let buffer = gpu.create_uniform_buffer("tile-uniforms", &uniforms);

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile-bg"),
            layout: &self.tile_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        PreparedTile {
            _buffer: buffer,
            bind_group,
        }
    }

    /// Render a single layer of tiles (backward-compatible convenience wrapper).
    ///
    /// `tiles` — renderable tiles (with fallback resolution).
    /// `texture_views` — map from TileCoord → GPU TextureView.
    pub fn render_frame(
        &self,
        gpu: &GpuContext,
        target: &wgpu::TextureView,
        viewport: &Viewport,
        tiles: &[RenderableTile],
        texture_views: &HashMap<TileCoord, &wgpu::TextureView>,
    ) {
        let single = RenderLayerData {
            name: "base",
            opacity: 1.0,
            tiles: tiles.to_vec(),
            texture_views: texture_views.clone(),
            tile_opacity_overrides: HashMap::new(),
        };
        self.render_frame_layered(gpu, target, viewport, &[single]);
    }

    /// Render multiple layers to the target surface.
    ///
    /// Layers are drawn bottom-to-top (the caller should pass them in z-order).
    /// Each layer gets its own render pass:
    /// - First layer: `Clear` color + depth
    /// - Subsequent layers: `Load` color + `Clear` depth (avoids cross-layer z-fighting)
    pub fn render_frame_layered(
        &self,
        gpu: &GpuContext,
        target: &wgpu::TextureView,
        viewport: &Viewport,
        layers: &[RenderLayerData],
    ) {
        if layers.is_empty() {
            return;
        }

        // 1. Update viewport uniforms (shared across all layers)
        let uniforms = viewport.to_uniforms();
        gpu.update_buffer(&self.viewport_buffer, &uniforms);

        // Compute f64 VP for per-tile MVP (eliminates high-zoom jitter)
        let vp_f64 = viewport.to_view_proj_f64();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        for (layer_idx, layer) in layers.iter().enumerate() {
            if layer.tiles.is_empty() {
                // Still need the first layer to clear, even if empty
                if layer_idx == 0 {
                    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("tile-clear-pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: target,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.08, g: 0.12, b: 0.18, a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Discard,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    // pass drops → ends
                }
                continue;
            }

            // 2. Build batched vertex/index mesh for this layer (RTE positions)
            let coords: Vec<TileCoord> = layer.tiles.iter().map(|t| t.coord).collect();
            let (vertices, indices) = build_tile_mesh(&coords);

            if vertices.is_empty() {
                continue;
            }

            let vertex_buffer = gpu.create_vertex_buffer(
                &format!("tile-vertices-{}", layer.name),
                &vertices,
            );
            let index_buffer = gpu.create_index_buffer(
                &format!("tile-indices-{}", layer.name),
                &indices,
            );

            // 3. Prepare per-tile bind groups (per-tile MVP + opacity)
            let prepared: Vec<PreparedTile> = layer
                .tiles
                .iter()
                .filter_map(|rt| {
                    layer.texture_views.get(&rt.texture_coord).map(|tex_view| {
                        let tile_opacity = layer
                            .tile_opacity_overrides
                            .get(&rt.coord)
                            .copied()
                            .unwrap_or(layer.opacity);
                        self.prepare_tile(gpu, &rt.coord, tex_view, tile_opacity, rt.uv_rect, &vp_f64)
                    })
                })
                .collect();

            // 4. Encode render pass for this layer
            let is_first = layer_idx == 0;
            let color_load = if is_first {
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.08, g: 0.12, b: 0.18, a: 1.0,
                })
            } else {
                wgpu::LoadOp::Load
            };
            // Always clear depth per layer to avoid cross-layer z-fighting
            let depth_load = wgpu::LoadOp::Clear(1.0);

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("tile-render-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: color_load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: depth_load,
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.viewport_bg, &[]);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                for (i, tile) in prepared.iter().enumerate() {
                    pass.set_bind_group(1, &tile.bind_group, &[]);
                    let start = (i * 6) as u32;
                    pass.draw_indexed(start..start + 6, 0, 0..1);
                }
            }
        }

        // 5. Submit all passes at once
        gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}
