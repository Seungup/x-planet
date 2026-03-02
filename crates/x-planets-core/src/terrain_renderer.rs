//! GPU terrain renderer using terrain_tile.wgsl.
//!
//! Renders terrain tiles as displaced 3D meshes (17×17 vertex grid per tile).
//! Each tile's vertices are displaced by elevation data on the CPU, then
//! rendered with the imagery texture draped on top.
//!
//! Uses the same bind group layouts as `TileRenderer` (viewport + tile uniforms
//! + texture + sampler) so the shaders share a uniform interface.

use x_planets_gpu::GpuContext;
use x_planets_math::{TileCoord, ViewportUniforms};

use crate::pipeline::{build_terrain_mesh, compute_height_scale, tile_uniforms_with_uv, RenderableTile};
use crate::render::TerrainVertex;
use crate::viewport::Viewport;
use std::collections::HashMap;

const TERRAIN_TILE_SHADER: &str = include_str!("../../../shaders/rendering/terrain_tile.wgsl");

/// CPU-side elevation data for a terrain tile.
pub struct TerrainTileData {
    pub elevation: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

/// Per-layer terrain data assembled each frame for rendering.
pub struct TerrainLayerData<'a> {
    /// Layer name (for debug labels).
    pub name: &'a str,
    /// Layer opacity (0.0–1.0).
    pub opacity: f32,
    /// Tiles to render (with fallback resolution).
    pub tiles: Vec<RenderableTile>,
    /// Imagery texture views (draped onto the terrain mesh).
    pub imagery_views: HashMap<TileCoord, &'a wgpu::TextureView>,
    /// Elevation data per tile (used to build displaced mesh on CPU).
    pub elevation_data: HashMap<TileCoord, &'a TerrainTileData>,
    /// Per-tile opacity overrides (for fade-in animation).
    pub tile_opacity_overrides: HashMap<TileCoord, f32>,
}

/// A terrain tile prepared for rendering (owns its GPU resources).
struct PreparedTerrainTile {
    _uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

/// Renders terrain tiles with 3D displaced meshes.
pub struct TerrainRenderer {
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
    /// Elevation exaggeration factor (default: 1.5 for visual effect).
    pub exaggeration: f64,
}

impl TerrainRenderer {
    /// Create a new terrain renderer.
    ///
    /// Requires a GpuContext with a surface (panics if headless).
    pub fn new(gpu: &GpuContext) -> Self {
        let format = gpu
            .surface_format()
            .expect("TerrainRenderer requires a surface");

        // ── Bind group layout 0: viewport uniforms ──
        let _viewport_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("terrain-viewport-bgl"),
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
                    label: Some("terrain-tile-bgl"),
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
                label: Some("terrain-tile-shader"),
                source: wgpu::ShaderSource::Wgsl(TERRAIN_TILE_SHADER.into()),
            });

        let pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("terrain-tile-layout"),
                    bind_group_layouts: &[&_viewport_bgl, &tile_bgl],
                    push_constant_ranges: &[],
                });

        let pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("terrain-tile-pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[TerrainVertex::layout()],
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
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
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

        // ── Viewport uniform buffer ──
        let viewport_uniforms = ViewportUniforms {
            view_proj: [0.0; 16],
            resolution: [0.0; 4],
            camera: [0.0; 4],
        };
        let viewport_buffer =
            gpu.create_uniform_buffer("terrain-viewport-uniforms", &viewport_uniforms);

        let viewport_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-viewport-bg"),
            layout: &_viewport_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: viewport_buffer.as_entire_binding(),
            }],
        });

        // ── Sampler ──
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain-tile-sampler"),
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
        let depth_view =
            Self::create_depth_texture(&gpu.device, surface_width, surface_height, depth_format);

        log::info!(
            "TerrainRenderer created (format: {:?}, depth: {:?})",
            format,
            depth_format
        );

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
            exaggeration: 1.5,
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
            label: Some("terrain-depth-texture"),
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
            self.depth_view =
                Self::create_depth_texture(device, width, height, self.depth_format);
        }
    }

    /// Prepare a single terrain tile: build displaced mesh + create bind group.
    fn prepare_tile(
        &self,
        gpu: &GpuContext,
        coord: &TileCoord,
        texture_view: &wgpu::TextureView,
        opacity: f32,
        uv_rect: [f32; 4],
        elevation: &TerrainTileData,
    ) -> PreparedTerrainTile {
        let height_scale = compute_height_scale(self.exaggeration);

        // Build displaced mesh from elevation data
        let (vertices, indices) = build_terrain_mesh(
            coord,
            &elevation.elevation,
            elevation.width,
            elevation.height,
            height_scale,
        );

        let vertex_buffer = gpu.create_vertex_buffer(
            &format!("terrain-verts-{}-{}-{}", coord.z, coord.x, coord.y),
            &vertices,
        );
        let index_buffer = gpu.create_index_buffer(
            &format!("terrain-idx-{}-{}-{}", coord.z, coord.x, coord.y),
            &indices,
        );

        let uniforms = tile_uniforms_with_uv(coord, opacity, uv_rect);
        let uniform_buffer = gpu.create_uniform_buffer("terrain-tile-uniforms", &uniforms);

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-tile-bg"),
            layout: &self.tile_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
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

        PreparedTerrainTile {
            _uniform_buffer: uniform_buffer,
            bind_group,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }

    /// Render terrain layers to the target surface.
    ///
    /// Terrain layers use `LoadOp::Load` for color (preserves raster layers already drawn)
    /// and `LoadOp::Load` for depth (shares depth with raster layers for proper occlusion).
    pub fn render_terrain_layered(
        &self,
        gpu: &GpuContext,
        target: &wgpu::TextureView,
        viewport: &Viewport,
        layers: &[TerrainLayerData],
    ) {
        if layers.is_empty() {
            return;
        }

        // Update viewport uniforms
        let uniforms = viewport.to_uniforms();
        gpu.update_buffer(&self.viewport_buffer, &uniforms);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        for layer in layers {
            // Prepare tiles that have both imagery textures and elevation data
            let prepared: Vec<PreparedTerrainTile> = layer
                .tiles
                .iter()
                .filter_map(|rt| {
                    let tex_view = layer.imagery_views.get(&rt.texture_coord)?;
                    let elev = layer.elevation_data.get(&rt.coord)?;
                    let tile_opacity = layer
                        .tile_opacity_overrides
                        .get(&rt.coord)
                        .copied()
                        .unwrap_or(layer.opacity);
                    Some(self.prepare_tile(gpu, &rt.coord, tex_view, tile_opacity, rt.uv_rect, elev))
                })
                .collect();

            if prepared.is_empty() {
                continue;
            }

            // Render pass: Load color (preserve raster), Load depth (share depth buffer)
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("terrain-render-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
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

                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.viewport_bg, &[]);

                for tile in &prepared {
                    pass.set_bind_group(1, &tile.bind_group, &[]);
                    pass.set_vertex_buffer(0, tile.vertex_buffer.slice(..));
                    pass.set_index_buffer(tile.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..tile.index_count, 0, 0..1);
                }
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}
