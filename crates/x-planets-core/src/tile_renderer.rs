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

use crate::pipeline::{
    build_globe_tile_mesh, build_polar_caps, tile_uniforms_for_globe,
    build_centered_tile_mesh, tile_uniforms_for_centered,
    tile_passes_angular_filter,
    RenderableTile,
};
use crate::render::{GlobeTileVertex, RenderLayerData};
use crate::viewport::Viewport;
use std::collections::HashMap;

const RASTER_TILE_GLOBE_SHADER: &str = include_str!("../../../shaders/rendering/raster_tile_globe.wgsl");

/// A tile prepared for rendering (owns its GPU resources).
pub struct PreparedTile {
    _buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// Renders raster tiles to the screen.
pub struct TileRenderer {
    globe_pipeline: wgpu::RenderPipeline,
    centered_pipeline: wgpu::RenderPipeline,
    _viewport_bgl: wgpu::BindGroupLayout,
    tile_bgl: wgpu::BindGroupLayout,
    viewport_buffer: wgpu::Buffer,
    viewport_bg: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    /// 1×1 white texture for polar caps and fallback rendering.
    polar_cap_texture_view: wgpu::TextureView,
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

        // ── Globe Shader + Pipeline ──
        let globe_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("raster-tile-globe-shader"),
                source: wgpu::ShaderSource::Wgsl(RASTER_TILE_GLOBE_SHADER.into()),
            });

        let globe_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("raster-tile-globe-layout"),
                    bind_group_layouts: &[&_viewport_bgl, &tile_bgl],
                    push_constant_ranges: &[],
                });

        let globe_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("raster-tile-globe-pipeline"),
                    layout: Some(&globe_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &globe_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[GlobeTileVertex::layout()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &globe_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        // Globe meshes use CCW winding from outside the sphere.
                        // Back-face culling hides the far hemisphere.
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: Self::depth_format(),
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::LessEqual,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });

        // ── Centered (oblique Mercator) pipeline: same shader as globe (vec3)
        //    but no back-face culling (flat z=0 meshes).
        let centered_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("raster-tile-centered-pipeline"),
                    layout: Some(&globe_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &globe_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[GlobeTileVertex::layout()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &globe_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(), // no back-face culling
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: Self::depth_format(),
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

        // ── 1×1 white texture for polar caps ──
        let polar_cap_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("polar-cap-texture"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Ice-white color: #E8EEF2
        gpu.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &polar_cap_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0xE8, 0xEE, 0xF2, 0xFF],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let polar_cap_texture_view = polar_cap_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // ── Depth texture ──
        let (surface_width, surface_height) = gpu
            .surface
            .as_ref()
            .map(|s| (s.config.width, s.config.height))
            .unwrap_or((800, 600));
        let depth_format = Self::depth_format();
        let depth_view = Self::create_depth_texture(&gpu.device, surface_width, surface_height, depth_format);

        log::info!("TileRenderer created (format: {:?}, depth: {:?})", format, depth_format);

        Self {
            globe_pipeline,
            centered_pipeline,
            _viewport_bgl,
            tile_bgl,
            viewport_buffer,
            viewport_bg,
            sampler,
            polar_cap_texture_view,
            depth_view,
            depth_format,
            surface_width,
            surface_height,
        }
    }

    /// Platform-appropriate depth format.
    /// Depth24Plus is safer on WebGL2 fallback; Depth32Float on native.
    fn depth_format() -> wgpu::TextureFormat {
        #[cfg(target_arch = "wasm32")]
        { wgpu::TextureFormat::Depth24Plus }
        #[cfg(not(target_arch = "wasm32"))]
        { wgpu::TextureFormat::Depth32Float }
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

    /// Prepare a tile for globe rendering: uses 3D sphere uniforms.
    fn prepare_tile_globe(
        &self,
        gpu: &GpuContext,
        rt: &RenderableTile,
        texture_view: &wgpu::TextureView,
        opacity: f32,
        vp_f64: &glam::DMat4,
    ) -> PreparedTile {
        let uniforms = tile_uniforms_for_globe(rt, opacity, vp_f64);
        let buffer = gpu.create_uniform_buffer("globe-tile-uniforms", &uniforms);

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globe-tile-bg"),
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

    /// Prepare a tile for centered Mercator rendering: uses oblique Mercator uniforms.
    fn prepare_tile_centered(
        &self,
        gpu: &GpuContext,
        rt: &RenderableTile,
        texture_view: &wgpu::TextureView,
        opacity: f32,
        vp_f64: &glam::DMat4,
        center_lat_rad: f64,
        center_lon_rad: f64,
    ) -> PreparedTile {
        let uniforms =
            tile_uniforms_for_centered(rt, opacity, vp_f64, center_lat_rad, center_lon_rad);
        let buffer = gpu.create_uniform_buffer("centered-tile-uniforms", &uniforms);

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("centered-tile-bg"),
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
        self.render_frame_layered_projected(
            gpu,
            target,
            viewport,
            layers,
            x_planets_math::ProjectionMode::Mercator,
        );
    }

    /// Like [`render_frame_layered`] but uses the given projection for tile positioning.
    pub fn render_frame_layered_projected(
        &self,
        gpu: &GpuContext,
        target: &wgpu::TextureView,
        viewport: &Viewport,
        layers: &[RenderLayerData],
        mode: x_planets_math::ProjectionMode,
    ) {
        if layers.is_empty() {
            // No raster layers to draw, but we still need to clear the surface
            // so subsequent passes (terrain, 3D tiles) compositing with
            // LoadOp::Load don't read undefined/stale data.
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("tile-clear-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.08,
                                g: 0.12,
                                b: 0.18,
                                a: 1.0,
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
            }
            gpu.queue.submit(Some(encoder.finish()));
            return;
        }

        // 1. Update viewport uniforms (shared across all layers)
        let uniforms = viewport.to_uniforms();
        gpu.update_buffer(&self.viewport_buffer, &uniforms);

        // Compute f64 VP for per-tile MVP (eliminates high-zoom jitter)
        let vp_f64 = viewport.to_view_proj_f64_projected(mode);

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

            // 2. Encode render pass for this layer
            let is_first = layer_idx == 0;
            let color_load = if is_first {
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.08, g: 0.12, b: 0.18, a: 1.0,
                })
            } else {
                wgpu::LoadOp::Load
            };
            let depth_load = wgpu::LoadOp::Clear(1.0);

            let is_globe = mode == x_planets_math::ProjectionMode::Globe;

            if is_globe {
                // ── Globe path: tessellated sphere mesh ──
                // Filter to tiles with available textures BEFORE building
                // the mesh so that mesh indices and bind groups stay aligned.
                let renderable_tiles: Vec<&RenderableTile> = layer
                    .tiles
                    .iter()
                    .filter(|rt| layer.texture_views.contains_key(&rt.texture_coord))
                    .collect();

                let renderable_refs: Vec<RenderableTile> =
                    renderable_tiles.iter().map(|rt| (*rt).clone()).collect();
                let (globe_verts, globe_idxs, tile_idx_counts) =
                    build_globe_tile_mesh(&renderable_refs);

                if globe_verts.is_empty() {
                    continue;
                }

                let vertex_buffer = gpu.create_vertex_buffer(
                    &format!("globe-vertices-{}", layer.name),
                    &globe_verts,
                );
                let index_buffer = gpu.create_index_buffer(
                    &format!("globe-indices-{}", layer.name),
                    &globe_idxs,
                );

                let prepared: Vec<PreparedTile> = renderable_tiles
                    .iter()
                    .map(|rt| {
                        let tex_view = layer.texture_views.get(&rt.texture_coord).unwrap();
                        let tile_opacity = layer
                            .tile_opacity_overrides
                            .get(&rt.coord)
                            .copied()
                            .unwrap_or(layer.opacity);
                        self.prepare_tile_globe(gpu, rt, tex_view, tile_opacity, &vp_f64)
                    })
                    .collect();

                // ── Polar cap buffers ──
                let (cap_verts, cap_idxs) = build_polar_caps();
                let cap_idx_count = cap_idxs.len() as u32;
                let cap_vb = gpu.create_vertex_buffer("polar-cap-vertices", &cap_verts);
                let cap_ib = gpu.create_index_buffer("polar-cap-indices", &cap_idxs);
                // Cap uses VP directly (no per-tile model translation)
                let cap_mvp = vp_f64.as_mat4();
                let cap_uniforms = x_planets_math::TileUniforms {
                    mvp: cap_mvp.to_cols_array(),
                    bounds: [0.0; 4],
                    meta: [0.0, 1.0, 0.0, 0.0], // zoom=0, opacity=1
                    uv_rect: [0.0, 0.0, 1.0, 1.0],
                };
                let cap_uniform_buf = gpu.create_uniform_buffer("polar-cap-uniforms", &cap_uniforms);
                let cap_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("polar-cap-bg"),
                    layout: &self.tile_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: cap_uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.polar_cap_texture_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                });

                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("globe-render-pass"),
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

                    pass.set_pipeline(&self.globe_pipeline);
                    pass.set_bind_group(0, &self.viewport_bg, &[]);
                    pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                    let mut idx_offset = 0u32;
                    for (i, tile) in prepared.iter().enumerate() {
                        pass.set_bind_group(1, &tile.bind_group, &[]);
                        let count = tile_idx_counts[i];
                        pass.draw_indexed(idx_offset..idx_offset + count, 0, 0..1);
                        idx_offset += count;
                    }

                    // ── Polar caps (fill holes at ±85.05° to ±90°) ──
                    pass.set_vertex_buffer(0, cap_vb.slice(..));
                    pass.set_index_buffer(cap_ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.set_bind_group(1, &cap_bg, &[]);
                    pass.draw_indexed(0..cap_idx_count, 0, 0..1);
                }
            } else {
                // ── Centered Mercator path: tessellated oblique Mercator mesh ──
                // Tiles are re-projected through oblique Mercator centered on
                // the viewport center, minimising distortion near the view.
                let center_lat_rad = viewport.center.lat.to_radians();
                let center_lon_rad = viewport.center.lon.to_radians();

                // Filter to tiles with available textures AND within the
                // valid oblique Mercator range BEFORE building the mesh so
                // that mesh indices and bind groups stay aligned.
                // The oblique Mercator has a singularity at ~90° from the
                // center; skip tiles whose angular distance exceeds 80° to
                // prevent extreme distortion (the V-shape artifact).
                let renderable_tiles: Vec<&RenderableTile> = layer
                    .tiles
                    .iter()
                    .filter(|rt| {
                        if !layer.texture_views.contains_key(&rt.texture_coord) {
                            return false;
                        }
                        tile_passes_angular_filter(
                            rt,
                            center_lat_rad,
                            center_lon_rad,
                            viewport.zoom,
                        )
                    })
                    .collect();

                let renderable_refs: Vec<RenderableTile> =
                    renderable_tiles.iter().map(|rt| (*rt).clone()).collect();
                let (centered_verts, centered_idxs, tile_idx_counts) =
                    build_centered_tile_mesh(&renderable_refs, center_lat_rad, center_lon_rad);

                if centered_verts.is_empty() {
                    continue;
                }

                let vertex_buffer = gpu.create_vertex_buffer(
                    &format!("centered-vertices-{}", layer.name),
                    &centered_verts,
                );
                let index_buffer = gpu.create_index_buffer(
                    &format!("centered-indices-{}", layer.name),
                    &centered_idxs,
                );

                let prepared: Vec<PreparedTile> = renderable_tiles
                    .iter()
                    .map(|rt| {
                        let tex_view = layer.texture_views.get(&rt.texture_coord).unwrap();
                        let tile_opacity = layer
                            .tile_opacity_overrides
                            .get(&rt.coord)
                            .copied()
                            .unwrap_or(layer.opacity);
                        self.prepare_tile_centered(
                            gpu,
                            rt,
                            tex_view,
                            tile_opacity,
                            &vp_f64,
                            center_lat_rad,
                            center_lon_rad,
                        )
                    })
                    .collect();

                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("centered-render-pass"),
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

                    pass.set_pipeline(&self.centered_pipeline);
                    pass.set_bind_group(0, &self.viewport_bg, &[]);
                    pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                    let mut idx_offset = 0u32;
                    for (i, tile) in prepared.iter().enumerate() {
                        pass.set_bind_group(1, &tile.bind_group, &[]);
                        let count = tile_idx_counts[i];
                        pass.draw_indexed(idx_offset..idx_offset + count, 0, 0..1);
                        idx_offset += count;
                    }
                }
            }
        }

        // 5. Submit all passes at once
        gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}
