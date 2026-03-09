//! GPU tile renderer using raster_tile.wgsl.
//!
//! Karpathy step: "One pipeline, one draw call per tile, correct on screen."
//!
//! This renderer:
//! - Creates the render pipeline from raster_tile.wgsl
//! - Uses shared viewport uniforms (group 0) from SharedRenderResources
//! - Creates per-tile bind groups (group 1: uniforms + texture + sampler)
//! - Batches tile quads into a single vertex/index buffer

use x_planets_gpu::GpuContext;
use x_planets_math::TileCoord;

use crate::pipeline::{
    build_globe_tile_mesh, build_polar_caps, tile_uniforms_for_globe,
    build_centered_tile_mesh, tile_uniforms_for_centered,
    tile_passes_angular_filter,
    RenderableTile,
};
use crate::render::{GlobeTileVertex, RenderLayerData};
use crate::shared_render_resources::SharedRenderResources;
use crate::viewport::Viewport;
use std::collections::HashMap;

const RASTER_TILE_GLOBE_SHADER: &str = include_str!("../../../shaders/rendering/raster_tile_globe.wgsl");

/// A tile prepared for rendering (owns its GPU resources).
struct PreparedTile {
    _buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// A layer's geometry + bind groups, ready to be drawn.
struct PreparedLayer {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    tiles: Vec<PreparedTile>,
    tile_idx_counts: Vec<u32>,
    use_globe_pipeline: bool,
}

/// Cached CPU-side mesh data to avoid recomputing trigonometric mesh every frame.
struct CachedTileMesh {
    /// Raw vertex data (GlobeTileVertex array as bytes).
    verts: Vec<GlobeTileVertex>,
    /// Index data.
    idxs: Vec<u32>,
    tile_idx_counts: Vec<u32>,
    /// Cache key: sorted tile coords that produced this mesh.
    tile_coords: Vec<TileCoord>,
    /// For centered mode: the quantized camera center used to build the mesh.
    center_lat_lon: Option<(i64, i64)>,
    is_globe: bool,
}

/// Quantize f64 radians to ~0.0001° precision for cache key comparison.
fn quantize_radians(rad: f64) -> i64 {
    (rad * 1_000_000.0) as i64
}

/// Renders raster tiles to the screen.
pub struct TileRenderer {
    globe_pipeline: wgpu::RenderPipeline,
    centered_pipeline: wgpu::RenderPipeline,
    /// 1×1 white texture for polar caps and fallback rendering.
    polar_cap_texture_view: wgpu::TextureView,
    /// Cached polar cap vertex/index buffers (never change after creation).
    cached_polar_caps: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    /// Cached mesh geometry per layer name.
    cached_meshes: HashMap<String, CachedTileMesh>,
}

impl TileRenderer {
    /// Create a new tile renderer.
    ///
    /// Requires a GpuContext with a surface (panics if headless).
    pub fn new(gpu: &GpuContext, shared: &SharedRenderResources) -> Self {
        let format = gpu
            .surface_format()
            .expect("TileRenderer requires a surface");

        let depth_format = SharedRenderResources::depth_format();

        let globe_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("raster-tile-globe-shader"),
                source: wgpu::ShaderSource::Wgsl(RASTER_TILE_GLOBE_SHADER.into()),
            });

        let pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("raster-tile-layout"),
                    bind_group_layouts: &[&shared.viewport_bgl, &shared.tile_bgl],
                    push_constant_ranges: &[],
                });

        let make_depth_stencil = || wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let color_targets = [Some(wgpu::ColorTargetState {
            format,
            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let vertex_buffers = [GlobeTileVertex::layout()];

        let globe_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("raster-tile-globe-pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &globe_shader,
                        entry_point: Some("vs_main"),
                        buffers: &vertex_buffers,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &globe_shader,
                        entry_point: Some("fs_main"),
                        targets: &color_targets,
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        ..Default::default()
                    },
                    depth_stencil: Some(make_depth_stencil()),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });

        let centered_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("raster-tile-centered-pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &globe_shader,
                        entry_point: Some("vs_main"),
                        buffers: &vertex_buffers,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &globe_shader,
                        entry_point: Some("fs_main"),
                        targets: &color_targets,
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(make_depth_stencil()),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
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
        gpu.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &polar_cap_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0xE8, 0xEE, 0xF2, 0xFF],
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let polar_cap_texture_view = polar_cap_texture.create_view(&wgpu::TextureViewDescriptor::default());

        log::info!("TileRenderer created (format: {:?})", format);

        Self {
            globe_pipeline,
            centered_pipeline,
            polar_cap_texture_view,
            cached_polar_caps: None,
            cached_meshes: HashMap::new(),
        }
    }

    /// Create a bind group for a single tile.
    fn prepare_tile(
        gpu: &GpuContext,
        shared: &SharedRenderResources,
        uniforms: &x_planets_math::TileUniforms,
        texture_view: &wgpu::TextureView,
    ) -> PreparedTile {
        let buffer = gpu.create_uniform_buffer("tile-uniforms", uniforms);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile-bg"),
            layout: &shared.tile_bgl,
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
                    resource: wgpu::BindingResource::Sampler(&shared.sampler),
                },
            ],
        });
        PreparedTile { _buffer: buffer, bind_group }
    }

    /// Prepare a layer: filter tiles, build mesh (with caching), create bind groups.
    ///
    /// Mesh geometry (vertex/index data) is cached per layer. On cache hit,
    /// the expensive trigonometric mesh computation is skipped and cached CPU
    /// data is re-uploaded to fresh GPU buffers.
    fn prepare_layer(
        &mut self,
        gpu: &GpuContext,
        shared: &SharedRenderResources,
        viewport: &Viewport,
        layer: &RenderLayerData,
        mode: x_planets_math::ProjectionMode,
        vp_f64: &glam::DMat4,
    ) -> Option<PreparedLayer> {
        let is_globe = mode == x_planets_math::ProjectionMode::Globe;
        let center_lat_rad = viewport.center.lat.to_radians();
        let center_lon_rad = viewport.center.lon.to_radians();

        // Filter tiles with available textures (+ angular filter for centered)
        let renderable_tiles: Vec<&RenderableTile> = layer
            .tiles
            .iter()
            .filter(|rt| {
                if !layer.texture_views.contains_key(&rt.texture_coord) {
                    return false;
                }
                if !is_globe {
                    tile_passes_angular_filter(rt, center_lat_rad, center_lon_rad, viewport.zoom)
                } else {
                    true
                }
            })
            .collect();

        if renderable_tiles.is_empty() {
            return None;
        }

        // Build sorted coord list for cache key comparison
        let mut sorted_coords: Vec<TileCoord> =
            renderable_tiles.iter().map(|rt| rt.coord).collect();
        sorted_coords.sort_by(|a, b| {
            a.z.cmp(&b.z)
                .then(a.x.cmp(&b.x))
                .then(a.y.cmp(&b.y))
        });

        let center_key = if is_globe {
            None
        } else {
            Some((quantize_radians(center_lat_rad), quantize_radians(center_lon_rad)))
        };

        // Check mesh cache: skip CPU mesh computation if tile set + center unchanged
        let cache_hit = self.cached_meshes.get(layer.name).map_or(false, |cached| {
            cached.is_globe == is_globe
                && cached.tile_coords == sorted_coords
                && cached.center_lat_lon == center_key
        });

        let (verts, idxs, tile_idx_counts) = if cache_hit {
            let cached = self.cached_meshes.get(layer.name).unwrap();
            (cached.verts.clone(), cached.idxs.clone(), cached.tile_idx_counts.clone())
        } else {
            let renderable_refs: Vec<RenderableTile> =
                renderable_tiles.iter().map(|rt| (*rt).clone()).collect();
            let (v, i, c) = if is_globe {
                build_globe_tile_mesh(&renderable_refs)
            } else {
                build_centered_tile_mesh(&renderable_refs, center_lat_rad, center_lon_rad)
            };

            if v.is_empty() {
                return None;
            }

            // Update cache with computed mesh data
            self.cached_meshes.insert(
                layer.name.to_string(),
                CachedTileMesh {
                    verts: v.clone(),
                    idxs: i.clone(),
                    tile_idx_counts: c.clone(),
                    tile_coords: sorted_coords,
                    center_lat_lon: center_key,
                    is_globe,
                },
            );

            (v, i, c)
        };

        let vertex_buffer = gpu.create_vertex_buffer("tile-vertices", &verts);
        let index_buffer = gpu.create_index_buffer("tile-indices", &idxs);

        // Prepare per-tile bind groups (projection-specific uniforms — always rebuilt)
        let tiles: Vec<PreparedTile> = renderable_tiles
            .iter()
            .filter_map(|rt| {
                let tex_view = layer.texture_views.get(&rt.texture_coord)?;
                let tile_opacity = layer
                    .tile_opacity_overrides
                    .get(&rt.coord)
                    .copied()
                    .unwrap_or(layer.opacity);
                let uniforms = if is_globe {
                    tile_uniforms_for_globe(rt, tile_opacity, vp_f64)
                } else {
                    tile_uniforms_for_centered(rt, tile_opacity, vp_f64, center_lat_rad, center_lon_rad)
                };
                Some(Self::prepare_tile(gpu, shared, &uniforms, tex_view))
            })
            .collect();

        Some(PreparedLayer {
            vertex_buffer,
            index_buffer,
            tiles,
            tile_idx_counts,
            use_globe_pipeline: is_globe,
        })
    }

    /// Render a single layer of tiles (backward-compatible convenience wrapper).
    pub fn render_frame(
        &mut self,
        gpu: &GpuContext,
        shared: &SharedRenderResources,
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
        self.render_frame_layered(gpu, shared, target, viewport, &[single]);
    }

    /// Render multiple layers to the target surface.
    pub fn render_frame_layered(
        &mut self,
        gpu: &GpuContext,
        shared: &SharedRenderResources,
        target: &wgpu::TextureView,
        viewport: &Viewport,
        layers: &[RenderLayerData],
    ) {
        self.render_frame_layered_projected(
            gpu,
            shared,
            target,
            viewport,
            layers,
            x_planets_math::ProjectionMode::Mercator,
        );
    }

    /// Emit a clear pass for the first layer.
    fn emit_clear_pass(
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        shared: &SharedRenderResources,
    ) {
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
                view: &shared.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
    }

    /// Like [`render_frame_layered`] but uses the given projection for tile positioning.
    pub fn render_frame_layered_projected(
        &mut self,
        gpu: &GpuContext,
        shared: &SharedRenderResources,
        target: &wgpu::TextureView,
        viewport: &Viewport,
        layers: &[RenderLayerData],
        mode: x_planets_math::ProjectionMode,
    ) {
        if layers.is_empty() {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            Self::emit_clear_pass(&mut encoder, target, shared);
            gpu.queue.submit(Some(encoder.finish()));
            return;
        }

        // Update viewport uniforms once per frame
        let uniforms = viewport.to_uniforms();
        shared.update_viewport(gpu, &uniforms);

        let vp_f64 = viewport.to_view_proj_f64_projected(mode);
        let is_globe = mode == x_planets_math::ProjectionMode::Globe;

        // Prepare polar caps (globe only, cached)
        let polar_cap_data = if is_globe {
            if self.cached_polar_caps.is_none() {
                let (cap_verts, cap_idxs) = build_polar_caps();
                let count = cap_idxs.len() as u32;
                let vb = gpu.create_vertex_buffer("polar-cap-vertices", &cap_verts);
                let ib = gpu.create_index_buffer("polar-cap-indices", &cap_idxs);
                self.cached_polar_caps = Some((vb, ib, count));
            }

            let cap_mvp = vp_f64.as_mat4();
            let cap_uniforms = x_planets_math::TileUniforms {
                mvp: cap_mvp.to_cols_array(),
                bounds: [0.0; 4],
                meta: [0.0, 1.0, 0.0, 0.0],
                uv_rect: [0.0, 0.0, 1.0, 1.0],
            };
            Some(Self::prepare_tile(gpu, shared, &cap_uniforms, &self.polar_cap_texture_view))
        } else {
            None
        };

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        for (layer_idx, layer) in layers.iter().enumerate() {
            let is_first = layer_idx == 0;

            if layer.tiles.is_empty() {
                if is_first {
                    Self::emit_clear_pass(&mut encoder, target, shared);
                }
                continue;
            }

            let prepared = match self.prepare_layer(gpu, shared, viewport, layer, mode, &vp_f64) {
                Some(p) => p,
                None => {
                    if is_first {
                        Self::emit_clear_pass(&mut encoder, target, shared);
                    }
                    continue;
                }
            };

            let color_load = if is_first {
                wgpu::LoadOp::Clear(wgpu::Color { r: 0.08, g: 0.12, b: 0.18, a: 1.0 })
            } else {
                wgpu::LoadOp::Load
            };

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
                        view: &shared.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                let pipeline = if prepared.use_globe_pipeline {
                    &self.globe_pipeline
                } else {
                    &self.centered_pipeline
                };
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &shared.viewport_bg, &[]);
                pass.set_vertex_buffer(0, prepared.vertex_buffer.slice(..));
                pass.set_index_buffer(prepared.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                let mut idx_offset = 0u32;
                for (i, tile) in prepared.tiles.iter().enumerate() {
                    pass.set_bind_group(1, &tile.bind_group, &[]);
                    let count = prepared.tile_idx_counts[i];
                    pass.draw_indexed(idx_offset..idx_offset + count, 0, 0..1);
                    idx_offset += count;
                }

                // Globe: draw polar caps at the end
                if let (Some(polar), Some((cap_vb, cap_ib, cap_count))) =
                    (&polar_cap_data, self.cached_polar_caps.as_ref())
                {
                    pass.set_vertex_buffer(0, cap_vb.slice(..));
                    pass.set_index_buffer(cap_ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.set_bind_group(1, &polar.bind_group, &[]);
                    pass.draw_indexed(0..*cap_count, 0, 0..1);
                }
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}
