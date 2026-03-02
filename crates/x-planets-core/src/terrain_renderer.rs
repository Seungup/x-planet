//! GPU terrain renderer using terrain_tile.wgsl.
//!
//! Renders terrain tiles as displaced 3D meshes (33×33 vertex grid per tile).
//! Each tile's vertices are displaced by elevation data on the CPU, then
//! rendered with the imagery texture draped on top.
//!
//! **Mesh caching**: vertex/index buffers are built once per tile and cached
//! until the elevation source changes or exaggeration is adjusted.
//! Only the per-tile uniform buffer + bind group are recreated each frame
//! (they depend on the camera's VP matrix).
//!
//! Uses the same bind group layouts as `TileRenderer` (viewport + tile uniforms
//! + texture + sampler) so the shaders share a uniform interface.

use x_planets_gpu::GpuContext;
use x_planets_math::{TileCoord, TileUniforms, ViewportUniforms};

use crate::pipeline::{build_terrain_mesh, compute_height_scale, fallback_uv_rect, tile_uniforms_with_uv, RenderableTile};
use crate::render::TerrainVertex;
use crate::viewport::Viewport;
use std::collections::{HashMap, HashSet};

const TERRAIN_TILE_SHADER: &str = include_str!("../../../shaders/rendering/terrain_tile.wgsl");

/// CPU-side terrain data for a tile.
///
/// Two variants:
/// - `Heightmap`: regular elevation grid from Terrain RGB / Terrarium decoding.
///   Mesh is built on-demand by [`build_terrain_mesh`].
/// - `PrebuiltMesh`: pre-built triangle mesh from Quantized Mesh 1.0 decoding.
///   Heights are stored in **metres** (not scaled); [`build_terrain_mesh_from_qm`]
///   is called once to produce this variant and `height_scale` is applied in
///   [`TerrainRenderer::get_or_build_mesh`] so exaggeration changes work without
///   re-fetching tiles.
pub enum TerrainTileData {
    /// Heightmap elevation grid (Terrain RGB / Terrarium).
    Heightmap {
        elevation: Vec<f32>,
        width: u32,
        height: u32,
    },
    /// Pre-built triangle mesh (Quantized Mesh 1.0).
    ///
    /// `positions[i][2]` is elevation in **metres** (not scaled by height_scale).
    /// Scaling is applied when building the GPU vertex buffer.
    PrebuiltMesh {
        /// Per-vertex data (position, normal, tex_coord).
        /// `position[2]` is raw metres; normal is in unscaled mesh space.
        vertices: Vec<crate::render::TerrainVertex>,
        /// Triangle indices.
        indices: Vec<u32>,
        /// Regular grid heightmap rasterized from the QM mesh.
        /// Used for over-zoom fallback: when a child tile beyond `max_zoom`
        /// needs elevation from this parent tile, it can sub-sample this
        /// heightmap via `elev_uv_rect` instead of rendering a flat placeholder.
        /// Grid is `fallback_grid_size × fallback_grid_size`, values in metres.
        fallback_heightmap: Vec<f32>,
        /// Side length of the square fallback heightmap grid.
        fallback_grid_size: u32,
    },
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
    /// Value is `(data, source_coord)`: when `source_coord != render_coord`,
    /// the data comes from a parent tile and only the relevant sub-rect
    /// should be sampled (computed via `fallback_uv_rect`).
    pub elevation_data: HashMap<TileCoord, (&'a TerrainTileData, TileCoord)>,
    /// Per-tile opacity overrides (for fade-in animation).
    pub tile_opacity_overrides: HashMap<TileCoord, f32>,
}

/// Cached GPU resources for a terrain tile.
///
/// Everything here is allocated once and reused across frames:
/// - Vertex/index buffers: rebuilt only when elevation source changes
/// - Uniform buffer: same allocation, contents updated via `write_buffer()`
/// - Bind group: reused as long as the imagery texture coord doesn't change
///   (uniform buffer is the same object, sampler never changes)
struct CachedMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    /// Which elevation tile was used (for mesh invalidation).
    elev_source: TileCoord,
    /// Reusable uniform buffer — updated every frame, never reallocated.
    uniform_buffer: wgpu::Buffer,
    /// Cached bind group — reused when imagery texture hasn't changed.
    bind_group: Option<wgpu::BindGroup>,
    /// The imagery texture coord used to build `bind_group`.
    /// When this changes (parent→child swap), bind group is recreated.
    last_texture_coord: Option<TileCoord>,
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
    /// Elevation exaggeration factor (default: 20.0 for visible terrain in Mercator view).
    pub exaggeration: f64,
    /// Cached vertex/index buffers keyed by render coord.
    mesh_cache: HashMap<TileCoord, CachedMesh>,
    /// Exaggeration value when the cache was last valid.
    /// If exaggeration changes, the entire cache is invalidated.
    cached_exaggeration: f64,
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
                        // Ccw because the VP matrix includes a flip_x (-1 on x-axis)
                        // which reverses winding order in clip space.
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
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

        let exaggeration = 1.5;

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
            exaggeration,
            mesh_cache: HashMap::new(),
            cached_exaggeration: exaggeration,
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

    /// Invalidate a cached mesh so it's rebuilt next frame.
    ///
    /// Used when terrain data is re-resampled (e.g., multi-source
    /// geographic heightmap update eliminates cliff walls).
    pub fn invalidate_mesh(&mut self, coord: &TileCoord) {
        self.mesh_cache.remove(coord);
    }

    /// Get or build the cached mesh for a terrain tile.
    ///
    /// Returns the cached vertex/index buffers if the elevation source
    /// hasn't changed, otherwise rebuilds the mesh and updates the cache.
    fn get_or_build_mesh(
        &mut self,
        gpu: &GpuContext,
        coord: &TileCoord,
        elevation: &TerrainTileData,
        elev_uv_rect: [f32; 4],
        elev_source: TileCoord,
        height_scale: f32,
    ) -> &CachedMesh {
        // Check if cache entry is still valid
        let needs_rebuild = match self.mesh_cache.get(coord) {
            Some(cached) => cached.elev_source != elev_source,
            None => true,
        };

        if needs_rebuild {
            let (vertices, indices) = match elevation {
                TerrainTileData::Heightmap { elevation: elev, width, height } => {
                    build_terrain_mesh(coord, elev, *width, *height, height_scale, elev_uv_rect)
                }
                TerrainTileData::PrebuiltMesh {
                    vertices,
                    indices,
                    fallback_heightmap,
                    fallback_grid_size,
                } => {
                    if elev_source != *coord {
                        // Parent's QM mesh covers the parent tile area, not this
                        // child's sub-region.  Use the rasterized fallback heightmap
                        // to provide elevation data.  `build_terrain_mesh` will
                        // sub-sample via `elev_uv_rect` just like regular heightmaps.
                        build_terrain_mesh(
                            coord,
                            fallback_heightmap,
                            *fallback_grid_size,
                            *fallback_grid_size,
                            height_scale,
                            elev_uv_rect,
                        )
                    } else {
                        // Pre-built mesh: apply height_scale to the z component.
                        // Heights are stored in metres; scale here so exaggeration
                        // changes (which clear the GPU mesh cache) work correctly.
                        let scaled: Vec<crate::render::TerrainVertex> = vertices
                            .iter()
                            .map(|v| crate::render::TerrainVertex {
                                position: [
                                    v.position[0],
                                    v.position[1],
                                    v.position[2] * height_scale,
                                ],
                                normal: v.normal,
                                tex_coord: v.tex_coord,
                            })
                            .collect();
                        (scaled, indices.clone())
                    }
                }
            };

            let vertex_buffer = gpu.create_vertex_buffer(
                &format!("terrain-verts-{}-{}-{}", coord.z, coord.x, coord.y),
                &vertices,
            );
            let index_buffer = gpu.create_index_buffer(
                &format!("terrain-idx-{}-{}-{}", coord.z, coord.x, coord.y),
                &indices,
            );
            // Uniform buffer allocated once, reused every frame via update_buffer.
            let zero_uniforms: TileUniforms = bytemuck::Zeroable::zeroed();
            let uniform_buffer = gpu.create_uniform_buffer(
                &format!("terrain-uni-{}-{}-{}", coord.z, coord.x, coord.y),
                &zero_uniforms,
            );

            self.mesh_cache.insert(*coord, CachedMesh {
                vertex_buffer,
                index_buffer,
                index_count: indices.len() as u32,
                elev_source,
                uniform_buffer,
                bind_group: None,
                last_texture_coord: None,
            });
        }

        self.mesh_cache.get(coord).unwrap()
    }

    /// Render terrain layers to the target surface.
    ///
    /// Terrain layers use `LoadOp::Load` for color (preserves raster layers already drawn)
    /// and `LoadOp::Clear` for depth (own depth buffer).
    pub fn render_terrain_layered(
        &mut self,
        gpu: &GpuContext,
        target: &wgpu::TextureView,
        viewport: &Viewport,
        layers: &[TerrainLayerData],
    ) {
        if layers.is_empty() {
            return;
        }

        // Invalidate mesh cache if exaggeration changed.
        if (self.exaggeration - self.cached_exaggeration).abs() > 1e-9 {
            self.mesh_cache.clear();
            self.cached_exaggeration = self.exaggeration;
        }

        let height_scale = compute_height_scale(self.exaggeration);

        // Update viewport uniforms
        let uniforms = viewport.to_uniforms();
        gpu.update_buffer(&self.viewport_buffer, &uniforms);

        // Compute f64 VP for per-tile MVP (eliminates high-zoom jitter)
        let vp_f64 = viewport.to_view_proj_f64();

        // Track which tiles are rendered this frame for cache eviction.
        let mut rendered_coords = HashSet::new();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        for layer in layers {
            // Phase 1: Ensure meshes are cached for all visible tiles.
            // This is the expensive part — but only runs for NEW tiles.
            let tile_data: Vec<_> = layer
                .tiles
                .iter()
                .filter_map(|rt| {
                    let tex_view = layer.imagery_views.get(&rt.texture_coord);
                    let elev_entry = layer.elevation_data.get(&rt.coord);
                    if tex_view.is_none() || elev_entry.is_none() {
                        eprintln!(
                            "[terrain] SKIP tile z={} x={} y={}: \
                             imagery_view={} (tex_coord z={} x={} y={}), \
                             elev_data={}",
                            rt.coord.z, rt.coord.x, rt.coord.y,
                            tex_view.is_some(),
                            rt.texture_coord.z, rt.texture_coord.x, rt.texture_coord.y,
                            elev_entry.is_some(),
                        );
                    }
                    let tex_view = tex_view?;
                    let (elev, elev_source) = elev_entry?;
                    let tile_opacity = layer
                        .tile_opacity_overrides
                        .get(&rt.coord)
                        .copied()
                        .unwrap_or(layer.opacity);
                    let elev_uv = fallback_uv_rect(&rt.coord, elev_source);

                    Some((rt, *tex_view, *elev, *elev_source, elev_uv, tile_opacity))
                })
                .collect();

            // Build/fetch cached meshes
            for &(rt, _, elev, elev_source, elev_uv, _) in &tile_data {
                self.get_or_build_mesh(gpu, &rt.coord, elev, elev_uv, elev_source, height_scale);
                rendered_coords.insert(rt.coord);
            }

            // Phase 2: Update uniform buffers + reuse/rebuild bind groups.
            //
            // Per-frame cost breakdown (steady-state, no new tiles):
            //   - uniform_buffer: queue.write_buffer() only (zero alloc)
            //   - bind_group:     REUSED from cache (zero alloc)
            // Bind group is only rebuilt when imagery texture_coord changes
            // (parent→child swap), which happens at most once per tile load.
            let render_coords: Vec<TileCoord> = tile_data
                .iter()
                .filter_map(|&(rt, tex_view, _, _, _, tile_opacity)| {
                    let cached = self.mesh_cache.get_mut(&rt.coord)?;

                    // Update uniform buffer in-place (no allocation).
                    let tile_uniforms = tile_uniforms_with_uv(&rt.coord, tile_opacity, rt.uv_rect, &vp_f64);
                    gpu.update_buffer(&cached.uniform_buffer, &tile_uniforms);

                    // Rebuild bind group only when imagery texture changes.
                    let tex_changed = cached.last_texture_coord != Some(rt.texture_coord);
                    if tex_changed || cached.bind_group.is_none() {
                        cached.bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("terrain-tile-bg"),
                            layout: &self.tile_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: cached.uniform_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(tex_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        }));
                        cached.last_texture_coord = Some(rt.texture_coord);
                    }

                    Some(rt.coord)
                })
                .collect();

            if render_coords.is_empty() {
                continue;
            }

            // Phase 3: Render pass — all buffers and bind groups live in mesh_cache.
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

                for coord in &render_coords {
                    if let Some(cached) = self.mesh_cache.get(coord) {
                        if let Some(bg) = &cached.bind_group {
                            pass.set_bind_group(1, bg, &[]);
                            pass.set_vertex_buffer(0, cached.vertex_buffer.slice(..));
                            pass.set_index_buffer(cached.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..cached.index_count, 0, 0..1);
                        }
                    }
                }
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Evict cached meshes for tiles no longer visible.
        // Keep a generous margin — only evict if cache is large AND tile is not rendered.
        if self.mesh_cache.len() > rendered_coords.len() + 64 {
            self.mesh_cache.retain(|coord, _| rendered_coords.contains(coord));
        }
    }
}
