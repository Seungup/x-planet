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

use crate::pipeline::{
    build_terrain_mesh, build_terrain_mesh_centered, build_terrain_mesh_globe,
    compute_height_scale_for, fallback_uv_rect,
    tile_passes_angular_filter,
    tile_uniforms_for_centered, tile_uniforms_for_globe,
    RenderableTile,
};
use crate::render::TerrainVertex;
use crate::terrain_data::TerrainTileData;
use crate::viewport::Viewport;
use std::collections::{HashMap, HashSet};

const TERRAIN_TILE_SHADER: &str = include_str!("../../../shaders/rendering/terrain_tile.wgsl");

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
    /// Which projection mode was used (mesh geometry depends on projection).
    _projection_mode: x_planets_math::ProjectionMode,
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
    /// Elevation exaggeration factor (default: 1.5).
    pub exaggeration: f64,
    /// Cached vertex/index buffers keyed by render coord.
    mesh_cache: HashMap<TileCoord, CachedMesh>,
    /// Exaggeration value when the cache was last valid.
    /// If exaggeration changes, the entire cache is invalidated.
    cached_exaggeration: f64,
    /// Projection mode when the cache was last valid.
    /// If projection changes, the entire cache is invalidated
    /// (mesh geometry is projection-dependent).
    cached_projection_mode: x_planets_math::ProjectionMode,
}

/// Projection-specific parameters for terrain mesh building.
#[allow(dead_code)]
enum TerrainMeshParams {
    Mercator,
    Centered {
        center_lat_rad: f64,
        center_lon_rad: f64,
        tile_center_2d: glam::DVec2,
    },
    Globe {
        tile_center_3d: glam::DVec3,
    },
}

impl TerrainMeshParams {
    fn mode(&self) -> x_planets_math::ProjectionMode {
        match self {
            Self::Mercator => x_planets_math::ProjectionMode::Mercator,
            Self::Centered { .. } => x_planets_math::ProjectionMode::Mercator, // uses Mercator enum for centered
            Self::Globe { .. } => x_planets_math::ProjectionMode::Globe,
        }
    }
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
                        visibility: wgpu::ShaderStages::VERTEX
                            | wgpu::ShaderStages::FRAGMENT,
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
                        // No back-face culling: the terrain pipeline serves both
                        // centered Mercator (VP includes flip_x → CW becomes CCW)
                        // and Globe (no flip_x → CCW stays CCW).  A single
                        // front_face setting can't satisfy both modes, and the
                        // fragment shader's small-circle clipping + depth buffer
                        // already discard invisible fragments.
                        // Matches the raster centered_pipeline approach.
                        cull_mode: None,
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
            clip_sphere: [0.0; 4],
            terrain: [0.0; 4],
            sun_dir: [0.0; 4],
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
            cached_projection_mode: x_planets_math::ProjectionMode::Mercator,
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
    ///
    /// `mesh_params` provides projection-specific parameters for mesh building.
    fn get_or_build_mesh(
        &mut self,
        gpu: &GpuContext,
        coord: &TileCoord,
        elevation: &TerrainTileData,
        elev_uv_rect: [f32; 4],
        elev_source: TileCoord,
        height_scale: f32,
        mesh_params: &TerrainMeshParams,
    ) -> &CachedMesh {
        // Check if cache entry is still valid
        let needs_rebuild = match self.mesh_cache.get(coord) {
            Some(cached) => cached.elev_source != elev_source,
            None => true,
        };

        if needs_rebuild {
            // Get raw elevation data (heightmap) for projection-aware builders.
            // For PrebuiltMesh with fallback, use the fallback heightmap.
            // For PrebuiltMesh with own data, we need to handle differently per projection.
            let (vertices, indices) = match mesh_params {
                TerrainMeshParams::Mercator => {
                    self.build_mesh_standard(coord, elevation, elev_uv_rect, elev_source, height_scale)
                }
                TerrainMeshParams::Centered { center_lat_rad, center_lon_rad, tile_center_2d } => {
                    self.build_mesh_for_heightmap(
                        coord, elevation, elev_uv_rect, elev_source, height_scale,
                        |elev, w, h, uv_rect, hs| {
                            build_terrain_mesh_centered(
                                coord, elev, w, h, hs, uv_rect,
                                *center_lat_rad, *center_lon_rad, *tile_center_2d,
                            )
                        },
                    )
                }
                TerrainMeshParams::Globe { tile_center_3d } => {
                    self.build_mesh_for_heightmap(
                        coord, elevation, elev_uv_rect, elev_source, height_scale,
                        |elev, w, h, uv_rect, hs| {
                            build_terrain_mesh_globe(
                                coord, elev, w, h, hs, uv_rect, *tile_center_3d,
                            )
                        },
                    )
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
                _projection_mode: mesh_params.mode(),
                uniform_buffer,
                bind_group: None,
                last_texture_coord: None,
            });
        }

        self.mesh_cache.get(coord).expect("mesh_cache: just-inserted entry missing")
    }

    /// Build mesh using standard Mercator (original logic).
    fn build_mesh_standard(
        &self,
        coord: &TileCoord,
        elevation: &TerrainTileData,
        elev_uv_rect: [f32; 4],
        elev_source: TileCoord,
        height_scale: f32,
    ) -> (Vec<TerrainVertex>, Vec<u32>) {
        match elevation {
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
                    build_terrain_mesh(
                        coord, fallback_heightmap, *fallback_grid_size, *fallback_grid_size,
                        height_scale, elev_uv_rect,
                    )
                } else {
                    let scaled: Vec<TerrainVertex> = vertices
                        .iter()
                        .map(|v| TerrainVertex {
                            position: [v.position[0], v.position[1], v.position[2] * height_scale],
                            normal: v.normal,
                            tex_coord: v.tex_coord,
                        })
                        .collect();
                    (scaled, indices.clone())
                }
            }
        }
    }

    /// Build mesh using a projection-aware builder that takes heightmap data.
    ///
    /// For PrebuiltMesh (QM), always uses the fallback heightmap since the
    /// projection-aware builders need to recompute vertex positions from scratch
    /// (the pre-built QM positions are in standard Mercator space).
    fn build_mesh_for_heightmap<F>(
        &self,
        _coord: &TileCoord,
        elevation: &TerrainTileData,
        elev_uv_rect: [f32; 4],
        _elev_source: TileCoord,
        height_scale: f32,
        builder: F,
    ) -> (Vec<TerrainVertex>, Vec<u32>)
    where
        F: FnOnce(&[f32], u32, u32, [f32; 4], f32) -> (Vec<TerrainVertex>, Vec<u32>),
    {
        match elevation {
            TerrainTileData::Heightmap { elevation: elev, width, height } => {
                builder(elev, *width, *height, elev_uv_rect, height_scale)
            }
            TerrainTileData::PrebuiltMesh {
                fallback_heightmap,
                fallback_grid_size,
                ..
            } => {
                builder(
                    fallback_heightmap, *fallback_grid_size, *fallback_grid_size,
                    elev_uv_rect, height_scale,
                )
            }
        }
    }

    /// Render terrain layers to the target surface.
    ///
    /// Terrain layers use `LoadOp::Load` for color (preserves raster layers already drawn)
    /// and `LoadOp::Clear` for depth (own depth buffer).
    ///
    /// `mode` controls how terrain meshes are built and positioned:
    /// - `Mercator`: standard flat Mercator space (original behavior)
    /// - `Globe`: unit sphere with radial elevation displacement
    /// The centered Mercator mode is auto-detected when `mode == Mercator` and
    /// the raster renderer uses centered Mercator (same projection pipeline).
    pub fn render_terrain_layered(
        &mut self,
        gpu: &GpuContext,
        target: &wgpu::TextureView,
        viewport: &Viewport,
        layers: &[TerrainLayerData],
        mode: x_planets_math::ProjectionMode,
    ) {
        if layers.is_empty() {
            return;
        }

        // Invalidate mesh cache if exaggeration or projection changed.
        if (self.exaggeration - self.cached_exaggeration).abs() > 1e-9
            || self.cached_projection_mode != mode
        {
            self.mesh_cache.clear();
            self.cached_exaggeration = self.exaggeration;
            self.cached_projection_mode = mode;
        }

        // For centered Mercator (non-Globe Mercator), invalidate mesh cache
        // every frame because the oblique center changes with viewport pan.
        let is_centered = mode != x_planets_math::ProjectionMode::Globe;
        if is_centered {
            // Centered Mercator meshes depend on the viewport center,
            // which changes as the user pans.  Must rebuild every frame.
            self.mesh_cache.clear();
        }

        let height_scale = compute_height_scale_for(self.exaggeration, viewport.body.circumference);

        // Update viewport uniforms
        let uniforms = viewport.to_uniforms();
        gpu.update_buffer(&self.viewport_buffer, &uniforms);

        // Compute f64 VP using the same projection as the raster renderer.
        let vp_f64 = viewport.to_view_proj_f64_projected(mode);

        let center_lat_rad = viewport.center.lat.to_radians();
        let center_lon_rad = viewport.center.lon.to_radians();

        // Track which tiles are rendered this frame for cache eviction.
        let mut rendered_coords = HashSet::new();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        for layer in layers {
            // Phase 1: Ensure meshes are cached for all visible tiles.
            let tile_data: Vec<_> = layer
                .tiles
                .iter()
                .filter(|rt| {
                    // In centered Mercator mode, skip tiles beyond the angular
                    // threshold to avoid degenerate geometry from the oblique
                    // Mercator singularity.  Same filter the raster renderer uses.
                    if is_centered {
                        tile_passes_angular_filter(
                            rt, center_lat_rad, center_lon_rad, viewport.zoom,
                        )
                    } else {
                        true
                    }
                })
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

            // Build/fetch cached meshes with projection-aware geometry.
            for &(rt, _, elev, elev_source, elev_uv, _) in &tile_data {
                let mesh_params = match mode {
                    x_planets_math::ProjectionMode::Globe => {
                        let tile_center_3d = crate::pipeline::tile_mesh::globe_tile_center(
                            &rt.coord, rt.display_x,
                        );
                        TerrainMeshParams::Globe { tile_center_3d }
                    }
                    _ => {
                        // Centered Mercator — use oblique Mercator reprojection
                        let tile_center_2d = crate::pipeline::tile_mesh::centered_tile_center(
                            &rt.coord, rt.display_x, center_lat_rad, center_lon_rad,
                        );
                        TerrainMeshParams::Centered {
                            center_lat_rad,
                            center_lon_rad,
                            tile_center_2d,
                        }
                    }
                };
                self.get_or_build_mesh(gpu, &rt.coord, elev, elev_uv, elev_source, height_scale, &mesh_params);
                rendered_coords.insert(rt.coord);
            }

            // Phase 2: Update uniform buffers + reuse/rebuild bind groups.
            let render_coords: Vec<TileCoord> = tile_data
                .iter()
                .filter_map(|&(rt, tex_view, _, _, _, tile_opacity)| {
                    let cached = self.mesh_cache.get_mut(&rt.coord)?;

                    // Compute per-tile uniforms using the correct projection.
                    let tile_uniforms = match mode {
                        x_planets_math::ProjectionMode::Globe => {
                            tile_uniforms_for_globe(rt, tile_opacity, &vp_f64)
                        }
                        _ => {
                            tile_uniforms_for_centered(
                                rt, tile_opacity, &vp_f64,
                                center_lat_rad, center_lon_rad,
                            )
                        }
                    };
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
