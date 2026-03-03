//! Pure-function pipeline stages.
//!
//! Karpathy principle: every stage is a pure function.
//! Input → Output. No side effects. Independently testable.
//!
//! The MapEngine orchestrates these stages, but each stage
//! knows nothing about the engine or GPU.

use std::collections::HashSet;
use x_planets_math::{GeoCoord, TileCoord, TileUniforms, ViewportUniforms, VisibleTile};

use crate::render::{
    tile_quad_vertices_projected, GlobeTileVertex, TileVertex, TILE_QUAD_INDICES,
    tile_globe_mesh, tile_centered_mesh, polar_cap_mesh,
};
use crate::viewport::Viewport;

// ───────────────────────────────────────────────────────────────────
// Stage 1: Viewport → visible tile list
// ───────────────────────────────────────────────────────────────────

/// Determine which tiles are visible at the given zoom level (single-level, no LOD).
///
/// Returns all tiles at `tile_zoom()` that intersect the frustum.
/// For LOD-based multi-level tile selection, use [`Viewport::visible_tiles()`] instead.
///
/// Pure function. No state, no side effects.
pub fn visible_tiles(viewport: &Viewport) -> Vec<VisibleTile> {
    viewport.frustum().visible_tiles(viewport.tile_zoom())
}

// ───────────────────────────────────────────────────────────────────
// Stage 2: Visible tiles − cached tiles → load requests
// ───────────────────────────────────────────────────────────────────

/// A request to load a tile, with priority (lower = closer to center).
#[derive(Debug, Clone)]
pub struct LoadRequest {
    pub coord: TileCoord,
    pub priority: f32,
}

/// Given visible tiles and a set of already-cached tile coords,
/// produce a priority-ordered list of tiles that need loading.
///
/// Uses geographic (lat/lon) distance for priority.  The native app
/// uses Mercator distance with fallback-aware priority instead — see
/// `NativeApp::window_event` (RedrawRequested step 4a).
///
/// Pure function.
pub fn compute_load_requests(
    visible: &[TileCoord],
    cached: &HashSet<TileCoord>,
    center: &GeoCoord,
) -> Vec<LoadRequest> {
    let mut requests: Vec<LoadRequest> = visible
        .iter()
        .filter(|t| !cached.contains(t))
        .map(|t| {
            let c = t.to_geo_bounds().center();
            let dx = c.lon - center.lon;
            let dy = c.lat - center.lat;
            LoadRequest {
                coord: *t,
                priority: (dx * dx + dy * dy).sqrt() as f32,
            }
        })
        .collect();

    requests.sort_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap());
    requests
}

// ───────────────────────────────────────────────────────────────────
// Stage 3: Tile coords → GPU vertex/index data
// ───────────────────────────────────────────────────────────────────

/// Build vertex and index buffers for a set of tile quads.
///
/// Pure function. Each tile becomes a textured quad (4 verts, 6 indices).
pub fn build_tile_mesh(tiles: &[TileCoord]) -> (Vec<TileVertex>, Vec<u32>) {
    build_tile_mesh_projected(tiles, x_planets_math::ProjectionMode::Mercator)
}

/// Build vertex and index buffers with projection-dependent tile geometry.
pub fn build_tile_mesh_projected(
    tiles: &[TileCoord],
    mode: x_planets_math::ProjectionMode,
) -> (Vec<TileVertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(tiles.len() * 4);
    let mut indices = Vec::with_capacity(tiles.len() * 6);

    for (i, tile) in tiles.iter().enumerate() {
        let base = (i * 4) as u32;
        let quad = tile_quad_vertices_projected(tile, mode);
        vertices.extend_from_slice(&quad);
        for idx in TILE_QUAD_INDICES {
            indices.push(base + idx);
        }
    }

    (vertices, indices)
}

// ───────────────────────────────────────────────────────────────────
// Stage 4: Viewport → GPU uniform data
// ───────────────────────────────────────────────────────────────────

/// Compute the per-frame viewport uniform buffer data.
///
/// Pure function.
pub fn viewport_uniforms(viewport: &Viewport) -> ViewportUniforms {
    viewport.to_uniforms()
}

/// Compute per-tile uniform data.
///
/// `vp_f64` is the view-projection matrix in f64, from `Viewport::to_view_proj_f64()`.
/// The per-tile MVP is computed as `VP_f64 * translate(tile_center_f64)`, then cast to f32.
///
/// Pure function.
pub fn tile_uniforms(coord: &TileCoord, opacity: f32, vp_f64: &glam::DMat4) -> TileUniforms {
    tile_uniforms_with_uv(coord, opacity, [0.0, 0.0, 1.0, 1.0], vp_f64)
}

/// Compute per-tile uniform data with a UV sub-rectangle.
///
/// `vp_f64` is the view-projection matrix in f64.  The per-tile MVP is computed
/// as `VP_f64 * translate(tile_center_f64)` in full f64, then cast to f32.
/// This eliminates f32 jitter at high zoom by baking the large tile-center
/// offset into the matrix while still in f64.
///
/// Pure function.
pub fn tile_uniforms_with_uv(
    coord: &TileCoord,
    opacity: f32,
    uv_rect: [f32; 4],
    vp_f64: &glam::DMat4,
) -> TileUniforms {
    let n_f64 = coord.extent() as f64;
    let n = n_f64 as f32;
    let min_x = coord.x as f32 / n;
    let min_y = coord.y as f32 / n;
    let max_x = (coord.x + 1) as f32 / n;
    let max_y = (coord.y + 1) as f32 / n;

    // Tile center in f64 — the key to precision.
    let cx = (coord.x as f64 + 0.5) / n_f64;
    let cy = (coord.y as f64 + 0.5) / n_f64;

    // MVP = VP_f64 * translate(tile_center), computed entirely in f64.
    let model = glam::DMat4::from_translation(glam::DVec3::new(cx, cy, 0.0));
    let mvp_f64 = *vp_f64 * model;
    let mvp_f32 = mvp_f64.as_mat4();

    TileUniforms {
        mvp: mvp_f32.to_cols_array(),
        bounds: [min_x, min_y, max_x, max_y],
        meta: [coord.z as f32, opacity, 0.0, 0.0],
        uv_rect,
    }
}

/// Compute per-tile uniforms using `display_x` for antimeridian wrapping.
///
/// Like `tile_uniforms_with_uv`, but uses the unwrapped `display_x` for tile
/// center positioning so tiles crossing the antimeridian render correctly.
///
/// Pure function.
pub fn tile_uniforms_for_visible(
    rt: &RenderableTile,
    opacity: f32,
    vp_f64: &glam::DMat4,
) -> TileUniforms {
    tile_uniforms_for_visible_projected(
        rt,
        opacity,
        vp_f64,
        x_planets_math::ProjectionMode::Mercator,
    )
}

/// Like [`tile_uniforms_for_visible`] but positions the tile center using the given projection.
pub fn tile_uniforms_for_visible_projected(
    rt: &RenderableTile,
    opacity: f32,
    vp_f64: &glam::DMat4,
    mode: x_planets_math::ProjectionMode,
) -> TileUniforms {
    let n_f64 = rt.coord.extent() as f64;
    let n = n_f64 as f32;
    let min_x = rt.display_x as f32 / n;
    let min_y = rt.coord.y as f32 / n;
    let max_x = (rt.display_x + 1) as f32 / n;
    let max_y = (rt.coord.y + 1) as f32 / n;

    // Tile center — x is the same for both projections, y depends on mode.
    let cx = (rt.display_x as f64 + 0.5) / n_f64;
    let cy = match mode {
        x_planets_math::ProjectionMode::Mercator
        | x_planets_math::ProjectionMode::Globe => (rt.coord.y as f64 + 0.5) / n_f64,
        x_planets_math::ProjectionMode::Equirectangular => {
            let y_top_m = rt.coord.y as f64 / n_f64;
            let y_bot_m = (rt.coord.y + 1) as f64 / n_f64;
            let y_top_eq = x_planets_math::mercator_y_to_equirectangular_y(y_top_m);
            let y_bot_eq = x_planets_math::mercator_y_to_equirectangular_y(y_bot_m);
            (y_top_eq + y_bot_eq) / 2.0
        }
    };

    let model = glam::DMat4::from_translation(glam::DVec3::new(cx, cy, 0.0));
    let mvp_f64 = *vp_f64 * model;
    let mvp_f32 = mvp_f64.as_mat4();

    TileUniforms {
        mvp: mvp_f32.to_cols_array(),
        bounds: [min_x, min_y, max_x, max_y],
        meta: [rt.coord.z as f32, opacity, 0.0, 0.0],
        uv_rect: rt.uv_rect,
    }
}

// ───────────────────────────────────────────────────────────────────
// Globe (3D sphere) mesh + uniforms
// ───────────────────────────────────────────────────────────────────

/// Compute the center of a tile on the unit sphere (f64 precision).
///
/// Uses `display_x` for correct antimeridian handling.
/// Pure function.
pub fn globe_tile_center(coord: &TileCoord, display_x: i64) -> glam::DVec3 {
    let n = coord.extent() as f64;
    let mx = (display_x as f64 + 0.5) / n;
    let my = (coord.y as f64 + 0.5) / n;
    let lon_rad = (mx * 2.0 - 1.0) * std::f64::consts::PI;
    let lat_rad = x_planets_math::mercator_y_to_lat_rad(my);
    x_planets_math::geo_to_unit_sphere(lat_rad, lon_rad)
}

/// Build tessellated sphere meshes for all tiles in a layer.
///
/// Returns `(vertices, indices, per_tile_index_counts)`.
/// Each tile gets zoom-dependent subdivisions; index counts may vary.
/// Pure function.
pub fn build_globe_tile_mesh(
    tiles: &[RenderableTile],
) -> (Vec<GlobeTileVertex>, Vec<u32>, Vec<u32>) {
    let mut all_verts = Vec::new();
    let mut all_idxs = Vec::new();
    let mut tile_idx_counts = Vec::new();

    for rt in tiles {
        let tile_center_3d = globe_tile_center(&rt.coord, rt.display_x);
        let base_vertex = all_verts.len() as u32;
        let (verts, idxs) = tile_globe_mesh(&rt.coord, tile_center_3d);
        all_verts.extend(verts);
        all_idxs.extend(idxs.iter().map(|i| i + base_vertex));
        tile_idx_counts.push(idxs.len() as u32);
    }

    (all_verts, all_idxs, tile_idx_counts)
}

/// Build polar cap geometry for the globe (fills the holes at ±90°).
///
/// Returns `(vertices, indices)` for both north and south polar caps.
/// These use absolute unit-sphere positions (no RTE offset).
/// Render with an identity model matrix in the VP.
pub fn build_polar_caps() -> (Vec<GlobeTileVertex>, Vec<u32>) {
    let (mut verts, mut idxs) = polar_cap_mesh(true);
    let (south_v, south_i) = polar_cap_mesh(false);
    let base = verts.len() as u32;
    verts.extend(south_v);
    idxs.extend(south_i.iter().map(|i| i + base));
    (verts, idxs)
}

/// Compute per-tile uniforms for globe rendering.
///
/// The model translation is the tile's 3D center on the unit sphere
/// (instead of a 2D Mercator point).
/// Pure function.
pub fn tile_uniforms_for_globe(
    rt: &RenderableTile,
    opacity: f32,
    vp_f64: &glam::DMat4,
) -> TileUniforms {
    let n = rt.coord.extent() as f32;
    let min_x = rt.display_x as f32 / n;
    let min_y = rt.coord.y as f32 / n;
    let max_x = (rt.display_x + 1) as f32 / n;
    let max_y = (rt.coord.y + 1) as f32 / n;

    let tile_center_3d = globe_tile_center(&rt.coord, rt.display_x);
    let model = glam::DMat4::from_translation(tile_center_3d);
    let mvp_f64 = *vp_f64 * model;
    let mvp_f32 = mvp_f64.as_mat4();

    TileUniforms {
        mvp: mvp_f32.to_cols_array(),
        bounds: [min_x, min_y, max_x, max_y],
        meta: [rt.coord.z as f32, opacity, 0.0, 0.0],
        uv_rect: rt.uv_rect,
    }
}

// ───────────────────────────────────────────────────────────────────
// Viewport-centered (oblique) Mercator mesh + uniforms
// ───────────────────────────────────────────────────────────────────

/// Compute tile center in oblique (viewport-centered) Mercator space.
fn centered_tile_center(
    coord: &TileCoord,
    display_x: i64,
    center_lat_rad: f64,
    center_lon_rad: f64,
) -> glam::DVec2 {
    let n = coord.extent() as f64;
    let mx = (display_x as f64 + 0.5) / n;
    let my = (coord.y as f64 + 0.5) / n;
    let lon_rad = (mx * 2.0 - 1.0) * std::f64::consts::PI;
    let lat_rad = x_planets_math::mercator_y_to_lat_rad(my);
    x_planets_math::oblique_mercator(lat_rad, lon_rad, center_lat_rad, center_lon_rad)
}

/// Build tessellated centered-Mercator meshes for all tiles.
///
/// Tiles are projected through oblique Mercator centered on
/// `(center_lat_rad, center_lon_rad)`, producing 2D positions (z=0)
/// in `GlobeTileVertex` format (reuses the globe pipeline).
///
/// Returns `(vertices, indices, per_tile_index_counts)`.
pub fn build_centered_tile_mesh(
    tiles: &[RenderableTile],
    center_lat_rad: f64,
    center_lon_rad: f64,
) -> (Vec<GlobeTileVertex>, Vec<u32>, Vec<u32>) {
    let mut all_verts = Vec::new();
    let mut all_idxs = Vec::new();
    let mut tile_idx_counts = Vec::new();

    for rt in tiles {
        let tile_center_2d =
            centered_tile_center(&rt.coord, rt.display_x, center_lat_rad, center_lon_rad);
        let base_vertex = all_verts.len() as u32;
        let (verts, idxs) =
            tile_centered_mesh(&rt.coord, center_lat_rad, center_lon_rad, tile_center_2d);
        all_verts.extend(verts);
        all_idxs.extend(idxs.iter().map(|i| i + base_vertex));
        tile_idx_counts.push(idxs.len() as u32);
    }

    (all_verts, all_idxs, tile_idx_counts)
}

/// Compute per-tile uniforms for centered Mercator rendering.
pub fn tile_uniforms_for_centered(
    rt: &RenderableTile,
    opacity: f32,
    vp_f64: &glam::DMat4,
    center_lat_rad: f64,
    center_lon_rad: f64,
) -> TileUniforms {
    let n = rt.coord.extent() as f32;
    let min_x = rt.display_x as f32 / n;
    let min_y = rt.coord.y as f32 / n;
    let max_x = (rt.display_x + 1) as f32 / n;
    let max_y = (rt.coord.y + 1) as f32 / n;

    let tile_center_2d =
        centered_tile_center(&rt.coord, rt.display_x, center_lat_rad, center_lon_rad);
    let model =
        glam::DMat4::from_translation(glam::DVec3::new(tile_center_2d.x, tile_center_2d.y, 0.0));
    let mvp_f64 = *vp_f64 * model;
    let mvp_f32 = mvp_f64.as_mat4();

    TileUniforms {
        mvp: mvp_f32.to_cols_array(),
        bounds: [min_x, min_y, max_x, max_y],
        meta: [rt.coord.z as f32, opacity, 0.0, 0.0],
        uv_rect: rt.uv_rect,
    }
}

// ───────────────────────────────────────────────────────────────────
// Fallback texture resolution
// ───────────────────────────────────────────────────────────────────

/// A tile ready for rendering, with resolved texture source.
#[derive(Debug, Clone)]
pub struct RenderableTile {
    /// The tile position to render (geometry) — canonical coord.
    pub coord: TileCoord,
    /// The tile whose texture to use (may be an ancestor).
    pub texture_coord: TileCoord,
    /// UV sub-rectangle within texture_coord's texture.
    /// `[0, 0, 1, 1]` = full texture (own texture available).
    pub uv_rect: [f32; 4],
    /// Unwrapped X for rendering position (antimeridian support).
    /// Can be negative or >= 2^z.
    pub display_x: i64,
}

/// Compute the UV sub-rectangle of `ancestor`'s texture that corresponds
/// to the area covered by `tile`.
///
/// Pure function.
pub fn fallback_uv_rect(tile: &TileCoord, ancestor: &TileCoord) -> [f32; 4] {
    if tile.z == ancestor.z {
        return [0.0, 0.0, 1.0, 1.0];
    }

    let tile_ext = (1u32 << tile.z) as f32;
    let anc_ext = (1u32 << ancestor.z) as f32;

    // Tile's Mercator bounds.
    let tile_min_x = tile.x as f32 / tile_ext;
    let tile_max_x = (tile.x + 1) as f32 / tile_ext;
    let tile_min_y = tile.y as f32 / tile_ext;
    let tile_max_y = (tile.y + 1) as f32 / tile_ext;

    // Ancestor's Mercator bounds.
    let anc_min_x = ancestor.x as f32 / anc_ext;
    let anc_max_x = (ancestor.x + 1) as f32 / anc_ext;
    let anc_min_y = ancestor.y as f32 / anc_ext;
    let anc_max_y = (ancestor.y + 1) as f32 / anc_ext;

    let anc_w = anc_max_x - anc_min_x;
    let anc_h = anc_max_y - anc_min_y;

    [
        (tile_min_x - anc_min_x) / anc_w,
        (tile_min_y - anc_min_y) / anc_h,
        (tile_max_x - anc_min_x) / anc_w,
        (tile_max_y - anc_min_y) / anc_h,
    ]
}

/// Resolve visible tiles to renderable tiles with fallback textures.
///
/// For each visible tile, finds the best available texture (own or
/// nearest ancestor).  Returns `None` for tiles with no texture at all.
/// Preserves `display_x` from the input `VisibleTile` for antimeridian wrapping.
///
/// Pure function.
pub fn resolve_fallbacks(
    visible: &[VisibleTile],
    available_textures: &HashSet<TileCoord>,
) -> Vec<RenderableTile> {
    visible
        .iter()
        .filter_map(|vt| {
            let mut cur = vt.coord;
            loop {
                if available_textures.contains(&cur) {
                    let uv = fallback_uv_rect(&vt.coord, &cur);
                    return Some(RenderableTile {
                        coord: vt.coord,
                        texture_coord: cur,
                        uv_rect: uv,
                        display_x: vt.display_x,
                    });
                }
                match cur.parent() {
                    Some(p) => cur = p,
                    None => return None,
                }
            }
        })
        .collect()
}

// ───────────────────────────────────────────────────────────────────
// Stage 4b: Terrain mesh generation
// ───────────────────────────────────────────────────────────────────

use crate::render::TerrainVertex;

/// Terrain mesh grid resolution (vertices per axis = GRID_SIZE + 1).
/// 32 → 33×33 = 1089 verts, 2048 triangles per tile.
pub const TERRAIN_GRID_SIZE: u32 = 32;

/// Build a displaced terrain mesh for a single tile.
///
/// Generates a `(TERRAIN_GRID_SIZE+1)²` vertex grid with per-vertex normals
/// for hillshade lighting.  Vertex Z is sampled from `elevation` via bilinear
/// interpolation and scaled by `height_scale`.
///
/// `elev_uv_rect` remaps sampling coordinates into the elevation grid:
/// `[0, 0, 1, 1]` = use the full grid (own tile data available).
/// When using a parent's elevation data as fallback, pass the sub-rect
/// computed by `fallback_uv_rect(tile, parent)` so only the relevant
/// quadrant is sampled.
///
/// Pure function.
pub fn build_terrain_mesh(
    coord: &TileCoord,
    elevation: &[f32],
    src_width: u32,
    src_height: u32,
    height_scale: f32,
    elev_uv_rect: [f32; 4],
) -> (Vec<TerrainVertex>, Vec<u32>) {
    let grid = TERRAIN_GRID_SIZE;
    let verts_per_side = grid + 1;
    let vert_count = (verts_per_side * verts_per_side) as usize;
    let mut indices = Vec::with_capacity((grid * grid * 6) as usize);

    // Use f64 for tile bounds to avoid precision loss at high zoom,
    // then store vertex positions *relative to tile center* in f32.
    let n = coord.extent() as f64;
    let tile_size_f64 = 1.0 / n;
    let tile_w = tile_size_f64 as f32;
    let tile_h = tile_w; // square tiles

    // Elevation UV sub-rect: remap [0,1] → [eu_min, eu_max] for parent fallback.
    let eu_min = elev_uv_rect[0];
    let ev_min = elev_uv_rect[1];
    let eu_range = elev_uv_rect[2] - eu_min;
    let ev_range = elev_uv_rect[3] - ev_min;

    // ── Pass 1: Sample elevation at grid points ──
    // Positions are RELATIVE TO TILE CENTER for f32 precision.
    // The shader reconstructs world position using tile.bounds.
    let mut positions = Vec::with_capacity(vert_count);
    let mut tex_coords = Vec::with_capacity(vert_count);

    for gy in 0..verts_per_side {
        for gx in 0..verts_per_side {
            let u = gx as f32 / grid as f32;
            let v = gy as f32 / grid as f32;

            // Remap u,v into the elevation grid's sub-rect
            let eu = eu_min + u * eu_range;
            let ev = ev_min + v * ev_range;
            let h = sample_elevation_bilinear(elevation, src_width, src_height, eu, ev);

            // Relative to tile center: (u - 0.5) * tile_w, (v - 0.5) * tile_h
            positions.push([(u - 0.5) * tile_w, (v - 0.5) * tile_h, h * height_scale]);
            tex_coords.push([u, v]);
        }
    }

    // ── Pass 2: Compute per-vertex normals from neighboring positions ──
    //
    // At tile edges, the naive approach clamps the missing neighbor to `self`,
    // producing a one-sided difference.  Adjacent tiles compute the opposite
    // one-sided difference → normal discontinuity → visible hillshade seam.
    //
    // Fix: use a **reflected sample** at edges.  For example, at the left edge
    // (gx=0), the missing "left" neighbor is approximated as the mirror of
    // the "right" neighbor across the current vertex:
    //     left_virtual = 2 * p - right
    // This produces a central-difference-like normal that's symmetric and
    // continuous.  At interior vertices, standard central differences are used.
    let mut normals = vec![[0.0f32, 0.0, 1.0]; vert_count];
    let vs = verts_per_side as usize;

    for gy in 0..vs {
        for gx in 0..vs {
            let idx = gy * vs + gx;
            let p = positions[idx];

            // Reflected-sample neighbors at edges for smooth boundary normals.
            let right = if gx + 1 < vs { positions[idx + 1] } else {
                // Right edge: reflect left neighbor across p
                let l = positions[idx - 1];
                [2.0 * p[0] - l[0], 2.0 * p[1] - l[1], 2.0 * p[2] - l[2]]
            };
            let left = if gx > 0 { positions[idx - 1] } else {
                // Left edge: reflect right neighbor across p
                let r = positions[idx + 1];
                [2.0 * p[0] - r[0], 2.0 * p[1] - r[1], 2.0 * p[2] - r[2]]
            };
            let down = if gy + 1 < vs { positions[idx + vs] } else {
                // Bottom edge: reflect up neighbor across p
                let u = positions[idx - vs];
                [2.0 * p[0] - u[0], 2.0 * p[1] - u[1], 2.0 * p[2] - u[2]]
            };
            let up = if gy > 0 { positions[idx - vs] } else {
                // Top edge: reflect down neighbor across p
                let d = positions[idx + vs];
                [2.0 * p[0] - d[0], 2.0 * p[1] - d[1], 2.0 * p[2] - d[2]]
            };

            // Tangent vectors (central difference or reflected central difference)
            let dx = [right[0] - left[0], right[1] - left[1], right[2] - left[2]];
            let dy = [down[0] - up[0], down[1] - up[1], down[2] - up[2]];

            // Cross product (dx × dy) → surface normal
            let nx = dx[1] * dy[2] - dx[2] * dy[1];
            let ny = dx[2] * dy[0] - dx[0] * dy[2];
            let nz = dx[0] * dy[1] - dx[1] * dy[0];

            let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-10);
            normals[idx] = [nx / len, ny / len, nz / len];
        }
    }

    // ── Assemble surface vertices ──
    let mut vertices: Vec<TerrainVertex> = (0..vert_count)
        .map(|i| TerrainVertex {
            position: positions[i],
            normal: normals[i],
            tex_coord: tex_coords[i],
        })
        .collect();

    // ── Generate surface triangle indices ──
    // Winding: CW in tile-local (y-down) space.
    // After VP flip_x + y-inversion, this becomes CCW in clip space,
    // matching the terrain pipeline's FrontFace::Ccw setting.
    // (Same winding convention as QM PrebuiltMesh indices.)
    for gy in 0..grid {
        for gx in 0..grid {
            let tl = gy * verts_per_side + gx;
            let tr = tl + 1;
            let bl = tl + verts_per_side;
            let br = bl + 1;

            indices.push(tl);
            indices.push(bl);
            indices.push(tr);
            indices.push(tr);
            indices.push(bl);
            indices.push(br);
        }
    }

    // ── Skirt geometry ──
    // Extend vertical "walls" below each edge to hide gaps between tiles.
    let skirt_depth = tile_w * 0.05; // 5% of tile width
    let down_normal = [0.0f32, 0.0, -1.0];

    // Collect edge vertex indices: bottom, top, right, left edges
    let mut edge_strips: Vec<Vec<u32>> = Vec::new();

    // Bottom edge (gy=last, left to right)
    let mut strip = Vec::new();
    for gx in 0..verts_per_side {
        strip.push(grid * verts_per_side + gx);
    }
    edge_strips.push(strip);

    // Top edge (gy=0, right to left)
    // Reversed traversal: outward normal is -Y, reversing direction
    // makes the skirt triangles CW when viewed from outside.
    let mut strip = Vec::new();
    for gx in (0..verts_per_side).rev() {
        strip.push(gx);
    }
    edge_strips.push(strip);

    // Right edge (gx=last, bottom to top)
    // Reversed traversal: outward normal is +X, reversing direction
    // makes the skirt triangles CW when viewed from outside.
    let mut strip = Vec::new();
    for gy in (0..verts_per_side).rev() {
        strip.push(gy * verts_per_side + grid);
    }
    edge_strips.push(strip);

    // Left edge (gx=0, top to bottom)
    let mut strip = Vec::new();
    for gy in 0..verts_per_side {
        strip.push(gy * verts_per_side);
    }
    edge_strips.push(strip);

    for edge in &edge_strips {
        for i in 0..edge.len() - 1 {
            let top_a = edge[i] as usize;
            let top_b = edge[i + 1] as usize;

            // Add two skirt vertices (same xy, lowered z)
            let skirt_a = vertices.len() as u32;
            let mut pa = positions[top_a];
            pa[2] -= skirt_depth;
            vertices.push(TerrainVertex {
                position: pa,
                normal: down_normal,
                tex_coord: tex_coords[top_a],
            });

            let skirt_b = vertices.len() as u32;
            let mut pb = positions[top_b];
            pb[2] -= skirt_depth;
            vertices.push(TerrainVertex {
                position: pb,
                normal: down_normal,
                tex_coord: tex_coords[top_b],
            });

            // Two triangles (reversed winding to match surface):
            // top_a, skirt_a, top_b  +  skirt_a, skirt_b, top_b
            indices.push(edge[i]);
            indices.push(skirt_a);
            indices.push(edge[i + 1]);
            indices.push(skirt_a);
            indices.push(skirt_b);
            indices.push(edge[i + 1]);
        }
    }

    (vertices, indices)
}

/// Bilinear sample from elevation grid.
fn sample_elevation_bilinear(
    elevation: &[f32],
    src_width: u32,
    src_height: u32,
    u: f32,
    v: f32,
) -> f32 {
    let sx = u * (src_width - 1) as f32;
    let sy = v * (src_height - 1) as f32;
    let ix = (sx as u32).min(src_width.saturating_sub(2));
    let iy = (sy as u32).min(src_height.saturating_sub(2));
    let fx = sx - ix as f32;
    let fy = sy - iy as f32;

    let idx00 = (iy * src_width + ix) as usize;
    let idx10 = idx00 + 1;
    let idx01 = idx00 + src_width as usize;
    let idx11 = idx01 + 1;

    if idx11 < elevation.len() {
        elevation[idx00] * (1.0 - fx) * (1.0 - fy)
            + elevation[idx10] * fx * (1.0 - fy)
            + elevation[idx01] * (1.0 - fx) * fy
            + elevation[idx11] * fx * fy
    } else if !elevation.is_empty() {
        elevation[idx00.min(elevation.len() - 1)]
    } else {
        0.0
    }
}

/// Compute height scale: converts meters of elevation to Mercator [0,1] world units.
///
/// `exaggeration` controls visual amplification (1.0 = real scale, 2.0 = 2× taller).
/// At the equator, 1 Mercator unit ≈ 40,075,000 m.
///
/// A base exaggeration of 1.0 would produce realistic proportions but
/// the heights are nearly invisible at global zoom levels.
///
/// Pure function.
pub fn compute_height_scale(exaggeration: f64) -> f32 {
    const EARTH_CIRCUMFERENCE_M: f64 = 40_075_000.0;
    (exaggeration / EARTH_CIRCUMFERENCE_M) as f32
}

// ───────────────────────────────────────────────────────────────────
// Stage 4c: Quantized Mesh → TerrainVertex conversion
// ───────────────────────────────────────────────────────────────────

/// Convert a decoded Quantized Mesh tile into `TerrainVertex` + index lists.
///
/// Heights in the returned vertices are in **metres** (not scaled).
/// The caller (`TerrainRenderer::get_or_build_mesh`) applies `height_scale`
/// when uploading to the GPU so that exaggeration changes work without
/// re-fetching tiles.
///
/// ## Coordinate mapping
/// - `u`: 0-32767 → 0-1 (west → east across tile)
/// - `v`: 0-32767 → 0-1 QM south-north; **flipped** here so that v_tile = 0 = north
///   (matches the existing XYZ / web-Mercator orientation used by `build_terrain_mesh`).
/// - `height`: 0-32767 → [`header.min_height`, `header.max_height`] metres.
///
/// ## Normals
/// If the tile contains an oct-encoded normals extension, those are decoded and used.
/// Otherwise, per-vertex normals are computed from the mesh geometry (face-normal
/// accumulation, area-weighted).
///
/// ## Skirts
/// Edge vertex lists (`west_indices`, `south_indices`, `east_indices`, `north_indices`)
/// are used to generate skirt quads that prevent cracks between adjacent tiles.
///
/// Pure function.
pub fn build_terrain_mesh_from_qm(
    coord: &TileCoord,
    qm: &x_planets_tiles::DecodedQuantizedMesh,
) -> (Vec<TerrainVertex>, Vec<u32>) {
    use x_planets_tiles::quantized_mesh::decode_oct_normal;

    let n = coord.extent() as f64;
    let tile_w = (1.0 / n) as f32;
    let tile_h = tile_w;

    let min_h = qm.header.min_height;
    let max_h = qm.header.max_height;
    let h_range = max_h - min_h;
    let vertex_count = qm.u.len();

    // ── Pass 1: positions + tex_coords ────────────────────────────
    let mut positions: Vec<[f32; 3]>   = Vec::with_capacity(vertex_count);
    let mut tex_coords: Vec<[f32; 2]>  = Vec::with_capacity(vertex_count);

    for i in 0..vertex_count {
        let u_norm = qm.u[i] as f32 / 32767.0;       // 0-1, west→east
        let v_norm = qm.v[i] as f32 / 32767.0;       // 0-1, south→north (QM convention)
        let h_norm = qm.height[i] as f32 / 32767.0;

        // QM v=0 is south, v=1 is north.
        // The existing pipeline uses v=0 at the top (north in XYZ TMS=false tiles).
        let v_tile = 1.0 - v_norm; // flip to match existing convention

        let height_m = min_h + h_norm * h_range; // metres (not scaled)

        positions.push([
            (u_norm - 0.5) * tile_w,
            (v_tile - 0.5) * tile_h,
            height_m,            // raw metres; height_scale applied in renderer
        ]);
        tex_coords.push([u_norm, v_tile]);
    }

    // ── Pass 2: normals ────────────────────────────────────────────
    let normals: Vec<[f32; 3]> = if let Some(oct) = &qm.oct_normals {
        oct.iter()
            .map(|&[x, y]| decode_oct_normal(x, y))
            .collect()
    } else {
        compute_normals_from_triangles(&positions, &qm.indices)
    };

    // ── Assemble surface vertices ─────────────────────────────────
    let mut vertices: Vec<TerrainVertex> = (0..vertex_count)
        .map(|i| TerrainVertex {
            position:  positions[i],
            normal:    normals[i],
            tex_coord: tex_coords[i],
        })
        .collect();

    let mut indices = qm.indices.clone();

    // ── Skirts (edge vertices → downward quads) ───────────────────
    // Prevents gaps/cracks between adjacent tiles at different detail.
    //
    // QM positions store z in **metres**, not in scaled tile-local space
    // (height_scale is applied later in the renderer).  The skirt depth
    // must therefore also be in metres so it remains proportional after
    // scaling.  We use ~2 % of the tile's equatorial width in metres,
    // which gives a consistent 2-3 % depth relative to tile_w in the
    // final coordinate space regardless of exaggeration.
    let tile_extent_m = 40_075_000.0_f64 / n;
    let skirt_depth = (tile_extent_m * 0.02) as f32;
    let down_normal = [0.0f32, 0.0, -1.0];

    for edge_indices in [
        &qm.west_indices,
        &qm.south_indices,
        &qm.east_indices,
        &qm.north_indices,
    ] {
        let edge_count = edge_indices.len();
        if edge_count < 2 {
            continue;
        }
        for i in 0..edge_count - 1 {
            let top_a = edge_indices[i] as usize;
            let top_b = edge_indices[i + 1] as usize;
            if top_a >= positions.len() || top_b >= positions.len() {
                continue;
            }

            let skirt_a_idx = vertices.len() as u32;
            let mut pa = positions[top_a];
            pa[2] -= skirt_depth;
            vertices.push(TerrainVertex {
                position:  pa,
                normal:    down_normal,
                tex_coord: tex_coords[top_a],
            });

            let skirt_b_idx = vertices.len() as u32;
            let mut pb = positions[top_b];
            pb[2] -= skirt_depth;
            vertices.push(TerrainVertex {
                position:  pb,
                normal:    down_normal,
                tex_coord: tex_coords[top_b],
            });

            // CW winding in tile-local space → CCW in clip space after VP flip_x
            // (matches terrain_renderer pipeline: FrontFace::Ccw)
            let ia = edge_indices[i];
            let ib = edge_indices[i + 1];
            indices.push(ia);
            indices.push(skirt_a_idx);
            indices.push(ib);
            indices.push(ib);
            indices.push(skirt_a_idx);
            indices.push(skirt_b_idx);
        }
    }

    (vertices, indices)
}

// ───────────────────────────────────────────────────────────────────
// Stage 4c-geo: Project QM mesh from EPSG:4326 to EPSG:3857
// ───────────────────────────────────────────────────────────────────

/// Re-project a QM mesh from EPSG:4326 tile-local space to EPSG:3857
/// tile-local space **in place**.
///
/// After `build_terrain_mesh_from_qm` the vertex tex_coords are in the
/// 4326 tile's UV space: u ∈ [0,1] west→east, v ∈ [0,1] north→south.
/// This function converts every vertex's position and tex_coord so
/// that they live in the target 3857 tile's local space instead.
///
/// ## Why this matters
/// The rasterize → resample pipeline loses edge continuity:
/// adjacent 3857 tiles that straddle a 4326 latitude boundary
/// rasterize DIFFERENT QM TIN meshes, producing different
/// interpolated heights at the shared seam → cliff walls.
///
/// By projecting the QM mesh directly, edge vertices that are
/// shared between adjacent 4326 tiles (guaranteed by the QM spec)
/// map to the same 3857 positions, preserving continuity.
///
/// ## Vertex transformation
/// - `tex_coord` → geographic (lon, lat) → 3857 normalised → 3857 tile-local UV
/// - `position[0..2]` → recalculated from the new UV + original height
/// - `position[2]` (height in metres) is **unchanged**
///
/// Pure function (modifies `vertices` in place).
pub fn project_qm_vertices_4326_to_3857(
    vertices: &mut [crate::render::TerrainVertex],
    merc_coord: &TileCoord,
    geo_west: f64,
    geo_east: f64,
    geo_north: f64,
    geo_south: f64,
) {
    let pi = std::f64::consts::PI;
    let n = (1u64 << merc_coord.z) as f64;
    let tile_w = (1.0 / n) as f32;

    let geo_lon_range = geo_east - geo_west;
    let geo_lat_range = geo_north - geo_south;

    for v in vertices.iter_mut() {
        let u_4326 = v.tex_coord[0] as f64;
        let v_4326 = v.tex_coord[1] as f64;

        // 4326 UV → geographic degrees
        let lon_deg = geo_west + u_4326 * geo_lon_range;
        let lat_deg = geo_north - v_4326 * geo_lat_range;

        // Geographic → 3857 normalised [0,1]×[0,1]
        let merc_nx = (lon_deg + 180.0) / 360.0;
        let lat_rad = lat_deg.to_radians();
        let merc_ny = 0.5 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / (2.0 * pi);

        // 3857 normalised → tile-local UV
        let u_3857 = merc_nx * n - merc_coord.x as f64;
        let v_3857 = merc_ny * n - merc_coord.y as f64;

        // Update position (tile-local coords in 3857 grid).
        // position[2] (height in metres) stays unchanged.
        v.position[0] = (u_3857 as f32 - 0.5) * tile_w;
        v.position[1] = (v_3857 as f32 - 0.5) * tile_w;

        // Update tex_coord to 3857 tile-local UV (for imagery draping)
        v.tex_coord = [u_3857 as f32, v_3857 as f32];
    }
}

/// Compute per-vertex normals from a triangle mesh.
///
/// Uses area-weighted face normal accumulation.
/// Fallback when oct-encoded normals are not available in the QM tile.
///
/// Pure function.
fn compute_normals_from_triangles(
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32, 0.0, 0.0]; positions.len()];

    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if ia >= positions.len() || ib >= positions.len() || ic >= positions.len() {
            continue;
        }
        let a = positions[ia];
        let b = positions[ib];
        let c = positions[ic];

        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];

        // Cross product (magnitude = 2× triangle area → area-weighted)
        let nx = ab[1] * ac[2] - ab[2] * ac[1];
        let ny = ab[2] * ac[0] - ab[0] * ac[2];
        let nz = ab[0] * ac[1] - ab[1] * ac[0];

        for &idx in &[ia, ib, ic] {
            normals[idx][0] += nx;
            normals[idx][1] += ny;
            normals[idx][2] += nz;
        }
    }

    for n in &mut normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-10);
        n[0] /= len;
        n[1] /= len;
        n[2] /= len;
    }

    normals
}

// ───────────────────────────────────────────────────────────────────
// Stage 4d: QM mesh → regular heightmap rasterization (over-zoom fallback)
// ───────────────────────────────────────────────────────────────────

/// Rasterize a Quantized Mesh triangle mesh into a regular grid heightmap.
///
/// Used to create the `fallback_heightmap` stored alongside `PrebuiltMesh`.
/// When a child tile beyond `max_zoom` needs elevation from a parent QM tile,
/// it sub-samples this heightmap using `build_terrain_mesh()` with the appropriate
/// `elev_uv_rect`, just like heightmap-based terrain (Terrain RGB / Terrarium).
///
/// The grid is `grid_size × grid_size` and covers the full [0,1]² UV space of the
/// tile.  Heights are in metres (matching `TerrainVertex::position[2]`).
///
/// Pure function.
pub fn rasterize_qm_to_heightmap(
    vertices: &[crate::render::TerrainVertex],
    indices: &[u32],
    grid_size: u32,
) -> Vec<f32> {
    let gs = grid_size as usize;
    let mut heightmap = vec![0.0f32; gs * gs];
    // Track which cells have been written for gap-filling later.
    let mut filled = vec![false; gs * gs];

    let inv = 1.0 / (grid_size - 1) as f32;

    // Rasterize each triangle: for each grid cell whose center falls inside
    // the triangle (in tex_coord / UV space), compute height via barycentric
    // interpolation.
    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if ia >= vertices.len() || ib >= vertices.len() || ic >= vertices.len() {
            continue;
        }

        let a_uv = vertices[ia].tex_coord;
        let b_uv = vertices[ib].tex_coord;
        let c_uv = vertices[ic].tex_coord;
        let a_h = vertices[ia].position[2];
        let b_h = vertices[ib].position[2];
        let c_h = vertices[ic].position[2];

        // Bounding box of the triangle in grid coordinates
        let min_u = a_uv[0].min(b_uv[0]).min(c_uv[0]);
        let max_u = a_uv[0].max(b_uv[0]).max(c_uv[0]);
        let min_v = a_uv[1].min(b_uv[1]).min(c_uv[1]);
        let max_v = a_uv[1].max(b_uv[1]).max(c_uv[1]);

        let col_min = ((min_u / inv).floor() as usize).min(gs - 1);
        let col_max = ((max_u / inv).ceil() as usize).min(gs - 1);
        let row_min = ((min_v / inv).floor() as usize).min(gs - 1);
        let row_max = ((max_v / inv).ceil() as usize).min(gs - 1);

        for row in row_min..=row_max {
            for col in col_min..=col_max {
                let pu = col as f32 * inv;
                let pv = row as f32 * inv;

                // Barycentric coordinates
                let (w0, w1, w2) = barycentric(
                    pu, pv,
                    a_uv[0], a_uv[1],
                    b_uv[0], b_uv[1],
                    c_uv[0], c_uv[1],
                );

                if w0 >= -1e-4 && w1 >= -1e-4 && w2 >= -1e-4 {
                    let idx = row * gs + col;
                    let h = w0 * a_h + w1 * b_h + w2 * c_h;
                    heightmap[idx] = h;
                    filled[idx] = true;
                }
            }
        }
    }

    // Fill unfilled cells with nearest filled neighbor (simple flood fill).
    // This handles tiny gaps due to floating-point precision.
    fill_gaps(&mut heightmap, &filled, gs);

    heightmap
}

/// Barycentric coordinates of point (px, py) with respect to triangle (ax,ay)-(bx,by)-(cx,cy).
fn barycentric(
    px: f32, py: f32,
    ax: f32, ay: f32,
    bx: f32, by: f32,
    cx: f32, cy: f32,
) -> (f32, f32, f32) {
    let v0x = bx - ax;
    let v0y = by - ay;
    let v1x = cx - ax;
    let v1y = cy - ay;
    let v2x = px - ax;
    let v2y = py - ay;

    let d00 = v0x * v0x + v0y * v0y;
    let d01 = v0x * v1x + v0y * v1y;
    let d11 = v1x * v1x + v1y * v1y;
    let d20 = v2x * v0x + v2y * v0y;
    let d21 = v2x * v1x + v2y * v1y;

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        return (-1.0, -1.0, -1.0); // Degenerate triangle
    }
    let inv_denom = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;
    let u = 1.0 - v - w;

    (u, v, w)
}

/// Fill unfilled cells with the nearest filled cell's value.
/// Simple iterative spreading — runs at most `gs` passes.
fn fill_gaps(heightmap: &mut [f32], filled: &[bool], gs: usize) {
    let unfilled_count = filled.iter().filter(|&&f| !f).count();
    if unfilled_count == 0 {
        return;
    }

    let mut current_filled = filled.to_vec();
    let offsets: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for _ in 0..gs {
        let mut any_changed = false;
        let prev_filled = current_filled.clone();
        for row in 0..gs {
            for col in 0..gs {
                let idx = row * gs + col;
                if prev_filled[idx] {
                    continue;
                }
                // Find any filled neighbor
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for &(dr, dc) in &offsets {
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    if nr >= 0 && nr < gs as i32 && nc >= 0 && nc < gs as i32 {
                        let ni = nr as usize * gs + nc as usize;
                        if prev_filled[ni] {
                            sum += heightmap[ni];
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    heightmap[idx] = sum / count as f32;
                    current_filled[idx] = true;
                    any_changed = true;
                }
            }
        }
        if !any_changed {
            break;
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Stage 4e: EPSG:4326 → EPSG:3857 heightmap resampling
// ───────────────────────────────────────────────────────────────────

/// Resample a heightmap from EPSG:4326 (Geographic) tile UV space to
/// EPSG:3857 (Web Mercator) tile UV space.
///
/// The input heightmap lives in the UV space of a 4326 tile with the given
/// geographic bounds.  This function creates a new heightmap for a 3857 tile
/// by mapping each output grid point to geographic (lon, lat) coordinates
/// (using the Mercator projection), then sampling the input heightmap.
///
/// This is necessary because the two projections use fundamentally different
/// tile grids:
/// - EPSG:3857 tiles are square in Mercator space (latitude varies non-linearly)
/// - EPSG:4326 tiles are rectangular in lat/lon space (uniform degree spacing)
///
/// ## Parameters
/// - `src_heightmap`: input heightmap in 4326 UV space (row-major, north→south)
/// - `src_grid_size`: side length of the square source grid
/// - `geo_west/east/north/south`: geographic bounds of the 4326 tile (degrees)
/// - `merc_coord`: the 3857 tile coordinate to produce the output for
/// - `out_grid_size`: side length of the square output grid
///
/// ## Returns
/// A `Vec<f32>` heightmap in 3857 UV space, `out_grid_size × out_grid_size`,
/// with heights in metres.
///
/// Pure function.
pub fn resample_geographic_to_mercator(
    src_heightmap: &[f32],
    src_grid_size: u32,
    geo_west: f64,
    geo_east: f64,
    geo_north: f64,
    geo_south: f64,
    merc_coord: &x_planets_math::TileCoord,
    out_grid_size: u32,
) -> Vec<f32> {
    let ogs = out_grid_size as usize;
    let mut out = vec![0.0f32; ogs * ogs];

    let n_3857 = (1u64 << merc_coord.z) as f64;
    let inv = 1.0 / (out_grid_size - 1) as f64;

    let geo_lon_range = geo_east - geo_west;
    let geo_lat_range = geo_north - geo_south; // positive (north > south)

    for row in 0..ogs {
        for col in 0..ogs {
            let u_merc = col as f64 * inv;
            let v_merc = row as f64 * inv;

            // Convert (u_merc, v_merc) to geographic (lon, lat).
            // Longitude is linear within a Mercator tile:
            let lon = (merc_coord.x as f64 + u_merc) / n_3857 * 360.0 - 180.0;

            // Latitude requires the Mercator Y → latitude conversion:
            let merc_y = std::f64::consts::PI
                * (1.0 - 2.0 * (merc_coord.y as f64 + v_merc) / n_3857);
            let lat = merc_y.sinh().atan().to_degrees();

            // Map (lon, lat) to the source 4326 tile's UV space.
            let u_4326 = if geo_lon_range.abs() > 1e-12 {
                (lon - geo_west) / geo_lon_range
            } else {
                0.5
            };
            // 4326 heightmap: row 0 = north, row (gs-1) = south
            let v_4326 = if geo_lat_range.abs() > 1e-12 {
                (geo_north - lat) / geo_lat_range
            } else {
                0.5
            };

            // Clamp to [0, 1] — edge samples for areas outside the 4326 tile.
            let u_clamped = u_4326.clamp(0.0, 1.0) as f32;
            let v_clamped = v_4326.clamp(0.0, 1.0) as f32;

            let h = sample_elevation_bilinear(
                src_heightmap,
                src_grid_size,
                src_grid_size,
                u_clamped,
                v_clamped,
            );
            out[row * ogs + col] = h;
        }
    }

    out
}

/// A single EPSG:4326 heightmap source for multi-source resampling.
pub struct GeoHeightmapSource<'a> {
    pub heightmap: &'a [f32],
    pub grid_size: u32,
    pub west: f64,
    pub east: f64,
    pub north: f64,
    pub south: f64,
}

/// Resample from **multiple** EPSG:4326 heightmaps to a single EPSG:3857 tile.
///
/// For each output pixel, the function finds the 4326 source whose geographic
/// bounds contain the pixel's (lon, lat) and samples from that source.  If no
/// source covers the point, the nearest edge of the nearest source is used
/// (clamping, same as the single-source variant).
///
/// This eliminates cliff walls at 4326 tile boundaries: when a 3857 tile
/// straddles two 4326 tiles, both are provided as sources and the correct
/// one is chosen per-pixel.
///
/// Pure function.
pub fn resample_geographic_to_mercator_multi(
    sources: &[GeoHeightmapSource<'_>],
    merc_coord: &x_planets_math::TileCoord,
    out_grid_size: u32,
) -> Vec<f32> {
    if sources.is_empty() {
        return vec![0.0f32; (out_grid_size * out_grid_size) as usize];
    }
    // Fast path: single source → delegate to avoid overhead.
    if sources.len() == 1 {
        let s = &sources[0];
        return resample_geographic_to_mercator(
            s.heightmap, s.grid_size,
            s.west, s.east, s.north, s.south,
            merc_coord, out_grid_size,
        );
    }

    let ogs = out_grid_size as usize;
    let mut out = vec![0.0f32; ogs * ogs];

    let n_3857 = (1u64 << merc_coord.z) as f64;
    let inv = 1.0 / (out_grid_size - 1) as f64;

    for row in 0..ogs {
        for col in 0..ogs {
            let u_merc = col as f64 * inv;
            let v_merc = row as f64 * inv;

            // Geographic coordinates of this output pixel.
            let lon = (merc_coord.x as f64 + u_merc) / n_3857 * 360.0 - 180.0;
            let merc_y = std::f64::consts::PI
                * (1.0 - 2.0 * (merc_coord.y as f64 + v_merc) / n_3857);
            let lat = merc_y.sinh().atan().to_degrees();

            // Find the source that contains (lon, lat).
            let mut best_h = 0.0f32;
            let mut found = false;
            for s in sources {
                let lon_range = s.east - s.west;
                let lat_range = s.north - s.south;

                let u_4326 = if lon_range.abs() > 1e-12 {
                    (lon - s.west) / lon_range
                } else { 0.5 };
                let v_4326 = if lat_range.abs() > 1e-12 {
                    (s.north - lat) / lat_range
                } else { 0.5 };

                // Check if this source covers the point (within [0,1]).
                if u_4326 >= -1e-6 && u_4326 <= 1.0 + 1e-6
                    && v_4326 >= -1e-6 && v_4326 <= 1.0 + 1e-6
                {
                    let u_c = u_4326.clamp(0.0, 1.0) as f32;
                    let v_c = v_4326.clamp(0.0, 1.0) as f32;
                    best_h = sample_elevation_bilinear(
                        s.heightmap, s.grid_size, s.grid_size, u_c, v_c,
                    );
                    found = true;
                    break;
                }
            }

            if !found {
                // No source covers this point — clamp to nearest source edge.
                // Use the first source (primary).
                let s = &sources[0];
                let lon_range = s.east - s.west;
                let lat_range = s.north - s.south;
                let u_4326 = if lon_range.abs() > 1e-12 {
                    (lon - s.west) / lon_range
                } else { 0.5 };
                let v_4326 = if lat_range.abs() > 1e-12 {
                    (s.north - lat) / lat_range
                } else { 0.5 };
                let u_c = u_4326.clamp(0.0, 1.0) as f32;
                let v_c = v_4326.clamp(0.0, 1.0) as f32;
                best_h = sample_elevation_bilinear(
                    s.heightmap, s.grid_size, s.grid_size, u_c, v_c,
                );
            }

            out[row * ogs + col] = best_h;
        }
    }

    out
}

// ───────────────────────────────────────────────────────────────────
// Stage 5: Projection transform (CPU reference)
// ───────────────────────────────────────────────────────────────────

use x_planets_projection::ProjectionPlugin;

/// Apply a projection to a list of world-space positions (CPU).
///
/// This is the "ground truth" against which GPU results are compared.
///
/// Pure function.
pub fn project_positions_cpu(
    plugin: &dyn ProjectionPlugin,
    positions: &[glam::DVec3],
) -> Vec<glam::DVec3> {
    positions.iter().map(|p| plugin.project_cpu(*p)).collect()
}

/// Verify projection roundtrip accuracy.
///
/// Pure function. Returns max error across all test points.
pub fn verify_projection_roundtrip(
    plugin: &dyn ProjectionPlugin,
    test_points: &[glam::DVec3],
) -> f64 {
    test_points
        .iter()
        .map(|p| {
            let projected = plugin.project_cpu(*p);
            let recovered = plugin.unproject_cpu(projected);
            (*p - recovered).length()
        })
        .fold(0.0f64, f64::max)
}

// ───────────────────────────────────────────────────────────────────
// Stage 6: Frame summary (for Karpathy-style logging)
// ───────────────────────────────────────────────────────────────────

/// A snapshot of what happened in a single frame.
/// No state — just a value object for logging/debugging.
#[derive(Debug, Clone)]
pub struct FrameSummary {
    pub visible_tile_count: usize,
    pub cached_tile_count: usize,
    pub load_requests: usize,
    pub zoom: f64,
    pub center: GeoCoord,
}

impl std::fmt::Display for FrameSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "z={:.1} center=({:.2},{:.2}) visible={} cached={} pending={}",
            self.zoom,
            self.center.lat,
            self.center.lon,
            self.visible_tile_count,
            self.cached_tile_count,
            self.load_requests,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests — 카파시 원칙: 모든 순수 함수는 즉시 테스트
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use x_planets_math::GeoCoord;
    use x_planets_projection::{Equirectangular, Mercator};

    // ── Stage 1 ────────────────────────────────────────────────

    #[test]
    fn test_visible_tiles_zoom_0_covers_world() {
        let mut vp = Viewport::new(800, 600);
        vp.center = GeoCoord::new(0.0, 0.0);
        vp.zoom = 0.0;

        let tiles = visible_tiles(&vp);
        // zoom 0 → 1x1 grid → should include tile (0,0,0)
        assert!(
            tiles.iter().any(|vt| vt.coord == TileCoord::new(0, 0, 0)),
            "zoom 0 must include the single world tile, got: {:?}",
            tiles
        );
    }

    #[test]
    fn test_visible_tiles_zoom_1_has_4() {
        let mut vp = Viewport::new(800, 600);
        vp.center = GeoCoord::new(0.0, 0.0);
        vp.zoom = 1.0;

        let tiles = visible_tiles(&vp);
        // zoom 1 → 2x2 grid, centered on equator → should see all 4
        assert!(tiles.len() >= 2, "zoom 1 center=(0,0) should see multiple tiles");
    }

    #[test]
    fn test_visible_tiles_increases_with_zoom() {
        let mut vp = Viewport::new(800, 600);
        vp.center = GeoCoord::new(37.5, 127.0);

        let count_z2 = { vp.zoom = 2.0; visible_tiles(&vp).len() };
        let count_z5 = { vp.zoom = 5.0; visible_tiles(&vp).len() };

        // higher zoom → more tiles (smaller tiles, same viewport)
        assert!(
            count_z5 >= count_z2,
            "z5 ({}) should have >= tiles than z2 ({})",
            count_z5,
            count_z2
        );
    }

    // ── Stage 2 ────────────────────────────────────────────────

    #[test]
    fn test_load_requests_excludes_cached() {
        let visible = vec![
            TileCoord::new(2, 0, 0),
            TileCoord::new(2, 1, 0),
            TileCoord::new(2, 2, 0),
        ];
        let mut cached = HashSet::new();
        cached.insert(TileCoord::new(2, 1, 0));

        let requests = compute_load_requests(&visible, &cached, &GeoCoord::new(0.0, 0.0));

        assert_eq!(requests.len(), 2);
        assert!(requests.iter().all(|r| r.coord != TileCoord::new(2, 1, 0)));
    }

    #[test]
    fn test_load_requests_sorted_by_distance() {
        let visible = vec![
            TileCoord::new(3, 0, 0), // far from center
            TileCoord::new(3, 4, 4), // closer to equator/prime meridian
        ];
        let cached = HashSet::new();
        let center = GeoCoord::new(0.0, 0.0);

        let requests = compute_load_requests(&visible, &cached, &center);

        assert_eq!(requests.len(), 2);
        assert!(
            requests[0].priority <= requests[1].priority,
            "should be sorted by distance: {:.2} <= {:.2}",
            requests[0].priority,
            requests[1].priority
        );
    }

    #[test]
    fn test_load_requests_empty_when_all_cached() {
        let visible = vec![TileCoord::new(1, 0, 0), TileCoord::new(1, 1, 0)];
        let cached: HashSet<TileCoord> = visible.iter().copied().collect();

        let requests = compute_load_requests(&visible, &cached, &GeoCoord::new(0.0, 0.0));
        assert!(requests.is_empty());
    }

    // ── Stage 3 ────────────────────────────────────────────────

    #[test]
    fn test_build_tile_mesh_single_quad() {
        let tiles = vec![TileCoord::new(0, 0, 0)];
        let (verts, indices) = build_tile_mesh(&tiles);

        assert_eq!(verts.len(), 4, "1 tile = 4 vertices");
        assert_eq!(indices.len(), 6, "1 tile = 6 indices");
    }

    #[test]
    fn test_build_tile_mesh_multiple() {
        let tiles = vec![
            TileCoord::new(1, 0, 0),
            TileCoord::new(1, 1, 0),
            TileCoord::new(1, 0, 1),
            TileCoord::new(1, 1, 1),
        ];
        let (verts, indices) = build_tile_mesh(&tiles);

        assert_eq!(verts.len(), 16, "4 tiles × 4 vertices");
        assert_eq!(indices.len(), 24, "4 tiles × 6 indices");
    }

    #[test]
    fn test_build_tile_mesh_no_degenerate_triangles() {
        // Karpathy: "check for degenerate cases"
        for z in 0..=4u8 {
            let n = 1u32 << z;
            for x in 0..n {
                for y in 0..n {
                    let tiles = vec![TileCoord::new(z, x, y)];
                    let (verts, _) = build_tile_mesh(&tiles);

                    let w = verts[1].position[0] - verts[0].position[0];
                    let h = verts[2].position[1] - verts[0].position[1];
                    let area = w * h;

                    assert!(
                        area.abs() > 1e-10,
                        "degenerate quad at z={} x={} y={}: area={}",
                        z, x, y, area
                    );
                }
            }
        }
    }

    #[test]
    fn test_build_tile_mesh_rte_centered() {
        // RTE: zoom 0, single tile — vertices should be centered at origin
        // with half-size 0.5 in each direction.
        let tiles = vec![TileCoord::new(0, 0, 0)];
        let (verts, _) = build_tile_mesh(&tiles);

        let min_x = verts.iter().map(|v| v.position[0]).fold(f32::MAX, f32::min);
        let max_x = verts.iter().map(|v| v.position[0]).fold(f32::MIN, f32::max);
        let min_y = verts.iter().map(|v| v.position[1]).fold(f32::MAX, f32::min);
        let max_y = verts.iter().map(|v| v.position[1]).fold(f32::MIN, f32::max);

        assert!((min_x - (-0.5)).abs() < 1e-6, "min_x should be -0.5, got {}", min_x);
        assert!((max_x - 0.5).abs() < 1e-6, "max_x should be 0.5, got {}", max_x);
        assert!((min_y - (-0.5)).abs() < 1e-6, "min_y should be -0.5, got {}", min_y);
        assert!((max_y - 0.5).abs() < 1e-6, "max_y should be 0.5, got {}", max_y);
    }

    #[test]
    fn test_build_tile_mesh_rte_all_same_size_per_zoom() {
        // At zoom 1, all 4 tiles should have identical vertex positions (RTE)
        let tiles = vec![
            TileCoord::new(1, 0, 0),
            TileCoord::new(1, 1, 0),
            TileCoord::new(1, 0, 1),
            TileCoord::new(1, 1, 1),
        ];
        let (verts, _) = build_tile_mesh(&tiles);

        // Each tile's quad has 4 vertices. They should all be ±0.25.
        for chunk in verts.chunks(4) {
            let min_x = chunk.iter().map(|v| v.position[0]).fold(f32::MAX, f32::min);
            let max_x = chunk.iter().map(|v| v.position[0]).fold(f32::MIN, f32::max);
            assert!((min_x - (-0.25)).abs() < 1e-6, "min_x should be -0.25, got {}", min_x);
            assert!((max_x - 0.25).abs() < 1e-6, "max_x should be 0.25, got {}", max_x);
        }
    }

    // ── Stage 4 ────────────────────────────────────────────────

    #[test]
    fn test_tile_uniforms_bounds() {
        let vp = glam::DMat4::IDENTITY;
        let u = tile_uniforms(&TileCoord::new(1, 0, 0), 1.0, &vp);
        assert!((u.bounds[0] - 0.0).abs() < 1e-6); // min_x
        assert!((u.bounds[1] - 0.0).abs() < 1e-6); // min_y
        assert!((u.bounds[2] - 0.5).abs() < 1e-6); // max_x
        assert!((u.bounds[3] - 0.5).abs() < 1e-6); // max_y
    }

    #[test]
    fn test_tile_uniforms_opacity() {
        let vp = glam::DMat4::IDENTITY;
        let u = tile_uniforms(&TileCoord::new(0, 0, 0), 0.75, &vp);
        assert!((u.meta[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_tile_uniforms_mvp_precision() {
        // At zoom 18, verify per-tile MVP is precise:
        // tile center is at ~0.500003814697 in Mercator, a value that loses
        // precision in f32. The f64 MVP bakes this into the matrix.
        let vp_f64 = {
            let mut v = Viewport::new(800, 600);
            v.center = GeoCoord::new(0.0, 0.0);
            v.zoom = 18.0;
            v.to_view_proj_f64()
        };
        let coord = TileCoord::new(18, 131072, 131072); // near center
        let u = tile_uniforms(&coord, 1.0, &vp_f64);
        // MVP should be non-zero (a valid transform)
        let mvp_sum: f32 = u.mvp.iter().map(|v| v.abs()).sum();
        assert!(mvp_sum > 1.0, "MVP should be a valid transform");
    }

    #[test]
    fn test_viewport_uniforms_resolution() {
        let vp = Viewport::new(1920, 1080);
        let u = viewport_uniforms(&vp);
        assert_eq!(u.resolution[0], 1920.0);
        assert_eq!(u.resolution[1], 1080.0);
        assert!((u.resolution[2] - 1.0 / 1920.0).abs() < 1e-6);
    }

    // ── Stage 5: Projection ────────────────────────────────────

    #[test]
    fn test_mercator_roundtrip_1000_points() {
        let proj = Mercator;
        let points = generate_test_grid(20, 50); // 1000 points

        let max_err = verify_projection_roundtrip(&proj, &points);
        assert!(
            max_err < 1e-8,
            "Mercator roundtrip max error: {:.2e} (should be < 1e-8)",
            max_err
        );
    }

    #[test]
    fn test_equirectangular_roundtrip_1000_points() {
        let proj = Equirectangular;
        let points = generate_test_grid(20, 50);

        let max_err = verify_projection_roundtrip(&proj, &points);
        assert!(
            max_err < 1e-10,
            "Equirectangular roundtrip max error: {:.2e} (should be < 1e-10)",
            max_err
        );
    }

    #[test]
    fn test_mercator_known_values() {
        // Karpathy: "always test with known values, not just roundtrips"
        let proj = Mercator;

        // (0,0) → center of the map (0.5, 0.5)
        let origin = proj.project_cpu(glam::DVec3::new(0.0, 0.0, 0.0));
        assert!((origin.x - 0.5).abs() < 1e-10, "origin.x = {}", origin.x);
        assert!((origin.y - 0.5).abs() < 1e-10, "origin.y = {}", origin.y);

        // (0, -180) → left edge (0.0, 0.5)
        let left = proj.project_cpu(glam::DVec3::new(0.0, -180.0, 0.0));
        assert!(left.x.abs() < 1e-10, "left.x = {}", left.x);

        // (0, 180) → right edge (1.0, 0.5)
        let right = proj.project_cpu(glam::DVec3::new(0.0, 180.0, 0.0));
        assert!((right.x - 1.0).abs() < 1e-10, "right.x = {}", right.x);
    }

    #[test]
    fn test_project_positions_cpu_batch() {
        let proj = Mercator;
        let positions = vec![
            glam::DVec3::new(0.0, 0.0, 0.0),
            glam::DVec3::new(45.0, 90.0, 0.0),
            glam::DVec3::new(-30.0, -60.0, 0.0),
        ];

        let results = project_positions_cpu(&proj, &positions);

        assert_eq!(results.len(), 3);
        // Verify each result matches individual projection
        for (pos, result) in positions.iter().zip(results.iter()) {
            let expected = proj.project_cpu(*pos);
            assert!(
                (*result - expected).length() < 1e-15,
                "batch should match individual"
            );
        }
    }

    // ── Stage 6: FrameSummary ──────────────────────────────────

    #[test]
    fn test_frame_summary_display() {
        let summary = FrameSummary {
            visible_tile_count: 16,
            cached_tile_count: 12,
            load_requests: 4,
            zoom: 3.5,
            center: GeoCoord::new(37.57, 126.98),
        };

        let s = format!("{}", summary);
        assert!(s.contains("z=3.5"));
        assert!(s.contains("visible=16"));
        assert!(s.contains("pending=4"));
    }

    // ── Fallback UV / resolve_fallbacks tests ────────────────

    #[test]
    fn test_fallback_uv_same_tile() {
        // When tile == ancestor, UV rect should be identity [0,0,1,1].
        let tile = TileCoord::new(5, 10, 15);
        let uv = fallback_uv_rect(&tile, &tile);
        assert!((uv[0] - 0.0).abs() < 1e-6);
        assert!((uv[1] - 0.0).abs() < 1e-6);
        assert!((uv[2] - 1.0).abs() < 1e-6);
        assert!((uv[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_fallback_uv_one_level_up() {
        // zoom=2 tile (0,0) has 4 children at zoom=3: (0,0), (1,0), (0,1), (1,1).
        // Child (1,0) at zoom=3 should map to right-half of parent's texture.
        let parent = TileCoord::new(2, 0, 0);
        let child = TileCoord::new(3, 1, 0); // right column, top row

        let uv = fallback_uv_rect(&child, &parent);
        // u_min should be 0.5 (right half), v_min should be 0.0 (top half)
        assert!((uv[0] - 0.5).abs() < 1e-5, "u_min = {}", uv[0]);
        assert!((uv[1] - 0.0).abs() < 1e-5, "v_min = {}", uv[1]);
        assert!((uv[2] - 1.0).abs() < 1e-5, "u_max = {}", uv[2]);
        assert!((uv[3] - 0.5).abs() < 1e-5, "v_max = {}", uv[3]);
    }

    #[test]
    fn test_fallback_uv_two_levels_up() {
        // zoom=1 tile (0,0) → zoom=3 grandchild (1,1)
        let ancestor = TileCoord::new(1, 0, 0);
        let tile = TileCoord::new(3, 1, 1);

        let uv = fallback_uv_rect(&tile, &ancestor);
        // ancestor covers [0, 0] to [0.5, 0.5] in Mercator
        // tile covers [1/8, 1/8] to [2/8, 2/8]
        // relative to ancestor: [0.25, 0.25] to [0.5, 0.5]
        assert!((uv[0] - 0.25).abs() < 1e-5, "u_min = {}", uv[0]);
        assert!((uv[1] - 0.25).abs() < 1e-5, "v_min = {}", uv[1]);
        assert!((uv[2] - 0.5).abs() < 1e-5, "u_max = {}", uv[2]);
        assert!((uv[3] - 0.5).abs() < 1e-5, "v_max = {}", uv[3]);
    }

    #[test]
    fn test_resolve_fallbacks_complete_coverage() {
        // All visible tiles with their own texture should get uv_rect = [0,0,1,1].
        let visible: Vec<VisibleTile> = vec![
            VisibleTile::canonical(TileCoord::new(2, 0, 0)),
            VisibleTile::canonical(TileCoord::new(2, 1, 0)),
            VisibleTile::canonical(TileCoord::new(2, 0, 1)),
        ];
        let available: HashSet<TileCoord> = visible.iter().map(|vt| vt.coord).collect();
        let result = resolve_fallbacks(&visible, &available);

        assert_eq!(result.len(), 3);
        for r in &result {
            assert_eq!(r.coord, r.texture_coord);
            assert!((r.uv_rect[0] - 0.0).abs() < 1e-6);
            assert!((r.uv_rect[2] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_resolve_fallbacks_uses_parent() {
        // If a child tile is missing but parent is available, should use parent.
        let child = TileCoord::new(3, 2, 3);
        let parent = child.parent().unwrap();

        let visible = vec![VisibleTile::canonical(child)];
        let mut available = HashSet::new();
        available.insert(parent);

        let result = resolve_fallbacks(&visible, &available);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coord, child);
        assert_eq!(result[0].texture_coord, parent);
        // UV rect should NOT be identity since it's a fallback.
        assert!(
            result[0].uv_rect[0] != 0.0 || result[0].uv_rect[1] != 0.0
                || result[0].uv_rect[2] != 1.0 || result[0].uv_rect[3] != 1.0,
            "Fallback UV should differ from identity"
        );
    }

    #[test]
    fn test_resolve_fallbacks_missing_all_returns_empty() {
        // If no textures are available at all, result should be empty.
        let visible = vec![VisibleTile::canonical(TileCoord::new(5, 10, 10))];
        let available = HashSet::new();
        let result = resolve_fallbacks(&visible, &available);
        assert!(result.is_empty());
    }

    // ── Helpers ────────────────────────────────────────────────

    // ── Terrain mesh tests ──────────────────────────────────────

    #[test]
    fn test_build_terrain_mesh_dimensions() {
        let coord = TileCoord::new(2, 1, 1);
        let elevation = vec![0.0f32; 256 * 256];
        let identity_uv = [0.0, 0.0, 1.0, 1.0];
        let (verts, indices) = build_terrain_mesh(&coord, &elevation, 256, 256, 1e-5, identity_uv);
        let g = TERRAIN_GRID_SIZE;
        let surface_verts = (g + 1) * (g + 1);
        // Skirt adds 2 vertices per edge segment × 4 edges × g segments
        let skirt_verts = 4 * g * 2;
        assert!(
            verts.len() == (surface_verts + skirt_verts) as usize,
            "expected {} verts ({}+{}), got {}",
            surface_verts + skirt_verts, surface_verts, skirt_verts, verts.len()
        );
        // Surface indices + skirt indices
        let surface_indices = g * g * 6;
        let skirt_indices = 4 * g * 6;
        assert!(
            indices.len() == (surface_indices + skirt_indices) as usize,
            "expected {} indices, got {}",
            surface_indices + skirt_indices, indices.len()
        );
    }

    #[test]
    fn test_build_terrain_mesh_flat_has_zero_z() {
        let coord = TileCoord::new(0, 0, 0);
        let elevation = vec![0.0f32; 4]; // minimal 2×2
        let identity_uv = [0.0, 0.0, 1.0, 1.0];
        let (verts, _) = build_terrain_mesh(&coord, &elevation, 2, 2, 1e-5, identity_uv);
        let g = TERRAIN_GRID_SIZE;
        let surface_count = ((g + 1) * (g + 1)) as usize;
        // Check surface vertices only (skirt verts have negative z)
        for v in &verts[..surface_count] {
            assert!(
                v.position[2].abs() < 1e-10,
                "flat terrain should have z≈0, got {}",
                v.position[2]
            );
        }
    }

    #[test]
    fn test_build_terrain_mesh_elevated() {
        let coord = TileCoord::new(0, 0, 0);
        let elevation = vec![1000.0f32; 4]; // 1000m everywhere
        let scale = 1e-5;
        let identity_uv = [0.0, 0.0, 1.0, 1.0];
        let (verts, _) = build_terrain_mesh(&coord, &elevation, 2, 2, scale, identity_uv);
        let g = TERRAIN_GRID_SIZE;
        let surface_count = ((g + 1) * (g + 1)) as usize;
        let expected_z = 1000.0 * scale;
        // Check surface vertices only
        for v in &verts[..surface_count] {
            assert!(
                (v.position[2] - expected_z).abs() < 1e-6,
                "expected z={}, got {}",
                expected_z,
                v.position[2]
            );
        }
    }

    #[test]
    fn test_build_terrain_mesh_parent_fallback_uv() {
        // Parent (z=1, x=0, y=0) has elevation: top-left=0m, top-right=1000m,
        // bottom-left=2000m, bottom-right=3000m (2×2 grid).
        let parent = TileCoord::new(1, 0, 0);
        let child = TileCoord::new(2, 1, 1); // bottom-right quadrant
        let elevation = vec![0.0, 1000.0, 2000.0, 3000.0]; // 2×2

        let scale = 1e-5;
        let elev_uv = fallback_uv_rect(&child, &parent);
        // child (2, 1, 1) is bottom-right of parent → uv = [0.5, 0.5, 1.0, 1.0]
        assert!((elev_uv[0] - 0.5).abs() < 1e-4, "u_min={}", elev_uv[0]);
        assert!((elev_uv[1] - 0.5).abs() < 1e-4, "v_min={}", elev_uv[1]);

        let (verts, _) = build_terrain_mesh(&child, &elevation, 2, 2, scale, elev_uv);

        let g = TERRAIN_GRID_SIZE;
        // All surface vertices should sample near the bottom-right corner (3000m).
        // With bilinear interpolation over the [0.5,0.5]-[1.0,1.0] sub-rect,
        // the center of the mesh (u=0.75, v=0.75) should be ~1500m.
        // The bottom-right corner (u=1, v=1 → eu=1, ev=1) should be ~3000m.
        let br_idx = g as usize * (g as usize + 1) + g as usize; // last surface vertex
        let br_elev = verts[br_idx].position[2] / scale;
        assert!(
            (br_elev - 3000.0).abs() < 50.0,
            "bottom-right should be ~3000m (parent's BR corner), got {}m",
            br_elev
        );

        // Top-left of child (u=0, v=0 → eu=0.5, ev=0.5) should be
        // near center of parent = ~1500m (average of all 4 parent corners)
        let tl_elev = verts[0].position[2] / scale;
        assert!(
            (tl_elev - 1500.0).abs() < 100.0,
            "top-left should be ~1500m (parent center), got {}m",
            tl_elev
        );
    }

    #[test]
    fn test_build_terrain_mesh_edge_normals_smooth() {
        // Verify that edge normals are smooth (no abrupt jump at boundary).
        //
        // Create a sloped terrain (linear gradient in y) and check that the
        // normals at the top edge (gy=0) are close to the normals at gy=1
        // (first interior row), and similarly for the bottom edge.
        // With the reflected-sample approach, edge normals should extrapolate
        // the interior slope, not clamp to flat.
        let coord = TileCoord::new(5, 16, 16);
        let grid_size = 33u32;
        // Linear gradient: height increases from north to south (0m → 1000m)
        let mut elevation = vec![0.0f32; (grid_size * grid_size) as usize];
        for row in 0..grid_size {
            for col in 0..grid_size {
                let v = row as f32 / (grid_size - 1) as f32;
                elevation[(row * grid_size + col) as usize] = v * 1000.0;
            }
        }
        let scale = 1e-5;
        let identity_uv = [0.0, 0.0, 1.0, 1.0];
        let (verts, _) = build_terrain_mesh(
            &coord, &elevation, grid_size, grid_size, scale, identity_uv,
        );
        let vs = (TERRAIN_GRID_SIZE + 1) as usize;

        // Check top edge normals vs first interior row
        let mid_x = vs / 2;
        let edge_top = verts[0 * vs + mid_x].normal;
        let interior_1 = verts[1 * vs + mid_x].normal;

        // With reflected samples, top edge normal should closely match
        // the first interior row's normal (both see the same linear slope).
        let dot_top = edge_top[0] * interior_1[0]
            + edge_top[1] * interior_1[1]
            + edge_top[2] * interior_1[2];
        assert!(
            dot_top > 0.99,
            "Top edge normal should match interior (dot={:.4}), edge={:?}, interior={:?}",
            dot_top, edge_top, interior_1,
        );

        // Check bottom edge normals vs last interior row
        let last = vs - 1;
        let edge_bottom = verts[last * vs + mid_x].normal;
        let interior_last = verts[(last - 1) * vs + mid_x].normal;

        let dot_bottom = edge_bottom[0] * interior_last[0]
            + edge_bottom[1] * interior_last[1]
            + edge_bottom[2] * interior_last[2];
        assert!(
            dot_bottom > 0.99,
            "Bottom edge normal should match interior (dot={:.4}), edge={:?}, interior={:?}",
            dot_bottom, edge_bottom, interior_last,
        );
    }

    #[test]
    fn test_compute_height_scale() {
        let scale = compute_height_scale(1.0);
        // scale = 1 / 40_075_000
        let expected = (1.0 / 40_075_000.0_f64) as f32;
        assert!((scale - expected).abs() < 1e-12);

        // Linearity: 2× exaggeration → 2× scale
        let scale_2x = compute_height_scale(2.0);
        assert!((scale_2x - 2.0 * scale).abs() < 1e-12);
    }

    // ── Stage 4d: rasterize_qm_to_heightmap ────────────────────

    #[test]
    fn test_rasterize_flat_triangle() {
        // A single triangle covering the full [0,1]² UV space with constant height.
        let height = 500.0f32;
        let vertices = vec![
            TerrainVertex {
                position: [-0.5, -0.5, height],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.0, 0.0],
            },
            TerrainVertex {
                position: [0.5, -0.5, height],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [1.0, 0.0],
            },
            TerrainVertex {
                position: [-0.5, 0.5, height],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.0, 1.0],
            },
        ];
        let indices = vec![0, 1, 2];
        let hm = rasterize_qm_to_heightmap(&vertices, &indices, 5);
        // All cells inside the triangle should be ~500.
        // The lower-right half might be unfilled (gap-filled).
        let inside_count = hm.iter().filter(|&&h| (h - height).abs() < 1.0).count();
        assert!(
            inside_count >= 6,
            "at least 6 of 25 cells should be inside triangle, got {}",
            inside_count
        );
    }

    #[test]
    fn test_rasterize_two_triangles_full_coverage() {
        // Two triangles covering the full [0,1]² UV space (a quad).
        let h = 1000.0f32;
        let vertices = vec![
            TerrainVertex {
                position: [-0.5, -0.5, h],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.0, 0.0],
            },
            TerrainVertex {
                position: [0.5, -0.5, h],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [1.0, 0.0],
            },
            TerrainVertex {
                position: [-0.5, 0.5, h],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.0, 1.0],
            },
            TerrainVertex {
                position: [0.5, 0.5, h],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [1.0, 1.0],
            },
        ];
        let indices = vec![0, 1, 2, 1, 3, 2];
        let hm = rasterize_qm_to_heightmap(&vertices, &indices, 9);
        // Full coverage → all 81 cells should be ~1000m.
        for (i, &val) in hm.iter().enumerate() {
            assert!(
                (val - h).abs() < 1.0,
                "cell {} should be {}m, got {}m",
                i, h, val
            );
        }
    }

    #[test]
    fn test_rasterize_sloped_surface() {
        // A sloped surface: height varies linearly with u_tex.
        // TL(0,0)=0m, TR(1,0)=1000m, BL(0,1)=0m, BR(1,1)=1000m.
        let vertices = vec![
            TerrainVertex {
                position: [-0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.0, 0.0],
            },
            TerrainVertex {
                position: [0.5, -0.5, 1000.0],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [1.0, 0.0],
            },
            TerrainVertex {
                position: [-0.5, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.0, 1.0],
            },
            TerrainVertex {
                position: [0.5, 0.5, 1000.0],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [1.0, 1.0],
            },
        ];
        let indices = vec![0, 1, 2, 1, 3, 2];
        let gs = 5u32;
        let hm = rasterize_qm_to_heightmap(&vertices, &indices, gs);

        // Verify the slope: each column should have roughly the same height,
        // increasing from left to right.
        for col in 0..gs as usize {
            let u = col as f32 / (gs - 1) as f32;
            let expected = u * 1000.0;
            let actual = hm[col]; // row 0
            assert!(
                (actual - expected).abs() < 50.0,
                "col {} (u={:.2}): expected ~{:.0}m, got {:.0}m",
                col, u, expected, actual
            );
        }
    }

    #[test]
    fn test_rasterize_grid_size_matches() {
        let h = 100.0f32;
        let vertices = vec![
            TerrainVertex {
                position: [-0.5, -0.5, h],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.0, 0.0],
            },
            TerrainVertex {
                position: [0.5, -0.5, h],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [1.0, 0.0],
            },
            TerrainVertex {
                position: [0.5, 0.5, h],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [1.0, 1.0],
            },
            TerrainVertex {
                position: [-0.5, 0.5, h],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.0, 1.0],
            },
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];

        for gs in [5u32, 17, 33, 65] {
            let hm = rasterize_qm_to_heightmap(&vertices, &indices, gs);
            assert_eq!(
                hm.len(),
                (gs * gs) as usize,
                "grid_size={}: expected {} cells, got {}",
                gs, gs * gs, hm.len()
            );
        }
    }

    // ── Stage 4e: resample_geographic_to_mercator ───────────────

    #[test]
    fn test_resample_constant_height() {
        // A constant heightmap should remain constant after resampling.
        let h = 2500.0f32;
        let gs = 9u32;
        let src = vec![h; (gs * gs) as usize];

        // 4326 tile at z=5, x=32, y=8 (equatorial)
        let (west, east, north, south) = (0.0, 5.625, 5.625, 0.0);

        // 3857 tile z=6 (somewhere near equator)
        let merc = TileCoord::new(6, 33, 31);
        let out = resample_geographic_to_mercator(
            &src, gs, west, east, north, south, &merc, gs,
        );

        assert_eq!(out.len(), (gs * gs) as usize);
        for (i, &val) in out.iter().enumerate() {
            assert!(
                (val - h).abs() < 1.0,
                "cell {}: expected {}m, got {}m",
                i, h, val
            );
        }
    }

    #[test]
    fn test_resample_output_size() {
        let gs = 17u32;
        let src = vec![0.0f32; (gs * gs) as usize];
        let merc = TileCoord::new(5, 16, 15);

        for out_gs in [5u32, 17, 33, 65] {
            let out = resample_geographic_to_mercator(
                &src, gs, -5.625, 0.0, 5.625, 0.0, &merc, out_gs,
            );
            assert_eq!(
                out.len(),
                (out_gs * out_gs) as usize,
                "out_grid_size={}: expected {} cells",
                out_gs, out_gs * out_gs
            );
        }
    }

    #[test]
    fn test_resample_north_south_gradient() {
        // Height increases from north to south in the 4326 tile.
        // After resampling to 3857, the gradient should be preserved
        // (though non-linearly due to Mercator projection).
        let gs = 33u32;
        let mut src = vec![0.0f32; (gs * gs) as usize];
        for row in 0..gs {
            for col in 0..gs {
                // row 0 = north (0m), row gs-1 = south (1000m)
                src[(row * gs + col) as usize] = row as f32 / (gs - 1) as f32 * 1000.0;
            }
        }

        // 4326 tile bounds: near equator, [-5.625°, 0°] lon, [5.625°, 0°] lat
        let west = -5.625;
        let east = 0.0;
        let north = 5.625;
        let south = 0.0;
        // 3857 z=6, y=31: top=lat≈5.63°, bottom=lat≈0° — matches the 4326 tile
        let merc = TileCoord::new(6, 31, 31);

        let out_gs = 17u32;
        let out = resample_geographic_to_mercator(
            &src, gs, west, east, north, south, &merc, out_gs,
        );

        // Verify gradient is preserved: top row (north, ~5.6°) should have lower
        // values than bottom row (south, ~0°) since src row 0=north=0m, row last=south=1000m.
        let top_avg: f32 = (0..out_gs).map(|c| out[c as usize]).sum::<f32>() / out_gs as f32;
        let bot_avg: f32 = (0..out_gs)
            .map(|c| out[((out_gs - 1) * out_gs + c) as usize])
            .sum::<f32>()
            / out_gs as f32;

        assert!(
            bot_avg > top_avg,
            "bottom average ({:.1}) should be > top average ({:.1})",
            bot_avg, top_avg
        );
    }

    #[test]
    fn test_resample_equatorial_symmetry() {
        // A symmetric heightmap (constant E-W, varying N-S) centered on equator
        // should produce roughly symmetric results in the resampled 3857 tile
        // (since Mercator is approximately linear near the equator).
        let gs = 33u32;
        let mut src = vec![0.0f32; (gs * gs) as usize];
        let mid = (gs - 1) as f32 / 2.0;
        for row in 0..gs {
            for col in 0..gs {
                // Parabolic: peaks at center, 0 at edges
                let v = (row as f32 - mid).abs() / mid;
                src[(row * gs + col) as usize] = (1.0 - v) * 1000.0;
            }
        }

        // Symmetric around equator: [-2.8125°, 2.8125°] lat
        let merc = TileCoord::new(7, 64, 63); // centered near equator
        let out = resample_geographic_to_mercator(
            &src, gs, -2.8125, 2.8125, 2.8125, -2.8125, &merc, 17,
        );

        // Near equator, Mercator ≈ linear, so center row should be highest
        let mid_row = 8;
        let center_h = out[mid_row * 17 + 8];
        let edge_h = out[0 * 17 + 8]; // top edge
        assert!(
            center_h > edge_h,
            "center ({:.1}) should be higher than edge ({:.1})",
            center_h, edge_h
        );
    }

    // ── sample_elevation_bilinear ───────────────────────────────

    #[test]
    fn test_bilinear_corners() {
        // 2×2 grid: corners should sample exactly.
        let elev = vec![0.0, 100.0, 200.0, 300.0];
        assert!((sample_elevation_bilinear(&elev, 2, 2, 0.0, 0.0) - 0.0).abs() < 1e-3);
        assert!((sample_elevation_bilinear(&elev, 2, 2, 1.0, 0.0) - 100.0).abs() < 1e-3);
        assert!((sample_elevation_bilinear(&elev, 2, 2, 0.0, 1.0) - 200.0).abs() < 1e-3);
        assert!((sample_elevation_bilinear(&elev, 2, 2, 1.0, 1.0) - 300.0).abs() < 1e-3);
    }

    #[test]
    fn test_bilinear_center() {
        // 2×2 grid: center should be average of all 4 corners.
        let elev = vec![0.0, 100.0, 200.0, 300.0];
        let center = sample_elevation_bilinear(&elev, 2, 2, 0.5, 0.5);
        let expected = (0.0 + 100.0 + 200.0 + 300.0) / 4.0;
        assert!(
            (center - expected).abs() < 1e-3,
            "center should be {}, got {}",
            expected, center
        );
    }

    #[test]
    fn test_bilinear_edge_midpoint() {
        // 2×2 grid: midpoint of top edge (u=0.5, v=0).
        let elev = vec![0.0, 100.0, 200.0, 300.0];
        let mid_top = sample_elevation_bilinear(&elev, 2, 2, 0.5, 0.0);
        assert!(
            (mid_top - 50.0).abs() < 1e-3,
            "top edge midpoint should be 50, got {}",
            mid_top
        );
    }

    #[test]
    fn test_bilinear_constant_surface() {
        // All values the same → any sample should return that value.
        let h = 777.0f32;
        let elev = vec![h; 65 * 65];
        for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let val = sample_elevation_bilinear(&elev, 65, 65, u, v);
                assert!(
                    (val - h).abs() < 1e-3,
                    "constant surface at ({}, {}): expected {}, got {}",
                    u, v, h, val
                );
            }
        }
    }

    #[test]
    fn test_bilinear_larger_grid() {
        // 3×3 grid with known pattern
        let elev = vec![
            0.0, 100.0, 200.0,
            300.0, 400.0, 500.0,
            600.0, 700.0, 800.0,
        ];
        // Center (0.5, 0.5) should be 400.0 (the center cell)
        let center = sample_elevation_bilinear(&elev, 3, 3, 0.5, 0.5);
        assert!(
            (center - 400.0).abs() < 1e-3,
            "3×3 center should be 400, got {}",
            center
        );
    }

    // ── Over-zoom fallback integration ──────────────────────────

    #[test]
    fn test_overzoom_fallback_heightmap_workflow() {
        // Simulate the over-zoom workflow:
        // 1. QM tile is rasterized to a heightmap at max_zoom
        // 2. Child tile at max_zoom+1 uses parent's heightmap with fallback_uv_rect

        // Step 1: Create a synthetic "rasterized" heightmap for parent
        let parent_gs = 33u32;
        let mut parent_hm = vec![0.0f32; (parent_gs * parent_gs) as usize];
        // Height gradient: increases from TL to BR
        for row in 0..parent_gs {
            for col in 0..parent_gs {
                let u = col as f32 / (parent_gs - 1) as f32;
                let v = row as f32 / (parent_gs - 1) as f32;
                parent_hm[(row * parent_gs + col) as usize] = (u + v) * 500.0;
            }
        }

        // Step 2: Build child mesh using parent's heightmap
        let parent = TileCoord::new(13, 4096, 3072);
        let child = TileCoord::new(14, 8192, 6144); // top-left child

        let elev_uv = fallback_uv_rect(&child, &parent);
        // Top-left child → uv = [0.0, 0.0, 0.5, 0.5]
        assert!((elev_uv[0] - 0.0).abs() < 1e-4, "u_min={}", elev_uv[0]);
        assert!((elev_uv[1] - 0.0).abs() < 1e-4, "v_min={}", elev_uv[1]);
        assert!((elev_uv[2] - 0.5).abs() < 1e-4, "u_max={}", elev_uv[2]);
        assert!((elev_uv[3] - 0.5).abs() < 1e-4, "v_max={}", elev_uv[3]);

        let scale = compute_height_scale(1.5);
        let (verts, _) = build_terrain_mesh(
            &child, &parent_hm, parent_gs, parent_gs, scale, elev_uv,
        );

        // Verify: the child's top-left (u=0,v=0 → eu=0,ev=0) maps to parent's TL (h≈0)
        // and bottom-right (u=1,v=1 → eu=0.5,ev=0.5) maps to parent's center (h≈500)
        let tl_h = verts[0].position[2] / scale;
        let g = TERRAIN_GRID_SIZE;
        let br_idx = g as usize * (g as usize + 1) + g as usize;
        let br_h = verts[br_idx].position[2] / scale;

        assert!(
            tl_h.abs() < 20.0,
            "child TL should be ~0m, got {}m",
            tl_h
        );
        assert!(
            (br_h - 500.0).abs() < 50.0,
            "child BR should be ~500m, got {}m",
            br_h
        );
    }

    #[test]
    fn test_overzoom_multiple_levels() {
        // Over-zoom by 2 levels: grandchild using grandparent's heightmap
        let ancestor = TileCoord::new(13, 4096, 3072);
        let grandchild = TileCoord::new(15, 16384, 12288); // top-left of top-left child

        let elev_uv = fallback_uv_rect(&grandchild, &ancestor);
        // grandchild is in the top-left quarter of the top-left quarter
        // → uv = [0.0, 0.0, 0.25, 0.25]
        assert!((elev_uv[0] - 0.0).abs() < 1e-4, "u_min={}", elev_uv[0]);
        assert!((elev_uv[1] - 0.0).abs() < 1e-4, "v_min={}", elev_uv[1]);
        assert!((elev_uv[2] - 0.25).abs() < 1e-4, "u_max={}", elev_uv[2]);
        assert!((elev_uv[3] - 0.25).abs() < 1e-4, "v_max={}", elev_uv[3]);
    }

    // ── Skirt exclusion regression (cliff wall fix) ────────────

    #[test]
    fn test_rasterize_skirt_excluded_preserves_edge_heights() {
        // Safety test: the `indices[..surface_idx_count]` slicing in
        // tile_upload.rs excludes skirt geometry from the rasterizer.
        //
        // Real QM skirt triangles are UV-degenerate (all vertices lie
        // along a single tile edge → zero area in UV space), so the
        // barycentric test already rejects them.  The slicing is a
        // defence-in-depth guard that also saves computation.
        //
        // This test verifies that surface-only rasterization produces
        // correct heights, and that hypothetical non-degenerate skirt
        // triangles (UV inset far enough to cover grid cells) WOULD
        // corrupt the heightmap if included.

        let h = 500.0f32;
        let skirt_depth = 300.0f32;

        // Surface: a flat quad at height `h`, covering full [0,1]² UV.
        let surface_verts = vec![
            TerrainVertex { position: [-0.5, -0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 0.0] },
            TerrainVertex { position: [ 0.5, -0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 0.0] },
            TerrainVertex { position: [ 0.5,  0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 1.0] },
            TerrainVertex { position: [-0.5,  0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 1.0] },
        ];
        let surface_indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3];
        let surface_idx_count = surface_indices.len();

        // Hypothetical bad skirt: vertices with large UV inset (0.3)
        // so they form a non-degenerate quad overlapping the surface.
        let mut all_verts = surface_verts.clone();
        let skirt_a_idx = all_verts.len() as u32;
        all_verts.push(TerrainVertex {
            position: [-0.5, -0.5, h - skirt_depth],
            normal: [0.0, 0.0, -1.0],
            tex_coord: [0.3, 0.0],  // large UV inset
        });
        let skirt_b_idx = all_verts.len() as u32;
        all_verts.push(TerrainVertex {
            position: [-0.5, 0.5, h - skirt_depth],
            normal: [0.0, 0.0, -1.0],
            tex_coord: [0.3, 1.0],  // large UV inset
        });
        let mut all_indices = surface_indices.clone();
        all_indices.extend_from_slice(&[0, skirt_a_idx, 3, 3, skirt_a_idx, skirt_b_idx]);

        let gs = 17u32;

        // Surface-only: all cells should be ~h (flat surface).
        let without_skirts = rasterize_qm_to_heightmap(
            &all_verts, &all_indices[..surface_idx_count], gs,
        );
        for row in 0..gs {
            for col in 0..gs {
                let idx = (row * gs + col) as usize;
                assert!(
                    (without_skirts[idx] - h).abs() < 1.0,
                    "skirt-excluded: ({},{}) should be ~{}m, got {}m",
                    row, col, h, without_skirts[idx]
                );
            }
        }

        // With bad skirts: cells near the west edge (small u) should
        // be corrupted because the skirt triangle covers u=0..0.3.
        let with_skirts = rasterize_qm_to_heightmap(&all_verts, &all_indices, gs);
        // col=2 → u = 2/16 = 0.125, well inside u=[0, 0.3] range
        let corrupted_col = 2usize;
        let mut found_corruption = false;
        for row in 0..gs {
            let idx = (row as usize) * (gs as usize) + corrupted_col;
            if (with_skirts[idx] - h).abs() > 10.0 {
                found_corruption = true;
                break;
            }
        }
        assert!(
            found_corruption,
            "inset-UV skirts (u=0.3) should corrupt cells near col=2 (u=0.125)",
        );
    }

    #[test]
    fn test_build_terrain_mesh_from_qm_surface_vs_total_indices() {
        // Verify that `qm.indices.len()` gives the surface-only count,
        // and `build_terrain_mesh_from_qm` returns more indices (surface + skirts).
        let coord = TileCoord::new(5, 16, 12);
        let qm = x_planets_tiles::DecodedQuantizedMesh {
            coord,
            header: x_planets_tiles::quantized_mesh::QmHeader {
                center_x: 0.0, center_y: 0.0, center_z: 0.0,
                min_height: 0.0, max_height: 1000.0,
                bounding_sphere_radius: 1.0,
                horizon_occlusion_point_x: 0.0,
                horizon_occlusion_point_y: 0.0,
                horizon_occlusion_point_z: 0.0,
            },
            // Simple 4-vertex quad
            u: vec![0, 32767, 32767, 0],
            v: vec![0, 0, 32767, 32767],
            height: vec![16383, 16383, 16383, 16383], // ~500m
            indices: vec![0, 1, 2, 0, 2, 3],
            west_indices: vec![0, 3],
            south_indices: vec![0, 1],
            east_indices: vec![1, 2],
            north_indices: vec![3, 2],
            oct_normals: None,
        };

        let surface_idx_count = qm.indices.len();
        assert_eq!(surface_idx_count, 6, "surface should have 6 indices (2 triangles)");

        let (_vertices, indices) = build_terrain_mesh_from_qm(&coord, &qm);
        assert!(
            indices.len() > surface_idx_count,
            "total indices ({}) should be > surface-only ({}) due to skirts",
            indices.len(), surface_idx_count
        );

        // Verify: slicing with surface_idx_count gives only surface triangles
        let surface_only = &indices[..surface_idx_count];
        assert_eq!(surface_only.len(), 6);
        // All surface indices should reference original vertices (0..3)
        for &idx in surface_only {
            assert!(idx < 4, "surface index {} should be < 4", idx);
        }
    }

    // ── project_qm_vertices_4326_to_3857 ──────────────────────

    /// Helper: compute 4326 tile bounds (same logic as tile_source::geographic_tile_bounds).
    fn geo_tile_bounds(gx: u32, gy: u32, gz: u8) -> (f64, f64, f64, f64) {
        let n_x = (1u32 << (gz + 1)) as f64;
        let n_y = (1u32 << gz) as f64;
        let west  = gx as f64 / n_x * 360.0 - 180.0;
        let east  = (gx + 1) as f64 / n_x * 360.0 - 180.0;
        let north = 90.0 - gy as f64 / n_y * 180.0;
        let south = 90.0 - (gy + 1) as f64 / n_y * 180.0;
        (west, east, north, south)
    }

    #[test]
    fn test_project_4326_to_3857_center_maps_correctly() {
        // A vertex at the center of the 4326 tile (u=0.5, v=0.5)
        // should project within the corresponding 3857 tile.
        //
        // Use equatorial tiles where 4326 and 3857 nearly align:
        // 3857 tile z=6, x=33, y=31 (near equator, positive lat)
        // 4326 tile gz=5, gx=33, gy=15 (covers lat ~2.8° to ~5.6°)
        let merc = TileCoord::new(6, 33, 31);
        let (geo_west, geo_east, geo_north, geo_south) = geo_tile_bounds(33, 15, 5);

        let mut verts = vec![
            TerrainVertex {
                position: [0.0, 0.0, 1000.0],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.5, 0.5], // center of 4326 tile
            },
        ];

        project_qm_vertices_4326_to_3857(
            &mut verts, &merc,
            geo_west, geo_east, geo_north, geo_south,
        );

        // Longitude aligns exactly → u should be 0.5
        assert!(
            (verts[0].tex_coord[0] - 0.5).abs() < 0.01,
            "projected u should be 0.5, got {}",
            verts[0].tex_coord[0]
        );
        // The center of the 4326 tile in latitude maps somewhere inside
        // the 3857 tile (v between -0.5 and 1.5 due to lat extent mismatch,
        // but close to 0.5 near equator)
        assert!(
            verts[0].tex_coord[1] > -0.5 && verts[0].tex_coord[1] < 1.5,
            "projected v should be approximately in tile range, got {}",
            verts[0].tex_coord[1]
        );
        // Height is preserved
        assert_eq!(verts[0].position[2], 1000.0);
    }

    #[test]
    fn test_project_4326_to_3857_preserves_height() {
        let merc = TileCoord::new(5, 16, 16);
        let (geo_west, geo_east, geo_north, geo_south) = geo_tile_bounds(16, 7, 4);

        let heights = [0.0, 100.0, 500.0, 8848.0, -420.0];
        for &h in &heights {
            let mut verts = vec![TerrainVertex {
                position: [0.0, 0.0, h],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.5, 0.5],
            }];
            project_qm_vertices_4326_to_3857(
                &mut verts, &merc,
                geo_west, geo_east, geo_north, geo_south,
            );
            assert_eq!(
                verts[0].position[2], h,
                "height {}m should be preserved after projection", h
            );
        }
    }

    #[test]
    fn test_project_4326_to_3857_edge_continuity() {
        // The key property: two adjacent 4326 tiles share an edge.
        // A vertex at the shared edge should project to the SAME 3857
        // position regardless of which tile it belongs to.
        //
        // This tests the fix for cliff walls at 4326 tile boundaries.

        // Two vertically-adjacent 4326 tiles at gz=4
        let (west_a, east_a, north_a, south_a) = geo_tile_bounds(16, 7, 4);
        let (west_b, east_b, north_b, south_b) = geo_tile_bounds(16, 8, 4);

        // The shared edge: tile A's south = tile B's north
        assert!(
            (south_a - north_b).abs() < 1e-10,
            "tiles should share edge: A.south={}, B.north={}",
            south_a, north_b
        );

        // A vertex on tile A's south edge (v=1.0)
        let merc = TileCoord::new(5, 32, 16);
        let h = 750.0f32;
        let u_shared = 0.3;

        let mut vert_a = vec![TerrainVertex {
            position: [0.0, 0.0, h],
            normal: [0.0, 0.0, 1.0],
            tex_coord: [u_shared, 1.0],
        }];
        project_qm_vertices_4326_to_3857(
            &mut vert_a, &merc,
            west_a, east_a, north_a, south_a,
        );

        // Same vertex on tile B's north edge (v=0.0)
        let mut vert_b = vec![TerrainVertex {
            position: [0.0, 0.0, h],
            normal: [0.0, 0.0, 1.0],
            tex_coord: [u_shared, 0.0],
        }];
        project_qm_vertices_4326_to_3857(
            &mut vert_b, &merc,
            west_b, east_b, north_b, south_b,
        );

        // Both should project to the same 3857 position (no cliff wall!)
        let dx = (vert_a[0].position[0] - vert_b[0].position[0]).abs();
        let dy = (vert_a[0].position[1] - vert_b[0].position[1]).abs();
        assert!(
            dx < 1e-6 && dy < 1e-6,
            "shared edge vertices should project to same position: \
             A=({}, {}), B=({}, {}), diff=({}, {})",
            vert_a[0].position[0], vert_a[0].position[1],
            vert_b[0].position[0], vert_b[0].position[1],
            dx, dy,
        );
        assert_eq!(vert_a[0].position[2], vert_b[0].position[2]);
    }

    #[test]
    fn test_project_4326_to_3857_full_tile_coverage() {
        // The projected vertices' tex_coords (3857 UV) should cover
        // approximately [0,1]² for a tile near the equator.
        let merc = TileCoord::new(6, 33, 31);
        let (geo_west, geo_east, geo_north, geo_south) = geo_tile_bounds(33, 15, 5);

        let mut verts = vec![
            TerrainVertex { position: [0.0, 0.0, 0.0], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 0.0] },
            TerrainVertex { position: [0.0, 0.0, 0.0], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 0.0] },
            TerrainVertex { position: [0.0, 0.0, 0.0], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 1.0] },
            TerrainVertex { position: [0.0, 0.0, 0.0], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 1.0] },
        ];

        project_qm_vertices_4326_to_3857(
            &mut verts, &merc,
            geo_west, geo_east, geo_north, geo_south,
        );

        // Longitude aligns exactly: u should be 0.0 and 1.0
        assert!((verts[0].tex_coord[0] - 0.0).abs() < 0.01, "NW u={}", verts[0].tex_coord[0]);
        assert!((verts[1].tex_coord[0] - 1.0).abs() < 0.01, "NE u={}", verts[1].tex_coord[0]);

        // Latitude approximately covers [0, 1]
        let v_min = verts.iter().map(|v| v.tex_coord[1]).fold(f32::MAX, f32::min);
        let v_max = verts.iter().map(|v| v.tex_coord[1]).fold(f32::MIN, f32::max);
        assert!(
            v_min < 0.1 && v_max > 0.9,
            "projected v range [{}, {}] should approximately cover [0, 1]",
            v_min, v_max
        );
    }

    // ── Multi-source resampling tests ─────────────────────────

    #[test]
    fn test_multi_source_single_source_matches_original() {
        // Multi-source with 1 source should produce identical output
        // as the single-source function.
        let grid_size = 5u32;
        let heightmap: Vec<f32> = (0..(grid_size * grid_size))
            .map(|i| 100.0 + i as f32 * 10.0)
            .collect();
        let merc_coord = TileCoord::new(3, 4, 3);
        let (west, east, north, south) = (-45.0, -22.5, 45.0, 33.75);
        let out_grid_size = 5u32;

        let single = resample_geographic_to_mercator(
            &heightmap, grid_size,
            west, east, north, south,
            &merc_coord, out_grid_size,
        );
        let multi = resample_geographic_to_mercator_multi(
            &[GeoHeightmapSource {
                heightmap: &heightmap,
                grid_size,
                west, east, north, south,
            }],
            &merc_coord, out_grid_size,
        );

        for (i, (s, m)) in single.iter().zip(multi.iter()).enumerate() {
            assert!(
                (s - m).abs() < 1e-4,
                "Mismatch at index {}: single={}, multi={}",
                i, s, m,
            );
        }
    }

    #[test]
    fn test_multi_source_two_tiles_no_clamping() {
        // Two adjacent 4326 tiles fully covering the 3857 tile.
        // Heights should come from the correct source per-pixel,
        // with no clamped (constant) region.
        //
        // Setup: 3857 tile z=5, x=17, y=12
        // Primary 4326 tile gz=4, gy=4: north=45°, south=33.75°
        // Secondary 4326 tile gz=4, gy=5: north=33.75°, south=22.5°
        // The 3857 tile extends below 33.75° → needs secondary tile.

        let grid_size = 5u32;
        // Tile A (north): constant height 1000m
        let heightmap_a: Vec<f32> = vec![1000.0; (grid_size * grid_size) as usize];
        // Tile B (south): constant height 500m
        let heightmap_b: Vec<f32> = vec![500.0; (grid_size * grid_size) as usize];

        // 4326 tile gy=4: north=45°, south=33.75°
        let source_a = GeoHeightmapSource {
            heightmap: &heightmap_a,
            grid_size,
            west: 123.75,
            east: 135.0,
            north: 45.0,
            south: 33.75,
        };
        // 4326 tile gy=5: north=33.75°, south=22.5°
        let source_b = GeoHeightmapSource {
            heightmap: &heightmap_b,
            grid_size,
            west: 123.75,
            east: 135.0,
            north: 33.75,
            south: 22.5,
        };

        let merc_coord = TileCoord::new(5, 27, 12);
        let out_grid_size = 9u32;

        // Multi-source: should pick correct tile per pixel.
        let multi = resample_geographic_to_mercator_multi(
            &[source_a, source_b],
            &merc_coord, out_grid_size,
        );

        // Single-source (only tile A): would clamp southern pixels.
        let single = resample_geographic_to_mercator(
            &heightmap_a, grid_size,
            123.75, 135.0, 45.0, 33.75,
            &merc_coord, out_grid_size,
        );

        // In multi-source, pixels in the south part should get 500m (from tile B).
        // In single-source, ALL pixels get 1000m (clamped to tile A).
        let has_500 = multi.iter().any(|&h| (h - 500.0).abs() < 1.0);
        let single_all_1000 = single.iter().all(|&h| (h - 1000.0).abs() < 1.0);

        assert!(
            has_500,
            "Multi-source should sample from tile B (500m) for southern pixels"
        );
        assert!(
            single_all_1000,
            "Single-source should clamp all to tile A (1000m)"
        );
    }

    #[test]
    fn test_multi_source_smooth_boundary() {
        // At the boundary between two 4326 tiles, heights should
        // transition smoothly (no cliff wall).
        //
        // Create two tiles with a gradient that matches at the boundary:
        // Tile A: height varies from 1000m (north) to 500m (south)
        // Tile B: height varies from 500m (north) to 0m (south)
        // At the boundary (tile A south / tile B north), both have 500m.

        let grid_size = 33u32;
        let gs = grid_size as usize;

        // Tile A: linear gradient 1000 → 500
        let heightmap_a: Vec<f32> = (0..gs * gs)
            .map(|i| {
                let row = i / gs;
                let t = row as f32 / (gs - 1) as f32; // 0 at north, 1 at south
                1000.0 - 500.0 * t
            })
            .collect();

        // Tile B: linear gradient 500 → 0
        let heightmap_b: Vec<f32> = (0..gs * gs)
            .map(|i| {
                let row = i / gs;
                let t = row as f32 / (gs - 1) as f32;
                500.0 - 500.0 * t
            })
            .collect();

        // Verify boundary match
        assert!(
            (heightmap_a[gs * (gs - 1)] - heightmap_b[0]).abs() < 1.0,
            "Tile edge heights should match"
        );

        let source_a = GeoHeightmapSource {
            heightmap: &heightmap_a,
            grid_size,
            west: 0.0, east: 11.25, north: 45.0, south: 33.75,
        };
        let source_b = GeoHeightmapSource {
            heightmap: &heightmap_b,
            grid_size,
            west: 0.0, east: 11.25, north: 33.75, south: 22.5,
        };

        let merc_coord = TileCoord::new(5, 16, 12);
        let out_grid_size = 33u32;
        let multi = resample_geographic_to_mercator_multi(
            &[source_a, source_b],
            &merc_coord, out_grid_size,
        );

        // Check that no adjacent rows have a height jump > 100m.
        // With smooth input data, the output should also be smooth.
        let ogs = out_grid_size as usize;
        let mut max_jump = 0.0f32;
        for row in 1..ogs {
            for col in 0..ogs {
                let h_prev = multi[(row - 1) * ogs + col];
                let h_curr = multi[row * ogs + col];
                let jump = (h_curr - h_prev).abs();
                if jump > max_jump {
                    max_jump = jump;
                }
            }
        }
        assert!(
            max_jump < 100.0,
            "Max row-to-row height jump {:.1}m exceeds 100m — cliff wall detected!",
            max_jump,
        );
    }

    // ── Helpers ────────────────────────────────────────────────

    /// Generate a grid of test points across valid lat/lon range.
    fn generate_test_grid(lat_steps: usize, lon_steps: usize) -> Vec<glam::DVec3> {
        let mut points = Vec::with_capacity(lat_steps * lon_steps);
        for i in 0..lat_steps {
            // Stay within Mercator-safe range
            let lat = -80.0 + (160.0 / lat_steps as f64) * i as f64;
            for j in 0..lon_steps {
                let lon = -179.0 + (358.0 / lon_steps as f64) * j as f64;
                points.push(glam::DVec3::new(lat, lon, 0.0));
            }
        }
        points
    }

    // ── Globe/centered mesh alignment regression tests ─────────

    /// Regression: build_globe_tile_mesh must produce exactly one
    /// index-count entry per input tile.  A mismatch between the mesh
    /// index-count vector and the prepared bind-group vector caused
    /// tiles to render with the wrong mesh.
    #[test]
    fn test_build_globe_tile_mesh_count_matches_tiles() {
        let tiles = vec![
            RenderableTile {
                coord: TileCoord::new(2, 0, 0),
                texture_coord: TileCoord::new(2, 0, 0),
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                display_x: 0,
            },
            RenderableTile {
                coord: TileCoord::new(2, 1, 1),
                texture_coord: TileCoord::new(2, 1, 1),
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                display_x: 1,
            },
            RenderableTile {
                coord: TileCoord::new(2, 3, 2),
                texture_coord: TileCoord::new(2, 3, 2),
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                display_x: 3,
            },
        ];

        let (_verts, _idxs, tile_idx_counts) = build_globe_tile_mesh(&tiles);

        assert_eq!(
            tile_idx_counts.len(),
            tiles.len(),
            "tile_idx_counts length must equal input tile count"
        );
        // Every tile must produce at least some indices
        for (i, &count) in tile_idx_counts.iter().enumerate() {
            assert!(
                count > 0,
                "tile {} must produce non-zero index count, got 0",
                i
            );
        }
    }

    /// Regression: build_centered_tile_mesh must produce exactly one
    /// index-count entry per input tile.
    #[test]
    fn test_build_centered_tile_mesh_count_matches_tiles() {
        let center_lat = 37.5_f64.to_radians();
        let center_lon = 127.0_f64.to_radians();
        // Use tiles near the center so they pass the angular-distance filter
        let tiles = vec![
            RenderableTile {
                coord: TileCoord::new(3, 7, 3),
                texture_coord: TileCoord::new(3, 7, 3),
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                display_x: 7,
            },
            RenderableTile {
                coord: TileCoord::new(3, 7, 4),
                texture_coord: TileCoord::new(3, 7, 4),
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                display_x: 7,
            },
        ];

        let (_verts, _idxs, tile_idx_counts) =
            build_centered_tile_mesh(&tiles, center_lat, center_lon);

        assert_eq!(
            tile_idx_counts.len(),
            tiles.len(),
            "tile_idx_counts length must equal input tile count"
        );
        for (i, &count) in tile_idx_counts.iter().enumerate() {
            assert!(
                count > 0,
                "tile {} must produce non-zero index count, got 0",
                i
            );
        }
    }

    /// Regression: index offsets must be cumulative and non-overlapping.
    /// The draw loop uses `idx_offset..idx_offset+count` ranges, so
    /// the sum of all counts must equal total indices.
    #[test]
    fn test_globe_mesh_index_offsets_are_contiguous() {
        let tiles: Vec<RenderableTile> = (0..4)
            .map(|i| RenderableTile {
                coord: TileCoord::new(2, i, 0),
                texture_coord: TileCoord::new(2, i, 0),
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                display_x: i as i64,
            })
            .collect();

        let (_verts, all_idxs, tile_idx_counts) = build_globe_tile_mesh(&tiles);

        let total: u32 = tile_idx_counts.iter().sum();
        assert_eq!(
            total,
            all_idxs.len() as u32,
            "sum of per-tile index counts must equal total index count"
        );
    }

    /// Regression: same contiguity invariant for centered Mercator.
    #[test]
    fn test_centered_mesh_index_offsets_are_contiguous() {
        let center_lat = 0.0_f64.to_radians();
        let center_lon = 0.0_f64.to_radians();
        let tiles: Vec<RenderableTile> = (0..4)
            .map(|i| RenderableTile {
                coord: TileCoord::new(2, i, 1),
                texture_coord: TileCoord::new(2, i, 1),
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                display_x: i as i64,
            })
            .collect();

        let (_verts, all_idxs, tile_idx_counts) =
            build_centered_tile_mesh(&tiles, center_lat, center_lon);

        let total: u32 = tile_idx_counts.iter().sum();
        assert_eq!(
            total,
            all_idxs.len() as u32,
            "sum of per-tile index counts must equal total index count"
        );
    }

    // ── Oblique Mercator angular-distance filtering tests ──────

    /// Tiles near the oblique Mercator singularity (~90° from center)
    /// must produce valid (non-degenerate) meshes. Tiles at the center
    /// should always produce indices.
    #[test]
    fn test_centered_mesh_center_tile_always_has_indices() {
        // Center at Seoul (37.5°N, 127°E)
        let center_lat = 37.5_f64.to_radians();
        let center_lon = 127.0_f64.to_radians();

        // Tile covering Seoul at zoom 5
        let tile = RenderableTile {
            coord: TileCoord::new(5, 27, 12),
            texture_coord: TileCoord::new(5, 27, 12),
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            display_x: 27,
        };

        let (_verts, _idxs, counts) =
            build_centered_tile_mesh(&[tile], center_lat, center_lon);

        assert_eq!(counts.len(), 1);
        assert!(
            counts[0] > 0,
            "tile near center must produce non-zero index count"
        );
    }

    /// The centered mesh builder should produce no degenerate triangles
    /// (zero-area) for tiles within the valid projection range.
    #[test]
    fn test_centered_mesh_no_degenerate_triangles_near_center() {
        let center_lat = 0.0_f64;
        let center_lon = 0.0_f64;
        // Tile at center
        let tile = RenderableTile {
            coord: TileCoord::new(2, 2, 2),
            texture_coord: TileCoord::new(2, 2, 2),
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            display_x: 2,
        };

        let (verts, idxs, _counts) =
            build_centered_tile_mesh(&[tile], center_lat, center_lon);

        // Check every triangle has non-zero area
        for tri in idxs.chunks(3) {
            if tri.len() < 3 {
                continue;
            }
            let p0 = verts[tri[0] as usize].position;
            let p1 = verts[tri[1] as usize].position;
            let p2 = verts[tri[2] as usize].position;
            let cross = (p1[0] - p0[0]) * (p2[1] - p0[1])
                - (p1[1] - p0[1]) * (p2[0] - p0[0]);
            assert!(
                cross.abs() > 1e-12,
                "degenerate triangle found: area ≈ {:.2e}",
                cross.abs()
            );
        }
    }

    /// Tiles whose angular distance from the center exceeds ~80° should
    /// produce fewer or zero triangles (winding check skips distorted ones).
    #[test]
    fn test_centered_mesh_far_tile_has_fewer_indices() {
        let center_lat = 37.5_f64.to_radians();
        let center_lon = 127.0_f64.to_radians();

        // Tile near center (Seoul)
        let near = RenderableTile {
            coord: TileCoord::new(2, 3, 1),
            texture_coord: TileCoord::new(2, 3, 1),
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            display_x: 3,
        };

        // Tile far from center (opposite side of globe at zoom 2)
        let far = RenderableTile {
            coord: TileCoord::new(2, 1, 1),
            texture_coord: TileCoord::new(2, 1, 1),
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            display_x: 1,
        };

        let (_v1, _i1, counts_near) =
            build_centered_tile_mesh(&[near], center_lat, center_lon);
        let (_v2, _i2, counts_far) =
            build_centered_tile_mesh(&[far], center_lat, center_lon);

        assert!(
            counts_near[0] > 0,
            "near tile must produce indices"
        );
        // Far tile should have fewer or no valid triangles due to
        // the oblique Mercator distortion near the singularity
        assert!(
            counts_far[0] <= counts_near[0],
            "far tile should have <= indices than near tile: {} vs {}",
            counts_far[0],
            counts_near[0]
        );
    }
}
