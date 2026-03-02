//! Pure-function pipeline stages.
//!
//! Karpathy principle: every stage is a pure function.
//! Input → Output. No side effects. Independently testable.
//!
//! The MapEngine orchestrates these stages, but each stage
//! knows nothing about the engine or GPU.

use std::collections::HashSet;
use x_planets_math::{GeoCoord, TileCoord, TileUniforms, ViewportUniforms};

use crate::render::{tile_quad_vertices, TileVertex, TILE_QUAD_INDICES};
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
pub fn visible_tiles(viewport: &Viewport) -> Vec<TileCoord> {
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
    let mut vertices = Vec::with_capacity(tiles.len() * 4);
    let mut indices = Vec::with_capacity(tiles.len() * 6);

    for (i, tile) in tiles.iter().enumerate() {
        let base = (i * 4) as u32;
        let quad = tile_quad_vertices(tile);
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

// ───────────────────────────────────────────────────────────────────
// Fallback texture resolution
// ───────────────────────────────────────────────────────────────────

/// A tile ready for rendering, with resolved texture source.
#[derive(Debug, Clone)]
pub struct RenderableTile {
    /// The tile position to render (geometry).
    pub coord: TileCoord,
    /// The tile whose texture to use (may be an ancestor).
    pub texture_coord: TileCoord,
    /// UV sub-rectangle within texture_coord's texture.
    /// `[0, 0, 1, 1]` = full texture (own texture available).
    pub uv_rect: [f32; 4],
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
///
/// Pure function.
pub fn resolve_fallbacks(
    visible: &[TileCoord],
    available_textures: &HashSet<TileCoord>,
) -> Vec<RenderableTile> {
    visible
        .iter()
        .filter_map(|tile| {
            let mut cur = *tile;
            loop {
                if available_textures.contains(&cur) {
                    let uv = fallback_uv_rect(tile, &cur);
                    return Some(RenderableTile {
                        coord: *tile,
                        texture_coord: cur,
                        uv_rect: uv,
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
    let mut normals = vec![[0.0f32, 0.0, 1.0]; vert_count];
    let vs = verts_per_side as usize;

    for gy in 0..vs {
        for gx in 0..vs {
            let idx = gy * vs + gx;
            let p = positions[idx];

            // Finite-difference neighbors (clamped at edges)
            let left = if gx > 0 { positions[idx - 1] } else { p };
            let right = if gx + 1 < vs { positions[idx + 1] } else { p };
            let up = if gy > 0 { positions[idx - vs] } else { p };
            let down = if gy + 1 < vs { positions[idx + vs] } else { p };

            // Tangent vectors
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
    for gy in 0..grid {
        for gx in 0..grid {
            let tl = gy * verts_per_side + gx;
            let tr = tl + 1;
            let bl = tl + verts_per_side;
            let br = bl + 1;

            indices.push(tl);
            indices.push(tr);
            indices.push(bl);
            indices.push(bl);
            indices.push(tr);
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
        strip.push((grid * verts_per_side + gx) as u32);
    }
    edge_strips.push(strip);

    // Top edge (gy=0, left to right)
    let mut strip = Vec::new();
    for gx in 0..verts_per_side {
        strip.push(gx as u32);
    }
    edge_strips.push(strip);

    // Right edge (gx=last, top to bottom)
    let mut strip = Vec::new();
    for gy in 0..verts_per_side {
        strip.push((gy * verts_per_side + grid) as u32);
    }
    edge_strips.push(strip);

    // Left edge (gx=0, top to bottom)
    let mut strip = Vec::new();
    for gy in 0..verts_per_side {
        strip.push((gy * verts_per_side) as u32);
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

            // Two triangles: top_a, top_b, skirt_a  +  skirt_a, top_b, skirt_b
            indices.push(edge[i]);
            indices.push(edge[i + 1]);
            indices.push(skirt_a);
            indices.push(skirt_a);
            indices.push(edge[i + 1]);
            indices.push(skirt_b);
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
            tiles.contains(&TileCoord::new(0, 0, 0)),
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
        let visible = vec![
            TileCoord::new(2, 0, 0),
            TileCoord::new(2, 1, 0),
            TileCoord::new(2, 0, 1),
        ];
        let available: HashSet<TileCoord> = visible.iter().copied().collect();
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

        let visible = vec![child];
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
        let visible = vec![TileCoord::new(5, 10, 10)];
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
    fn test_compute_height_scale() {
        let scale = compute_height_scale(1.0);
        // scale = 1 / 40_075_000
        let expected = (1.0 / 40_075_000.0_f64) as f32;
        assert!((scale - expected).abs() < 1e-12);

        // Linearity: 2× exaggeration → 2× scale
        let scale_2x = compute_height_scale(2.0);
        assert!((scale_2x - 2.0 * scale).abs() < 1e-12);
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
}
