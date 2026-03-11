//! Stage 3: Tile coords → GPU vertex/index data (flat, globe, centered Mercator).

use x_planets_math::TileCoord;

use crate::render::{
    tile_quad_vertices_projected, GlobeTileVertex, TileVertex, TILE_QUAD_INDICES,
    tile_globe_mesh, tile_centered_mesh, polar_cap_mesh,
};

use super::tile_uniforms::RenderableTile;

// ───────────────────────────────────────────────────────────────────
// Flat Mercator mesh
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
// Globe (3D sphere) mesh
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

// ───────────────────────────────────────────────────────────────────
// Viewport-centered (oblique) Mercator mesh
// ───────────────────────────────────────────────────────────────────

/// Compute tile center in oblique (viewport-centered) Mercator space.
pub(crate) fn centered_tile_center(
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

/// Angular-distance threshold (degrees) for the oblique Mercator
/// pre-filter.  The fragment shader performs per-pixel small-circle
/// clipping at 85°, so this CPU-side filter is only needed to avoid
/// projecting tiles that are completely beyond the visible hemisphere
/// (which would waste GPU work and produce degenerate geometry near
/// the 90° singularity).
///
/// Returns the threshold in degrees.
pub fn centered_angular_threshold_deg(_zoom: f64) -> f64 {
    // Must match the shader's clip_sphere angle (85°).
    // Tiles between 85°–90° produce degenerate oblique Mercator vertex
    // positions.  Even though the shader clips fragments beyond 85°,
    // triangles straddling the boundary interpolate between correct and
    // wildly distorted vertices, creating massive rendering artifacts
    // (stretched tiles, elevated walls).
    85.0
}

/// Returns `true` if a tile passes the angular-distance pre-filter
/// for the centered Mercator rendering path.
///
/// Pure function.  Mirrors the filter in `TileRenderer::render_frame_layered_projected`.
pub fn tile_passes_angular_filter(
    tile: &RenderableTile,
    viewport_center_lat_rad: f64,
    viewport_center_lon_rad: f64,
    zoom: f64,
) -> bool {
    let center_sphere =
        x_planets_math::geo_to_unit_sphere(viewport_center_lat_rad, viewport_center_lon_rad);
    let cos_threshold = centered_angular_threshold_deg(zoom).to_radians().cos();

    let n = tile.coord.extent() as f64;

    // Check tile center AND all four corners.  At low zoom levels tiles
    // are very large in geographic extent, so the center can be beyond
    // the threshold even though a large portion of the tile is within it.
    // If ANY sample point is within the threshold, keep the tile.
    let sample_points: [(f64, f64); 5] = [
        // center
        ((tile.display_x as f64 + 0.5) / n, (tile.coord.y as f64 + 0.5) / n),
        // corners
        (tile.display_x as f64 / n, tile.coord.y as f64 / n),
        ((tile.display_x + 1) as f64 / n, tile.coord.y as f64 / n),
        (tile.display_x as f64 / n, (tile.coord.y + 1) as f64 / n),
        ((tile.display_x + 1) as f64 / n, (tile.coord.y + 1) as f64 / n),
    ];

    for &(mx, my) in &sample_points {
        let lon_rad = (mx * 2.0 - 1.0) * std::f64::consts::PI;
        let lat_rad = x_planets_math::mercator_y_to_lat_rad(my);
        let tile_sphere = x_planets_math::geo_to_unit_sphere(lat_rad, lon_rad);
        let cos_angle = center_sphere.dot(tile_sphere);
        if cos_angle > cos_threshold {
            return true;
        }
    }
    false
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

// ───────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify centered tile meshes produce valid (finite, non-NaN) vertices
    /// at zoom=15 with pitch=60 — the 3D buildings preset scenario.
    #[test]
    fn test_centered_mesh_valid_at_high_zoom_pitch() {
        use crate::viewport::Viewport;
        use x_planets_math::GeoCoord;

        let mut vp = Viewport::new(1920, 1080);
        vp.center = GeoCoord::new(40.6892, -74.0445);
        vp.zoom = 15.0;
        vp.pitch = 60.0;

        let center_lat_rad = vp.center.lat.to_radians();
        let center_lon_rad = vp.center.lon.to_radians();

        let tiles = vp.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);
        assert!(!tiles.is_empty(), "Should have visible tiles at zoom=15 pitch=60");

        let renderables: Vec<RenderableTile> = tiles.iter().map(|vt| RenderableTile {
            coord: vt.coord,
            texture_coord: vt.coord,
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            display_x: vt.display_x,
        }).collect();

        let (verts, idxs, counts) =
            build_centered_tile_mesh(&renderables, center_lat_rad, center_lon_rad);

        assert!(!verts.is_empty(), "Should produce vertices");
        assert!(!idxs.is_empty(), "Should produce indices");
        assert_eq!(counts.len(), renderables.len(), "One count per tile");

        // Check all vertices are finite
        let mut nan_count = 0;
        for v in &verts {
            for &p in &v.position {
                if !p.is_finite() { nan_count += 1; }
            }
            for &s in &v.sphere_pos {
                if !s.is_finite() { nan_count += 1; }
            }
        }
        assert_eq!(nan_count, 0, "All vertex positions and sphere_pos must be finite");
    }
}
