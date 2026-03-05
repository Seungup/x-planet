//! Stage 4: Tile uniforms, fallback UV resolution, and RenderableTile.

use std::collections::HashSet;
use x_planets_math::{TileCoord, TileUniforms, VisibleTile};

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

// ───────────────────────────────────────────────────────────────────
// Stage 4: Viewport → GPU uniform data
// ───────────────────────────────────────────────────────────────────

/// Compute the per-frame viewport uniform buffer data.
///
/// Pure function.
pub fn viewport_uniforms(viewport: &crate::viewport::Viewport) -> x_planets_math::ViewportUniforms {
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
        _ => (rt.coord.y as f64 + 0.5) / n_f64,
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
// Globe (3D sphere) uniforms
// ───────────────────────────────────────────────────────────────────

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

    let tile_center_3d = super::tile_mesh::globe_tile_center(&rt.coord, rt.display_x);
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
// Centered Mercator uniforms
// ───────────────────────────────────────────────────────────────────

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
        super::tile_mesh::centered_tile_center(&rt.coord, rt.display_x, center_lat_rad, center_lon_rad);
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
// Fallback UV resolution
// ───────────────────────────────────────────────────────────────────

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
