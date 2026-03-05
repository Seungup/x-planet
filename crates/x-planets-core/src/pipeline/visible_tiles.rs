//! Stage 1–2: Viewport → visible tile list → load requests.

use std::collections::HashSet;
use x_planets_math::{GeoCoord, TileCoord, VisibleTile};

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
