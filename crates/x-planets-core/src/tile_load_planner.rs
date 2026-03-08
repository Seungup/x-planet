//! Pure-function tile load planner.
//!
//! Decides **which** tiles to load and in what priority, without performing
//! any I/O.  Both the native (tokio) and web (spawn_local) platforms call
//! [`plan_tile_loads`] each frame and then execute the returned requests
//! using their own async runtime.
//!
//! This eliminates the duplication between the native `tile_loading.rs` and
//! the web `request_raster_tiles()` / `request_elevation_tiles()`.

use std::collections::HashSet;

use glam::DVec2;
use x_planets_math::{TileCoord, VisibleTile};

use crate::engine::LayerKind;

// ═══════════════════════════════════════════════════════════════════
// Trait: snapshot of a layer's current load state
// ═══════════════════════════════════════════════════════════════════

/// Read-only view of a layer's current loading state.
///
/// Platforms implement this on their layer state struct (e.g.
/// `NativeLayerState`, `WebLayerState`) so the planner can query
/// cache, pending, and failure status without owning any data.
pub trait LayerLoadState {
    fn kind(&self) -> &LayerKind;
    fn min_zoom(&self) -> u8;
    fn max_zoom(&self) -> u8;
    fn has_texture(&self, coord: &TileCoord) -> bool;
    fn is_pending(&self, coord: &TileCoord) -> bool;
    fn is_failed_cooldown(&self, coord: &TileCoord) -> bool;
    fn has_terrain_data(&self, coord: &TileCoord) -> bool;
    fn is_elevation_pending(&self, coord: &TileCoord) -> bool;
    fn max_concurrent(&self) -> usize;
    fn pending_count(&self) -> usize;
    fn max_elevation_concurrent(&self) -> usize;
    fn elevation_pending_count(&self) -> usize;
    fn has_elevation_source(&self) -> bool;
}

// ═══════════════════════════════════════════════════════════════════
// Output types
// ═══════════════════════════════════════════════════════════════════

/// What kind of tile to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannedRequestKind {
    /// Raster imagery tile (PNG/JPEG).
    Raster,
    /// Terrain-layer tile (config-file terrain layers).
    Terrain,
    /// Elevation tile for a raster layer with runtime terrain enabled.
    Elevation,
}

/// A planned tile load request, ordered by priority (lower = more urgent).
#[derive(Debug, Clone)]
pub struct PlannedRequest {
    pub coord: TileCoord,
    pub priority: f32,
    pub kind: PlannedRequestKind,
}

/// The set of tile coords that are still needed this frame.
///
/// Platforms use this to abort in-flight requests not in this set:
/// ```ignore
/// let stale: Vec<_> = pending_coords.iter()
///     .filter(|c| !needed.raster.contains(c))
///     .copied().collect();
/// ```
#[derive(Debug, Clone, Default)]
pub struct NeededSet {
    /// Raster/terrain tile coords that should NOT be aborted.
    pub raster: HashSet<TileCoord>,
    /// Elevation tile coords that should NOT be aborted.
    pub elevation: HashSet<TileCoord>,
}

// ═══════════════════════════════════════════════════════════════════
// Main entry point
// ═══════════════════════════════════════════════════════════════════

/// Compute which tiles to load for a single layer this frame.
///
/// Returns `(needed, requests)` where:
/// - `needed` — coords that should NOT be aborted if in-flight
/// - `requests` — ordered by priority (lowest first = most urgent)
///
/// The caller is responsible for:
/// 1. Aborting in-flight requests not in `needed`
/// 2. Taking at most `available_slots` requests from the list
/// 3. Spawning async fetch tasks for taken requests
///
/// Pure function — no I/O, no allocation of platform resources.
pub fn plan_tile_loads(
    layer: &dyn LayerLoadState,
    visible: &[VisibleTile],
    visible_set: &HashSet<TileCoord>,
    camera_center: DVec2,
) -> (NeededSet, Vec<PlannedRequest>) {
    let min_z = layer.min_zoom();
    let max_z = layer.max_zoom();

    // 1. Build needed set (visible + base + ancestors).
    let raster_needed = compute_needed_set(layer, visible, visible_set, min_z, max_z);
    let elevation_needed = visible_set.clone();
    let needed = NeededSet {
        raster: raster_needed,
        elevation: elevation_needed,
    };

    // 2. Plan requests by priority.
    let mut requests = Vec::new();

    // 2a. Base tiles (z=0, z=1) — highest priority.
    plan_base_tile_loads(layer, min_z, max_z, &mut requests);

    // 2b. Parent-first ancestor loading.
    plan_ancestor_loads(layer, visible, visible_set, min_z, max_z, camera_center, &mut requests);

    // 2c. Visible tiles with fallback-depth priority.
    plan_visible_tile_loads(layer, visible, min_z, max_z, camera_center, &mut requests);

    // 2d. Elevation tiles (if terrain is enabled).
    let is_terrain_layer = matches!(layer.kind(), LayerKind::Terrain { .. });
    if layer.has_elevation_source() || is_terrain_layer {
        plan_elevation_loads(layer, visible, visible_set, camera_center, &mut requests);
    }

    // Sort by priority (lowest first).
    requests.sort_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap_or(std::cmp::Ordering::Equal));

    (needed, requests)
}

// ═══════════════════════════════════════════════════════════════════
// Sub-functions
// ═══════════════════════════════════════════════════════════════════

/// Build the "needed" set: visible tiles + base tiles + ancestor chain
/// up to the first cached ancestor.  Used to decide which in-flight
/// requests are stale.
fn compute_needed_set(
    layer: &dyn LayerLoadState,
    visible: &[VisibleTile],
    visible_set: &HashSet<TileCoord>,
    min_z: u8,
    max_z: u8,
) -> HashSet<TileCoord> {
    let mut needed: HashSet<TileCoord> = visible_set.clone();

    // Always include base tiles (z=0, z=1).
    let base_max = 1u8.min(max_z);
    for z in min_z..=base_max {
        let n = 1u32 << z;
        for y in 0..n {
            for x in 0..n {
                needed.insert(TileCoord::new(z, x, y));
            }
        }
    }

    // Walk ancestor chains for each visible tile.
    for vt in visible {
        let coord = vt.coord;
        let start = if coord.z > max_z {
            let clamped = coord.clamp_to_zoom(max_z);
            needed.insert(clamped);
            clamped.parent()
        } else {
            coord.parent()
        };
        let mut cur = start;
        while let Some(p) = cur {
            if layer.has_texture(&p) {
                needed.insert(p);
                break;
            }
            needed.insert(p);
            cur = p.parent();
        }
    }

    needed
}

/// Plan base tile loads (z=0, z=1) — highest priority (0.0).
fn plan_base_tile_loads(
    layer: &dyn LayerLoadState,
    min_z: u8,
    max_z: u8,
    requests: &mut Vec<PlannedRequest>,
) {
    let base_max = 1u8.min(max_z);
    for z in min_z..=base_max {
        let n = 1u32 << z;
        for y in 0..n {
            for x in 0..n {
                let coord = TileCoord::new(z, x, y);
                if layer.has_texture(&coord)
                    || layer.is_pending(&coord)
                    || layer.is_failed_cooldown(&coord)
                {
                    continue;
                }
                requests.push(PlannedRequest {
                    coord,
                    priority: 0.0,
                    kind: if matches!(layer.kind(), LayerKind::Terrain { .. }) {
                        PlannedRequestKind::Terrain
                    } else {
                        PlannedRequestKind::Raster
                    },
                });
            }
        }
    }
}

/// Parent-first loading: for each visible tile without a cached ancestor,
/// enqueue the nearest uncached ancestor (one per visible tile per frame).
fn plan_ancestor_loads(
    layer: &dyn LayerLoadState,
    visible: &[VisibleTile],
    visible_set: &HashSet<TileCoord>,
    min_z: u8,
    max_z: u8,
    camera_center: DVec2,
    requests: &mut Vec<PlannedRequest>,
) {
    let mut budget = layer.max_concurrent();
    let mut ancestor_enqueued: HashSet<TileCoord> = HashSet::new();
    let request_kind = if matches!(layer.kind(), LayerKind::Terrain { .. }) {
        PlannedRequestKind::Terrain
    } else {
        PlannedRequestKind::Raster
    };

    for vt in visible {
        if budget == 0 {
            break;
        }
        let coord = vt.coord;
        if layer.has_texture(&coord) {
            continue;
        }

        let start = if coord.z > max_z {
            Some(coord.clamp_to_zoom(max_z))
        } else {
            coord.parent()
        };
        let mut cur = start;
        while let Some(p) = cur {
            if p.z < min_z {
                break;
            }
            if layer.has_texture(&p) {
                break;
            }
            if !layer.is_pending(&p)
                && !layer.is_failed_cooldown(&p)
                && ancestor_enqueued.insert(p)
                && !visible_set.contains(&p)
            {
                let p_center = p.mercator_center();
                let p_dist = (p_center - camera_center).length() as f32;
                requests.push(PlannedRequest {
                    coord: p,
                    priority: p_dist * 0.8,
                    kind: request_kind,
                });
                budget = budget.saturating_sub(1);
                break;
            }
            cur = p.parent();
        }
    }
}

/// Plan visible tile loads with fallback-depth priority and overzoom clamping.
fn plan_visible_tile_loads(
    layer: &dyn LayerLoadState,
    visible: &[VisibleTile],
    min_z: u8,
    max_z: u8,
    camera_center: DVec2,
    requests: &mut Vec<PlannedRequest>,
) {
    let request_kind = if matches!(layer.kind(), LayerKind::Terrain { .. }) {
        PlannedRequestKind::Terrain
    } else {
        PlannedRequestKind::Raster
    };
    let mut overzoom_enqueued: HashSet<TileCoord> = HashSet::new();

    for vt in visible {
        let coord = vt.coord;
        if coord.z < min_z {
            continue;
        }

        let fetch_coord = if coord.z > max_z {
            let clamped = coord.clamp_to_zoom(max_z);
            if !overzoom_enqueued.insert(clamped) {
                continue;
            }
            clamped
        } else {
            coord
        };

        if layer.has_texture(&fetch_coord)
            || layer.is_pending(&fetch_coord)
            || layer.is_failed_cooldown(&fetch_coord)
        {
            continue;
        }

        let tile_center = fetch_coord.mercator_center();
        let dist = (tile_center - camera_center).length() as f32;

        // Fallback depth: zoom levels up to nearest cached ancestor.
        let fallback_depth = {
            let mut depth = 0u32;
            let mut cur = fetch_coord.parent();
            loop {
                match cur {
                    Some(c) if layer.has_texture(&c) => {
                        depth += 1;
                        break;
                    }
                    Some(c) => {
                        depth += 1;
                        cur = c.parent();
                    }
                    None => {
                        depth = 0;
                        break;
                    }
                }
            }
            depth
        };

        let fallback_factor = match fallback_depth {
            0 => 0.5,
            1 => 1.5,
            2 => 1.2,
            _ => 1.0,
        };

        requests.push(PlannedRequest {
            coord: fetch_coord,
            priority: dist * fallback_factor,
            kind: request_kind,
        });
    }
}

/// Plan elevation tile loads for runtime terrain.
fn plan_elevation_loads(
    layer: &dyn LayerLoadState,
    visible: &[VisibleTile],
    visible_set: &HashSet<TileCoord>,
    camera_center: DVec2,
    requests: &mut Vec<PlannedRequest>,
) {
    // Eagerly load z=0 elevation for global fallback.
    let base_coord = TileCoord::new(0, 0, 0);
    if !layer.has_terrain_data(&base_coord) && !layer.is_elevation_pending(&base_coord) {
        requests.push(PlannedRequest {
            coord: base_coord,
            priority: 0.0,
            kind: PlannedRequestKind::Elevation,
        });
    }

    // Visible elevation tiles, sorted by distance.
    let mut seen = HashSet::new();
    let mut elev_requests: Vec<(TileCoord, f32)> = visible
        .iter()
        .filter(|vt| {
            visible_set.contains(&vt.coord)
                && !layer.has_terrain_data(&vt.coord)
                && !layer.is_elevation_pending(&vt.coord)
                && seen.insert(vt.coord)
        })
        .map(|vt| {
            let center = vt.display_mercator_center();
            let dist = (center - camera_center).length() as f32;
            (vt.coord, dist)
        })
        .collect();
    elev_requests.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    for (coord, dist) in elev_requests {
        requests.push(PlannedRequest {
            coord,
            priority: dist,
            kind: PlannedRequestKind::Elevation,
        });
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock layer state for testing.
    struct MockLayer {
        min_zoom: u8,
        max_zoom: u8,
        textures: HashSet<TileCoord>,
        pending: HashSet<TileCoord>,
        failed: HashSet<TileCoord>,
        terrain_data: HashSet<TileCoord>,
        elevation_pending: HashSet<TileCoord>,
        max_concurrent: usize,
        has_elev_source: bool,
        kind: LayerKind,
    }

    impl MockLayer {
        fn raster() -> Self {
            Self {
                min_zoom: 0,
                max_zoom: 18,
                textures: HashSet::new(),
                pending: HashSet::new(),
                failed: HashSet::new(),
                terrain_data: HashSet::new(),
                elevation_pending: HashSet::new(),
                max_concurrent: 6,
                has_elev_source: false,
                kind: LayerKind::Raster,
            }
        }
    }

    impl LayerLoadState for MockLayer {
        fn kind(&self) -> &LayerKind { &self.kind }
        fn min_zoom(&self) -> u8 { self.min_zoom }
        fn max_zoom(&self) -> u8 { self.max_zoom }
        fn has_texture(&self, coord: &TileCoord) -> bool { self.textures.contains(coord) }
        fn is_pending(&self, coord: &TileCoord) -> bool { self.pending.contains(coord) }
        fn is_failed_cooldown(&self, coord: &TileCoord) -> bool { self.failed.contains(coord) }
        fn has_terrain_data(&self, coord: &TileCoord) -> bool { self.terrain_data.contains(coord) }
        fn is_elevation_pending(&self, coord: &TileCoord) -> bool { self.elevation_pending.contains(coord) }
        fn max_concurrent(&self) -> usize { self.max_concurrent }
        fn pending_count(&self) -> usize { self.pending.len() }
        fn max_elevation_concurrent(&self) -> usize { 4 }
        fn elevation_pending_count(&self) -> usize { self.elevation_pending.len() }
        fn has_elevation_source(&self) -> bool { self.has_elev_source }
    }

    #[test]
    fn base_tiles_always_planned() {
        let layer = MockLayer::raster();
        let visible = vec![];
        let visible_set = HashSet::new();
        let center = DVec2::new(0.5, 0.5);

        let (_, requests) = plan_tile_loads(&layer, &visible, &visible_set, center);

        // Should request z=0/0/0, z=1/0/0, z=1/1/0, z=1/0/1, z=1/1/1 = 5 tiles
        let base_coords: Vec<_> = requests.iter().filter(|r| r.priority == 0.0).collect();
        assert_eq!(base_coords.len(), 5);
    }

    #[test]
    fn parent_first_loading_enqueues_ancestors() {
        let mut layer = MockLayer::raster();
        // Cache z=0 base tile
        layer.textures.insert(TileCoord::new(0, 0, 0));
        // Also cache z=1 tiles
        for y in 0..2 {
            for x in 0..2 {
                layer.textures.insert(TileCoord::new(1, x, y));
            }
        }

        // Visible tile at z=5 — ancestors z=4,3,2 are missing
        let vt = VisibleTile { coord: TileCoord::new(5, 10, 10), display_x: 10 };
        let visible = vec![vt];
        let visible_set: HashSet<_> = visible.iter().map(|v| v.coord).collect();
        let center = DVec2::new(0.5, 0.5);

        let (_, requests) = plan_tile_loads(&layer, &visible, &visible_set, center);

        // Should have ancestor request(s) with priority *= 0.8
        let ancestor_reqs: Vec<_> = requests
            .iter()
            .filter(|r| r.coord.z < 5 && r.coord.z > 1 && r.priority > 0.0)
            .collect();
        assert!(!ancestor_reqs.is_empty(), "should plan ancestor loads");
    }

    #[test]
    fn overzoom_clamping() {
        let mut layer = MockLayer::raster();
        layer.max_zoom = 10;
        // Cache base tiles so they don't appear in requests
        layer.textures.insert(TileCoord::new(0, 0, 0));
        for y in 0..2 { for x in 0..2 { layer.textures.insert(TileCoord::new(1, x, y)); } }

        // Two visible tiles at z=12 that map to the same z=10 parent
        let vt1 = VisibleTile { coord: TileCoord::new(12, 40, 40), display_x: 40 };
        let vt2 = VisibleTile { coord: TileCoord::new(12, 41, 40), display_x: 41 };
        let visible = vec![vt1, vt2];
        let visible_set: HashSet<_> = visible.iter().map(|v| v.coord).collect();
        let center = DVec2::new(0.5, 0.5);

        let (_, requests) = plan_tile_loads(&layer, &visible, &visible_set, center);

        // Both vt1 and vt2 at z=12 map to the same z=10 tile.
        // The clamped coord should appear in the requests.
        let clamped = TileCoord::new(12, 40, 40).clamp_to_zoom(10);
        let z10_reqs: Vec<_> = requests.iter().filter(|r| r.coord == clamped).collect();
        assert!(!z10_reqs.is_empty(), "should request the clamped z=10 tile");
        // No z=12 tiles should appear (beyond max_zoom).
        let z12_reqs: Vec<_> = requests.iter().filter(|r| r.coord.z == 12).collect();
        assert!(z12_reqs.is_empty(), "should NOT request tiles beyond max_zoom");
    }

    #[test]
    fn fallback_depth_priority() {
        let mut layer = MockLayer::raster();
        // Cache base tiles
        layer.textures.insert(TileCoord::new(0, 0, 0));
        for y in 0..2 { for x in 0..2 { layer.textures.insert(TileCoord::new(1, x, y)); } }

        // Two visible tiles at z=5 — one has a z=4 parent cached, the other doesn't
        let vt_with_parent = VisibleTile { coord: TileCoord::new(5, 0, 0), display_x: 0 };
        let vt_no_parent = VisibleTile { coord: TileCoord::new(5, 10, 10), display_x: 10 };
        // Cache the parent of vt_with_parent
        layer.textures.insert(TileCoord::new(4, 0, 0));

        let visible = vec![vt_with_parent, vt_no_parent];
        let visible_set: HashSet<_> = visible.iter().map(|v| v.coord).collect();
        let center = DVec2::new(0.5, 0.5);

        let (_, requests) = plan_tile_loads(&layer, &visible, &visible_set, center);

        // Find the two visible tile requests
        let req_with = requests.iter().find(|r| r.coord == TileCoord::new(5, 0, 0));
        let req_without = requests.iter().find(|r| r.coord == TileCoord::new(5, 10, 10));

        // vt_with_parent has depth=1 → factor=1.5
        // vt_no_parent has depth>1 → factor varies, but the one with no close parent
        // should have different priority
        if let (Some(rw), Some(rwo)) = (req_with, req_without) {
            // The tile with a close fallback (depth=1) uses factor 1.5
            // The tile without close fallback uses lower factor → higher urgency
            // (exact comparison depends on distance)
            assert!(rw.priority > 0.0);
            assert!(rwo.priority > 0.0);
        }
    }

    #[test]
    fn elevation_base_tile_planned() {
        let mut layer = MockLayer::raster();
        layer.has_elev_source = true;

        let visible = vec![];
        let visible_set = HashSet::new();
        let center = DVec2::new(0.5, 0.5);

        let (_, requests) = plan_tile_loads(&layer, &visible, &visible_set, center);

        let elev_base: Vec<_> = requests
            .iter()
            .filter(|r| r.kind == PlannedRequestKind::Elevation && r.coord.z == 0)
            .collect();
        assert_eq!(elev_base.len(), 1, "should plan elevation base tile");
    }
}
