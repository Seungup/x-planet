//! Tile visibility tracking and crossfade overlay computation.

use std::collections::{HashMap, HashSet};

use x_planets_math::{TileCoord, VisibleTile};

use super::FADE_DURATION;
use crate::pipeline::RenderableTile;

/// Smooth Hermite interpolation (smoothstep): 3t² − 2t³.
/// Produces ease-in-out curve that avoids perceptual pop at start/end.
#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

// ═══════════════════════════════════════════════════════════════════
// Tile visibility tracking (shared between native and web)
// ═══════════════════════════════════════════════════════════════════

/// Track tile visibility changes and register fade-in for cached tiles
/// newly entering the viewport.
///
/// Call once per frame after tile uploads, before crossfade computation.
///
/// - `visible`: current frame's visible tiles
/// - `available`: tiles with textures in GPU cache
/// - `prev_visible_available`: the set returned from the previous frame
/// - `fade_elapsed_fn`: returns elapsed seconds since tile was loaded, or `None`
/// - `register_fade_fn`: called with each tile coord that needs a new fade entry
/// - `departing_tiles`: map of coord → departure timestamp (mutated in place)
///
/// Returns the new `visible_available` set to store for next frame.
pub fn update_tile_visibility<F, R>(
    visible: &[VisibleTile],
    available: &HashSet<TileCoord>,
    prev_visible_available: &HashSet<TileCoord>,
    fade_elapsed_fn: F,
    mut register_fade_fn: R,
    departing_tiles: &mut HashMap<TileCoord, f64>,
    now_secs: f64,
) -> HashSet<TileCoord>
where
    F: Fn(&TileCoord) -> Option<f64>,
    R: FnMut(TileCoord),
{
    let visible_set: HashSet<TileCoord> = visible.iter().map(|vt| vt.coord).collect();

    let visible_available: HashSet<TileCoord> = visible
        .iter()
        .filter(|vt| available.contains(&vt.coord))
        .map(|vt| vt.coord)
        .collect();

    // Register fade for cached tiles newly entering viewport (e.g. tiles
    // coming from behind the globe, or re-entering after being off-screen
    // long enough for the fade entry to be GC'd).
    for &coord in &visible_available {
        if !prev_visible_available.contains(&coord)
            && fade_elapsed_fn(&coord).is_none()
        {
            register_fade_fn(coord);
        }
    }

    // Track departing tiles (were visible+available, now gone) for
    // zoom-out fade-out overlay.
    for &coord in prev_visible_available {
        if !visible_set.contains(&coord) {
            departing_tiles.entry(coord).or_insert(now_secs);
        }
    }
    departing_tiles.retain(|_, start| now_secs - *start < FADE_DURATION);

    visible_available
}

// ═══════════════════════════════════════════════════════════════════
// Crossfade computation (shared between native and web)
// ═══════════════════════════════════════════════════════════════════

/// Compute crossfade tiles: identify tiles transitioning parent → child.
///
/// During the fade-in period, exclude child tiles from the `available` set
/// so `resolve_fallbacks` picks the parent texture as the base. Returns
/// the modified available set and a list of `(coord, fade_t)` pairs for
/// the overlay pass.
///
/// `tile_fade_elapsed_fn` returns the elapsed seconds since a tile was loaded,
/// or `None` if the tile is not being tracked for fade-in.
/// Crossfade tile entry with display_x for antimeridian wrapping.
pub type CrossfadeTile = (TileCoord, f32, i64);

pub fn compute_crossfade<F>(
    visible: &[VisibleTile],
    available: &HashSet<TileCoord>,
    tile_fade_elapsed_fn: F,
) -> (HashSet<TileCoord>, Vec<CrossfadeTile>)
where
    F: Fn(&TileCoord) -> Option<f64>,
{
    let mut available_for_base = available.clone();
    let mut crossfade_tiles: Vec<CrossfadeTile> = Vec::new();

    for vt in visible {
        let coord = vt.coord;
        if !available.contains(&coord) {
            continue;
        }
        if let Some(elapsed) = tile_fade_elapsed_fn(&coord) {
            if elapsed < FADE_DURATION {
                // Search for an available parent (unbounded depth).
                // Base tiles (z=0, z=1) are always eagerly loaded, so this
                // search is guaranteed to find an ancestor quickly.
                // Without full-depth search, tiles many levels above their
                // nearest cached ancestor skip crossfade and flash in.
                let has_parent = {
                    let mut c = coord.parent();
                    let mut found = false;
                    while let Some(p) = c {
                        if available.contains(&p) {
                            found = true;
                            break;
                        }
                        c = p.parent();
                    }
                    found
                };
                if has_parent {
                    available_for_base.remove(&coord);
                    let linear_t = ((elapsed / FADE_DURATION) as f32).clamp(0.0, 1.0);
                    let fade_t = smoothstep(linear_t);
                    crossfade_tiles.push((coord, fade_t, vt.display_x));
                }
            }
        }
    }

    (available_for_base, crossfade_tiles)
}

/// Compute per-tile opacity overrides for tiles with no parent coverage
/// (first-time appearance, fade from near-zero).
pub fn compute_fade_overrides<F>(
    renderable: &[RenderableTile],
    layer_opacity: f32,
    tile_fade_elapsed_fn: F,
) -> HashMap<TileCoord, f32>
where
    F: Fn(&TileCoord) -> Option<f64>,
{
    let mut overrides = HashMap::new();
    for rt in renderable {
        // Only apply to tiles using their own texture (not parent fallback)
        if rt.texture_coord != rt.coord {
            continue;
        }
        if let Some(elapsed) = tile_fade_elapsed_fn(&rt.coord) {
            if elapsed < FADE_DURATION {
                let linear_t = ((elapsed / FADE_DURATION) as f32).clamp(0.0, 1.0);
                let t = smoothstep(linear_t);
                overrides.insert(rt.coord, layer_opacity * t);
            }
        }
    }
    overrides
}

/// Build crossfade overlay [`RenderableTile`]s and their opacity overrides
/// from the crossfade tile list.
pub fn build_crossfade_overlay(
    crossfade_tiles: &[CrossfadeTile],
    layer_opacity: f32,
) -> (Vec<RenderableTile>, HashMap<TileCoord, f32>) {
    let mut tiles = Vec::with_capacity(crossfade_tiles.len());
    let mut opacity_map = HashMap::with_capacity(crossfade_tiles.len());
    for &(coord, fade_t, display_x) in crossfade_tiles {
        tiles.push(RenderableTile {
            coord,
            texture_coord: coord,
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            display_x,
        });
        opacity_map.insert(coord, layer_opacity * fade_t);
    }
    (tiles, opacity_map)
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossfade_with_parent() {
        let parent = TileCoord::new(1, 0, 0);
        let child = TileCoord::new(2, 0, 0);
        let visible = vec![VisibleTile::canonical(child)];
        let available: HashSet<TileCoord> = [parent, child].into_iter().collect();

        // Child is mid-fade (0.15s elapsed)
        let (base_available, crossfade) =
            compute_crossfade(&visible, &available, |coord| {
                if *coord == child {
                    Some(0.15)
                } else {
                    None
                }
            });

        // Child excluded from base (parent will be used as fallback)
        assert!(!base_available.contains(&child));
        assert!(base_available.contains(&parent));
        // Child in crossfade overlay
        assert_eq!(crossfade.len(), 1);
        assert_eq!(crossfade[0].0, child);
        assert!(crossfade[0].1 > 0.0 && crossfade[0].1 < 1.0);
    }

    #[test]
    fn test_crossfade_without_parent() {
        let child = TileCoord::new(2, 0, 0);
        let visible = vec![VisibleTile::canonical(child)];
        let available: HashSet<TileCoord> = [child].into_iter().collect();

        let (base_available, crossfade) =
            compute_crossfade(&visible, &available, |coord| {
                if *coord == child {
                    Some(0.15)
                } else {
                    None
                }
            });

        // No parent → child stays in base, no crossfade
        assert!(base_available.contains(&child));
        assert!(crossfade.is_empty());
    }
}
