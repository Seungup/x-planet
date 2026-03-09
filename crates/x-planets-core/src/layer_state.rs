//! Shared layer state fields used by both native and web platforms.
//!
//! Platforms embed [`LayerStateBase`] in their own layer state struct
//! and delegate common field access through it.

use std::collections::HashSet;

use x_planets_math::TileCoord;

use crate::engine::LayerKind;

/// Platform-agnostic layer state fields.
///
/// Both `NativeLayerState` and `WebLayerState` embed this struct and
/// delegate [`LayerLoadState`](crate::tile_load_planner::LayerLoadState)
/// methods to it.
#[derive(Debug, Clone)]
pub struct LayerStateBase {
    /// Human-readable layer name (unique key).
    pub name: String,
    /// Layer rendering kind.
    pub kind: LayerKind,
    /// Tile source URL template.
    pub url_template: String,
    /// Minimum zoom level served by the tile source.
    pub min_zoom: u8,
    /// Maximum zoom level served by the tile source.
    pub max_zoom: u8,
    /// Maximum concurrent tile loads.
    pub max_concurrent: usize,
    /// Maximum concurrent elevation tile loads.
    pub max_elevation_concurrent: usize,
    /// Tile coords currently being fetched (raster/terrain).
    pub pending_coords: HashSet<TileCoord>,
    /// Tile coords currently being fetched (elevation).
    pub pending_elevation_coords: HashSet<TileCoord>,
    /// Cached set of available raster tile coords (rebuilt each frame).
    pub available_coords_cache: HashSet<TileCoord>,
}

impl LayerStateBase {
    /// Create a new base state from layer configuration.
    pub fn new(
        name: String,
        kind: LayerKind,
        url_template: String,
        max_concurrent: usize,
    ) -> Self {
        Self {
            name,
            kind,
            url_template,
            min_zoom: 0,
            max_zoom: 22,
            max_concurrent,
            max_elevation_concurrent: max_concurrent.min(4),
            pending_coords: HashSet::new(),
            pending_elevation_coords: HashSet::new(),
            available_coords_cache: HashSet::new(),
        }
    }

    /// Refresh the available coords cache from an iterator of cache keys.
    pub fn refresh_available_cache(&mut self, keys: impl Iterator<Item = TileCoord>) {
        self.available_coords_cache = keys.collect();
    }
}
