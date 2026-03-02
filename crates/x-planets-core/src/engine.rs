//! Main map engine: coordinates rendering, tile loading, and user input.

use crate::viewport::{CameraController, Viewport};
use x_planets_math::GeoCoord;
use x_planets_projection::ProjectionRegistry;
use x_planets_tiles::{TileCache, TileLoader};

/// Configuration for the map engine.
pub struct MapConfig {
    /// Initial viewport center.
    pub center: GeoCoord,
    /// Initial zoom level.
    pub zoom: f64,
    /// Default projection name.
    pub projection: String,
    /// Maximum concurrent tile loads.
    pub max_concurrent_loads: usize,
    /// Maximum cached tiles in memory.
    pub max_cached_tiles: usize,
}

impl Default for MapConfig {
    fn default() -> Self {
        Self {
            center: GeoCoord::new(0.0, 0.0),
            zoom: 2.0,
            projection: "Web Mercator".to_string(),
            max_concurrent_loads: 6,
            max_cached_tiles: 256,
        }
    }
}

/// The main map engine that coordinates all subsystems.
pub struct MapEngine {
    pub viewport: Viewport,
    pub camera: CameraController,
    pub projection_registry: ProjectionRegistry,
    pub tile_loader: TileLoader,
    pub tile_cache: TileCache<Vec<u8>>, // Raw decoded pixels for now
    pub active_projection: String,
    needs_redraw: bool,
}

impl MapEngine {
    /// Create a new map engine with the given configuration.
    pub fn new(config: MapConfig, width: u32, height: u32) -> Self {
        let mut viewport = Viewport::new(width, height);
        viewport.center = config.center;
        viewport.zoom = config.zoom;

        Self {
            viewport,
            camera: CameraController::new(),
            projection_registry: ProjectionRegistry::new(),
            tile_loader: TileLoader::new(config.max_concurrent_loads),
            tile_cache: TileCache::new(config.max_cached_tiles),
            active_projection: config.projection,
            needs_redraw: true,
        }
    }

    /// Resize the viewport.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.viewport.width = width;
        self.viewport.height = height;
        self.needs_redraw = true;
    }

    /// Pan the map by pixel delta.
    pub fn pan(&mut self, dx: f64, dy: f64) {
        self.camera.pan(&mut self.viewport, dx, dy);
        self.needs_redraw = true;
    }

    /// Zoom the map at its center.
    pub fn zoom(&mut self, delta: f64) {
        self.camera.zoom(&mut self.viewport, delta);
        self.needs_redraw = true;
    }

    /// Zoom toward a specific screen point (zoom-to-pointer).
    pub fn zoom_at(&mut self, delta: f64, screen_x: f64, screen_y: f64) {
        self.camera.zoom_at(&mut self.viewport, delta, screen_x, screen_y);
        self.needs_redraw = true;
    }

    /// Adjust pitch by delta degrees (positive = more tilt, negative = flatter).
    pub fn pitch(&mut self, delta: f64) {
        let new_pitch = self.viewport.pitch + delta;
        self.camera.set_pitch(&mut self.viewport, new_pitch);
        self.needs_redraw = true;
    }

    /// Rotate the map by delta degrees (positive = clockwise).
    pub fn rotate(&mut self, delta: f64) {
        let new_bearing = self.viewport.bearing + delta;
        self.camera.set_bearing(&mut self.viewport, new_bearing);
        self.needs_redraw = true;
    }

    /// Set the active projection by name.
    pub fn set_projection(&mut self, name: &str) -> bool {
        if self.projection_registry.get(name).is_some() {
            self.active_projection = name.to_string();
            self.needs_redraw = true;
            true
        } else {
            log::warn!("Projection not found: {}", name);
            false
        }
    }

    /// Get the currently active projection.
    pub fn active_projection(&self) -> Option<std::sync::Arc<dyn x_planets_projection::ProjectionPlugin>> {
        self.projection_registry.get(&self.active_projection)
    }

    /// Whether a redraw is needed.
    pub fn needs_redraw(&self) -> bool {
        self.needs_redraw
    }

    /// Mark the frame as rendered.
    pub fn frame_rendered(&mut self) {
        self.needs_redraw = false;
    }

    /// Request a redraw.
    pub fn request_redraw(&mut self) {
        self.needs_redraw = true;
    }

    /// Update the engine state (call once per frame).
    ///
    /// This determines which tiles are visible and queues loads for missing tiles.
    pub fn update(&mut self) {
        let visible = self.viewport.visible_tiles();

        for tile_coord in &visible {
            if !self.tile_cache.contains(tile_coord) {
                // Calculate priority (distance from viewport center)
                let tile_bounds = tile_coord.to_geo_bounds();
                let tile_center = tile_bounds.center();
                let dx = tile_center.lon - self.viewport.center.lon;
                let dy = tile_center.lat - self.viewport.center.lat;
                let distance = (dx * dx + dy * dy).sqrt() as f32;

                self.tile_loader.enqueue(x_planets_tiles::TileRequest {
                    coord: *tile_coord,
                    priority: distance,
                });
            }
        }

        log::trace!(
            "Visible tiles: {}, Pending loads: {}, Cached: {}",
            visible.len(),
            self.tile_loader.pending_count(),
            self.tile_cache.len(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = MapEngine::new(MapConfig::default(), 800, 600);
        assert!(engine.needs_redraw());
        assert!(engine.active_projection().is_some());
    }

    #[test]
    fn test_engine_resize() {
        let mut engine = MapEngine::new(MapConfig::default(), 800, 600);
        engine.frame_rendered();
        assert!(!engine.needs_redraw());

        engine.resize(1920, 1080);
        assert!(engine.needs_redraw());
        assert_eq!(engine.viewport.width, 1920);
    }

    #[test]
    fn test_engine_set_projection() {
        let mut engine = MapEngine::new(MapConfig::default(), 800, 600);
        assert!(engine.set_projection("Equirectangular"));
        assert!(!engine.set_projection("NonExistent"));
    }
}
