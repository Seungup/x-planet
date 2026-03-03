//! Main map engine: coordinates rendering, tile loading, and user input.
//!
//! The engine manages a **layer stack** of tile sources.  Each [`TileLayer`]
//! holds platform-agnostic metadata ([`LayerConfig`]).  The platform app
//! (native / wasm) owns the per-layer GPU state (textures, loaders, etc.).

use crate::viewport::{CameraController, Viewport};
use x_planets_math::GeoCoord;
use x_planets_projection::ProjectionRegistry;
use x_planets_tiles::TerrainEncoding;

// ═══════════════════════════════════════════════════════════════════
// Layer types
// ═══════════════════════════════════════════════════════════════════

/// The kind of tile layer, determining which rendering pipeline to use.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum LayerKind {
    /// Standard raster imagery (PNG/JPEG tiles rendered as flat quads).
    #[default]
    Raster,
    /// Terrain elevation tiles rendered as displaced meshes.
    /// The `imagery_layer` names the companion raster layer whose textures are
    /// draped onto the terrain mesh.
    /// The `encoding` selects the elevation decoding formula (MapboxRgb or Terrarium).
    Terrain {
        imagery_layer: String,
        encoding: TerrainEncoding,
    },
    /// OGC 3D Tiles (glTF/B3DM models in ECEF coordinates).
    /// Requires `cesium_ion_token` + `cesium_ion_asset_id` or `google_api_key`
    /// to be set on the [`LayerConfig`].
    Tiles3d,
}

/// Configuration for a single tile layer.
#[derive(Debug, Clone)]
pub struct LayerConfig {
    /// Human-readable layer name (unique key).
    pub name: String,
    /// Tile source URL template (e.g., `"https://tile.openstreetmap.org/{z}/{x}/{y}.png"`).
    pub tile_source_url: String,
    /// Opacity 0.0 (transparent) – 1.0 (opaque).
    pub opacity: f32,
    /// Whether this layer is rendered.
    pub visible: bool,
    /// Stacking order – lower values are drawn first (bottom).
    pub z_order: i32,
    /// Per-layer maximum cached tiles.
    pub max_cached_tiles: usize,
    /// Per-layer maximum concurrent tile loads.
    pub max_concurrent_loads: usize,
    /// Layer rendering kind (raster, terrain, etc.).
    pub kind: LayerKind,
    /// Cesium Ion account token (for `Tiles3d` layers).
    pub cesium_ion_token: Option<String>,
    /// Cesium Ion asset ID (for `Tiles3d` layers, e.g. 96188 = OSM Buildings).
    pub cesium_ion_asset_id: Option<u64>,
    /// Google Maps Platform API key (for `Tiles3d` layers).
    pub google_api_key: Option<String>,
    /// Whether the terrain encoding was explicitly set in config.
    /// When `false`, TileJSON auto-detection may override the encoding.
    pub terrain_encoding_explicit: bool,
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            name: "base".to_string(),
            tile_source_url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png".to_string(),
            opacity: 1.0,
            visible: true,
            z_order: 0,
            max_cached_tiles: 256,
            max_concurrent_loads: 6,
            kind: LayerKind::Raster,
            cesium_ion_token: None,
            cesium_ion_asset_id: None,
            google_api_key: None,
            terrain_encoding_explicit: false,
        }
    }
}

/// A tile layer managed by the engine (platform-agnostic metadata).
pub struct TileLayer {
    pub config: LayerConfig,
}

// ═══════════════════════════════════════════════════════════════════
// Map configuration
// ═══════════════════════════════════════════════════════════════════

/// Configuration for the map engine.
pub struct MapConfig {
    /// Initial viewport center.
    pub center: GeoCoord,
    /// Initial zoom level.
    pub zoom: f64,
    /// Default projection name.
    pub projection: String,
    /// Tile source URL template – used as the default "base" layer when `layers` is empty.
    pub tile_source_url: String,
    /// Maximum concurrent tile loads (default layer).
    pub max_concurrent_loads: usize,
    /// Maximum cached tiles in memory (default layer).
    pub max_cached_tiles: usize,
    /// Explicit layer stack.  When non-empty, `tile_source_url` / concurrent / cached
    /// fields above are ignored and each [`LayerConfig`] is used instead.
    pub layers: Vec<LayerConfig>,
    /// Terrain height exaggeration factor (default: 1.5).
    /// Higher values make mountains more prominent in the Mercator view.
    pub terrain_exaggeration: f64,
}

impl Default for MapConfig {
    fn default() -> Self {
        Self {
            center: GeoCoord::new(0.0, 0.0),
            zoom: 2.0,
            projection: "Web Mercator".to_string(),
            tile_source_url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png".to_string(),
            max_concurrent_loads: 6,
            max_cached_tiles: 256,
            layers: Vec::new(),
            terrain_exaggeration: 1.5,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Map engine
// ═══════════════════════════════════════════════════════════════════

/// The main map engine that coordinates all subsystems.
///
/// Tile I/O and GPU state live in the platform app – the engine only stores
/// layer metadata and viewport / camera state.
pub struct MapEngine {
    pub viewport: Viewport,
    pub camera: CameraController,
    pub projection_registry: ProjectionRegistry,
    /// Ordered layer stack (sorted by `z_order`).
    pub layers: Vec<TileLayer>,
    pub active_projection: String,
    needs_redraw: bool,
}

impl MapEngine {
    /// Create a new map engine with the given configuration.
    ///
    /// If `config.layers` is empty, a default "base" layer is created from
    /// `config.tile_source_url`.
    pub fn new(config: MapConfig, width: u32, height: u32) -> Self {
        let mut viewport = Viewport::new(width, height);
        viewport.center = config.center;
        viewport.zoom = config.zoom;

        let layers = if config.layers.is_empty() {
            vec![TileLayer {
                config: LayerConfig {
                    name: "base".to_string(),
                    tile_source_url: config.tile_source_url.clone(),
                    opacity: 1.0,
                    visible: true,
                    z_order: 0,
                    max_cached_tiles: config.max_cached_tiles,
                    max_concurrent_loads: config.max_concurrent_loads,
                    kind: LayerKind::Raster,
                    ..Default::default()
                },
            }]
        } else {
            config
                .layers
                .into_iter()
                .map(|c| TileLayer { config: c })
                .collect()
        };

        let mut engine = Self {
            viewport,
            camera: CameraController::new(),
            projection_registry: ProjectionRegistry::new(),
            layers,
            active_projection: config.projection,
            needs_redraw: true,
        };
        engine.sort_layers();
        engine
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

    /// Pan the map by pixel delta, projection-aware.
    pub fn pan_for_mode(&mut self, dx: f64, dy: f64, mode: x_planets_math::ProjectionMode) {
        self.camera.pan_for_mode(&mut self.viewport, dx, dy, mode);
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

    /// Zoom toward a screen point, projection-aware.
    pub fn zoom_at_for_mode(
        &mut self,
        delta: f64,
        screen_x: f64,
        screen_y: f64,
        mode: x_planets_math::ProjectionMode,
    ) {
        self.camera
            .zoom_at_for_mode(&mut self.viewport, delta, screen_x, screen_y, mode);
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

    // ── Layer management ─────────────────────────────────────────

    /// Add a new layer. Returns the index in the sorted stack.
    pub fn add_layer(&mut self, config: LayerConfig) -> usize {
        let name = config.name.clone();
        self.layers.push(TileLayer { config });
        self.sort_layers();
        self.needs_redraw = true;
        self.layers
            .iter()
            .position(|l| l.config.name == name)
            .unwrap_or(self.layers.len() - 1)
    }

    /// Remove a layer by name. Returns `true` if found.
    pub fn remove_layer(&mut self, name: &str) -> bool {
        let before = self.layers.len();
        self.layers.retain(|l| l.config.name != name);
        let removed = self.layers.len() < before;
        if removed {
            self.needs_redraw = true;
        }
        removed
    }

    /// Look up a layer by name (read-only).
    pub fn get_layer(&self, name: &str) -> Option<&TileLayer> {
        self.layers.iter().find(|l| l.config.name == name)
    }

    /// Look up a layer by name (mutable).
    pub fn get_layer_mut(&mut self, name: &str) -> Option<&mut TileLayer> {
        self.layers.iter_mut().find(|l| l.config.name == name)
    }

    /// Set layer opacity (0.0–1.0). Returns `true` if the layer was found.
    pub fn set_layer_opacity(&mut self, name: &str, opacity: f32) -> bool {
        if let Some(layer) = self.get_layer_mut(name) {
            layer.config.opacity = opacity.clamp(0.0, 1.0);
            self.needs_redraw = true;
            true
        } else {
            false
        }
    }

    /// Set layer visibility. Returns `true` if the layer was found.
    pub fn set_layer_visible(&mut self, name: &str, visible: bool) -> bool {
        if let Some(layer) = self.get_layer_mut(name) {
            layer.config.visible = visible;
            self.needs_redraw = true;
            true
        } else {
            false
        }
    }

    /// Iterator over visible layers (in z-order, bottom-to-top).
    pub fn visible_layers(&self) -> impl Iterator<Item = &TileLayer> {
        self.layers.iter().filter(|l| l.config.visible)
    }

    /// Sort layers by `z_order` (stable sort preserves insertion order for ties).
    fn sort_layers(&mut self) {
        self.layers.sort_by_key(|l| l.config.z_order);
    }

    /// Total number of layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

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

    #[test]
    fn test_default_config_creates_base_layer() {
        let engine = MapEngine::new(MapConfig::default(), 800, 600);
        assert_eq!(engine.layer_count(), 1);
        let base = engine.get_layer("base").unwrap();
        assert_eq!(base.config.opacity, 1.0);
        assert!(base.config.visible);
        assert_eq!(base.config.z_order, 0);
    }

    #[test]
    fn test_engine_add_layer() {
        let mut engine = MapEngine::new(MapConfig::default(), 800, 600);
        engine.frame_rendered();

        engine.add_layer(LayerConfig {
            name: "satellite".to_string(),
            tile_source_url: "https://example.com/{z}/{x}/{y}.png".to_string(),
            opacity: 0.8,
            z_order: 10,
            ..Default::default()
        });

        assert_eq!(engine.layer_count(), 2);
        assert!(engine.needs_redraw());
        let sat = engine.get_layer("satellite").unwrap();
        assert_eq!(sat.config.opacity, 0.8);
    }

    #[test]
    fn test_engine_remove_layer() {
        let mut engine = MapEngine::new(MapConfig::default(), 800, 600);
        engine.add_layer(LayerConfig {
            name: "overlay".to_string(),
            z_order: 5,
            ..Default::default()
        });
        assert_eq!(engine.layer_count(), 2);
        assert!(engine.remove_layer("overlay"));
        assert_eq!(engine.layer_count(), 1);
        assert!(!engine.remove_layer("nonexistent"));
    }

    #[test]
    fn test_engine_layer_opacity() {
        let mut engine = MapEngine::new(MapConfig::default(), 800, 600);
        engine.frame_rendered();
        assert!(engine.set_layer_opacity("base", 0.5));
        assert_eq!(engine.get_layer("base").unwrap().config.opacity, 0.5);
        assert!(engine.needs_redraw());

        // Clamp
        engine.set_layer_opacity("base", 2.0);
        assert_eq!(engine.get_layer("base").unwrap().config.opacity, 1.0);
        engine.set_layer_opacity("base", -1.0);
        assert_eq!(engine.get_layer("base").unwrap().config.opacity, 0.0);

        assert!(!engine.set_layer_opacity("nonexistent", 0.5));
    }

    #[test]
    fn test_engine_layer_visibility() {
        let mut engine = MapEngine::new(MapConfig::default(), 800, 600);
        engine.frame_rendered();
        assert!(engine.set_layer_visible("base", false));
        assert!(!engine.get_layer("base").unwrap().config.visible);
        assert!(engine.needs_redraw());
        assert_eq!(engine.visible_layers().count(), 0);

        engine.set_layer_visible("base", true);
        assert_eq!(engine.visible_layers().count(), 1);
    }

    #[test]
    fn test_engine_layer_z_order() {
        let config = MapConfig {
            layers: vec![
                LayerConfig {
                    name: "top".to_string(),
                    z_order: 100,
                    ..Default::default()
                },
                LayerConfig {
                    name: "mid".to_string(),
                    z_order: 50,
                    ..Default::default()
                },
                LayerConfig {
                    name: "bottom".to_string(),
                    z_order: 0,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let engine = MapEngine::new(config, 800, 600);
        let names: Vec<&str> = engine.layers.iter().map(|l| l.config.name.as_str()).collect();
        assert_eq!(names, vec!["bottom", "mid", "top"]);
    }
}
