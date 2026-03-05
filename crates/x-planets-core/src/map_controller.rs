//! Unified map controller — high-level API for all platforms.
//!
//! `MapController` wraps `MapEngine` + `AnimationController` + crossfade tracking
//! into a single struct that any platform (native, web, C#, Python) can drive with
//! minimal boilerplate.
//!
//! ## Responsibility split
//!
//! - **MapController** (core): camera, layers, animation, visibility, render data assembly
//! - **Platform** (native/web): tile I/O, GPU cache, async runtime, event loop

use std::collections::{HashMap, HashSet};

use x_planets_math::{GeoCoord, ProjectionMode, TileCoord, VisibleTile};
use x_planets_tiles::TerrainEncoding;

use crate::engine::{LayerConfig, LayerKind, MapConfig, MapEngine};
use crate::interaction::{
    build_crossfade_overlay, compute_crossfade, compute_fade_overrides, update_tile_visibility,
    AnimationController, FADE_DURATION,
};
use crate::pipeline::{resolve_fallbacks, RenderableTile};
use crate::render::RenderLayerData;
use crate::terrain_renderer::{TerrainLayerData, TerrainTileData};

// ═══════════════════════════════════════════════════════════════════
// LayerStateView — platform implements this to provide GPU cache state
// ═══════════════════════════════════════════════════════════════════

/// Platform-agnostic view into a layer's GPU cache state.
///
/// Platforms (native/web) implement this trait on their per-layer state struct
/// so that `MapController::build_render_data()` can assemble render data without
/// knowing about platform-specific GPU types.
pub trait LayerStateView {
    /// Layer name (must match the `LayerConfig::name`).
    fn name(&self) -> &str;

    /// Set of tile coords that have raster textures in the GPU cache.
    fn available_raster_coords(&self) -> &HashSet<TileCoord>;

    /// Terrain tile data for a given coord, or `None` if not cached.
    fn terrain_tile_data(&self, coord: &TileCoord) -> Option<&TerrainTileData>;
}

// ═══════════════════════════════════════════════════════════════════
// RenderOutput — what build_render_data() produces
// ═══════════════════════════════════════════════════════════════════

/// Assembled render data for one frame (passed to renderers by the platform).
pub struct RenderOutput<'a> {
    /// Raster tile layers (flat quads with crossfade overlays).
    pub raster_layers: Vec<RenderLayerData<'a>>,
    /// Terrain layers (displaced meshes with draped imagery).
    pub terrain_layers: Vec<TerrainLayerData<'a>>,
    /// Terrain crossfade overlay layers.
    pub terrain_overlay_layers: Vec<TerrainLayerData<'a>>,
}

// ═══════════════════════════════════════════════════════════════════
// Terrain state — rendering property, not a layer
// ═══════════════════════════════════════════════════════════════════

/// Terrain rendering configuration.
///
/// Terrain is a **rendering property** on the map controller, not a separate
/// layer in the engine's layer stack.  When enabled, tiles with available
/// elevation data are rendered as displaced meshes; tiles without elevation
/// continue rendering as flat raster.
pub struct TerrainState {
    /// Elevation tile URL template (e.g. `https://…/{z}/{x}/{y}.png`).
    pub url: String,
    /// Elevation encoding format.
    pub encoding: TerrainEncoding,
    /// Name of the raster layer whose imagery is draped onto the terrain mesh.
    pub imagery_layer_name: String,
}

// ═══════════════════════════════════════════════════════════════════
// MapController
// ═══════════════════════════════════════════════════════════════════

/// Unified map controller for all platforms.
///
/// Owns the `MapEngine` (layer metadata + viewport), `AnimationController`
/// (smooth zoom, inertia, tile fade), and crossfade/departing tracking state.
pub struct MapController {
    pub engine: MapEngine,
    pub anim: AnimationController,

    /// Previous frame's visible+available tile set (for crossfade transition detection).
    prev_visible_available: HashSet<TileCoord>,
    /// Tiles that recently left the visible set, with departure timestamp.
    departing_tiles: HashMap<TileCoord, f64>,

    /// Terrain rendering state (rendering property, not a layer).
    pub terrain: Option<TerrainState>,
}

impl MapController {
    // ── Constructor ──────────────────────────────────────────────

    pub fn new(config: MapConfig, width: u32, height: u32) -> Self {
        let initial_zoom = config.zoom;
        let engine = MapEngine::new(config, width, height);
        Self {
            engine,
            anim: AnimationController::new(initial_zoom),
            prev_visible_available: HashSet::new(),
            departing_tiles: HashMap::new(),
            terrain: None,
        }
    }

    // ── Camera ──────────────────────────────────────────────────

    pub fn pan(&mut self, dx: f64, dy: f64) {
        let mode = self.rendering_mode();
        self.engine.pan_for_mode(dx, dy, mode);
    }

    pub fn zoom(&mut self, delta: f64) {
        self.engine.zoom(delta);
        self.anim.zoom_target = self.engine.viewport.zoom;
    }

    pub fn zoom_at(&mut self, delta: f64, screen_x: f64, screen_y: f64) {
        let mode = self.rendering_mode();
        self.engine.zoom_at_for_mode(delta, screen_x, screen_y, mode);
        self.anim.zoom_target = self.engine.viewport.zoom;
    }

    pub fn rotate(&mut self, degrees: f64) {
        self.engine.rotate(degrees);
    }

    pub fn pitch(&mut self, degrees: f64) {
        self.engine.pitch(degrees);
    }

    pub fn set_center(&mut self, lat: f64, lon: f64) {
        self.engine.viewport.center = GeoCoord::new(lat, lon);
        self.engine.request_redraw();
    }

    pub fn set_zoom(&mut self, zoom: f64) {
        self.engine.viewport.zoom = zoom.clamp(self.engine.camera.min_zoom, self.engine.camera.max_zoom);
        self.anim.zoom_target = self.engine.viewport.zoom;
        self.engine.request_redraw();
    }

    pub fn center(&self) -> (f64, f64) {
        (self.engine.viewport.center.lat, self.engine.viewport.center.lon)
    }

    pub fn zoom_level(&self) -> f64 {
        self.engine.viewport.zoom
    }

    pub fn bearing(&self) -> f64 {
        self.engine.viewport.bearing
    }

    pub fn pitch_angle(&self) -> f64 {
        self.engine.viewport.pitch
    }

    // ── Viewport ────────────────────────────────────────────────

    pub fn resize(&mut self, width: u32, height: u32) {
        self.engine.resize(width, height);
    }

    pub fn width(&self) -> u32 {
        self.engine.viewport.width
    }

    pub fn height(&self) -> u32 {
        self.engine.viewport.height
    }

    // ── Projection ──────────────────────────────────────────────

    pub fn set_projection(&mut self, name: &str) -> bool {
        self.engine.set_projection(name)
    }

    pub fn projection_name(&self) -> &str {
        &self.engine.active_projection
    }

    pub fn rendering_mode(&self) -> ProjectionMode {
        self.engine.rendering_mode()
    }

    pub fn cycle_projection(&mut self) -> String {
        let mut names: Vec<String> = self
            .engine
            .projection_registry
            .list()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        names.sort();
        let current = &self.engine.active_projection;
        let idx = names.iter().position(|n| n == current).unwrap_or(0);
        let next_idx = (idx + 1) % names.len();
        self.engine.set_projection(&names[next_idx]);
        self.engine.active_projection.clone()
    }

    // ── Layer management ────────────────────────────────────────

    pub fn add_layer(&mut self, config: LayerConfig) -> usize {
        self.engine.add_layer(config)
    }

    pub fn remove_layer(&mut self, name: &str) -> bool {
        self.engine.remove_layer(name)
    }

    pub fn set_layer_visible(&mut self, name: &str, visible: bool) -> bool {
        self.engine.set_layer_visible(name, visible)
    }

    pub fn set_layer_opacity(&mut self, name: &str, opacity: f32) -> bool {
        self.engine.set_layer_opacity(name, opacity)
    }

    pub fn layer_count(&self) -> usize {
        self.engine.layer_count()
    }

    // ── Terrain high-level API ──────────────────────────────────

    /// Toggle terrain on/off.  Returns the new state (true = terrain ON).
    ///
    /// Terrain is a rendering property, not a layer.  No layers are added or
    /// removed from the engine.  Platform code should start/stop loading
    /// elevation data on the raster layer when this returns true/false.
    pub fn toggle_terrain(&mut self, url: &str, encoding: TerrainEncoding) -> bool {
        if self.terrain.is_some() {
            // Turn OFF
            self.terrain = None;
            self.engine.request_redraw();
            log::info!("Terrain: OFF");
            false
        } else {
            // Turn ON — find the first raster layer to use as imagery
            let imagery_name = self
                .engine
                .layers
                .iter()
                .find(|l| matches!(l.config.kind, LayerKind::Raster))
                .map(|l| l.config.name.clone())
                .unwrap_or_else(|| "base".to_string());

            self.terrain = Some(TerrainState {
                url: url.to_string(),
                encoding,
                imagery_layer_name: imagery_name,
            });

            self.engine.request_redraw();
            log::info!("Terrain: ON");
            true
        }
    }

    /// Whether terrain is currently enabled.
    pub fn terrain_enabled(&self) -> bool {
        self.terrain.is_some()
    }

    /// The terrain elevation URL template, if terrain is enabled.
    pub fn terrain_url(&self) -> Option<&str> {
        self.terrain.as_ref().map(|ts| ts.url.as_str())
    }

    /// The imagery layer name used by terrain, if terrain is enabled.
    pub fn terrain_imagery_name(&self) -> Option<&str> {
        self.terrain.as_ref().map(|ts| ts.imagery_layer_name.as_str())
    }

    /// The terrain encoding, if terrain is enabled.
    pub fn terrain_encoding(&self) -> Option<TerrainEncoding> {
        self.terrain.as_ref().map(|ts| ts.encoding)
    }

    // ── Per-frame orchestration ─────────────────────────────────

    /// Advance animations (smooth zoom, inertia pan). Call once per frame.
    pub fn tick(&mut self, dt_secs: f64) {
        let mode = self.rendering_mode();
        self.anim.tick_with_mode(&mut self.engine, dt_secs, mode);
    }

    pub fn needs_redraw(&self) -> bool {
        self.engine.needs_redraw()
            || self.anim.is_animating(self.engine.viewport.zoom)
    }

    pub fn frame_rendered(&mut self) {
        self.engine.frame_rendered();
    }

    pub fn visible_tiles(&self) -> Vec<VisibleTile> {
        let mode = self.rendering_mode();
        self.engine.viewport.visible_tiles_for_mode(mode)
    }

    // ── Drag / interaction forwarding ───────────────────────────

    pub fn begin_drag(&mut self) {
        self.anim.begin_drag();
    }

    pub fn record_drag(&mut self, x: f64, y: f64, time_secs: f64) {
        self.anim.record_drag((x, y), time_secs);
    }

    pub fn end_drag(&mut self, time_secs: f64) {
        self.anim.compute_release_velocity(time_secs);
    }

    pub fn check_double_click(&mut self, x: f64, y: f64, time_secs: f64) -> bool {
        self.anim.check_double_click((x, y), time_secs)
    }

    // ── Tile loading callbacks ──────────────────────────────────

    pub fn register_tile_loaded(&mut self, coord: TileCoord, time_secs: f64) {
        self.anim.register_tile_loaded(coord, time_secs);
    }

    pub fn tile_fade_elapsed(&self, coord: &TileCoord, now_secs: f64) -> Option<f64> {
        self.anim.tile_fade_elapsed(coord, now_secs)
    }

    pub fn gc_fades(&mut self, now_secs: f64) {
        self.anim.gc_fades(now_secs);
    }

    // ── Tile visibility tracking ────────────────────────────────

    /// Update tile visibility tracking (call once per frame after GPU upload).
    ///
    /// Returns the set of newly visible+available tiles (for the platform to
    /// register fade-in entries).
    pub fn update_visibility(
        &mut self,
        visible: &[VisibleTile],
        available: &HashSet<TileCoord>,
        now_secs: f64,
    ) -> Vec<TileCoord> {
        let prev = std::mem::take(&mut self.prev_visible_available);
        let mut to_register: Vec<TileCoord> = Vec::new();
        let new_prev = update_tile_visibility(
            visible,
            available,
            &prev,
            |coord| self.anim.tile_fade_elapsed(coord, now_secs),
            |coord| to_register.push(coord),
            &mut self.departing_tiles,
            now_secs,
        );
        for &coord in &to_register {
            self.anim.register_tile_loaded(coord, now_secs);
        }
        self.prev_visible_available = new_prev;
        to_register
    }

    // ── Render data assembly ────────────────────────────────────

    /// Build render data for all visible layers.
    ///
    /// Platform code provides:
    /// - `layer_views`: per-layer GPU cache state (implements `LayerStateView`)
    /// - `texture_fn`: closure that returns `&wgpu::TextureView` for a layer+coord
    ///
    /// Returns assembled `RenderOutput` ready to pass to renderers.
    pub fn build_render_data<'a>(
        &'a self,
        layer_views: &[&'a dyn LayerStateView],
        texture_fn: &dyn Fn(&str, &TileCoord) -> Option<&'a wgpu::TextureView>,
        visible: &[VisibleTile],
        now_secs: f64,
    ) -> RenderOutput<'a> {
        let mut raster_layers: Vec<RenderLayerData<'a>> = Vec::new();
        let mut terrain_layers: Vec<TerrainLayerData<'a>> = Vec::new();
        let mut terrain_overlay_layers: Vec<TerrainLayerData<'a>> = Vec::new();

        // Collect raster layer names consumed as terrain imagery by config-file
        // terrain layers (NOT the runtime toggle — that uses per-tile split).
        let config_terrain_imagery: HashSet<&str> = self
            .engine
            .visible_layers()
            .filter_map(|l| match &l.config.kind {
                LayerKind::Terrain { imagery_layer, .. } => Some(imagery_layer.as_str()),
                _ => None,
            })
            .collect();

        let fade_fn = |coord: &TileCoord| self.anim.tile_fade_elapsed(coord, now_secs);

        // Is this raster layer the terrain imagery target (runtime toggle)?
        let terrain_imagery_layer = self
            .terrain
            .as_ref()
            .map(|ts| ts.imagery_layer_name.as_str());

        for layer in self.engine.visible_layers() {
            let lv = layer_views
                .iter()
                .find(|v| v.name() == layer.config.name);

            match &layer.config.kind {
                LayerKind::Raster => {
                    // Skip if consumed by a config-file terrain layer
                    if config_terrain_imagery.contains(layer.config.name.as_str()) {
                        continue;
                    }
                    if let Some(lv) = lv {
                        // Per-tile split: terrain-enabled raster imagery layer
                        let is_terrain_imagery = terrain_imagery_layer == Some(layer.config.name.as_str());
                        if is_terrain_imagery {
                            // Split visible tiles: elevation available → terrain, rest → flat raster
                            let (flat_visible, terrain_visible) =
                                split_by_elevation(visible, *lv);

                            // Flat raster for tiles without elevation
                            if !flat_visible.is_empty() {
                                let (base, overlay) = build_raster_layer(
                                    &layer.config.name,
                                    layer.config.opacity,
                                    *lv,
                                    texture_fn,
                                    &flat_visible,
                                    &fade_fn,
                                );
                                raster_layers.push(base);
                                if let Some(ovl) = overlay {
                                    raster_layers.push(ovl);
                                }
                            }

                            // Terrain mesh for tiles with elevation (single layer view)
                            if !terrain_visible.is_empty() {
                                let (base, overlay) = build_terrain_layer(
                                    &layer.config.name,
                                    layer.config.opacity,
                                    *lv, // elevation data
                                    *lv, // imagery textures (same layer)
                                    texture_fn,
                                    &terrain_visible,
                                    &fade_fn,
                                );
                                terrain_layers.push(base);
                                if let Some(ovl) = overlay {
                                    terrain_overlay_layers.push(ovl);
                                }
                            }
                        } else {
                            let (base, overlay) = build_raster_layer(
                                &layer.config.name,
                                layer.config.opacity,
                                *lv,
                                texture_fn,
                                visible,
                                &fade_fn,
                            );
                            raster_layers.push(base);
                            if let Some(ovl) = overlay {
                                raster_layers.push(ovl);
                            }
                        }
                    }
                }
                LayerKind::Terrain { imagery_layer, .. } => {
                    // Config-file terrain layers (backward compat)
                    let terrain_lv = lv;
                    let imagery_lv = layer_views.iter().find(|v| v.name() == imagery_layer.as_str());

                    if let (Some(terrain_lv), Some(imagery_lv)) = (terrain_lv, imagery_lv) {
                        let (base, overlay) = build_terrain_layer(
                            &layer.config.name,
                            layer.config.opacity,
                            *terrain_lv,
                            *imagery_lv,
                            texture_fn,
                            visible,
                            &fade_fn,
                        );
                        terrain_layers.push(base);
                        if let Some(ovl) = overlay {
                            terrain_overlay_layers.push(ovl);
                        }
                    }
                }
                LayerKind::Tiles3d => {}
            }
        }

        // Build departing tiles overlay (zoom-out fade-out)
        if !self.departing_tiles.is_empty() {
            if let Some(first_raster_lv) = layer_views.first() {
                let mut overlay_tiles = Vec::new();
                let mut overlay_opacity = HashMap::new();
                let mut texture_views = HashMap::new();

                for (&coord, &start) in &self.departing_tiles {
                    let elapsed = now_secs - start;
                    let fade_out = (1.0 - elapsed / FADE_DURATION).max(0.0) as f32;
                    if fade_out > 0.01 {
                        if let Some(tv) = texture_fn(first_raster_lv.name(), &coord) {
                            overlay_tiles.push(RenderableTile {
                                coord,
                                texture_coord: coord,
                                uv_rect: [0.0, 0.0, 1.0, 1.0],
                                display_x: coord.x as i64,
                            });
                            overlay_opacity.insert(coord, fade_out);
                            texture_views.insert(coord, tv);
                        }
                    }
                }

                if !overlay_tiles.is_empty() {
                    raster_layers.push(RenderLayerData {
                        name: "zoom-out-overlay",
                        opacity: 1.0,
                        tiles: overlay_tiles,
                        texture_views,
                        tile_opacity_overrides: overlay_opacity,
                    });
                }
            }
        }

        RenderOutput {
            raster_layers,
            terrain_layers,
            terrain_overlay_layers,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Free functions for render data assembly (avoids borrow conflicts)
// ═══════════════════════════════════════════════════════════════════

fn build_raster_layer<'a>(
    layer_name: &'a str,
    layer_opacity: f32,
    lv: &'a dyn LayerStateView,
    texture_fn: &dyn Fn(&str, &TileCoord) -> Option<&'a wgpu::TextureView>,
    visible: &[VisibleTile],
    fade_fn: &dyn Fn(&TileCoord) -> Option<f64>,
) -> (RenderLayerData<'a>, Option<RenderLayerData<'a>>) {
    let available = lv.available_raster_coords();

    let texture_views: HashMap<TileCoord, &'a wgpu::TextureView> = available
        .iter()
        .filter_map(|coord| texture_fn(lv.name(), coord).map(|tv| (*coord, tv)))
        .collect();

    let (available_for_base, crossfade_tiles) =
        compute_crossfade(visible, available, fade_fn);

    let renderable = resolve_fallbacks(visible, &available_for_base);

    let tile_opacity_overrides = compute_fade_overrides(
        &renderable,
        layer_opacity,
        fade_fn,
    );

    let base = RenderLayerData {
        name: layer_name,
        opacity: layer_opacity,
        tiles: renderable,
        texture_views: texture_views.clone(),
        tile_opacity_overrides,
    };

    let overlay = if !crossfade_tiles.is_empty() {
        let (overlay_tiles, overlay_opacity) =
            build_crossfade_overlay(&crossfade_tiles, layer_opacity);
        Some(RenderLayerData {
            name: "crossfade-overlay",
            opacity: layer_opacity,
            tiles: overlay_tiles,
            texture_views,
            tile_opacity_overrides: overlay_opacity,
        })
    } else {
        None
    };

    (base, overlay)
}

/// Split visible tiles into (flat, terrain) based on elevation data availability.
///
/// A tile goes to the terrain list if the layer state has elevation data for it
/// OR any of its ancestor tiles (parent fallback).  Otherwise it stays in flat.
fn split_by_elevation(
    visible: &[VisibleTile],
    lv: &dyn LayerStateView,
) -> (Vec<VisibleTile>, Vec<VisibleTile>) {
    let mut flat = Vec::new();
    let mut terrain = Vec::new();
    for vt in visible {
        let has_elev = {
            let mut c = Some(vt.coord);
            let mut found = false;
            while let Some(candidate) = c {
                if lv.terrain_tile_data(&candidate).is_some() {
                    found = true;
                    break;
                }
                c = candidate.parent();
            }
            found
        };
        if has_elev {
            terrain.push(vt.clone());
        } else {
            flat.push(vt.clone());
        }
    }
    (flat, terrain)
}

fn build_terrain_layer<'a>(
    layer_name: &'a str,
    layer_opacity: f32,
    terrain_lv: &'a dyn LayerStateView,
    imagery_lv: &'a dyn LayerStateView,
    texture_fn: &dyn Fn(&str, &TileCoord) -> Option<&'a wgpu::TextureView>,
    visible: &[VisibleTile],
    fade_fn: &dyn Fn(&TileCoord) -> Option<f64>,
) -> (TerrainLayerData<'a>, Option<TerrainLayerData<'a>>) {
    let available = imagery_lv.available_raster_coords();

    let imagery_views: HashMap<TileCoord, &'a wgpu::TextureView> = available
        .iter()
        .filter_map(|coord| texture_fn(imagery_lv.name(), coord).map(|tv| (*coord, tv)))
        .collect();

    let (available_for_base, crossfade_tiles) =
        compute_crossfade(visible, available, fade_fn);

    let renderable = resolve_fallbacks(visible, &available_for_base);

    // Elevation data with parent fallback
    let mut elevation_data: HashMap<TileCoord, (&'a TerrainTileData, TileCoord)> = HashMap::new();
    let all_needed: HashSet<TileCoord> = renderable
        .iter()
        .map(|rt| rt.coord)
        .chain(crossfade_tiles.iter().map(|&(c, _, _)| c))
        .collect();

    for &coord in &all_needed {
        let mut c = Some(coord);
        while let Some(candidate) = c {
            if let Some(data) = terrain_lv.terrain_tile_data(&candidate) {
                elevation_data.insert(coord, (data, candidate));
                break;
            }
            c = candidate.parent();
        }
    }

    let base = TerrainLayerData {
        name: layer_name,
        opacity: layer_opacity,
        tiles: renderable,
        imagery_views: imagery_views.clone(),
        elevation_data: elevation_data.clone(),
        tile_opacity_overrides: HashMap::new(),
    };

    let overlay = if !crossfade_tiles.is_empty() {
        let (overlay_tiles, overlay_opacity) =
            build_crossfade_overlay(&crossfade_tiles, layer_opacity);
        let overlay_elev: HashMap<TileCoord, (&'a TerrainTileData, TileCoord)> =
            crossfade_tiles
                .iter()
                .filter_map(|&(coord, _, _)| {
                    elevation_data.get(&coord).map(|&v| (coord, v))
                })
                .collect();
        Some(TerrainLayerData {
            name: "terrain-crossfade",
            opacity: layer_opacity,
            tiles: overlay_tiles,
            imagery_views,
            elevation_data: overlay_elev,
            tile_opacity_overrides: overlay_opacity,
        })
    } else {
        None
    };

    (base, overlay)
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_creation() {
        let ctrl = MapController::new(MapConfig::default(), 800, 600);
        assert!(ctrl.needs_redraw());
        assert!(!ctrl.terrain_enabled());
        assert_eq!(ctrl.layer_count(), 1);
    }

    #[test]
    fn test_controller_camera() {
        let mut ctrl = MapController::new(MapConfig::default(), 800, 600);
        ctrl.set_center(37.5665, 126.9780);
        let (lat, lon) = ctrl.center();
        assert!((lat - 37.5665).abs() < 0.001);
        assert!((lon - 126.9780).abs() < 0.001);
    }

    #[test]
    fn test_controller_zoom() {
        let mut ctrl = MapController::new(MapConfig::default(), 800, 600);
        ctrl.set_zoom(5.0);
        assert!((ctrl.zoom_level() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_controller_projection() {
        let mut ctrl = MapController::new(MapConfig::default(), 800, 600);
        assert!(ctrl.set_projection("Globe"));
        assert_eq!(ctrl.projection_name(), "Globe");
        assert!(!ctrl.set_projection("NonExistent"));
    }

    #[test]
    fn test_controller_cycle_projection() {
        let mut ctrl = MapController::new(MapConfig::default(), 800, 600);
        let name = ctrl.cycle_projection();
        assert!(!name.is_empty());
    }

    #[test]
    fn test_controller_terrain_toggle() {
        let mut ctrl = MapController::new(MapConfig::default(), 800, 600);
        let url = "https://example.com/{z}/{x}/{y}.png";

        assert!(!ctrl.terrain_enabled());
        assert_eq!(ctrl.layer_count(), 1);

        // Toggle ON — no layer added (terrain is a rendering property)
        let on = ctrl.toggle_terrain(url, TerrainEncoding::Terrarium);
        assert!(on);
        assert!(ctrl.terrain_enabled());
        assert_eq!(ctrl.layer_count(), 1); // unchanged
        assert!(ctrl.terrain_imagery_name().is_some());
        assert!(ctrl.terrain_url().is_some());
        assert_eq!(ctrl.terrain_encoding(), Some(TerrainEncoding::Terrarium));

        // Toggle OFF
        let off = ctrl.toggle_terrain(url, TerrainEncoding::Terrarium);
        assert!(!off);
        assert!(!ctrl.terrain_enabled());
        assert_eq!(ctrl.layer_count(), 1);
    }

    #[test]
    fn test_controller_tick() {
        let mut ctrl = MapController::new(MapConfig::default(), 800, 600);
        ctrl.anim.zoom_target = 5.0;
        ctrl.tick(0.016);
        // Zoom should have moved toward target
        assert!(ctrl.zoom_level() > 2.0);
    }

    #[test]
    fn test_controller_drag() {
        let mut ctrl = MapController::new(MapConfig::default(), 800, 600);
        ctrl.begin_drag();
        ctrl.record_drag(100.0, 100.0, 1.0);
        ctrl.record_drag(200.0, 100.0, 1.05);
        ctrl.end_drag(1.05);
        // Should have inertia velocity
        assert!(ctrl.anim.pan_velocity.0.abs() > 0.0);
    }

    #[test]
    fn test_controller_resize() {
        let mut ctrl = MapController::new(MapConfig::default(), 800, 600);
        ctrl.resize(1920, 1080);
        assert_eq!(ctrl.width(), 1920);
        assert_eq!(ctrl.height(), 1080);
    }
}
