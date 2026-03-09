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
    AnimationController, CameraAnimation, EasingMode,
};
use crate::pipeline::{resolve_fallbacks, RenderableTile};
#[cfg(feature = "gpu")]
use crate::render::RenderLayerData;
use crate::terrain_data::TerrainTileData;
#[cfg(feature = "gpu")]
use crate::terrain_renderer::TerrainLayerData;

// ═══════════════════════════════════════════════════════════════════
// MapEvent — platform-agnostic event signals
// ═══════════════════════════════════════════════════════════════════

/// Events emitted by [`MapController`] when viewport state changes.
///
/// Platform code (web/native) can drain these each frame and dispatch them
/// to registered callbacks (e.g. JS `on("move", fn)` handlers).
#[derive(Debug, Clone)]
pub enum MapEvent {
    Move { lat: f64, lon: f64 },
    Zoom { zoom: f64 },
    Pitch { pitch: f64 },
    Bearing { bearing: f64 },
    MoveEnd,
    ZoomEnd,
    Click { lat: f64, lon: f64, x: f64, y: f64 },
}

impl MapEvent {
    /// Event name string for JS dispatch.
    pub fn name(&self) -> &'static str {
        match self {
            MapEvent::Move { .. } => "move",
            MapEvent::Zoom { .. } => "zoom",
            MapEvent::Pitch { .. } => "pitch",
            MapEvent::Bearing { .. } => "bearing",
            MapEvent::MoveEnd => "moveend",
            MapEvent::ZoomEnd => "zoomend",
            MapEvent::Click { .. } => "click",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// LayerInfo — public read-only layer metadata
// ═══════════════════════════════════════════════════════════════════

/// Read-only layer metadata returned by [`MapController::get_layer_info`].
#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub name: String,
    pub url: String,
    pub opacity: f32,
    pub visible: bool,
    pub z_order: i32,
    pub kind: String,
}

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
// RenderOutput — what build_render_data() produces (GPU feature only)
// ═══════════════════════════════════════════════════════════════════

/// Assembled render data for one frame (passed to renderers by the platform).
///
/// Only available with the `gpu` feature since it contains wgpu texture references.
#[cfg(feature = "gpu")]
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

    /// Terrain elevation URL template (set from config, changeable at runtime).
    terrain_source_url: String,
    /// Terrain encoding (set from config, changeable at runtime).
    terrain_source_encoding: TerrainEncoding,

    // ── Event tracking ──
    pending_events: Vec<MapEvent>,
    /// Previous frame viewport state for change detection.
    prev_center: (f64, f64),
    prev_zoom: f64,
    prev_bearing: f64,
    prev_pitch: f64,
    /// Whether the camera was animating last frame (for *End events).
    was_moving: bool,
    was_zooming: bool,
}

impl MapController {
    // ── Constructor ──────────────────────────────────────────────

    pub fn new(config: MapConfig, width: u32, height: u32) -> Self {
        let initial_zoom = config.zoom;
        let center = (config.center.lat, config.center.lon);
        let has_terrain = config.layers.iter().any(|l| matches!(l.kind, LayerKind::Terrain { .. }));
        let terrain_url = config.terrain_url.clone();
        let terrain_encoding = config.terrain_encoding;
        let mut engine = MapEngine::new(config, width, height);
        if has_terrain {
            engine.viewport.frustum_margin = 0.15;
        }
        Self {
            engine,
            anim: AnimationController::new(initial_zoom),
            prev_visible_available: HashSet::new(),
            departing_tiles: HashMap::new(),
            terrain: None,
            terrain_source_url: terrain_url,
            terrain_source_encoding: terrain_encoding,
            pending_events: Vec::new(),
            prev_center: center,
            prev_zoom: initial_zoom,
            prev_bearing: 0.0,
            prev_pitch: 0.0,
            was_moving: false,
            was_zooming: false,
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

    // ── Camera Limits ──────────────────────────────────────────

    pub fn min_zoom(&self) -> f64 {
        self.engine.camera.min_zoom
    }

    pub fn set_min_zoom(&mut self, zoom: f64) {
        self.engine.camera.min_zoom = zoom;
    }

    pub fn max_zoom(&self) -> f64 {
        self.engine.camera.max_zoom
    }

    pub fn set_max_zoom(&mut self, zoom: f64) {
        self.engine.camera.max_zoom = zoom;
    }

    pub fn max_pitch(&self) -> f64 {
        self.engine.camera.max_pitch
    }

    pub fn set_max_pitch(&mut self, degrees: f64) {
        self.engine.camera.max_pitch = degrees;
        // Clamp current pitch to new limit
        if self.engine.viewport.pitch > degrees {
            self.engine.viewport.pitch = degrees;
        }
    }

    pub fn tile_budget(&self) -> usize {
        self.engine.viewport.tile_budget
    }

    pub fn set_tile_budget(&mut self, budget: usize) {
        self.engine.viewport.tile_budget = budget;
    }

    /// Set bearing (rotation) directly in degrees.
    pub fn set_bearing(&mut self, degrees: f64) {
        self.engine.camera.set_bearing(&mut self.engine.viewport, degrees);
        self.engine.request_redraw();
    }

    /// Set pitch (tilt) directly in degrees.
    pub fn set_pitch(&mut self, degrees: f64) {
        self.engine.camera.set_pitch(&mut self.engine.viewport, degrees);
        self.engine.request_redraw();
    }

    /// Animate the camera to a new position (smooth ease-in-out).
    ///
    /// `duration_secs` defaults to 2.0 if `None`.
    pub fn fly_to(
        &mut self,
        lat: f64,
        lon: f64,
        zoom: f64,
        duration_secs: Option<f64>,
        bearing: Option<f64>,
        pitch: Option<f64>,
    ) {
        self.start_camera_anim(lat, lon, zoom, duration_secs, bearing, pitch, EasingMode::FlyTo);
    }

    /// Animate the camera to a new position (linear interpolation).
    ///
    /// `duration_secs` defaults to 1.0 if `None`.
    pub fn ease_to(
        &mut self,
        lat: f64,
        lon: f64,
        zoom: f64,
        duration_secs: Option<f64>,
        bearing: Option<f64>,
        pitch: Option<f64>,
    ) {
        self.start_camera_anim(lat, lon, zoom, duration_secs, bearing, pitch, EasingMode::EaseTo);
    }

    /// Jump the camera to a new position instantly (no animation).
    pub fn jump_to(&mut self, lat: f64, lon: f64, zoom: f64, bearing: Option<f64>, pitch: Option<f64>) {
        self.anim.stop_animation();
        self.set_center(lat, lon);
        self.set_zoom(zoom);
        if let Some(b) = bearing {
            self.set_bearing(b);
        }
        if let Some(p) = pitch {
            self.set_pitch(p);
        }
    }

    /// Cancel any running camera animation.
    pub fn stop_animation(&mut self) {
        self.anim.stop_animation();
    }

    fn start_camera_anim(
        &mut self,
        lat: f64,
        lon: f64,
        zoom: f64,
        duration_secs: Option<f64>,
        bearing: Option<f64>,
        pitch: Option<f64>,
        easing: EasingMode,
    ) {
        let default_dur = match easing {
            EasingMode::FlyTo => 2.0,
            EasingMode::EaseTo => 1.0,
        };
        let duration = duration_secs.unwrap_or(default_dur).max(0.01);

        let target_bearing = bearing.unwrap_or(self.engine.viewport.bearing);
        let target_pitch = pitch.unwrap_or(self.engine.viewport.pitch);

        // Shortest-path bearing interpolation
        let mut start_bearing = self.engine.viewport.bearing;
        let delta_b = target_bearing - start_bearing;
        if delta_b > 180.0 {
            start_bearing += 360.0;
        } else if delta_b < -180.0 {
            start_bearing -= 360.0;
        }

        let anim = CameraAnimation {
            start_center: self.engine.viewport.center,
            target_center: GeoCoord::new(lat, lon),
            start_zoom: self.engine.viewport.zoom,
            target_zoom: zoom.clamp(self.engine.camera.min_zoom, self.engine.camera.max_zoom),
            start_bearing,
            target_bearing,
            start_pitch: self.engine.viewport.pitch,
            target_pitch,
            duration,
            elapsed: 0.0,
            easing,
        };
        self.anim.start_camera_animation(anim);
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

    // ── Coordinate conversion ─────────────────────────────────

    /// Convert geographic (lat, lon) to screen pixel coordinates.
    ///
    /// Returns `None` if the point is outside the visible area.
    pub fn project(&self, lat: f64, lon: f64) -> Option<(f64, f64)> {
        let vp = &self.engine.viewport;
        let merc = x_planets_math::geo_to_mercator(&GeoCoord::new(lat, lon));
        let center_merc = x_planets_math::geo_to_mercator(&vp.center);

        let scale = 2.0_f64.powf(vp.zoom);
        let aspect = vp.width as f64 / vp.height as f64;

        // Offset in Mercator space (handling wrap-around)
        let mut dx = merc.x - center_merc.x;
        if dx > 0.5 { dx -= 1.0; }
        if dx < -0.5 { dx += 1.0; }
        let dy = merc.y - center_merc.y;

        // Apply bearing rotation
        let bearing_rad = vp.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();
        let rx = cos_b * dx - sin_b * dy;
        let ry = sin_b * dx + cos_b * dy;

        // Convert to screen pixels
        let sx = vp.width as f64 * 0.5 + rx * scale / aspect * vp.width as f64;
        let sy = vp.height as f64 * 0.5 + ry * scale * vp.height as f64;

        Some((sx, sy))
    }

    /// Convert screen pixel coordinates to geographic (lat, lon).
    ///
    /// Returns `None` if the point cannot be unprojected.
    pub fn unproject(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let vp = &self.engine.viewport;
        let center_merc = x_planets_math::geo_to_mercator(&vp.center);

        let scale = 2.0_f64.powf(vp.zoom);
        let aspect = vp.width as f64 / vp.height as f64;

        // Screen → normalized offset
        let dx_screen = (x - vp.width as f64 * 0.5) / vp.width as f64 * aspect / scale;
        let dy_screen = (y - vp.height as f64 * 0.5) / vp.height as f64 / scale;

        // Reverse bearing rotation
        let bearing_rad = vp.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();
        let mx = cos_b * dx_screen + sin_b * dy_screen;
        let my = -sin_b * dx_screen + cos_b * dy_screen;

        let merc_x = (center_merc.x + mx).rem_euclid(1.0);
        let merc_y = center_merc.y + my;
        if merc_y < 0.0 || merc_y > 1.0 {
            return None;
        }

        let geo = x_planets_math::mercator_to_geo(glam::DVec2::new(merc_x, merc_y));
        Some((geo.lat, geo.lon))
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

    /// Get all layer names in z-order (bottom to top).
    pub fn layer_names(&self) -> Vec<String> {
        self.engine.layers.iter().map(|l| l.config.name.clone()).collect()
    }

    /// Get layer info by name: (url, opacity, visible, z_order, kind).
    pub fn get_layer_info(&self, name: &str) -> Option<LayerInfo> {
        self.engine.get_layer(name).map(|l| LayerInfo {
            name: l.config.name.clone(),
            url: l.config.tile_source_url.clone(),
            opacity: l.config.opacity,
            visible: l.config.visible,
            z_order: l.config.z_order,
            kind: match &l.config.kind {
                LayerKind::Raster => "raster".to_string(),
                LayerKind::Terrain { .. } => "terrain".to_string(),
                LayerKind::Tiles3d => "3dtiles".to_string(),
            },
        })
    }

    // ── Terrain high-level API ──────────────────────────────────

    /// Toggle terrain on/off.  Returns the new state (true = terrain ON).
    ///
    /// Uses the terrain URL and encoding from the `MapConfig` stored in the
    /// engine.  Call [`set_terrain_source`] first if you need to change them.
    ///
    /// Terrain is a rendering property, not a layer.  No layers are added or
    /// removed from the engine.  Platform code should start/stop loading
    /// elevation data on the raster layer when this returns true/false.
    pub fn toggle_terrain(&mut self) -> bool {
        if self.terrain.is_some() {
            // Turn OFF
            self.terrain = None;
            self.engine.viewport.frustum_margin = 0.05;
            self.engine.request_redraw();
            log::info!("Terrain: OFF");
            false
        } else {
            let url = self.terrain_source_url.clone();
            let encoding = self.terrain_source_encoding;
            if url.is_empty() {
                log::warn!("Terrain toggle ignored: no terrain URL configured");
                return false;
            }

            // Turn ON — find the first raster layer to use as imagery
            let imagery_name = self
                .engine
                .layers
                .iter()
                .find(|l| matches!(l.config.kind, LayerKind::Raster))
                .map(|l| l.config.name.clone())
                .unwrap_or_else(|| "base".to_string());

            self.terrain = Some(TerrainState {
                url,
                encoding,
                imagery_layer_name: imagery_name,
            });

            // Widen frustum margin — terrain displacement can shift tiles
            // into the viewport even when the flat ground-plane check says
            // they're off-screen.
            self.engine.viewport.frustum_margin = 0.15;
            self.engine.request_redraw();
            log::info!("Terrain: ON");
            true
        }
    }

    /// Set the terrain elevation source URL and encoding.
    ///
    /// This configures which elevation tiles to fetch when terrain is toggled on.
    /// If terrain is already enabled, the change takes effect on the next toggle cycle.
    pub fn set_terrain_source(&mut self, url: &str, encoding: TerrainEncoding) {
        self.terrain_source_url = url.to_string();
        self.terrain_source_encoding = encoding;
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
    ///
    /// Also detects viewport state changes and pushes [`MapEvent`]s.
    pub fn tick(&mut self, dt_secs: f64) {
        let mode = self.rendering_mode();
        self.anim.tick_with_mode(&mut self.engine, dt_secs, mode);

        // ── Detect state changes and emit events ──
        let vp = &self.engine.viewport;
        let cur_center = (vp.center.lat, vp.center.lon);
        let cur_zoom = vp.zoom;
        let cur_bearing = vp.bearing;
        let cur_pitch = vp.pitch;

        let is_moving = (cur_center.0 - self.prev_center.0).abs() > 1e-9
            || (cur_center.1 - self.prev_center.1).abs() > 1e-9;
        let is_zooming = (cur_zoom - self.prev_zoom).abs() > 1e-6;

        if is_moving {
            self.pending_events.push(MapEvent::Move { lat: cur_center.0, lon: cur_center.1 });
        }
        if is_zooming {
            self.pending_events.push(MapEvent::Zoom { zoom: cur_zoom });
        }
        if (cur_bearing - self.prev_bearing).abs() > 1e-6 {
            self.pending_events.push(MapEvent::Bearing { bearing: cur_bearing });
        }
        if (cur_pitch - self.prev_pitch).abs() > 1e-6 {
            self.pending_events.push(MapEvent::Pitch { pitch: cur_pitch });
        }

        // Emit *End events when movement/zoom stops
        if self.was_moving && !is_moving {
            self.pending_events.push(MapEvent::MoveEnd);
        }
        if self.was_zooming && !is_zooming {
            self.pending_events.push(MapEvent::ZoomEnd);
        }

        self.prev_center = cur_center;
        self.prev_zoom = cur_zoom;
        self.prev_bearing = cur_bearing;
        self.prev_pitch = cur_pitch;
        self.was_moving = is_moving;
        self.was_zooming = is_zooming;
    }

    /// Drain all pending events since the last call.
    pub fn drain_events(&mut self) -> Vec<MapEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Push a click event (called by platform input handler).
    pub fn push_click(&mut self, lat: f64, lon: f64, x: f64, y: f64) {
        self.pending_events.push(MapEvent::Click { lat, lon, x, y });
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

    // ── Render data assembly (GPU feature only) ────────────────

    /// Build render data for all visible layers.
    ///
    /// Platform code provides:
    /// - `layer_views`: per-layer GPU cache state (implements `LayerStateView`)
    /// - `texture_fn`: closure that returns `&wgpu::TextureView` for a layer+coord
    ///
    /// Returns assembled `RenderOutput` ready to pass to renderers.
    ///
    /// Only available with the `gpu` feature.
    #[cfg(feature = "gpu")]
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
                            // Always render ALL tiles as flat raster first — this
                            // provides a stable, flicker-free base layer.  Terrain
                            // tiles are then rendered on top (LoadOp::Load) and
                            // overdraw the raster underneath with displaced meshes.
                            //
                            // The old approach split tiles exclusively into flat vs
                            // terrain sets, which caused flickering as elevation data
                            // loaded asynchronously (tiles popping between renderers).
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

                            // Terrain overlay for tiles with elevation data
                            let (_, terrain_visible) =
                                split_by_elevation(visible, *lv, *lv);

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
                    let fade_out = (1.0 - elapsed / self.anim.config.fade_duration).max(0.0) as f32;
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
// Free functions for render data assembly (GPU feature only)
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "gpu")]
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

#[cfg(feature = "gpu")]
/// Split visible tiles into (flat, terrain) based on elevation data availability.
///
/// A tile goes to the terrain list if the layer state has elevation data for it
/// OR any of its ancestor tiles (parent fallback).  Otherwise it stays in flat.
fn split_by_elevation(
    visible: &[VisibleTile],
    terrain_lv: &dyn LayerStateView,
    imagery_lv: &dyn LayerStateView,
) -> (Vec<VisibleTile>, Vec<VisibleTile>) {
    let available_raster = imagery_lv.available_raster_coords();
    let mut flat = Vec::new();
    let mut terrain = Vec::new();
    for vt in visible {
        let has_elev = {
            let mut c = Some(vt.coord);
            let mut found = false;
            while let Some(candidate) = c {
                if terrain_lv.terrain_tile_data(&candidate).is_some() {
                    found = true;
                    break;
                }
                c = candidate.parent();
            }
            found
        };
        // Only classify as terrain if imagery is also available (exact or parent fallback).
        // Without this check, tiles with elevation but no imagery texture get skipped by
        // the terrain renderer AND excluded from the raster pass, causing black holes.
        let has_imagery = has_elev && {
            let mut c = Some(vt.coord);
            let mut found = false;
            while let Some(candidate) = c {
                if available_raster.contains(&candidate) {
                    found = true;
                    break;
                }
                c = candidate.parent();
            }
            found
        };
        if has_imagery {
            terrain.push(vt.clone());
        } else {
            flat.push(vt.clone());
        }
    }
    (flat, terrain)
}

#[cfg(feature = "gpu")]
fn build_terrain_layer<'a>(
    layer_name: &'a str,
    layer_opacity: f32,
    terrain_lv: &'a dyn LayerStateView,
    imagery_lv: &'a dyn LayerStateView,
    texture_fn: &dyn Fn(&str, &TileCoord) -> Option<&'a wgpu::TextureView>,
    visible: &[VisibleTile],
    _fade_fn: &dyn Fn(&TileCoord) -> Option<f64>,
) -> (TerrainLayerData<'a>, Option<TerrainLayerData<'a>>) {
    let available = imagery_lv.available_raster_coords();

    let imagery_views: HashMap<TileCoord, &'a wgpu::TextureView> = available
        .iter()
        .filter_map(|coord| texture_fn(imagery_lv.name(), coord).map(|tv| (*coord, tv)))
        .collect();

    // Terrain skips crossfade entirely.  Unlike flat raster tiles, terrain
    // meshes have 3D displaced geometry that differs between LOD levels.
    // Crossfade alpha-blends two mismatched 3D surfaces → depth-test
    // failures, holes, and shimmer.  The raster base layer (always rendered
    // underneath) already provides smooth visual transitions, so terrain
    // tiles can snap in at full opacity without visual popping.
    let renderable = resolve_fallbacks(visible, available);

    // Elevation data with parent fallback
    let mut elevation_data: HashMap<TileCoord, (&'a TerrainTileData, TileCoord)> = HashMap::new();
    let all_needed: HashSet<TileCoord> = renderable
        .iter()
        .map(|rt| rt.coord)
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
        imagery_views,
        elevation_data,
        tile_opacity_overrides: HashMap::new(),
    };

    // No overlay — terrain transitions are instant.
    (base, None)
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

        assert!(!ctrl.terrain_enabled());
        assert_eq!(ctrl.layer_count(), 1);

        // Set terrain source first
        ctrl.set_terrain_source(
            "https://example.com/{z}/{x}/{y}.png",
            TerrainEncoding::Terrarium,
        );

        // Toggle ON — no layer added (terrain is a rendering property)
        let on = ctrl.toggle_terrain();
        assert!(on);
        assert!(ctrl.terrain_enabled());
        assert_eq!(ctrl.layer_count(), 1); // unchanged
        assert!(ctrl.terrain_imagery_name().is_some());
        assert!(ctrl.terrain_url().is_some());
        assert_eq!(ctrl.terrain_encoding(), Some(TerrainEncoding::Terrarium));

        // Toggle OFF
        let off = ctrl.toggle_terrain();
        assert!(!off);
        assert!(!ctrl.terrain_enabled());
        assert_eq!(ctrl.layer_count(), 1);
    }

    #[test]
    fn test_controller_terrain_toggle_without_url() {
        let mut ctrl = MapController::new(MapConfig::default(), 800, 600);
        // No terrain source configured — toggle should return false
        let on = ctrl.toggle_terrain();
        assert!(!on);
        assert!(!ctrl.terrain_enabled());
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
