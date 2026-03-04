//! Animation state for smooth zoom, inertia panning, and tile fade-in.
//!
//! Uses shared constants and `exp_decay` from `x_planets_core::interaction`.
//! Native uses `std::time::Instant` for timing; the shared module uses `f64`
//! timestamps — conversion happens at call boundaries.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use x_planets_core::interaction::{
    exp_decay, DRAG_SAMPLE_WINDOW, INERTIA_FRICTION, INERTIA_MIN_SPEED, ZOOM_ANIM_SPEED,
};
use x_planets_core::MapEngine;
use x_planets_math::TileCoord;

// Re-export the shared fade duration so existing code (`render_layers.rs`) compiles.
pub(crate) use x_planets_core::interaction::FADE_DURATION;

/// Tracks all running animations so the render loop can tick them each frame.
pub(crate) struct AnimationState {
    // ── Smooth zoom ──
    pub zoom_target: f64,
    pub zoom_anchor: Option<(f64, f64)>,

    // ── Inertia pan ──
    pub pan_velocity: (f64, f64),
    pub last_drag_positions: Vec<((f64, f64), Instant)>,

    // ── Tile fade-in ──
    pub tile_fade_start: HashMap<TileCoord, Instant>,

    // ── Tile visibility tracking ──
    /// Tiles that were visible+available last frame (for shared
    /// `update_tile_visibility` in core).
    pub prev_visible_available: HashSet<TileCoord>,
    /// Tiles that departed the visible set (for zoom-out fade-out).
    pub departing_tiles: HashMap<TileCoord, f64>,

    // ── Double-click detection ──
    pub last_click_time: Option<Instant>,
    pub last_click_pos: Option<(f64, f64)>,

    // ── Reference epoch for f64 timestamps ──
    /// Used to convert `Instant` to f64 seconds for shared core functions.
    start_time: Instant,
}

impl AnimationState {
    pub fn new(initial_zoom: f64) -> Self {
        Self {
            zoom_target: initial_zoom,
            zoom_anchor: None,
            pan_velocity: (0.0, 0.0),
            last_drag_positions: Vec::new(),
            tile_fade_start: HashMap::new(),
            prev_visible_available: HashSet::new(),
            departing_tiles: HashMap::new(),
            last_click_time: None,
            last_click_pos: None,
            start_time: Instant::now(),
        }
    }

    pub fn is_animating(&self, current_zoom: f64) -> bool {
        (self.zoom_target - current_zoom).abs() > 0.001
            || (self.pan_velocity.0.powi(2) + self.pan_velocity.1.powi(2)).sqrt()
                > INERTIA_MIN_SPEED
            || !self.tile_fade_start.is_empty()
    }

    pub fn tick_zoom_for_mode(
        &mut self,
        engine: &mut MapEngine,
        dt: f64,
        mode: x_planets_math::ProjectionMode,
    ) {
        let current = engine.viewport.zoom;
        let target = self
            .zoom_target
            .clamp(engine.camera.min_zoom, engine.camera.max_zoom);
        let diff = target - current;
        if diff.abs() < 0.001 {
            if (engine.viewport.zoom - target).abs() > 1e-9 {
                engine.viewport.zoom = target;
                engine.request_redraw();
            }
            return;
        }
        let new_zoom = exp_decay(current, target, ZOOM_ANIM_SPEED, dt);
        let delta = new_zoom - current;
        match self.zoom_anchor {
            Some((mx, my)) => engine.zoom_at_for_mode(delta, mx, my, mode),
            None => engine.zoom_for_mode(delta, mode),
        }
    }

    pub fn tick_pan_for_mode(
        &mut self,
        engine: &mut MapEngine,
        dt: f64,
        mode: x_planets_math::ProjectionMode,
    ) {
        let (vx, vy) = self.pan_velocity;
        let speed = (vx * vx + vy * vy).sqrt();
        if speed < INERTIA_MIN_SPEED {
            self.pan_velocity = (0.0, 0.0);
            return;
        }
        engine.pan_for_mode(vx * dt, -(vy * dt), mode);
        let friction = (-INERTIA_FRICTION * dt).exp();
        self.pan_velocity = (vx * friction, vy * friction);
    }

    pub fn record_drag(&mut self, pos: (f64, f64)) {
        let now = Instant::now();
        let window_ms = (DRAG_SAMPLE_WINDOW * 1000.0) as u128;
        self.last_drag_positions
            .retain(|(_, t)| now.duration_since(*t).as_millis() < window_ms);
        self.last_drag_positions.push((pos, now));
    }

    pub fn compute_release_velocity(&mut self) {
        let now = Instant::now();
        let window_ms = (DRAG_SAMPLE_WINDOW * 1000.0) as u128;
        self.last_drag_positions
            .retain(|(_, t)| now.duration_since(*t).as_millis() < window_ms);
        if self.last_drag_positions.len() < 2 {
            self.pan_velocity = (0.0, 0.0);
            return;
        }
        let first = &self.last_drag_positions[0];
        let last = &self.last_drag_positions[self.last_drag_positions.len() - 1];
        let dt = last.1.duration_since(first.1).as_secs_f64();
        if dt < 0.001 {
            self.pan_velocity = (0.0, 0.0);
            return;
        }
        let vx = (last.0 .0 - first.0 .0) / dt;
        let vy = (last.0 .1 - first.0 .1) / dt;
        self.pan_velocity = (vx, vy);
        self.last_drag_positions.clear();
    }

    pub fn gc_fades(&mut self, now: Instant) {
        self.tile_fade_start.retain(|_, start| {
            now.duration_since(*start).as_secs_f64() < FADE_DURATION + 0.1
        });
    }

    /// Get elapsed seconds since a tile was loaded, for use with shared crossfade functions.
    pub fn tile_fade_elapsed(&self, coord: &TileCoord, now: Instant) -> Option<f64> {
        self.tile_fade_start
            .get(coord)
            .map(|&start| now.duration_since(start).as_secs_f64())
    }

    /// Convert an `Instant` to f64 seconds since app start, for shared core
    /// functions that use f64 timestamps.
    pub fn to_secs_f64(&self, instant: Instant) -> f64 {
        instant.duration_since(self.start_time).as_secs_f64()
    }
}
