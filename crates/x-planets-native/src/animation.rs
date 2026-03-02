//! Animation state for smooth zoom, inertia panning, and tile fade-in.

use std::collections::HashMap;
use std::time::Instant;

use x_planets_core::MapEngine;
use x_planets_math::TileCoord;

/// Frame-rate-independent exponential decay interpolation.
///
/// Moves `current` toward `target` at a rate determined by `speed`.
/// `speed = 12.0` reaches ~63% of the remaining distance per ~83ms.
fn exp_decay(current: f64, target: f64, speed: f64, dt: f64) -> f64 {
    current + (target - current) * (1.0 - (-speed * dt).exp())
}

/// Duration (seconds) for newly loaded tiles to fade from 0→1 opacity.
pub(crate) const FADE_DURATION: f64 = 0.3;

/// Tracks all running animations so the render loop can tick them each frame.
pub(crate) struct AnimationState {
    // ── Smooth zoom ──
    /// Target zoom level (accumulated from scroll/keyboard input).
    pub zoom_target: f64,
    /// Screen-space anchor point for zoom-toward-cursor.  `None` = zoom at center.
    pub zoom_anchor: Option<(f64, f64)>,

    // ── Inertia pan ──
    /// Current pan velocity in screen pixels/sec.
    pub pan_velocity: (f64, f64),
    /// Recent drag samples for velocity estimation (position, timestamp).
    pub last_drag_positions: Vec<((f64, f64), Instant)>,

    // ── Tile fade-in ──
    /// Maps newly loaded tile coords → the `Instant` they first appeared as GPU textures.
    pub tile_fade_start: HashMap<TileCoord, Instant>,

    // ── Double-click detection ──
    pub last_click_time: Option<Instant>,
    pub last_click_pos: Option<(f64, f64)>,
}

impl AnimationState {
    pub fn new(initial_zoom: f64) -> Self {
        Self {
            zoom_target: initial_zoom,
            zoom_anchor: None,
            pan_velocity: (0.0, 0.0),
            last_drag_positions: Vec::new(),
            tile_fade_start: HashMap::new(),
            last_click_time: None,
            last_click_pos: None,
        }
    }

    /// Returns `true` if any animation is still running and requires continuous redraws.
    pub fn is_animating(&self, current_zoom: f64) -> bool {
        (self.zoom_target - current_zoom).abs() > 0.001
            || (self.pan_velocity.0.powi(2) + self.pan_velocity.1.powi(2)).sqrt() > 1.0
            || !self.tile_fade_start.is_empty()
    }

    /// Advance smooth zoom toward `zoom_target`.
    pub fn tick_zoom(&mut self, engine: &mut MapEngine, dt: f64) {
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
        let new_zoom = exp_decay(current, target, 12.0, dt);
        let delta = new_zoom - current;
        match self.zoom_anchor {
            Some((mx, my)) => engine.zoom_at(delta, mx, my),
            None => engine.zoom(delta),
        }
    }

    /// Advance inertia panning (friction-based velocity decay).
    pub fn tick_pan(&mut self, engine: &mut MapEngine, dt: f64) {
        let (vx, vy) = self.pan_velocity;
        let speed = (vx * vx + vy * vy).sqrt();
        if speed < 1.0 {
            self.pan_velocity = (0.0, 0.0);
            return;
        }
        // Move by velocity × dt (negate vy because screen Y is inverted for pan).
        engine.pan(vx * dt, -(vy * dt));
        // Exponential friction decay.
        let friction = (-6.0 * dt).exp();
        self.pan_velocity = (vx * friction, vy * friction);
    }

    /// Record a drag position sample for velocity estimation.
    pub fn record_drag(&mut self, pos: (f64, f64)) {
        let now = Instant::now();
        // Keep only the last 100ms of samples.
        self.last_drag_positions
            .retain(|(_, t)| now.duration_since(*t).as_millis() < 100);
        self.last_drag_positions.push((pos, now));
    }

    /// Compute pan velocity from recent drag samples (called on mouse-up).
    pub fn compute_release_velocity(&mut self) {
        let now = Instant::now();
        // Need at least 2 samples within the last 100ms.
        self.last_drag_positions
            .retain(|(_, t)| now.duration_since(*t).as_millis() < 100);
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

    /// Garbage-collect finished fade-in entries.
    pub fn gc_fades(&mut self, now: Instant) {
        self.tile_fade_start.retain(|_, start| {
            now.duration_since(*start).as_secs_f64() < FADE_DURATION + 0.1
        });
    }
}
