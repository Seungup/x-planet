//! Constants, exponential decay utility, and [`AnimationController`].

use std::collections::HashMap;

use x_planets_math::{GeoCoord, TileCoord};

use crate::engine::MapEngine;

// ═══════════════════════════════════════════════════════════════════
// Constants (shared between all platforms)
// ═══════════════════════════════════════════════════════════════════

/// Duration (seconds) for newly loaded tiles to fade from 0→1 opacity.
pub const FADE_DURATION: f64 = 0.5;
/// Double-click time window (seconds).
pub const DOUBLE_CLICK_TIME: f64 = 0.3;
/// Double-click max distance (pixels).
pub const DOUBLE_CLICK_DIST: f64 = 10.0;
/// Pan distance per arrow-key press (pixels).
pub const PAN_AMOUNT: f64 = 50.0;
/// Zoom step per +/- key press.
pub const ZOOM_STEP: f64 = 0.5;
/// Rotation per Q/E key press (degrees).
pub const KEYBOARD_ROTATE: f64 = 10.0;
/// Sensitivity for pitch drag (degrees per pixel).
pub const PITCH_SENSITIVITY: f64 = 0.3;
/// Sensitivity for rotation drag (degrees per pixel).
pub const ROTATE_SENSITIVITY: f64 = 0.3;
/// Grace period after touch count changes (seconds).
/// Suppresses gesture actions during finger transitions to prevent jitter.
pub const TOUCH_GRACE_PERIOD: f64 = 0.08;
/// Exponential friction speed for inertia decay.
pub const INERTIA_FRICTION: f64 = 4.5;
/// Minimum speed (px/sec) below which inertia stops.
pub const INERTIA_MIN_SPEED: f64 = 1.0;
/// Exponential decay speed for smooth zoom animation.
pub const ZOOM_ANIM_SPEED: f64 = 16.0;
/// Time window (seconds) for drag velocity sampling.
pub const DRAG_SAMPLE_WINDOW: f64 = 0.1;

// ═══════════════════════════════════════════════════════════════════
// InteractionConfig — runtime-tunable interaction parameters
// ═══════════════════════════════════════════════════════════════════

/// Runtime-tunable interaction parameters.
///
/// Default values match the `pub const` values above.  Modify these on
/// `AnimationController::config` to adjust interaction behaviour at runtime.
#[derive(Debug, Clone)]
pub struct InteractionConfig {
    pub fade_duration: f64,
    pub double_click_time: f64,
    pub double_click_dist: f64,
    pub pan_amount: f64,
    pub zoom_step: f64,
    pub keyboard_rotate: f64,
    pub pitch_sensitivity: f64,
    pub rotate_sensitivity: f64,
    pub touch_grace_period: f64,
    pub inertia_friction: f64,
    pub inertia_min_speed: f64,
    pub zoom_anim_speed: f64,
    pub drag_sample_window: f64,
}

impl Default for InteractionConfig {
    fn default() -> Self {
        Self {
            fade_duration: FADE_DURATION,
            double_click_time: DOUBLE_CLICK_TIME,
            double_click_dist: DOUBLE_CLICK_DIST,
            pan_amount: PAN_AMOUNT,
            zoom_step: ZOOM_STEP,
            keyboard_rotate: KEYBOARD_ROTATE,
            pitch_sensitivity: PITCH_SENSITIVITY,
            rotate_sensitivity: ROTATE_SENSITIVITY,
            touch_grace_period: TOUCH_GRACE_PERIOD,
            inertia_friction: INERTIA_FRICTION,
            inertia_min_speed: INERTIA_MIN_SPEED,
            zoom_anim_speed: ZOOM_ANIM_SPEED,
            drag_sample_window: DRAG_SAMPLE_WINDOW,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Utility
// ═══════════════════════════════════════════════════════════════════

/// Frame-rate-independent exponential decay interpolation.
///
/// Moves `current` toward `target` at a rate determined by `speed`.
/// `speed = 12.0` reaches ~63% of remaining distance per ~83ms.
pub fn exp_decay(current: f64, target: f64, speed: f64, dt: f64) -> f64 {
    current + (target - current) * (1.0 - (-speed * dt).exp())
}

// ═══════════════════════════════════════════════════════════════════
// FlyTo / EaseTo animation
// ═══════════════════════════════════════════════════════════════════

/// Easing mode for camera transition animations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EasingMode {
    /// Ease-in-out (smooth start and end, like Mapbox `flyTo`).
    FlyTo,
    /// Linear interpolation (constant speed, like Mapbox `easeTo`).
    EaseTo,
}

/// State for an in-progress camera transition animation.
#[derive(Clone, Debug)]
pub struct CameraAnimation {
    pub start_center: GeoCoord,
    pub target_center: GeoCoord,
    pub start_zoom: f64,
    pub target_zoom: f64,
    pub start_bearing: f64,
    pub target_bearing: f64,
    pub start_pitch: f64,
    pub target_pitch: f64,
    pub duration: f64,
    pub elapsed: f64,
    pub easing: EasingMode,
}

impl CameraAnimation {
    /// Compute the interpolation factor `t` in [0, 1] based on elapsed/duration.
    fn progress(&self) -> f64 {
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0);
        match self.easing {
            EasingMode::EaseTo => t,
            EasingMode::FlyTo => {
                // Smooth ease-in-out: 3t² - 2t³
                t * t * (3.0 - 2.0 * t)
            }
        }
    }

    /// Whether the animation has finished.
    pub fn is_done(&self) -> bool {
        self.elapsed >= self.duration
    }
}

// ═══════════════════════════════════════════════════════════════════
// AnimationController
// ═══════════════════════════════════════════════════════════════════

/// Platform-agnostic animation controller for smooth zoom, inertia panning,
/// double-click detection, and tile fade-in timing.
///
/// All timestamps are `f64` seconds (relative to an arbitrary epoch).
pub struct AnimationController {
    /// Runtime-tunable interaction parameters.
    pub config: InteractionConfig,

    // ── Smooth zoom ──
    /// Target zoom level (accumulated from scroll/keyboard).
    pub zoom_target: f64,
    /// Screen-space anchor for zoom-toward-cursor. `None` = zoom at center.
    pub zoom_anchor: Option<(f64, f64)>,

    // ── Inertia pan ──
    /// Current pan velocity in screen pixels/sec.
    pub pan_velocity: (f64, f64),
    /// Recent drag samples: (position, time_secs).
    drag_samples: Vec<((f64, f64), f64)>,

    // ── Double-click ──
    last_click_time: Option<f64>,
    last_click_pos: Option<(f64, f64)>,

    // ── Tile fade-in ──
    tile_fade_start: HashMap<TileCoord, f64>,

    // ── Camera animation (flyTo / easeTo) ──
    pub camera_anim: Option<CameraAnimation>,

    // ── Misc ──
    /// Last known mouse position (for zoom anchor fallback).
    pub last_mouse_pos: Option<(f64, f64)>,
}

impl AnimationController {
    pub fn new(initial_zoom: f64) -> Self {
        Self {
            config: InteractionConfig::default(),
            zoom_target: initial_zoom,
            zoom_anchor: None,
            pan_velocity: (0.0, 0.0),
            drag_samples: Vec::new(),
            last_click_time: None,
            last_click_pos: None,
            tile_fade_start: HashMap::new(),
            camera_anim: None,
            last_mouse_pos: None,
        }
    }

    /// Returns `true` if any animation is still running and requires continuous redraws.
    pub fn is_animating(&self, current_zoom: f64) -> bool {
        let zoom_active = (self.zoom_target - current_zoom).abs() > 0.001;
        let pan_active = {
            let (vx, vy) = self.pan_velocity;
            (vx * vx + vy * vy).sqrt() > self.config.inertia_min_speed
        };
        let fades_active = !self.tile_fade_start.is_empty();
        let camera_active = self.camera_anim.as_ref().map_or(false, |a| !a.is_done());
        zoom_active || pan_active || fades_active || camera_active
    }

    /// Advance smooth zoom and inertia pan. Call once per frame with `dt` in seconds.
    pub fn tick(&mut self, engine: &mut MapEngine, dt: f64) {
        self.tick_with_mode(engine, dt, x_planets_math::ProjectionMode::Mercator);
    }

    /// Advance smooth zoom and inertia pan, projection-aware.
    pub fn tick_with_mode(
        &mut self,
        engine: &mut MapEngine,
        dt: f64,
        mode: x_planets_math::ProjectionMode,
    ) {
        // ── Camera animation (flyTo / easeTo) ──
        if let Some(ref mut anim) = self.camera_anim {
            anim.elapsed += dt;
            let t = anim.progress();

            let lat = anim.start_center.lat + (anim.target_center.lat - anim.start_center.lat) * t;
            let lon = anim.start_center.lon + (anim.target_center.lon - anim.start_center.lon) * t;
            let zoom = anim.start_zoom + (anim.target_zoom - anim.start_zoom) * t;
            let bearing = anim.start_bearing + (anim.target_bearing - anim.start_bearing) * t;
            let pitch = anim.start_pitch + (anim.target_pitch - anim.start_pitch) * t;

            engine.viewport.center = GeoCoord::new(lat, lon);
            engine.viewport.zoom = zoom.clamp(engine.camera.min_zoom, engine.camera.max_zoom);
            engine.camera.set_bearing(&mut engine.viewport, bearing);
            engine.camera.set_pitch(&mut engine.viewport, pitch);
            engine.request_redraw();

            if anim.is_done() {
                self.zoom_target = engine.viewport.zoom;
                self.camera_anim = None;
            }
            return; // Camera animation overrides smooth zoom / inertia
        }

        // ── Smooth zoom ──
        let current = engine.viewport.zoom;
        let target = self
            .zoom_target
            .clamp(engine.camera.min_zoom, engine.camera.max_zoom);
        let diff = target - current;
        if diff.abs() > 0.001 {
            let new_zoom = exp_decay(current, target, self.config.zoom_anim_speed, dt);
            let delta = new_zoom - current;
            match self.zoom_anchor {
                Some((mx, my)) => engine.zoom_at_for_mode(delta, mx, my, mode),
                None => engine.zoom(delta),
            }
        } else if (current - target).abs() > 1e-9 {
            engine.viewport.zoom = target;
            engine.request_redraw();
        }

        // ── Inertia pan ──
        let (vx, vy) = self.pan_velocity;
        let speed = (vx * vx + vy * vy).sqrt();
        if speed > self.config.inertia_min_speed {
            engine.pan_for_mode(vx * dt, -(vy * dt), mode);
            let friction = (-self.config.inertia_friction * dt).exp();
            self.pan_velocity = (vx * friction, vy * friction);
        } else {
            self.pan_velocity = (0.0, 0.0);
        }
    }

    /// Record a drag position sample for velocity estimation.
    pub fn record_drag(&mut self, pos: (f64, f64), time_secs: f64) {
        self.drag_samples
            .retain(|(_, t)| time_secs - t < self.config.drag_sample_window);
        self.drag_samples.push((pos, time_secs));
    }

    /// Compute pan velocity from recent drag samples (call on mouse-up/touch-end).
    pub fn compute_release_velocity(&mut self, time_secs: f64) {
        self.drag_samples
            .retain(|(_, t)| time_secs - t < self.config.drag_sample_window);
        if self.drag_samples.len() < 2 {
            self.pan_velocity = (0.0, 0.0);
            return;
        }
        let first = &self.drag_samples[0];
        let last = &self.drag_samples[self.drag_samples.len() - 1];
        let dt = last.1 - first.1;
        if dt < 0.001 {
            self.pan_velocity = (0.0, 0.0);
            return;
        }
        let vx = (last.0 .0 - first.0 .0) / dt;
        let vy = (last.0 .1 - first.0 .1) / dt;
        self.pan_velocity = (vx, vy);
        self.drag_samples.clear();
    }

    /// Called on mouse-down / touch-start to stop inertia and prepare for new drag.
    pub fn begin_drag(&mut self) {
        self.pan_velocity = (0.0, 0.0);
        self.drag_samples.clear();
    }

    /// Check if a click at `pos` at `time_secs` is a double-click.
    /// Returns `true` if double-click detected, also resets state to prevent triple-click.
    pub fn check_double_click(&mut self, pos: (f64, f64), time_secs: f64) -> bool {
        let is_double = self
            .last_click_time
            .map(|t| time_secs - t < self.config.double_click_time)
            .unwrap_or(false)
            && self
                .last_click_pos
                .map(|(lx, ly)| {
                    ((pos.0 - lx).powi(2) + (pos.1 - ly).powi(2)).sqrt() < self.config.double_click_dist
                })
                .unwrap_or(false);

        if is_double {
            self.last_click_time = None; // prevent triple-click
            self.last_click_pos = None;
        } else {
            self.last_click_time = Some(time_secs);
            self.last_click_pos = Some(pos);
        }
        is_double
    }

    // ── Camera animation ──

    /// Start a flyTo or easeTo camera animation.
    pub fn start_camera_animation(&mut self, anim: CameraAnimation) {
        self.pan_velocity = (0.0, 0.0); // Stop inertia
        self.camera_anim = Some(anim);
    }

    /// Cancel any running camera animation immediately.
    pub fn stop_animation(&mut self) {
        self.camera_anim = None;
    }

    // ── Tile fade-in ──

    /// Register that a tile texture became available at `time_secs`.
    pub fn register_tile_loaded(&mut self, coord: TileCoord, time_secs: f64) {
        self.tile_fade_start.insert(coord, time_secs);
    }

    /// Get elapsed time since tile was loaded. Returns `None` if not tracked.
    pub fn tile_fade_elapsed(&self, coord: &TileCoord, now_secs: f64) -> Option<f64> {
        self.tile_fade_start.get(coord).map(|&start| now_secs - start)
    }

    /// Garbage-collect finished fade entries.
    pub fn gc_fades(&mut self, now_secs: f64) {
        self.tile_fade_start
            .retain(|_, start| now_secs - *start < self.config.fade_duration + 0.1);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_decay_converges() {
        let val = exp_decay(0.0, 10.0, 12.0, 1.0);
        assert!(val > 9.99, "should nearly converge after 1s at speed=12");
    }

    #[test]
    fn test_exp_decay_no_overshoot() {
        let val = exp_decay(0.0, 10.0, 12.0, 0.016);
        assert!(val > 0.0 && val < 10.0, "should not overshoot");
    }

    #[test]
    fn test_double_click_detection() {
        let mut anim = AnimationController::new(5.0);
        let pos = (100.0, 100.0);
        // First click
        assert!(!anim.check_double_click(pos, 1.0));
        // Second click within threshold
        assert!(anim.check_double_click(pos, 1.2));
        // Third click should NOT be double (state was reset)
        assert!(!anim.check_double_click(pos, 1.4));
    }

    #[test]
    fn test_double_click_too_far() {
        let mut anim = AnimationController::new(5.0);
        assert!(!anim.check_double_click((100.0, 100.0), 1.0));
        assert!(!anim.check_double_click((200.0, 200.0), 1.2)); // too far
    }

    #[test]
    fn test_double_click_too_slow() {
        let mut anim = AnimationController::new(5.0);
        let pos = (100.0, 100.0);
        assert!(!anim.check_double_click(pos, 1.0));
        assert!(!anim.check_double_click(pos, 2.0)); // >300ms
    }

    #[test]
    fn test_drag_velocity_estimation() {
        let mut anim = AnimationController::new(5.0);
        anim.record_drag((0.0, 0.0), 1.0);
        anim.record_drag((100.0, 0.0), 1.05);
        anim.record_drag((200.0, 0.0), 1.1);
        anim.compute_release_velocity(1.1);
        let (vx, vy) = anim.pan_velocity;
        assert!((vx - 2000.0).abs() < 1.0, "vx={} expected ~2000", vx);
        assert!(vy.abs() < 1.0, "vy={} expected ~0", vy);
    }

    #[test]
    fn test_drag_velocity_too_few_samples() {
        let mut anim = AnimationController::new(5.0);
        anim.record_drag((0.0, 0.0), 1.0);
        anim.compute_release_velocity(1.1);
        assert_eq!(anim.pan_velocity, (0.0, 0.0));
    }

    #[test]
    fn test_tile_fade_lifecycle() {
        let mut anim = AnimationController::new(5.0);
        let coord = TileCoord::new(5, 10, 10);
        anim.register_tile_loaded(coord, 1.0);

        // During fade
        let elapsed = anim.tile_fade_elapsed(&coord, 1.15);
        assert!(elapsed.is_some());
        assert!((elapsed.unwrap() - 0.15).abs() < 0.001);

        // After fade
        let elapsed = anim.tile_fade_elapsed(&coord, 1.0 + FADE_DURATION + 0.1);
        assert!(elapsed.unwrap() > FADE_DURATION);

        // GC removes old entries
        anim.gc_fades(1.0 + FADE_DURATION + 0.2);
        assert!(anim.tile_fade_elapsed(&coord, 1.0 + FADE_DURATION + 0.2).is_none());
    }

    #[test]
    fn test_animation_controller_is_animating_idle() {
        let anim = AnimationController::new(5.0);
        // When zoom_target == current_zoom and no velocity or fades
        assert!(!anim.is_animating(5.0));
    }

    #[test]
    fn test_animation_controller_is_animating_zoom() {
        let mut anim = AnimationController::new(5.0);
        anim.zoom_target = 8.0;
        assert!(anim.is_animating(5.0));
    }

    #[test]
    fn test_animation_controller_is_animating_pan() {
        let mut anim = AnimationController::new(5.0);
        anim.pan_velocity = (100.0, 0.0);
        assert!(anim.is_animating(5.0));
    }

    #[test]
    fn test_animation_controller_tick_zoom_converges() {
        let mut anim = AnimationController::new(5.0);
        anim.zoom_target = 8.0;

        let config = crate::engine::MapConfig::default();
        let mut engine = MapEngine::new(config, 800, 600);

        // Tick many frames
        for _ in 0..100 {
            anim.tick(&mut engine, 0.016);
        }

        // Zoom should have converged close to target
        assert!(
            (engine.viewport.zoom - 8.0).abs() < 0.01,
            "Zoom should converge to target: got {}",
            engine.viewport.zoom
        );
    }

    #[test]
    fn test_animation_controller_tick_inertia_decays() {
        let mut anim = AnimationController::new(5.0);
        anim.pan_velocity = (500.0, 0.0);

        let config = crate::engine::MapConfig::default();
        let mut engine = MapEngine::new(config, 800, 600);

        // Tick several frames
        for _ in 0..60 {
            anim.tick(&mut engine, 0.016);
        }

        // Pan velocity should have decayed
        let (vx, _) = anim.pan_velocity;
        assert!(
            vx.abs() < 10.0,
            "Inertia should decay: velocity still at {}",
            vx
        );
    }

    #[test]
    fn test_animation_controller_begin_drag_stops_inertia() {
        let mut anim = AnimationController::new(5.0);
        anim.pan_velocity = (500.0, 300.0);

        anim.begin_drag();

        assert_eq!(anim.pan_velocity, (0.0, 0.0));
    }

    #[test]
    fn test_animation_controller_gc_fades() {
        let mut anim = AnimationController::new(5.0);
        let c1 = TileCoord::new(5, 10, 10);
        let c2 = TileCoord::new(5, 11, 10);
        anim.register_tile_loaded(c1, 1.0);
        anim.register_tile_loaded(c2, 2.0);

        // At t=2.0, c1 has been loaded for 1.0s (past FADE_DURATION + 0.1s grace)
        anim.gc_fades(2.0);
        assert!(anim.tile_fade_elapsed(&c1, 2.0).is_none(), "c1 should be GC'd");
        assert!(anim.tile_fade_elapsed(&c2, 2.0).is_some(), "c2 should remain");
    }
}
