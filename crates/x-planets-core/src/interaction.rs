//! Shared input, animation, and gesture logic for all platforms.
//!
//! Uses `f64` timestamps (seconds) instead of `std::time::Instant` so the same
//! code works on both native (Instant → f64) and WASM (performance.now()/1000).
//!
//! # Modules
//! - [`AnimationController`]: smooth zoom, inertia panning, double-click, tile fades
//! - [`TouchGestureState`]: multi-touch gestures with grace period for finger transitions
//! - [`crossfade`]: tile crossfade overlay computation (shared between native and web)

use std::collections::{HashMap, HashSet};

use x_planets_math::{TileCoord, VisibleTile};

use crate::engine::MapEngine;
use crate::pipeline::RenderableTile;

// ═══════════════════════════════════════════════════════════════════
// Constants (shared between all platforms)
// ═══════════════════════════════════════════════════════════════════

/// Duration (seconds) for newly loaded tiles to fade from 0→1 opacity.
pub const FADE_DURATION: f64 = 0.3;
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
// AnimationController
// ═══════════════════════════════════════════════════════════════════

/// Platform-agnostic animation controller for smooth zoom, inertia panning,
/// double-click detection, and tile fade-in timing.
///
/// All timestamps are `f64` seconds (relative to an arbitrary epoch).
pub struct AnimationController {
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

    // ── Misc ──
    /// Last known mouse position (for zoom anchor fallback).
    pub last_mouse_pos: Option<(f64, f64)>,
}

impl AnimationController {
    pub fn new(initial_zoom: f64) -> Self {
        Self {
            zoom_target: initial_zoom,
            zoom_anchor: None,
            pan_velocity: (0.0, 0.0),
            drag_samples: Vec::new(),
            last_click_time: None,
            last_click_pos: None,
            tile_fade_start: HashMap::new(),
            last_mouse_pos: None,
        }
    }

    /// Returns `true` if any animation is still running and requires continuous redraws.
    pub fn is_animating(&self, current_zoom: f64) -> bool {
        let zoom_active = (self.zoom_target - current_zoom).abs() > 0.001;
        let pan_active = {
            let (vx, vy) = self.pan_velocity;
            (vx * vx + vy * vy).sqrt() > INERTIA_MIN_SPEED
        };
        let fades_active = !self.tile_fade_start.is_empty();
        zoom_active || pan_active || fades_active
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
        // ── Smooth zoom ──
        let current = engine.viewport.zoom;
        let target = self
            .zoom_target
            .clamp(engine.camera.min_zoom, engine.camera.max_zoom);
        let diff = target - current;
        if diff.abs() > 0.001 {
            let new_zoom = exp_decay(current, target, ZOOM_ANIM_SPEED, dt);
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
        if speed > INERTIA_MIN_SPEED {
            engine.pan_for_mode(vx * dt, -(vy * dt), mode);
            let friction = (-INERTIA_FRICTION * dt).exp();
            self.pan_velocity = (vx * friction, vy * friction);
        } else {
            self.pan_velocity = (0.0, 0.0);
        }
    }

    /// Record a drag position sample for velocity estimation.
    pub fn record_drag(&mut self, pos: (f64, f64), time_secs: f64) {
        self.drag_samples
            .retain(|(_, t)| time_secs - t < DRAG_SAMPLE_WINDOW);
        self.drag_samples.push((pos, time_secs));
    }

    /// Compute pan velocity from recent drag samples (call on mouse-up/touch-end).
    pub fn compute_release_velocity(&mut self, time_secs: f64) {
        self.drag_samples
            .retain(|(_, t)| time_secs - t < DRAG_SAMPLE_WINDOW);
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
            .map(|t| time_secs - t < DOUBLE_CLICK_TIME)
            .unwrap_or(false)
            && self
                .last_click_pos
                .map(|(lx, ly)| {
                    ((pos.0 - lx).powi(2) + (pos.1 - ly).powi(2)).sqrt() < DOUBLE_CLICK_DIST
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
            .retain(|_, start| now_secs - *start < FADE_DURATION + 0.1);
    }
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
                // Search for an available parent at most 4 levels up.
                // Unbounded search can cause issues at high zoom with many
                // cached ancestor tiles, and at polar regions where tiles
                // change rapidly.
                let has_parent = {
                    let mut c = coord.parent();
                    let mut found = false;
                    let mut depth = 0u8;
                    while let Some(p) = c {
                        if depth >= 4 {
                            break;
                        }
                        if available.contains(&p) {
                            found = true;
                            break;
                        }
                        c = p.parent();
                        depth += 1;
                    }
                    found
                };
                if has_parent {
                    available_for_base.remove(&coord);
                    let fade_t = ((elapsed / FADE_DURATION) as f32).clamp(1.0 / 60.0, 1.0);
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
                let t = ((elapsed / FADE_DURATION) as f32).clamp(1.0 / 60.0, 1.0);
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
// TouchGestureState (with grace period)
// ═══════════════════════════════════════════════════════════════════

/// Multi-touch gesture state with grace period for finger transitions.
///
/// When the number of active touches changes (e.g. 1→2 or 2→1), a
/// grace period suppresses gesture actions to prevent jitter from the
/// momentary single-finger state during a two-finger gesture setup.
pub struct TouchGestureState {
    /// Active touch points: id → (x, y)
    touches: HashMap<i32, (f64, f64)>,
    /// Previous pinch distance
    prev_pinch_dist: Option<f64>,
    /// Previous pinch angle (degrees)
    prev_pinch_angle: Option<f64>,
    /// Previous midpoint
    prev_midpoint: Option<(f64, f64)>,
    /// Timestamp when touch count last changed
    count_change_time: f64,
    /// Touch count before last change (for detecting transitions)
    prev_count: usize,
}

impl TouchGestureState {
    pub fn new() -> Self {
        Self {
            touches: HashMap::new(),
            prev_pinch_dist: None,
            prev_pinch_angle: None,
            prev_midpoint: None,
            count_change_time: 0.0,
            prev_count: 0,
        }
    }

    /// Returns `true` if we're in a grace period after a touch count change.
    pub fn in_grace_period(&self, now_secs: f64) -> bool {
        now_secs - self.count_change_time < TOUCH_GRACE_PERIOD
    }

    /// Current number of active touches.
    pub fn touch_count(&self) -> usize {
        self.touches.len()
    }

    /// Register a new touch point. Call for each `touchstart` changed touch.
    pub fn touch_start(&mut self, id: i32, x: f64, y: f64, now_secs: f64) {
        let old_count = self.touches.len();
        self.touches.insert(id, (x, y));
        let new_count = self.touches.len();
        if new_count != old_count {
            self.count_change_time = now_secs;
            self.prev_count = old_count;
        }
        // Reset pinch state on count change
        self.prev_pinch_dist = self.pinch_distance();
        self.prev_pinch_angle = self.pinch_angle();
        self.prev_midpoint = self.midpoint();
    }

    /// Remove a touch point. Call for each `touchend`/`touchcancel` changed touch.
    pub fn touch_end(&mut self, id: i32, now_secs: f64) {
        let old_count = self.touches.len();
        self.touches.remove(&id);
        let new_count = self.touches.len();
        if new_count != old_count {
            self.count_change_time = now_secs;
            self.prev_count = old_count;
        }
        // Reset pinch state
        self.prev_pinch_dist = self.pinch_distance();
        self.prev_pinch_angle = self.pinch_angle();
        self.prev_midpoint = self.midpoint();
    }

    /// Process touch moves and return computed gesture actions.
    ///
    /// `changes` is a list of `(id, x, y)` for each changed touch.
    /// `dpr` is the device pixel ratio for scaling single-finger pan.
    /// `now_secs` is the current timestamp.
    ///
    /// Returns `None` if in grace period (suppress all gestures).
    pub fn process_moves(
        &mut self,
        changes: &[(i32, f64, f64)],
        dpr: f64,
        now_secs: f64,
    ) -> Option<GestureAction> {
        // Snapshot previous positions for single-touch pan
        let prev_positions: HashMap<i32, (f64, f64)> = self.touches.clone();

        // Update positions
        for &(id, x, y) in changes {
            if self.touches.contains_key(&id) {
                self.touches.insert(id, (x, y));
            }
        }

        // Suppress during grace period — but NOT for 0→1 transitions.
        // When a single finger first touches down there is no prior gesture to
        // conflict with, so skipping the delay makes pan start feel instant.
        if self.in_grace_period(now_secs) && self.prev_count > 0 {
            // Still update pinch state so we don't get a jump after grace ends
            self.prev_pinch_dist = self.pinch_distance();
            self.prev_pinch_angle = self.pinch_angle();
            self.prev_midpoint = self.midpoint();
            return None;
        }

        let count = self.touches.len();

        if count == 1 {
            // ── Single finger: pan ──
            // Only if we didn't just come from a multi-touch gesture
            if self.prev_count > 1 && now_secs - self.count_change_time < TOUCH_GRACE_PERIOD * 2.0
            {
                return None;
            }
            if let Some(&(id, x, y)) = changes.first() {
                if let Some(&(px, py)) = prev_positions.get(&id) {
                    let dx = (x - px) * dpr;
                    let dy = (y - py) * dpr;
                    return Some(GestureAction::Pan { dx, dy });
                }
            }
        } else if count == 2 {
            // ── Two fingers: pinch zoom + rotate + pitch ──
            let new_dist = self.pinch_distance();
            let new_angle = self.pinch_angle();
            let new_mid = self.midpoint();

            let mut action = MultiTouchAction::default();

            // Zoom from pinch
            if let (Some(prev_d), Some(new_d)) = (self.prev_pinch_dist, new_dist) {
                if prev_d > 1.0 {
                    let zoom_delta = (new_d / prev_d).log2();
                    if let Some((mx, my)) = new_mid {
                        action.zoom = Some((zoom_delta, mx * dpr, my * dpr));
                    }
                }
            }

            // Rotate from angle change
            if let (Some(prev_a), Some(new_a)) = (self.prev_pinch_angle, new_angle) {
                let mut delta = new_a - prev_a;
                if delta > 180.0 {
                    delta -= 360.0;
                }
                if delta < -180.0 {
                    delta += 360.0;
                }
                if delta.abs() < 30.0 {
                    action.rotate = Some(-delta);
                }
            }

            // Pitch from vertical midpoint drag
            if let (Some((_, prev_my)), Some((_, new_my))) = (self.prev_midpoint, new_mid) {
                let dy = new_my - prev_my;
                if dy.abs() > 0.5 {
                    action.pitch = Some(-dy * PITCH_SENSITIVITY);
                }
            }

            self.prev_pinch_dist = new_dist;
            self.prev_pinch_angle = new_angle;
            self.prev_midpoint = new_mid;

            return Some(GestureAction::MultiTouch(action));
        }

        None
    }

    fn pinch_distance(&self) -> Option<f64> {
        if self.touches.len() == 2 {
            let pts: Vec<_> = self.touches.values().collect();
            let dx = pts[1].0 - pts[0].0;
            let dy = pts[1].1 - pts[0].1;
            Some((dx * dx + dy * dy).sqrt())
        } else {
            None
        }
    }

    fn pinch_angle(&self) -> Option<f64> {
        if self.touches.len() == 2 {
            let pts: Vec<_> = self.touches.values().collect();
            let dx = pts[1].0 - pts[0].0;
            let dy = pts[1].1 - pts[0].1;
            Some(dy.atan2(dx).to_degrees())
        } else {
            None
        }
    }

    fn midpoint(&self) -> Option<(f64, f64)> {
        if self.touches.len() == 2 {
            let pts: Vec<_> = self.touches.values().collect();
            Some(((pts[0].0 + pts[1].0) / 2.0, (pts[0].1 + pts[1].1) / 2.0))
        } else {
            None
        }
    }
}

/// Action produced by a single-finger gesture.
#[derive(Debug)]
pub enum GestureAction {
    /// Single-finger pan (dx, dy in physical pixels).
    Pan { dx: f64, dy: f64 },
    /// Multi-finger gesture combining zoom, rotate, and pitch.
    MultiTouch(MultiTouchAction),
}

/// Combined two-finger gesture deltas.
#[derive(Debug, Default)]
pub struct MultiTouchAction {
    /// Pinch zoom: (delta_zoom_levels, center_x, center_y).
    pub zoom: Option<(f64, f64, f64)>,
    /// Rotation in degrees.
    pub rotate: Option<f64>,
    /// Pitch delta in degrees.
    pub pitch: Option<f64>,
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
        let elapsed = anim.tile_fade_elapsed(&coord, 1.5);
        assert!(elapsed.unwrap() > FADE_DURATION);

        // GC removes old entries
        anim.gc_fades(1.5);
        assert!(anim.tile_fade_elapsed(&coord, 1.5).is_none());
    }

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

    #[test]
    fn test_touch_grace_period() {
        let mut ts = TouchGestureState::new();
        ts.touch_start(0, 100.0, 100.0, 1.0);
        assert!(ts.in_grace_period(1.0)); // just changed
        assert!(ts.in_grace_period(1.05)); // still in grace
        assert!(!ts.in_grace_period(1.1)); // grace over
    }

    #[test]
    fn test_touch_grace_skipped_for_first_finger() {
        // 0→1 transition: grace period is skipped so pan starts instantly.
        let mut ts = TouchGestureState::new();
        ts.touch_start(0, 100.0, 100.0, 1.0);

        // Move during grace period for first finger → Pan (no suppression)
        let action = ts.process_moves(&[(0, 150.0, 100.0)], 1.0, 1.02);
        assert!(matches!(action, Some(GestureAction::Pan { .. })));

        // Move after grace period → still Pan
        let action = ts.process_moves(&[(0, 200.0, 100.0)], 1.0, 1.1);
        assert!(matches!(action, Some(GestureAction::Pan { .. })));
    }

    #[test]
    fn test_touch_grace_suppresses_multi_finger_transition() {
        // 1→2 transition: grace period IS active to prevent jitter.
        let mut ts = TouchGestureState::new();
        ts.touch_start(0, 100.0, 100.0, 0.0);
        // Wait past initial grace
        let _ = ts.process_moves(&[(0, 110.0, 100.0)], 1.0, 0.2);
        // Second finger arrives → new grace period (prev_count=1 > 0)
        ts.touch_start(1, 200.0, 100.0, 0.3);
        let action = ts.process_moves(&[(0, 115.0, 100.0), (1, 205.0, 100.0)], 1.0, 0.32);
        assert!(action.is_none(), "Should suppress during 1→2 grace period");
    }

    #[test]
    fn test_touch_two_finger_transition() {
        let mut ts = TouchGestureState::new();
        // First finger
        ts.touch_start(0, 100.0, 100.0, 1.0);
        // Wait past grace
        let _ = ts.process_moves(&[(0, 110.0, 100.0)], 1.0, 1.1);

        // Second finger arrives — triggers new grace period
        ts.touch_start(1, 200.0, 100.0, 1.15);
        assert!(ts.in_grace_period(1.15));

        // Two-finger move during grace → suppressed
        let action = ts.process_moves(&[(0, 115.0, 100.0), (1, 205.0, 100.0)], 1.0, 1.18);
        assert!(action.is_none());

        // Two-finger move after grace → MultiTouch
        let action = ts.process_moves(&[(0, 120.0, 100.0), (1, 210.0, 100.0)], 1.0, 1.25);
        assert!(matches!(action, Some(GestureAction::MultiTouch(_))));
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
        anim.register_tile_loaded(c2, 1.5);

        // At t=1.5, c1 has been loaded for 0.5s (past FADE_DURATION=0.3)
        anim.gc_fades(1.5);
        assert!(anim.tile_fade_elapsed(&c1, 1.5).is_none(), "c1 should be GC'd");
        assert!(anim.tile_fade_elapsed(&c2, 1.5).is_some(), "c2 should remain");
    }
}
