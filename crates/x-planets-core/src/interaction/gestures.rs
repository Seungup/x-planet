//! Multi-touch gesture state with grace period for finger transitions.

use std::collections::HashMap;

use super::{PITCH_SENSITIVITY, TOUCH_GRACE_PERIOD};

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
}
