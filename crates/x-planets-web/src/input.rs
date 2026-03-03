//! Touch and mouse input handling for the web map.
//!
//! Desktop mouse features (matching native):
//! - Left-drag: pan with inertia
//! - Right-drag: pitch (vertical) + rotate (horizontal)
//! - Middle-drag: rotate
//! - Scroll wheel: smooth animated zoom toward cursor
//! - Double-click: smooth zoom in +1 level
//! - Keyboard: Arrow keys (pan), +/- (zoom), Q/E (rotate), Home (reset)
//!
//! Mobile touch:
//! - Single finger: pan
//! - Two fingers: pinch zoom, rotate, pitch (vertical drag)

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::app::WebApp;

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn now_ms() -> f64 {
    web_sys::window()
        .unwrap()
        .performance()
        .unwrap()
        .now()
}

// ═══════════════════════════════════════════════════════════════════
// Touch state for gesture detection
// ═══════════════════════════════════════════════════════════════════

struct TouchState {
    /// Active touch points: id → (x, y)
    touches: HashMap<i32, (f64, f64)>,
    /// Previous pinch distance (for zoom)
    prev_pinch_dist: Option<f64>,
    /// Previous pinch angle (for rotation)
    prev_pinch_angle: Option<f64>,
    /// Previous midpoint (for two-finger pitch)
    prev_midpoint: Option<(f64, f64)>,
}

impl TouchState {
    fn new() -> Self {
        Self {
            touches: HashMap::new(),
            prev_pinch_dist: None,
            prev_pinch_angle: None,
            prev_midpoint: None,
        }
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

// ═══════════════════════════════════════════════════════════════════
// Mouse drag state
// ═══════════════════════════════════════════════════════════════════

struct MouseDragState {
    /// Left-button drag: last position
    left: Option<(f64, f64)>,
    /// Left button currently pressed
    left_pressed: bool,
    /// Right-button drag: last position
    right: Option<(f64, f64)>,
    /// Middle-button drag: last X position (rotate only)
    middle_x: Option<f64>,
}

impl MouseDragState {
    fn new() -> Self {
        Self {
            left: None,
            left_pressed: false,
            right: None,
            middle_x: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Event registration
// ═══════════════════════════════════════════════════════════════════

pub fn register_events(canvas: &web_sys::HtmlCanvasElement, app: Rc<RefCell<WebApp>>) {
    let touch_state = Rc::new(RefCell::new(TouchState::new()));
    let mouse_state = Rc::new(RefCell::new(MouseDragState::new()));

    // ── Mouse events ──
    register_mouse_events(canvas, Rc::clone(&app), Rc::clone(&mouse_state));

    // ── Wheel event ──
    register_wheel_event(canvas, Rc::clone(&app));

    // ── Keyboard events ──
    register_keyboard_events(Rc::clone(&app));

    // ── Touch events ──
    register_touch_events(canvas, Rc::clone(&app), Rc::clone(&touch_state));
}

// ═══════════════════════════════════════════════════════════════════
// Mouse events: left-drag (pan+inertia), right-drag (pitch+rotate),
//               middle-drag (rotate), double-click (zoom)
// ═══════════════════════════════════════════════════════════════════

fn register_mouse_events(
    canvas: &web_sys::HtmlCanvasElement,
    app: Rc<RefCell<WebApp>>,
    mouse: Rc<RefCell<MouseDragState>>,
) {
    // Disable context menu so right-click drag works
    {
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
            e.prevent_default();
        });
        canvas
            .add_event_listener_with_callback("contextmenu", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // mousedown
    {
        let ms = Rc::clone(&mouse);
        let app = Rc::clone(&app);
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
            let pos = (e.offset_x() as f64, e.offset_y() as f64);
            let mut ms = ms.borrow_mut();
            match e.button() {
                0 => {
                    // ── Double-click detection ──
                    let now = now_ms();
                    let mut app = app.borrow_mut();
                    let is_double_click = app
                        .last_click_time_ms
                        .map(|t| now - t < 300.0)
                        .unwrap_or(false)
                        && app
                            .last_click_pos
                            .map(|(lx, ly)| {
                                ((pos.0 - lx).powi(2) + (pos.1 - ly).powi(2)).sqrt() < 10.0
                            })
                            .unwrap_or(false);

                    if is_double_click {
                        // Double-click: smooth zoom in +1 level at cursor
                        app.zoom_target += 1.0;
                        app.zoom_anchor = Some(pos);
                        app.last_click_time_ms = None; // prevent triple-click
                    } else {
                        app.last_click_time_ms = Some(now);
                        app.last_click_pos = Some(pos);
                    }

                    // Stop inertia when starting a new drag
                    app.pan_velocity = (0.0, 0.0);
                    app.drag_samples.clear();

                    ms.left = Some(pos);
                    ms.left_pressed = true;
                }
                1 => {
                    ms.middle_x = Some(pos.0);
                }
                2 => {
                    ms.right = Some(pos);
                }
                _ => {}
            }
        });
        canvas
            .add_event_listener_with_callback("mousedown", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // mousemove
    {
        let ms = Rc::clone(&mouse);
        let app = Rc::clone(&app);
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
            let x = e.offset_x() as f64;
            let y = e.offset_y() as f64;
            let mut ms = ms.borrow_mut();
            let mut app = app.borrow_mut();

            // Track mouse position for zoom anchor fallback
            app.last_mouse_pos = Some((x, y));

            // Left-drag: pan
            if ms.left_pressed {
                if let Some((lx, ly)) = ms.left {
                    let dx = x - lx;
                    let dy = y - ly;
                    app.engine.pan(dx, -dy);
                }
                // Record sample for inertia velocity estimation
                let now = now_ms();
                app.record_drag((x, y), now);
                ms.left = Some((x, y));
            }

            // Right-drag: pitch (vertical) + rotate (horizontal)
            if let Some((lx, ly)) = ms.right {
                let dx = x - lx;
                let dy = y - ly;
                app.engine.pitch(-dy * 0.3); // drag up = more tilt
                app.engine.rotate(dx * 0.3); // drag right = clockwise
                ms.right = Some((x, y));
            }

            // Middle-drag: rotate
            if let Some(last_x) = ms.middle_x {
                let dx = x - last_x;
                app.engine.rotate(dx * 0.3);
                ms.middle_x = Some(x);
            }
        });
        canvas
            .add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // mouseup
    {
        let ms = Rc::clone(&mouse);
        let app = Rc::clone(&app);
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
            let mut ms = ms.borrow_mut();
            match e.button() {
                0 => {
                    ms.left = None;
                    ms.left_pressed = false;
                    // Compute release velocity for inertia
                    let now = now_ms();
                    app.borrow_mut().compute_release_velocity(now);
                }
                1 => {
                    ms.middle_x = None;
                }
                2 => {
                    ms.right = None;
                }
                _ => {}
            }
        });
        canvas
            .add_event_listener_with_callback("mouseup", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Wheel: smooth animated zoom toward cursor
// ═══════════════════════════════════════════════════════════════════

fn register_wheel_event(canvas: &web_sys::HtmlCanvasElement, app: Rc<RefCell<WebApp>>) {
    let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::WheelEvent| {
        e.prevent_default();
        let delta = -e.delta_y() / 300.0;
        let x = e.offset_x() as f64;
        let y = e.offset_y() as f64;
        let mut app = app.borrow_mut();
        // Accumulate into zoom target for smooth animation
        app.zoom_target += delta;
        // Set anchor to cursor position for zoom-toward-pointer
        app.zoom_anchor = Some((x, y));
    });
    canvas
        .add_event_listener_with_callback("wheel", cb.as_ref().unchecked_ref())
        .unwrap();
    cb.forget();
}

// ═══════════════════════════════════════════════════════════════════
// Keyboard: Arrow keys (pan), +/- (zoom), Q/E (rotate), Home (reset)
// ═══════════════════════════════════════════════════════════════════

fn register_keyboard_events(app: Rc<RefCell<WebApp>>) {
    let window = web_sys::window().unwrap();
    let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::KeyboardEvent| {
        let key = e.code();
        let mut app = app.borrow_mut();
        let pan_amount = 50.0;

        match key.as_str() {
            "ArrowLeft" => {
                app.engine.pan(-pan_amount, 0.0);
            }
            "ArrowRight" => {
                app.engine.pan(pan_amount, 0.0);
            }
            "ArrowUp" => {
                app.engine.pan(0.0, -pan_amount);
            }
            "ArrowDown" => {
                app.engine.pan(0.0, pan_amount);
            }
            "Equal" | "NumpadAdd" => {
                app.zoom_target += 0.5;
                app.zoom_anchor = None;
            }
            "Minus" | "NumpadSubtract" => {
                app.zoom_target -= 0.5;
                app.zoom_anchor = None;
            }
            "KeyQ" => {
                app.engine.rotate(-10.0);
            }
            "KeyE" => {
                app.engine.rotate(10.0);
            }
            "Home" => {
                app.engine.viewport.center = x_planets_math::GeoCoord::new(0.0, 0.0);
                app.engine.viewport.zoom = 2.0;
                app.engine.viewport.pitch = 0.0;
                app.engine.viewport.bearing = 0.0;
                app.zoom_target = 2.0;
                app.pan_velocity = (0.0, 0.0);
                app.engine.request_redraw();
            }
            _ => return, // Don't prevent default for unhandled keys
        }
        e.prevent_default();
    });
    window
        .add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref())
        .unwrap();
    cb.forget();
}

// ═══════════════════════════════════════════════════════════════════
// Touch events
// ═══════════════════════════════════════════════════════════════════

fn register_touch_events(
    canvas: &web_sys::HtmlCanvasElement,
    app: Rc<RefCell<WebApp>>,
    touch_state: Rc<RefCell<TouchState>>,
) {
    let dpr = web_sys::window().unwrap().device_pixel_ratio();

    // touchstart
    {
        let ts = Rc::clone(&touch_state);
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::TouchEvent| {
            e.prevent_default();
            let mut ts = ts.borrow_mut();
            let touches = e.changed_touches();
            for i in 0..touches.length() {
                if let Some(t) = touches.get(i) {
                    ts.touches.insert(
                        t.identifier(),
                        (t.client_x() as f64, t.client_y() as f64),
                    );
                }
            }
            // Reset pinch state when touch count changes
            ts.prev_pinch_dist = ts.pinch_distance();
            ts.prev_pinch_angle = ts.pinch_angle();
            ts.prev_midpoint = ts.midpoint();
        });
        canvas
            .add_event_listener_with_callback("touchstart", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // touchmove
    {
        let ts = Rc::clone(&touch_state);
        let app = Rc::clone(&app);
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::TouchEvent| {
            e.prevent_default();
            let mut ts = ts.borrow_mut();

            // Snapshot previous positions for single-touch pan
            let prev_positions: HashMap<i32, (f64, f64)> = ts.touches.clone();

            // Update all touch positions
            let touches = e.changed_touches();
            for i in 0..touches.length() {
                if let Some(t) = touches.get(i) {
                    ts.touches.insert(
                        t.identifier(),
                        (t.client_x() as f64, t.client_y() as f64),
                    );
                }
            }

            let mut app = app.borrow_mut();

            if ts.touches.len() == 1 {
                // ── Single finger: pan ──
                let changed = e.changed_touches();
                if let Some(t) = changed.get(0) {
                    let id = t.identifier();
                    if let Some(&(px, py)) = prev_positions.get(&id) {
                        let x = t.client_x() as f64;
                        let y = t.client_y() as f64;
                        let dx = (x - px) * dpr;
                        let dy = (y - py) * dpr;
                        app.engine.pan(dx, -dy);
                    }
                }
            } else if ts.touches.len() == 2 {
                // ── Two fingers: pinch zoom + rotate ──
                let new_dist = ts.pinch_distance();
                let new_angle = ts.pinch_angle();
                let midpoint = ts.midpoint();

                // Zoom from pinch
                if let (Some(prev_d), Some(new_d)) = (ts.prev_pinch_dist, new_dist) {
                    if prev_d > 1.0 {
                        let zoom_delta = (new_d / prev_d).log2();
                        if let Some((mx, my)) = midpoint {
                            app.engine.zoom_at(zoom_delta, mx * dpr, my * dpr);
                        }
                    }
                }

                // Rotate from angle change
                if let (Some(prev_a), Some(new_a)) = (ts.prev_pinch_angle, new_angle) {
                    let mut delta_angle = new_a - prev_a;
                    // Normalize to [-180, 180]
                    if delta_angle > 180.0 {
                        delta_angle -= 360.0;
                    }
                    if delta_angle < -180.0 {
                        delta_angle += 360.0;
                    }
                    if delta_angle.abs() < 30.0 {
                        app.engine.rotate(-delta_angle);
                    }
                }

                // Pitch from two-finger vertical drag
                if let (Some((_, prev_my)), Some((_, new_my))) =
                    (ts.prev_midpoint, midpoint)
                {
                    let dy = new_my - prev_my;
                    if dy.abs() > 0.5 {
                        app.engine.pitch(-dy * 0.3);
                    }
                }

                ts.prev_pinch_dist = new_dist;
                ts.prev_pinch_angle = new_angle;
                ts.prev_midpoint = midpoint;
            }
        });
        canvas
            .add_event_listener_with_callback("touchmove", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // touchend / touchcancel
    {
        let ts = Rc::clone(&touch_state);
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::TouchEvent| {
            e.prevent_default();
            let mut ts = ts.borrow_mut();
            let touches = e.changed_touches();
            for i in 0..touches.length() {
                if let Some(t) = touches.get(i) {
                    ts.touches.remove(&t.identifier());
                }
            }
            // Reset pinch state
            ts.prev_pinch_dist = ts.pinch_distance();
            ts.prev_pinch_angle = ts.pinch_angle();
            ts.prev_midpoint = ts.midpoint();
        });
        canvas
            .add_event_listener_with_callback("touchend", cb.as_ref().unchecked_ref())
            .unwrap();
        canvas
            .add_event_listener_with_callback("touchcancel", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }
}
