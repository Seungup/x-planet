//! Touch and mouse input handling for the web map.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::app::WebApp;

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
// Event registration
// ═══════════════════════════════════════════════════════════════════

pub fn register_events(canvas: &web_sys::HtmlCanvasElement, app: Rc<RefCell<WebApp>>) {
    let touch_state = Rc::new(RefCell::new(TouchState::new()));
    let mouse_state: Rc<RefCell<Option<(f64, f64)>>> = Rc::new(RefCell::new(None));

    // ── Mouse events ──
    register_mouse_events(canvas, Rc::clone(&app), Rc::clone(&mouse_state));

    // ── Wheel event ──
    register_wheel_event(canvas, Rc::clone(&app));

    // ── Touch events ──
    register_touch_events(canvas, Rc::clone(&app), Rc::clone(&touch_state));
}

fn register_mouse_events(
    canvas: &web_sys::HtmlCanvasElement,
    app: Rc<RefCell<WebApp>>,
    mouse_state: Rc<RefCell<Option<(f64, f64)>>>,
) {
    // Right-click drag state: last position for pitch/rotate
    let right_mouse_state: Rc<RefCell<Option<(f64, f64)>>> = Rc::new(RefCell::new(None));

    // Disable context menu on canvas so right-click drag works
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
        let ms = Rc::clone(&mouse_state);
        let rms = Rc::clone(&right_mouse_state);
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
            let pos = (e.offset_x() as f64, e.offset_y() as f64);
            match e.button() {
                0 => *ms.borrow_mut() = Some(pos),  // Left button
                2 => *rms.borrow_mut() = Some(pos),  // Right button
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
        let ms = Rc::clone(&mouse_state);
        let rms = Rc::clone(&right_mouse_state);
        let app = Rc::clone(&app);
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
            let x = e.offset_x() as f64;
            let y = e.offset_y() as f64;

            // Left-drag: pan
            {
                let mut ms = ms.borrow_mut();
                if let Some((lx, ly)) = *ms {
                    let dx = x - lx;
                    let dy = y - ly;
                    app.borrow_mut().engine.pan(dx, -dy);
                    *ms = Some((x, y));
                }
            }

            // Right-drag: pitch (vertical) + rotate (horizontal)
            {
                let mut rms = rms.borrow_mut();
                if let Some((lx, ly)) = *rms {
                    let dx = x - lx;
                    let dy = y - ly;
                    let mut app = app.borrow_mut();
                    app.engine.pitch(-dy * 0.3);   // drag up = more tilt
                    app.engine.rotate(dx * 0.3);   // drag right = clockwise
                    *rms = Some((x, y));
                }
            }
        });
        canvas
            .add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }

    // mouseup
    {
        let ms = Rc::clone(&mouse_state);
        let rms = Rc::clone(&right_mouse_state);
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::MouseEvent| {
            match e.button() {
                0 => *ms.borrow_mut() = None,
                2 => *rms.borrow_mut() = None,
                _ => {}
            }
        });
        canvas
            .add_event_listener_with_callback("mouseup", cb.as_ref().unchecked_ref())
            .unwrap();
        cb.forget();
    }
}

fn register_wheel_event(canvas: &web_sys::HtmlCanvasElement, app: Rc<RefCell<WebApp>>) {
    let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::WheelEvent| {
        e.prevent_default();
        let delta = -e.delta_y() / 300.0;
        let x = e.offset_x() as f64;
        let y = e.offset_y() as f64;
        app.borrow_mut().engine.zoom_at(delta, x, y);
    });
    canvas
        .add_event_listener_with_callback("wheel", cb.as_ref().unchecked_ref())
        .unwrap();
    cb.forget();
}

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
                // Negate: screen-clockwise finger rotation → positive atan2 delta,
                // but we want map to rotate clockwise (bearing decrease visually).
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
                // Drag up (negative dy in screen coords) → increase pitch (tilt more)
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
