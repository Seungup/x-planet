//! Touch and mouse input handling for the web map.
//!
//! Uses shared `AnimationController` and `TouchGestureState` from
//! `x_planets_core::interaction` to avoid duplicating logic with native.
//!
//! Desktop mouse:
//! - Left-drag: pan with inertia
//! - Right-drag: pitch (vertical) + rotate (horizontal)
//! - Middle-drag: rotate
//! - Scroll wheel: smooth animated zoom toward cursor
//! - Double-click: smooth zoom in +1 level
//! - Keyboard: Arrow keys (pan), +/- (zoom), Q/E (rotate), Home (reset)
//!
//! Mobile touch (via shared TouchGestureState with grace period):
//! - Single finger: pan
//! - Two fingers: pinch zoom, rotate, pitch (vertical drag)

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use x_planets_core::interaction::{
    GestureAction, TouchGestureState,
    PAN_AMOUNT, ZOOM_STEP, KEYBOARD_ROTATE, PITCH_SENSITIVITY, ROTATE_SENSITIVITY,
};

use crate::app::WebApp;

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn now_secs() -> f64 {
    web_sys::window()
        .unwrap()
        .performance()
        .unwrap()
        .now()
        / 1000.0
}

/// Register an event listener with `{ passive: false }` so that
/// `preventDefault()` actually works. Mobile browsers default touch
/// and wheel listeners to passive, silently ignoring `preventDefault()`.
fn add_non_passive_listener(
    target: &web_sys::EventTarget,
    event_type: &str,
    cb: &js_sys::Function,
) {
    let opts = web_sys::AddEventListenerOptions::new();
    opts.set_passive(false);
    target
        .add_event_listener_with_callback_and_add_event_listener_options(event_type, cb, &opts)
        .unwrap();
}

// ═══════════════════════════════════════════════════════════════════
// Mouse drag state (platform-specific: tracks which buttons are held)
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
    let touch_state = Rc::new(RefCell::new(TouchGestureState::new()));
    let mouse_state = Rc::new(RefCell::new(MouseDragState::new()));

    register_mouse_events(canvas, Rc::clone(&app), Rc::clone(&mouse_state));
    register_wheel_event(canvas, Rc::clone(&app));
    register_keyboard_events(Rc::clone(&app));
    register_touch_events(canvas, Rc::clone(&app), Rc::clone(&touch_state));
}

// ═══════════════════════════════════════════════════════════════════
// Mouse events
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
                    let now = now_secs();
                    let mut app = app.borrow_mut();

                    if app.anim.check_double_click(pos, now) {
                        // Double-click: smooth zoom in +1 level at cursor
                        app.anim.zoom_target += 1.0;
                        app.anim.zoom_anchor = Some(pos);
                    }

                    // Stop inertia when starting a new drag
                    app.anim.begin_drag();
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
            app.anim.last_mouse_pos = Some((x, y));

            // Left-drag: pan
            if ms.left_pressed {
                if let Some((lx, ly)) = ms.left {
                    let dx = x - lx;
                    let dy = y - ly;
                    app.engine.pan(dx, -dy);
                }
                app.anim.record_drag((x, y), now_secs());
                ms.left = Some((x, y));
            }

            // Right-drag: pitch + rotate
            if let Some((lx, ly)) = ms.right {
                let dx = x - lx;
                let dy = y - ly;
                app.engine.pitch(-dy * PITCH_SENSITIVITY);
                app.engine.rotate(dx * ROTATE_SENSITIVITY);
                ms.right = Some((x, y));
            }

            // Middle-drag: rotate
            if let Some(last_x) = ms.middle_x {
                let dx = x - last_x;
                app.engine.rotate(dx * ROTATE_SENSITIVITY);
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
                    app.borrow_mut().anim.compute_release_velocity(now_secs());
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
        app.anim.zoom_target += delta;
        app.anim.zoom_anchor = Some((x, y));
    });
    // Must be non-passive so preventDefault() stops browser scroll/zoom
    add_non_passive_listener(canvas, "wheel", cb.as_ref().unchecked_ref());
    cb.forget();
}

// ═══════════════════════════════════════════════════════════════════
// Keyboard
// ═══════════════════════════════════════════════════════════════════

fn register_keyboard_events(app: Rc<RefCell<WebApp>>) {
    let window = web_sys::window().unwrap();
    let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::KeyboardEvent| {
        let key = e.code();
        let mut app = app.borrow_mut();

        match key.as_str() {
            "ArrowLeft" => app.engine.pan(-PAN_AMOUNT, 0.0),
            "ArrowRight" => app.engine.pan(PAN_AMOUNT, 0.0),
            "ArrowUp" => app.engine.pan(0.0, -PAN_AMOUNT),
            "ArrowDown" => app.engine.pan(0.0, PAN_AMOUNT),
            "Equal" | "NumpadAdd" => {
                app.anim.zoom_target += ZOOM_STEP;
                app.anim.zoom_anchor = None;
            }
            "Minus" | "NumpadSubtract" => {
                app.anim.zoom_target -= ZOOM_STEP;
                app.anim.zoom_anchor = None;
            }
            "KeyQ" => app.engine.rotate(-KEYBOARD_ROTATE),
            "KeyE" => app.engine.rotate(KEYBOARD_ROTATE),
            "Home" => {
                app.engine.viewport.center = x_planets_math::GeoCoord::new(0.0, 0.0);
                app.engine.viewport.zoom = 2.0;
                app.engine.viewport.pitch = 0.0;
                app.engine.viewport.bearing = 0.0;
                app.anim.zoom_target = 2.0;
                app.anim.pan_velocity = (0.0, 0.0);
                app.engine.request_redraw();
            }
            _ => return,
        }
        e.prevent_default();
    });
    window
        .add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref())
        .unwrap();
    cb.forget();
}

// ═══════════════════════════════════════════════════════════════════
// Touch events (using shared TouchGestureState with grace period)
// ═══════════════════════════════════════════════════════════════════

fn register_touch_events(
    canvas: &web_sys::HtmlCanvasElement,
    app: Rc<RefCell<WebApp>>,
    touch_state: Rc<RefCell<TouchGestureState>>,
) {
    let dpr = web_sys::window().unwrap().device_pixel_ratio();

    // touchstart (must be non-passive so preventDefault() works on mobile)
    {
        let ts = Rc::clone(&touch_state);
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::TouchEvent| {
            e.prevent_default();
            let now = now_secs();
            let mut ts = ts.borrow_mut();
            let touches = e.changed_touches();
            for i in 0..touches.length() {
                if let Some(t) = touches.get(i) {
                    ts.touch_start(
                        t.identifier(),
                        t.client_x() as f64,
                        t.client_y() as f64,
                        now,
                    );
                }
            }
        });
        add_non_passive_listener(canvas, "touchstart", cb.as_ref().unchecked_ref());
        cb.forget();
    }

    // touchmove (must be non-passive so preventDefault() works on mobile)
    {
        let ts = Rc::clone(&touch_state);
        let app = Rc::clone(&app);
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::TouchEvent| {
            e.prevent_default();
            let now = now_secs();

            // Collect changed touches
            let changed = e.changed_touches();
            let mut changes = Vec::with_capacity(changed.length() as usize);
            for i in 0..changed.length() {
                if let Some(t) = changed.get(i) {
                    changes.push((t.identifier(), t.client_x() as f64, t.client_y() as f64));
                }
            }

            let mut ts = ts.borrow_mut();
            if let Some(action) = ts.process_moves(&changes, dpr, now) {
                let mut app = app.borrow_mut();
                match action {
                    GestureAction::Pan { dx, dy } => {
                        app.engine.pan(dx, -dy);
                    }
                    GestureAction::MultiTouch(mt) => {
                        if let Some((delta, cx, cy)) = mt.zoom {
                            app.engine.zoom_at(delta, cx, cy);
                        }
                        if let Some(deg) = mt.rotate {
                            app.engine.rotate(deg);
                        }
                        if let Some(deg) = mt.pitch {
                            app.engine.pitch(deg);
                        }
                    }
                }
            }
        });
        add_non_passive_listener(canvas, "touchmove", cb.as_ref().unchecked_ref());
        cb.forget();
    }

    // touchend / touchcancel (must be non-passive so preventDefault() works on mobile)
    {
        let ts = Rc::clone(&touch_state);
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::TouchEvent| {
            e.prevent_default();
            let now = now_secs();
            let mut ts = ts.borrow_mut();
            let touches = e.changed_touches();
            for i in 0..touches.length() {
                if let Some(t) = touches.get(i) {
                    ts.touch_end(t.identifier(), now);
                }
            }
        });
        add_non_passive_listener(canvas, "touchend", cb.as_ref().unchecked_ref());
        add_non_passive_listener(canvas, "touchcancel", cb.as_ref().unchecked_ref());
        cb.forget();
    }
}
