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
//! - Keyboard: Arrow keys (pan), +/- (zoom), Q/E (rotate), P (projection), Home (reset)
//!
//! Mobile touch (via shared TouchGestureState with grace period):
//! - Single finger: pan with inertia
//! - Double-tap: smooth zoom in +1 level
//! - Two fingers: pinch zoom, rotate, pitch (vertical drag)

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use x_planets_core::interaction::{
    GestureAction, TouchGestureState, KEYBOARD_ROTATE, PAN_AMOUNT, PITCH_SENSITIVITY,
    ROTATE_SENSITIVITY, ZOOM_STEP,
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
    /// Left-button mousedown position (CSS pixels) for click detection.
    left_down_pos: Option<(f64, f64)>,
}

impl MouseDragState {
    fn new() -> Self {
        Self {
            left: None,
            left_pressed: false,
            right: None,
            middle_x: None,
            left_down_pos: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Touch tap state (for double-tap zoom detection)
// ═══════════════════════════════════════════════════════════════════

struct TouchTapState {
    /// Position when single finger first touched down (CSS pixels).
    start_pos: Option<(f64, f64)>,
    /// Timestamp when single finger first touched down.
    start_time: Option<f64>,
}

impl TouchTapState {
    fn new() -> Self {
        Self {
            start_pos: None,
            start_time: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Event registration
// ═══════════════════════════════════════════════════════════════════

pub fn register_events(canvas: &web_sys::HtmlCanvasElement, app: Rc<RefCell<WebApp>>) {
    let touch_state = Rc::new(RefCell::new(TouchGestureState::new()));
    let mouse_state = Rc::new(RefCell::new(MouseDragState::new()));
    let tap_state = Rc::new(RefCell::new(TouchTapState::new()));

    register_mouse_events(canvas, Rc::clone(&app), Rc::clone(&mouse_state));
    register_wheel_event(canvas, Rc::clone(&app));
    register_keyboard_events(Rc::clone(&app));
    register_touch_events(
        canvas,
        Rc::clone(&app),
        Rc::clone(&touch_state),
        Rc::clone(&tap_state),
    );
}

// ═══════════════════════════════════════════════════════════════════
// Mouse events
// ═══════════════════════════════════════════════════════════════════

fn register_mouse_events(
    canvas: &web_sys::HtmlCanvasElement,
    app: Rc<RefCell<WebApp>>,
    mouse: Rc<RefCell<MouseDragState>>,
) {
    // Mouse offset_x/y returns CSS pixels, but viewport dimensions are
    // physical pixels (CSS × DPR).  Scale position-sensitive values by
    // DPR so pan deltas and zoom anchors match the viewport coordinate space.
    let dpr = web_sys::window().unwrap().device_pixel_ratio();

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

                    if app.controller.anim.check_double_click(pos, now) {
                        // Double-click: smooth zoom in +1 level at cursor
                        app.controller.anim.zoom_target += 1.0;
                        app.controller.anim.zoom_anchor = Some((pos.0 * dpr, pos.1 * dpr));
                    }

                    // Stop inertia when starting a new drag
                    app.controller.anim.begin_drag();
                    ms.left = Some(pos);
                    ms.left_pressed = true;
                    ms.left_down_pos = Some(pos);
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

            // Track mouse position for zoom anchor fallback (physical pixels)
            app.controller.anim.last_mouse_pos = Some((x * dpr, y * dpr));

            // Left-drag: pan (projection-aware via MapController::pan)
            if ms.left_pressed {
                if let Some((lx, ly)) = ms.left {
                    let dx = (x - lx) * dpr;
                    let dy = (y - ly) * dpr;
                    app.controller.pan(dx, -dy);
                }
                app.controller.anim.record_drag((x * dpr, y * dpr), now_secs());
                ms.left = Some((x, y));
            }

            // Right-drag: pitch + rotate (via MapController)
            if let Some((lx, ly)) = ms.right {
                let dx = x - lx;
                let dy = y - ly;
                app.controller.pitch(-dy * PITCH_SENSITIVITY);
                app.controller.rotate(dx * ROTATE_SENSITIVITY);
                ms.right = Some((x, y));
            }

            // Middle-drag: rotate (via MapController)
            if let Some(last_x) = ms.middle_x {
                let dx = x - last_x;
                app.controller.rotate(dx * ROTATE_SENSITIVITY);
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
                    // Click detection: if mousedown→mouseup distance < 5px, emit click event
                    if let Some(down_pos) = ms.left_down_pos.take() {
                        let x = e.offset_x() as f64;
                        let y = e.offset_y() as f64;
                        let dx = x - down_pos.0;
                        let dy = y - down_pos.1;
                        if dx * dx + dy * dy < 25.0 {
                            // Unproject screen coords (physical pixels) to geographic
                            let px = x * dpr;
                            let py = y * dpr;
                            if let Some((lat, lon)) = app.borrow().controller.unproject(px, py) {
                                app.borrow_mut().controller.push_click(lat, lon, x, y);
                            }
                        }
                    }
                    ms.left = None;
                    ms.left_pressed = false;
                    app.borrow_mut().controller.anim.compute_release_velocity(now_secs());
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
    let dpr = web_sys::window().unwrap().device_pixel_ratio();
    let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::WheelEvent| {
        e.prevent_default();
        let delta = -e.delta_y() / 300.0;
        let x = e.offset_x() as f64 * dpr;
        let y = e.offset_y() as f64 * dpr;
        let mut app = app.borrow_mut();
        app.controller.anim.zoom_target += delta;
        app.controller.anim.zoom_anchor = Some((x, y));
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
            // Pan (projection-aware via MapController::pan)
            "ArrowLeft" => app.controller.pan(-PAN_AMOUNT, 0.0),
            "ArrowRight" => app.controller.pan(PAN_AMOUNT, 0.0),
            "ArrowUp" => app.controller.pan(0.0, -PAN_AMOUNT),
            "ArrowDown" => app.controller.pan(0.0, PAN_AMOUNT),
            "Equal" | "NumpadAdd" => {
                app.controller.anim.zoom_target += ZOOM_STEP;
                app.controller.anim.zoom_anchor = None;
            }
            "Minus" | "NumpadSubtract" => {
                app.controller.anim.zoom_target -= ZOOM_STEP;
                app.controller.anim.zoom_anchor = None;
            }
            "KeyQ" => app.controller.rotate(-KEYBOARD_ROTATE),
            "KeyE" => app.controller.rotate(KEYBOARD_ROTATE),
            "KeyP" => {
                let name = app.cycle_projection();
                update_projection_button(&name);
            }
            "KeyT" => {
                let enabled = app.toggle_terrain();
                update_altitude_button(enabled);
            }
            "Home" => {
                app.controller.engine.viewport.center = x_planets_math::GeoCoord::new(0.0, 0.0);
                app.controller.engine.viewport.zoom = 2.0;
                app.controller.engine.viewport.pitch = 0.0;
                app.controller.engine.viewport.bearing = 0.0;
                app.controller.anim.zoom_target = 2.0;
                app.controller.anim.pan_velocity = (0.0, 0.0);
                app.controller.engine.request_redraw();
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
// Touch events (with inertia + double-tap zoom)
// ═══════════════════════════════════════════════════════════════════

/// Maximum duration (seconds) and distance (CSS px) for a touch to
/// count as a "tap" for double-tap detection.
const TAP_MAX_DURATION: f64 = 0.3;
const TAP_MAX_DISTANCE: f64 = 20.0;

fn register_touch_events(
    canvas: &web_sys::HtmlCanvasElement,
    app: Rc<RefCell<WebApp>>,
    touch_state: Rc<RefCell<TouchGestureState>>,
    tap_state: Rc<RefCell<TouchTapState>>,
) {
    let dpr = web_sys::window().unwrap().device_pixel_ratio();

    // touchstart (must be non-passive so preventDefault() works on mobile)
    {
        let ts = Rc::clone(&touch_state);
        let app = Rc::clone(&app);
        let tap = Rc::clone(&tap_state);
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

            if ts.touch_count() == 1 {
                // Single finger: stop any running inertia and begin new drag.
                app.borrow_mut().controller.anim.begin_drag();
                // Record tap start for double-tap detection.
                if let Some(t) = e.changed_touches().get(0) {
                    let mut tap = tap.borrow_mut();
                    tap.start_pos = Some((t.client_x() as f64, t.client_y() as f64));
                    tap.start_time = Some(now);
                }
            } else {
                // Multi-touch → not a tap.
                let mut tap = tap.borrow_mut();
                tap.start_pos = None;
                tap.start_time = None;
            }
        });
        add_non_passive_listener(canvas, "touchstart", cb.as_ref().unchecked_ref());
        cb.forget();
    }

    // touchmove (must be non-passive so preventDefault() works on mobile)
    {
        let ts = Rc::clone(&touch_state);
        let app = Rc::clone(&app);
        let tap = Rc::clone(&tap_state);
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
                        app.controller.pan(dx, -dy);
                        // Record drag position for inertia velocity estimation.
                        if let Some(&(_, x, y)) = changes.first() {
                            app.controller.anim.record_drag((x * dpr, y * dpr), now);
                        }
                    }
                    GestureAction::MultiTouch(mt) => {
                        if let Some((delta, cx, cy)) = mt.zoom {
                            app.controller.zoom_at(delta, cx, cy);
                        }
                        if let Some(deg) = mt.rotate {
                            app.controller.rotate(deg);
                        }
                        if let Some(deg) = mt.pitch {
                            app.controller.pitch(deg);
                        }
                        // Multi-touch gesture invalidates tap.
                        let mut tap = tap.borrow_mut();
                        tap.start_pos = None;
                        tap.start_time = None;
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
        let app = Rc::clone(&app);
        let tap = Rc::clone(&tap_state);
        let cb = Closure::<dyn FnMut(_)>::new(move |e: web_sys::TouchEvent| {
            e.prevent_default();
            let now = now_secs();
            let mut ts = ts.borrow_mut();
            let was_single = ts.touch_count() == 1;

            let touches = e.changed_touches();
            for i in 0..touches.length() {
                if let Some(t) = touches.get(i) {
                    let end_pos = (t.client_x() as f64, t.client_y() as f64);
                    ts.touch_end(t.identifier(), now);

                    // 1→0 transition: finalize single-finger gesture.
                    if was_single && ts.touch_count() == 0 {
                        let mut app = app.borrow_mut();
                        let mut tap = tap.borrow_mut();

                        // Check for double-tap zoom.
                        if let (Some(start_pos), Some(start_time)) =
                            (tap.start_pos, tap.start_time)
                        {
                            let duration = now - start_time;
                            let dist = ((end_pos.0 - start_pos.0).powi(2)
                                + (end_pos.1 - start_pos.1).powi(2))
                            .sqrt();
                            if duration < TAP_MAX_DURATION && dist < TAP_MAX_DISTANCE {
                                // Use CSS pixels for distance check (matches mouse),
                                // physical pixels only for the zoom anchor.
                                if app.controller.anim.check_double_click(start_pos, now) {
                                    app.controller.anim.zoom_target += 1.0;
                                    app.controller.anim.zoom_anchor =
                                        Some((start_pos.0 * dpr, start_pos.1 * dpr));
                                }
                            }
                        }

                        // Compute inertia velocity from drag samples.
                        app.controller.anim.compute_release_velocity(now);

                        tap.start_pos = None;
                        tap.start_time = None;
                    }
                }
            }
        });
        add_non_passive_listener(canvas, "touchend", cb.as_ref().unchecked_ref());
        add_non_passive_listener(canvas, "touchcancel", cb.as_ref().unchecked_ref());
        cb.forget();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Button text updates (buttons are created by TypeScript, updated here
// when keyboard shortcuts trigger projection/terrain changes)
// ═══════════════════════════════════════════════════════════════════

/// Update the projection button text to match the current projection.
fn update_projection_button(name: &str) {
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Some(btn) = doc.get_element_by_id("proj-btn") {
            btn.set_text_content(Some(name));
        }
    }
}

/// Update the altitude button text and style to match the current state.
fn update_altitude_button(enabled: bool) {
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Some(btn) = doc.get_element_by_id("alt-btn") {
            if enabled {
                btn.set_text_content(Some("Terrain ON"));
                let _ = btn.set_attribute("class", "xp-btn active");
            } else {
                btn.set_text_content(Some("Terrain OFF"));
                let _ = btn.set_attribute("class", "xp-btn");
            }
        }
    }
}
