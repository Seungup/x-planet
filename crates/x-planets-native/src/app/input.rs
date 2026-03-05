//! Input event handling: keyboard, mouse, cursor, and scroll wheel.
//!
//! Mirrors `x-planets-web/src/input.rs` exactly — uses `MapController`'s
//! high-level methods (`pan`, `rotate`, `pitch`, `zoom_at`) so that
//! projection-specific behavior is handled consistently across platforms.

use std::time::Instant;

use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, PhysicalKey};

use x_planets_core::interaction::{
    KEYBOARD_ROTATE, PAN_AMOUNT, PITCH_SENSITIVITY, ROTATE_SENSITIVITY, ZOOM_STEP,
};

use super::NativeApp;

impl NativeApp {
    pub(super) fn handle_keyboard_input(&mut self, event: winit::event::KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }

        if let Some(ctrl) = &mut self.controller {
            match event.physical_key {
                // ── Pan (projection-aware via MapController::pan) ──
                PhysicalKey::Code(KeyCode::ArrowLeft) => ctrl.pan(-PAN_AMOUNT, 0.0),
                PhysicalKey::Code(KeyCode::ArrowRight) => ctrl.pan(PAN_AMOUNT, 0.0),
                PhysicalKey::Code(KeyCode::ArrowUp) => ctrl.pan(0.0, -PAN_AMOUNT),
                PhysicalKey::Code(KeyCode::ArrowDown) => ctrl.pan(0.0, PAN_AMOUNT),

                // ── Zoom (animated) ──
                PhysicalKey::Code(KeyCode::Equal) | PhysicalKey::Code(KeyCode::NumpadAdd) => {
                    ctrl.anim.zoom_target += ZOOM_STEP;
                    ctrl.anim.zoom_anchor = None;
                }
                PhysicalKey::Code(KeyCode::Minus) | PhysicalKey::Code(KeyCode::NumpadSubtract) => {
                    ctrl.anim.zoom_target -= ZOOM_STEP;
                    ctrl.anim.zoom_anchor = None;
                }

                // ── Rotate ──
                PhysicalKey::Code(KeyCode::KeyQ) => ctrl.rotate(-KEYBOARD_ROTATE),
                PhysicalKey::Code(KeyCode::KeyE) => ctrl.rotate(KEYBOARD_ROTATE),

                // ── Projection cycle ──
                PhysicalKey::Code(KeyCode::KeyP) => {
                    let name = ctrl.cycle_projection();
                    log::info!("Projection: {}", name);
                }

                // ── Terrain toggle (same as web KeyT) ──
                PhysicalKey::Code(KeyCode::KeyT) => {
                    // Must drop ctrl borrow before calling self.toggle_terrain()
                }

                // ── Reset view (same as web Home) ──
                PhysicalKey::Code(KeyCode::Home) => {
                    ctrl.set_center(0.0, 0.0);
                    ctrl.set_zoom(2.0);
                    ctrl.engine.viewport.pitch = 0.0;
                    ctrl.engine.viewport.bearing = 0.0;
                    ctrl.anim.zoom_target = 2.0;
                    ctrl.anim.pan_velocity = (0.0, 0.0);
                    ctrl.engine.request_redraw();
                }

                _ => return,
            }
        }

        // Handle terrain toggle separately (needs &mut self, not &mut controller)
        if event.physical_key == PhysicalKey::Code(KeyCode::KeyT) {
            let enabled = self.toggle_terrain();
            log::info!("Terrain: {}", if enabled { "ON" } else { "OFF" });
        }

        self.window.as_ref().unwrap().request_redraw();
    }

    pub(super) fn handle_mouse_input(&mut self, state: ElementState, button: MouseButton) {
        let pressed = state == ElementState::Pressed;
        match button {
            MouseButton::Left => {
                if pressed {
                    let now_secs = self.now_secs(Instant::now());
                    let pos = self.last_mouse_pos.unwrap_or((0.0, 0.0));

                    if let Some(ctrl) = &mut self.controller {
                        // Double-click: smooth zoom in +1 level (same as web)
                        if ctrl.anim.check_double_click(pos, now_secs) {
                            ctrl.anim.zoom_target += 1.0;
                            ctrl.anim.zoom_anchor = Some(pos);
                        }
                        // Stop inertia when starting a new drag
                        ctrl.anim.begin_drag();
                    }
                } else {
                    // Mouse up: compute release velocity for inertia
                    let now_secs = self.now_secs(Instant::now());
                    if let Some(ctrl) = &mut self.controller {
                        ctrl.anim.compute_release_velocity(now_secs);
                    }
                    self.last_mouse_pos = None;
                }
                self.mouse_pressed = pressed;
            }
            MouseButton::Right => {
                self.right_mouse_pressed = pressed;
                if !pressed {
                    self.last_right_pos = None;
                }
            }
            MouseButton::Middle => {
                self.middle_mouse_pressed = pressed;
                if !pressed {
                    self.last_rotate_x = None;
                }
            }
            _ => {}
        }
    }

    pub(super) fn handle_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        let (x, y) = (position.x, position.y);
        let now_secs = self.now_secs(Instant::now());

        if let Some(ctrl) = &mut self.controller {
            // Track mouse position for zoom anchor fallback (same as web)
            ctrl.anim.last_mouse_pos = Some((x, y));

            // Left-drag: pan (projection-aware via MapController::pan)
            if self.mouse_pressed {
                if let Some((lx, ly)) = self.last_mouse_pos {
                    let dx = x - lx;
                    let dy = y - ly;
                    ctrl.pan(dx, -dy);
                }
                // Record drag position for inertia velocity estimation
                ctrl.anim.record_drag((x, y), now_secs);
            }

            // Right-drag: pitch (vertical) + rotate (horizontal)
            if self.right_mouse_pressed {
                if let Some((lx, ly)) = self.last_right_pos {
                    let dx = x - lx;
                    let dy = y - ly;
                    ctrl.pitch(-dy * PITCH_SENSITIVITY);
                    ctrl.rotate(dx * ROTATE_SENSITIVITY);
                }
                self.last_right_pos = Some((x, y));
            }

            // Middle-drag: rotate
            if self.middle_mouse_pressed {
                if let Some(last_x) = self.last_rotate_x {
                    let dx = x - last_x;
                    ctrl.rotate(dx * ROTATE_SENSITIVITY);
                }
                self.last_rotate_x = Some(x);
            }
        }

        self.last_mouse_pos = Some((x, y));

        if self.mouse_pressed || self.right_mouse_pressed || self.middle_mouse_pressed {
            self.window.as_ref().unwrap().request_redraw();
        }
    }

    pub(super) fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        let scroll_y = match delta {
            MouseScrollDelta::LineDelta(_, y) => y as f64 * 0.3,
            MouseScrollDelta::PixelDelta(p) => p.y * 0.003,
        };
        if let Some(ctrl) = &mut self.controller {
            ctrl.anim.zoom_target += scroll_y;
            ctrl.anim.zoom_anchor = self.last_mouse_pos;
        }
        self.window.as_ref().unwrap().request_redraw();
    }
}
