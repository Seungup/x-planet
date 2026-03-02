//! Input event handling: keyboard, mouse, cursor, and scroll wheel.

use std::time::Instant;

use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, PhysicalKey};

use super::NativeApp;

impl NativeApp {
    pub(super) fn handle_keyboard_input(&mut self, event: winit::event::KeyEvent) {
        if event.state == ElementState::Pressed {
            if let Some(engine) = &mut self.engine {
                let pan_amount = 50.0;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::ArrowLeft) => {
                        engine.pan(-pan_amount, 0.0);
                    }
                    PhysicalKey::Code(KeyCode::ArrowRight) => {
                        engine.pan(pan_amount, 0.0);
                    }
                    PhysicalKey::Code(KeyCode::ArrowUp) => {
                        engine.pan(0.0, -pan_amount);
                    }
                    PhysicalKey::Code(KeyCode::ArrowDown) => {
                        engine.pan(0.0, pan_amount);
                    }
                    // +/-: smooth animated zoom (target ±0.5)
                    PhysicalKey::Code(KeyCode::Equal)
                    | PhysicalKey::Code(KeyCode::NumpadAdd) => {
                        self.anim.zoom_target += 0.5;
                        self.anim.zoom_anchor = None;
                    }
                    PhysicalKey::Code(KeyCode::Minus)
                    | PhysicalKey::Code(KeyCode::NumpadSubtract) => {
                        self.anim.zoom_target -= 0.5;
                        self.anim.zoom_anchor = None;
                    }
                    // Q/E: rotate counter-clockwise / clockwise
                    PhysicalKey::Code(KeyCode::KeyQ) => {
                        engine.rotate(-10.0);
                    }
                    PhysicalKey::Code(KeyCode::KeyE) => {
                        engine.rotate(10.0);
                    }
                    PhysicalKey::Code(KeyCode::Home) => {
                        engine.viewport.center =
                            x_planets_math::GeoCoord::new(0.0, 0.0);
                        engine.viewport.zoom = 2.0;
                        engine.viewport.pitch = 0.0;
                        engine.viewport.bearing = 0.0;
                        self.anim.zoom_target = 2.0;
                        self.anim.pan_velocity = (0.0, 0.0);
                        engine.request_redraw();
                    }
                    _ => {}
                }
            }
            self.window.as_ref().unwrap().request_redraw();
        }
    }

    pub(super) fn handle_mouse_input(&mut self, state: ElementState, button: MouseButton) {
        let pressed = state == ElementState::Pressed;
        match button {
            MouseButton::Left => {
                if pressed {
                    // ── Double-click detection ──
                    let now = Instant::now();
                    let current_pos = self.last_mouse_pos.unwrap_or((0.0, 0.0));
                    let is_double_click = self
                        .anim
                        .last_click_time
                        .map(|t| now.duration_since(t).as_millis() < 300)
                        .unwrap_or(false)
                        && self
                            .anim
                            .last_click_pos
                            .map(|(lx, ly)| {
                                let (cx, cy) = current_pos;
                                ((cx - lx).powi(2) + (cy - ly).powi(2)).sqrt() < 10.0
                            })
                            .unwrap_or(false);

                    if is_double_click {
                        // Double-click: smooth zoom in +1 level
                        self.anim.zoom_target += 1.0;
                        self.anim.zoom_anchor = Some(current_pos);
                        self.anim.last_click_time = None; // prevent triple-click
                        self.window.as_ref().unwrap().request_redraw();
                    } else {
                        self.anim.last_click_time = Some(now);
                        self.anim.last_click_pos = Some(current_pos);
                    }

                    // Stop inertia when starting a new drag
                    self.anim.pan_velocity = (0.0, 0.0);
                    self.anim.last_drag_positions.clear();
                } else {
                    // Mouse up: compute release velocity for inertia
                    self.anim.compute_release_velocity();
                    if self.anim.pan_velocity.0.abs() > 1.0
                        || self.anim.pan_velocity.1.abs() > 1.0
                    {
                        self.window.as_ref().unwrap().request_redraw();
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
        let pos = (position.x, position.y);

        // Left-drag: pan
        if self.mouse_pressed {
            if let Some(last) = self.last_mouse_pos {
                let dx = pos.0 - last.0;
                let dy = pos.1 - last.1;
                if let Some(engine) = &mut self.engine {
                    engine.pan(dx, -dy);
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            // Record sample for inertia velocity estimation
            self.anim.record_drag(pos);
        }

        // Right-drag: pitch (vertical) + rotate (horizontal)
        if self.right_mouse_pressed {
            if let Some(last) = self.last_right_pos {
                let dx = pos.0 - last.0;
                let dy = pos.1 - last.1;
                if let Some(engine) = &mut self.engine {
                    engine.pitch(-dy * 0.3); // drag up = more tilt
                    engine.rotate(dx * 0.3); // drag right = clockwise
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            self.last_right_pos = Some(pos);
        }

        // Middle-drag: rotate (drag right = clockwise)
        if self.middle_mouse_pressed {
            if let Some(last_x) = self.last_rotate_x {
                let dx = pos.0 - last_x;
                if let Some(engine) = &mut self.engine {
                    engine.rotate(dx * 0.3);
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            self.last_rotate_x = Some(pos.0);
        }

        self.last_mouse_pos = Some(pos);
    }

    pub(super) fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        let scroll_y = match delta {
            MouseScrollDelta::LineDelta(_, y) => y as f64 * 0.3,
            MouseScrollDelta::PixelDelta(p) => p.y * 0.003,
        };
        // Accumulate into zoom target for smooth animation
        self.anim.zoom_target += scroll_y;
        // Set anchor to cursor position for zoom-toward-pointer
        if let Some((mx, my)) = self.last_mouse_pos {
            self.anim.zoom_anchor = Some((mx, my));
        } else {
            self.anim.zoom_anchor = None;
        }
        self.window.as_ref().unwrap().request_redraw();
    }
}
