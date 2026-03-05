//! Input event handling: keyboard, mouse, cursor, and scroll wheel.
//!
//! Uses shared constants from `x_planets_core::interaction` and drives
//! `MapController` (the same controller used by the web platform).

use std::time::Instant;

use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, PhysicalKey};

use x_planets_core::interaction::{
    KEYBOARD_ROTATE, PAN_AMOUNT, PITCH_SENSITIVITY, ROTATE_SENSITIVITY, ZOOM_STEP,
};
use x_planets_tiles::TerrainEncoding;

use super::NativeApp;

impl NativeApp {
    pub(super) fn handle_keyboard_input(&mut self, event: winit::event::KeyEvent) {
        if event.state == ElementState::Pressed {
            if let Some(ctrl) = &mut self.controller {
                let mode = ctrl.rendering_mode();
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::ArrowLeft) => {
                        ctrl.engine.pan_for_mode(-PAN_AMOUNT, 0.0, mode);
                    }
                    PhysicalKey::Code(KeyCode::ArrowRight) => {
                        ctrl.engine.pan_for_mode(PAN_AMOUNT, 0.0, mode);
                    }
                    PhysicalKey::Code(KeyCode::ArrowUp) => {
                        ctrl.engine.pan_for_mode(0.0, -PAN_AMOUNT, mode);
                    }
                    PhysicalKey::Code(KeyCode::ArrowDown) => {
                        ctrl.engine.pan_for_mode(0.0, PAN_AMOUNT, mode);
                    }
                    PhysicalKey::Code(KeyCode::Equal)
                    | PhysicalKey::Code(KeyCode::NumpadAdd) => {
                        ctrl.anim.zoom_target += ZOOM_STEP;
                        ctrl.anim.zoom_anchor = None;
                    }
                    PhysicalKey::Code(KeyCode::Minus)
                    | PhysicalKey::Code(KeyCode::NumpadSubtract) => {
                        ctrl.anim.zoom_target -= ZOOM_STEP;
                        ctrl.anim.zoom_anchor = None;
                    }
                    PhysicalKey::Code(KeyCode::KeyQ) => {
                        ctrl.engine.rotate(-KEYBOARD_ROTATE);
                    }
                    PhysicalKey::Code(KeyCode::KeyE) => {
                        ctrl.engine.rotate(KEYBOARD_ROTATE);
                    }
                    PhysicalKey::Code(KeyCode::KeyP) => {
                        let name = ctrl.cycle_projection();
                        log::info!("Projection: {}", name);
                    }
                    PhysicalKey::Code(KeyCode::KeyT) => {
                        let url = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
                        let enabled = ctrl.toggle_terrain(url, TerrainEncoding::Terrarium);
                        log::info!("Terrain: {}", if enabled { "ON" } else { "OFF" });

                        if enabled {
                            // Add NativeLayerState for the new terrain layer
                            if let Some(terrain_name) = ctrl.terrain_layer_name() {
                                let terrain_url = ctrl.terrain_url().unwrap_or(url).to_string();
                                let imagery_layer = ctrl.terrain_imagery_name()
                                    .unwrap_or("base").to_string();
                                self.layer_states.push(crate::tile_source::NativeLayerState {
                                    name: terrain_name.to_string(),
                                    kind: x_planets_core::engine::LayerKind::Terrain {
                                        imagery_layer,
                                        encoding: TerrainEncoding::Terrarium,
                                    },
                                    tile_source: std::sync::Arc::new(
                                        crate::tile_source::NativeTileSource::new(terrain_url),
                                    ),
                                    tile_textures: x_planets_tiles::TileCache::new(256),
                                    tile_loader: x_planets_tiles::TileLoader::new(6),
                                    pending_coords: std::collections::HashSet::new(),
                                    terrain_data: x_planets_tiles::TileCache::new(256),
                                    failed_cooldowns: std::collections::HashMap::new(),
                                    min_zoom: 0,
                                    max_zoom: 15,
                                    tile_scale: 1.0,
                                    geographic: false,
                                    geo_heightmap_cache: std::collections::HashMap::new(),
                                    available_coords_cache: std::collections::HashSet::new(),
                                });
                            }
                        } else {
                            // Remove terrain layer states
                            self.layer_states.retain(|ls| {
                                !matches!(ls.kind, x_planets_core::engine::LayerKind::Terrain { .. })
                            });
                        }
                    }
                    PhysicalKey::Code(KeyCode::Home) => {
                        ctrl.engine.viewport.center =
                            x_planets_math::GeoCoord::new(0.0, 0.0);
                        ctrl.engine.viewport.zoom = 2.0;
                        ctrl.engine.viewport.pitch = 0.0;
                        ctrl.engine.viewport.bearing = 0.0;
                        ctrl.anim.zoom_target = 2.0;
                        ctrl.anim.pan_velocity = (0.0, 0.0);
                        ctrl.engine.request_redraw();
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
                    let now = Instant::now();
                    let now_secs = self.now_secs(now);
                    let current_pos = self.last_mouse_pos.unwrap_or((0.0, 0.0));

                    if let Some(ctrl) = &mut self.controller {
                        if ctrl.check_double_click(current_pos.0, current_pos.1, now_secs) {
                            ctrl.anim.zoom_target += 1.0;
                            ctrl.anim.zoom_anchor = Some(current_pos);
                        }
                        ctrl.begin_drag();
                    }
                } else {
                    // Mouse up: compute release velocity for inertia
                    let now_secs = self.now_secs(Instant::now());
                    if let Some(ctrl) = &mut self.controller {
                        ctrl.end_drag(now_secs);
                        if ctrl.anim.pan_velocity.0.abs() > 1.0
                            || ctrl.anim.pan_velocity.1.abs() > 1.0
                        {
                            self.window.as_ref().unwrap().request_redraw();
                        }
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
                if let Some(ctrl) = &mut self.controller {
                    let mode = ctrl.rendering_mode();
                    ctrl.engine.pan_for_mode(dx, -dy, mode);
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            // Record sample for inertia velocity estimation
            let now_secs = self.now_secs(Instant::now());
            if let Some(ctrl) = &mut self.controller {
                ctrl.record_drag(pos.0, pos.1, now_secs);
            }
        }

        // Right-drag: pitch (vertical) + rotate (horizontal)
        if self.right_mouse_pressed {
            if let Some(last) = self.last_right_pos {
                let dx = pos.0 - last.0;
                let dy = pos.1 - last.1;
                if let Some(ctrl) = &mut self.controller {
                    ctrl.engine.pitch(-dy * PITCH_SENSITIVITY);
                    ctrl.engine.rotate(dx * ROTATE_SENSITIVITY);
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            self.last_right_pos = Some(pos);
        }

        // Middle-drag: rotate
        if self.middle_mouse_pressed {
            if let Some(last_x) = self.last_rotate_x {
                let dx = pos.0 - last_x;
                if let Some(ctrl) = &mut self.controller {
                    ctrl.engine.rotate(dx * ROTATE_SENSITIVITY);
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
        if let Some(ctrl) = &mut self.controller {
            ctrl.anim.zoom_target += scroll_y;
            if let Some((mx, my)) = self.last_mouse_pos {
                ctrl.anim.zoom_anchor = Some((mx, my));
            } else {
                ctrl.anim.zoom_anchor = None;
            }
        }
        self.window.as_ref().unwrap().request_redraw();
    }
}
