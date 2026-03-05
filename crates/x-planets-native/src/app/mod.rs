//! Native application event loop (winit + wgpu).
//!
//! The `NativeApp` struct and `ApplicationHandler` implementation that drives
//! rendering.  Sub-modules split the logic into focused responsibilities.
//!
//! Uses `MapController` from `x_planets_core` — the same controller used by the
//! web platform — for unified animation, camera, layer management, and
//! render-data assembly.

mod init;
mod input;
mod tile_loading;
mod tile_upload;
mod tiles3d_frame;

use std::collections::HashSet;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::EventLoop,
    window::{Window, WindowAttributes},
};
use x_planets_core::engine::MapConfig;
use x_planets_core::map_controller::LayerStateView;
use x_planets_core::{MapController, Model3dRenderer, TerrainRenderer, TileRenderer};
use x_planets_gpu::{GpuContext, TextureManager};
use x_planets_math::TileCoord;

use crate::tile_source::{LayerTileResult, NativeLayerState};
use crate::tiles3d_native::{Tiles3dLayerState, Tiles3dMessage};

/// Run the native application with an event loop.
///
/// Controls:
///   Arrow keys / left-drag        — pan (with inertia on release)
///   +/- / scroll wheel            — smooth zoom
///   Double-click                   — smooth zoom in (+1 level)
///   Right-drag (up/down)          — pitch (tilt)
///   Middle-drag (left/right)      — rotate (bearing)
///   Q / E keys                    — rotate left / right
///   T key                         — toggle terrain
///   Home                          — reset view
pub fn run_native(config: MapConfig) -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    log::info!("Starting x-planets native viewer...");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");

    let (tile_tx, tile_rx) = mpsc::channel();
    let (tiles3d_tx, tiles3d_rx) = mpsc::channel();

    let event_loop = EventLoop::new()?;
    let mut app = NativeApp {
        config,
        window: None,
        gpu: None,
        controller: None,
        renderer: None,
        terrain_renderer: None,
        model3d_renderer: None,
        tex_manager: None,
        // Per-layer GPU state (created in `resumed`)
        layer_states: Vec::new(),
        // 3D Tiles state
        tiles3d_states: Vec::new(),
        tiles3d_tx,
        tiles3d_rx,
        // Shared async tile channel
        rt,
        tile_tx,
        tile_rx,
        // Frame timing
        start_time: Instant::now(),
        last_frame_time: None,
        frame_count: 0,
        fps_update_time: None,
        // Mouse state
        mouse_pressed: false,
        last_mouse_pos: None,
        right_mouse_pressed: false,
        last_right_pos: None,
        middle_mouse_pressed: false,
        last_rotate_x: None,
    };
    event_loop.run_app(&mut app)?;

    Ok(())
}

pub(crate) struct NativeApp {
    config: MapConfig,
    pub(super) window: Option<Arc<Window>>,
    pub(super) gpu: Option<GpuContext>,
    /// Unified map controller (same as web platform).
    pub(super) controller: Option<MapController>,
    pub(super) renderer: Option<TileRenderer>,
    pub(super) terrain_renderer: Option<TerrainRenderer>,
    pub(super) model3d_renderer: Option<Model3dRenderer>,
    pub(super) tex_manager: Option<TextureManager>,
    /// Per-layer tile source, texture cache, loader, pending set.
    pub(super) layer_states: Vec<NativeLayerState>,
    // ── 3D Tiles state ──
    pub(super) tiles3d_states: Vec<Tiles3dLayerState>,
    pub(super) tiles3d_tx: mpsc::Sender<Tiles3dMessage>,
    pub(super) tiles3d_rx: mpsc::Receiver<Tiles3dMessage>,
    // Shared async tile channel (results tagged with layer name)
    pub(super) rt: tokio::runtime::Runtime,
    pub(super) tile_tx: mpsc::Sender<LayerTileResult>,
    tile_rx: mpsc::Receiver<LayerTileResult>,
    // ── Timing ──
    /// Reference epoch for converting Instant → f64 seconds.
    pub(super) start_time: Instant,
    last_frame_time: Option<Instant>,
    frame_count: u32,
    fps_update_time: Option<Instant>,
    // Left-click drag: pan
    pub(super) mouse_pressed: bool,
    pub(super) last_mouse_pos: Option<(f64, f64)>,
    // Right-click drag: pitch (vertical) + rotate (horizontal)
    pub(super) right_mouse_pressed: bool,
    pub(super) last_right_pos: Option<(f64, f64)>,
    // Middle-click drag: rotate (bearing, alternative)
    pub(super) middle_mouse_pressed: bool,
    pub(super) last_rotate_x: Option<f64>,
}

impl NativeApp {
    /// Convert an Instant to f64 seconds since app start (for AnimationController).
    pub(super) fn now_secs(&self, instant: Instant) -> f64 {
        instant.duration_since(self.start_time).as_secs_f64()
    }

    /// Toggle terrain rendering on/off. Returns the new state.
    ///
    /// No layers are added or removed.  Elevation data is loaded on the
    /// raster imagery layer as a secondary data stream.
    pub(super) fn toggle_terrain(&mut self) -> bool {
        let url = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
        let ctrl = self.controller.as_mut().unwrap();
        let enabled = ctrl.toggle_terrain(url, x_planets_tiles::TerrainEncoding::Terrarium);

        if enabled {
            // Set elevation source on the imagery layer
            let imagery_name = ctrl.terrain_imagery_name()
                .unwrap_or("base").to_string();
            let terrain_url = ctrl.terrain_url().unwrap_or(url).to_string();
            if let Some(ls) = self.layer_states.iter_mut().find(|ls| ls.name == imagery_name) {
                ls.elevation_source = Some(std::sync::Arc::new(
                    crate::tile_source::NativeTileSource::new(terrain_url),
                ));
            }
        } else {
            // Clear elevation data on all layers
            for ls in &mut self.layer_states {
                ls.elevation_source = None;
                ls.terrain_data = x_planets_tiles::TileCache::new(256);
                ls.pending_elevation_coords.clear();
            }
        }

        enabled
    }
}

impl ApplicationHandler for NativeApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("x-planets — Arrows: pan | +/-: zoom | RMB: pitch | MMB/Q/E: rotate | T: terrain")
            .with_inner_size(winit::dpi::PhysicalSize::new(800, 600));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let size = window.inner_size();
        self.window = Some(window.clone());

        // ── GPU ──
        let gpu = pollster::block_on(GpuContext::new_with_window(
            window,
            size.width,
            size.height,
        ))
        .expect("Failed to initialize GPU context");

        log::info!("GPU adapter: {}", gpu.adapter_info().name);

        // ── TileRenderer ──
        let renderer = TileRenderer::new(&gpu);

        // ── TerrainRenderer ──
        let mut terrain_renderer = TerrainRenderer::new(&gpu);
        terrain_renderer.exaggeration = self.config.terrain_exaggeration;

        // ── Model3dRenderer ──
        let model3d_renderer = Model3dRenderer::new(&gpu);

        // ── TextureManager (creates per-tile label textures on demand) ──
        let tex_manager = TextureManager::new(&gpu.device);

        // ── MapController (unified: same as web platform) ──
        let config = std::mem::take(&mut self.config);
        let controller = MapController::new(config, size.width, size.height);

        // ── Per-layer GPU state (raster + terrain) ──
        self.layer_states = init::init_layer_states(&self.rt, &controller.engine);

        // ── Per-layer 3D Tiles state ──
        self.tiles3d_states = init::init_tiles3d_states(&controller.engine);

        log::info!(
            "Engine ready: {} layers, center=({:.2},{:.2}) zoom={:.1}",
            controller.engine.layers.len(),
            controller.engine.viewport.center.lat,
            controller.engine.viewport.center.lon,
            controller.engine.viewport.zoom,
        );

        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.terrain_renderer = Some(terrain_renderer);
        self.model3d_renderer = Some(model3d_renderer);
        self.tex_manager = Some(tex_manager);
        self.controller = Some(controller);
        self.start_time = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Window closed.");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize_surface(size.width, size.height);
                    if let Some(renderer) = &mut self.renderer {
                        renderer.resize(&gpu.device, size.width, size.height);
                    }
                    if let Some(terrain_renderer) = &mut self.terrain_renderer {
                        terrain_renderer.resize(&gpu.device, size.width, size.height);
                    }
                    if let Some(model3d_renderer) = &mut self.model3d_renderer {
                        model3d_renderer.resize(&gpu.device, size.width, size.height);
                    }
                }
                if let Some(ctrl) = &mut self.controller {
                    ctrl.resize(size.width, size.height);
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard_input(event);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.handle_mouse_input(state, button);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.handle_cursor_moved(position);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.handle_mouse_wheel(delta);
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
            }
            _ => {}
        }
    }
}

impl NativeApp {
    fn render_frame(&mut self) {
        if self.gpu.is_none()
            || self.controller.is_none()
            || self.renderer.is_none()
            || self.tex_manager.is_none()
        {
            return;
        }

        // ── 1. Frame timing ──
        let now = Instant::now();
        let now_secs = self.now_secs(now);
        let dt = self
            .last_frame_time
            .map(|t| now.duration_since(t).as_secs_f64())
            .unwrap_or(1.0 / 60.0)
            .min(0.1);
        self.last_frame_time = Some(now);

        // ── 2. Tick animations via MapController (same as web) ──
        let ctrl = self.controller.as_mut().unwrap();
        ctrl.tick(dt);
        ctrl.gc_fades(now_secs);

        // ── 3. Surface setup ──
        let (frame, view) = {
            let gpu = self.gpu.as_ref().unwrap();
            let surf = match gpu.surface.as_ref() {
                Some(s) => s,
                None => return,
            };

            let frame = match surf.surface.get_current_texture() {
                Ok(f) => f,
                Err(wgpu::SurfaceError::Lost) => {
                    surf.surface.configure(&gpu.device, &surf.config);
                    return;
                }
                Err(_) => return,
            };

            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            (frame, view)
        };

        // ── 4. Per-layer async tile loading pipeline ──
        let ctrl = self.controller.as_ref().unwrap();
        let visible = ctrl.visible_tiles();
        let camera_center = x_planets_math::geo_to_mercator(&ctrl.engine.viewport.center);
        let visible_set: HashSet<TileCoord> = visible.iter().map(|vt| vt.coord).collect();

        self.run_tile_loading(&visible, &visible_set, camera_center, now);
        self.poll_tile_results(now);

        // ── 4a. Refresh available coords caches ──
        for ls in &mut self.layer_states {
            ls.refresh_available_cache();
        }

        // ── 4b. Tile visibility tracking via MapController ──
        {
            let mut all_available: HashSet<TileCoord> = HashSet::new();
            for ls in &self.layer_states {
                for coord in &ls.available_coords_cache {
                    all_available.insert(*coord);
                }
            }
            let ctrl = self.controller.as_mut().unwrap();
            ctrl.update_visibility(&visible, &all_available, now_secs);
        }

        // ── 5. LRU bump all layers ──
        for ls in &mut self.layer_states {
            {
                let base_max = 1u8.min(ls.max_zoom);
                for z in ls.min_zoom..=base_max {
                    let n = 1u32 << z;
                    for y in 0..n {
                        for x in 0..n {
                            let c = x_planets_math::TileCoord::new(z, x, y);
                            let _ = ls.tile_textures.get(&c);
                            let _ = ls.terrain_data.get(&c);
                        }
                    }
                }
            }
            for vt in &visible {
                let coord = vt.coord;
                let _ = ls.tile_textures.get(&coord);
                let _ = ls.terrain_data.get(&coord);
                let mut parent = coord.parent();
                while let Some(p) = parent {
                    let tex_found = ls.tile_textures.get(&p).is_some();
                    let _ = ls.terrain_data.get(&p);
                    if tex_found {
                        break;
                    }
                    parent = p.parent();
                }
            }
        }

        // ── 6. Build render data via MapController (same as web) ──
        // ── 7. Render ──
        {
            let ctrl = self.controller.as_ref().unwrap();
            let layer_view_refs: Vec<&dyn LayerStateView> = self
                .layer_states
                .iter()
                .map(|ls| ls as &dyn LayerStateView)
                .collect();

            let render_output = ctrl.build_render_data(
                &layer_view_refs,
                &|layer_name, coord| {
                    self.layer_states
                        .iter()
                        .find(|ls| ls.name == layer_name)
                        .and_then(|ls| ls.tile_textures.peek(coord))
                        .map(|tex| &tex.view)
                },
                &visible,
                now_secs,
            );

            let renderer = self.renderer.as_ref().unwrap();
            let gpu = self.gpu.as_ref().unwrap();

            let proj_mode = ctrl.rendering_mode();
            renderer.render_frame_layered_projected(
                gpu,
                &view,
                &ctrl.engine.viewport,
                &render_output.raster_layers,
                proj_mode,
            );

            // Render terrain layers on top of raster
            if !render_output.terrain_layers.is_empty()
                || !render_output.terrain_overlay_layers.is_empty()
            {
                if let Some(terrain_renderer) = &mut self.terrain_renderer {
                    if !render_output.terrain_layers.is_empty() {
                        terrain_renderer.render_terrain_layered(
                            gpu,
                            &view,
                            &ctrl.engine.viewport,
                            &render_output.terrain_layers,
                        );
                    }
                    if !render_output.terrain_overlay_layers.is_empty() {
                        terrain_renderer.render_terrain_layered(
                            gpu,
                            &view,
                            &ctrl.engine.viewport,
                            &render_output.terrain_overlay_layers,
                        );
                    }
                }
            }
        }

        // ── 7b. 3D Tiles: init, load, traverse, render ──
        self.tiles3d_spawn_init();
        self.tiles3d_poll_messages();
        self.tiles3d_traverse_and_render(&view);

        frame.present();

        // ── 8. FPS counter ──
        let ctrl = self.controller.as_ref().unwrap();
        self.frame_count += 1;
        let fps_elapsed = self
            .fps_update_time
            .map(|t| now.duration_since(t).as_secs_f64())
            .unwrap_or(1.0);
        if fps_elapsed >= 0.5 {
            let fps = self.frame_count as f64 / fps_elapsed;
            if let Some(window) = &self.window {
                window.set_title(&format!(
                    "x-planets — {:.0} FPS | z={:.1} | {} tiles",
                    fps,
                    ctrl.engine.viewport.zoom,
                    visible.len(),
                ));
            }
            self.frame_count = 0;
            self.fps_update_time = Some(now);
        }

        // ── 9. Continue rendering if animations or loading are in progress ──
        let any_pending = self.layer_states.iter().any(|ls| {
            !ls.pending_coords.is_empty()
                || visible.iter().any(|vt| !ls.tile_textures.contains(&vt.coord))
        });
        let any_tiles3d_pending = self.tiles3d_states.iter().any(|ts| {
            !ts.pending_uris.is_empty() || !ts.is_initialized()
        });
        if any_pending
            || any_tiles3d_pending
            || ctrl.needs_redraw()
        {
            self.window.as_ref().unwrap().request_redraw();
        }
    }
}
