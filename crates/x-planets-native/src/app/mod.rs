//! Native application event loop (winit + wgpu).
//!
//! The `NativeApp` struct and `ApplicationHandler` implementation that drives
//! rendering.  Sub-modules split the logic into focused responsibilities.

mod init;
mod input;
mod render_layers;
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
use x_planets_core::{MapEngine, Model3dRenderer, TerrainRenderer, TileRenderer};
use x_planets_gpu::{GpuContext, TextureManager};
use x_planets_math::TileCoord;

use crate::animation::AnimationState;
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
        engine: None,
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
        // Animation
        anim: AnimationState::new(2.0), // will be re-initialized in resumed()
        // Frame timing
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

struct NativeApp {
    config: MapConfig,
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    engine: Option<MapEngine>,
    renderer: Option<TileRenderer>,
    terrain_renderer: Option<TerrainRenderer>,
    model3d_renderer: Option<Model3dRenderer>,
    tex_manager: Option<TextureManager>,
    /// Per-layer tile source, texture cache, loader, pending set.
    layer_states: Vec<NativeLayerState>,
    // ── 3D Tiles state ──
    tiles3d_states: Vec<Tiles3dLayerState>,
    tiles3d_tx: mpsc::Sender<Tiles3dMessage>,
    tiles3d_rx: mpsc::Receiver<Tiles3dMessage>,
    // Shared async tile channel (results tagged with layer name)
    rt: tokio::runtime::Runtime,
    tile_tx: mpsc::Sender<LayerTileResult>,
    tile_rx: mpsc::Receiver<LayerTileResult>,
    // Animation state
    anim: AnimationState,
    // Frame timing
    last_frame_time: Option<Instant>,
    frame_count: u32,
    fps_update_time: Option<Instant>,
    // Left-click drag: pan
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    // Right-click drag: pitch (vertical) + rotate (horizontal)
    right_mouse_pressed: bool,
    last_right_pos: Option<(f64, f64)>,
    // Middle-click drag: rotate (bearing, alternative)
    middle_mouse_pressed: bool,
    last_rotate_x: Option<f64>,
}

impl NativeApp {
    fn resolve_projection_mode(&self) -> x_planets_math::ProjectionMode {
        match self.engine.as_ref() {
            Some(engine) => engine.rendering_mode(),
            None => x_planets_math::ProjectionMode::Mercator,
        }
    }
}

impl ApplicationHandler for NativeApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("x-planets — Arrows: pan | +/-: zoom | RMB: pitch | MMB/Q/E: rotate")
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

        // ── MapEngine ──
        let config = std::mem::take(&mut self.config);
        let engine = MapEngine::new(config, size.width, size.height);

        // ── Animation state (sync zoom target with engine) ──
        self.anim = AnimationState::new(engine.viewport.zoom);

        // ── Per-layer GPU state (raster + terrain) ──
        self.layer_states = init::init_layer_states(&self.rt, &engine);

        // ── Per-layer 3D Tiles state ──
        self.tiles3d_states = init::init_tiles3d_states(&engine);

        log::info!(
            "Engine ready: {} layers, center=({:.2},{:.2}) zoom={:.1}",
            engine.layers.len(),
            engine.viewport.center.lat,
            engine.viewport.center.lon,
            engine.viewport.zoom,
        );

        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.terrain_renderer = Some(terrain_renderer);
        self.model3d_renderer = Some(model3d_renderer);
        self.tex_manager = Some(tex_manager);
        self.engine = Some(engine);
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
                    // Resize depth texture alongside surface.
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
                if let Some(engine) = &mut self.engine {
                    engine.resize(size.width, size.height);
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
            || self.engine.is_none()
            || self.renderer.is_none()
            || self.tex_manager.is_none()
        {
            return;
        }

        // ── 1. Frame timing ──
        let now = Instant::now();
        let dt = self
            .last_frame_time
            .map(|t| now.duration_since(t).as_secs_f64())
            .unwrap_or(1.0 / 60.0)
            .min(0.1); // clamp: 100ms max (prevents jump after tab switch)
        self.last_frame_time = Some(now);

        // ── 2. Tick animations (needs mutable engine) ──
        {
            let mode = self.resolve_projection_mode();
            let engine = self.engine.as_mut().unwrap();
            self.anim.tick_zoom_for_mode(engine, dt, mode);
            self.anim.tick_pan_for_mode(engine, dt, mode);
        }
        self.anim.gc_fades(now);

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
        let proj_mode_for_tiles = self.resolve_projection_mode();
        let visible = self.engine.as_ref().unwrap().viewport.visible_tiles_for_mode(proj_mode_for_tiles);
        let camera_center =
            x_planets_math::geo_to_mercator(&self.engine.as_ref().unwrap().viewport.center);
        let visible_set: HashSet<TileCoord> = visible.iter().map(|vt| vt.coord).collect();

        self.run_tile_loading(&visible, &visible_set, camera_center, now);
        self.poll_tile_results(now);

        // ── 4b. Tile visibility tracking (shared core logic) ──
        // Register fade-in for cached tiles newly entering viewport and
        // track departing tiles for zoom-out fade-out.
        {
            let mut all_available: HashSet<TileCoord> = HashSet::new();
            for ls in &self.layer_states {
                for coord in ls.tile_textures.keys() {
                    all_available.insert(*coord);
                }
            }
            let now_secs = self.anim.to_secs_f64(now);
            let prev = std::mem::take(&mut self.anim.prev_visible_available);
            // Split borrows: fade_start is read by fade_elapsed_fn and
            // written by register_fade_fn, departing_tiles is separate.
            let fade_start = &self.anim.tile_fade_start;
            let departing = &mut self.anim.departing_tiles;
            let mut to_register: Vec<TileCoord> = Vec::new();
            let new_prev = x_planets_core::interaction::update_tile_visibility(
                &visible,
                &all_available,
                &prev,
                |coord| fade_start.get(coord).map(|&s| now.duration_since(s).as_secs_f64()),
                |coord| to_register.push(coord),
                departing,
                now_secs,
            );
            for coord in to_register {
                self.anim.tile_fade_start.insert(coord, now);
            }
            self.anim.prev_visible_available = new_prev;
        }

        // ── 5. LRU bump all layers (mutable pass) ──
        // Bump visible tiles AND their fallback ancestors to prevent
        // parent tiles from being evicted while still needed as fallback
        // coverage for unloaded children.
        for ls in &mut self.layer_states {
            // Always bump base tiles (z=0, z=1) to prevent LRU eviction.
            // These provide global fallback coverage for all tiles.
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
                // Also bump ancestor tiles that might serve as fallbacks
                let mut parent = coord.parent();
                while let Some(p) = parent {
                    let tex_found = ls.tile_textures.get(&p).is_some();
                    let _ = ls.terrain_data.get(&p);
                    if tex_found {
                        break; // bumped — ancestors above are even older, skip
                    }
                    parent = p.parent();
                }
            }
        }

        // ── 6. Build RenderLayerData for each visible layer (immutable pass) ──
        // ── 7. Render (raster first, then terrain on top) ──
        {
            let (render_layers, terrain_layers, terrain_overlay_layers) =
                render_layers::build_all_layers(
                    self.engine.as_ref().unwrap(),
                    &self.layer_states,
                    &self.anim,
                    &visible,
                    now,
                );

            let renderer = self.renderer.as_ref().unwrap();
            let gpu = self.gpu.as_ref().unwrap();
            let engine = self.engine.as_ref().unwrap();

            let proj_mode = engine.rendering_mode();
            renderer.render_frame_layered_projected(gpu, &view, &engine.viewport, &render_layers, proj_mode);

            // Render terrain layers (displaced meshes) on top of raster
            if !terrain_layers.is_empty() || !terrain_overlay_layers.is_empty() {
                if let Some(terrain_renderer) = &mut self.terrain_renderer {
                    // Base terrain pass: parent fallback imagery for stable coverage.
                    if !terrain_layers.is_empty() {
                        terrain_renderer.render_terrain_layered(
                            gpu,
                            &view,
                            &engine.viewport,
                            &terrain_layers,
                        );
                    }
                    // Cross-fade overlay pass: child imagery fading in.
                    // Must be a SEPARATE call because the mesh cache has
                    // one uniform buffer per tile coord — the overlay needs
                    // different uniform values (child texture + fade opacity)
                    // for the same coords.  Separate submission ensures the
                    // base pass uniforms are consumed before being overwritten.
                    if !terrain_overlay_layers.is_empty() {
                        terrain_renderer.render_terrain_layered(
                            gpu,
                            &view,
                            &engine.viewport,
                            &terrain_overlay_layers,
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

        // ── 8. FPS counter (update window title every 500ms) ──
        let engine = self.engine.as_ref().unwrap();
        self.frame_count += 1;
        let fps_elapsed = self
            .fps_update_time
            .map(|t| now.duration_since(t).as_secs_f64())
            .unwrap_or(1.0); // trigger immediately on first frame
        if fps_elapsed >= 0.5 {
            let fps = self.frame_count as f64 / fps_elapsed;
            if let Some(window) = &self.window {
                window.set_title(&format!(
                    "x-planets — {:.0} FPS | z={:.1} | {} tiles",
                    fps,
                    engine.viewport.zoom,
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
            || self.anim.is_animating(engine.viewport.zoom)
        {
            self.window.as_ref().unwrap().request_redraw();
        }
    }
}
