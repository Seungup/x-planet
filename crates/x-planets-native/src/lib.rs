//! x-planets-native: Native platform backend.
//!
//! Provides desktop windowing (winit), async I/O (tokio),
//! and HTTP client (reqwest) implementations.
//!
//! Phase 1.2: TileRenderer + MapEngine + keyboard/mouse pan/zoom/pitch/rotate.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::sync::Arc;

use async_trait::async_trait;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};
use x_planets_core::engine::MapConfig;
use x_planets_core::render::RenderLayerData;
use x_planets_core::{MapEngine, TileRenderer};
use x_planets_gpu::{GpuContext, GpuTexture, TextureManager};
use x_planets_math::TileCoord;
use x_planets_tiles::{
    DecodedRasterTile, RasterTileDecoder, TileCache, TileDecoder, TileLoader, TileRequest,
    TileSource,
};

/// Result from a layer tile fetch+decode, tagged with the layer name.
struct LayerTileResult {
    layer_name: String,
    result: Result<DecodedRasterTile, (TileCoord, String)>,
}

/// Per-layer GPU state: tile source, texture cache, loader, pending set.
struct NativeLayerState {
    name: String,
    tile_source: Arc<NativeTileSource>,
    tile_textures: TileCache<GpuTexture>,
    tile_loader: TileLoader,
    pending_coords: HashSet<TileCoord>,
}

// ═══════════════════════════════════════════════════════════════════
// Tile Source
// ═══════════════════════════════════════════════════════════════════

/// Native HTTP-based tile source using reqwest.
pub struct NativeTileSource {
    client: reqwest::Client,
    url_template: String,
    tms: bool,
}

impl NativeTileSource {
    pub fn new(url_template: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("x-planets/0.1")
                .build()
                .expect("Failed to create HTTP client"),
            url_template: url_template.into(),
            tms: false,
        }
    }

    pub fn with_tms(mut self, tms: bool) -> Self {
        self.tms = tms;
        self
    }
}

#[async_trait]
impl TileSource for NativeTileSource {
    async fn fetch(&self, coord: TileCoord) -> Result<Vec<u8>, x_planets_tiles::LoadError> {
        let url = self.tile_url(&coord);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| x_planets_tiles::LoadError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status == 404 {
            return Err(x_planets_tiles::LoadError::NotFound(coord));
        }
        if !response.status().is_success() {
            return Err(x_planets_tiles::LoadError::HttpError { status, url });
        }

        response
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| x_planets_tiles::LoadError::Network(e.to_string()))
    }

    fn tile_url(&self, coord: &TileCoord) -> String {
        let y = if self.tms {
            (1u32 << coord.z) - 1 - coord.y
        } else {
            coord.y
        };

        self.url_template
            .replace("{z}", &coord.z.to_string())
            .replace("{x}", &coord.x.to_string())
            .replace("{y}", &y.to_string())
    }
}

// ═══════════════════════════════════════════════════════════════════
// Native Application
// ═══════════════════════════════════════════════════════════════════

/// Run the native application with an event loop.
///
/// Controls:
///   Arrow keys / left-drag        — pan
///   +/- / scroll wheel            — zoom
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

    let event_loop = EventLoop::new()?;
    let mut app = NativeApp {
        config,
        window: None,
        gpu: None,
        engine: None,
        renderer: None,
        tex_manager: None,
        // Per-layer GPU state (created in `resumed`)
        layer_states: Vec::new(),
        // Shared async tile channel
        rt,
        tile_tx,
        tile_rx,
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
    tex_manager: Option<TextureManager>,
    /// Per-layer tile source, texture cache, loader, pending set.
    layer_states: Vec<NativeLayerState>,
    // Shared async tile channel (results tagged with layer name)
    rt: tokio::runtime::Runtime,
    tile_tx: mpsc::Sender<LayerTileResult>,
    tile_rx: mpsc::Receiver<LayerTileResult>,
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

        // ── TextureManager (creates per-tile label textures on demand) ──
        let tex_manager = TextureManager::new(&gpu.device);

        // ── MapEngine ──
        let config = std::mem::take(&mut self.config);
        let engine = MapEngine::new(config, size.width, size.height);

        // ── Per-layer GPU state ──
        self.layer_states = engine
            .layers
            .iter()
            .map(|layer| {
                let cfg = &layer.config;
                log::info!(
                    "Creating layer '{}' → {} (max_concurrent={}, max_cached={})",
                    cfg.name, cfg.tile_source_url,
                    cfg.max_concurrent_loads, cfg.max_cached_tiles,
                );
                NativeLayerState {
                    name: cfg.name.clone(),
                    tile_source: Arc::new(NativeTileSource::new(&cfg.tile_source_url)),
                    tile_textures: TileCache::new(cfg.max_cached_tiles),
                    tile_loader: TileLoader::new(cfg.max_concurrent_loads),
                    pending_coords: HashSet::new(),
                }
            })
            .collect();

        log::info!(
            "Engine ready: {} layers, center=({:.2},{:.2}) zoom={:.1}",
            engine.layers.len(),
            engine.viewport.center.lat,
            engine.viewport.center.lon,
            engine.viewport.zoom,
        );

        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
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
                }
                if let Some(engine) = &mut self.engine {
                    engine.resize(size.width, size.height);
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == winit::event::ElementState::Pressed {
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
                            PhysicalKey::Code(KeyCode::Equal)
                            | PhysicalKey::Code(KeyCode::NumpadAdd) => {
                                engine.zoom(0.5);
                            }
                            PhysicalKey::Code(KeyCode::Minus)
                            | PhysicalKey::Code(KeyCode::NumpadSubtract) => {
                                engine.zoom(-0.5);
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
                                engine.request_redraw();
                            }
                            _ => {}
                        }
                    }
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        self.mouse_pressed = pressed;
                        if !pressed { self.last_mouse_pos = None; }
                    }
                    MouseButton::Right => {
                        self.right_mouse_pressed = pressed;
                        if !pressed { self.last_right_pos = None; }
                    }
                    MouseButton::Middle => {
                        self.middle_mouse_pressed = pressed;
                        if !pressed { self.last_rotate_x = None; }
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
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
                }

                // Right-drag: pitch (vertical) + rotate (horizontal)
                if self.right_mouse_pressed {
                    if let Some(last) = self.last_right_pos {
                        let dx = pos.0 - last.0;
                        let dy = pos.1 - last.1;
                        if let Some(engine) = &mut self.engine {
                            engine.pitch(-dy * 0.3);  // drag up = more tilt
                            engine.rotate(dx * 0.3);  // drag right = clockwise
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
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64 * 0.3,
                    MouseScrollDelta::PixelDelta(p) => p.y * 0.003,
                };
                if let Some(engine) = &mut self.engine {
                    if let Some((mx, my)) = self.last_mouse_pos {
                        engine.zoom_at(scroll_y, mx, my);
                    } else {
                        engine.zoom(scroll_y);
                    }
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let (Some(gpu), Some(engine), Some(_renderer), Some(tex_mgr)) = (
                    self.gpu.as_ref(),
                    self.engine.as_ref(),
                    self.renderer.as_ref(),
                    self.tex_manager.as_ref(),
                ) else {
                    return;
                };

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

                // ── Per-layer async tile loading pipeline ──
                let visible = engine.viewport.visible_tiles();
                let camera_center = x_planets_math::geo_to_mercator(&engine.viewport.center);

                // 1. For each layer: enqueue missing tiles & spawn fetch tasks
                for ls in &mut self.layer_states {
                    // Enqueue
                    for &coord in &visible {
                        if ls.tile_textures.contains(&coord) || ls.pending_coords.contains(&coord) {
                            continue;
                        }
                        let tile_center = coord.mercator_center();
                        let dist = (tile_center - camera_center).length() as f32;
                        ls.tile_loader.enqueue(TileRequest { coord, priority: dist });
                        ls.pending_coords.insert(coord);
                    }

                    // Dequeue & spawn
                    while let Some(req) = ls.tile_loader.dequeue() {
                        let source = Arc::clone(&ls.tile_source);
                        let tx = self.tile_tx.clone();
                        let layer_name = ls.name.clone();
                        self.rt.spawn(async move {
                            let result = match source.fetch(req.coord).await {
                                Ok(bytes) => {
                                    let decoder = RasterTileDecoder::default();
                                    match decoder.decode(req.coord, &bytes).await {
                                        Ok(decoded) => Ok(decoded),
                                        Err(e) => Err((req.coord, e.to_string())),
                                    }
                                }
                                Err(e) => Err((req.coord, e.to_string())),
                            };
                            let _ = tx.send(LayerTileResult { layer_name, result });
                        });
                    }
                }

                // 2. Poll completed tiles & create GPU textures (dispatched by layer name)
                while let Ok(msg) = self.tile_rx.try_recv() {
                    if let Some(ls) = self.layer_states.iter_mut().find(|s| s.name == msg.layer_name) {
                        ls.tile_loader.complete();
                        match msg.result {
                            Ok(decoded) => {
                                log::debug!(
                                    "[{}] Tile loaded: z={} x={} y={} ({}×{})",
                                    ls.name,
                                    decoded.coord.z, decoded.coord.x, decoded.coord.y,
                                    decoded.width, decoded.height,
                                );
                                ls.pending_coords.remove(&decoded.coord);
                                let tex = tex_mgr.create_rgba_texture(
                                    &gpu.device, &gpu.queue,
                                    &format!("{}-tile-{}-{}-{}", ls.name,
                                        decoded.coord.z, decoded.coord.x, decoded.coord.y),
                                    decoded.width, decoded.height, &decoded.pixels,
                                );
                                ls.tile_textures.insert(decoded.coord, tex);
                            }
                            Err((coord, err_msg)) => {
                                log::warn!("[{}] Tile load failed {}: {}", ls.name, coord, err_msg);
                                ls.pending_coords.remove(&coord);
                            }
                        }
                    }
                }

                // 3a. LRU bump all layers (mutable pass)
                for ls in &mut self.layer_states {
                    for &coord in &visible {
                        let _ = ls.tile_textures.get(&coord);
                    }
                }

                // 3b. Build RenderLayerData for each visible layer (immutable pass)
                let engine = self.engine.as_ref().unwrap();
                let mut render_layers: Vec<RenderLayerData> = Vec::new();

                for layer in engine.visible_layers() {
                    if let Some(ls) = self.layer_states.iter().find(|s| s.name == layer.config.name) {
                        // Resolve fallbacks
                        let available: HashSet<TileCoord> =
                            ls.tile_textures.keys().copied().collect();
                        let renderable = x_planets_core::pipeline::resolve_fallbacks(&visible, &available);

                        // Build texture view map
                        let texture_views: HashMap<TileCoord, &wgpu::TextureView> = ls
                            .tile_textures
                            .iter()
                            .map(|(k, v)| (*k, &v.view))
                            .collect();

                        render_layers.push(RenderLayerData {
                            name: &layer.config.name,
                            opacity: layer.config.opacity,
                            tiles: renderable,
                            texture_views,
                        });
                    }
                }

                let renderer = self.renderer.as_ref().unwrap();
                let gpu = self.gpu.as_ref().unwrap();

                renderer.render_frame_layered(gpu, &view, &engine.viewport, &render_layers);
                frame.present();

                // Request redraw when any layer is still loading.
                let any_pending = self.layer_states.iter().any(|ls| {
                    !ls.pending_coords.is_empty()
                        || visible.iter().any(|c| !ls.tile_textures.contains(c))
                });
                if any_pending {
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            _ => {}
        }
    }
}
