//! x-planets-native: Native platform backend.
//!
//! Provides desktop windowing (winit), async I/O (tokio),
//! and HTTP client (reqwest) implementations.
//!
//! Supports raster, terrain, and OGC 3D Tiles layers.

pub mod config;
mod tiles3d_native;

use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};
use x_planets_core::engine::{LayerKind, MapConfig};
use x_planets_core::render::RenderLayerData;
use x_planets_core::{
    MapEngine, Model3dRenderer, TerrainLayerData, TerrainRenderer, TerrainTileData, TileRenderer,
};
use x_planets_gpu::{GpuContext, GpuTexture, TextureManager};
use x_planets_math::TileCoord;
use x_planets_tiles::{
    DecodedRasterTile, DecodedTerrainTile, RasterTileDecoder, TerrainEncoding, TerrainRgbDecoder,
    TerrariumDecoder, TileCache, TileDecoder, TileLoader, TileRequest, TileSource,
};

use tiles3d_native::{
    cesium_resolve_endpoint, fetch_tile_content, fetch_tileset, Tiles3dAuthKind,
    Tiles3dLayerState, Tiles3dMessage,
};

// ═══════════════════════════════════════════════════════════════════
// Animation utilities
// ═══════════════════════════════════════════════════════════════════

/// Frame-rate-independent exponential decay interpolation.
///
/// Moves `current` toward `target` at a rate determined by `speed`.
/// `speed = 12.0` reaches ~63% of the remaining distance per ~83ms.
fn exp_decay(current: f64, target: f64, speed: f64, dt: f64) -> f64 {
    current + (target - current) * (1.0 - (-speed * dt).exp())
}

/// Duration (seconds) for newly loaded tiles to fade from 0→1 opacity.
const FADE_DURATION: f64 = 0.3;

// ═══════════════════════════════════════════════════════════════════
// Animation state
// ═══════════════════════════════════════════════════════════════════

/// Tracks all running animations so the render loop can tick them each frame.
struct AnimationState {
    // ── Smooth zoom ──
    /// Target zoom level (accumulated from scroll/keyboard input).
    zoom_target: f64,
    /// Screen-space anchor point for zoom-toward-cursor.  `None` = zoom at center.
    zoom_anchor: Option<(f64, f64)>,

    // ── Inertia pan ──
    /// Current pan velocity in screen pixels/sec.
    pan_velocity: (f64, f64),
    /// Recent drag samples for velocity estimation (position, timestamp).
    last_drag_positions: Vec<((f64, f64), Instant)>,

    // ── Tile fade-in ──
    /// Maps newly loaded tile coords → the `Instant` they first appeared as GPU textures.
    tile_fade_start: HashMap<TileCoord, Instant>,

    // ── Double-click detection ──
    last_click_time: Option<Instant>,
    last_click_pos: Option<(f64, f64)>,
}

impl AnimationState {
    fn new(initial_zoom: f64) -> Self {
        Self {
            zoom_target: initial_zoom,
            zoom_anchor: None,
            pan_velocity: (0.0, 0.0),
            last_drag_positions: Vec::new(),
            tile_fade_start: HashMap::new(),
            last_click_time: None,
            last_click_pos: None,
        }
    }

    /// Returns `true` if any animation is still running and requires continuous redraws.
    fn is_animating(&self, current_zoom: f64) -> bool {
        (self.zoom_target - current_zoom).abs() > 0.001
            || (self.pan_velocity.0.powi(2) + self.pan_velocity.1.powi(2)).sqrt() > 1.0
            || !self.tile_fade_start.is_empty()
    }

    /// Advance smooth zoom toward `zoom_target`.
    fn tick_zoom(&mut self, engine: &mut MapEngine, dt: f64) {
        let current = engine.viewport.zoom;
        let target = self
            .zoom_target
            .clamp(engine.camera.min_zoom, engine.camera.max_zoom);
        let diff = target - current;
        if diff.abs() < 0.001 {
            if (engine.viewport.zoom - target).abs() > 1e-9 {
                engine.viewport.zoom = target;
                engine.request_redraw();
            }
            return;
        }
        let new_zoom = exp_decay(current, target, 12.0, dt);
        let delta = new_zoom - current;
        match self.zoom_anchor {
            Some((mx, my)) => engine.zoom_at(delta, mx, my),
            None => engine.zoom(delta),
        }
    }

    /// Advance inertia panning (friction-based velocity decay).
    fn tick_pan(&mut self, engine: &mut MapEngine, dt: f64) {
        let (vx, vy) = self.pan_velocity;
        let speed = (vx * vx + vy * vy).sqrt();
        if speed < 1.0 {
            self.pan_velocity = (0.0, 0.0);
            return;
        }
        // Move by velocity × dt (negate vy because screen Y is inverted for pan).
        engine.pan(vx * dt, -(vy * dt));
        // Exponential friction decay.
        let friction = (-6.0 * dt).exp();
        self.pan_velocity = (vx * friction, vy * friction);
    }

    /// Record a drag position sample for velocity estimation.
    fn record_drag(&mut self, pos: (f64, f64)) {
        let now = Instant::now();
        // Keep only the last 100ms of samples.
        self.last_drag_positions
            .retain(|(_, t)| now.duration_since(*t).as_millis() < 100);
        self.last_drag_positions.push((pos, now));
    }

    /// Compute pan velocity from recent drag samples (called on mouse-up).
    fn compute_release_velocity(&mut self) {
        let now = Instant::now();
        // Need at least 2 samples within the last 100ms.
        self.last_drag_positions
            .retain(|(_, t)| now.duration_since(*t).as_millis() < 100);
        if self.last_drag_positions.len() < 2 {
            self.pan_velocity = (0.0, 0.0);
            return;
        }
        let first = &self.last_drag_positions[0];
        let last = &self.last_drag_positions[self.last_drag_positions.len() - 1];
        let dt = last.1.duration_since(first.1).as_secs_f64();
        if dt < 0.001 {
            self.pan_velocity = (0.0, 0.0);
            return;
        }
        let vx = (last.0 .0 - first.0 .0) / dt;
        let vy = (last.0 .1 - first.0 .1) / dt;
        self.pan_velocity = (vx, vy);
        self.last_drag_positions.clear();
    }

    /// Garbage-collect finished fade-in entries.
    fn gc_fades(&mut self, now: Instant) {
        self.tile_fade_start.retain(|_, start| {
            now.duration_since(*start).as_secs_f64() < FADE_DURATION + 0.1
        });
    }
}

// ═══════════════════════════════════════════════════════════════════
// Layer tile result + per-layer state
// ═══════════════════════════════════════════════════════════════════

/// A decoded tile result, discriminated by layer kind.
enum TileResult {
    Raster(DecodedRasterTile),
    Terrain(DecodedTerrainTile),
}

/// Result from a layer tile fetch+decode, tagged with the layer name.
struct LayerTileResult {
    layer_name: String,
    result: Result<TileResult, (TileCoord, String)>,
}

/// Per-layer GPU state: tile source, texture cache, loader, pending set.
struct NativeLayerState {
    name: String,
    kind: LayerKind,
    tile_source: Arc<NativeTileSource>,
    tile_textures: TileCache<GpuTexture>,
    tile_loader: TileLoader,
    pending_coords: HashSet<TileCoord>,
    /// Elevation data for terrain layers (CPU-side, used for mesh generation).
    terrain_data: HashMap<TileCoord, TerrainTileData>,
    /// Cooldown for failed tiles: don't retry until the Instant has passed.
    /// Prevents infinite retry loops when the server returns 429 / transient errors.
    failed_cooldowns: HashMap<TileCoord, Instant>,
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
        let terrain_renderer = TerrainRenderer::new(&gpu);

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
        self.layer_states = engine
            .layers
            .iter()
            .filter(|layer| !matches!(layer.config.kind, LayerKind::Tiles3d))
            .map(|layer| {
                let cfg = &layer.config;
                log::info!(
                    "Creating layer '{}' → {} (max_concurrent={}, max_cached={})",
                    cfg.name, cfg.tile_source_url,
                    cfg.max_concurrent_loads, cfg.max_cached_tiles,
                );
                NativeLayerState {
                    name: cfg.name.clone(),
                    kind: cfg.kind.clone(),
                    tile_source: Arc::new(NativeTileSource::new(&cfg.tile_source_url)),
                    tile_textures: TileCache::new(cfg.max_cached_tiles),
                    tile_loader: TileLoader::new(cfg.max_concurrent_loads),
                    pending_coords: HashSet::new(),
                    terrain_data: HashMap::new(),
                    failed_cooldowns: HashMap::new(),
                }
            })
            .collect();

        // ── Per-layer 3D Tiles state ──
        self.tiles3d_states = engine
            .layers
            .iter()
            .filter(|layer| matches!(layer.config.kind, LayerKind::Tiles3d))
            .map(|layer| {
                let cfg = &layer.config;
                let auth = if let Some(token) = &cfg.cesium_ion_token {
                    let asset_id = cfg.cesium_ion_asset_id.unwrap_or(1);
                    log::info!(
                        "Creating 3D Tiles layer '{}' → Cesium Ion asset {}",
                        cfg.name, asset_id,
                    );
                    Tiles3dAuthKind::CesiumIon {
                        account_token: token.clone(),
                        asset_id,
                    }
                } else if let Some(key) = &cfg.google_api_key {
                    log::info!(
                        "Creating 3D Tiles layer '{}' → Google 3D Tiles",
                        cfg.name,
                    );
                    Tiles3dAuthKind::Google {
                        api_key: key.clone(),
                    }
                } else {
                    log::warn!(
                        "3D Tiles layer '{}' has no auth config, skipping",
                        cfg.name,
                    );
                    // Use a dummy — will fail on init
                    Tiles3dAuthKind::CesiumIon {
                        account_token: String::new(),
                        asset_id: 0,
                    }
                };
                Tiles3dLayerState::new(cfg.name.clone(), auth)
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
            WindowEvent::MouseInput { state, button, .. } => {
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
            WindowEvent::MouseWheel { delta, .. } => {
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
            WindowEvent::RedrawRequested => {
                let (Some(gpu), Some(_engine), Some(_renderer), Some(tex_mgr)) = (
                    self.gpu.as_ref(),
                    self.engine.as_ref(),
                    self.renderer.as_ref(),
                    self.tex_manager.as_ref(),
                ) else {
                    return;
                };

                // ── 1. Frame timing ──
                let now = Instant::now();
                let dt = self
                    .last_frame_time
                    .map(|t| now.duration_since(t).as_secs_f64())
                    .unwrap_or(1.0 / 60.0)
                    .min(0.1); // clamp: 100ms max (prevents jump after tab switch)
                self.last_frame_time = Some(now);

                // ── 2. Tick animations (needs mutable engine) ──
                let engine = self.engine.as_mut().unwrap();
                self.anim.tick_zoom(engine, dt);
                self.anim.tick_pan(engine, dt);
                self.anim.gc_fades(now);

                // ── 3. Surface setup ──
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

                // ── 4. Per-layer async tile loading pipeline ──
                let visible = engine.viewport.visible_tiles();
                let camera_center = x_planets_math::geo_to_mercator(&engine.viewport.center);

                // 4a. For each layer: abort stale, enqueue visible tiles, spawn fetch tasks
                let current_zoom = engine.viewport.tile_zoom();
                let visible_set: HashSet<TileCoord> = visible.iter().copied().collect();

                for ls in &mut self.layer_states {
                    // ── Abort stale requests ──
                    // Clear the priority queue every frame.  Tiles that were queued
                    // but never dequeued (max_concurrent reached) are NOT in
                    // pending_coords, so they'll be naturally re-enqueued below
                    // with fresh priorities.
                    ls.tile_loader.clear();

                    // Prune pending_coords for tiles at distant zoom levels.
                    // These are truly in-flight (task spawned), so freeing their
                    // concurrency slot lets current-view tiles load faster.
                    let stale_coords: Vec<TileCoord> = ls.pending_coords
                        .iter()
                        .filter(|c| {
                            let zoom_diff = (c.z as i32 - current_zoom as i32).unsigned_abs();
                            zoom_diff > 1 && !visible_set.contains(c)
                        })
                        .copied()
                        .collect();
                    for coord in stale_coords {
                        ls.pending_coords.remove(&coord);
                        ls.tile_loader.complete(); // free concurrency slot
                    }

                    // GC expired cooldowns (once per frame is cheap).
                    ls.failed_cooldowns.retain(|_, expire| now < *expire);

                    // Enqueue visible tiles that are not yet loaded or in-flight.
                    // Priority: distance from camera × fallback penalty.
                    // Tiles with no/distant fallback texture are prioritized
                    // (lower value = higher priority in the min-heap).
                    for &coord in &visible {
                        if ls.tile_textures.contains(&coord)
                            || ls.pending_coords.contains(&coord)
                            || ls.failed_cooldowns.contains_key(&coord)
                        {
                            continue;
                        }
                        let tile_center = coord.mercator_center();
                        let dist = (tile_center - camera_center).length() as f32;

                        // Fallback depth: how many zoom levels up to the nearest
                        // cached ancestor?  0 = no ancestor at all (blank tile!).
                        let fallback_depth = {
                            let mut depth = 0u32;
                            let mut cur = coord.parent();
                            loop {
                                match cur {
                                    Some(c) if ls.tile_textures.contains(&c) => {
                                        depth += 1;
                                        break;
                                    }
                                    Some(c) => {
                                        depth += 1;
                                        cur = c.parent();
                                    }
                                    None => {
                                        depth = 0; // no ancestor found
                                        break;
                                    }
                                }
                            }
                            depth
                        };
                        // No fallback (depth=0) → factor=0.5 (boost priority)
                        // Close fallback (depth=1) → factor=1.5 (deprioritize)
                        // Distant fallback (depth≥3) → factor=1.0 (normal)
                        let fallback_factor = match fallback_depth {
                            0 => 0.5,
                            1 => 1.5,
                            2 => 1.2,
                            _ => 1.0,
                        };
                        ls.tile_loader.enqueue(TileRequest {
                            coord,
                            priority: dist * fallback_factor,
                        });
                        // NOTE: Do NOT insert into pending_coords here!
                        // pending_coords tracks only truly in-flight tasks (spawned).
                        // Tiles that stay in the queue are dropped by clear() next
                        // frame and re-enqueued with fresh priorities.
                    }

                    // Dequeue & spawn (route decoder by layer kind).
                    // Insert into pending_coords ONLY when a task is actually spawned.
                    let terrain_encoding = match &ls.kind {
                        LayerKind::Terrain { encoding, .. } => Some(*encoding),
                        _ => None,
                    };
                    while let Some(req) = ls.tile_loader.dequeue() {
                        ls.pending_coords.insert(req.coord);
                        let source = Arc::clone(&ls.tile_source);
                        let tx = self.tile_tx.clone();
                        let layer_name = ls.name.clone();
                        if let Some(enc) = terrain_encoding {
                            self.rt.spawn(async move {
                                let result = match source.fetch(req.coord).await {
                                    Ok(bytes) => {
                                        let decoded = match enc {
                                            TerrainEncoding::MapboxRgb => {
                                                TerrainRgbDecoder.decode(req.coord, &bytes).await
                                            }
                                            TerrainEncoding::Terrarium => {
                                                TerrariumDecoder.decode(req.coord, &bytes).await
                                            }
                                        };
                                        match decoded {
                                            Ok(decoded) => Ok(TileResult::Terrain(decoded)),
                                            Err(e) => Err((req.coord, e.to_string())),
                                        }
                                    }
                                    Err(e) => Err((req.coord, e.to_string())),
                                };
                                let _ = tx.send(LayerTileResult { layer_name, result });
                            });
                        } else {
                            self.rt.spawn(async move {
                                let result = match source.fetch(req.coord).await {
                                    Ok(bytes) => {
                                        let decoder = RasterTileDecoder::default();
                                        match decoder.decode(req.coord, &bytes).await {
                                            Ok(decoded) => Ok(TileResult::Raster(decoded)),
                                            Err(e) => Err((req.coord, e.to_string())),
                                        }
                                    }
                                    Err(e) => Err((req.coord, e.to_string())),
                                };
                                let _ = tx.send(LayerTileResult { layer_name, result });
                            });
                        }
                    }
                }

                // 4b. Poll completed tiles & create GPU textures (dispatched by layer name)
                while let Ok(msg) = self.tile_rx.try_recv() {
                    if let Some(ls) = self.layer_states.iter_mut().find(|s| s.name == msg.layer_name) {
                        match msg.result {
                            Ok(TileResult::Raster(decoded)) => {
                                // Only decrement active_count if tile was still tracked as
                                // pending.  Stale tiles pruned during abort already freed
                                // their concurrency slot.
                                if ls.pending_coords.remove(&decoded.coord) {
                                    ls.tile_loader.complete();
                                }
                                log::debug!(
                                    "[{}] Raster tile loaded: z={} x={} y={} ({}×{})",
                                    ls.name,
                                    decoded.coord.z, decoded.coord.x, decoded.coord.y,
                                    decoded.width, decoded.height,
                                );
                                self.anim.tile_fade_start.insert(decoded.coord, now);
                                let tex = tex_mgr.create_rgba_texture(
                                    &gpu.device, &gpu.queue,
                                    &format!("{}-tile-{}-{}-{}", ls.name,
                                        decoded.coord.z, decoded.coord.x, decoded.coord.y),
                                    decoded.width, decoded.height, &decoded.pixels,
                                );
                                ls.tile_textures.insert(decoded.coord, tex);
                            }
                            Ok(TileResult::Terrain(decoded)) => {
                                if ls.pending_coords.remove(&decoded.coord) {
                                    ls.tile_loader.complete();
                                }
                                log::debug!(
                                    "[{}] Terrain tile loaded: z={} x={} y={} elev=[{:.0}..{:.0}]m",
                                    ls.name,
                                    decoded.coord.z, decoded.coord.x, decoded.coord.y,
                                    decoded.min_elevation, decoded.max_elevation,
                                );
                                self.anim.tile_fade_start.insert(decoded.coord, now);
                                // Store elevation data on CPU for mesh generation
                                ls.terrain_data.insert(
                                    decoded.coord,
                                    TerrainTileData {
                                        elevation: decoded.elevation,
                                        width: decoded.width,
                                        height: decoded.height,
                                    },
                                );
                                // Also insert a placeholder texture so the tile is considered "loaded"
                                // (the actual imagery texture comes from the companion raster layer)
                                let tex = tex_mgr.create_rgba_texture(
                                    &gpu.device, &gpu.queue,
                                    &format!("{}-terrain-{}-{}-{}", ls.name,
                                        decoded.coord.z, decoded.coord.x, decoded.coord.y),
                                    1, 1, &[128, 128, 128, 255], // 1×1 gray placeholder
                                );
                                ls.tile_textures.insert(decoded.coord, tex);
                            }
                            Err((coord, err_msg)) => {
                                if ls.pending_coords.remove(&coord) {
                                    ls.tile_loader.complete();
                                }
                                // Backoff cooldown: 429 → 30s, other errors → 5s.
                                let cooldown_secs = if err_msg.contains("429") { 30 } else { 5 };
                                ls.failed_cooldowns.insert(
                                    coord,
                                    now + std::time::Duration::from_secs(cooldown_secs),
                                );
                                log::warn!(
                                    "[{}] Tile load failed {} (retry in {}s): {}",
                                    ls.name, coord, cooldown_secs, err_msg,
                                );
                            }
                        }
                    }
                }

                // ── 5. LRU bump all layers (mutable pass) ──
                for ls in &mut self.layer_states {
                    for &coord in &visible {
                        let _ = ls.tile_textures.get(&coord);
                    }
                }

                // ── 6. Build RenderLayerData for each visible layer (immutable pass) ──
                let engine = self.engine.as_ref().unwrap();
                let mut render_layers: Vec<RenderLayerData> = Vec::new();
                let mut terrain_layers: Vec<TerrainLayerData> = Vec::new();

                // Collect raster layer names that are used as imagery for terrain layers.
                // These will be skipped in the flat raster render pass — they're already
                // draped onto the 3D terrain mesh.
                let terrain_imagery_names: HashSet<&str> = engine
                    .visible_layers()
                    .filter_map(|l| match &l.config.kind {
                        LayerKind::Terrain { imagery_layer, .. } => Some(imagery_layer.as_str()),
                        _ => None,
                    })
                    .collect();

                for layer in engine.visible_layers() {
                    if let Some(ls) = self.layer_states.iter().find(|s| s.name == layer.config.name) {
                        match &layer.config.kind {
                            LayerKind::Raster => {
                                // Skip raster layers that serve as terrain imagery —
                                // they're already draped onto the 3D terrain mesh.
                                if terrain_imagery_names.contains(layer.config.name.as_str()) {
                                    continue;
                                }

                                // Resolve fallbacks
                                let available: HashSet<TileCoord> =
                                    ls.tile_textures.keys().copied().collect();
                                let renderable =
                                    x_planets_core::pipeline::resolve_fallbacks(&visible, &available);

                                // Build texture view map
                                let texture_views: HashMap<TileCoord, &wgpu::TextureView> = ls
                                    .tile_textures
                                    .iter()
                                    .map(|(k, v)| (*k, &v.view))
                                    .collect();

                                // Compute per-tile fade-in opacity overrides
                                let mut tile_opacity_overrides = HashMap::new();
                                for rt in &renderable {
                                    if let Some(&start) =
                                        self.anim.tile_fade_start.get(&rt.texture_coord)
                                    {
                                        let elapsed = now.duration_since(start).as_secs_f64();
                                        if elapsed < FADE_DURATION {
                                            let t = (elapsed / FADE_DURATION).min(1.0) as f32;
                                            tile_opacity_overrides
                                                .insert(rt.coord, layer.config.opacity * t);
                                        }
                                    }
                                }

                                render_layers.push(RenderLayerData {
                                    name: &layer.config.name,
                                    opacity: layer.config.opacity,
                                    tiles: renderable,
                                    texture_views,
                                    tile_opacity_overrides,
                                });
                            }
                            LayerKind::Tiles3d => {
                                // 3D Tiles layers are handled separately below.
                            }
                            LayerKind::Terrain { imagery_layer, .. } => {
                                // Find companion imagery layer's texture views
                                let imagery_ls = self
                                    .layer_states
                                    .iter()
                                    .find(|s| s.name == *imagery_layer);

                                if let Some(img_ls) = imagery_ls {
                                    // Available = imagery tiles that are loaded.
                                    // Elevation will be looked up with parent fallback below.
                                    let available: HashSet<TileCoord> =
                                        img_ls.tile_textures.keys().copied().collect();
                                    let renderable = x_planets_core::pipeline::resolve_fallbacks(
                                        &visible, &available,
                                    );

                                    // Imagery texture views from companion layer
                                    let imagery_views: HashMap<TileCoord, &wgpu::TextureView> =
                                        img_ls
                                            .tile_textures
                                            .iter()
                                            .map(|(k, v)| (*k, &v.view))
                                            .collect();

                                    // Elevation data with parent fallback:
                                    // If elevation for a tile's exact coord isn't available,
                                    // walk up to parent coords until we find one.
                                    // Value = (data, source_coord) so the renderer can compute
                                    // the correct UV sub-rect for parent-tile sampling.
                                    let mut elevation_data: HashMap<TileCoord, (&TerrainTileData, TileCoord)> =
                                        HashMap::new();
                                    let all_needed: HashSet<TileCoord> = renderable
                                        .iter()
                                        .map(|rt| rt.coord)
                                        .collect();
                                    for &coord in &all_needed {
                                        // Try exact match first, then walk up to parents
                                        let mut c = Some(coord);
                                        while let Some(candidate) = c {
                                            if let Some(data) = ls.terrain_data.get(&candidate) {
                                                elevation_data.insert(coord, (data, candidate));
                                                break;
                                            }
                                            c = candidate.parent();
                                        }
                                    }

                                    // Compute per-tile fade-in opacity overrides
                                    let mut tile_opacity_overrides = HashMap::new();
                                    for rt in &renderable {
                                        if let Some(&start) =
                                            self.anim.tile_fade_start.get(&rt.texture_coord)
                                        {
                                            let elapsed = now.duration_since(start).as_secs_f64();
                                            if elapsed < FADE_DURATION {
                                                let t =
                                                    (elapsed / FADE_DURATION).min(1.0) as f32;
                                                tile_opacity_overrides
                                                    .insert(rt.coord, layer.config.opacity * t);
                                            }
                                        }
                                    }

                                    terrain_layers.push(TerrainLayerData {
                                        name: &layer.config.name,
                                        opacity: layer.config.opacity,
                                        tiles: renderable,
                                        imagery_views,
                                        elevation_data,
                                        tile_opacity_overrides,
                                    });
                                }
                            }
                        }
                    }
                }

                // ── 7. Render (raster first, then terrain on top) ──
                let renderer = self.renderer.as_ref().unwrap();
                let gpu = self.gpu.as_ref().unwrap();

                renderer.render_frame_layered(gpu, &view, &engine.viewport, &render_layers);

                // Render terrain layers (displaced meshes) on top of raster
                if !terrain_layers.is_empty() {
                    if let Some(terrain_renderer) = &mut self.terrain_renderer {
                        terrain_renderer.render_terrain_layered(
                            gpu,
                            &view,
                            &engine.viewport,
                            &terrain_layers,
                        );
                    }
                }

                // ── 7b. 3D Tiles: init, load, traverse, render ──
                // Spawn init tasks for uninitialized layers.
                for ts3d in &mut self.tiles3d_states {
                    if !ts3d.is_initialized() && !ts3d.init_spawned {
                        ts3d.init_spawned = true;
                        let client = ts3d.client.clone();
                        let tx = self.tiles3d_tx.clone();
                        let layer_name = ts3d.name.clone();

                        match &ts3d.auth {
                            Tiles3dAuthKind::CesiumIon {
                                account_token,
                                asset_id,
                            } => {
                                let token = account_token.clone();
                                let aid = *asset_id;
                                self.rt.spawn(async move {
                                    let result =
                                        match cesium_resolve_endpoint(&client, &token, aid).await {
                                            Ok((endpoint_url, access_token)) => {
                                                match fetch_tileset(
                                                    &client,
                                                    &endpoint_url,
                                                    Some(&access_token),
                                                )
                                                .await
                                                {
                                                    Ok((tileset, base_url)) => Ok((
                                                        tileset,
                                                        base_url,
                                                        Some(access_token),
                                                    )),
                                                    Err(e) => Err(e),
                                                }
                                            }
                                            Err(e) => Err(e),
                                        };
                                    let _ = tx.send(Tiles3dMessage::Initialized {
                                        layer_name,
                                        result,
                                    });
                                });
                            }
                            Tiles3dAuthKind::Google { api_key } => {
                                let key = api_key.clone();
                                self.rt.spawn(async move {
                                    let url = format!(
                                        "https://tile.googleapis.com/v1/3dtiles/root.json?key={}",
                                        key
                                    );
                                    let result = match fetch_tileset(&client, &url, None).await {
                                        Ok((tileset, base_url)) => {
                                            Ok((tileset, base_url, None))
                                        }
                                        Err(e) => Err(e),
                                    };
                                    let _ = tx.send(Tiles3dMessage::Initialized {
                                        layer_name,
                                        result,
                                    });
                                });
                            }
                        }
                    }
                }

                // Poll 3D Tiles messages.
                while let Ok(msg) = self.tiles3d_rx.try_recv() {
                    match msg {
                        Tiles3dMessage::Initialized {
                            layer_name,
                            result,
                        } => {
                            if let Some(ts3d) = self
                                .tiles3d_states
                                .iter_mut()
                                .find(|s| s.name == layer_name)
                            {
                                match result {
                                    Ok((tileset, base_url, access_token)) => {
                                        let tile_count =
                                            x_planets_tiles::tiles3d::tileset::tile_count(
                                                &tileset.root,
                                            );
                                        log::info!(
                                            "[{}] 3D Tiles initialized: {} tiles",
                                            layer_name,
                                            tile_count,
                                        );
                                        ts3d.tileset = Some(tileset);
                                        ts3d.base_url = base_url;
                                        ts3d.access_token = access_token;
                                    }
                                    Err(e) => {
                                        log::error!(
                                            "[{}] 3D Tiles init failed: {}",
                                            layer_name,
                                            e
                                        );
                                    }
                                }
                            }
                        }
                        Tiles3dMessage::ContentLoaded {
                            layer_name,
                            content_uri,
                            result,
                        } => {
                            if let Some(ts3d) = self
                                .tiles3d_states
                                .iter_mut()
                                .find(|s| s.name == layer_name)
                            {
                                ts3d.pending_uris.remove(&content_uri);
                                match result {
                                    Ok(decoded) => {
                                        log::debug!(
                                            "[{}] 3D tile loaded: {} ({} meshes)",
                                            layer_name,
                                            content_uri,
                                            decoded.meshes.len(),
                                        );
                                        let gpu = self.gpu.as_ref().unwrap();
                                        let renderer =
                                            self.model3d_renderer.as_ref().unwrap();
                                        ts3d.upload_decoded_tile(
                                            gpu,
                                            renderer,
                                            &content_uri,
                                            &decoded,
                                        );
                                    }
                                    Err(e) => {
                                        log::warn!(
                                            "[{}] 3D tile failed: {} - {}",
                                            layer_name,
                                            content_uri,
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // Traverse, load, and render 3D Tiles layers.
                let engine = self.engine.as_ref().unwrap();
                for ts3d in &mut self.tiles3d_states {
                    if !ts3d.is_initialized() {
                        continue;
                    }

                    let tileset = ts3d.tileset.as_ref().unwrap();
                    let camera =
                        x_planets_core::tiles3d_pipeline::viewport_to_traversal_camera(
                            &engine.viewport,
                        );
                    let config =
                        x_planets_tiles::tiles3d::traversal::TraversalConfig {
                            max_sse: 16.0,
                            tile_budget: 256,
                            screen_height: engine.viewport.height as f64,
                            fov_y: x_planets_core::tiles3d_pipeline::traversal_fov_y(),
                        };

                    let traversal =
                        x_planets_tiles::tiles3d::traversal::traverse_tileset(
                            tileset,
                            &ts3d.base_url,
                            &camera,
                            &ts3d.loaded_uris,
                            &config,
                        );

                    // Spawn loads for missing tiles.
                    for req in &traversal.load_requests {
                        if ts3d.pending_uris.contains(&req.content_uri) {
                            continue;
                        }
                        if ts3d.pending_uris.len() >= ts3d.max_concurrent {
                            break;
                        }
                        ts3d.pending_uris.insert(req.content_uri.clone());

                        let client = ts3d.client.clone();
                        let tx = self.tiles3d_tx.clone();
                        let layer_name = ts3d.name.clone();
                        let content_uri = req.content_uri.clone();
                        let access_token = ts3d.access_token.clone();

                        self.rt.spawn(async move {
                            let fetch_result = fetch_tile_content(
                                &client,
                                &content_uri,
                                access_token.as_deref(),
                            )
                            .await;
                            let result = match fetch_result {
                                Ok(bytes) => {
                                    x_planets_tiles::tiles3d::decoder::decode_3d_tile(
                                        &bytes,
                                        &content_uri,
                                    )
                                    .map_err(|e| e.to_string())
                                }
                                Err(e) => Err(e),
                            };
                            let _ = tx.send(Tiles3dMessage::ContentLoaded {
                                layer_name,
                                content_uri,
                                result,
                            });
                        });
                    }

                    // Unload tiles no longer needed.
                    for uri in &traversal.unload_set {
                        ts3d.gpu_tiles.remove(uri);
                        ts3d.loaded_uris.remove(uri);
                    }

                    // Update transforms and render.
                    if !traversal.render_set.is_empty() {
                        let (uniforms, camera_ecef) =
                            x_planets_core::tiles3d_pipeline::build_tiles3d_uniforms(
                                &engine.viewport,
                            );

                        ts3d.update_render_transforms(
                            &gpu.queue,
                            &traversal.render_set,
                            camera_ecef,
                            1.0, // opacity
                        );

                        let models = ts3d.collect_render_models(&traversal.render_set);
                        if let Some(model3d_renderer) = &self.model3d_renderer {
                            model3d_renderer.render_models_with_uniforms(
                                gpu,
                                &view,
                                &uniforms,
                                &models,
                            );
                        }
                    }
                }

                frame.present();

                // ── 8. FPS counter (update window title every 500ms) ──
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
                        || visible.iter().any(|c| !ls.tile_textures.contains(c))
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
            _ => {}
        }
    }
}
