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
// TileJSON support
// ═══════════════════════════════════════════════════════════════════

/// Minimal TileJSON 2.x/3.x metadata — only the fields we need.
///
/// Spec: <https://github.com/mapbox/tilejson-spec>
#[derive(serde::Deserialize)]
struct TileJson {
    /// One or more tile URL templates (e.g. `"https://…/{z}/{x}/{y}.webp"`).
    tiles: Vec<String>,
    #[serde(default)]
    minzoom: Option<u8>,
    #[serde(default)]
    maxzoom: Option<u8>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    scheme: Option<String>,
    /// Tile pixel density (e.g. `"1.000000"` = 256×256, `"2.000000"` = 512×512).
    /// MapTiler extension; not in the original TileJSON spec.
    #[serde(default)]
    scale: Option<String>,
    /// Tile data format, e.g. `"quantized-mesh-1.0"`, `"terrarium"`, `"webp"`, `"png"`.
    /// Used to auto-detect terrain encoding without requiring explicit config.
    #[serde(default)]
    format: Option<String>,
}

/// Resolved metadata from a TileJSON endpoint.
struct TileJsonMeta {
    /// The first tile URL template from the `tiles` array.
    tile_url: String,
    /// Whether the tile scheme is TMS (y-axis flipped).
    tms: bool,
    /// Minimum zoom level served by the source.
    min_zoom: Option<u8>,
    /// Maximum zoom level served by the source.
    max_zoom: Option<u8>,
    /// Tile pixel density (1.0 = 256px, 2.0 = 512px).
    scale: f32,
    /// Terrain encoding auto-detected from the TileJSON `format` field.
    /// `None` if the format is not a recognized terrain format (raster).
    detected_encoding: Option<TerrainEncoding>,
}

/// Returns `true` if the URL looks like a TileJSON endpoint
/// (ends in `.json` but is not a 3D Tiles `tileset.json`).
fn is_tilejson_url(url: &str) -> bool {
    let lower = url.split('?').next().unwrap_or(url).to_ascii_lowercase();
    lower.ends_with(".json")
        && !lower.ends_with("tileset.json")
}

/// Fetch a TileJSON endpoint and extract tile URL template + metadata.
///
/// Returns [`TileJsonMeta`] with the resolved URL, TMS flag, zoom range, and scale.
async fn resolve_tilejson(
    client: &reqwest::Client,
    url: &str,
) -> Result<TileJsonMeta, String> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("TileJSON fetch failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("TileJSON HTTP {}", resp.status()));
    }
    let json: TileJson = resp
        .json()
        .await
        .map_err(|e| format!("TileJSON parse failed: {e}"))?;
    let tile_url = json
        .tiles
        .into_iter()
        .next()
        .ok_or_else(|| "TileJSON has empty `tiles` array".to_string())?;
    let tms = json
        .scheme
        .as_deref()
        .map(|s| s.eq_ignore_ascii_case("tms"))
        .unwrap_or(false);
    let scale = json
        .scale
        .as_deref()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);

    // Auto-detect terrain encoding from TileJSON `format` field.
    let detected_encoding = match json.format.as_deref() {
        Some(f) if f.starts_with("quantized-mesh") => Some(TerrainEncoding::QuantizedMesh),
        Some("terrarium") => Some(TerrainEncoding::Terrarium),
        _ => None,
    };

    log::info!(
        "TileJSON resolved: \"{}\" (zoom {}-{}, scale={}x, tms={}, format={:?})",
        json.name.as_deref().unwrap_or("(unnamed)"),
        json.minzoom.unwrap_or(0),
        json.maxzoom.unwrap_or(22),
        scale,
        tms,
        json.format.as_deref().unwrap_or("(none)"),
    );
    Ok(TileJsonMeta {
        tile_url,
        tms,
        min_zoom: json.minzoom,
        max_zoom: json.maxzoom,
        scale,
        detected_encoding,
    })
}

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
    /// Quantized Mesh 1.0 terrain tile (pre-built triangle mesh).
    QuantizedMesh(x_planets_tiles::DecodedQuantizedMesh),
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
    /// LRU-evicted alongside tile_textures to prevent unbounded memory growth.
    terrain_data: TileCache<TerrainTileData>,
    /// Cooldown for failed tiles: don't retry until the Instant has passed.
    /// Prevents infinite retry loops when the server returns 429 / transient errors.
    failed_cooldowns: HashMap<TileCoord, Instant>,
    /// Minimum zoom level served by the tile source (from TileJSON `minzoom`).
    min_zoom: u8,
    /// Maximum zoom level served by the tile source (from TileJSON `maxzoom`).
    /// Tiles beyond this zoom are never requested; the fallback system
    /// renders them with parent tiles at `max_zoom`.
    max_zoom: u8,
    /// Tile pixel density (1.0 = 256px, 2.0 = 512px).
    /// Parsed from TileJSON `scale` field.  Reserved for future LOD calculations.
    #[allow(dead_code)]
    tile_scale: f32,
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
        // Resolve TileJSON endpoints (URLs ending in `.json`) before creating
        // tile sources.  This fetches the metadata JSON at startup and extracts
        // the actual tile URL template from the `tiles` array.
        let tilejson_client = reqwest::Client::builder()
            .user_agent("x-planets/0.1")
            .build()
            .expect("Failed to create HTTP client for TileJSON");

        self.layer_states = engine
            .layers
            .iter()
            .filter(|layer| !matches!(layer.config.kind, LayerKind::Tiles3d))
            .map(|layer| {
                let cfg = &layer.config;
                let meta = if is_tilejson_url(&cfg.tile_source_url) {
                    log::info!(
                        "Layer '{}': resolving TileJSON → {}",
                        cfg.name, cfg.tile_source_url,
                    );
                    match self.rt.block_on(resolve_tilejson(&tilejson_client, &cfg.tile_source_url)) {
                        Ok(m) => {
                            log::info!(
                                "  → resolved to: {}",
                                if m.tile_url.len() > 80 { format!("{}…", &m.tile_url[..80]) } else { m.tile_url.clone() },
                            );
                            m
                        }
                        Err(e) => {
                            log::warn!("  → TileJSON resolution failed, using URL as-is: {}", e);
                            TileJsonMeta {
                                tile_url: cfg.tile_source_url.clone(),
                                tms: false,
                                min_zoom: None,
                                max_zoom: None,
                                scale: 1.0,
                                detected_encoding: None,
                            }
                        }
                    }
                } else {
                    TileJsonMeta {
                        tile_url: cfg.tile_source_url.clone(),
                        tms: false,
                        min_zoom: None,
                        max_zoom: None,
                        scale: 1.0,
                        detected_encoding: None,
                    }
                };

                // If TileJSON detected a terrain encoding (e.g. "quantized-mesh-1.0"),
                // override the encoding in the layer kind so explicit config is not required.
                let kind = if let (LayerKind::Terrain { ref imagery_layer, .. }, Some(enc)) =
                    (&cfg.kind, meta.detected_encoding)
                {
                    eprintln!(
                        "[x-planets] Layer '{}': TileJSON auto-detected encoding → {:?}",
                        cfg.name, enc
                    );
                    log::info!(
                        "  → auto-detected terrain encoding: {:?}",
                        enc
                    );
                    LayerKind::Terrain {
                        imagery_layer: imagery_layer.clone(),
                        encoding: enc,
                    }
                } else {
                    cfg.kind.clone()
                };

                log::info!(
                    "Creating layer '{}' → {} (zoom {}-{}, scale={}x, max_concurrent={}, max_cached={})",
                    cfg.name, meta.tile_url,
                    meta.min_zoom.unwrap_or(0), meta.max_zoom.unwrap_or(22),
                    meta.scale,
                    cfg.max_concurrent_loads, cfg.max_cached_tiles,
                );
                let source = NativeTileSource::new(&meta.tile_url).with_tms(meta.tms);
                NativeLayerState {
                    name: cfg.name.clone(),
                    kind,
                    tile_source: Arc::new(source),
                    tile_textures: TileCache::new(cfg.max_cached_tiles),
                    tile_loader: TileLoader::new(cfg.max_concurrent_loads),
                    pending_coords: HashSet::new(),
                    terrain_data: TileCache::new(cfg.max_cached_tiles),
                    failed_cooldowns: HashMap::new(),
                    min_zoom: meta.min_zoom.unwrap_or(0),
                    max_zoom: meta.max_zoom.unwrap_or(22),
                    tile_scale: meta.scale,
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
                let visible_set: HashSet<TileCoord> = visible.iter().copied().collect();

                for ls in &mut self.layer_states {
                    // ── Abort stale requests ──
                    // Clear the priority queue every frame.  Tiles that were queued
                    // but never dequeued (max_concurrent reached) are NOT in
                    // pending_coords, so they'll be naturally re-enqueued below
                    // with fresh priorities.
                    ls.tile_loader.clear();

                    // Build the "needed" set: visible tiles + uncached ancestors
                    // that serve as fallback coverage.  Only abort in-flight
                    // tiles NOT in this set.  This prevents the old zoom_diff
                    // heuristic from killing ancestor tiles spawned by
                    // parent-first loading (which caused infinite re-spawn loops).
                    //
                    // For over-zoomed tiles (z > max_zoom), the max_zoom ancestor
                    // is the deepest tile we can fetch, so include it in the needed set.
                    let mut needed_coords: HashSet<TileCoord> = visible_set.clone();
                    for &coord in &visible {
                        // If tile exceeds max_zoom, start the ancestor chain
                        // from the corresponding tile AT max_zoom.
                        let start = if coord.z > ls.max_zoom {
                            let dz = coord.z - ls.max_zoom;
                            let clamped = TileCoord::new(
                                ls.max_zoom,
                                coord.x >> dz,
                                coord.y >> dz,
                            );
                            needed_coords.insert(clamped);
                            clamped.parent()
                        } else {
                            coord.parent()
                        };
                        let mut cur = start;
                        while let Some(p) = cur {
                            if ls.tile_textures.contains(&p) {
                                // Cached ancestor found — it and everything
                                // above it are already available.
                                needed_coords.insert(p);
                                break;
                            }
                            needed_coords.insert(p);
                            cur = p.parent();
                        }
                    }

                    let stale_coords: Vec<TileCoord> = ls.pending_coords
                        .iter()
                        .filter(|c| !needed_coords.contains(c))
                        .copied()
                        .collect();
                    for coord in stale_coords {
                        ls.pending_coords.remove(&coord);
                        ls.tile_loader.complete(); // free concurrency slot
                    }

                    // GC expired cooldowns (once per frame is cheap).
                    ls.failed_cooldowns.retain(|_, expire| now < *expire);

                    // ── Parent-first loading ──
                    // For each visible tile missing a cached ancestor, enqueue
                    // the NEAREST uncached ancestor (one level at a time).
                    // Once that ancestor loads, next frame discovers the next
                    // one.  This avoids flooding the queue with deep ancestor
                    // chains (z=0..z=14) that block visible tile loading.
                    //
                    // Ancestors share a reserved portion of concurrency:
                    //   2 out of max_concurrent slots.  The rest go to
                    //   visible tiles so current-view loading isn't starved.
                    {
                        let max_ancestor_slots = 2usize;
                        let ancestor_in_flight = ls.pending_coords
                            .iter()
                            .filter(|c| !visible_set.contains(c))
                            .count();

                        if ancestor_in_flight < max_ancestor_slots {
                            let mut ancestor_enqueued: HashSet<TileCoord> = HashSet::new();
                            let mut budget = max_ancestor_slots - ancestor_in_flight;
                            for &coord in &visible {
                                if budget == 0 { break; }
                                // Start from the closest fetchable ancestor
                                // (skip children beyond max_zoom).
                                let start = if coord.z > ls.max_zoom {
                                    let dz = coord.z - ls.max_zoom;
                                    // The max_zoom tile covering this visible tile
                                    // might itself be needed — enqueue it as a
                                    // visible-priority tile, not just an ancestor.
                                    let clamped = TileCoord::new(
                                        ls.max_zoom,
                                        coord.x >> dz,
                                        coord.y >> dz,
                                    );
                                    Some(clamped)
                                } else {
                                    coord.parent()
                                };
                                let mut cur = start;
                                while let Some(p) = cur {
                                    if p.z < ls.min_zoom { break; }
                                    if ls.tile_textures.contains(&p) {
                                        break; // ancestor cached, chain OK
                                    }
                                    if !ls.pending_coords.contains(&p)
                                        && !ls.failed_cooldowns.contains_key(&p)
                                        && ancestor_enqueued.insert(p)
                                        && !visible_set.contains(&p)
                                    {
                                        // Enqueue the nearest uncached ancestor.
                                        // Priority: slightly better than the
                                        // worst visible tile so it loads soon
                                        // but doesn't starve visible tiles.
                                        let p_center = p.mercator_center();
                                        let p_dist = (p_center - camera_center)
                                            .length() as f32;
                                        ls.tile_loader.enqueue(TileRequest {
                                            coord: p,
                                            priority: p_dist * 0.8,
                                        });
                                        budget = budget.saturating_sub(1);
                                        break; // only nearest ancestor per visible tile
                                    }
                                    cur = p.parent();
                                }
                            }
                        }
                    }

                    // Enqueue visible tiles that are not yet loaded or in-flight.
                    // Priority: distance from camera × fallback penalty.
                    // Tiles with no/distant fallback texture are prioritized
                    // (lower value = higher priority in the min-heap).
                    //
                    // Zoom clamping: tiles beyond `max_zoom` are never requested.
                    // The fallback system renders them with parent tiles at `max_zoom`.
                    // Tiles below `min_zoom` are also skipped (rare edge case).
                    //
                    // Over-zoom: for visible tiles at z > max_zoom, we enqueue the
                    // corresponding tile at max_zoom so the fallback system can use
                    // it.  Multiple over-zoomed children may map to the SAME max_zoom
                    // tile, so we deduplicate.
                    let mut overzoom_enqueued: HashSet<TileCoord> = HashSet::new();
                    for &coord in &visible {
                        if coord.z < ls.min_zoom {
                            continue;
                        }
                        // Clamp over-zoomed tiles: enqueue the deepest fetchable tile.
                        let fetch_coord = if coord.z > ls.max_zoom {
                            let dz = coord.z - ls.max_zoom;
                            let clamped = TileCoord::new(
                                ls.max_zoom,
                                coord.x >> dz,
                                coord.y >> dz,
                            );
                            if !overzoom_enqueued.insert(clamped) {
                                continue; // already enqueued this max_zoom tile
                            }
                            clamped
                        } else {
                            coord
                        };

                        if ls.tile_textures.contains(&fetch_coord)
                            || ls.pending_coords.contains(&fetch_coord)
                            || ls.failed_cooldowns.contains_key(&fetch_coord)
                        {
                            continue;
                        }
                        let tile_center = fetch_coord.mercator_center();
                        let dist = (tile_center - camera_center).length() as f32;

                        // Fallback depth: how many zoom levels up to the nearest
                        // cached ancestor?  0 = no ancestor at all (blank tile!).
                        let fallback_depth = {
                            let mut depth = 0u32;
                            let mut cur = fetch_coord.parent();
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
                            coord: fetch_coord,
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
                                    Ok(bytes) => match enc {
                                        TerrainEncoding::MapboxRgb => {
                                            match TerrainRgbDecoder.decode(req.coord, &bytes).await {
                                                Ok(d) => Ok(TileResult::Terrain(d)),
                                                Err(e) => Err((req.coord, e.to_string())),
                                            }
                                        }
                                        TerrainEncoding::Terrarium => {
                                            match TerrariumDecoder.decode(req.coord, &bytes).await {
                                                Ok(d) => Ok(TileResult::Terrain(d)),
                                                Err(e) => Err((req.coord, e.to_string())),
                                            }
                                        }
                                        TerrainEncoding::QuantizedMesh => {
                                            match x_planets_tiles::parse_quantized_mesh(req.coord, &bytes) {
                                                Ok(qm) => Ok(TileResult::QuantizedMesh(qm)),
                                                Err(e) => Err((req.coord, e.to_string())),
                                            }
                                        }
                                    },
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

                // 4b. Poll completed tiles & create GPU textures (dispatched by layer name).
                // Cap per frame to avoid frame-time spikes when many tiles arrive at once
                // (each raster tile = ~256KB GPU upload, each terrain mesh = CPU build).
                // Remaining tiles stay in the channel and are processed next frame.
                const MAX_TILES_PER_FRAME: usize = 4;
                let mut tiles_this_frame = 0;
                while tiles_this_frame < MAX_TILES_PER_FRAME {
                    let msg = match self.tile_rx.try_recv() {
                        Ok(m) => m,
                        Err(_) => break,
                    };
                    tiles_this_frame += 1;
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
                                // Note: we do NOT insert terrain tile_fade_start here.
                                // Terrain imagery comes from the companion raster layer,
                                // whose fade_start is recorded when the raster tile loads.
                                // Inserting here would overwrite the imagery fade_start,
                                // causing incorrect cross-fade timing.

                                // Store elevation data on CPU for mesh generation
                                ls.terrain_data.insert(
                                    decoded.coord,
                                    TerrainTileData::Heightmap {
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
                            Ok(TileResult::QuantizedMesh(qm)) => {
                                if ls.pending_coords.remove(&qm.coord) {
                                    ls.tile_loader.complete();
                                }
                                log::info!(
                                    "[{}] QM tile loaded: z={} x={} y={} ({} verts, {} tris, h=[{:.0}..{:.0}]m)",
                                    ls.name,
                                    qm.coord.z, qm.coord.x, qm.coord.y,
                                    qm.u.len(),
                                    qm.indices.len() / 3,
                                    qm.header.min_height, qm.header.max_height,
                                );
                                // Convert raw QM data to pre-built vertices/indices.
                                // Heights stay in metres; height_scale applied in renderer
                                // so exaggeration changes work without re-fetching.
                                let (vertices, indices) =
                                    x_planets_core::pipeline::build_terrain_mesh_from_qm(
                                        &qm.coord, &qm,
                                    );
                                ls.terrain_data.insert(
                                    qm.coord,
                                    TerrainTileData::PrebuiltMesh { vertices, indices },
                                );
                                // Placeholder texture (imagery from companion raster layer)
                                let tex = tex_mgr.create_rgba_texture(
                                    &gpu.device, &gpu.queue,
                                    &format!("{}-qm-{}-{}-{}", ls.name,
                                        qm.coord.z, qm.coord.x, qm.coord.y),
                                    1, 1, &[128, 128, 128, 255],
                                );
                                ls.tile_textures.insert(qm.coord, tex);
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
                                // Always print tile failures to stderr so the user can
                                // see errors even without RUST_LOG=debug.
                                eprintln!(
                                    "[x-planets] TILE FAIL [{}] z={} x={} y={} (retry {}s): {}",
                                    ls.name, coord.z, coord.x, coord.y, cooldown_secs, err_msg,
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
                // Bump visible tiles AND their fallback ancestors to prevent
                // parent tiles from being evicted while still needed as fallback
                // coverage for unloaded children.
                for ls in &mut self.layer_states {
                    for &coord in &visible {
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
                let engine = self.engine.as_ref().unwrap();
                let mut render_layers: Vec<RenderLayerData> = Vec::new();
                let mut terrain_layers: Vec<TerrainLayerData> = Vec::new();
                let mut terrain_overlay_layers: Vec<TerrainLayerData> = Vec::new();

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

                                let available: HashSet<TileCoord> =
                                    ls.tile_textures.keys().copied().collect();

                                // Build texture view map
                                let texture_views: HashMap<TileCoord, &wgpu::TextureView> = ls
                                    .tile_textures
                                    .iter()
                                    .map(|(k, v)| (*k, &v.view))
                                    .collect();

                                // ── Cross-fade: identify tiles transitioning parent → child ──
                                //
                                // During the fade-in period, exclude child tiles from
                                // the "available" set so resolve_fallbacks picks the
                                // parent texture as the base.  The child tiles are then
                                // rendered as a separate overlay layer at fade opacity.
                                // Alpha blending: output = child×t + parent×(1−t).
                                let mut available_for_base = available.clone();
                                let mut crossfade_tiles: Vec<(TileCoord, f32)> = Vec::new();

                                for &coord in &visible {
                                    if !available.contains(&coord) { continue; }
                                    if let Some(&start) =
                                        self.anim.tile_fade_start.get(&coord)
                                    {
                                        let elapsed = now.duration_since(start).as_secs_f64();
                                        if elapsed < FADE_DURATION {
                                            let has_parent = {
                                                let mut c = coord.parent();
                                                let mut found = false;
                                                while let Some(p) = c {
                                                    if available.contains(&p) {
                                                        found = true;
                                                        break;
                                                    }
                                                    c = p.parent();
                                                }
                                                found
                                            };
                                            if has_parent {
                                                available_for_base.remove(&coord);
                                                let fade_t = ((elapsed / FADE_DURATION) as f32)
                                                    .max(1.0 / 60.0)
                                                    .min(1.0);
                                                crossfade_tiles.push((coord, fade_t));
                                            }
                                        }
                                    }
                                }

                                let renderable =
                                    x_planets_core::pipeline::resolve_fallbacks(
                                        &visible, &available_for_base,
                                    );

                                // Opacity overrides: only for tiles with NO parent
                                // coverage (first-time appearance, fade from zero).
                                let mut tile_opacity_overrides = HashMap::new();
                                for rt in &renderable {
                                    if rt.texture_coord != rt.coord {
                                        continue; // using parent fallback → full opacity
                                    }
                                    if let Some(&start) =
                                        self.anim.tile_fade_start.get(&rt.coord)
                                    {
                                        let elapsed = now.duration_since(start).as_secs_f64();
                                        if elapsed < FADE_DURATION {
                                            // No parent coverage → fade from near-zero
                                            let t = ((elapsed / FADE_DURATION) as f32)
                                                .max(1.0 / 60.0)
                                                .min(1.0);
                                            tile_opacity_overrides.insert(
                                                rt.coord,
                                                layer.config.opacity * t,
                                            );
                                        }
                                    }
                                }

                                // Base layer: parent fallbacks for crossfading tiles,
                                // own textures for tiles that finished fading or have
                                // no parent coverage.
                                render_layers.push(RenderLayerData {
                                    name: &layer.config.name,
                                    opacity: layer.config.opacity,
                                    tiles: renderable,
                                    texture_views: texture_views.clone(),
                                    tile_opacity_overrides,
                                });

                                // Cross-fade overlay: child tiles fading in over parent.
                                // Rendered as a separate layer — each layer gets its own
                                // render pass with cleared depth, so the overlay composites
                                // correctly via alpha blending.
                                if !crossfade_tiles.is_empty() {
                                    let mut overlay_tiles = Vec::new();
                                    let mut overlay_opacity = HashMap::new();
                                    for &(coord, fade_t) in &crossfade_tiles {
                                        overlay_tiles.push(
                                            x_planets_core::pipeline::RenderableTile {
                                                coord,
                                                texture_coord: coord,
                                                uv_rect: [0.0, 0.0, 1.0, 1.0],
                                            },
                                        );
                                        overlay_opacity.insert(
                                            coord,
                                            layer.config.opacity * fade_t,
                                        );
                                    }
                                    render_layers.push(RenderLayerData {
                                        name: "crossfade-overlay",
                                        opacity: layer.config.opacity,
                                        tiles: overlay_tiles,
                                        texture_views,
                                        tile_opacity_overrides: overlay_opacity,
                                    });
                                }
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

                                if imagery_ls.is_none() {
                                    eprintln!(
                                        "[x-planets] TERRAIN WARN: layer '{}' references imagery_layer '{}' \
                                         which was not found — terrain cannot render without it.",
                                        layer.config.name, imagery_layer
                                    );
                                }
                                if let Some(img_ls) = imagery_ls {
                                    let available: HashSet<TileCoord> =
                                        img_ls.tile_textures.keys().copied().collect();

                                    // Imagery texture views from companion layer
                                    let imagery_views: HashMap<TileCoord, &wgpu::TextureView> =
                                        img_ls
                                            .tile_textures
                                            .iter()
                                            .map(|(k, v)| (*k, &v.view))
                                            .collect();

                                    // ── Cross-fade for terrain imagery ──
                                    // Same principle as raster: exclude fading child
                                    // tiles so the base pass uses parent fallback,
                                    // then render child as overlay at fade opacity.
                                    let mut available_for_base = available.clone();
                                    let mut crossfade_tiles: Vec<(TileCoord, f32)> = Vec::new();

                                    for &coord in &visible {
                                        if !available.contains(&coord) { continue; }
                                        if let Some(&start) =
                                            self.anim.tile_fade_start.get(&coord)
                                        {
                                            let elapsed = now.duration_since(start).as_secs_f64();
                                            if elapsed < FADE_DURATION {
                                                let has_parent = {
                                                    let mut c = coord.parent();
                                                    let mut found = false;
                                                    while let Some(p) = c {
                                                        if available.contains(&p) {
                                                            found = true;
                                                            break;
                                                        }
                                                        c = p.parent();
                                                    }
                                                    found
                                                };
                                                if has_parent {
                                                    available_for_base.remove(&coord);
                                                    let fade_t = ((elapsed / FADE_DURATION) as f32)
                                                        .max(1.0 / 60.0)
                                                        .min(1.0);
                                                    crossfade_tiles.push((coord, fade_t));
                                                }
                                            }
                                        }
                                    }

                                    let renderable = x_planets_core::pipeline::resolve_fallbacks(
                                        &visible, &available_for_base,
                                    );

                                    // Elevation data with parent fallback.
                                    // Include coords for both base and overlay tiles.
                                    let mut elevation_data: HashMap<TileCoord, (&TerrainTileData, TileCoord)> =
                                        HashMap::new();
                                    let all_needed: HashSet<TileCoord> = renderable
                                        .iter()
                                        .map(|rt| rt.coord)
                                        .chain(crossfade_tiles.iter().map(|&(c, _)| c))
                                        .collect();
                                    for &coord in &all_needed {
                                        let mut c = Some(coord);
                                        while let Some(candidate) = c {
                                            if let Some(data) = ls.terrain_data.peek(&candidate) {
                                                // Accept any elevation data including parent
                                                // PrebuiltMesh tiles.  When a parent QM mesh is
                                                // used for a child coord the renderer generates a
                                                // flat placeholder so the imagery is visible
                                                // immediately during the parent-first loading phase.
                                                elevation_data.insert(coord, (data, candidate));
                                                break;
                                            }
                                            c = candidate.parent();
                                        }
                                    }

                                    // Debug: report terrain pipeline state on every frame
                                    // so we can diagnose why rendering stops.
                                    eprintln!(
                                        "[terrain-pipeline] '{}': \
                                         img_available={} renderable={} elev_data={} \
                                         terrain_data_size=? crossfade={}",
                                        layer.config.name,
                                        available.len(),
                                        renderable.len(),
                                        elevation_data.len(),
                                        crossfade_tiles.len(),
                                    );

                                    // Base terrain layer (parent fallback imagery for
                                    // crossfading tiles, own imagery for stable tiles).
                                    terrain_layers.push(TerrainLayerData {
                                        name: &layer.config.name,
                                        opacity: layer.config.opacity,
                                        tiles: renderable,
                                        imagery_views: imagery_views.clone(),
                                        elevation_data: elevation_data.clone(),
                                        tile_opacity_overrides: HashMap::new(),
                                    });

                                    // Cross-fade overlay: child imagery fading in.
                                    // Must be rendered in a SEPARATE render_terrain_layered
                                    // call because the mesh cache shares uniform buffers
                                    // per coord — a single call would overwrite the base
                                    // pass uniforms before submission.
                                    if !crossfade_tiles.is_empty() {
                                        let overlay_tiles: Vec<_> = crossfade_tiles
                                            .iter()
                                            .map(|&(coord, _)| {
                                                x_planets_core::pipeline::RenderableTile {
                                                    coord,
                                                    texture_coord: coord,
                                                    uv_rect: [0.0, 0.0, 1.0, 1.0],
                                                }
                                            })
                                            .collect();
                                        let mut overlay_opacity = HashMap::new();
                                        for &(coord, fade_t) in &crossfade_tiles {
                                            overlay_opacity.insert(
                                                coord,
                                                layer.config.opacity * fade_t,
                                            );
                                        }
                                        let overlay_elev: HashMap<TileCoord, (&TerrainTileData, TileCoord)> =
                                            crossfade_tiles
                                                .iter()
                                                .filter_map(|&(coord, _)| {
                                                    elevation_data.get(&coord).map(|&v| (coord, v))
                                                })
                                                .collect();
                                        terrain_overlay_layers.push(TerrainLayerData {
                                            name: "terrain-crossfade",
                                            opacity: layer.config.opacity,
                                            tiles: overlay_tiles,
                                            imagery_views,
                                            elevation_data: overlay_elev,
                                            tile_opacity_overrides: overlay_opacity,
                                        });
                                    }
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
