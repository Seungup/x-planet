//! Web application state and render loop.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use x_planets_core::engine::LayerKind;
use x_planets_core::map_controller::LayerStateView;
use x_planets_core::MapController;
use x_planets_render::{TerrainLayerData, TerrainRenderer, TerrainTileData, TileRenderer};
use x_planets_gpu::{GpuContext, GpuTexture, TextureManager};
use x_planets_math::{TileCoord, VisibleTile};
use x_planets_tiles::{RasterTileDecoder, TerrainEncoding, TerrariumDecoder, TileCache, TileDecoder};
use wasm_bindgen_futures::JsFuture;

// ═══════════════════════════════════════════════════════════════════
// Per-layer tile loading result (produced by async fetch, consumed each frame)
// ═══════════════════════════════════════════════════════════════════

enum CompletedTileResult {
    Raster {
        coord: TileCoord,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    },
    Elevation {
        coord: TileCoord,
        elevation: Vec<f32>,
        width: u32,
        height: u32,
    },
}

// ═══════════════════════════════════════════════════════════════════
// WebLayerState — per-layer GPU cache and async loading state
// ═══════════════════════════════════════════════════════════════════

pub(crate) struct WebLayerState {
    pub name: String,
    pub kind: LayerKind,
    pub url_template: String,
    pub tile_textures: TileCache<GpuTexture>,
    /// Elevation data (populated when terrain is enabled on this raster layer).
    pub terrain_data: HashMap<TileCoord, TerrainTileData>,
    pub pending_coords: HashSet<TileCoord>,
    completed_queue: Rc<RefCell<Vec<CompletedTileResult>>>,
    pub failed_queue: Rc<RefCell<Vec<TileCoord>>>,
    pub max_concurrent: usize,
    /// Cached set of available raster tile coords (rebuilt each frame).
    available_coords_cache: HashSet<TileCoord>,
    /// Elevation tile URL template (set when terrain is toggled on).
    pub elevation_url: Option<String>,
    /// Pending elevation tile fetches (separate from raster pending).
    pub pending_elevation_coords: HashSet<TileCoord>,
    pub max_elevation_concurrent: usize,
    /// Failed elevation tile fetches (separate from raster failures).
    pub failed_elevation_queue: Rc<RefCell<Vec<TileCoord>>>,
}

impl WebLayerState {
    fn new(name: String, kind: LayerKind, url_template: String) -> Self {
        Self {
            name,
            kind,
            url_template,
            tile_textures: TileCache::new(256),
            terrain_data: HashMap::new(),
            pending_coords: HashSet::new(),
            completed_queue: Rc::new(RefCell::new(Vec::new())),
            failed_queue: Rc::new(RefCell::new(Vec::new())),
            max_concurrent: 6,
            available_coords_cache: HashSet::new(),
            elevation_url: None,
            pending_elevation_coords: HashSet::new(),
            max_elevation_concurrent: 4,
            failed_elevation_queue: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// Refresh the cached available coords set from tile_textures.
    fn refresh_available_cache(&mut self) {
        self.available_coords_cache = self.tile_textures.keys().copied().collect();
    }
}

impl LayerStateView for WebLayerState {
    fn name(&self) -> &str {
        &self.name
    }

    fn available_raster_coords(&self) -> &HashSet<TileCoord> {
        &self.available_coords_cache
    }

    fn terrain_tile_data(&self, coord: &TileCoord) -> Option<&TerrainTileData> {
        self.terrain_data.get(coord)
    }
}

// ═══════════════════════════════════════════════════════════════════
// WebApp
// ═══════════════════════════════════════════════════════════════════

pub struct WebApp {
    pub gpu: GpuContext,
    pub controller: MapController,
    renderer: TileRenderer,
    pub(crate) terrain_renderer: TerrainRenderer,
    tex_manager: TextureManager,
    pub(crate) canvas: web_sys::HtmlCanvasElement,
    pub dpr: f64,
    last_width: u32,
    last_height: u32,

    // Per-layer state
    layer_states: Vec<WebLayerState>,

    /// Previous frame timestamp (ms) for dt calculation.
    last_frame_ms: Option<f64>,

    /// Registered JS event handlers: event_name → [callback, ...].
    pub(crate) event_handlers: HashMap<String, Vec<js_sys::Function>>,

    /// Whether the app has been destroyed (stops render loop).
    pub(crate) destroyed: bool,
}

impl WebApp {
    pub fn new(
        gpu: GpuContext,
        controller: MapController,
        renderer: TileRenderer,
        terrain_renderer: TerrainRenderer,
        tex_manager: TextureManager,
        canvas: web_sys::HtmlCanvasElement,
        dpr: f64,
    ) -> Self {
        let width = canvas.width();
        let height = canvas.height();

        // Create initial layer states from the engine's layers.
        let layer_states: Vec<WebLayerState> = controller
            .engine
            .layers
            .iter()
            .map(|l| WebLayerState::new(
                l.config.name.clone(),
                l.config.kind.clone(),
                l.config.tile_source_url.clone(),
            ))
            .collect();

        Self {
            gpu,
            controller,
            renderer,
            terrain_renderer,
            tex_manager,
            canvas,
            dpr,
            last_width: width,
            last_height: height,
            layer_states,
            last_frame_ms: None,
            event_handlers: HashMap::new(),
            destroyed: false,
        }
    }

    /// Add a new layer state for a dynamically added layer.
    pub fn add_layer_state(&mut self, name: &str, url: &str) {
        self.layer_states.push(WebLayerState::new(
            name.to_string(),
            LayerKind::Raster,
            url.to_string(),
        ));
    }

    /// Resolve the active projection name to a `ProjectionMode` enum.
    pub(crate) fn resolve_projection_mode(&self) -> x_planets_math::ProjectionMode {
        self.controller.rendering_mode()
    }

    /// Toggle terrain rendering on/off. Returns the new state.
    ///
    /// `url` and `encoding` are optional.  When `None`, defaults to
    /// AWS Terrarium tiles.  Supported encoding strings: "terrarium",
    /// "mapbox", "quantized-mesh".
    ///
    /// No layers are added or removed.  Elevation data is loaded on the
    /// raster imagery layer as a secondary data stream.
    pub fn toggle_terrain_with(
        &mut self,
        url: Option<&str>,
        encoding: Option<&str>,
    ) -> bool {
        let default_url = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
        let url = url.unwrap_or(default_url);
        let encoding = match encoding {
            Some("mapbox") => TerrainEncoding::MapboxRgb,
            Some("quantized-mesh") => TerrainEncoding::QuantizedMesh,
            _ => TerrainEncoding::Terrarium,
        };
        let enabled = self.controller.toggle_terrain(url, encoding);

        if enabled {
            // Set elevation URL on the imagery layer so it starts loading elevation
            let imagery_name = self.controller.terrain_imagery_name()
                .unwrap_or("base").to_string();
            let terrain_url = self.controller.terrain_url()
                .unwrap_or(url).to_string();
            if let Some(ls) = self.layer_states.iter_mut().find(|ls| ls.name == imagery_name) {
                ls.elevation_url = Some(terrain_url);
            }
        } else {
            // Clear elevation data and pending on the imagery layer
            for ls in &mut self.layer_states {
                ls.elevation_url = None;
                ls.terrain_data.clear();
                ls.pending_elevation_coords.clear();
            }
        }

        enabled
    }

    /// Cycle to the next projection and return its name.
    pub fn cycle_projection(&mut self) -> String {
        self.controller.cycle_projection()
    }

    /// Start the requestAnimationFrame render loop.
    pub fn start_render_loop(app: Rc<RefCell<Self>>) {
        let f: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
        let g = f.clone();

        *g.borrow_mut() = Some(Closure::new(move |timestamp_ms: f64| {
            // Render frame (holds borrow_mut, then releases it)
            app.borrow_mut().render_frame(timestamp_ms);

            // Drain events and dispatch to JS callbacks OUTSIDE the borrow.
            // This allows callbacks to call back into the map API (e.g. getZoom()).
            let events_with_handlers = {
                let mut app_ref = app.borrow_mut();
                let events = app_ref.controller.drain_events();
                if events.is_empty() {
                    Vec::new()
                } else {
                    events.into_iter().filter_map(|event| {
                        let name = event.name().to_string();
                        app_ref.event_handlers.get(&name).map(|handlers| {
                            (event, handlers.clone())
                        })
                    }).collect::<Vec<_>>()
                }
            }; // borrow_mut dropped here

            for (event, handlers) in &events_with_handlers {
                let js_data = map_event_to_js(event);
                for handler in handlers {
                    let _ = handler.call1(&JsValue::NULL, &js_data);
                }
            }

            // Check if destroyed (stop loop)
            if app.borrow().destroyed {
                return; // Don't request next frame
            }

            request_animation_frame(f.borrow().as_ref().unwrap());
        }));

        request_animation_frame(g.borrow().as_ref().unwrap());
    }

    fn render_frame(&mut self, timestamp_ms: f64) {
        let now_secs = timestamp_ms / 1000.0;

        // ── 0. Tick animations (smooth zoom, inertia pan) ──
        if let Some(prev_ms) = self.last_frame_ms {
            let dt = ((timestamp_ms - prev_ms) / 1000.0).min(0.1);
            self.controller.tick(dt);
        }
        self.last_frame_ms = Some(timestamp_ms);

        // GC finished tile fades
        self.controller.gc_fades(now_secs);

        // ── 1. Handle resize ──
        self.check_resize();

        // ── 2. Process completed tile fetches → GPU upload ──
        self.upload_completed_tiles(now_secs);

        // ── 3. Request missing tiles for all layers ──
        let visible = self.controller.visible_tiles();
        self.request_tiles_for_all_layers(&visible);

        // ── 4. Refresh available coords caches ──
        for ls in &mut self.layer_states {
            ls.refresh_available_cache();
        }

        // ── 4a. Update tile visibility tracking ──
        {
            // Use the first raster layer's available coords for visibility tracking
            let available: HashSet<TileCoord> = self
                .layer_states
                .iter()
                .find(|ls| matches!(ls.kind, LayerKind::Raster))
                .map(|ls| ls.available_coords_cache.clone())
                .unwrap_or_default();

            self.controller.update_visibility(&visible, &available, now_secs);
        }

        // ── 5. Build render data via MapController ──
        let layer_view_refs: Vec<&dyn LayerStateView> = self
            .layer_states
            .iter()
            .map(|ls| ls as &dyn LayerStateView)
            .collect();

        let render_output = self.controller.build_render_data(
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

        // ── 6. Render ──
        let surface = match self.gpu.surface.as_ref() {
            Some(s) => s,
            None => return,
        };
        let frame = match surface.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost) => {
                self.gpu.resize_surface(self.last_width, self.last_height);
                return;
            }
            Err(_) => return,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mode = self.resolve_projection_mode();

        // Always render raster base first
        self.renderer.render_frame_layered_projected(
            &self.gpu,
            &view,
            &self.controller.engine.viewport,
            &render_output.raster_layers,
            mode,
        );

        // Render terrain layers on top (LoadOp::Load)
        if !render_output.terrain_layers.is_empty() {
            let all_terrain: Vec<TerrainLayerData> = render_output
                .terrain_layers
                .into_iter()
                .chain(render_output.terrain_overlay_layers)
                .collect();
            self.terrain_renderer.render_terrain_layered(
                &self.gpu,
                &view,
                &self.controller.engine.viewport,
                &all_terrain,
                mode,
            );
        }

        frame.present();

        // ── 7. LRU bump visible tiles + base tiles ──
        for ls in &mut self.layer_states {
            if matches!(ls.kind, LayerKind::Raster) {
                for z in 0..=1u8 {
                    let n = 1u32 << z;
                    for y in 0..n {
                        for x in 0..n {
                            let _ = ls.tile_textures.get(&TileCoord::new(z, x, y));
                        }
                    }
                }
                for vt in &visible {
                    let _ = ls.tile_textures.get(&vt.coord);
                }
            }
        }
    }

    fn check_resize(&mut self) {
        let css_w = self.canvas.client_width() as f64;
        let css_h = self.canvas.client_height() as f64;
        let w = (css_w * self.dpr).max(1.0) as u32;
        let h = (css_h * self.dpr).max(1.0) as u32;
        if w != self.last_width || h != self.last_height {
            self.canvas.set_width(w);
            self.canvas.set_height(h);
            self.gpu.resize_surface(w, h);
            self.controller.resize(w, h);
            self.renderer.resize(&self.gpu.device, w, h);
            self.terrain_renderer.resize(&self.gpu.device, w, h);
            self.last_width = w;
            self.last_height = h;
            log::info!("Resized: {}x{}", w, h);
        }
    }

    fn upload_completed_tiles(&mut self, now_secs: f64) {
        for ls in &mut self.layer_states {
            // Drain failed raster fetch notifications
            for coord in ls.failed_queue.borrow_mut().drain(..) {
                ls.pending_coords.remove(&coord);
            }
            // Drain failed elevation fetch notifications
            for coord in ls.failed_elevation_queue.borrow_mut().drain(..) {
                ls.pending_elevation_coords.remove(&coord);
            }

            // Drain completed results
            let completed: Vec<CompletedTileResult> =
                ls.completed_queue.borrow_mut().drain(..).collect();
            for result in completed {
                match result {
                    CompletedTileResult::Raster { coord, width, height, pixels } => {
                        ls.pending_coords.remove(&coord);
                        let label = format!("tile-{}/{}/{}", coord.z, coord.x, coord.y);
                        let gpu_tex = self.tex_manager.create_rgba_texture(
                            &self.gpu.device,
                            &self.gpu.queue,
                            &label,
                            width,
                            height,
                            &pixels,
                        );
                        ls.tile_textures.insert(coord, gpu_tex);
                        self.controller.register_tile_loaded(coord, now_secs);
                    }
                    CompletedTileResult::Elevation { coord, elevation, width, height } => {
                        ls.pending_elevation_coords.remove(&coord);
                        ls.terrain_data.insert(
                            coord,
                            TerrainTileData::Heightmap { elevation, width, height },
                        );
                    }
                }
            }
        }
    }

    /// Request missing tiles for all layers.
    fn request_tiles_for_all_layers(&mut self, visible: &[VisibleTile]) {
        let camera_center = x_planets_math::geo_to_mercator(&self.controller.engine.viewport.center);

        for ls in &mut self.layer_states {
            let visible_set: HashSet<TileCoord> = visible.iter().map(|vt| vt.coord).collect();
            ls.pending_coords.retain(|c| c.z <= 1 || visible_set.contains(c));
            ls.pending_elevation_coords.retain(|c| visible_set.contains(c));

            match &ls.kind {
                LayerKind::Raster => {
                    request_raster_tiles(ls, visible, camera_center);
                    // Also request elevation tiles if terrain is enabled on this layer
                    if let Some(elev_url) = ls.elevation_url.clone() {
                        request_elevation_tiles(ls, visible, camera_center, &elev_url);
                    }
                }
                LayerKind::Terrain { .. } => {
                    // Config-file terrain layers (backward compat)
                    request_elevation_tiles(ls, visible, camera_center, &ls.url_template.clone());
                }
                LayerKind::Tiles3d => {}
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tile loading — free functions to avoid borrow issues
// ═══════════════════════════════════════════════════════════════════

fn request_raster_tiles(
    ls: &mut WebLayerState,
    visible: &[VisibleTile],
    camera_center: x_planets_math::DVec2,
) {
    let mut missing: Vec<(TileCoord, f64)> = Vec::new();

    // Always eagerly load z=0 and z=1 base tiles
    for z in 0..=1u8 {
        let n = 1u32 << z;
        for y in 0..n {
            for x in 0..n {
                let coord = TileCoord::new(z, x, y);
                if !ls.tile_textures.contains(&coord) && !ls.pending_coords.contains(&coord) {
                    missing.push((coord, 0.0));
                }
            }
        }
    }

    // Deduplicate and sort by distance
    let mut seen = HashSet::new();
    let mut visible_missing: Vec<(TileCoord, f64)> = visible
        .iter()
        .filter(|vt| {
            !ls.tile_textures.contains(&vt.coord)
                && !ls.pending_coords.contains(&vt.coord)
                && seen.insert(vt.coord)
        })
        .map(|vt| {
            let center = vt.display_mercator_center();
            let dist = (center - camera_center).length();
            (vt.coord, dist)
        })
        .collect();
    visible_missing.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    missing.extend(visible_missing);

    let slots = ls.max_concurrent.saturating_sub(ls.pending_coords.len());
    for (coord, _) in missing.into_iter().take(slots) {
        ls.pending_coords.insert(coord);
        let queue = Rc::clone(&ls.completed_queue);
        let failed = Rc::clone(&ls.failed_queue);
        let url = tile_url(&ls.url_template, &coord);

        wasm_bindgen_futures::spawn_local(async move {
            match fetch_bytes(&url).await {
                Ok(bytes) => {
                    let decoder = RasterTileDecoder::default();
                    match decoder.decode(coord, &bytes).await {
                        Ok(decoded) => {
                            queue.borrow_mut().push(CompletedTileResult::Raster {
                                coord,
                                width: decoded.width,
                                height: decoded.height,
                                pixels: decoded.pixels,
                            });
                        }
                        Err(e) => {
                            log::warn!("Decode {}: {}", coord, e);
                            failed.borrow_mut().push(coord);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Fetch {}: {}", coord, e);
                    failed.borrow_mut().push(coord);
                }
            }
        });
    }
}

fn request_elevation_tiles(
    ls: &mut WebLayerState,
    visible: &[VisibleTile],
    camera_center: x_planets_math::DVec2,
    elev_url_template: &str,
) {
    let mut missing: Vec<(TileCoord, f64)> = Vec::new();

    let mut seen = HashSet::new();
    let mut visible_missing: Vec<(TileCoord, f64)> = visible
        .iter()
        .filter(|vt| {
            !ls.terrain_data.contains_key(&vt.coord)
                && !ls.pending_elevation_coords.contains(&vt.coord)
                && seen.insert(vt.coord)
        })
        .map(|vt| {
            let center = vt.display_mercator_center();
            let dist = (center - camera_center).length();
            (vt.coord, dist)
        })
        .collect();
    visible_missing.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    missing.extend(visible_missing);

    let slots = ls.max_elevation_concurrent.saturating_sub(ls.pending_elevation_coords.len());
    for (coord, _) in missing.into_iter().take(slots) {
        ls.pending_elevation_coords.insert(coord);
        let queue = Rc::clone(&ls.completed_queue);
        let failed = Rc::clone(&ls.failed_elevation_queue);
        let url = tile_url(elev_url_template, &coord);

        wasm_bindgen_futures::spawn_local(async move {
            match fetch_bytes(&url).await {
                Ok(bytes) => {
                    let decoder = TerrariumDecoder;
                    match decoder.decode(coord, &bytes).await {
                        Ok(decoded) => {
                            queue.borrow_mut().push(CompletedTileResult::Elevation {
                                coord,
                                elevation: decoded.elevation,
                                width: decoded.width,
                                height: decoded.height,
                            });
                        }
                        Err(e) => {
                            log::warn!("Elev decode {}: {}", coord, e);
                            failed.borrow_mut().push(coord);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Elev fetch {}: {}", coord, e);
                    failed.borrow_mut().push(coord);
                }
            }
        });
    }
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn tile_url(template: &str, coord: &TileCoord) -> String {
    template
        .replace("{z}", &coord.z.to_string())
        .replace("{x}", &coord.x.to_string())
        .replace("{y}", &coord.y.to_string())
}

async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or("No window")?;
    let resp = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|e| format!("{:?}", e))?;
    let resp: web_sys::Response = resp
        .dyn_into()
        .map_err(|_| "Response cast failed".to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let buf = JsFuture::from(
        resp.array_buffer()
            .map_err(|e| format!("{:?}", e))?,
    )
    .await
    .map_err(|e| format!("{:?}", e))?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

fn request_animation_frame(f: &Closure<dyn FnMut(f64)>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}

/// Convert a [`MapEvent`] to a JS object for dispatch to callbacks.
fn map_event_to_js(event: &x_planets_core::map_controller::MapEvent) -> JsValue {
    use x_planets_core::map_controller::MapEvent;
    let obj = js_sys::Object::new();
    match event {
        MapEvent::Move { lat, lon } => {
            let _ = js_sys::Reflect::set(&obj, &"lat".into(), &(*lat).into());
            let _ = js_sys::Reflect::set(&obj, &"lon".into(), &(*lon).into());
        }
        MapEvent::Zoom { zoom } => {
            let _ = js_sys::Reflect::set(&obj, &"zoom".into(), &(*zoom).into());
        }
        MapEvent::Pitch { pitch } => {
            let _ = js_sys::Reflect::set(&obj, &"pitch".into(), &(*pitch).into());
        }
        MapEvent::Bearing { bearing } => {
            let _ = js_sys::Reflect::set(&obj, &"bearing".into(), &(*bearing).into());
        }
        MapEvent::Click { lat, lon, x, y } => {
            let _ = js_sys::Reflect::set(&obj, &"lat".into(), &(*lat).into());
            let _ = js_sys::Reflect::set(&obj, &"lon".into(), &(*lon).into());
            let _ = js_sys::Reflect::set(&obj, &"x".into(), &(*x).into());
            let _ = js_sys::Reflect::set(&obj, &"y".into(), &(*y).into());
        }
        MapEvent::MoveEnd | MapEvent::ZoomEnd => {}
    }
    obj.into()
}
