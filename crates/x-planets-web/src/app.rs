//! Web application state and render loop.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use x_planets_core::engine::LayerKind;
use x_planets_core::map_controller::LayerStateView;
use x_planets_core::model3d_renderer::Model3dRenderer;
use x_planets_core::tile_load_planner::{plan_tile_loads, LayerLoadState, PlannedRequestKind};
use x_planets_core::MapController;
use x_planets_render::{TerrainLayerData, TerrainRenderer, TerrainTileData, TileRenderer};
use x_planets_gpu::{GpuContext, GpuTexture, TextureManager};
use x_planets_math::{TileCoord, VisibleTile};
use x_planets_tiles::{RasterTileDecoder, TerrainEncoding, TerrainRgbDecoder, TerrariumDecoder, TileCache, TileDecoder};
use wasm_bindgen_futures::JsFuture;

use crate::tiles3d_web::Tiles3dWebState;

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
    /// Terrain encoding for elevation decoding.
    pub terrain_encoding: TerrainEncoding,
    /// Pending elevation tile fetches (separate from raster pending).
    pub pending_elevation_coords: HashSet<TileCoord>,
    pub max_elevation_concurrent: usize,
    /// Failed elevation tile fetches (separate from raster failures).
    pub failed_elevation_queue: Rc<RefCell<Vec<TileCoord>>>,
    /// Per-coord failure count for elevation tiles.
    elevation_fail_count: HashMap<TileCoord, u8>,
    /// Elevation tiles that have permanently failed (exceeded max retries).
    failed_elevation_permanent: HashSet<TileCoord>,
}

impl WebLayerState {
    fn new(name: String, kind: LayerKind, url_template: String, max_cached: usize, max_concurrent: usize) -> Self {
        // Extract encoding from the LayerKind if it's a terrain layer.
        let terrain_encoding = match &kind {
            LayerKind::Terrain { encoding, .. } => *encoding,
            _ => TerrainEncoding::Terrarium,
        };
        Self {
            name,
            kind,
            url_template,
            tile_textures: TileCache::new(max_cached),
            terrain_data: HashMap::new(),
            pending_coords: HashSet::new(),
            completed_queue: Rc::new(RefCell::new(Vec::new())),
            failed_queue: Rc::new(RefCell::new(Vec::new())),
            max_concurrent,
            available_coords_cache: HashSet::new(),
            elevation_url: None,
            terrain_encoding,
            pending_elevation_coords: HashSet::new(),
            max_elevation_concurrent: max_concurrent.min(4),
            failed_elevation_queue: Rc::new(RefCell::new(Vec::new())),
            elevation_fail_count: HashMap::new(),
            failed_elevation_permanent: HashSet::new(),
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

impl LayerLoadState for WebLayerState {
    fn kind(&self) -> &LayerKind {
        &self.kind
    }
    fn min_zoom(&self) -> u8 {
        0
    }
    fn max_zoom(&self) -> u8 {
        22
    }
    fn has_texture(&self, coord: &TileCoord) -> bool {
        self.tile_textures.contains(coord)
    }
    fn is_pending(&self, coord: &TileCoord) -> bool {
        self.pending_coords.contains(coord)
    }
    fn is_failed_cooldown(&self, _coord: &TileCoord) -> bool {
        // Web doesn't track per-coord cooldowns (failed tiles just get retried next frame).
        false
    }
    fn has_terrain_data(&self, coord: &TileCoord) -> bool {
        self.terrain_data.contains_key(coord)
    }
    fn is_elevation_pending(&self, coord: &TileCoord) -> bool {
        self.pending_elevation_coords.contains(coord)
    }
    fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
    fn pending_count(&self) -> usize {
        self.pending_coords.len()
    }
    fn max_elevation_concurrent(&self) -> usize {
        self.max_elevation_concurrent
    }
    fn elevation_pending_count(&self) -> usize {
        self.pending_elevation_coords.len()
    }
    fn has_elevation_source(&self) -> bool {
        self.elevation_url.is_some()
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
    model3d_renderer: Option<Model3dRenderer>,
    tex_manager: TextureManager,
    pub(crate) canvas: web_sys::HtmlCanvasElement,
    pub dpr: f64,
    last_width: u32,
    last_height: u32,

    // Per-layer state
    layer_states: Vec<WebLayerState>,
    /// 3D Tiles layer states.
    pub(crate) tiles3d_states: Vec<Tiles3dWebState>,

    /// Previous frame timestamp (ms) for dt calculation.
    last_frame_ms: Option<f64>,

    /// Registered JS event handlers: event_name → [callback, ...].
    pub(crate) event_handlers: HashMap<String, Vec<js_sys::Function>>,

    /// Whether the app has been destroyed (stops render loop).
    pub(crate) destroyed: bool,

    /// Cached visible_tiles() result: (center_lat, center_lon, zoom, pitch, bearing, width, height, tiles)
    cached_visible: Option<(f64, f64, f64, f64, f64, u32, u32, Vec<VisibleTile>)>,
}

impl WebApp {
    pub fn new(
        gpu: GpuContext,
        controller: MapController,
        renderer: TileRenderer,
        terrain_renderer: TerrainRenderer,
        model3d_renderer: Option<Model3dRenderer>,
        tex_manager: TextureManager,
        canvas: web_sys::HtmlCanvasElement,
        dpr: f64,
        tiles3d_states: Vec<Tiles3dWebState>,
    ) -> Self {
        let width = canvas.width();
        let height = canvas.height();

        // Create initial layer states from the engine's layers.
        let layer_states: Vec<WebLayerState> = controller
            .engine
            .layers
            .iter()
            .filter(|l| !matches!(l.config.kind, LayerKind::Tiles3d))
            .map(|l| WebLayerState::new(
                l.config.name.clone(),
                l.config.kind.clone(),
                l.config.tile_source_url.clone(),
                l.config.max_cached_tiles,
                l.config.max_concurrent_loads,
            ))
            .collect();

        Self {
            gpu,
            controller,
            renderer,
            terrain_renderer,
            model3d_renderer,
            tex_manager,
            canvas,
            dpr,
            last_width: width,
            last_height: height,
            layer_states,
            tiles3d_states,
            last_frame_ms: None,
            event_handlers: HashMap::new(),
            destroyed: false,
            cached_visible: None,
        }
    }

    /// Add a new layer state for a dynamically added layer.
    pub fn add_layer_state(&mut self, name: &str, url: &str, kind: LayerKind, max_cached: usize, max_concurrent: usize) {
        self.layer_states.push(WebLayerState::new(
            name.to_string(),
            kind,
            url.to_string(),
            max_cached,
            max_concurrent,
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
    pub fn toggle_terrain(&mut self) -> bool {
        let enabled = self.controller.toggle_terrain();

        if enabled {
            // Set elevation URL and encoding on the imagery layer so it starts loading elevation
            let imagery_name = self.controller.terrain_imagery_name()
                .unwrap_or("base").to_string();
            let terrain_url = self.controller.terrain_url()
                .unwrap_or("").to_string();
            let encoding = self.controller.terrain_encoding()
                .unwrap_or(TerrainEncoding::Terrarium);
            if let Some(ls) = self.layer_states.iter_mut().find(|ls| ls.name == imagery_name) {
                ls.elevation_url = Some(terrain_url);
                ls.terrain_encoding = encoding;
            }
        } else {
            // Clear elevation data and pending on the imagery layer
            for ls in &mut self.layer_states {
                ls.elevation_url = None;
                ls.terrain_data.clear();
                ls.pending_elevation_coords.clear();
                ls.elevation_fail_count.clear();
                ls.failed_elevation_permanent.clear();
            }
        }

        enabled
    }

    /// Set terrain elevation source URL and encoding at runtime.
    pub fn set_terrain_source(&mut self, url: &str, encoding: &str) {
        let enc = match encoding {
            "mapbox" | "mapbox-rgb" => TerrainEncoding::MapboxRgb,
            "quantized-mesh" | "qm" => TerrainEncoding::QuantizedMesh,
            _ => TerrainEncoding::Terrarium,
        };
        self.controller.set_terrain_source(url, enc);
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
        // Cache visible_tiles() result: skip recalculation if viewport hasn't changed.
        let vp = &self.controller.engine.viewport;
        let vp_key = (vp.center.lat, vp.center.lon, vp.zoom, vp.pitch, vp.bearing, vp.width, vp.height);
        let visible = if self.cached_visible.as_ref().map_or(true, |c| {
            (c.0, c.1, c.2, c.3, c.4, c.5, c.6) != vp_key
        }) {
            let v = self.controller.visible_tiles();
            self.cached_visible = Some((vp_key.0, vp_key.1, vp_key.2, vp_key.3, vp_key.4, vp_key.5, vp_key.6, v.clone()));
            v
        } else {
            self.cached_visible.as_ref().unwrap().7.clone()
        };
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

        // ── 6b. 3D Tiles: init, poll, traverse, render ──
        self.tiles3d_render(&view);

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
        // Use getBoundingClientRect for reliable sizing on mobile Safari.
        // clientWidth/clientHeight can return 0 before the first layout pass
        // on iOS Safari, causing the viewport to stay at 1x1.
        let rect = self.canvas.get_bounding_client_rect();
        let mut css_w = rect.width();
        let mut css_h = rect.height();
        if css_w < 1.0 || css_h < 1.0 {
            css_w = self.canvas.client_width() as f64;
            css_h = self.canvas.client_height() as f64;
        }
        if css_w < 1.0 || css_h < 1.0 {
            if let Some(window) = web_sys::window() {
                css_w = window.inner_width()
                    .ok().and_then(|v| v.as_f64()).unwrap_or(css_w);
                css_h = window.inner_height()
                    .ok().and_then(|v| v.as_f64()).unwrap_or(css_h);
            }
        }
        let w = (css_w * self.dpr).max(1.0) as u32;
        let h = (css_h * self.dpr).max(1.0) as u32;
        if w != self.last_width || h != self.last_height {
            self.canvas.set_width(w);
            self.canvas.set_height(h);
            self.gpu.resize_surface(w, h);
            self.controller.resize(w, h);
            self.renderer.resize(&self.gpu.device, w, h);
            self.terrain_renderer.resize(&self.gpu.device, w, h);
            if let Some(m3d) = &mut self.model3d_renderer {
                m3d.resize(&self.gpu.device, w, h);
            }
            self.last_width = w;
            self.last_height = h;
            log::info!("Resized: {}x{} (css: {}x{})", w, h, css_w as u32, css_h as u32);
        }
    }

    fn upload_completed_tiles(&mut self, now_secs: f64) {
        const MAX_UPLOADS_PER_FRAME: usize = 4;
        let mut upload_count: usize = 0;

        for ls in &mut self.layer_states {
            // Drain failed raster fetch notifications
            for coord in ls.failed_queue.borrow_mut().drain(..) {
                ls.pending_coords.remove(&coord);
            }
            // Drain failed elevation fetch notifications — track retry counts
            for coord in ls.failed_elevation_queue.borrow_mut().drain(..) {
                ls.pending_elevation_coords.remove(&coord);
                let count = ls.elevation_fail_count.entry(coord).or_insert(0);
                *count += 1;
                if *count >= 3 {
                    ls.failed_elevation_permanent.insert(coord);
                }
            }

            // Process completed results with a per-frame upload budget.
            // Raster uploads are expensive (GPU texture write), so we cap them.
            // Elevation data is CPU-only (no GPU upload), so it's always processed.
            let mut completed_ref = ls.completed_queue.borrow_mut();
            let mut remaining = Vec::new();
            for result in completed_ref.drain(..) {
                match result {
                    CompletedTileResult::Raster { coord, width, height, pixels } => {
                        if upload_count < MAX_UPLOADS_PER_FRAME {
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
                            upload_count += 1;
                        } else {
                            remaining.push(CompletedTileResult::Raster { coord, width, height, pixels });
                        }
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
            // Put back any raster results that exceeded the budget
            for r in remaining {
                completed_ref.push(r);
            }
            drop(completed_ref);
        }
    }

    /// 3D Tiles: init, poll messages, traverse, and render.
    fn tiles3d_render(&mut self, view: &wgpu::TextureView) {
        if self.tiles3d_states.is_empty() {
            return;
        }

        // Spawn init for any uninitialized layers.
        for ts3d in &mut self.tiles3d_states {
            if !ts3d.is_initialized() && !ts3d.init_spawned {
                ts3d.spawn_init();
            }
        }

        // Poll messages (GPU upload of decoded tiles).
        if let Some(renderer) = &self.model3d_renderer {
            // Need to split borrow: take states out temporarily.
            let mut states = std::mem::take(&mut self.tiles3d_states);
            for ts3d in &mut states {
                ts3d.poll_messages(&self.gpu, renderer);
            }
            self.tiles3d_states = states;
        }

        // Traverse and render each initialized layer.
        let viewport = &self.controller.engine.viewport;
        for ts3d in &mut self.tiles3d_states {
            if !ts3d.is_initialized() {
                continue;
            }

            let camera = x_planets_core::tiles3d_pipeline::viewport_to_traversal_camera(viewport);
            let fov_y = x_planets_core::tiles3d_pipeline::traversal_fov_y();
            let render_set = ts3d.traverse_and_spawn_loads(
                &camera, viewport.height as f64, fov_y,
            );

            // Diagnostic: log traversal results periodically
            if ts3d.generation % 300 == 1 {
                log::info!(
                    "[3dtiles] traversal: render_set={}, loaded={}, pending={}, gpu_tiles={}, camera=({:.0},{:.0},{:.0})",
                    render_set.len(),
                    ts3d.loaded_uris.len(),
                    ts3d.pending_uris.len(),
                    ts3d.gpu_tiles.len(),
                    camera.position_ecef.x, camera.position_ecef.y, camera.position_ecef.z,
                );
                if render_set.is_empty() && !ts3d.loaded_uris.is_empty() {
                    // Log first few loaded URIs for debugging URI mismatch
                    for (i, uri) in ts3d.loaded_uris.iter().take(3).enumerate() {
                        log::info!("[3dtiles] loaded_uri[{}]: {}", i, uri);
                    }
                }
                if !render_set.is_empty() {
                    for (i, tile) in render_set.iter().take(3).enumerate() {
                        let in_gpu = ts3d.gpu_tiles.contains_key(&tile.content_uri);
                        log::info!(
                            "[3dtiles] render[{}]: uri={}, in_gpu={}, sse={:.1}, transform_t=({:.0},{:.0},{:.0})",
                            i, tile.content_uri, in_gpu, tile.sse,
                            tile.transform.col(3).x, tile.transform.col(3).y, tile.transform.col(3).z,
                        );
                    }
                }
            }

            if render_set.is_empty() {
                continue;
            }

            let (uniforms, camera_ecef) =
                x_planets_core::tiles3d_pipeline::build_tiles3d_uniforms(viewport);

            ts3d.update_render_transforms(&self.gpu.queue, &render_set, camera_ecef, 1.0);

            let models = ts3d.collect_render_models(&render_set);
            if let Some(renderer) = &self.model3d_renderer {
                renderer.render_models_with_uniforms(&self.gpu, view, &uniforms, &models);
            }
        }
    }

    /// Request missing tiles for all layers using the shared planner.
    fn request_tiles_for_all_layers(&mut self, visible: &[VisibleTile]) {
        let camera_center = x_planets_math::geo_to_mercator(&self.controller.engine.viewport.center);
        let visible_set: HashSet<TileCoord> = visible.iter().map(|vt| vt.coord).collect();

        for ls in &mut self.layer_states {
            if matches!(ls.kind, LayerKind::Tiles3d) {
                continue;
            }

            // ── Core planner: what to load and what's still needed ──
            let (needed, requests) = plan_tile_loads(ls, visible, &visible_set, camera_center);

            // ── Abort stale raster requests ──
            ls.pending_coords.retain(|c| needed.raster.contains(c));

            // ── Abort stale elevation requests ──
            ls.pending_elevation_coords
                .retain(|c| needed.elevation.contains(c));

            // ── Spawn raster/terrain tasks ──
            let raster_slots = ls.max_concurrent.saturating_sub(ls.pending_coords.len());
            let mut raster_count = 0usize;

            // Determine elevation URL for this layer
            let elev_url = match &ls.kind {
                LayerKind::Terrain { .. } => Some(ls.url_template.clone()),
                _ => ls.elevation_url.clone(),
            };
            let encoding = ls.terrain_encoding;

            let elev_slots = ls
                .max_elevation_concurrent
                .saturating_sub(ls.pending_elevation_coords.len());
            let mut elev_count = 0usize;

            for req in &requests {
                match req.kind {
                    PlannedRequestKind::Raster => {
                        if raster_count >= raster_slots {
                            continue;
                        }
                        if ls.pending_coords.contains(&req.coord) {
                            continue;
                        }
                        ls.pending_coords.insert(req.coord);
                        raster_count += 1;

                        let queue = Rc::clone(&ls.completed_queue);
                        let failed = Rc::clone(&ls.failed_queue);
                        let url = tile_url(&ls.url_template, &req.coord);
                        let coord = req.coord;

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
                    PlannedRequestKind::Terrain => {
                        // Config-file terrain layers use raster slots.
                        if raster_count >= raster_slots {
                            continue;
                        }
                        if ls.pending_coords.contains(&req.coord) {
                            continue;
                        }
                        ls.pending_coords.insert(req.coord);
                        raster_count += 1;

                        let queue = Rc::clone(&ls.completed_queue);
                        let failed = Rc::clone(&ls.failed_queue);
                        let url = tile_url(&ls.url_template, &req.coord);
                        let coord = req.coord;

                        wasm_bindgen_futures::spawn_local(async move {
                            match fetch_bytes(&url).await {
                                Ok(bytes) => {
                                    let result = match encoding {
                                        TerrainEncoding::MapboxRgb => {
                                            TerrainRgbDecoder.decode(coord, &bytes).await
                                        }
                                        TerrainEncoding::Terrarium => {
                                            TerrariumDecoder.decode(coord, &bytes).await
                                        }
                                        TerrainEncoding::QuantizedMesh => {
                                            log::warn!("QM decoding not yet supported in web");
                                            failed.borrow_mut().push(coord);
                                            return;
                                        }
                                    };
                                    match result {
                                        Ok(decoded) => {
                                            queue.borrow_mut().push(CompletedTileResult::Elevation {
                                                coord,
                                                elevation: decoded.elevation,
                                                width: decoded.width,
                                                height: decoded.height,
                                            });
                                        }
                                        Err(e) => {
                                            log::warn!("Terrain decode {}: {}", coord, e);
                                            failed.borrow_mut().push(coord);
                                        }
                                    }
                                }
                                Err(e) => {
                                    log::warn!("Terrain fetch {}: {}", coord, e);
                                    failed.borrow_mut().push(coord);
                                }
                            }
                        });
                    }
                    PlannedRequestKind::Elevation => {
                        if elev_count >= elev_slots {
                            continue;
                        }
                        if ls.pending_elevation_coords.contains(&req.coord) {
                            continue;
                        }
                        if ls.failed_elevation_permanent.contains(&req.coord) {
                            continue;
                        }
                        let Some(ref elev_url_template) = elev_url else {
                            continue;
                        };
                        ls.pending_elevation_coords.insert(req.coord);
                        elev_count += 1;

                        let queue = Rc::clone(&ls.completed_queue);
                        let failed = Rc::clone(&ls.failed_elevation_queue);
                        let url = tile_url(elev_url_template, &req.coord);
                        let coord = req.coord;

                        wasm_bindgen_futures::spawn_local(async move {
                            match fetch_bytes(&url).await {
                                Ok(bytes) => {
                                    let result = match encoding {
                                        TerrainEncoding::MapboxRgb => {
                                            TerrainRgbDecoder.decode(coord, &bytes).await
                                        }
                                        TerrainEncoding::Terrarium => {
                                            TerrariumDecoder.decode(coord, &bytes).await
                                        }
                                        TerrainEncoding::QuantizedMesh => {
                                            log::warn!("QM decoding not yet supported in web");
                                            failed.borrow_mut().push(coord);
                                            return;
                                        }
                                    };
                                    match result {
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
            }
        }
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
    let opts = web_sys::RequestInit::new();
    opts.set_mode(web_sys::RequestMode::Cors);
    let request = web_sys::Request::new_with_str_and_init(url, &opts)
        .map_err(|e| format!("{:?}", e))?;
    let resp = JsFuture::from(window.fetch_with_request(&request))
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
