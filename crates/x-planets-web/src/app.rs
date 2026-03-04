//! Web application state and render loop.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use x_planets_core::engine::MapEngine;
use x_planets_core::interaction::{
    AnimationController, FADE_DURATION, build_crossfade_overlay, compute_crossfade,
    compute_fade_overrides,
};
use x_planets_core::pipeline::{resolve_fallbacks, RenderableTile};
use x_planets_core::render::RenderLayerData;
use x_planets_core::TileRenderer;
use x_planets_gpu::{GpuContext, GpuTexture, TextureManager};
use x_planets_math::{TileCoord, VisibleTile};
use x_planets_tiles::{RasterTileDecoder, TileCache, TileDecoder};
use wasm_bindgen_futures::JsFuture;

// ═══════════════════════════════════════════════════════════════════
// Completed tile result (produced by async fetch, consumed each frame)
// ═══════════════════════════════════════════════════════════════════

struct CompletedTile {
    coord: TileCoord,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

// ═══════════════════════════════════════════════════════════════════
// WebApp
// ═══════════════════════════════════════════════════════════════════

pub struct WebApp {
    pub gpu: GpuContext,
    pub engine: MapEngine,
    renderer: TileRenderer,
    tex_manager: TextureManager,
    canvas: web_sys::HtmlCanvasElement,
    dpr: f64,
    last_width: u32,
    last_height: u32,

    // Tile loading state
    url_template: String,
    tile_textures: TileCache<GpuTexture>,
    pending_coords: HashSet<TileCoord>,
    completed_queue: Rc<RefCell<Vec<CompletedTile>>>,
    /// Failed fetch notifications (coord pushed from async task, drained each frame).
    failed_queue: Rc<RefCell<Vec<TileCoord>>>,
    max_concurrent: usize,

    // ── Shared animation controller (from x-planets-core) ──
    pub anim: AnimationController,
    /// Previous frame timestamp (ms) for dt calculation.
    last_frame_ms: Option<f64>,

    // ── Crossfade tracking ──
    /// Tiles that were visible+available last frame (for detecting transitions).
    prev_visible_available: HashSet<TileCoord>,
    /// Tiles that recently left the visible set, rendered as a fading-out overlay.
    /// Maps coord → departure time (seconds).
    departing_tiles: HashMap<TileCoord, f64>,
}

impl WebApp {
    pub fn new(
        gpu: GpuContext,
        engine: MapEngine,
        renderer: TileRenderer,
        tex_manager: TextureManager,
        canvas: web_sys::HtmlCanvasElement,
        dpr: f64,
    ) -> Self {
        let url_template = engine
            .layers
            .first()
            .map(|l| l.config.tile_source_url.clone())
            .unwrap_or_else(|| "https://tile.openstreetmap.org/{z}/{x}/{y}.png".into());

        let width = canvas.width();
        let height = canvas.height();

        let initial_zoom = engine.viewport.zoom;
        Self {
            gpu,
            engine,
            renderer,
            tex_manager,
            canvas,
            dpr,
            last_width: width,
            last_height: height,
            url_template,
            tile_textures: TileCache::new(256),
            pending_coords: HashSet::new(),
            completed_queue: Rc::new(RefCell::new(Vec::new())),
            failed_queue: Rc::new(RefCell::new(Vec::new())),
            max_concurrent: 6,
            anim: AnimationController::new(initial_zoom),
            last_frame_ms: None,
            prev_visible_available: HashSet::new(),
            departing_tiles: HashMap::new(),
        }
    }

    /// Resolve the active projection name to a `ProjectionMode` enum.
    ///
    /// Delegates to the `ProjectionPlugin::rendering_mode()` declared by
    /// the active projection in the registry.
    pub(crate) fn resolve_projection_mode(&self) -> x_planets_math::ProjectionMode {
        self.engine.rendering_mode()
    }

    /// Cycle to the next projection and return its name.
    pub fn cycle_projection(&mut self) -> String {
        let mut names: Vec<String> = self
            .engine
            .projection_registry
            .list()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        names.sort();
        let current = &self.engine.active_projection;
        let idx = names.iter().position(|n| n == current).unwrap_or(0);
        let next_idx = (idx + 1) % names.len();
        self.engine.set_projection(&names[next_idx]);
        log::info!("Projection: {}", self.engine.active_projection);
        self.engine.active_projection.clone()
    }

    /// Start the requestAnimationFrame render loop.
    pub fn start_render_loop(app: Rc<RefCell<Self>>) {
        let f: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
        let g = f.clone();

        *g.borrow_mut() = Some(Closure::new(move |timestamp_ms: f64| {
            app.borrow_mut().render_frame(timestamp_ms);
            request_animation_frame(f.borrow().as_ref().unwrap());
        }));

        request_animation_frame(g.borrow().as_ref().unwrap());
    }

    fn render_frame(&mut self, timestamp_ms: f64) {
        let now_secs = timestamp_ms / 1000.0;

        // ── 0. Tick animations (smooth zoom, inertia pan) ──
        if let Some(prev_ms) = self.last_frame_ms {
            let dt = ((timestamp_ms - prev_ms) / 1000.0).min(0.1);
            let mode = self.resolve_projection_mode();
            self.anim.tick_with_mode(&mut self.engine, dt, mode);
        }
        self.last_frame_ms = Some(timestamp_ms);

        // GC finished tile fades
        self.anim.gc_fades(now_secs);

        // ── 1. Handle resize ──
        self.check_resize();

        // ── 2. Process completed tile fetches → GPU upload ──
        self.upload_completed_tiles(now_secs);

        // ── 3. Request missing tiles ──
        let proj_mode = self.resolve_projection_mode();
        let visible = self.engine.viewport.visible_tiles_for_mode(proj_mode);
        self.request_missing_tiles(&visible);

        // ── 4. Build render data with crossfade ──
        let available: HashSet<TileCoord> = self.tile_textures.keys().copied().collect();
        // Use canonical coords for set operations
        let visible_set: HashSet<TileCoord> = visible.iter().map(|vt| vt.coord).collect();

        // 4a. Register fade for tiles that just became visible+available
        //     (handles cached tiles re-entering view and zoom-out transitions).
        let visible_available: HashSet<TileCoord> = visible.iter()
            .filter(|vt| available.contains(&vt.coord))
            .map(|vt| vt.coord)
            .collect();
        for &coord in &visible_available {
            if !self.prev_visible_available.contains(&coord) {
                if self.anim.tile_fade_elapsed(&coord, now_secs).is_none() {
                    self.anim.register_tile_loaded(coord, now_secs);
                }
            }
        }

        // 4b. Track departing tiles (were visible+available, now gone) for
        //     zoom-out fade-out overlay.
        for &coord in &self.prev_visible_available {
            if !visible_set.contains(&coord) {
                self.departing_tiles.entry(coord).or_insert(now_secs);
            }
        }
        self.departing_tiles.retain(|_, start| now_secs - *start < FADE_DURATION);
        self.prev_visible_available = visible_available;

        let texture_views: HashMap<TileCoord, &wgpu::TextureView> = self
            .tile_textures
            .iter()
            .map(|(k, v)| (*k, &v.view))
            .collect();

        // Crossfade: exclude fading children from base, render them as overlay
        let (available_for_base, crossfade_tiles) =
            compute_crossfade(&visible, &available, |coord| {
                self.anim.tile_fade_elapsed(coord, now_secs)
            });

        let renderable = resolve_fallbacks(&visible, &available_for_base);

        // Opacity overrides for tiles with no parent (fade from zero)
        let tile_opacity_overrides =
            compute_fade_overrides(&renderable, 1.0, |coord| {
                self.anim.tile_fade_elapsed(coord, now_secs)
            });

        let base_layer = RenderLayerData {
            name: "base",
            opacity: 1.0,
            tiles: renderable,
            texture_views: texture_views.clone(),
            tile_opacity_overrides,
        };

        // Build crossfade overlay layer (zoom-in: child fading in over parent)
        let mut layers: Vec<RenderLayerData> = vec![base_layer];
        if !crossfade_tiles.is_empty() {
            let (overlay_tiles, overlay_opacity) =
                build_crossfade_overlay(&crossfade_tiles, 1.0);
            layers.push(RenderLayerData {
                name: "crossfade-overlay",
                opacity: 1.0,
                tiles: overlay_tiles,
                texture_views: texture_views.clone(),
                tile_opacity_overrides: overlay_opacity,
            });
        }

        // Build departing tiles overlay (zoom-out: old tiles fading out)
        if !self.departing_tiles.is_empty() {
            let mut overlay_tiles = Vec::new();
            let mut overlay_opacity = HashMap::new();
            for (&coord, &start) in &self.departing_tiles {
                if texture_views.contains_key(&coord) {
                    let elapsed = now_secs - start;
                    let fade_out = (1.0 - elapsed / FADE_DURATION).max(0.0) as f32;
                    if fade_out > 0.01 {
                        overlay_tiles.push(RenderableTile {
                            coord,
                            texture_coord: coord,
                            uv_rect: [0.0, 0.0, 1.0, 1.0],
                            display_x: coord.x as i64,
                        });
                        overlay_opacity.insert(coord, fade_out);
                    }
                }
            }
            if !overlay_tiles.is_empty() {
                layers.push(RenderLayerData {
                    name: "zoom-out-overlay",
                    opacity: 1.0,
                    tiles: overlay_tiles,
                    texture_views: texture_views.clone(),
                    tile_opacity_overrides: overlay_opacity,
                });
            }
        }

        // ── 5. Render ──
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
        self.renderer
            .render_frame_layered_projected(&self.gpu, &view, &self.engine.viewport, &layers, mode);

        frame.present();

        // ── 6. LRU bump visible tiles + base tiles ──
        // Always bump base tiles (z=0, z=1) to prevent LRU eviction.
        for z in 0..=1u8 {
            let n = 1u32 << z;
            for y in 0..n {
                for x in 0..n {
                    let _ = self.tile_textures.get(&TileCoord::new(z, x, y));
                }
            }
        }
        for vt in &visible {
            let _ = self.tile_textures.get(&vt.coord);
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
            self.engine.resize(w, h);
            self.renderer.resize(&self.gpu.device, w, h);
            self.last_width = w;
            self.last_height = h;
            log::info!("Resized: {}x{}", w, h);
        }
    }

    fn upload_completed_tiles(&mut self, now_secs: f64) {
        // Drain failed fetch notifications so their pending slots are freed.
        for coord in self.failed_queue.borrow_mut().drain(..) {
            self.pending_coords.remove(&coord);
        }

        let completed: Vec<CompletedTile> = self.completed_queue.borrow_mut().drain(..).collect();
        for tile in completed {
            self.pending_coords.remove(&tile.coord);
            let label = format!("tile-{}/{}/{}", tile.coord.z, tile.coord.x, tile.coord.y);
            let gpu_tex = self.tex_manager.create_rgba_texture(
                &self.gpu.device,
                &self.gpu.queue,
                &label,
                tile.width,
                tile.height,
                &tile.pixels,
            );
            self.tile_textures.insert(tile.coord, gpu_tex);
            // Register for fade-in animation
            self.anim.register_tile_loaded(tile.coord, now_secs);
        }
    }

    fn request_missing_tiles(&mut self, visible: &[VisibleTile]) {
        // Use canonical coords for cache/pending lookups
        let visible_set: HashSet<TileCoord> = visible.iter().map(|vt| vt.coord).collect();

        // Free pending slots for tiles no longer visible (keep base tiles).
        // The in-flight fetches can't be cancelled, but freeing the slot
        // lets new (now-visible) tiles start loading immediately.
        self.pending_coords.retain(|c| c.z <= 1 || visible_set.contains(c));

        let camera_center = x_planets_math::geo_to_mercator(&self.engine.viewport.center);

        // ── Base tile loading ──
        // Always eagerly load z=0 and z=1 tiles (5 total) so that
        // resolve_fallbacks() always finds a cached ancestor.
        let mut missing: Vec<(TileCoord, f64)> = Vec::new();
        for z in 0..=1u8 {
            let n = 1u32 << z;
            for y in 0..n {
                for x in 0..n {
                    let coord = TileCoord::new(z, x, y);
                    if !self.tile_textures.contains(&coord)
                        && !self.pending_coords.contains(&coord)
                    {
                        // Highest priority (distance 0)
                        missing.push((coord, 0.0));
                    }
                }
            }
        }

        // Deduplicate by canonical coord (same tile may appear in multiple
        // wrapped positions) and sort by display distance (closest first).
        let mut seen = HashSet::new();
        let mut visible_missing: Vec<(TileCoord, f64)> = visible
            .iter()
            .filter(|vt| {
                !self.tile_textures.contains(&vt.coord)
                    && !self.pending_coords.contains(&vt.coord)
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

        // Respect max concurrent
        let slots = self.max_concurrent.saturating_sub(self.pending_coords.len());
        for (coord, _) in missing.into_iter().take(slots) {
            self.pending_coords.insert(coord);
            let queue = Rc::clone(&self.completed_queue);
            let failed = Rc::clone(&self.failed_queue);
            let url = tile_url(&self.url_template, &coord);

            wasm_bindgen_futures::spawn_local(async move {
                match fetch_bytes(&url).await {
                    Ok(bytes) => {
                        let decoder = RasterTileDecoder::default();
                        match decoder.decode(coord, &bytes).await {
                            Ok(decoded) => {
                                queue.borrow_mut().push(CompletedTile {
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
