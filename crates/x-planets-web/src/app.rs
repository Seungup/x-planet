//! Web application state and render loop.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use x_planets_core::engine::MapEngine;
use x_planets_core::interaction::{
    AnimationController, build_crossfade_overlay, compute_crossfade, compute_fade_overrides,
};
use x_planets_core::pipeline::resolve_fallbacks;
use x_planets_core::render::RenderLayerData;
use x_planets_core::TileRenderer;
use x_planets_gpu::{GpuContext, GpuTexture, TextureManager};
use x_planets_math::TileCoord;
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
    max_concurrent: usize,

    // ── Shared animation controller (from x-planets-core) ──
    pub anim: AnimationController,
    /// Previous frame timestamp (ms) for dt calculation.
    last_frame_ms: Option<f64>,
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
            max_concurrent: 6,
            anim: AnimationController::new(initial_zoom),
            last_frame_ms: None,
        }
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
            self.anim.tick(&mut self.engine, dt);
        }
        self.last_frame_ms = Some(timestamp_ms);

        // GC finished tile fades
        self.anim.gc_fades(now_secs);

        // ── 1. Handle resize ──
        self.check_resize();

        // ── 2. Process completed tile fetches → GPU upload ──
        self.upload_completed_tiles(now_secs);

        // ── 3. Request missing tiles ──
        let visible = self.engine.viewport.visible_tiles();
        self.request_missing_tiles(&visible);

        // ── 4. Build render data with crossfade ──
        let available: HashSet<TileCoord> = self.tile_textures.keys().copied().collect();

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

        // Build crossfade overlay layer (if any tiles are transitioning)
        let mut layers: Vec<RenderLayerData> = vec![base_layer];
        if !crossfade_tiles.is_empty() {
            let (overlay_tiles, overlay_opacity) =
                build_crossfade_overlay(&crossfade_tiles, 1.0);
            layers.push(RenderLayerData {
                name: "crossfade-overlay",
                opacity: 1.0,
                tiles: overlay_tiles,
                texture_views,
                tile_opacity_overrides: overlay_opacity,
            });
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

        self.renderer
            .render_frame_layered(&self.gpu, &view, &self.engine.viewport, &layers);

        frame.present();

        // ── 6. LRU bump visible tiles ──
        for coord in &visible {
            let _ = self.tile_textures.get(coord);
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

    fn request_missing_tiles(&mut self, visible: &[TileCoord]) {
        let camera_center = x_planets_math::geo_to_mercator(&self.engine.viewport.center);

        // Sort by distance from camera (closest first)
        let mut missing: Vec<(TileCoord, f64)> = visible
            .iter()
            .filter(|c| {
                !self.tile_textures.contains(c) && !self.pending_coords.contains(c)
            })
            .map(|c| {
                let center = c.mercator_center();
                let dist = (center - camera_center).length();
                (*c, dist)
            })
            .collect();
        missing.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Respect max concurrent
        let slots = self.max_concurrent.saturating_sub(self.pending_coords.len());
        for (coord, _) in missing.into_iter().take(slots) {
            self.pending_coords.insert(coord);
            let queue = Rc::clone(&self.completed_queue);
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
                            Err(e) => log::warn!("Decode {}: {}", coord, e),
                        }
                    }
                    Err(e) => log::warn!("Fetch {}: {}", coord, e),
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
