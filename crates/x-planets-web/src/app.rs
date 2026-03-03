//! Web application state and render loop.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use x_planets_core::engine::MapEngine;
use wasm_bindgen_futures::JsFuture;

use x_planets_core::pipeline::resolve_fallbacks;
use x_planets_core::render::RenderLayerData;
use x_planets_core::TileRenderer;
use x_planets_gpu::{GpuContext, GpuTexture, TextureManager};
use x_planets_math::TileCoord;
use x_planets_tiles::{RasterTileDecoder, TileCache, TileDecoder};

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

    // Max concurrent tile loads
    max_concurrent: usize,

    // ── Animation state ──
    /// Target zoom level (accumulated from scroll/keyboard, animated toward).
    pub zoom_target: f64,
    /// Screen-space anchor for zoom-toward-cursor. `None` = zoom at center.
    pub zoom_anchor: Option<(f64, f64)>,
    /// Current pan velocity in screen pixels/sec (for inertia).
    pub pan_velocity: (f64, f64),
    /// Recent drag samples: (position, timestamp_ms) for velocity estimation.
    pub drag_samples: Vec<((f64, f64), f64)>,
    /// Timestamp (ms) of last left-click for double-click detection.
    pub last_click_time_ms: Option<f64>,
    /// Position of last left-click for double-click detection.
    pub last_click_pos: Option<(f64, f64)>,
    /// Previous frame timestamp (ms) for dt calculation.
    pub last_frame_ms: Option<f64>,
    /// Last known mouse position (for zoom anchor fallback).
    pub last_mouse_pos: Option<(f64, f64)>,
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
            zoom_target: initial_zoom,
            zoom_anchor: None,
            pan_velocity: (0.0, 0.0),
            drag_samples: Vec::new(),
            last_click_time_ms: None,
            last_click_pos: None,
            last_frame_ms: None,
            last_mouse_pos: None,
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
        // ── 0. Tick animations (smooth zoom, inertia pan) ──
        self.tick_animations(timestamp_ms);

        // ── 1. Handle resize ──
        self.check_resize();

        // ── 2. Process completed tile fetches → GPU upload ──
        self.upload_completed_tiles();

        // ── 3. Request missing tiles ──
        let visible = self.engine.viewport.visible_tiles();
        self.request_missing_tiles(&visible);

        // ── 4. Build render data ──
        let available: HashSet<TileCoord> = self.tile_textures.keys().copied().collect();
        let renderable = resolve_fallbacks(&visible, &available);

        let texture_views: HashMap<TileCoord, &wgpu::TextureView> = self
            .tile_textures
            .iter()
            .map(|(k, v)| (*k, &v.view))
            .collect();

        let layer = RenderLayerData {
            name: "base",
            opacity: 1.0,
            tiles: renderable,
            texture_views,
            tile_opacity_overrides: HashMap::new(),
        };

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
            .render_frame_layered(&self.gpu, &view, &self.engine.viewport, &[layer]);

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

    fn upload_completed_tiles(&mut self) {
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
    // ── Animation methods ──

    fn tick_animations(&mut self, timestamp_ms: f64) {
        let dt = match self.last_frame_ms {
            Some(prev) => ((timestamp_ms - prev) / 1000.0).min(0.1), // cap at 100ms
            None => {
                self.last_frame_ms = Some(timestamp_ms);
                return;
            }
        };
        self.last_frame_ms = Some(timestamp_ms);

        // Smooth zoom: exponential decay toward target
        let current = self.engine.viewport.zoom;
        let target = self
            .zoom_target
            .clamp(self.engine.camera.min_zoom, self.engine.camera.max_zoom);
        let diff = target - current;
        if diff.abs() > 0.001 {
            let new_zoom = current + diff * (1.0 - (-12.0 * dt).exp());
            let delta = new_zoom - current;
            match self.zoom_anchor {
                Some((mx, my)) => self.engine.zoom_at(delta, mx, my),
                None => self.engine.zoom(delta),
            }
        } else if (current - target).abs() > 1e-9 {
            self.engine.viewport.zoom = target;
            self.engine.request_redraw();
        }

        // Inertia pan: friction-based velocity decay
        let (vx, vy) = self.pan_velocity;
        let speed = (vx * vx + vy * vy).sqrt();
        if speed > 1.0 {
            self.engine.pan(vx * dt, -(vy * dt));
            let friction = (-6.0 * dt).exp();
            self.pan_velocity = (vx * friction, vy * friction);
        } else {
            self.pan_velocity = (0.0, 0.0);
        }
    }

    /// Record a drag position sample for velocity estimation.
    pub fn record_drag(&mut self, pos: (f64, f64), timestamp_ms: f64) {
        // Keep only the last 100ms of samples.
        self.drag_samples
            .retain(|(_, t)| timestamp_ms - t < 100.0);
        self.drag_samples.push((pos, timestamp_ms));
    }

    /// Compute pan velocity from recent drag samples (called on mouse-up).
    pub fn compute_release_velocity(&mut self, timestamp_ms: f64) {
        self.drag_samples
            .retain(|(_, t)| timestamp_ms - t < 100.0);
        if self.drag_samples.len() < 2 {
            self.pan_velocity = (0.0, 0.0);
            return;
        }
        let first = &self.drag_samples[0];
        let last = &self.drag_samples[self.drag_samples.len() - 1];
        let dt = (last.1 - first.1) / 1000.0; // seconds
        if dt < 0.001 {
            self.pan_velocity = (0.0, 0.0);
            return;
        }
        let vx = (last.0 .0 - first.0 .0) / dt;
        let vy = (last.0 .1 - first.0 .1) / dt;
        self.pan_velocity = (vx, vy);
        self.drag_samples.clear();
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
