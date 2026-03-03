//! x-planets-web: Web/WASM platform backend.
//!
//! Provides browser-based implementations for canvas rendering,
//! fetch API tile loading, and touch/mouse input handling.
//!
//! This crate is compiled to WebAssembly and exposed to JavaScript
//! via wasm-bindgen.

#[cfg(target_arch = "wasm32")]
mod app;
#[cfg(target_arch = "wasm32")]
mod input;

#[cfg(target_arch = "wasm32")]
mod web_impl {
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    use crate::app::WebApp;

    /// Initialize the WASM module, then launch the map.
    #[wasm_bindgen(start)]
    pub fn wasm_init() {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).unwrap();
        log::info!("x-planets WASM module initialized");

        wasm_bindgen_futures::spawn_local(async {
            if let Err(e) = run().await {
                log::error!("Fatal: {:?}", e);
            }
        });
    }

    async fn run() -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("No window")?;
        let document = window.document().ok_or("No document")?;
        let canvas = document
            .get_element_by_id("x-planets-canvas")
            .ok_or("Canvas not found")?
            .dyn_into::<web_sys::HtmlCanvasElement>()?;

        // ── DPR-aware canvas sizing ──
        let dpr = window.device_pixel_ratio();
        let css_w = canvas.client_width() as f64;
        let css_h = canvas.client_height() as f64;
        let width = (css_w * dpr).max(1.0) as u32;
        let height = (css_h * dpr).max(1.0) as u32;
        canvas.set_width(width);
        canvas.set_height(height);

        log::info!("Canvas: {}x{} (DPR: {:.1})", width, height, dpr);

        // ── GPU ──
        let surface_target = wgpu::SurfaceTarget::Canvas(canvas.clone());
        let gpu = x_planets_gpu::GpuContext::new_with_window(surface_target, width, height)
            .await
            .map_err(|e| JsValue::from_str(&format!("GPU init failed: {}", e)))?;

        log::info!("GPU: {}", gpu.adapter_info().name);

        // ── MapEngine (default config = OSM base layer) ──
        let config = x_planets_core::engine::MapConfig::default();
        let engine = x_planets_core::MapEngine::new(config, width, height);

        // ── TileRenderer ──
        let renderer = x_planets_core::TileRenderer::new(&gpu);

        // ── TextureManager ──
        let tex_manager = x_planets_gpu::TextureManager::new(&gpu.device);

        // ── WebApp ──
        let app = WebApp::new(gpu, engine, renderer, tex_manager, canvas.clone(), dpr);
        let app = std::rc::Rc::new(std::cell::RefCell::new(app));

        // ── Input events ──
        crate::input::register_events(&canvas, std::rc::Rc::clone(&app));

        // ── Projection switcher button ──
        crate::input::setup_projection_button(std::rc::Rc::clone(&app));

        // ── Start render loop ──
        WebApp::start_render_loop(std::rc::Rc::clone(&app));

        log::info!("x-planets web started!");
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════
// Web Tile Source (Fetch API)
// ═══════════════════════════════════════════════════════════════════

#[cfg(target_arch = "wasm32")]
mod web_tile_source {
    use async_trait::async_trait;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use x_planets_math::TileCoord;
    use x_planets_tiles::{LoadError, TileSource};

    /// Web-based tile source using the browser Fetch API.
    pub struct WebTileSource {
        url_template: String,
        tms: bool,
    }

    impl WebTileSource {
        pub fn new(url_template: impl Into<String>) -> Self {
            Self {
                url_template: url_template.into(),
                tms: false,
            }
        }

        #[allow(dead_code)]
        pub fn with_tms(mut self, tms: bool) -> Self {
            self.tms = tms;
            self
        }
    }

    #[async_trait(?Send)]
    impl TileSource for WebTileSource {
        async fn fetch(&self, coord: TileCoord) -> Result<Vec<u8>, LoadError> {
            let url = self.tile_url(&coord);

            let window = web_sys::window()
                .ok_or_else(|| LoadError::Network("No window object".into()))?;

            let resp_value = JsFuture::from(window.fetch_with_str(&url))
                .await
                .map_err(|e| LoadError::Network(format!("{:?}", e)))?;

            let resp: web_sys::Response = resp_value
                .dyn_into()
                .map_err(|_| LoadError::Network("Response cast failed".into()))?;

            if !resp.ok() {
                let status = resp.status();
                if status == 404 {
                    return Err(LoadError::NotFound(coord));
                }
                return Err(LoadError::HttpError { status, url });
            }

            let array_buffer = JsFuture::from(
                resp.array_buffer()
                    .map_err(|e| LoadError::Network(format!("{:?}", e)))?,
            )
            .await
            .map_err(|e| LoadError::Network(format!("{:?}", e)))?;

            let uint8_array = js_sys::Uint8Array::new(&array_buffer);
            Ok(uint8_array.to_vec())
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
}

#[cfg(target_arch = "wasm32")]
pub use web_tile_source::WebTileSource;

// Non-WASM stub for compilation checks
#[cfg(not(target_arch = "wasm32"))]
pub fn _placeholder() {
    // This module is only meaningful when compiled to wasm32.
}
