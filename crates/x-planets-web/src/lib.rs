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

        // ── MapController (default config = OSM base layer) ──
        let config = x_planets_core::engine::MapConfig::default();
        let controller = x_planets_core::MapController::new(config, width, height);

        // ── TileRenderer ──
        let renderer = x_planets_render::TileRenderer::new(&gpu);

        // ── TerrainRenderer ──
        let terrain_renderer = x_planets_render::TerrainRenderer::new(&gpu);

        // ── TextureManager ──
        let tex_manager = x_planets_gpu::TextureManager::new(&gpu.device);

        // ── WebApp ──
        let app = WebApp::new(gpu, controller, renderer, terrain_renderer, tex_manager, canvas.clone(), dpr);
        let app = std::rc::Rc::new(std::cell::RefCell::new(app));

        // ── Input events ──
        crate::input::register_events(&canvas, std::rc::Rc::clone(&app));

        // ── Start render loop ──
        WebApp::start_render_loop(std::rc::Rc::clone(&app));

        // ── Expose JS API as window.xplanets ──
        let xplanets = XPlanetsMap {
            app: std::rc::Rc::clone(&app),
        };
        js_sys::Reflect::set(
            &window,
            &JsValue::from_str("xplanets"),
            &xplanets.into(),
        )?;

        // ── Notify TypeScript that the API is ready ──
        let event = web_sys::CustomEvent::new("xplanets-ready")
            .map_err(|e| JsValue::from_str(&format!("Event error: {:?}", e)))?;
        window.dispatch_event(&event)?;

        log::info!("x-planets web started!");
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════
    // XPlanetsMap — JavaScript API exposed via wasm-bindgen
    // ═══════════════════════════════════════════════════════════════

    /// JavaScript API for controlling the x-planets map.
    ///
    /// Exposed as `window.xplanets` after initialization.
    ///
    /// ```javascript
    /// const map = window.xplanets;
    /// map.toggleTerrain();                  // terrain ON/OFF
    /// map.setTerrainExaggeration(2.0);      // height exaggeration
    /// map.setCenter(37.5665, 126.9780);     // move to Seoul
    /// map.zoomTo(12);                       // zoom level
    /// map.setProjection("Equirectangular"); // projection mode
    /// map.setLayerOpacity("base", 0.5);     // layer transparency
    /// ```
    #[wasm_bindgen]
    pub struct XPlanetsMap {
        app: std::rc::Rc<std::cell::RefCell<WebApp>>,
    }

    #[wasm_bindgen]
    impl XPlanetsMap {
        // ── Map Control ──

        /// Pan the map by pixel delta (dx, dy).
        #[wasm_bindgen(js_name = "panBy")]
        pub fn pan_by(&self, dx: f64, dy: f64) {
            self.app.borrow_mut().controller.pan(dx, dy);
        }

        /// Set the zoom level directly.
        #[wasm_bindgen(js_name = "zoomTo")]
        pub fn zoom_to(&self, zoom: f64) {
            self.app.borrow_mut().controller.set_zoom(zoom);
        }

        /// Set the map center to (lat, lon) in degrees.
        #[wasm_bindgen(js_name = "setCenter")]
        pub fn set_center(&self, lat: f64, lon: f64) {
            self.app.borrow_mut().controller.set_center(lat, lon);
        }

        /// Get the current map center as [lat, lon].
        #[wasm_bindgen(js_name = "getCenter")]
        pub fn get_center(&self) -> Vec<f64> {
            let (lat, lon) = self.app.borrow().controller.center();
            vec![lat, lon]
        }

        /// Get the current zoom level.
        #[wasm_bindgen(js_name = "getZoom")]
        pub fn get_zoom(&self) -> f64 {
            self.app.borrow().controller.zoom_level()
        }

        /// Get the current bearing (rotation) in degrees.
        #[wasm_bindgen(js_name = "getBearing")]
        pub fn get_bearing(&self) -> f64 {
            self.app.borrow().controller.bearing()
        }

        /// Get the current pitch angle in degrees.
        #[wasm_bindgen(js_name = "getPitch")]
        pub fn get_pitch(&self) -> f64 {
            self.app.borrow().controller.pitch_angle()
        }

        // ── Projection ──

        /// Set the projection by name (e.g. "Web Mercator", "Globe", "Equirectangular").
        /// Returns true if the projection was found.
        #[wasm_bindgen(js_name = "setProjection")]
        pub fn set_projection(&self, name: &str) -> bool {
            self.app.borrow_mut().controller.set_projection(name)
        }

        /// Cycle to the next available projection. Returns the new projection name.
        #[wasm_bindgen(js_name = "cycleProjection")]
        pub fn cycle_projection(&self) -> String {
            self.app.borrow_mut().cycle_projection()
        }

        /// Get the current projection name.
        #[wasm_bindgen(js_name = "getProjection")]
        pub fn get_projection(&self) -> String {
            self.app.borrow().controller.projection_name().to_string()
        }

        // ── Terrain ──

        /// Toggle terrain on/off. Returns the new state (true = terrain ON).
        ///
        /// Optional parameters:
        /// - `url`: Tile URL template (default: AWS Terrarium)
        /// - `encoding`: "terrarium" | "mapbox" | "quantized-mesh" (default: "terrarium")
        #[wasm_bindgen(js_name = "toggleTerrain")]
        pub fn toggle_terrain(&self, url: Option<String>, encoding: Option<String>) -> bool {
            self.app.borrow_mut().toggle_terrain_with(
                url.as_deref(),
                encoding.as_deref(),
            )
        }

        /// Whether terrain is currently enabled.
        #[wasm_bindgen(js_name = "terrainEnabled")]
        pub fn terrain_enabled(&self) -> bool {
            self.app.borrow().controller.terrain_enabled()
        }

        /// Set terrain height exaggeration factor.
        #[wasm_bindgen(js_name = "setTerrainExaggeration")]
        pub fn set_terrain_exaggeration(&self, value: f64) {
            self.app.borrow_mut().terrain_renderer.exaggeration = value;
        }

        /// Get terrain height exaggeration factor.
        #[wasm_bindgen(js_name = "getTerrainExaggeration")]
        pub fn get_terrain_exaggeration(&self) -> f64 {
            self.app.borrow().terrain_renderer.exaggeration
        }

        // ── Layer Management ──

        /// Set layer visibility. Returns true if the layer was found.
        #[wasm_bindgen(js_name = "setLayerVisible")]
        pub fn set_layer_visible(&self, name: &str, visible: bool) -> bool {
            self.app.borrow_mut().controller.set_layer_visible(name, visible)
        }

        /// Set layer opacity (0.0–1.0). Returns true if the layer was found.
        #[wasm_bindgen(js_name = "setLayerOpacity")]
        pub fn set_layer_opacity(&self, name: &str, opacity: f32) -> bool {
            self.app.borrow_mut().controller.set_layer_opacity(name, opacity)
        }

        /// Remove a layer by name. Returns true if the layer was found.
        #[wasm_bindgen(js_name = "removeLayer")]
        pub fn remove_layer(&self, name: &str) -> bool {
            self.app.borrow_mut().controller.remove_layer(name)
        }

        /// Get the number of layers.
        #[wasm_bindgen(js_name = "layerCount")]
        pub fn layer_count(&self) -> usize {
            self.app.borrow().controller.layer_count()
        }

        // ── Viewport ──

        /// Resize the map viewport.
        pub fn resize(&self, width: u32, height: u32) {
            self.app.borrow_mut().controller.resize(width, height);
        }
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
