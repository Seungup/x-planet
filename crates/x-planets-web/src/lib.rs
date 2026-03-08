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
    use x_planets_core::engine::{LayerConfig, LayerKind};

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

        /// Set the bearing (rotation) in degrees.
        #[wasm_bindgen(js_name = "setBearing")]
        pub fn set_bearing(&self, degrees: f64) {
            self.app.borrow_mut().controller.set_bearing(degrees);
        }

        /// Set the pitch (tilt) in degrees (0 = top-down, 60 = max tilt).
        #[wasm_bindgen(js_name = "setPitch")]
        pub fn set_pitch(&self, degrees: f64) {
            self.app.borrow_mut().controller.set_pitch(degrees);
        }

        /// Smoothly animate the camera to a new position (ease-in-out).
        ///
        /// Parameters: lat, lon, zoom, duration (seconds, default 2.0),
        /// bearing (degrees, optional), pitch (degrees, optional).
        #[wasm_bindgen(js_name = "flyTo")]
        pub fn fly_to(
            &self,
            lat: f64,
            lon: f64,
            zoom: f64,
            duration: Option<f64>,
            bearing: Option<f64>,
            pitch: Option<f64>,
        ) {
            self.app.borrow_mut().controller.fly_to(lat, lon, zoom, duration, bearing, pitch);
        }

        /// Smoothly animate the camera to a new position (linear interpolation).
        ///
        /// Parameters: lat, lon, zoom, duration (seconds, default 1.0),
        /// bearing (degrees, optional), pitch (degrees, optional).
        #[wasm_bindgen(js_name = "easeTo")]
        pub fn ease_to(
            &self,
            lat: f64,
            lon: f64,
            zoom: f64,
            duration: Option<f64>,
            bearing: Option<f64>,
            pitch: Option<f64>,
        ) {
            self.app.borrow_mut().controller.ease_to(lat, lon, zoom, duration, bearing, pitch);
        }

        /// Jump the camera to a new position instantly (no animation).
        #[wasm_bindgen(js_name = "jumpTo")]
        pub fn jump_to(
            &self,
            lat: f64,
            lon: f64,
            zoom: f64,
            bearing: Option<f64>,
            pitch: Option<f64>,
        ) {
            self.app.borrow_mut().controller.jump_to(lat, lon, zoom, bearing, pitch);
        }

        /// Cancel any running camera animation.
        #[wasm_bindgen(js_name = "stopAnimation")]
        pub fn stop_animation(&self) {
            self.app.borrow_mut().controller.stop_animation();
        }

        // ── Coordinate Conversion ──

        /// Convert geographic (lat, lon) to screen pixel coordinates.
        /// Returns [x, y] or null if outside the visible area.
        pub fn project(&self, lat: f64, lon: f64) -> Option<Vec<f64>> {
            self.app.borrow().controller.project(lat, lon).map(|(x, y)| vec![x, y])
        }

        /// Convert screen pixel coordinates to geographic (lat, lon).
        /// Returns [lat, lon] or null if outside the map.
        pub fn unproject(&self, x: f64, y: f64) -> Option<Vec<f64>> {
            self.app.borrow().controller.unproject(x, y).map(|(lat, lon)| vec![lat, lon])
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

        /// Add a raster tile layer. Returns the layer index.
        ///
        /// Parameters:
        /// - `name`: unique layer name
        /// - `url`: tile URL template with {z}/{x}/{y}
        /// - `z_order`: stacking order (optional, default 0)
        #[wasm_bindgen(js_name = "addLayer")]
        pub fn add_layer(&self, name: &str, url: &str, z_order: Option<i32>) -> usize {
            let config = LayerConfig {
                name: name.to_string(),
                tile_source_url: url.to_string(),
                z_order: z_order.unwrap_or(0),
                kind: LayerKind::Raster,
                ..Default::default()
            };
            let max_cached = config.max_cached_tiles;
            let max_concurrent = config.max_concurrent_loads;
            let mut app = self.app.borrow_mut();
            let idx = app.controller.add_layer(config);
            // Add corresponding WebLayerState
            app.add_layer_state(name, url, max_cached, max_concurrent);
            idx
        }

        /// Get layer info as a JS object. Returns null if not found.
        ///
        /// Object shape: { name, url, opacity, visible, zOrder, kind }
        #[wasm_bindgen(js_name = "getLayer")]
        pub fn get_layer(&self, name: &str) -> JsValue {
            match self.app.borrow().controller.get_layer_info(name) {
                Some(info) => {
                    let obj = js_sys::Object::new();
                    let _ = js_sys::Reflect::set(&obj, &"name".into(), &info.name.into());
                    let _ = js_sys::Reflect::set(&obj, &"url".into(), &info.url.into());
                    let _ = js_sys::Reflect::set(&obj, &"opacity".into(), &(info.opacity as f64).into());
                    let _ = js_sys::Reflect::set(&obj, &"visible".into(), &info.visible.into());
                    let _ = js_sys::Reflect::set(&obj, &"zOrder".into(), &info.z_order.into());
                    let _ = js_sys::Reflect::set(&obj, &"kind".into(), &info.kind.into());
                    obj.into()
                }
                None => JsValue::NULL,
            }
        }

        /// Get all layer names in z-order (bottom to top).
        #[wasm_bindgen(js_name = "getLayers")]
        pub fn get_layers(&self) -> Vec<String> {
            self.app.borrow().controller.layer_names()
        }

        // ── Viewport ──

        /// Resize the map viewport.
        pub fn resize(&self, width: u32, height: u32) {
            self.app.borrow_mut().controller.resize(width, height);
        }

        // ── Camera Limits ──

        /// Get the minimum zoom level.
        #[wasm_bindgen(js_name = "getMinZoom")]
        pub fn get_min_zoom(&self) -> f64 {
            self.app.borrow().controller.min_zoom()
        }

        /// Set the minimum zoom level.
        #[wasm_bindgen(js_name = "setMinZoom")]
        pub fn set_min_zoom(&self, zoom: f64) {
            self.app.borrow_mut().controller.set_min_zoom(zoom);
        }

        /// Get the maximum zoom level.
        #[wasm_bindgen(js_name = "getMaxZoom")]
        pub fn get_max_zoom(&self) -> f64 {
            self.app.borrow().controller.max_zoom()
        }

        /// Set the maximum zoom level.
        #[wasm_bindgen(js_name = "setMaxZoom")]
        pub fn set_max_zoom(&self, zoom: f64) {
            self.app.borrow_mut().controller.set_max_zoom(zoom);
        }

        /// Get the maximum pitch angle in degrees.
        #[wasm_bindgen(js_name = "getMaxPitch")]
        pub fn get_max_pitch(&self) -> f64 {
            self.app.borrow().controller.max_pitch()
        }

        /// Set the maximum pitch angle in degrees.
        #[wasm_bindgen(js_name = "setMaxPitch")]
        pub fn set_max_pitch(&self, degrees: f64) {
            self.app.borrow_mut().controller.set_max_pitch(degrees);
        }

        /// Get the tile budget (max tiles per frame).
        #[wasm_bindgen(js_name = "getTileBudget")]
        pub fn get_tile_budget(&self) -> usize {
            self.app.borrow().controller.tile_budget()
        }

        /// Set the tile budget (max tiles per frame).
        #[wasm_bindgen(js_name = "setTileBudget")]
        pub fn set_tile_budget(&self, budget: usize) {
            self.app.borrow_mut().controller.set_tile_budget(budget);
        }

        // ── Events ──

        /// Register an event listener. Supported events:
        /// "move", "zoom", "pitch", "bearing", "moveend", "zoomend", "click"
        pub fn on(&self, event: &str, callback: js_sys::Function) {
            self.app.borrow_mut()
                .event_handlers
                .entry(event.to_string())
                .or_default()
                .push(callback);
        }

        /// Remove an event listener.
        pub fn off(&self, event: &str, callback: js_sys::Function) {
            if let Some(handlers) = self.app.borrow_mut().event_handlers.get_mut(event) {
                handlers.retain(|f| f != &callback);
            }
        }

        // ── Lifecycle ──

        /// Destroy the map instance, stopping the render loop and cleaning up resources.
        ///
        /// After calling this, the map will no longer render and all event
        /// listeners will be removed.
        pub fn destroy(&self) {
            let mut app = self.app.borrow_mut();
            app.destroyed = true;
            app.event_handlers.clear();
            log::info!("x-planets map destroyed");
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // XPlanets — Factory for Promise-based initialization
    // ═══════════════════════════════════════════════════════════════

    /// Factory for creating x-planets map instances.
    ///
    /// ```javascript
    /// const map = await XPlanets.create("my-canvas", { center: [37.5, 127], zoom: 10 });
    /// map.on("move", console.log);
    /// ```
    #[wasm_bindgen]
    pub struct XPlanets;

    #[wasm_bindgen]
    impl XPlanets {
        /// Create a new x-planets map instance on the given canvas element.
        ///
        /// - `canvas_id`: The DOM id of the canvas element.
        /// - `config`: Optional configuration object with:
        ///   - `center`: `[lat, lon]` (default: [0, 0])
        ///   - `zoom`: number (default: 2)
        ///   - `projection`: string (default: "Web Mercator")
        ///   - `layers`: array of `{ name, url, kind?, zOrder? }`
        pub async fn create(canvas_id: &str, config: JsValue) -> Result<XPlanetsMap, JsValue> {
            console_error_panic_hook::set_once();
            console_log::init_with_level(log::Level::Info).ok();

            let window = web_sys::window().ok_or("No window")?;
            let document = window.document().ok_or("No document")?;
            let canvas = document
                .get_element_by_id(canvas_id)
                .ok_or_else(|| JsValue::from_str(&format!("Canvas '{}' not found", canvas_id)))?
                .dyn_into::<web_sys::HtmlCanvasElement>()?;

            // DPR-aware canvas sizing
            let dpr = window.device_pixel_ratio();
            let css_w = canvas.client_width() as f64;
            let css_h = canvas.client_height() as f64;
            let width = (css_w * dpr).max(1.0) as u32;
            let height = (css_h * dpr).max(1.0) as u32;
            canvas.set_width(width);
            canvas.set_height(height);

            // GPU
            let surface_target = wgpu::SurfaceTarget::Canvas(canvas.clone());
            let gpu = x_planets_gpu::GpuContext::new_with_window(surface_target, width, height)
                .await
                .map_err(|e| JsValue::from_str(&format!("GPU init failed: {}", e)))?;

            // Parse JS config
            let map_config = parse_js_config(&config);

            // Set projection after creation if specified
            let projection_name = if !config.is_undefined() && !config.is_null() {
                js_sys::Reflect::get(&config, &"projection".into())
                    .ok()
                    .and_then(|v| v.as_string())
            } else {
                None
            };

            let mut controller = x_planets_core::MapController::new(map_config, width, height);
            if let Some(proj) = projection_name {
                controller.set_projection(&proj);
            }

            let renderer = x_planets_render::TileRenderer::new(&gpu);
            let terrain_renderer = x_planets_render::TerrainRenderer::new(&gpu);
            let tex_manager = x_planets_gpu::TextureManager::new(&gpu.device);

            let app = WebApp::new(gpu, controller, renderer, terrain_renderer, tex_manager, canvas.clone(), dpr);
            let app = std::rc::Rc::new(std::cell::RefCell::new(app));

            // Input events
            crate::input::register_events(&canvas, std::rc::Rc::clone(&app));

            // Start render loop
            WebApp::start_render_loop(std::rc::Rc::clone(&app));

            Ok(XPlanetsMap { app })
        }
    }

    /// Parse a JS config object into a [`MapConfig`].
    fn parse_js_config(val: &JsValue) -> x_planets_core::engine::MapConfig {
        let mut config = x_planets_core::engine::MapConfig::default();
        if val.is_undefined() || val.is_null() {
            return config;
        }

        // center: [lat, lon]
        if let Ok(center) = js_sys::Reflect::get(val, &"center".into()) {
            if let Some(arr) = center.dyn_ref::<js_sys::Array>() {
                if arr.length() >= 2 {
                    if let (Some(lat), Some(lon)) = (arr.get(0).as_f64(), arr.get(1).as_f64()) {
                        config.center = x_planets_math::GeoCoord::new(lat, lon);
                    }
                }
            }
        }

        // zoom: number
        if let Ok(zoom) = js_sys::Reflect::get(val, &"zoom".into()) {
            if let Some(z) = zoom.as_f64() {
                config.zoom = z;
            }
        }

        // minZoom: number
        if let Ok(v) = js_sys::Reflect::get(val, &"minZoom".into()) {
            if let Some(n) = v.as_f64() {
                config.min_zoom = n;
            }
        }

        // maxZoom: number
        if let Ok(v) = js_sys::Reflect::get(val, &"maxZoom".into()) {
            if let Some(n) = v.as_f64() {
                config.max_zoom = n;
            }
        }

        // maxPitch: number
        if let Ok(v) = js_sys::Reflect::get(val, &"maxPitch".into()) {
            if let Some(n) = v.as_f64() {
                config.max_pitch = n;
            }
        }

        // tileBudget: number
        if let Ok(v) = js_sys::Reflect::get(val, &"tileBudget".into()) {
            if let Some(n) = v.as_f64() {
                config.tile_budget = n as usize;
            }
        }

        // layers: [{ name, url, kind?, zOrder?, opacity? }]
        if let Ok(layers) = js_sys::Reflect::get(val, &"layers".into()) {
            if let Some(arr) = layers.dyn_ref::<js_sys::Array>() {
                let mut layer_configs = Vec::new();
                for i in 0..arr.length() {
                    let item = arr.get(i);
                    let name = js_sys::Reflect::get(&item, &"name".into())
                        .ok().and_then(|v| v.as_string()).unwrap_or_default();
                    let url = js_sys::Reflect::get(&item, &"url".into())
                        .ok().and_then(|v| v.as_string()).unwrap_or_default();
                    let z_order = js_sys::Reflect::get(&item, &"zOrder".into())
                        .ok().and_then(|v| v.as_f64()).unwrap_or(i as f64) as i32;
                    let opacity = js_sys::Reflect::get(&item, &"opacity".into())
                        .ok().and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

                    if !name.is_empty() && !url.is_empty() {
                        layer_configs.push(LayerConfig {
                            name,
                            tile_source_url: url,
                            z_order,
                            opacity,
                            kind: LayerKind::Raster,
                            ..Default::default()
                        });
                    }
                }
                if !layer_configs.is_empty() {
                    config.layers = layer_configs;
                }
            }
        }

        config
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
