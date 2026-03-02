//! x-planets-web: Web/WASM platform backend.
//!
//! Provides browser-based implementations for canvas rendering,
//! fetch API tile loading, and IndexedDB caching.
//!
//! This crate is compiled to WebAssembly and exposed to JavaScript
//! via wasm-bindgen.

#[cfg(target_arch = "wasm32")]
mod web_impl {
    use wasm_bindgen::prelude::*;

    /// Initialize the WASM module.
    #[wasm_bindgen(start)]
    pub fn wasm_init() {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).unwrap();
        log::info!("x-planets WASM module initialized");
    }

    /// Create and run the map engine on a canvas element.
    #[wasm_bindgen]
    pub async fn create_map(canvas_id: &str) -> Result<(), JsValue> {
        log::info!("Creating map on canvas: {}", canvas_id);

        let window = web_sys::window().ok_or("No window")?;
        let document = window.document().ok_or("No document")?;
        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or("Canvas not found")?
            .dyn_into::<web_sys::HtmlCanvasElement>()?;

        let width = canvas.client_width() as u32;
        let height = canvas.client_height() as u32;

        log::info!("Canvas size: {}x{}", width, height);

        // TODO: Initialize wgpu surface from canvas
        // TODO: Create MapEngine and start render loop

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
    ///
    /// Fetches tile images via `window.fetch()` and returns raw bytes.
    /// Designed for single-threaded wasm32 execution.
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
