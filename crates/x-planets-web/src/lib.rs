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

// Non-WASM stub for compilation checks
#[cfg(not(target_arch = "wasm32"))]
pub fn _placeholder() {
    // This module is only meaningful when compiled to wasm32.
}
