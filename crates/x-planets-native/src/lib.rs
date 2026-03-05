//! x-planets-native: Native platform backend.
//!
//! Provides desktop windowing (winit), async I/O (tokio),
//! and HTTP client (reqwest) implementations.
//!
//! Supports raster, terrain, and OGC 3D Tiles layers.

pub mod config;
mod app;
mod tile_source;
mod tilejson;
mod tiles3d_native;

pub use app::run_native;
pub use tile_source::NativeTileSource;

/// Shared HTTP client constructor used across the crate.
pub(crate) fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("x-planets/0.1")
        .build()
        .expect("Failed to create HTTP client")
}
