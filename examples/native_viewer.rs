//! Native Viewer Example
//!
//! A minimal desktop application that renders OpenStreetMap raster tiles
//! using the x-planets engine with Web Mercator projection.
//!
//! Usage:
//!   cargo run --example native_viewer

use x_planets_core::engine::{MapConfig, MapEngine};
use x_planets_math::GeoCoord;

fn main() {
    env_logger::init();
    log::info!("x-planets native viewer starting...");

    // Configure the map engine
    let config = MapConfig {
        center: GeoCoord::new(37.5665, 126.9780), // Seoul
        zoom: 5.0,
        projection: "Web Mercator".to_string(),
        tile_source_url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png".to_string(),
        max_concurrent_loads: 8,
        max_cached_tiles: 512,
    };

    // Create the engine
    let mut engine = MapEngine::new(config, 1280, 720);

    // Demonstrate the engine API
    log::info!("Active projection: {}", engine.active_projection
        .as_ref()
        .map(|p| p.name().to_string())
        .unwrap_or_else(|| {
            engine.active_projection()
                .map(|p| p.name().to_string())
                .unwrap_or_default()
        })
    );

    let visible = engine.viewport.visible_tiles();
    log::info!(
        "Viewport center: ({:.4}, {:.4}), zoom: {:.1}",
        engine.viewport.center.lat,
        engine.viewport.center.lon,
        engine.viewport.zoom,
    );
    log::info!("Visible tiles at zoom {}: {}", engine.viewport.tile_zoom(), visible.len());

    for tile in visible.iter().take(5) {
        log::info!("  Tile: {}", tile);
    }
    if visible.len() > 5 {
        log::info!("  ... and {} more", visible.len() - 5);
    }

    // Test pan and zoom
    engine.pan(100.0, 50.0);
    engine.zoom(1.0);

    let visible_after = engine.viewport.visible_tiles();
    log::info!(
        "After pan/zoom - center: ({:.4}, {:.4}), zoom: {:.1}, tiles: {}",
        engine.viewport.center.lat,
        engine.viewport.center.lon,
        engine.viewport.zoom,
        visible_after.len(),
    );

    // Switch projection
    if engine.set_projection("Globe") {
        log::info!("Switched to Globe projection");
    }

    log::info!("Layers: {}", engine.layer_count());
    for layer in &engine.layers {
        log::info!("  Layer '{}' (z={}, opacity={}, visible={})",
            layer.config.name, layer.config.z_order,
            layer.config.opacity, layer.config.visible);
    }

    log::info!("Native viewer demo complete.");
    log::info!("Full winit event loop integration coming in Phase 2.");
}
