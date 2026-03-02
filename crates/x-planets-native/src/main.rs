//! x-planets native viewer binary.
//!
//! Usage: cargo run -p x-planets-native

use x_planets_core::engine::MapConfig;
use x_planets_math::GeoCoord;

fn main() {
    let config = MapConfig {
        center: GeoCoord::new(37.5665, 126.9780), // Seoul
        zoom: 5.0,
        tile_source_url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png".to_string(),
        max_concurrent_loads: 8,
        max_cached_tiles: 512,
        ..Default::default()
    };

    if let Err(e) = x_planets_native::run_native(config) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
