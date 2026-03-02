//! Step 09: MapEngine + TileRenderer integration test.
//!
//! Runs the full native viewer with:
//! - MapEngine managing viewport state
//! - TileRenderer drawing checkerboard tiles at Mercator positions
//! - Keyboard pan/zoom controls
//!
//! Controls:
//!   Arrow keys — pan
//!   +/- — zoom
//!   Home — reset to center

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = x_planets_core::engine::MapConfig::default();
    x_planets_native::run_native(config)
}
