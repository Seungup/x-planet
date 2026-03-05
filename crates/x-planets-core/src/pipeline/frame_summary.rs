//! Stage 6: Frame summary (for Karpathy-style logging).

use x_planets_math::GeoCoord;

/// A snapshot of what happened in a single frame.
/// No state — just a value object for logging/debugging.
#[derive(Debug, Clone)]
pub struct FrameSummary {
    pub visible_tile_count: usize,
    pub cached_tile_count: usize,
    pub load_requests: usize,
    pub zoom: f64,
    pub center: GeoCoord,
}

impl std::fmt::Display for FrameSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "z={:.1} center=({:.2},{:.2}) visible={} cached={} pending={}",
            self.zoom,
            self.center.lat,
            self.center.lon,
            self.visible_tile_count,
            self.cached_tile_count,
            self.load_requests,
        )
    }
}
