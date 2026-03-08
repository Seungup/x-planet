//! Native HTTP tile source and per-layer GPU state.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use x_planets_core::engine::LayerKind;
use x_planets_core::map_controller::LayerStateView;
use x_planets_render::TerrainTileData;
use x_planets_gpu::GpuTexture;
use x_planets_math::TileCoord;
use x_planets_tiles::{
    DecodedRasterTile, DecodedTerrainTile, TileCache, TileLoader, TileSource,
};

/// A decoded tile result, discriminated by layer kind.
pub(crate) enum TileResult {
    Raster(DecodedRasterTile),
    Terrain(DecodedTerrainTile),
    /// Quantized Mesh 1.0 terrain tile (pre-built triangle mesh).
    QuantizedMesh(Box<x_planets_tiles::DecodedQuantizedMesh>),
}

/// Result from a layer tile fetch+decode, tagged with the layer name.
pub(crate) struct LayerTileResult {
    pub layer_name: String,
    pub result: Result<TileResult, (TileCoord, String)>,
}

/// Per-layer GPU state: tile source, texture cache, loader, pending set.
pub(crate) struct NativeLayerState {
    pub name: String,
    pub kind: LayerKind,
    pub tile_source: Arc<NativeTileSource>,
    pub tile_textures: TileCache<GpuTexture>,
    pub tile_loader: TileLoader,
    pub pending_coords: HashSet<TileCoord>,
    /// Elevation data for terrain layers (CPU-side, used for mesh generation).
    /// LRU-evicted alongside tile_textures to prevent unbounded memory growth.
    pub terrain_data: TileCache<TerrainTileData>,
    /// Cooldown for failed tiles: don't retry until the Instant has passed.
    /// Prevents infinite retry loops when the server returns 429 / transient errors.
    pub failed_cooldowns: HashMap<TileCoord, Instant>,
    /// Minimum zoom level served by the tile source (from TileJSON `minzoom`).
    pub min_zoom: u8,
    /// Maximum zoom level served by the tile source (from TileJSON `maxzoom`).
    /// Tiles beyond this zoom are never requested; the fallback system
    /// renders them with parent tiles at `max_zoom`.
    pub max_zoom: u8,
    /// Tile pixel density (1.0 = 256px, 2.0 = 512px).
    /// Parsed from TileJSON `scale` field.  Reserved for future LOD calculations.
    #[allow(dead_code)]
    pub tile_scale: f32,
    /// True when the tile source uses EPSG:4326 (geographic) tile grid.
    /// QM tiles from geographic sources must be resampled to EPSG:3857 space
    /// before rendering (the engine uses a Mercator tile grid).
    pub geographic: bool,
    /// Cache of rasterized EPSG:4326 heightmaps, keyed by `(gx, gy, gz)`.
    ///
    /// When a 3857 tile straddles two 4326 tiles, the multi-source resampling
    /// function can look up both heightmaps from this cache to produce a
    /// seamless 3857 heightmap without cliff walls.
    pub geo_heightmap_cache: HashMap<(u32, u32, u8), GeoHeightmapEntry>,
    /// Cached set of available raster tile coords (rebuilt each frame).
    pub available_coords_cache: HashSet<TileCoord>,
    /// Elevation tile source (set when terrain is toggled on for this raster layer).
    pub elevation_source: Option<Arc<NativeTileSource>>,
    /// Pending elevation tile fetches (separate from raster pending).
    pub pending_elevation_coords: HashSet<TileCoord>,
    /// Maximum concurrent elevation tile loads.
    pub max_elevation_concurrent: usize,
}

impl NativeLayerState {
    /// Refresh the cached available coords set from tile_textures.
    pub fn refresh_available_cache(&mut self) {
        self.available_coords_cache = self.tile_textures.keys().copied().collect();
    }
}

impl LayerStateView for NativeLayerState {
    fn name(&self) -> &str {
        &self.name
    }

    fn available_raster_coords(&self) -> &HashSet<TileCoord> {
        &self.available_coords_cache
    }

    fn terrain_tile_data(&self, coord: &TileCoord) -> Option<&TerrainTileData> {
        self.terrain_data.peek(coord)
    }
}

/// Native HTTP-based tile source using reqwest.
pub struct NativeTileSource {
    client: reqwest::Client,
    url_template: String,
    tms: bool,
    /// True when the tile source uses EPSG:4326 (geographic) tile grid.
    /// In this case, EPSG:3857 (Web Mercator) tile coordinates are converted
    /// to EPSG:4326 before building the URL.
    geographic: bool,
}

impl NativeTileSource {
    pub fn new(url_template: impl Into<String>) -> Self {
        Self {
            client: crate::http_client(),
            url_template: url_template.into(),
            tms: false,
            geographic: false,
        }
    }

    pub fn with_tms(mut self, tms: bool) -> Self {
        self.tms = tms;
        self
    }

    pub fn with_geographic(mut self, geographic: bool) -> Self {
        self.geographic = geographic;
        self
    }
}

/// Convert an EPSG:3857 (Web Mercator) XYZ tile coordinate to the
/// corresponding EPSG:4326 (Geographic) XYZ tile coordinate.
///
/// Uses zoom **z−1** for the geographic grid so that a single 4326 tile
/// covers approximately the same longitude range as the 3857 tile:
/// - 3857 tile at z: width = 360/2^z degrees
/// - 4326 tile at z: width = 360/2^(z+1) = 180/2^z degrees (HALF!)
/// - 4326 tile at z−1: width = 360/2^z degrees (MATCHES 3857 at z)
///
/// EPSG:4326 has a 2:1 tile grid: `2^(z+1)` tiles in x, `2^z` tiles in y.
/// The y coordinate follows XYZ convention (0 = north).
pub(crate) fn mercator_to_geographic_tile(coord: &TileCoord) -> (u32, u32, u8) {
    // Use z-1 so the 4326 tile covers the same longitude extent as the 3857 tile.
    let gz = if coord.z > 0 { coord.z - 1 } else { 0 };

    let n_3857 = (1u64 << coord.z) as f64;

    // Center of the 3857 tile in geographic coordinates
    let lon = (coord.x as f64 + 0.5) / n_3857 * 360.0 - 180.0;
    let y_center = (coord.y as f64 + 0.5) / n_3857;
    let lat_rad = (std::f64::consts::PI * (1.0 - 2.0 * y_center)).sinh().atan();
    let lat = lat_rad.to_degrees();

    // EPSG:4326 tile grid at gz: 2^(gz+1) in x, 2^gz in y
    let x_count = (1u64 << (gz as u64 + 1)) as f64;
    let y_count = (1u64 << gz) as f64;

    let x = ((lon + 180.0) / 360.0 * x_count).floor() as u32;
    // XYZ: y=0 at north pole (90°N)
    let y = ((90.0 - lat) / 180.0 * y_count)
        .floor()
        .clamp(0.0, y_count - 1.0) as u32;

    (x.min((1u32 << (gz + 1)) - 1), y, gz)
}

/// Compute the geographic (lon/lat) bounds of an EPSG:4326 tile.
///
/// Returns `(west, east, north, south)` in degrees.
/// 4326 grid: `2^(z+1)` tiles in x, `2^z` tiles in y.
/// XYZ convention: y=0 at 90°N (north pole).
pub(crate) fn geographic_tile_bounds(x: u32, y: u32, z: u8) -> (f64, f64, f64, f64) {
    let x_count = (1u64 << (z as u64 + 1)) as f64;
    let y_count = (1u64 << z) as f64;

    let west = x as f64 / x_count * 360.0 - 180.0;
    let east = (x as f64 + 1.0) / x_count * 360.0 - 180.0;
    let north = 90.0 - y as f64 / y_count * 180.0;
    let south = 90.0 - (y as f64 + 1.0) / y_count * 180.0;

    (west, east, north, south)
}

/// Cached rasterized EPSG:4326 heightmap for multi-source resampling.
///
/// Stored by `(gx, gy, gz)` key so that adjacent 3857 tiles sharing
/// the same 4326 tile can look it up and eliminate cliff walls.
pub(crate) struct GeoHeightmapEntry {
    pub heightmap: Vec<f32>,
    pub grid_size: u32,
    pub west: f64,
    pub east: f64,
    pub north: f64,
    pub south: f64,
}

/// Return ALL EPSG:4326 tile coordinates that overlap the given 3857 tile.
///
/// The primary tile is the one returned by [`mercator_to_geographic_tile`].
/// If the 3857 tile's latitude extent goes beyond the primary tile's
/// geographic bounds, the adjacent 4326 tile (above or below) is included.
///
/// Returns a `Vec` of `(gx, gy, gz)` with the primary tile first.
pub(crate) fn overlapping_geographic_tiles(merc_coord: &TileCoord) -> Vec<(u32, u32, u8)> {
    let (gx, gy, gz) = mercator_to_geographic_tile(merc_coord);
    let (_geo_west, _geo_east, geo_north, geo_south) =
        geographic_tile_bounds(gx, gy, gz);

    let mut result = vec![(gx, gy, gz)];

    // Compute the 3857 tile's north/south latitudes.
    let n_3857 = (1u64 << merc_coord.z) as f64;
    let merc_y_top = std::f64::consts::PI
        * (1.0 - 2.0 * merc_coord.y as f64 / n_3857);
    let north_lat = merc_y_top.sinh().atan().to_degrees();

    let merc_y_bot = std::f64::consts::PI
        * (1.0 - 2.0 * (merc_coord.y as f64 + 1.0) / n_3857);
    let south_lat = merc_y_bot.sinh().atan().to_degrees();

    let y_count = (1u64 << gz) as u32;

    // If the 3857 tile extends beyond the primary 4326 tile's north edge,
    // include the 4326 tile above (gy - 1).
    if north_lat > geo_north + 0.001 && gy > 0 {
        result.push((gx, gy - 1, gz));
    }

    // If the 3857 tile extends beyond the primary 4326 tile's south edge,
    // include the 4326 tile below (gy + 1).
    if south_lat < geo_south - 0.001 && gy + 1 < y_count {
        result.push((gx, gy + 1, gz));
    }

    result
}

#[async_trait]
impl TileSource for NativeTileSource {
    async fn fetch(&self, coord: TileCoord) -> Result<Vec<u8>, x_planets_tiles::LoadError> {
        let url = self.tile_url(&coord);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| x_planets_tiles::LoadError::Network(e.to_string()))?;

        let status = response.status().as_u16();
        if status == 404 {
            return Err(x_planets_tiles::LoadError::NotFound(coord));
        }
        if !response.status().is_success() {
            return Err(x_planets_tiles::LoadError::HttpError { status, url });
        }

        response
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| x_planets_tiles::LoadError::Network(e.to_string()))
    }

    fn tile_url(&self, coord: &TileCoord) -> String {
        let (x, mut y, z) = if self.geographic {
            mercator_to_geographic_tile(coord)
        } else {
            (coord.x, coord.y, coord.z)
        };

        if self.tms {
            y = (1u32 << z) - 1 - y;
        }

        self.url_template
            .replace("{z}", &z.to_string())
            .replace("{x}", &x.to_string())
            .replace("{y}", &y.to_string())
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── mercator_to_geographic_tile ──────────────────────────────

    #[test]
    fn test_merc_to_geo_zoom_0() {
        // 3857 z=0: single tile covering the world.
        // gz = 0-1 = 0 (clamped), 4326 at z=0: 2×1 tiles.
        let (gx, gy, gz) = mercator_to_geographic_tile(&TileCoord::new(0, 0, 0));
        assert_eq!(gz, 0, "gz should be 0 for z_3857=0");
        // The center of 3857(0,0,0) is lon=0, lat=0.
        // 4326 at z=0: x_count=2, y_count=1
        // x = ((0+180)/360 * 2).floor() = 1
        // y = ((90-0)/180 * 1).floor() = 0
        assert_eq!(gx, 1, "gx for center at lon=0");
        assert_eq!(gy, 0, "gy for center at lat=0");
    }

    #[test]
    fn test_merc_to_geo_zoom_1_center() {
        // 3857 z=1: 2×2 grid. Tile (1,1) center: lon=90°, lat≈-40.98°
        // gz = 0, 4326 at z=0: 2×1 tiles
        let coord = TileCoord::new(1, 1, 1);
        let (gx, _gy, gz) = mercator_to_geographic_tile(&coord);
        assert_eq!(gz, 0, "gz = z-1 = 0");
        // lon = (1.5/2)*360 - 180 = 90°
        // x = ((90+180)/360 * 2).floor() = 1.5.floor() = 1
        assert_eq!(gx, 1);
    }

    #[test]
    fn test_merc_to_geo_zoom_reduces_by_one() {
        // At any z > 0, gz should be z - 1.
        for z in 1..=18u8 {
            let coord = TileCoord::new(z, 0, 0);
            let (_, _, gz) = mercator_to_geographic_tile(&coord);
            assert_eq!(gz, z - 1, "gz should be z-1 at z={}", z);
        }
    }

    #[test]
    fn test_merc_to_geo_4326_tile_grid_size() {
        // At gz, the 4326 grid has 2^(gz+1) × 2^gz tiles.
        // Returned x should be < 2^(gz+1), y should be < 2^gz.
        for z in 1..=15u8 {
            let n = 1u32 << z;
            for &(x, y) in &[(0, 0), (n / 2, n / 2), (n - 1, n - 1)] {
                let coord = TileCoord::new(z, x, y);
                let (gx, gy, gz) = mercator_to_geographic_tile(&coord);
                let max_x = (1u32 << (gz + 1)) - 1;
                let max_y = (1u32 << gz) - 1;
                assert!(
                    gx <= max_x,
                    "gx={} should be <= {} at z={} (x={}, y={})",
                    gx, max_x, z, x, y
                );
                assert!(
                    gy <= max_y,
                    "gy={} should be <= {} at z={} (x={}, y={})",
                    gy, max_y, z, x, y
                );
            }
        }
    }

    #[test]
    fn test_merc_to_geo_longitude_coverage() {
        // A 3857 tile at z covers 360/2^z degrees of longitude.
        // A 4326 tile at gz=z-1 covers 360/2^(gz+1) = 360/2^z degrees.
        // So one 4326 tile at gz should cover the same longitude as one 3857 tile at z.
        // The center of a 3857 tile should always fall within the returned 4326 tile.
        for z in 1..=12u8 {
            let n = 1u32 << z;
            for x in [0, n / 4, n / 2, 3 * n / 4, n - 1] {
                let coord = TileCoord::new(z, x, n / 2);
                let (gx, gy, gz) = mercator_to_geographic_tile(&coord);
                let (west, east, north, south) = geographic_tile_bounds(gx, gy, gz);

                // Compute 3857 tile center lon/lat
                let lon = (x as f64 + 0.5) / n as f64 * 360.0 - 180.0;
                let y_center = (n as f64 / 2.0 + 0.5) / n as f64;
                let lat_rad = (std::f64::consts::PI * (1.0 - 2.0 * y_center)).sinh().atan();
                let lat = lat_rad.to_degrees();

                assert!(
                    lon >= west && lon <= east,
                    "lon={:.2} should be in [{:.2}, {:.2}] at z={} x={}",
                    lon, west, east, z, x
                );
                assert!(
                    lat >= south && lat <= north,
                    "lat={:.2} should be in [{:.2}, {:.2}] at z={} x={}",
                    lat, south, north, z, x
                );
            }
        }
    }

    #[test]
    fn test_merc_to_geo_known_alps_tile() {
        // Alps region: roughly lon=10°, lat=47° → 3857 z=13
        // At z=13, n=8192. Tile containing lon=10°, lat=47°:
        // x = ((10+180)/360 * 8192).floor() = (190/360 * 8192) ≈ 4324
        // y: lat_rad = 47° → merc_y ≈ 0.372 → y ≈ (0.5 - 0.372/pi*0.5) * 8192
        // We'll use the coord directly and check the 4326 mapping.
        let coord = TileCoord::new(13, 4324, 2865);
        let (gx, gy, gz) = mercator_to_geographic_tile(&coord);
        assert_eq!(gz, 12);
        // At 4326 z=12: x_count=8192, y_count=4096
        // The 4326 tile should be near the same Alps region
        let (west, east, north, south) = geographic_tile_bounds(gx, gy, gz);
        assert!(west < 11.0 && east > 9.0, "should contain lon≈10° (got [{}, {}])", west, east);
        assert!(south < 48.0 && north > 46.0, "should contain lat≈47° (got [{}, {}])", south, north);
    }

    // ── geographic_tile_bounds ──────────────────────────────────

    #[test]
    fn test_geo_bounds_z0_full_world() {
        // At z=0: 2×1 tiles. Left tile covers [-180°, 0°] × [-90°, 90°].
        let (west, east, north, south) = geographic_tile_bounds(0, 0, 0);
        assert!((west - (-180.0)).abs() < 1e-10);
        assert!((east - 0.0).abs() < 1e-10);
        assert!((north - 90.0).abs() < 1e-10);
        assert!((south - (-90.0)).abs() < 1e-10);

        // Right tile covers [0°, 180°] × [-90°, 90°].
        let (west, east, north, south) = geographic_tile_bounds(1, 0, 0);
        assert!((west - 0.0).abs() < 1e-10);
        assert!((east - 180.0).abs() < 1e-10);
        assert!((north - 90.0).abs() < 1e-10);
        assert!((south - (-90.0)).abs() < 1e-10);
    }

    #[test]
    fn test_geo_bounds_adjacent_tiles_share_edges() {
        // Adjacent tiles at the same zoom should share edges perfectly.
        for z in 0..=6u8 {
            let x_count = 1u32 << (z + 1);
            let y_count = 1u32 << z;
            // Check x-adjacency
            if x_count > 1 {
                let (_, east0, _, _) = geographic_tile_bounds(0, 0, z);
                let (west1, _, _, _) = geographic_tile_bounds(1, 0, z);
                assert!(
                    (east0 - west1).abs() < 1e-10,
                    "east of tile 0 ({}) should equal west of tile 1 ({}) at z={}",
                    east0, west1, z
                );
            }
            // Check y-adjacency
            if y_count > 1 {
                let (_, _, _, south0) = geographic_tile_bounds(0, 0, z);
                let (_, _, north1, _) = geographic_tile_bounds(0, 1, z);
                assert!(
                    (south0 - north1).abs() < 1e-10,
                    "south of tile y=0 ({}) should equal north of tile y=1 ({}) at z={}",
                    south0, north1, z
                );
            }
        }
    }

    #[test]
    fn test_geo_bounds_tile_dimensions_consistent() {
        // All tiles at the same zoom should have the same width and height.
        for z in 0..=5u8 {
            let x_count = 1u32 << (z + 1);
            let y_count = 1u32 << z;
            let expected_w = 360.0 / x_count as f64;
            let expected_h = 180.0 / y_count as f64;

            for x in 0..x_count.min(4) {
                for y in 0..y_count.min(4) {
                    let (west, east, north, south) = geographic_tile_bounds(x, y, z);
                    let w = east - west;
                    let h = north - south;
                    assert!(
                        (w - expected_w).abs() < 1e-10,
                        "tile ({},{},{}) width {:.6} != expected {:.6}",
                        x, y, z, w, expected_w
                    );
                    assert!(
                        (h - expected_h).abs() < 1e-10,
                        "tile ({},{},{}) height {:.6} != expected {:.6}",
                        x, y, z, h, expected_h
                    );
                }
            }
        }
    }

    #[test]
    fn test_geo_bounds_north_south_ordering() {
        // North should always be > south for XYZ convention.
        for z in 0..=5u8 {
            let y_count = 1u32 << z;
            for y in 0..y_count.min(4) {
                let (_, _, north, south) = geographic_tile_bounds(0, y, z);
                assert!(
                    north > south,
                    "north ({}) should be > south ({}) at z={} y={}",
                    north, south, z, y
                );
            }
        }
    }

    // ── TMS y-flip ──────────────────────────────────────────────

    #[test]
    fn test_tms_y_flip_formula() {
        // The tile_url method applies TMS flip: y_tms = 2^z - 1 - y_xyz
        // Verify the formula is correct for known values.
        for z in 0..=5u8 {
            let max_y = (1u32 << z) - 1;
            // XYZ y=0 (north) → TMS y=max (north)
            assert_eq!((1u32 << z) - 1 - 0, max_y);
            // XYZ y=max (south) → TMS y=0 (south)
            assert_eq!((1u32 << z) - 1 - max_y, 0);
        }
    }

    #[test]
    fn test_tile_url_geographic_with_tms() {
        // Verify that geographic+TMS tile URLs use the correct z/x/y.
        let source = NativeTileSource::new("https://example.com/{z}/{x}/{y}.terrain")
            .with_tms(true)
            .with_geographic(true);

        // 3857 z=1, x=1, y=0 (top-left in XYZ)
        let coord = TileCoord::new(1, 1, 0);
        let url = source.tile_url(&coord);

        // gz = 0. 4326 at z=0: 2×1 tiles.
        // Center of 3857(1, 0, 1): lon=90°, lat≈66.5°
        // gx = ((90+180)/360 * 2).floor() = 1
        // gy = ((90 - 66.5) / 180 * 1).floor() = 0
        // TMS flip: y_tms = (1 << 0) - 1 - gy = 0 → same since gz=0 has max_y=0.
        assert!(url.contains("/0/"), "URL should contain gz=0: {}", url);
    }

    #[test]
    fn test_tile_url_xyz_no_geo_no_tms() {
        let source = NativeTileSource::new("https://example.com/{z}/{x}/{y}.png");
        let coord = TileCoord::new(5, 10, 15);
        let url = source.tile_url(&coord);
        assert_eq!(url, "https://example.com/5/10/15.png");
    }

    // ── Over-zoom coordinate clamping ───────────────────────────

    #[test]
    fn test_clamp_to_zoom_identity() {
        let coord = TileCoord::new(10, 500, 300);
        let clamped = coord.clamp_to_zoom(10);
        assert_eq!(clamped, coord);
    }

    #[test]
    fn test_clamp_to_zoom_reduces() {
        // z=15 tile → clamped to z=13
        let coord = TileCoord::new(15, 16384, 12288);
        let clamped = coord.clamp_to_zoom(13);
        assert_eq!(clamped.z, 13);
        // Each zoom level halves the coordinate
        assert_eq!(clamped.x, 16384 >> 2);
        assert_eq!(clamped.y, 12288 >> 2);
    }

    #[test]
    fn test_overzoom_children_map_to_same_parent() {
        // Four z=14 children mapping to z=13 should all give the same tile.
        let max_zoom: u8 = 13;
        let parent = TileCoord::new(13, 4096, 3072);
        let children = [
            TileCoord::new(14, 8192, 6144),
            TileCoord::new(14, 8193, 6144),
            TileCoord::new(14, 8192, 6145),
            TileCoord::new(14, 8193, 6145),
        ];
        for child in &children {
            let clamped = child.clamp_to_zoom(max_zoom);
            assert_eq!(
                clamped, parent,
                "child {:?} should clamp to {:?}, got {:?}",
                child, parent, clamped
            );
        }
    }

    // ── Geographic max_zoom adjustment ──────────────────────────

    #[test]
    fn test_geographic_maxzoom_plus_one() {
        // TileJSON maxzoom=13 (4326) → max_zoom_3857 = 14
        // Because gz = z_3857 - 1, so 3857 z=14 → gz=13 (the max 4326 zoom).
        let geo_maxzoom: u8 = 13;
        let max_zoom_3857 = geo_maxzoom.saturating_add(1);
        assert_eq!(max_zoom_3857, 14);

        // At z_3857=14, gz=13 (valid)
        // At z_3857=15, gz=14 (over-zoom: requires clamping to z_3857=14)
        let coord_14 = TileCoord::new(14, 8000, 6000);
        let (_, _, gz_14) = mercator_to_geographic_tile(&coord_14);
        assert_eq!(gz_14, 13, "z_3857=14 should map to gz=13");

        let coord_15 = TileCoord::new(15, 16000, 12000);
        let (_, _, gz_15) = mercator_to_geographic_tile(&coord_15);
        assert_eq!(gz_15, 14, "z_3857=15 would map to gz=14 (beyond source)");

        // With max_zoom_3857=14, z_3857=15 gets clamped to 14
        let clamped = coord_15.clamp_to_zoom(max_zoom_3857);
        assert_eq!(clamped.z, 14);
        let (_, _, gz_clamped) = mercator_to_geographic_tile(&clamped);
        assert_eq!(gz_clamped, 13, "clamped coord should give gz=13");
    }

    // ── Roundtrip: merc→geo→bounds→contains_center ─────────────

    #[test]
    fn test_merc_to_geo_roundtrip_multiple_zooms() {
        // For various 3857 tiles, verify the 4326 tile's bounds contain
        // the 3857 tile's center point.
        let test_tiles = [
            TileCoord::new(5, 16, 12),   // Europe
            TileCoord::new(8, 215, 100),  // East Asia
            TileCoord::new(10, 512, 380), // Mediterranean
            TileCoord::new(12, 2048, 1400), // Mid-latitudes
            TileCoord::new(14, 8640, 5740),  // Alps
        ];

        for coord in &test_tiles {
            let (gx, gy, gz) = mercator_to_geographic_tile(coord);
            let (west, east, north, south) = geographic_tile_bounds(gx, gy, gz);

            let n_3857 = (1u64 << coord.z) as f64;
            let lon = (coord.x as f64 + 0.5) / n_3857 * 360.0 - 180.0;
            let y_center = (coord.y as f64 + 0.5) / n_3857;
            let lat_rad = (std::f64::consts::PI * (1.0 - 2.0 * y_center)).sinh().atan();
            let lat = lat_rad.to_degrees();

            assert!(
                lon >= west - 1e-6 && lon <= east + 1e-6,
                "z={} ({},{}): lon={:.4} not in [{:.4}, {:.4}]",
                coord.z, coord.x, coord.y, lon, west, east
            );
            assert!(
                lat >= south - 1e-6 && lat <= north + 1e-6,
                "z={} ({},{}): lat={:.4} not in [{:.4}, {:.4}]",
                coord.z, coord.x, coord.y, lat, south, north
            );
        }
    }

    // ── overlapping_geographic_tiles ────────────────────────────

    #[test]
    fn test_overlapping_equator_no_secondary() {
        // Tile near equator: 3857 tile should fit within a single 4326 tile.
        let coord = TileCoord::new(5, 16, 15); // near equator
        let tiles = overlapping_geographic_tiles(&coord);
        assert_eq!(tiles.len(), 1, "Equatorial tile should need only 1 source");
    }

    #[test]
    fn test_overlapping_high_latitude_needs_secondary() {
        // Tile at higher latitude: 3857 tile extends beyond single 4326 tile.
        // z=5, y=12 covers ~[32°, 41°] but primary 4326 tile covers [33.75°, 45°].
        // The south part (32° to 33.75°) extends beyond → needs secondary.
        let coord = TileCoord::new(5, 27, 12);
        let tiles = overlapping_geographic_tiles(&coord);
        assert!(
            tiles.len() >= 2,
            "High-latitude tile should need 2 sources, got {}",
            tiles.len(),
        );

        // Primary should be first
        let (primary_gx, primary_gy, primary_gz) = tiles[0];
        let (_w, _e, _primary_north, primary_south) =
            geographic_tile_bounds(primary_gx, primary_gy, primary_gz);

        // Secondary should be adjacent (gy+1, covering south)
        let (_, secondary_gy, _) = tiles[1];
        assert_eq!(
            secondary_gy,
            primary_gy + 1,
            "Secondary should be the tile below (gy+1)"
        );

        // Verify the secondary tile covers the missing latitude range.
        let (_w2, _e2, sec_north, _sec_south) =
            geographic_tile_bounds(primary_gx, secondary_gy, primary_gz);
        assert!(
            (sec_north - primary_south).abs() < 1e-6,
            "Secondary's north edge ({}) should match primary's south edge ({})",
            sec_north, primary_south,
        );
    }

    #[test]
    fn test_overlapping_primary_always_first() {
        // The primary tile (from mercator_to_geographic_tile) should always
        // be the first element.
        for y in 0..32u32 {
            let coord = TileCoord::new(5, 16, y);
            let tiles = overlapping_geographic_tiles(&coord);
            let (primary_gx, primary_gy, primary_gz) =
                mercator_to_geographic_tile(&coord);
            assert_eq!(
                tiles[0],
                (primary_gx, primary_gy, primary_gz),
                "Primary tile should be first for y={}",
                y,
            );
        }
    }
}
