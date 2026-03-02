//! TileJSON 2.x/3.x endpoint resolution.
//!
//! Fetches a TileJSON metadata endpoint and extracts the tile URL template,
//! TMS flag, zoom range, scale, and auto-detected terrain encoding.

use x_planets_tiles::TerrainEncoding;

/// Minimal TileJSON 2.x/3.x metadata — only the fields we need.
///
/// Spec: <https://github.com/mapbox/tilejson-spec>
#[derive(serde::Deserialize)]
struct TileJson {
    /// One or more tile URL templates (e.g. `"https://…/{z}/{x}/{y}.webp"`).
    tiles: Vec<String>,
    #[serde(default)]
    minzoom: Option<u8>,
    #[serde(default)]
    maxzoom: Option<u8>,
    #[serde(default)]
    name: Option<String>,
    /// TileJSON 2.x uses `"scheme"`, but MapTiler uses `"schema"`.
    /// Accept both spellings with serde alias.
    #[serde(default, alias = "schema")]
    scheme: Option<String>,
    /// Tile pixel density (e.g. `"1.000000"` = 256×256, `"2.000000"` = 512×512).
    /// MapTiler extension; not in the original TileJSON spec.
    #[serde(default)]
    scale: Option<String>,
    /// Tile data format, e.g. `"quantized-mesh-1.0"`, `"terrarium"`, `"webp"`, `"png"`.
    /// Used to auto-detect terrain encoding without requiring explicit config.
    #[serde(default)]
    format: Option<String>,
    /// Tile projection, e.g. `"EPSG:4326"`, `"EPSG:3857"`.
    /// When `EPSG:4326`, tiles use a geographic grid (2:1 at zoom 0).
    #[serde(default)]
    projection: Option<String>,
}

/// Resolved metadata from a TileJSON endpoint.
pub(crate) struct TileJsonMeta {
    /// The first tile URL template from the `tiles` array.
    pub tile_url: String,
    /// Whether the tile scheme is TMS (y-axis flipped).
    pub tms: bool,
    /// Minimum zoom level served by the source.
    pub min_zoom: Option<u8>,
    /// Maximum zoom level served by the source.
    pub max_zoom: Option<u8>,
    /// Tile pixel density (1.0 = 256px, 2.0 = 512px).
    pub scale: f32,
    /// Terrain encoding auto-detected from the TileJSON `format` field.
    /// `None` if the format is not a recognized terrain format (raster).
    pub detected_encoding: Option<TerrainEncoding>,
    /// `true` when the tile source uses EPSG:4326 (geographic) tile grid.
    /// EPSG:4326 has a 2:1 tile grid (2^(z+1) × 2^z) and requires
    /// coordinate conversion from the engine's EPSG:3857 tile coordinates.
    pub geographic: bool,
}

impl TileJsonMeta {
    /// Create a passthrough meta with defaults (no TileJSON resolution).
    pub fn passthrough(tile_url: String) -> Self {
        Self {
            tile_url,
            tms: false,
            min_zoom: None,
            max_zoom: None,
            scale: 1.0,
            detected_encoding: None,
            geographic: false,
        }
    }
}

/// Returns `true` if the URL looks like a TileJSON endpoint
/// (ends in `.json` but is not a 3D Tiles `tileset.json`).
pub(crate) fn is_tilejson_url(url: &str) -> bool {
    let lower = url.split('?').next().unwrap_or(url).to_ascii_lowercase();
    lower.ends_with(".json")
        && !lower.ends_with("tileset.json")
}

/// Fetch a TileJSON endpoint and extract tile URL template + metadata.
///
/// Returns [`TileJsonMeta`] with the resolved URL, TMS flag, zoom range, and scale.
pub(crate) async fn resolve_tilejson(
    client: &reqwest::Client,
    url: &str,
) -> Result<TileJsonMeta, String> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("TileJSON fetch failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("TileJSON HTTP {}", resp.status()));
    }
    let json: TileJson = resp
        .json()
        .await
        .map_err(|e| format!("TileJSON parse failed: {e}"))?;
    let tile_url = json
        .tiles
        .into_iter()
        .next()
        .ok_or_else(|| "TileJSON has empty `tiles` array".to_string())?;
    let tms = json
        .scheme
        .as_deref()
        .map(|s| s.eq_ignore_ascii_case("tms"))
        .unwrap_or(false);
    let scale = json
        .scale
        .as_deref()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0);

    // Auto-detect terrain encoding from TileJSON `format` field.
    let detected_encoding = match json.format.as_deref() {
        Some(f) if f.starts_with("quantized-mesh") => Some(TerrainEncoding::QuantizedMesh),
        Some("terrarium") => Some(TerrainEncoding::Terrarium),
        _ => None,
    };

    // Detect EPSG:4326 (geographic) tile grid from `projection` field.
    // When geographic, EPSG:3857 tile coordinates must be converted to
    // EPSG:4326 before building the tile URL.
    let geographic = json
        .projection
        .as_deref()
        .map(|p| p.contains("4326"))
        .unwrap_or(false);

    log::info!(
        "TileJSON resolved: \"{}\" (zoom {}-{}, scale={}x, tms={}, format={:?}, projection={}, geographic={})",
        json.name.as_deref().unwrap_or("(unnamed)"),
        json.minzoom.unwrap_or(0),
        json.maxzoom.unwrap_or(22),
        scale,
        tms,
        json.format.as_deref().unwrap_or("(none)"),
        json.projection.as_deref().unwrap_or("(default/3857)"),
        geographic,
    );
    Ok(TileJsonMeta {
        tile_url,
        tms,
        min_zoom: json.minzoom,
        max_zoom: json.maxzoom,
        scale,
        detected_encoding,
        geographic,
    })
}
