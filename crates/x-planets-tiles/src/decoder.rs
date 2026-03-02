//! Tile decoders for various tile formats.

use async_trait::async_trait;
use thiserror::Error;
use x_planets_math::TileCoord;

#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("Failed to decode image: {0}")]
    ImageDecode(String),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("Invalid tile data")]
    InvalidData,
}

/// Trait for decoding raw tile bytes into a usable format.
///
/// On native: requires `Send + Sync` for multi-threaded decoding.
/// On wasm32: single-threaded, no Send/Sync required.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait TileDecoder: Send + Sync {
    type Output: Send + Sync;

    /// Decode raw bytes into the tile's output format.
    async fn decode(&self, coord: TileCoord, data: &[u8]) -> Result<Self::Output, DecodeError>;

    /// File extension this decoder handles.
    fn extension(&self) -> &str;
}

/// Trait for decoding raw tile bytes (wasm32 — single-threaded).
#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait TileDecoder {
    type Output;

    /// Decode raw bytes into the tile's output format.
    async fn decode(&self, coord: TileCoord, data: &[u8]) -> Result<Self::Output, DecodeError>;

    /// File extension this decoder handles.
    fn extension(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Raster Tile Decoder (PNG / JPEG → RGBA pixels)
// ---------------------------------------------------------------------------

/// Decoded raster tile: RGBA pixel data ready for GPU upload.
pub struct DecodedRasterTile {
    pub coord: TileCoord,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>, // RGBA8
}

/// Decodes PNG/JPEG tile images into RGBA pixel data.
pub struct RasterTileDecoder {
    /// Expected tile size (typically 256 or 512).
    pub tile_size: u32,
}

impl RasterTileDecoder {
    pub fn new(tile_size: u32) -> Self {
        Self { tile_size }
    }
}

impl Default for RasterTileDecoder {
    fn default() -> Self {
        Self { tile_size: 256 }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl TileDecoder for RasterTileDecoder {
    type Output = DecodedRasterTile;

    async fn decode(&self, coord: TileCoord, data: &[u8]) -> Result<Self::Output, DecodeError> {
        // Decode image using the `image` crate
        let img = image::load_from_memory(data)
            .map_err(|e| DecodeError::ImageDecode(e.to_string()))?;

        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();

        Ok(DecodedRasterTile {
            coord,
            width,
            height,
            pixels: rgba.into_raw(),
        })
    }

    fn extension(&self) -> &str {
        "png"
    }
}

// ---------------------------------------------------------------------------
// Placeholder decoders for future implementation
// ---------------------------------------------------------------------------

/// Decoded vector tile containing feature geometry.
pub struct DecodedVectorTile {
    pub coord: TileCoord,
    pub layers: Vec<VectorLayer>,
}

/// A layer within a vector tile.
pub struct VectorLayer {
    pub name: String,
    pub features: Vec<VectorFeature>,
}

/// A single feature in a vector tile.
pub struct VectorFeature {
    pub geometry_type: GeometryType,
    pub coordinates: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryType {
    Point,
    LineString,
    Polygon,
}

/// Decoded terrain tile containing elevation data.
pub struct DecodedTerrainTile {
    pub coord: TileCoord,
    pub width: u32,
    pub height: u32,
    pub elevation: Vec<f32>,
    pub min_elevation: f32,
    pub max_elevation: f32,
}

// ---------------------------------------------------------------------------
// Terrain RGB Decoder (Mapbox Terrain RGB → elevation)
// ---------------------------------------------------------------------------

/// Decodes Mapbox Terrain RGB tiles into elevation data.
///
/// Height formula: `height = -10000 + ((R * 256 * 256 + G * 256 + B) * 0.1)`
///
/// Data source: `https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}@2x.pngraw?access_token=TOKEN`
pub struct TerrainRgbDecoder;

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl TileDecoder for TerrainRgbDecoder {
    type Output = DecodedTerrainTile;

    async fn decode(&self, coord: TileCoord, data: &[u8]) -> Result<Self::Output, DecodeError> {
        let img = image::load_from_memory(data)
            .map_err(|e| DecodeError::ImageDecode(e.to_string()))?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();

        let mut elevation = Vec::with_capacity((width * height) as usize);
        let mut min_elev = f32::MAX;
        let mut max_elev = f32::MIN;

        for pixel in rgba.pixels() {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            let h = -10000.0 + (r * 256.0 * 256.0 + g * 256.0 + b) * 0.1;
            min_elev = min_elev.min(h);
            max_elev = max_elev.max(h);
            elevation.push(h);
        }

        Ok(DecodedTerrainTile {
            coord,
            width,
            height,
            elevation,
            min_elevation: min_elev,
            max_elevation: max_elev,
        })
    }

    fn extension(&self) -> &str {
        "pngraw"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_raster_decoder_invalid_data() {
        let decoder = RasterTileDecoder::default();
        let coord = TileCoord::new(0, 0, 0);
        let result = decoder.decode(coord, b"not an image").await;
        assert!(result.is_err());
    }
}
