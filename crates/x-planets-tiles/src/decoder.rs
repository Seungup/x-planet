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
#[async_trait]
pub trait TileDecoder: Send + Sync {
    type Output: Send + Sync;

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

#[async_trait]
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
