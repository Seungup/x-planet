//! x-planets-tiles: Tile loading, decoding, and caching.
//!
//! Provides an async pipeline for fetching, decoding, and caching map tiles
//! of various formats (raster, vector, terrain).

pub mod cache;
pub mod decoder;
pub mod loader;
pub mod quantized_mesh;
pub mod tiles3d;

pub use cache::TileCache;
pub use decoder::{
    DecodedRasterTile, DecodedTerrainTile, RasterTileDecoder, TerrainEncoding, TerrainRgbDecoder,
    TerrariumDecoder, TileDecoder,
};
pub use loader::{LoadError, TileLoader, TileRequest, TileSource};
pub use quantized_mesh::{parse_quantized_mesh, DecodedQuantizedMesh};
