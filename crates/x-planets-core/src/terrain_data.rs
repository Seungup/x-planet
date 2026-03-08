//! CPU-side terrain tile data (wgpu-free).
//!
//! Extracted from `terrain_renderer.rs` so that it can be used without
//! the `gpu` feature (e.g. in FFI bindings).

use crate::render::TerrainVertex;

/// CPU-side terrain data for a tile.
///
/// Two variants:
/// - `Heightmap`: regular elevation grid from Terrain RGB / Terrarium decoding.
/// - `PrebuiltMesh`: pre-built triangle mesh from Quantized Mesh 1.0 decoding.
///   Heights are stored in **metres** (not scaled).
pub enum TerrainTileData {
    /// Heightmap elevation grid (Terrain RGB / Terrarium).
    Heightmap {
        elevation: Vec<f32>,
        width: u32,
        height: u32,
    },
    /// Pre-built triangle mesh (Quantized Mesh 1.0).
    ///
    /// `positions[i][2]` is elevation in **metres** (not scaled by height_scale).
    /// Scaling is applied when building the GPU vertex buffer.
    PrebuiltMesh {
        /// Per-vertex data (position, normal, tex_coord).
        /// `position[2]` is raw metres; normal is in unscaled mesh space.
        vertices: Vec<TerrainVertex>,
        /// Triangle indices.
        indices: Vec<u32>,
        /// Regular grid heightmap rasterized from the QM mesh.
        /// Used for over-zoom fallback.
        fallback_heightmap: Vec<f32>,
        /// Side length of the square fallback heightmap grid.
        fallback_grid_size: u32,
    },
}
