//! Render pipeline for tile layers.

use std::collections::HashMap;

use x_planets_math::TileCoord;

use crate::pipeline::RenderableTile;

// ═══════════════════════════════════════════════════════════════════
// Per-frame render data (passed to TileRenderer per layer)
// ═══════════════════════════════════════════════════════════════════

/// Per-layer data assembled each frame and handed to `TileRenderer::render_frame_layered`.
pub struct RenderLayerData<'a> {
    /// Layer name (for debug labels).
    pub name: &'a str,
    /// Layer opacity (0.0–1.0).
    pub opacity: f32,
    /// Tiles with fallback resolution.
    pub tiles: Vec<RenderableTile>,
    /// Map from TileCoord → GPU TextureView (both own + fallback textures).
    pub texture_views: HashMap<TileCoord, &'a wgpu::TextureView>,
    /// Per-tile opacity overrides (for fade-in animation).
    /// If a tile's coord is in this map, use this opacity instead of layer opacity.
    pub tile_opacity_overrides: HashMap<TileCoord, f32>,
}

// ═══════════════════════════════════════════════════════════════════
// Legacy layer types (step07 example compat)
// ═══════════════════════════════════════════════════════════════════

/// Describes a tile ready to be rendered.
pub struct RenderTile {
    pub coord: TileCoord,
    /// Index into the texture atlas or bind group array.
    pub texture_index: usize,
    /// Opacity for blending (0.0 - 1.0).
    pub opacity: f32,
}

/// A layer of tiles to render.
pub struct TileRenderLayer {
    pub name: String,
    pub tiles: Vec<RenderTile>,
    pub visible: bool,
    pub opacity: f32,
}

impl TileRenderLayer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tiles: Vec::new(),
            visible: true,
            opacity: 1.0,
        }
    }
}

/// Ordered stack of tile layers for compositing.
pub struct LayerStack {
    layers: Vec<TileRenderLayer>,
}

impl LayerStack {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: TileRenderLayer) {
        self.layers.push(layer);
    }

    pub fn remove_layer(&mut self, name: &str) {
        self.layers.retain(|l| l.name != name);
    }

    pub fn get_layer(&self, name: &str) -> Option<&TileRenderLayer> {
        self.layers.iter().find(|l| l.name == name)
    }

    pub fn get_layer_mut(&mut self, name: &str) -> Option<&mut TileRenderLayer> {
        self.layers.iter_mut().find(|l| l.name == name)
    }

    /// Get all visible layers in render order (bottom to top).
    pub fn visible_layers(&self) -> impl Iterator<Item = &TileRenderLayer> {
        self.layers.iter().filter(|l| l.visible)
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

impl Default for LayerStack {
    fn default() -> Self {
        Self::new()
    }
}

/// Quad vertices for a single tile (Relative-To-Center).
///
/// Positions are relative to the tile center, NOT absolute Mercator coordinates.
/// This avoids f32 precision loss at high zoom levels.  The per-tile MVP matrix
/// (computed in f64 on the CPU) handles the translation to world/clip space.
///
/// At zoom z, each tile spans `1 / 2^z` in Mercator space, so the half-size
/// is `0.5 / 2^z`.  Vertices are at `(±hw, ±hh)` centered on the origin.
pub fn tile_quad_vertices(coord: &TileCoord) -> [TileVertex; 4] {
    let n = coord.extent() as f32;
    let hw = 0.5 / n; // half-width
    let hh = 0.5 / n; // half-height

    [
        TileVertex { position: [-hw, -hh], tex_coord: [0.0, 0.0] },
        TileVertex { position: [ hw, -hh], tex_coord: [1.0, 0.0] },
        TileVertex { position: [-hw,  hh], tex_coord: [0.0, 1.0] },
        TileVertex { position: [ hw,  hh], tex_coord: [1.0, 1.0] },
    ]
}

/// Indices for a tile quad (two triangles).
pub const TILE_QUAD_INDICES: [u32; 6] = [0, 1, 2, 2, 1, 3];

/// Vertex layout for tile rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileVertex {
    pub position: [f32; 2],
    pub tex_coord: [f32; 2],
}

impl TileVertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Vertex layout for terrain tile rendering (3D displaced positions + normals).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainVertex {
    /// Position in Mercator x, y + elevation z.
    pub position: [f32; 3],
    /// Surface normal (for hillshade lighting).
    pub normal: [f32; 3],
    /// UV for imagery texture draping.
    pub tex_coord: [f32; 2],
}

impl TerrainVertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress, // 32 bytes
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3, // position xyz
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3, // normal xyz
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2, // tex_coord
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_quad_vertices_rte() {
        // RTE: vertices are relative to tile center, NOT absolute Mercator.
        let coord = TileCoord::new(1, 0, 0);
        let verts = tile_quad_vertices(&coord);
        // At zoom 1, tile half-size = 0.5 / 2 = 0.25
        // Vertices should be at (-0.25, -0.25) to (0.25, 0.25)
        assert!((verts[0].position[0] - (-0.25)).abs() < 1e-6);
        assert!((verts[0].position[1] - (-0.25)).abs() < 1e-6);
        assert!((verts[3].position[0] - 0.25).abs() < 1e-6);
        assert!((verts[3].position[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_tile_quad_vertices_zoom0() {
        // At zoom 0, single tile has half-size = 0.5
        let coord = TileCoord::new(0, 0, 0);
        let verts = tile_quad_vertices(&coord);
        assert!((verts[0].position[0] - (-0.5)).abs() < 1e-6);
        assert!((verts[3].position[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tile_quad_vertices_area() {
        // Area should be (1/n)^2 regardless of tile position.
        for z in 0..=4u8 {
            let n = 1u32 << z;
            for x in 0..n {
                for y in 0..n {
                    let verts = tile_quad_vertices(&TileCoord::new(z, x, y));
                    let w = verts[1].position[0] - verts[0].position[0];
                    let h = verts[2].position[1] - verts[0].position[1];
                    let expected_size = 1.0 / n as f32;
                    assert!(
                        (w - expected_size).abs() < 1e-6,
                        "z={} x={} y={}: width {} != {}",
                        z, x, y, w, expected_size
                    );
                    assert!(
                        (h - expected_size).abs() < 1e-6,
                        "z={} x={} y={}: height {} != {}",
                        z, x, y, h, expected_size
                    );
                }
            }
        }
    }

    #[test]
    fn test_tile_quad_vertices_centered_at_origin() {
        // All tiles should have vertices centered around (0, 0).
        for z in 1..=3u8 {
            let n = 1u32 << z;
            for x in 0..n {
                for y in 0..n {
                    let verts = tile_quad_vertices(&TileCoord::new(z, x, y));
                    let cx = (verts[0].position[0] + verts[3].position[0]) / 2.0;
                    let cy = (verts[0].position[1] + verts[3].position[1]) / 2.0;
                    assert!(
                        cx.abs() < 1e-6 && cy.abs() < 1e-6,
                        "z={} x={} y={}: center ({}, {}) should be (0, 0)",
                        z, x, y, cx, cy
                    );
                }
            }
        }
    }

    #[test]
    fn test_layer_stack() {
        let mut stack = LayerStack::new();
        stack.add_layer(TileRenderLayer::new("base"));
        stack.add_layer(TileRenderLayer::new("overlay"));

        assert_eq!(stack.layer_count(), 2);
        assert!(stack.get_layer("base").is_some());

        stack.remove_layer("base");
        assert_eq!(stack.layer_count(), 1);
    }
}
