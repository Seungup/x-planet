//! Render pipeline for tile layers.

use x_planets_math::TileCoord;

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

/// Quad vertices for a single tile.
/// Each tile is rendered as two triangles forming a quad.
pub fn tile_quad_vertices(coord: &TileCoord) -> [TileVertex; 4] {
    let n = coord.extent() as f32;
    let x0 = coord.x as f32 / n;
    let y0 = coord.y as f32 / n;
    let x1 = (coord.x + 1) as f32 / n;
    let y1 = (coord.y + 1) as f32 / n;

    [
        TileVertex { position: [x0, y0], tex_coord: [0.0, 0.0] },
        TileVertex { position: [x1, y0], tex_coord: [1.0, 0.0] },
        TileVertex { position: [x0, y1], tex_coord: [0.0, 1.0] },
        TileVertex { position: [x1, y1], tex_coord: [1.0, 1.0] },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_quad_vertices() {
        let coord = TileCoord::new(1, 0, 0);
        let verts = tile_quad_vertices(&coord);
        // At zoom 1, tile (0,0) covers 0..0.5 in both axes
        assert!((verts[0].position[0] - 0.0).abs() < 1e-6);
        assert!((verts[3].position[0] - 0.5).abs() < 1e-6);
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
