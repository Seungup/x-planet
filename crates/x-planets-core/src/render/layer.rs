//! Layer types for tile rendering.

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

#[cfg(test)]
mod tests {
    use super::*;

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
