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

/// Quad vertices for a single tile (Relative-To-Center) in Mercator space.
///
/// Positions are relative to the tile center, NOT absolute Mercator coordinates.
/// This avoids f32 precision loss at high zoom levels.  The per-tile MVP matrix
/// (computed in f64 on the CPU) handles the translation to world/clip space.
///
/// At zoom z, each tile spans `1 / 2^z` in Mercator space, so the half-size
/// is `0.5 / 2^z`.  Vertices are at `(±hw, ±hh)` centered on the origin.
pub fn tile_quad_vertices(coord: &TileCoord) -> [TileVertex; 4] {
    tile_quad_vertices_projected(coord, x_planets_math::ProjectionMode::Mercator)
}

/// Quad vertices with projection-dependent half-heights.
///
/// For Mercator, all tiles at the same zoom have equal height.
/// For Equirectangular, height varies by latitude (compressed near poles).
pub fn tile_quad_vertices_projected(
    coord: &TileCoord,
    mode: x_planets_math::ProjectionMode,
) -> [TileVertex; 4] {
    let n = coord.extent() as f32;
    let hw = 0.5 / n; // half-width (same for all projections — linear in longitude)

    let hh = match mode {
        x_planets_math::ProjectionMode::Mercator
        | x_planets_math::ProjectionMode::Globe => 0.5 / n,
        x_planets_math::ProjectionMode::Equirectangular => {
            let n_f64 = coord.extent() as f64;
            let y_top_m = coord.y as f64 / n_f64;
            let y_bot_m = (coord.y + 1) as f64 / n_f64;
            let y_top_eq = x_planets_math::mercator_y_to_equirectangular_y(y_top_m);
            let y_bot_eq = x_planets_math::mercator_y_to_equirectangular_y(y_bot_m);
            ((y_bot_eq - y_top_eq).abs() / 2.0) as f32
        }
    };

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

/// Vertex layout for globe tile rendering (3D position on sphere surface + UV).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobeTileVertex {
    /// Position (x, y, z) relative to tile center on the unit sphere (RTE).
    pub position: [f32; 3],
    /// Texture coordinate (0..1) within the tile.
    pub tex_coord: [f32; 2],
}

impl GlobeTileVertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress, // 20 bytes
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
                    format: wgpu::VertexFormat::Float32x2, // tex_coord
                },
            ],
        }
    }
}

/// Tessellation subdivisions for a globe tile based on zoom level.
///
/// Lower zoom → larger tiles → more curvature → more subdivisions needed.
pub fn globe_subdivisions(zoom: u8) -> u32 {
    match zoom {
        0..=2 => 32,
        3..=5 => 16,
        6..=9 => 8,
        10..=13 => 4,
        _ => 2,
    }
}

/// Build a tessellated globe mesh for a single tile on the unit sphere (RTE).
///
/// Each vertex is computed on the unit sphere surface, then offset relative
/// to `tile_center_3d` (computed in f64 for precision).
pub fn tile_globe_mesh(
    coord: &x_planets_math::TileCoord,
    tile_center_3d: glam::DVec3,
) -> (Vec<GlobeTileVertex>, Vec<u32>) {
    use std::f64::consts::PI;

    let subdiv = globe_subdivisions(coord.z);
    let seg = subdiv + 1; // vertices per axis
    let n = coord.extent() as f64;
    let mut vertices = Vec::with_capacity((seg * seg) as usize);
    let mut indices = Vec::with_capacity((subdiv * subdiv * 6) as usize);

    for j in 0..=subdiv {
        for i in 0..=subdiv {
            let u = i as f64 / subdiv as f64;
            let v = j as f64 / subdiv as f64;

            // Global Mercator position [0,1]
            let mx = (coord.x as f64 + u) / n;
            let my = (coord.y as f64 + v) / n;

            // Mercator → lat/lon (radians)
            let lon_rad = (mx * 2.0 - 1.0) * PI;
            let lat_rad = x_planets_math::mercator_y_to_lat_rad(my);

            // 3D position on unit sphere
            let pos_3d = x_planets_math::geo_to_unit_sphere(lat_rad, lon_rad);
            // RTE: subtract tile center in f64, then cast to f32
            let rte = pos_3d - tile_center_3d;

            vertices.push(GlobeTileVertex {
                position: [rte.x as f32, rte.y as f32, rte.z as f32],
                tex_coord: [u as f32, v as f32],
            });
        }
    }

    // Triangle indices (CCW winding from outside the sphere → back-face culling works)
    for j in 0..subdiv {
        for i in 0..subdiv {
            let tl = j * seg + i;
            let tr = j * seg + i + 1;
            let bl = (j + 1) * seg + i;
            let br = (j + 1) * seg + i + 1;
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    (vertices, indices)
}

/// Subdivisions for centered Mercator tessellation.
///
/// Same logic as globe — lower zoom needs more subdivisions because tiles
/// cover a larger angular extent and the oblique reprojection curves them.
pub fn centered_subdivisions(zoom: u8) -> u32 {
    match zoom {
        0..=2 => 16,
        3..=5 => 8,
        6..=9 => 4,
        _ => 2,
    }
}

/// Build a tessellated mesh for a tile using oblique (viewport-centered) Mercator.
///
/// Each vertex is projected through `oblique_mercator(lat, lon, center)` and
/// placed in 3D with z=0 (uses `GlobeTileVertex` layout for pipeline reuse).
/// `tile_center_2d` is the tile center in centered Mercator [0,1]² space.
pub fn tile_centered_mesh(
    coord: &x_planets_math::TileCoord,
    center_lat_rad: f64,
    center_lon_rad: f64,
    tile_center_2d: glam::DVec2,
) -> (Vec<GlobeTileVertex>, Vec<u32>) {
    use std::f64::consts::PI;

    let subdiv = centered_subdivisions(coord.z);
    let seg = subdiv + 1;
    let n = coord.extent() as f64;
    let mut vertices = Vec::with_capacity((seg * seg) as usize);
    let mut indices = Vec::with_capacity((subdiv * subdiv * 6) as usize);

    for j in 0..=subdiv {
        for i in 0..=subdiv {
            let u = i as f64 / subdiv as f64;
            let v = j as f64 / subdiv as f64;

            let mx = (coord.x as f64 + u) / n;
            let my = (coord.y as f64 + v) / n;

            // Standard Mercator → lat/lon
            let lon_rad = (mx * 2.0 - 1.0) * PI;
            let lat_rad = x_planets_math::mercator_y_to_lat_rad(my);

            // Oblique (centered) Mercator
            let centered = x_planets_math::oblique_mercator(
                lat_rad,
                lon_rad,
                center_lat_rad,
                center_lon_rad,
            );

            // RTE: subtract tile center in 2D
            let rx = (centered.x - tile_center_2d.x) as f32;
            let ry = (centered.y - tile_center_2d.y) as f32;

            vertices.push(GlobeTileVertex {
                position: [rx, ry, 0.0], // flat, z=0
                tex_coord: [u as f32, v as f32],
            });
        }
    }

    // Triangle indices — skip triangles whose winding has been inverted
    // by the oblique Mercator projection (happens near the antipodal point
    // of the projection center where the Mercator singularity flips geometry).
    for j in 0..subdiv {
        for i in 0..subdiv {
            let tl = j * seg + i;
            let tr = j * seg + i + 1;
            let bl = (j + 1) * seg + i;
            let br = (j + 1) * seg + i + 1;

            let p_tl = &vertices[tl as usize].position;
            let p_tr = &vertices[tr as usize].position;
            let p_bl = &vertices[bl as usize].position;
            let p_br = &vertices[br as usize].position;

            // 2D cross product: positive = CCW (normal winding), negative = CW (flipped)
            let cross1 = (p_tr[0] - p_tl[0]) * (p_bl[1] - p_tl[1])
                - (p_tr[1] - p_tl[1]) * (p_bl[0] - p_tl[0]);
            if cross1 > 0.0 {
                indices.extend_from_slice(&[tl, tr, bl]);
            }

            let cross2 = (p_tr[0] - p_bl[0]) * (p_br[1] - p_bl[1])
                - (p_tr[1] - p_bl[1]) * (p_br[0] - p_bl[0]);
            if cross2 > 0.0 {
                indices.extend_from_slice(&[bl, tr, br]);
            }
        }
    }

    (vertices, indices)
}

/// Number of segments for each polar cap ring.
const POLAR_CAP_SEGMENTS: u32 = 64;

/// Build a polar cap mesh (triangle fan from the pole to ±85.05° latitude).
///
/// `north`: true for north pole, false for south pole.
/// Returns (vertices, indices) with positions *not* RTE (absolute unit sphere coords).
/// The caller adds a center at the pole and the cap fills the gap
/// where no Mercator tiles exist.
pub fn polar_cap_mesh(north: bool) -> (Vec<GlobeTileVertex>, Vec<u32>) {
    use std::f64::consts::PI;

    // Mercator tile boundary latitude (rad)
    let cap_lat_deg: f64 = if north { 85.05112878 } else { -85.05112878 };
    let pole_lat_deg: f64 = if north { 90.0 } else { -90.0 };
    let cap_lat = cap_lat_deg.to_radians();
    let pole_lat = pole_lat_deg.to_radians();

    // Pole center vertex
    let pole_center = x_planets_math::geo_to_unit_sphere(pole_lat, 0.0);

    let seg = POLAR_CAP_SEGMENTS;
    let mut vertices = Vec::with_capacity(seg as usize + 1);
    let mut indices = Vec::with_capacity(seg as usize * 3);

    // Vertex 0 = pole
    vertices.push(GlobeTileVertex {
        position: [pole_center.x as f32, pole_center.y as f32, pole_center.z as f32],
        tex_coord: [0.5, 0.5],
    });

    // Ring of vertices at cap latitude
    for i in 0..=seg {
        let t = i as f64 / seg as f64;
        let lon = t * 2.0 * PI - PI;
        let pos = x_planets_math::geo_to_unit_sphere(cap_lat, lon);
        vertices.push(GlobeTileVertex {
            position: [pos.x as f32, pos.y as f32, pos.z as f32],
            tex_coord: [t as f32, if north { 0.0 } else { 1.0 }],
        });
    }

    // Triangle fan indices (CCW from outside for back-face culling)
    for i in 0..seg {
        if north {
            // North: pole at top, ring going counter-clockwise from outside
            indices.extend_from_slice(&[0, i + 2, i + 1]);
        } else {
            // South: opposite winding
            indices.extend_from_slice(&[0, i + 1, i + 2]);
        }
    }

    (vertices, indices)
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
