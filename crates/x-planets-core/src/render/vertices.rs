//! Vertex structs for tile, globe, and terrain rendering.
//!
//! The struct definitions are always available (pure `#[repr(C)]` + bytemuck POD).
//! The `layout()` methods that return `wgpu::VertexBufferLayout` are only available
//! when the `gpu` feature is enabled.

/// Vertex layout for tile rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileVertex {
    pub position: [f32; 2],
    pub tex_coord: [f32; 2],
}

#[cfg(feature = "gpu")]
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

/// Vertex layout for globe tile rendering (3D position on sphere surface + UV + sphere pos).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobeTileVertex {
    /// Position (x, y, z) relative to tile center on the unit sphere (RTE).
    pub position: [f32; 3],
    /// Texture coordinate (0..1) within the tile.
    pub tex_coord: [f32; 2],
    /// Original position on the unit sphere (for small-circle clipping in fragment shader).
    pub sphere_pos: [f32; 3],
}

#[cfg(feature = "gpu")]
impl GlobeTileVertex {
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
                    format: wgpu::VertexFormat::Float32x2, // tex_coord
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f32; 2]>())
                        as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3, // sphere_pos
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

#[cfg(feature = "gpu")]
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
