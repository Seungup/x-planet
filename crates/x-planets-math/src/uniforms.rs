use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// GPU-friendly Uniforms
// ---------------------------------------------------------------------------

/// Viewport uniforms that get uploaded to the GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ViewportUniforms {
    /// View-projection matrix
    pub view_proj: [f32; 16],
    /// Viewport resolution (width, height, 1/width, 1/height)
    pub resolution: [f32; 4],
    /// Camera center in world coordinates (x, y, zoom, _padding)
    pub camera: [f32; 4],
    /// Small-circle clipping: (center_x, center_y, center_z, cos_clip_angle)
    /// on the unit sphere.  The fragment shader discards pixels where
    /// dot(sphere_pos, clip_center.xyz) < clip_center.w.
    pub clip_sphere: [f32; 4],
    /// Terrain rendering params: (max_zoom, hillshade_strength, _pad, _pad).
    /// `max_zoom` drives depth-bias in shaders; `hillshade_strength` controls
    /// how strong the hillshade effect is (0.0 = flat, 1.0 = full).
    pub terrain: [f32; 4],
    /// Normalised sun direction for hillshade lighting (x, y, z, _pad).
    pub sun_dir: [f32; 4],
}

/// Per-tile uniforms uploaded to GPU.
///
/// Includes a per-tile model-view-projection matrix computed in f64 on the
/// CPU.  This eliminates f32 jitter at high zoom levels by baking the
/// tile-center translation into the matrix while still in f64.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TileUniforms {
    /// Per-tile model-view-projection matrix.
    /// `mvp = VP_f64 * translate(tile_center_f64)`, then cast to f32.
    /// The shader multiplies this by the RTE vertex position directly.
    pub mvp: [f32; 16],
    /// Tile world-space bounds (min_x, min_y, max_x, max_y)
    pub bounds: [f32; 4],
    /// Tile metadata (zoom_level, opacity, _pad, _pad)
    pub meta: [f32; 4],
    /// UV sub-rectangle within the texture (u_min, v_min, u_max, v_max).
    /// Default is [0, 0, 1, 1] for full texture; sub-rects are used
    /// when a parent tile's texture is used as fallback.
    pub uv_rect: [f32; 4],
}
