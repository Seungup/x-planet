// Raster Tile Rendering Shader
//
// Renders textured tile quads with projection support.
// Vertex shader: transforms tile coordinates through projection to clip space.
// Fragment shader: samples the tile texture with bilinear filtering.

// --- Uniforms ---

struct ViewportUniforms {
    view_proj: mat4x4<f32>,
    resolution: vec4<f32>,   // (width, height, 1/width, 1/height)
    camera: vec4<f32>,       // (center_x, center_y, zoom, _pad)
};

struct TileUniforms {
    bounds: vec4<f32>,       // (min_x, min_y, max_x, max_y) in Mercator space
    tile_meta: vec4<f32>,    // (zoom_level, opacity, _pad, _pad)
    uv_rect: vec4<f32>,      // (u_min, v_min, u_max, v_max) sub-rect in texture
};

@group(0) @binding(0)
var<uniform> viewport: ViewportUniforms;

@group(1) @binding(0)
var<uniform> tile: TileUniforms;

@group(1) @binding(1)
var tile_texture: texture_2d<f32>;

@group(1) @binding(2)
var tile_sampler: sampler;

// --- Vertex Shader ---

struct VertexInput {
    @location(0) position: vec2<f32>,    // Tile quad position (Mercator 0..1)
    @location(1) tex_coord: vec2<f32>,   // Texture coordinate (0..1)
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Position is already in Mercator space (0..1)
    let world_pos = vec4<f32>(input.position, 0.0, 1.0);

    // Apply view-projection matrix
    output.clip_position = viewport.view_proj * world_pos;

    // Depth bias: finer (higher zoom) tiles get smaller depth → render on top.
    // zoom_level ranges 0..22.  Bias shifts NDC z so higher zoom = closer.
    let depth_bias = (22.0 - tile.tile_meta.x) * 0.0001;
    output.clip_position.z = output.clip_position.z - depth_bias * output.clip_position.w;

    output.tex_coord = input.tex_coord;

    return output;
}

// --- Fragment Shader ---

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Remap tex_coord from [0,1] to the UV sub-rect.
    // For own texture: uv_rect = (0, 0, 1, 1) → identity.
    // For fallback parent: maps to the correct quadrant of the parent texture.
    let uv = mix(tile.uv_rect.xy, tile.uv_rect.zw, input.tex_coord);
    let color = textureSample(tile_texture, tile_sampler, uv);

    // Apply tile opacity
    let opacity = tile.tile_meta.y;
    return vec4<f32>(color.rgb, color.a * opacity);
}
