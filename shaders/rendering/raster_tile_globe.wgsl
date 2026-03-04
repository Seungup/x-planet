// Globe Raster Tile Rendering Shader
//
// Renders textured tile patches on a 3D unit-sphere surface.
// Vertex shader: transforms RTE 3D sphere positions through per-tile MVP.
// Fragment shader: samples the tile texture with bilinear filtering.
//
// Uses the same uniform layout as raster_tile.wgsl (TileUniforms).
// Only difference: vertex position is vec3 (3D on sphere) instead of vec2.

// --- Uniforms ---

struct ViewportUniforms {
    view_proj: mat4x4<f32>,
    resolution: vec4<f32>,   // (width, height, 1/width, 1/height)
    camera: vec4<f32>,       // (center_x, center_y, zoom, _pad)
};

struct TileUniforms {
    mvp: mat4x4<f32>,        // Per-tile MVP (VP_f64 * translate(tile_center_3d)), cast to f32
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
    @location(0) position: vec3<f32>,    // RTE: relative to tile center on unit sphere
    @location(1) tex_coord: vec2<f32>,   // Texture coordinate (0..1)
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // RTE: vertex position is relative to tile center on the unit sphere.
    // Per-tile MVP includes the tile-center translation (computed in f64 on CPU).
    output.clip_position = tile.mvp * vec4<f32>(input.position, 1.0);

    // Depth bias: finer (higher zoom) tiles render on top of coarser ones.
    // Higher zoom → more bias subtracted → lower z → closer to camera.
    let depth_bias = tile.tile_meta.x * 0.0001;
    output.clip_position.z = output.clip_position.z - depth_bias * output.clip_position.w;

    output.tex_coord = input.tex_coord;

    return output;
}

// --- Fragment Shader ---

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Remap tex_coord from [0,1] to the UV sub-rect.
    let uv = mix(tile.uv_rect.xy, tile.uv_rect.zw, input.tex_coord);
    let color = textureSample(tile_texture, tile_sampler, uv);

    // Apply tile opacity
    let opacity = tile.tile_meta.y;
    return vec4<f32>(color.rgb, color.a * opacity);
}
