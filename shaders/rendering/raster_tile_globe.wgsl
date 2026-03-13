// Globe / Centered-Mercator Raster Tile Rendering Shader
//
// Renders textured tile patches on a 3D unit-sphere surface or as
// oblique Mercator flat tiles.  Includes small-circle clipping: the
// fragment shader discards pixels beyond a configurable angular distance
// from the viewport center, producing a clean circular boundary.
//
// Uses the same uniform layout as raster_tile.wgsl (TileUniforms).
// Vertex position is vec3 (3D on sphere or 2D with z=0 for centered).
// sphere_pos carries the original unit-sphere position for clipping.

// --- Uniforms ---

struct ViewportUniforms {
    view_proj: mat4x4<f32>,
    resolution: vec4<f32>,   // (width, height, 1/width, 1/height)
    camera: vec4<f32>,       // (center_x, center_y, zoom, pitch)
    clip_sphere: vec4<f32>,  // (center_x, center_y, center_z, cos_clip_angle)
    terrain: vec4<f32>,      // (max_zoom, hillshade_strength, _pad, _pad)
    sun_dir: vec4<f32>,      // (sun_x, sun_y, sun_z, _pad)
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
    @location(2) sphere_pos: vec3<f32>,  // Original position on unit sphere
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) sphere_pos: vec3<f32>,  // Interpolated for fragment clipping
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
    output.sphere_pos = input.sphere_pos;

    return output;
}

// --- Fragment Shader ---

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Small-circle clipping: discard fragments beyond the clip angle
    // from the viewport center on the unit sphere.
    // clip_sphere.xyz = center direction, clip_sphere.w = cos(clip_angle).
    // When clip_sphere.w <= -1.0, clipping is disabled (debug bypass).
    if viewport.clip_sphere.w > -1.0 {
        let cos_angle = dot(normalize(input.sphere_pos), viewport.clip_sphere.xyz);
        if cos_angle < viewport.clip_sphere.w {
            discard;
        }
    }

    // Remap tex_coord from [0,1] to the UV sub-rect.
    let uv = mix(tile.uv_rect.xy, tile.uv_rect.zw, input.tex_coord);
    let color = textureSample(tile_texture, tile_sampler, uv);

    // Apply tile opacity (premultiplied alpha output)
    let opacity = tile.tile_meta.y;
    return vec4<f32>(color.rgb * opacity, color.a * opacity);
}
