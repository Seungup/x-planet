// Terrain Tile Rendering Shader
//
// Renders terrain mesh tiles with 3D displaced vertices and hillshade lighting.
// Vertex shader: transforms RTE 3D tile vertices through per-tile MVP to clip space,
//                and computes unit-sphere position for small-circle clipping.
// Fragment shader: samples the imagery texture draped onto the terrain,
//                  blended with a directional light hillshade.
//                  Includes small-circle clipping to discard fragments beyond the
//                  visible hemisphere in centered Mercator mode.

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
    mvp: mat4x4<f32>,        // Per-tile MVP (VP_f64 * translate(tile_center)), cast to f32
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
    @location(0) position: vec3<f32>,    // RTE: xy relative to tile center, z = elevation
    @location(1) normal: vec3<f32>,      // Surface normal
    @location(2) tex_coord: vec2<f32>,   // UV for imagery texture
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) sphere_pos: vec3<f32>,  // Unit-sphere position for small-circle clipping
};

const PI: f32 = 3.14159265358979323846;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // RTE (Relative-To-Center): vertex xy is relative to tile center.
    // Per-tile MVP already includes the tile-center translation (computed in f64 on CPU).
    // No need to reconstruct absolute position — just multiply directly.
    output.clip_position = tile.mvp * vec4<f32>(input.position, 1.0);

    // Depth bias: finer (higher zoom) tiles get smaller depth -> render on top.
    let depth_bias = (viewport.terrain.x - tile.tile_meta.x) * 0.0001;
    output.clip_position.z = output.clip_position.z - depth_bias * output.clip_position.w;

    output.tex_coord = input.tex_coord;
    output.normal = input.normal;

    // Reconstruct unit-sphere position from standard Mercator bounds + tex_coord.
    // This enables small-circle clipping in the fragment shader without adding
    // extra per-vertex data.
    let mx = mix(tile.bounds.x, tile.bounds.z, input.tex_coord.x);
    let my = mix(tile.bounds.y, tile.bounds.w, input.tex_coord.y);
    let lon = (mx * 2.0 - 1.0) * PI;
    let lat = 2.0 * atan(exp(PI * (1.0 - 2.0 * my))) - PI * 0.5;
    let cos_lat = cos(lat);
    output.sphere_pos = vec3<f32>(cos_lat * cos(lon), cos_lat * sin(lon), sin(lat));

    return output;
}

// --- Fragment Shader ---

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Small-circle clipping: discard fragments beyond the clip angle
    // from the viewport center on the unit sphere.
    // clip_sphere.xyz = center direction, clip_sphere.w = cos(clip_angle).
    let cos_angle = dot(normalize(input.sphere_pos), viewport.clip_sphere.xyz);
    if cos_angle < viewport.clip_sphere.w {
        discard;
    }

    // Remap tex_coord from [0,1] to the UV sub-rect (for fallback textures).
    let uv = mix(tile.uv_rect.xy, tile.uv_rect.zw, input.tex_coord);
    let color = textureSample(tile_texture, tile_sampler, uv);

    // -- Hillshade lighting --
    // Sun direction from uniform (configurable from CPU side)
    let sun = normalize(viewport.sun_dir.xyz);
    let n = normalize(input.normal);

    // Lambertian diffuse
    let ndotl = max(dot(n, sun), 0.0);

    // "Always daytime" hillshade controlled by hillshade_strength uniform.
    // strength=1.0 → classic relief shading; strength=0.0 → flat (no shading).
    let strength = viewport.terrain.y;
    let flat_illumination = sun.z;  // dot(vec3(0,0,1), sun)
    let shade = clamp((1.0 - 0.4 * strength) + 0.4 * strength * ndotl / flat_illumination,
                       1.0 - 0.4 * strength, 1.0 + 0.1 * strength);

    // Apply tile opacity (premultiplied alpha output)
    let opacity = tile.tile_meta.y;
    let shaded = color.rgb * shade;
    return vec4<f32>(shaded * opacity, color.a * opacity);
}
