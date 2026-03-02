// Terrain Tile Rendering Shader
//
// Renders terrain mesh tiles with 3D displaced vertices and hillshade lighting.
// Vertex shader: transforms 3D tile vertices through view-projection.
// Fragment shader: samples the imagery texture draped onto the terrain,
//                  blended with a directional light hillshade.

// --- Uniforms ---

struct ViewportUniforms {
    view_proj: mat4x4<f32>,
    resolution: vec4<f32>,   // (width, height, 1/width, 1/height)
    camera: vec4<f32>,       // (center_x, center_y, zoom, pitch)
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
    @location(0) position: vec3<f32>,    // RTE: xy relative to tile center, z = elevation
    @location(1) normal: vec3<f32>,      // Surface normal
    @location(2) tex_coord: vec2<f32>,   // UV for imagery texture
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // RTE (Relative-To-Center): vertex xy is relative to tile center.
    // Reconstruct absolute Mercator position from tile bounds.
    let tile_center = (tile.bounds.xy + tile.bounds.zw) * 0.5;
    let world_pos = vec4<f32>(input.position.xy + tile_center, input.position.z, 1.0);

    // Apply view-projection matrix
    output.clip_position = viewport.view_proj * world_pos;

    // Depth bias: finer (higher zoom) tiles get smaller depth -> render on top.
    let depth_bias = (22.0 - tile.tile_meta.x) * 0.0001;
    output.clip_position.z = output.clip_position.z - depth_bias * output.clip_position.w;

    output.tex_coord = input.tex_coord;
    output.normal = input.normal;

    return output;
}

// --- Fragment Shader ---

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Remap tex_coord from [0,1] to the UV sub-rect (for fallback textures).
    let uv = mix(tile.uv_rect.xy, tile.uv_rect.zw, input.tex_coord);
    let color = textureSample(tile_texture, tile_sampler, uv);

    // ── Hillshade lighting ──
    // Sun direction: northwest, 45° elevation (classic cartographic hillshade)
    let sun_dir = normalize(vec3<f32>(-0.5, -0.5, 0.7));
    let n = normalize(input.normal);

    // Lambertian diffuse
    let ndotl = max(dot(n, sun_dir), 0.0);

    // Blend: ambient + diffuse.  Keeps 40% base brightness + 60% sun contribution.
    let shade = 0.4 + 0.6 * ndotl;

    // Apply tile opacity
    let opacity = tile.tile_meta.y;
    return vec4<f32>(color.rgb * shade, color.a * opacity);
}
