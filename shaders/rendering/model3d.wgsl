// 3D Model Rendering Shader
//
// Renders 3D tile meshes (glTF/B3DM) with basic directional lighting.
// Vertex shader: transforms ECEF/local positions through model + view-projection.
// Fragment shader: texture sampling with directional light + ambient.

// --- Uniforms ---

struct ViewportUniforms {
    view_proj: mat4x4<f32>,
    resolution: vec4<f32>,   // (width, height, 1/width, 1/height)
    camera: vec4<f32>,       // (center_x, center_y, zoom, pitch)
    clip_sphere: vec4<f32>,  // (center_x, center_y, center_z, cos_clip_angle)
    terrain: vec4<f32>,      // (max_zoom, hillshade_strength, _pad, _pad)
    sun_dir: vec4<f32>,      // (sun_x, sun_y, sun_z, _pad)
};

struct ModelUniforms {
    model_matrix: mat4x4<f32>,   // ECEF → world space transform
    params: vec4<f32>,           // (opacity, has_texture, _pad, _pad)
};

@group(0) @binding(0)
var<uniform> viewport: ViewportUniforms;

@group(1) @binding(0)
var<uniform> model: ModelUniforms;

@group(1) @binding(1)
var model_texture: texture_2d<f32>;

@group(1) @binding(2)
var model_sampler: sampler;

// --- Constants ---

// Directional light (sun-like, from above-right)
const LIGHT_DIR: vec3<f32> = vec3<f32>(0.3, 0.8, 0.5);
const AMBIENT: f32 = 0.35;
const DIFFUSE_STRENGTH: f32 = 0.65;

// Default color when no texture is provided
const DEFAULT_COLOR: vec3<f32> = vec3<f32>(0.75, 0.75, 0.75);

// --- Vertex Shader ---

struct VertexInput {
    @location(0) position: vec3<f32>,    // Local or ECEF position
    @location(1) normal: vec3<f32>,      // Vertex normal
    @location(2) tex_coord: vec2<f32>,   // UV coordinates
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Transform position: model_matrix moves from local/ECEF to world space
    let world_pos = model.model_matrix * vec4<f32>(input.position, 1.0);

    // Apply view-projection
    output.clip_position = viewport.view_proj * world_pos;

    // Transform normal (using upper-3x3 of model matrix)
    let normal_matrix = mat3x3<f32>(
        model.model_matrix[0].xyz,
        model.model_matrix[1].xyz,
        model.model_matrix[2].xyz,
    );
    output.world_normal = normalize(normal_matrix * input.normal);

    output.tex_coord = input.tex_coord;

    return output;
}

// --- Fragment Shader ---

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Base color: texture or default
    var base_color: vec3<f32>;
    let has_texture = model.params.y;
    if (has_texture > 0.5) {
        base_color = textureSample(model_texture, model_sampler, input.tex_coord).rgb;
    } else {
        base_color = DEFAULT_COLOR;
    }

    // Simple directional lighting
    let normal = normalize(input.world_normal);
    let light_dir = normalize(LIGHT_DIR);
    let ndotl = max(dot(normal, light_dir), 0.0);
    let lighting = AMBIENT + DIFFUSE_STRENGTH * ndotl;

    let lit_color = base_color * lighting;

    // Apply opacity
    let opacity = model.params.x;
    return vec4<f32>(lit_color, opacity);
}
