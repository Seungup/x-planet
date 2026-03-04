//! Render and compute pipeline builders with fluent API.

use wgpu;

// ---------------------------------------------------------------------------
// Render Pipeline Builder
// ---------------------------------------------------------------------------

/// Builder for wgpu render pipelines.
pub struct RenderPipelineBuilder<'a> {
    device: &'a wgpu::Device,
    label: Option<&'a str>,
    vertex_shader: Option<wgpu::ShaderModule>,
    fragment_shader: Option<wgpu::ShaderModule>,
    vertex_buffers: Vec<wgpu::VertexBufferLayout<'a>>,
    bind_group_layouts: Vec<&'a wgpu::BindGroupLayout>,
    color_target_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    primitive_topology: wgpu::PrimitiveTopology,
    cull_mode: Option<wgpu::Face>,
    blend_state: Option<wgpu::BlendState>,
}

impl<'a> RenderPipelineBuilder<'a> {
    pub fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            label: None,
            vertex_shader: None,
            fragment_shader: None,
            vertex_buffers: Vec::new(),
            bind_group_layouts: Vec::new(),
            color_target_format: wgpu::TextureFormat::Bgra8UnormSrgb,
            depth_format: None,
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            blend_state: Some(wgpu::BlendState::ALPHA_BLENDING),
        }
    }

    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    pub fn vertex_shader(mut self, source: &str) -> Self {
        self.vertex_shader = Some(
            self.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("vertex-shader"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                }),
        );
        self
    }

    pub fn fragment_shader(mut self, source: &str) -> Self {
        self.fragment_shader = Some(
            self.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("fragment-shader"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                }),
        );
        self
    }

    pub fn shader(mut self, source: &str) -> Self {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: self.label,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        self.vertex_shader = Some(
            self.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: self.label,
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                }),
        );
        self.fragment_shader = Some(module);
        self
    }

    pub fn add_vertex_buffer(mut self, layout: wgpu::VertexBufferLayout<'a>) -> Self {
        self.vertex_buffers.push(layout);
        self
    }

    pub fn add_bind_group_layout(mut self, layout: &'a wgpu::BindGroupLayout) -> Self {
        self.bind_group_layouts.push(layout);
        self
    }

    pub fn color_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.color_target_format = format;
        self
    }

    pub fn depth_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.depth_format = Some(format);
        self
    }

    pub fn topology(mut self, topology: wgpu::PrimitiveTopology) -> Self {
        self.primitive_topology = topology;
        self
    }

    pub fn cull(mut self, face: wgpu::Face) -> Self {
        self.cull_mode = Some(face);
        self
    }

    pub fn blend(mut self, state: wgpu::BlendState) -> Self {
        self.blend_state = Some(state);
        self
    }

    pub fn no_blend(mut self) -> Self {
        self.blend_state = None;
        self
    }

    pub fn build(self) -> wgpu::RenderPipeline {
        let vs = self.vertex_shader.expect("vertex shader required");
        let fs = self.fragment_shader.expect("fragment shader required");

        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: self.label,
                    bind_group_layouts: &self.bind_group_layouts,
                    push_constant_ranges: &[],
                });

        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: self.label,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vs,
                    entry_point: Some("vs_main"),
                    buffers: &self.vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fs,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.color_target_format,
                        blend: self.blend_state,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: self.primitive_topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: self.cull_mode,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: self.depth_format.map(|format| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
    }
}

// ---------------------------------------------------------------------------
// Compute Pipeline Builder
// ---------------------------------------------------------------------------

/// Builder for wgpu compute pipelines.
pub struct ComputePipelineBuilder<'a> {
    device: &'a wgpu::Device,
    label: Option<&'a str>,
    shader_source: Option<String>,
    entry_point: String,
    bind_group_layouts: Vec<&'a wgpu::BindGroupLayout>,
}

impl<'a> ComputePipelineBuilder<'a> {
    pub fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            label: None,
            shader_source: None,
            entry_point: "main".to_string(),
            bind_group_layouts: Vec::new(),
        }
    }

    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    pub fn shader(mut self, source: &str) -> Self {
        self.shader_source = Some(source.to_string());
        self
    }

    pub fn entry_point(mut self, entry: &str) -> Self {
        self.entry_point = entry.to_string();
        self
    }

    pub fn add_bind_group_layout(mut self, layout: &'a wgpu::BindGroupLayout) -> Self {
        self.bind_group_layouts.push(layout);
        self
    }

    pub fn build(self) -> wgpu::ComputePipeline {
        let source = self.shader_source.expect("shader source required");

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: self.label,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: self.label,
                    bind_group_layouts: &self.bind_group_layouts,
                    push_constant_ranges: &[],
                });

        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: self.label,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(&self.entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
    }
}

// ---------------------------------------------------------------------------
// Shader Validator
// ---------------------------------------------------------------------------

/// Validate a WGSL shader source string using naga.
pub fn validate_wgsl(source: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| format!("Parse error: {e}"))?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    validator
        .validate(&module)
        .map_err(|e| format!("Validation error: {e}"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_wgsl_valid_shader() {
        let shader = r#"
            @vertex
            fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;
        assert!(validate_wgsl(shader).is_ok());
    }

    #[test]
    fn test_validate_wgsl_syntax_error() {
        let shader = "fn broken( { return; }";
        let result = validate_wgsl(shader);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Parse error"));
    }

    #[test]
    fn test_validate_wgsl_empty_shader() {
        // An empty shader is valid (no entry points, but parseable).
        assert!(validate_wgsl("").is_ok());
    }

    #[test]
    fn test_validate_wgsl_semantic_error() {
        // Referencing an undefined variable is a semantic error.
        let shader = r#"
            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return undefined_var;
            }
        "#;
        let result = validate_wgsl(shader);
        assert!(result.is_err());
    }

    #[test]
    fn test_shipped_shaders_all_valid() {
        // Validate all WGSL shaders shipped with the project.
        let shaders = [
            ("mercator", include_str!("../../../shaders/projections/mercator.wgsl")),
            ("equirectangular", include_str!("../../../shaders/projections/equirectangular.wgsl")),
            ("raster_tile", include_str!("../../../shaders/rendering/raster_tile.wgsl")),
            ("raster_tile_globe", include_str!("../../../shaders/rendering/raster_tile_globe.wgsl")),
            ("terrain_tile", include_str!("../../../shaders/rendering/terrain_tile.wgsl")),
            ("model3d", include_str!("../../../shaders/rendering/model3d.wgsl")),
        ];
        for (name, source) in &shaders {
            let result = validate_wgsl(source);
            assert!(
                result.is_ok(),
                "Shader '{}' failed validation: {}",
                name,
                result.unwrap_err()
            );
        }
    }
}
