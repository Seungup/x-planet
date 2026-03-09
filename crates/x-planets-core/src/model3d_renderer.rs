//! GPU 3D model renderer using model3d.wgsl.
//!
//! Renders glTF/B3DM meshes extracted from 3D Tiles with basic
//! directional lighting and texture support.
//!
//! Each model is uploaded as its own vertex/index buffer with a
//! per-model uniform buffer (model matrix + opacity).

use bytemuck::{Pod, Zeroable};
use x_planets_gpu::GpuContext;
use x_planets_math::ViewportUniforms;

use crate::viewport::Viewport;

const MODEL3D_SHADER: &str = include_str!("../../../shaders/rendering/model3d.wgsl");

// ═══════════════════════════════════════════════════════════════════
// Vertex type
// ═══════════════════════════════════════════════════════════════════

/// Vertex layout for 3D model rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Model3dVertex {
    /// Position (x, y, z) in local/ECEF coordinates.
    pub position: [f32; 3],
    /// Normal vector.
    pub normal: [f32; 3],
    /// Texture coordinate (UV).
    pub tex_coord: [f32; 2],
}

impl Model3dVertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress, // 32 bytes
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position: Float32x3 at location 0
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // normal: Float32x3 at location 1
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // tex_coord: Float32x2 at location 2
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() * 2) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Model uniforms
// ═══════════════════════════════════════════════════════════════════

/// Per-model uniform data uploaded to the GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ModelUniforms {
    /// Model matrix (ECEF → world space).
    pub model_matrix: [f32; 16],
    /// Parameters: (opacity, has_texture, _pad, _pad).
    pub params: [f32; 4],
}

// ═══════════════════════════════════════════════════════════════════
// GPU model (uploaded mesh)
// ═══════════════════════════════════════════════════════════════════

/// A 3D model uploaded to GPU, ready for rendering.
pub struct GpuModel3d {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Cached texture flag (1.0 = has texture, 0.0 = no texture).
    has_texture_flag: f32,
}

impl GpuModel3d {
    /// Update the model matrix and opacity for this model.
    ///
    /// Call this each frame when the camera moves to recompute
    /// ECEF-relative transforms for planet-scale rendering.
    pub fn update_transform(&self, queue: &wgpu::Queue, model_matrix: [f32; 16], opacity: f32) {
        let uniforms = ModelUniforms {
            model_matrix,
            params: [opacity, self.has_texture_flag, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}

// ═══════════════════════════════════════════════════════════════════
// 1×1 white texture (placeholder when no texture)
// ═══════════════════════════════════════════════════════════════════

fn create_white_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("model3d-white-1x1"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[255u8, 255, 255, 255],
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

// ═══════════════════════════════════════════════════════════════════
// Model3dRenderer
// ═══════════════════════════════════════════════════════════════════

/// Renderer for 3D tile models (glTF/B3DM meshes).
pub struct Model3dRenderer {
    pipeline: wgpu::RenderPipeline,
    _viewport_bgl: wgpu::BindGroupLayout,
    model_bgl: wgpu::BindGroupLayout,
    viewport_buffer: wgpu::Buffer,
    viewport_bg: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    depth_view: wgpu::TextureView,
    depth_format: wgpu::TextureFormat,
    surface_width: u32,
    surface_height: u32,
    /// 1×1 white placeholder texture for models without textures.
    white_texture_view: wgpu::TextureView,
}

impl Model3dRenderer {
    /// Create a new 3D model renderer.
    ///
    /// Requires a GpuContext with a surface (panics if headless).
    pub fn new(gpu: &GpuContext) -> Self {
        let format = gpu
            .surface_format()
            .expect("Model3dRenderer requires a surface");

        // ── Bind group layout 0: viewport uniforms ──
        let _viewport_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("model3d-viewport-bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        // ── Bind group layout 1: model uniforms + texture + sampler ──
        let model_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("model3d-model-bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX
                                | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float {
                                    filterable: true,
                                },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(
                                wgpu::SamplerBindingType::Filtering,
                            ),
                            count: None,
                        },
                    ],
                });

        // ── Shader + Pipeline ──
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("model3d-shader"),
                source: wgpu::ShaderSource::Wgsl(MODEL3D_SHADER.into()),
            });

        let pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("model3d-layout"),
                    bind_group_layouts: &[&_viewport_bgl, &model_bgl],
                    push_constant_ranges: &[],
                });

        let pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("model3d-pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[Model3dVertex::layout()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::LessEqual,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });

        // ── Viewport uniform buffer ──
        let viewport_uniforms = ViewportUniforms {
            view_proj: [0.0; 16],
            resolution: [0.0; 4],
            camera: [0.0; 4],
            clip_sphere: [0.0; 4],
            terrain: [0.0; 4],
            sun_dir: [0.0; 4],
        };
        let viewport_buffer =
            gpu.create_uniform_buffer("model3d-viewport-uniforms", &viewport_uniforms);

        let viewport_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model3d-viewport-bg"),
            layout: &_viewport_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: viewport_buffer.as_entire_binding(),
            }],
        });

        // ── Sampler ──
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("model3d-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ── Depth texture ──
        let (surface_width, surface_height) = gpu
            .surface
            .as_ref()
            .map(|s| (s.config.width, s.config.height))
            .unwrap_or((800, 600));
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let depth_view =
            Self::create_depth_texture(&gpu.device, surface_width, surface_height, depth_format);

        // ── White placeholder texture ──
        let white_texture_view = create_white_texture(&gpu.device, &gpu.queue);

        log::info!(
            "Model3dRenderer created (format: {:?}, depth: {:?})",
            format,
            depth_format
        );

        Self {
            pipeline,
            _viewport_bgl,
            model_bgl,
            viewport_buffer,
            viewport_bg,
            sampler,
            depth_view,
            depth_format,
            surface_width,
            surface_height,
            white_texture_view,
        }
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("model3d-depth-texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Recreate depth texture after window resize.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width != self.surface_width || height != self.surface_height {
            self.surface_width = width;
            self.surface_height = height;
            self.depth_view =
                Self::create_depth_texture(device, width, height, self.depth_format);
        }
    }

    /// Upload a mesh to the GPU and create a renderable model.
    ///
    /// `model_matrix` transforms from local/ECEF coordinates to world space.
    /// `opacity` is 0.0–1.0 for blending.
    /// `texture_view` is optional; if None, a white placeholder is used.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_mesh(
        &self,
        gpu: &GpuContext,
        label: &str,
        vertices: &[Model3dVertex],
        indices: &[u32],
        model_matrix: [f32; 16],
        opacity: f32,
        texture_view: Option<&wgpu::TextureView>,
    ) -> GpuModel3d {
        let vertex_buffer =
            gpu.create_vertex_buffer(&format!("model3d-verts-{}", label), vertices);
        let index_buffer =
            gpu.create_index_buffer(&format!("model3d-idx-{}", label), indices);

        let has_texture = if texture_view.is_some() { 1.0 } else { 0.0 };
        let uniforms = ModelUniforms {
            model_matrix,
            params: [opacity, has_texture, 0.0, 0.0],
        };
        let uniform_buffer = gpu.create_uniform_buffer("model3d-uniforms", &uniforms);

        let tex_view = texture_view.unwrap_or(&self.white_texture_view);

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("model3d-bg-{}", label)),
            layout: &self.model_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        GpuModel3d {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            uniform_buffer,
            bind_group,
            has_texture_flag: has_texture,
        }
    }

    /// Create a GPU texture from RGBA pixel data.
    pub fn create_texture(
        gpu: &GpuContext,
        label: &str,
        width: u32,
        height: u32,
        rgba_data: &[u8],
    ) -> wgpu::TextureView {
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        tex.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Render a set of 3D models to the target surface.
    ///
    /// Uses `LoadOp::Load` for color (preserves raster + terrain layers already drawn)
    /// and `LoadOp::Clear(1.0)` for depth.
    pub fn render_models(
        &self,
        gpu: &GpuContext,
        target: &wgpu::TextureView,
        viewport: &Viewport,
        models: &[&GpuModel3d],
    ) {
        if models.is_empty() {
            return;
        }

        // Update viewport uniforms from the standard Mercator viewport.
        let uniforms = viewport.to_uniforms();
        self.render_models_with_uniforms(gpu, target, &uniforms, models);
    }

    /// Render a set of 3D models with custom viewport uniforms.
    ///
    /// Use this for ECEF-based rendering where the view-projection matrix
    /// is computed in ECEF-relative space (from `tiles3d_pipeline`).
    pub fn render_models_with_uniforms(
        &self,
        gpu: &GpuContext,
        target: &wgpu::TextureView,
        uniforms: &ViewportUniforms,
        models: &[&GpuModel3d],
    ) {
        if models.is_empty() {
            return;
        }

        gpu.update_buffer(&self.viewport_buffer, uniforms);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("model3d-render-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.viewport_bg, &[]);

            for model in models {
                pass.set_bind_group(1, &model.bind_group, &[]);
                pass.set_vertex_buffer(0, model.vertex_buffer.slice(..));
                pass.set_index_buffer(model.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..model.index_count, 0, 0..1);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}
