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

use crate::shared_render_resources::SharedRenderResources;
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
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
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
    has_texture_flag: f32,
}

impl GpuModel3d {
    /// Update the model matrix and opacity for this model.
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
    /// 1×1 white placeholder texture for models without textures.
    white_texture_view: wgpu::TextureView,
}

impl Model3dRenderer {
    /// Create a new 3D model renderer.
    ///
    /// Requires a GpuContext with a surface (panics if headless).
    pub fn new(gpu: &GpuContext, shared: &SharedRenderResources) -> Self {
        let format = gpu
            .surface_format()
            .expect("Model3dRenderer requires a surface");

        let depth_format = SharedRenderResources::depth_format();

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
                    bind_group_layouts: &[&shared.viewport_bgl, &shared.tile_bgl],
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
                        format: depth_format,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::LessEqual,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });

        let white_texture_view = create_white_texture(&gpu.device, &gpu.queue);

        log::info!("Model3dRenderer created (format: {:?})", format);

        Self {
            pipeline,
            white_texture_view,
        }
    }

    /// Upload a mesh to the GPU and create a renderable model.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_mesh(
        &self,
        gpu: &GpuContext,
        shared: &SharedRenderResources,
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
            layout: &shared.tile_bgl,
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
                    resource: wgpu::BindingResource::Sampler(&shared.sampler),
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
    pub fn render_models(
        &self,
        gpu: &GpuContext,
        shared: &SharedRenderResources,
        target: &wgpu::TextureView,
        viewport: &Viewport,
        models: &[&GpuModel3d],
    ) {
        if models.is_empty() {
            return;
        }
        let uniforms = viewport.to_uniforms();
        self.render_models_with_uniforms(gpu, shared, target, &uniforms, models);
    }

    /// Render a set of 3D models with custom viewport uniforms.
    pub fn render_models_with_uniforms(
        &self,
        gpu: &GpuContext,
        shared: &SharedRenderResources,
        target: &wgpu::TextureView,
        uniforms: &ViewportUniforms,
        models: &[&GpuModel3d],
    ) {
        if models.is_empty() {
            return;
        }

        shared.update_viewport(gpu, uniforms);

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
                    view: &shared.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &shared.viewport_bg, &[]);

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
