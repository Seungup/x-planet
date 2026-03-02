//! Step 01: TileVertex Colored Quad
//!
//! Previous step proved: GPU works, shaders compile, pixels reach screen.
//! This step adds ONE thing: vertex buffers with our TileVertex layout.
//!
//! Goal: Render a colored quad using TileVertex (position + tex_coord).
//! tex_coord is used as color (R=u, G=v) so we can visually verify
//! that the vertex layout is correct.
//!
//! What this proves:
//!   ✓ TileVertex layout matches shader expectations
//!   ✓ Index buffer draws two triangles as a quad
//!   ✓ Vertex attributes are passed correctly to fragment shader
//!
//! What this deliberately does NOT do:
//!   ✗ No textures (tex_coord used as color for verification)
//!   ✗ No uniforms
//!   ✗ No projection
//!
//! Expected visual: A quad covering the screen with a gradient:
//!   top-left=black, top-right=red, bottom-left=green, bottom-right=yellow
//!
//! Run: cargo run --example step01_colored_quad

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::EventLoop,
    window::{Window, WindowAttributes},
};
use std::sync::Arc;
use wgpu::util::DeviceExt;

// Reuse the exact same TileVertex from our engine — this is the point.
// If this renders correctly, the layout is right.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TileVertex {
    position: [f32; 2],
    tex_coord: [f32; 2],
}

const VERTICES: &[TileVertex] = &[
    TileVertex { position: [-1.0, -1.0], tex_coord: [0.0, 1.0] }, // bottom-left
    TileVertex { position: [ 1.0, -1.0], tex_coord: [1.0, 1.0] }, // bottom-right
    TileVertex { position: [-1.0,  1.0], tex_coord: [0.0, 0.0] }, // top-left
    TileVertex { position: [ 1.0,  1.0], tex_coord: [1.0, 0.0] }, // top-right
];

const INDICES: &[u32] = &[0, 1, 2, 2, 1, 3];

const SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = vec4<f32>(input.position, 0.0, 1.0);
    output.uv = input.tex_coord;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Use tex_coord as color: R=u, G=v, B=0
    // This lets us visually verify the vertex layout is correct.
    return vec4<f32>(input.uv.x, input.uv.y, 0.0, 1.0);
}
"#;

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() { return; }

        let attrs = WindowAttributes::default()
            .with_title("Step 01: TileVertex Colored Quad")
            .with_inner_size(winit::dpi::PhysicalSize::new(800, 600));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());
        self.gpu = Some(pollster::block_on(init_gpu(window)));
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = s.width.max(1);
                    gpu.config.height = s.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(gpu) = &self.gpu {
                    render_frame(gpu);
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            _ => {}
        }
    }
}

async fn init_gpu(window: Arc<Window>) -> GpuState {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap();

    let size = window.inner_size();
    let config = surface
        .get_default_config(&adapter, size.width.max(1), size.height.max(1))
        .unwrap();
    surface.configure(&device, &config);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("step01-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("step01-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<TileVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    wgpu::VertexAttribute {
                        offset: 8,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step01-vertices"),
        contents: bytemuck::cast_slice(VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step01-indices"),
        contents: bytemuck::cast_slice(INDICES),
        usage: wgpu::BufferUsages::INDEX,
    });

    println!("✓ Vertex buffer: {} bytes ({} vertices)",
        std::mem::size_of_val(VERTICES), VERTICES.len());
    println!("✓ Index buffer: {} indices", INDICES.len());
    println!();
    println!("Expected: gradient quad (black→red→green→yellow)");
    println!("  top-left=black  top-right=red");
    println!("  bot-left=green  bot-right=yellow");

    GpuState {
        surface, device, queue, config, pipeline,
        vertex_buffer, index_buffer,
    }
}

fn render_frame(gpu: &GpuState) {
    let frame = match gpu.surface.get_current_texture() {
        Ok(f) => f,
        Err(_) => return,
    };
    let view = frame.texture.create_view(&Default::default());
    let mut enc = gpu.device.create_command_encoder(&Default::default());

    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });

        pass.set_pipeline(&gpu.pipeline);
        pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
        pass.set_index_buffer(gpu.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..6, 0, 0..1);
    }

    gpu.queue.submit(std::iter::once(enc.finish()));
    frame.present();
}

fn main() {
    env_logger::init();
    println!("═══════════════════════════════════════");
    println!("  Step 01: TileVertex Colored Quad");
    println!("═══════════════════════════════════════");

    let event_loop = EventLoop::new().unwrap();
    let mut app = App { window: None, gpu: None };
    event_loop.run_app(&mut app).unwrap();
}
