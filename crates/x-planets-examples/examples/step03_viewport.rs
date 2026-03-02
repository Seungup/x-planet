//! Step 03: ViewportUniforms — Pan & Zoom
//!
//! Karpathy: "Add ONE thing at a time. This step adds uniforms."
//!
//! Goal: Render a colored quad that moves and scales via ViewportUniforms.
//!       Arrow keys pan, +/- zoom. The quad should move with the camera.
//!
//! What this proves:
//!   ✓ Uniform buffer creation + upload works
//!   ✓ Bind group for uniforms works
//!   ✓ Vertex shader reads ViewportUniforms correctly
//!   ✓ Pan/zoom transforms are correct (CPU → GPU match)
//!
//! What this deliberately does NOT do:
//!   ✗ No textures (still using vertex colors)
//!   ✗ No tile loading
//!   ✗ No projection
//!
//! Run: cargo run -p x-planets-examples --example step03_viewport

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};

const SHADER: &str = r#"
struct ViewportUniforms {
    // [center_x, center_y, zoom, _pad]
    view: vec4<f32>,
    // [width, height, 1/width, 1/height]
    resolution: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> viewport: ViewportUniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Apply viewport transform:
    // 1. Shift by center
    // 2. Scale by zoom
    // 3. Map to clip space [-1, 1]
    let zoom = viewport.view.z;
    let center = viewport.view.xy;
    let aspect = viewport.resolution.x / viewport.resolution.y;

    let world = in.position - center;
    let scaled = world * zoom;

    // Correct for aspect ratio
    out.clip_pos = vec4<f32>(
        scaled.x / aspect * 2.0,
        -(scaled.y) * 2.0,  // flip Y for screen coords
        0.0,
        1.0
    );

    out.color = in.tex_coord;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Color from tex_coords: R=u, G=v, B=0.3
    return vec4<f32>(in.color.x, in.color.y, 0.3, 1.0);
}
"#;

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    // Viewport state — will match x_planets_core::Viewport logic
    center_x: f32,
    center_y: f32,
    zoom: f32,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    tex_coord: [f32; 2],
}

/// ViewportUniforms matching the shader struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewportUniformsGpu {
    view: [f32; 4],       // center_x, center_y, zoom, _pad
    resolution: [f32; 4], // width, height, 1/width, 1/height
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("Step 03: ViewportUniforms Pan/Zoom [arrows=pan, +/-=zoom]")
            .with_inner_size(winit::dpi::PhysicalSize::new(800, 600));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        let gpu = pollster::block_on(init_gpu(window));
        self.gpu = Some(gpu);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("✓ Window closed. Step 03 complete.");
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = new_size.width.max(1);
                    gpu.config.height = new_size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == winit::event::ElementState::Pressed {
                    let pan_speed = 0.05 / self.zoom;
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::ArrowLeft) => self.center_x -= pan_speed,
                        PhysicalKey::Code(KeyCode::ArrowRight) => self.center_x += pan_speed,
                        PhysicalKey::Code(KeyCode::ArrowUp) => self.center_y -= pan_speed,
                        PhysicalKey::Code(KeyCode::ArrowDown) => self.center_y += pan_speed,
                        PhysicalKey::Code(KeyCode::Equal) | PhysicalKey::Code(KeyCode::NumpadAdd) => {
                            self.zoom *= 1.2;
                        }
                        PhysicalKey::Code(KeyCode::Minus) | PhysicalKey::Code(KeyCode::NumpadSubtract) => {
                            self.zoom /= 1.2;
                        }
                        PhysicalKey::Code(KeyCode::Home) => {
                            // Reset
                            self.center_x = 0.5;
                            self.center_y = 0.5;
                            self.zoom = 1.0;
                        }
                        _ => {}
                    }
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(gpu) = &self.gpu {
                    // Update uniform buffer with current viewport state
                    let uniforms = ViewportUniformsGpu {
                        view: [self.center_x, self.center_y, self.zoom, 0.0],
                        resolution: [
                            gpu.config.width as f32,
                            gpu.config.height as f32,
                            1.0 / gpu.config.width as f32,
                            1.0 / gpu.config.height as f32,
                        ],
                    };
                    gpu.queue
                        .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

                    render_frame(gpu);
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            _ => {}
        }
    }
}

async fn init_gpu(window: Arc<Window>) -> GpuState {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone()).unwrap();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .expect("No suitable GPU adapter");

    println!("✓ GPU: {}", adapter.get_info().name);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap();

    let size = window.inner_size();
    let config = surface
        .get_default_config(&adapter, size.width.max(1), size.height.max(1))
        .unwrap();
    surface.configure(&device, &config);

    // Shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("step03-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    println!("✓ Shader compiled");

    // Quad vertices: covers [0,0]→[1,1] in world space
    let vertices = [
        Vertex { position: [0.0, 0.0], tex_coord: [0.0, 0.0] },
        Vertex { position: [1.0, 0.0], tex_coord: [1.0, 0.0] },
        Vertex { position: [0.0, 1.0], tex_coord: [0.0, 1.0] },
        Vertex { position: [1.0, 1.0], tex_coord: [1.0, 1.0] },
    ];
    let indices: [u32; 6] = [0, 1, 2, 2, 1, 3];

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step03-vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step03-indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    // Uniform buffer
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step03-uniforms"),
        contents: bytemuck::bytes_of(&ViewportUniformsGpu {
            view: [0.5, 0.5, 1.0, 0.0],
            resolution: [800.0, 600.0, 1.0 / 800.0, 1.0 / 600.0],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("step03-bgl"),
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("step03-bg"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("step03-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("step03-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
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
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

    println!("✓ Pipeline with ViewportUniforms created");
    println!();
    println!("Controls:");
    println!("  Arrow keys = pan");
    println!("  +/- = zoom in/out");
    println!("  Home = reset");
    println!();
    println!("You should see: a gradient quad that moves when you press arrows.");

    GpuState {
        surface,
        device,
        queue,
        config,
        pipeline,
        vertex_buffer,
        index_buffer,
        uniform_buffer,
        bind_group,
    }
}

fn render_frame(gpu: &GpuState) {
    let frame = match gpu.surface.get_current_texture() {
        Ok(f) => f,
        Err(_) => return,
    };

    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("step03-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.05,
                        g: 0.05,
                        b: 0.1,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &gpu.bind_group, &[]);
        pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
        pass.set_index_buffer(gpu.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..6, 0, 0..1);
    }

    gpu.queue.submit(std::iter::once(encoder.finish()));
    frame.present();
}

use wgpu::util::DeviceExt;

fn main() {
    env_logger::init();
    println!("═══════════════════════════════════════");
    println!("  Step 03: ViewportUniforms Pan/Zoom");
    println!("═══════════════════════════════════════");

    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        gpu: None,
        center_x: 0.5,
        center_y: 0.5,
        zoom: 1.0,
    };
    event_loop.run_app(&mut app).unwrap();
}
