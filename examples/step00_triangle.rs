//! Step 00: The Stupidest Thing That Works
//!
//! Karpathy: "Start with the absolute simplest end-to-end."
//!
//! Goal: Open a window. Clear it blue. Draw a white triangle.
//! If you see a white triangle on blue, step 00 is done.
//!
//! What this proves:
//!   ✓ wgpu initializes correctly
//!   ✓ Window + Surface works
//!   ✓ Shader compilation works
//!   ✓ Draw call reaches the screen
//!
//! What this deliberately does NOT do:
//!   ✗ No textures
//!   ✗ No uniforms
//!   ✗ No tile coordinates
//!   ✗ No projection
//!
//! Run: cargo run --example step00_triangle

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::EventLoop,
    window::{Window, WindowAttributes},
};
use std::sync::Arc;

const SHADER: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    // Hardcoded triangle vertices. No buffers needed.
    var pos = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.5),   // top
        vec2<f32>(-0.5, -0.5),   // bottom-left
        vec2<f32>( 0.5, -0.5),   // bottom-right
    );
    return vec4<f32>(pos[idx], 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0); // white
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("Step 00: White Triangle on Blue")
            .with_inner_size(winit::dpi::PhysicalSize::new(800, 600));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        // Initialize GPU — blocking for simplicity in step 00
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
                println!("✓ Window closed. Step 00 complete.");
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
        .expect("No suitable GPU adapter found");

    println!("✓ GPU adapter: {}", adapter.get_info().name);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .expect("Failed to create device");

    println!("✓ Device created");

    let size = window.inner_size();
    let config = surface
        .get_default_config(&adapter, size.width.max(1), size.height.max(1))
        .expect("Surface not supported");
    surface.configure(&device, &config);

    println!("✓ Surface configured ({}x{})", config.width, config.height);

    // Compile shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("step00-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    println!("✓ Shader compiled");

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("step00-layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("step00-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[], // No vertex buffers — positions hardcoded in shader
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

    println!("✓ Render pipeline created");
    println!();
    println!("You should see: white triangle on blue background.");
    println!("Close the window to finish step 00.");

    GpuState {
        surface,
        device,
        queue,
        config,
        pipeline,
    }
}

fn render_frame(gpu: &GpuState) {
    let frame = match gpu.surface.get_current_texture() {
        Ok(f) => f,
        Err(_) => return, // Skip frame on surface error
    };

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("step00-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.6,
                        a: 1.0, // blue background
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        pass.set_pipeline(&gpu.pipeline);
        pass.draw(0..3, 0..1); // 3 vertices, 1 instance
    }

    gpu.queue.submit(std::iter::once(encoder.finish()));
    frame.present();
}

fn main() {
    env_logger::init();
    println!("═══════════════════════════════════════");
    println!("  Step 00: The Stupidest Thing");
    println!("═══════════════════════════════════════");

    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        gpu: None,
    };
    event_loop.run_app(&mut app).unwrap();
}
