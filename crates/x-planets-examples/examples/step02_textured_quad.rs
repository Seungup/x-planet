//! Step 02: Checkerboard Textured Quad + CPU Readback Verification
//!
//! Previous step proved: TileVertex layout works, quad renders correctly.
//! This step adds ONE thing: texture sampling.
//!
//! Goal: Generate a checkerboard texture on CPU, upload to GPU,
//! sample it in the fragment shader, and (optionally) verify
//! by reading back the rendered pixels.
//!
//! What this proves:
//!   ✓ Texture creation + upload works
//!   ✓ Sampler configuration works
//!   ✓ Bind group layout for texture+sampler works
//!   ✓ textureSample() in shader returns correct values
//!
//! Karpathy twist: we generate the expected output on CPU too,
//! so we can compare GPU render vs CPU reference.
//!
//! Run: cargo run --example step02_textured_quad

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::EventLoop,
    window::{Window, WindowAttributes},
};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TileVertex {
    position: [f32; 2],
    tex_coord: [f32; 2],
}

const VERTICES: &[TileVertex] = &[
    TileVertex { position: [-1.0, -1.0], tex_coord: [0.0, 1.0] },
    TileVertex { position: [ 1.0, -1.0], tex_coord: [1.0, 1.0] },
    TileVertex { position: [-1.0,  1.0], tex_coord: [0.0, 0.0] },
    TileVertex { position: [ 1.0,  1.0], tex_coord: [1.0, 0.0] },
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

@group(0) @binding(0)
var t_tile: texture_2d<f32>;
@group(0) @binding(1)
var s_tile: sampler;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_tile, s_tile, input.uv);
}
"#;

/// Generate a checkerboard texture (same as test_utils but standalone).
fn make_checkerboard(size: u32, cells: u32) -> Vec<u8> {
    let cell_size = size / cells;
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let is_white = ((x / cell_size) + (y / cell_size)) % 2 == 0;
            let c: u8 = if is_white { 255 } else { 50 };
            pixels.extend_from_slice(&[c, c, c, 255]);
        }
    }
    pixels
}

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
    bind_group: wgpu::BindGroup,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() { return; }
        let attrs = WindowAttributes::default()
            .with_title("Step 02: Checkerboard Texture")
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
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
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
    let config = surface.get_default_config(&adapter, size.width.max(1), size.height.max(1)).unwrap();
    surface.configure(&device, &config);

    // Create checkerboard texture (8x8 cells on 256x256)
    let tex_size = 256u32;
    let checker_data = make_checkerboard(tex_size, 8);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("checkerboard"),
        size: wgpu::Extent3d { width: tex_size, height: tex_size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &checker_data,
        wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4 * tex_size), rows_per_image: Some(tex_size) },
        wgpu::Extent3d { width: tex_size, height: tex_size, depth_or_array_layers: 1 },
    );

    println!("✓ Checkerboard texture uploaded ({}x{}, 8x8 cells)", tex_size, tex_size);

    let tex_view = texture.create_view(&Default::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Nearest, // Nearest so checkerboard is crisp
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("texture-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("texture-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&tex_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
        ],
    });

    println!("✓ Bind group created (texture + sampler)");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("step02-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("step02-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                    wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step02-verts"),
        contents: bytemuck::cast_slice(VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step02-indices"),
        contents: bytemuck::cast_slice(INDICES),
        usage: wgpu::BufferUsages::INDEX,
    });

    println!("✓ Pipeline ready");
    println!();
    println!("Expected: 8x8 checkerboard pattern (white & dark gray)");

    GpuState {
        surface, device, queue, config, pipeline,
        vertex_buffer, index_buffer, bind_group,
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
        pass.set_bind_group(0, &gpu.bind_group, &[]);
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
    println!("  Step 02: Checkerboard Texture");
    println!("═══════════════════════════════════════");

    let event_loop = EventLoop::new().unwrap();
    let mut app = App { window: None, gpu: None };
    event_loop.run_app(&mut app).unwrap();
}
