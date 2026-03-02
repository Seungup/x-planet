//! Step 04: Render a Real OSM Tile
//!
//! Karpathy: "Now connect to real data. One tile. No fancy loading."
//!
//! Goal: Download one OSM raster tile (z=0, the world tile) and render it
//!       as a textured quad. ViewportUniforms for pan/zoom.
//!
//! What this proves:
//!   ✓ HTTP tile fetch works
//!   ✓ Image decode pipeline (PNG → RGBA) works
//!   ✓ Texture upload from real tile data works
//!   ✓ Full pipeline: fetch → decode → upload → render
//!
//! What this deliberately does NOT do:
//!   ✗ No multi-tile loading (just one tile)
//!   ✗ No tile cache
//!   ✗ No tile queue
//!   ✗ No projection transform (raw Mercator)
//!
//! Run: cargo run -p x-planets-examples --example step04_osm_tile

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
    view: vec4<f32>,       // center_x, center_y, zoom, _pad
    resolution: vec4<f32>, // width, height, 1/w, 1/h
};

@group(0) @binding(0) var<uniform> viewport: ViewportUniforms;
@group(1) @binding(0) var tile_texture: texture_2d<f32>;
@group(1) @binding(1) var tile_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let zoom = viewport.view.z;
    let center = viewport.view.xy;
    let aspect = viewport.resolution.x / viewport.resolution.y;

    let world = in.position - center;
    let scaled = world * zoom;

    out.clip_pos = vec4<f32>(
        scaled.x / aspect * 2.0,
        -(scaled.y) * 2.0,
        0.0,
        1.0
    );
    out.uv = in.tex_coord;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(tile_texture, tile_sampler, in.uv);
}
"#;

/// Fallback: CPU-generated world map placeholder (green land, blue ocean).
fn generate_placeholder_tile() -> (u32, u32, Vec<u8>) {
    let size = 256u32;
    let mut pixels = vec![0u8; (size * size * 4) as usize];

    for y in 0..size {
        for x in 0..size {
            let idx = ((y * size + x) * 4) as usize;
            // Simple pattern: blue ocean with "land" patches
            let nx = x as f32 / size as f32;
            let ny = y as f32 / size as f32;
            let land = ((nx * 7.0).sin() * (ny * 5.0).cos()).abs() > 0.4;

            if land {
                pixels[idx] = 100;     // R
                pixels[idx + 1] = 160; // G
                pixels[idx + 2] = 60;  // B
            } else {
                pixels[idx] = 40;      // R
                pixels[idx + 1] = 80;  // G
                pixels[idx + 2] = 180; // B
            }
            pixels[idx + 3] = 255;     // A
        }
    }
    (size, size, pixels)
}

/// Try to fetch the z=0 world tile from OSM.
async fn fetch_osm_tile() -> Option<(u32, u32, Vec<u8>)> {
    let url = "https://tile.openstreetmap.org/0/0/0.png";
    println!("  Fetching tile from: {}", url);

    let client = reqwest::Client::builder()
        .user_agent("x-planets/0.1 (karpathy-step04)")
        .build()
        .ok()?;

    let response = client.get(url).send().await.ok()?;
    if !response.status().is_success() {
        println!("  HTTP {}: using placeholder", response.status());
        return None;
    }

    let bytes = response.bytes().await.ok()?;
    println!("  Downloaded {} bytes", bytes.len());

    let img = image::load_from_memory(&bytes).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    println!("  Decoded: {}x{} RGBA", w, h);

    Some((w, h, rgba.into_raw()))
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    center_x: f32,
    center_y: f32,
    zoom: f32,
    tile_pixels: Option<(u32, u32, Vec<u8>)>,
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
    viewport_bind_group: wgpu::BindGroup,
    tile_bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    tex_coord: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewportUniformsGpu {
    view: [f32; 4],
    resolution: [f32; 4],
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("Step 04: Real OSM Tile [arrows=pan, +/-=zoom]")
            .with_inner_size(winit::dpi::PhysicalSize::new(800, 600));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        // Get tile data — try network, fallback to placeholder
        let (tw, th, pixels) = self.tile_pixels.take().unwrap_or_else(generate_placeholder_tile);

        let gpu = pollster::block_on(init_gpu(window, tw, th, &pixels));
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
                println!("✓ Window closed. Step 04 complete.");
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
                    let uniforms = ViewportUniformsGpu {
                        view: [self.center_x, self.center_y, self.zoom, 0.0],
                        resolution: [
                            gpu.config.width as f32,
                            gpu.config.height as f32,
                            1.0 / gpu.config.width as f32,
                            1.0 / gpu.config.height as f32,
                        ],
                    };
                    gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
                    render_frame(gpu);
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            _ => {}
        }
    }
}

async fn init_gpu(window: Arc<Window>, tile_w: u32, tile_h: u32, tile_pixels: &[u8]) -> GpuState {
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

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("step04-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    println!("✓ Shader compiled");

    // ── Vertex/Index buffers ────────────────────────────────────
    let vertices = [
        Vertex { position: [0.0, 0.0], tex_coord: [0.0, 0.0] },
        Vertex { position: [1.0, 0.0], tex_coord: [1.0, 0.0] },
        Vertex { position: [0.0, 1.0], tex_coord: [0.0, 1.0] },
        Vertex { position: [1.0, 1.0], tex_coord: [1.0, 1.0] },
    ];
    let indices: [u32; 6] = [0, 1, 2, 2, 1, 3];

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step04-vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step04-indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    // ── Viewport uniforms (group 0) ────────────────────────────
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step04-uniforms"),
        contents: bytemuck::bytes_of(&ViewportUniformsGpu {
            view: [0.5, 0.5, 1.0, 0.0],
            resolution: [800.0, 600.0, 1.0 / 800.0, 1.0 / 600.0],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let viewport_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("step04-viewport-bgl"),
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

    let viewport_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("step04-viewport-bg"),
        layout: &viewport_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    // ── Tile texture (group 1) ─────────────────────────────────
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("step04-tile-texture"),
        size: wgpu::Extent3d {
            width: tile_w,
            height: tile_h,
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
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        tile_pixels,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * tile_w),
            rows_per_image: Some(tile_h),
        },
        wgpu::Extent3d {
            width: tile_w,
            height: tile_h,
            depth_or_array_layers: 1,
        },
    );
    println!("✓ Tile texture uploaded ({}x{})", tile_w, tile_h);

    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("step04-tile-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
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

    let tile_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("step04-tile-bg"),
        layout: &tile_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    // ── Pipeline ────────────────────────────────────────────────
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("step04-layout"),
        bind_group_layouts: &[&viewport_bgl, &tile_bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("step04-pipeline"),
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

    println!("✓ Full tile rendering pipeline ready");
    println!();
    println!("You should see: the OSM world tile (or placeholder) with pan/zoom.");

    GpuState {
        surface,
        device,
        queue,
        config,
        pipeline,
        vertex_buffer,
        index_buffer,
        uniform_buffer,
        viewport_bind_group,
        tile_bind_group,
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
            label: Some("step04-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.02,
                        b: 0.05,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &gpu.viewport_bind_group, &[]);
        pass.set_bind_group(1, &gpu.tile_bind_group, &[]);
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
    println!("  Step 04: Real OSM Tile");
    println!("═══════════════════════════════════════");

    // Fetch tile before opening window (blocking)
    let tile_pixels = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async {
            match fetch_osm_tile().await {
                Some(tile) => {
                    println!("✓ OSM tile fetched successfully");
                    Some(tile)
                }
                None => {
                    println!("⚠ Could not fetch OSM tile, using placeholder");
                    None
                }
            }
        });

    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        gpu: None,
        center_x: 0.5,
        center_y: 0.5,
        zoom: 1.0,
        tile_pixels,
    };
    event_loop.run_app(&mut app).unwrap();
}
