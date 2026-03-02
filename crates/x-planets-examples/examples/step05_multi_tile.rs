//! Step 05: Multi-Tile Grid Rendering
//!
//! Karpathy: "One tile worked. Now render ALL visible tiles."
//!
//! Goal: Use Viewport's visible_tiles() + build_tile_mesh() + view_proj matrix
//!       to render a dynamic grid of tiles. Each tile gets a unique color/pattern.
//!
//! What this proves:
//!   ✓ Viewport frustum culling works (only visible tiles rendered)
//!   ✓ build_tile_mesh() produces gap-free geometry
//!   ✓ view_proj orthographic matrix transforms Mercator coords to clip space
//!   ✓ Per-tile uniforms (TileUniforms) work for coloring/identification
//!   ✓ Dynamic tile set changes as camera pans/zooms
//!
//! What this deliberately does NOT do:
//!   ✗ No real tile textures (CPU-generated patterns)
//!   ✗ No async loading or caching
//!   ✗ No layer compositing
//!
//! Run: cargo run -p x-planets-examples --example step05_multi_tile

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};

use x_planets_core::pipeline::{build_tile_mesh, tile_uniforms, viewport_uniforms, visible_tiles};
use x_planets_core::viewport::{CameraController, Viewport};
use x_planets_math::{GeoCoord, TileCoord, TileUniforms};

const SHADER: &str = r#"
struct ViewportUniforms {
    view_proj: mat4x4<f32>,
    resolution: vec4<f32>,
    camera: vec4<f32>,
};

struct TileUniforms {
    mvp: mat4x4<f32>,
    bounds: vec4<f32>,  // min_x, min_y, max_x, max_y
    tile_meta: vec4<f32>,    // zoom, opacity, _pad, _pad
    uv_rect: vec4<f32>,      // u_min, v_min, u_max, v_max
};

@group(0) @binding(0) var<uniform> viewport: ViewportUniforms;
@group(1) @binding(0) var<uniform> tile: TileUniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coord: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_pos: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = tile.mvp * vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.tex_coord;
    out.world_pos = in.position;
    return out;
}

// Hash function for tile coloring
fn hash_color(bounds: vec4<f32>) -> vec3<f32> {
    let seed = bounds.x * 127.1 + bounds.y * 311.7 + bounds.z * 74.7 + bounds.w * 183.3;
    let r = fract(sin(seed) * 43758.5453);
    let g = fract(sin(seed * 1.1 + 1.0) * 43758.5453);
    let b = fract(sin(seed * 1.2 + 2.0) * 43758.5453);
    // Ensure visible colors (not too dark)
    return vec3<f32>(r * 0.5 + 0.3, g * 0.5 + 0.3, b * 0.5 + 0.3);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_color = hash_color(tile.bounds);

    // Checker pattern within each tile
    let checker_x = floor(in.uv.x * 8.0);
    let checker_y = floor(in.uv.y * 8.0);
    let checker = (checker_x + checker_y) % 2.0;
    let pattern = mix(base_color * 0.85, base_color, checker);

    // Border highlight — 2px equivalent
    let border = 0.02;
    let is_border = select(0.0, 1.0,
        in.uv.x < border || in.uv.x > (1.0 - border) ||
        in.uv.y < border || in.uv.y > (1.0 - border));
    let final_color = mix(pattern, vec3<f32>(1.0, 1.0, 1.0), is_border * 0.6);

    return vec4<f32>(final_color, tile.tile_meta.y); // opacity from meta
}
"#;

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    viewport: Viewport,
    camera: CameraController,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    viewport_uniform_buffer: wgpu::Buffer,
    viewport_bind_group: wgpu::BindGroup,
    tile_uniform_buffer: wgpu::Buffer,
    tile_bind_group: wgpu::BindGroup,
    // Dynamic geometry (rebuilt when visible tiles change)
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    visible_tiles: Vec<TileCoord>,
}

use wgpu::util::DeviceExt;

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("Step 05: Multi-Tile Grid [arrows=pan, +/-=zoom]")
            .with_inner_size(winit::dpi::PhysicalSize::new(800, 600));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        let gpu = pollster::block_on(init_gpu(window, &self.viewport));
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
                println!("✓ Window closed. Step 05 complete.");
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                self.viewport.width = new_size.width.max(1);
                self.viewport.height = new_size.height.max(1);
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = new_size.width.max(1);
                    gpu.config.height = new_size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == winit::event::ElementState::Pressed {
                    let pan_amount = 50.0; // pixels
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::ArrowLeft) => {
                            self.camera.pan(&mut self.viewport, pan_amount, 0.0);
                        }
                        PhysicalKey::Code(KeyCode::ArrowRight) => {
                            self.camera.pan(&mut self.viewport, -pan_amount, 0.0);
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            self.camera.pan(&mut self.viewport, 0.0, pan_amount);
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => {
                            self.camera.pan(&mut self.viewport, 0.0, -pan_amount);
                        }
                        PhysicalKey::Code(KeyCode::Equal) | PhysicalKey::Code(KeyCode::NumpadAdd) => {
                            self.camera.zoom(&mut self.viewport, 0.5);
                        }
                        PhysicalKey::Code(KeyCode::Minus) | PhysicalKey::Code(KeyCode::NumpadSubtract) => {
                            self.camera.zoom(&mut self.viewport, -0.5);
                        }
                        PhysicalKey::Code(KeyCode::Home) => {
                            self.viewport.center = GeoCoord::new(0.0, 0.0);
                            self.viewport.zoom = 2.0;
                        }
                        _ => {}
                    }

                    // Rebuild tile geometry for new viewport
                    if let Some(gpu) = &mut self.gpu {
                        rebuild_tiles(gpu, &self.viewport);
                    }
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(gpu) = &self.gpu {
                    // Upload viewport uniforms
                    let vu = viewport_uniforms(&self.viewport);
                    let vp_f64 = self.viewport.to_view_proj_f64();
                    gpu.queue.write_buffer(
                        &gpu.viewport_uniform_buffer,
                        0,
                        bytemuck::bytes_of(&vu),
                    );
                    render_frame(gpu, &vp_f64);
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn rebuild_tiles(gpu: &mut GpuState, viewport: &Viewport) {
    let tiles = visible_tiles(viewport);
    if tiles.is_empty() {
        gpu.visible_tiles = tiles;
        return;
    }

    let (vertices, indices) = build_tile_mesh(&tiles);

    // Recreate buffers if needed (size may differ)
    gpu.vertex_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step05-vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    gpu.index_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step05-indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    println!(
        "  tiles={} verts={} zoom={:.1} center=({:.2},{:.2})",
        tiles.len(),
        vertices.len(),
        viewport.zoom,
        viewport.center.lat,
        viewport.center.lon,
    );

    gpu.visible_tiles = tiles;
}

async fn init_gpu(window: Arc<Window>, viewport: &Viewport) -> GpuState {
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
        label: Some("step05-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    println!("✓ Shader compiled");

    // ── Viewport uniforms (group 0) ────────────────────────────
    let vu = viewport_uniforms(viewport);
    let viewport_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step05-viewport-uniforms"),
        contents: bytemuck::bytes_of(&vu),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let viewport_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("step05-viewport-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let viewport_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("step05-viewport-bg"),
        layout: &viewport_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: viewport_uniform_buffer.as_entire_binding(),
        }],
    });

    // ── Tile uniforms (group 1) — updated per draw call ────────
    let tu = TileUniforms {
        mvp: [0.0; 16],
        bounds: [0.0, 0.0, 1.0, 1.0],
        meta: [0.0, 1.0, 0.0, 0.0],
        uv_rect: [0.0, 0.0, 1.0, 1.0],
    };
    let tile_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step05-tile-uniforms"),
        contents: bytemuck::bytes_of(&tu),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("step05-tile-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let tile_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("step05-tile-bg"),
        layout: &tile_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: tile_uniform_buffer.as_entire_binding(),
        }],
    });

    // ── Pipeline ────────────────────────────────────────────────
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("step05-layout"),
        bind_group_layouts: &[&viewport_bgl, &tile_bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("step05-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[x_planets_core::render::TileVertex::layout()],
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

    // ── Initial tile geometry ───────────────────────────────────
    let tiles = visible_tiles(viewport);
    let (vertices, indices) = build_tile_mesh(&tiles);

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step05-vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step05-indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    println!("✓ Multi-tile pipeline ready");
    println!("  Initial tiles: {} at zoom {:.1}", tiles.len(), viewport.zoom);
    println!();
    println!("You should see: a grid of uniquely colored tiles with borders.");
    println!("Pan/zoom changes which tiles are visible.");

    GpuState {
        surface,
        device,
        queue,
        config,
        pipeline,
        viewport_uniform_buffer,
        viewport_bind_group,
        tile_uniform_buffer,
        tile_bind_group,
        vertex_buffer,
        index_buffer,
        visible_tiles: tiles,
    }
}

fn render_frame(gpu: &GpuState, vp_f64: &glam::DMat4) {
    let frame = match gpu.surface.get_current_texture() {
        Ok(f) => f,
        Err(_) => return,
    };

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("step05-pass"),
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
        pass.set_bind_group(0, &gpu.viewport_bind_group, &[]);
        pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
        pass.set_index_buffer(gpu.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        // Draw each tile with its own TileUniforms
        for (i, coord) in gpu.visible_tiles.iter().enumerate() {
            let tu = tile_uniforms(coord, 1.0, vp_f64);
            gpu.queue.write_buffer(
                &gpu.tile_uniform_buffer,
                0,
                bytemuck::bytes_of(&tu),
            );

            pass.set_bind_group(1, &gpu.tile_bind_group, &[]);

            let idx_start = (i * 6) as u32;
            let idx_end = idx_start + 6;
            let vtx_offset = 0i32; // vertices are absolute
            pass.draw_indexed(idx_start..idx_end, vtx_offset, 0..1);
        }
    }

    gpu.queue.submit(std::iter::once(encoder.finish()));
    frame.present();
}

fn main() {
    env_logger::init();
    println!("═══════════════════════════════════════");
    println!("  Step 05: Multi-Tile Grid");
    println!("═══════════════════════════════════════");

    let mut viewport = Viewport::new(800, 600);
    viewport.center = GeoCoord::new(0.0, 0.0);
    viewport.zoom = 2.0;

    let camera = CameraController::new();

    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        gpu: None,
        viewport,
        camera,
    };
    event_loop.run_app(&mut app).unwrap();
}
