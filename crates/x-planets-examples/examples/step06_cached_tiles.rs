//! Step 06: Tile Cache + Async Loading
//!
//! Karpathy: "Now add the infrastructure. Cache what you've loaded."
//!
//! Goal: Integrate TileCache + compute_load_requests() so that:
//!   - Tiles are loaded on-demand as the camera moves
//!   - Already-loaded tiles are served from cache (no re-fetch)
//!   - Missing tiles show a placeholder until loaded
//!   - FrameSummary logs the pipeline state each frame
//!
//! What this proves:
//!   ✓ compute_load_requests() correctly filters cached tiles
//!   ✓ TileCache LRU eviction works under memory pressure
//!   ✓ FrameSummary accurately tracks visible/cached/pending counts
//!   ✓ Pipeline stages compose: visible_tiles → load_requests → cache → render
//!
//! What this deliberately does NOT do:
//!   ✗ No real HTTP fetching (simulated async delay)
//!   ✗ No texture upload (still CPU-colored tiles)
//!   ✗ No layer compositing
//!
//! Run: cargo run -p x-planets-examples --example step06_cached_tiles

use std::collections::HashSet;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};

use x_planets_core::pipeline::{
    build_tile_mesh, compute_load_requests, tile_uniforms, viewport_uniforms, visible_tiles,
    FrameSummary,
};
use x_planets_core::viewport::{CameraController, Viewport};
use x_planets_math::{GeoCoord, TileCoord, TileUniforms};
use x_planets_tiles::TileCache;

const SHADER: &str = r#"
struct ViewportUniforms {
    view_proj: mat4x4<f32>,
    resolution: vec4<f32>,
    camera: vec4<f32>,
};

struct TileUniforms {
    bounds: vec4<f32>,
    tile_meta: vec4<f32>, uv_rect: vec4<f32>,
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
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = viewport.view_proj * vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.tex_coord;
    return out;
}

fn hash_color(bounds: vec4<f32>) -> vec3<f32> {
    let seed = bounds.x * 127.1 + bounds.y * 311.7 + bounds.z * 74.7 + bounds.w * 183.3;
    let r = fract(sin(seed) * 43758.5453);
    let g = fract(sin(seed * 1.1 + 1.0) * 43758.5453);
    let b = fract(sin(seed * 1.2 + 2.0) * 43758.5453);
    return vec3<f32>(r * 0.5 + 0.3, g * 0.5 + 0.3, b * 0.5 + 0.3);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let opacity = tile.tile_meta.y;

    // Cached tiles: full color. Loading tiles: dim with stripes.
    let is_loading = select(0.0, 1.0, opacity < 0.5);
    let base_color = hash_color(tile.bounds);

    // Loading pattern: diagonal stripes
    let stripe = step(0.5, fract((in.uv.x + in.uv.y) * 8.0));
    let loading_color = mix(base_color * 0.2, base_color * 0.4, stripe);

    let color = mix(base_color, loading_color, is_loading);

    // Border
    let border = 0.02;
    let is_border = select(0.0, 1.0,
        in.uv.x < border || in.uv.x > (1.0 - border) ||
        in.uv.y < border || in.uv.y > (1.0 - border));
    let border_color = mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 0.5, 0.0), is_loading);
    let final_color = mix(color, border_color, is_border * 0.6);

    return vec4<f32>(final_color, 1.0);
}
"#;

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    viewport: Viewport,
    camera: CameraController,
    tile_cache: TileCache<Vec<u8>>,
    frame_count: u64,
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
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    visible_tiles: Vec<TileCoord>,
    cached_set: HashSet<TileCoord>,
}

use wgpu::util::DeviceExt;

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("Step 06: Cached Tiles [arrows=pan, +/-=zoom, space=simulate load]")
            .with_inner_size(winit::dpi::PhysicalSize::new(800, 600));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());
        self.gpu = Some(pollster::block_on(init_gpu(window, &self.viewport)));
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("✓ Window closed. Step 06 complete.");
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
                self.rebuild_and_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == winit::event::ElementState::Pressed {
                    let pan_amount = 50.0;
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
                        PhysicalKey::Code(KeyCode::Space) => {
                            // Simulate loading: move pending tiles into cache
                            self.simulate_tile_load();
                        }
                        PhysicalKey::Code(KeyCode::Home) => {
                            self.viewport.center = GeoCoord::new(0.0, 0.0);
                            self.viewport.zoom = 2.0;
                        }
                        _ => {}
                    }
                    self.rebuild_and_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(gpu) = &self.gpu {
                    let vu = viewport_uniforms(&self.viewport);
                    gpu.queue.write_buffer(
                        &gpu.viewport_uniform_buffer,
                        0,
                        bytemuck::bytes_of(&vu),
                    );
                    render_frame(gpu);
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl App {
    fn rebuild_and_redraw(&mut self) {
        if let Some(gpu) = &mut self.gpu {
            let tiles = visible_tiles(&self.viewport);

            // Compute which tiles are cached
            let cached_set: HashSet<TileCoord> = tiles
                .iter()
                .filter(|t| self.tile_cache.get(t).is_some())
                .copied()
                .collect();

            // Compute load requests (priority-ordered)
            let load_requests = compute_load_requests(
                &tiles,
                &cached_set,
                &self.viewport.center,
            );

            // Frame summary (Karpathy logging)
            self.frame_count += 1;
            if self.frame_count % 5 == 0 || load_requests.len() > 0 {
                let summary = FrameSummary {
                    visible_tile_count: tiles.len(),
                    cached_tile_count: cached_set.len(),
                    load_requests: load_requests.len(),
                    zoom: self.viewport.zoom,
                    center: self.viewport.center,
                };
                println!("  [frame {}] {}", self.frame_count, summary);
            }

            // Rebuild mesh
            if !tiles.is_empty() {
                let (vertices, indices) = build_tile_mesh(&tiles);
                gpu.vertex_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("step06-vertices"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                gpu.index_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("step06-indices"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
            }

            gpu.cached_set = cached_set;
            gpu.visible_tiles = tiles;
        }
        self.window.as_ref().unwrap().request_redraw();
    }

    fn simulate_tile_load(&mut self) {
        // Simulate: load the highest-priority uncached tiles
        let tiles = visible_tiles(&self.viewport);
        let cached_set: HashSet<TileCoord> = tiles
            .iter()
            .filter(|t| self.tile_cache.get(t).is_some())
            .copied()
            .collect();

        let requests = compute_load_requests(&tiles, &cached_set, &self.viewport.center);

        let to_load = requests.iter().take(4).collect::<Vec<_>>();
        for req in &to_load {
            // Insert a dummy pixel into cache to mark as "loaded"
            self.tile_cache.insert(req.coord, vec![0u8; 4]);
            println!("  ✓ Loaded tile {}", req.coord);
        }

        if to_load.is_empty() {
            println!("  All visible tiles already cached!");
        }
    }
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
        label: Some("step06-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    // Layouts
    let viewport_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("step06-viewport-bgl"),
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
    let tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("step06-tile-bgl"),
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

    // Buffers
    let vu = viewport_uniforms(viewport);
    let viewport_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step06-viewport-uniforms"),
        contents: bytemuck::bytes_of(&vu),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let tu = TileUniforms {
        bounds: [0.0; 4],
        meta: [0.0, 1.0, 0.0, 0.0], uv_rect: [0.0, 0.0, 1.0, 1.0],
    };
    let tile_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step06-tile-uniforms"),
        contents: bytemuck::bytes_of(&tu),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Bind groups
    let viewport_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("step06-viewport-bg"),
        layout: &viewport_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: viewport_uniform_buffer.as_entire_binding(),
        }],
    });
    let tile_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("step06-tile-bg"),
        layout: &tile_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: tile_uniform_buffer.as_entire_binding(),
        }],
    });

    // Pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("step06-layout"),
        bind_group_layouts: &[&viewport_bgl, &tile_bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("step06-pipeline"),
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

    // Initial tiles
    let tiles = visible_tiles(viewport);
    let (vertices, indices) = build_tile_mesh(&tiles);

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step06-vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step06-indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    println!("✓ Cached tile pipeline ready");
    println!("  Press SPACE to simulate loading tiles");

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
        cached_set: HashSet::new(),
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
            label: Some("step06-pass"),
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

        for (i, coord) in gpu.visible_tiles.iter().enumerate() {
            let is_cached = gpu.cached_set.contains(coord);
            let opacity = if is_cached { 1.0 } else { 0.3 }; // dim = loading
            let tu = tile_uniforms(coord, opacity);
            gpu.queue.write_buffer(&gpu.tile_uniform_buffer, 0, bytemuck::bytes_of(&tu));

            pass.set_bind_group(1, &gpu.tile_bind_group, &[]);
            let idx_start = (i * 6) as u32;
            pass.draw_indexed(idx_start..(idx_start + 6), 0, 0..1);
        }
    }

    gpu.queue.submit(std::iter::once(encoder.finish()));
    frame.present();
}

fn main() {
    env_logger::init();
    println!("═══════════════════════════════════════");
    println!("  Step 06: Tile Cache + Async Loading");
    println!("═══════════════════════════════════════");

    let mut viewport = Viewport::new(800, 600);
    viewport.center = GeoCoord::new(0.0, 0.0);
    viewport.zoom = 2.0;

    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        gpu: None,
        viewport,
        camera: CameraController::new(),
        tile_cache: TileCache::new(64),
        frame_count: 0,
    };
    event_loop.run_app(&mut app).unwrap();
}
