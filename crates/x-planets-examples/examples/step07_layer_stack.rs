//! Step 07: LayerStack Compositing
//!
//! Karpathy: "Now compose multiple data sources. Layers are just arrays."
//!
//! Goal: Use the LayerStack system to render two independent tile layers
//!       with separate visibility and opacity controls:
//!   - "base" layer: checker-patterned tiles (simulating terrain)
//!   - "overlay" layer: striped tiles with transparency (simulating labels)
//!
//! What this proves:
//!   ✓ LayerStack ordering (bottom-to-top render order)
//!   ✓ Per-layer opacity blending works
//!   ✓ Layer visibility toggle works
//!   ✓ Multiple layers use the same vertex/projection pipeline
//!
//! What this deliberately does NOT do:
//!   ✗ No real tile data (CPU-generated patterns)
//!   ✗ No per-tile textures
//!   ✗ No projection switching
//!
//! Run: cargo run -p x-planets-examples --example step07_layer_stack

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};

use x_planets_core::pipeline::{build_tile_mesh, tile_uniforms, viewport_uniforms, visible_tiles};
use x_planets_core::render::{LayerStack, TileRenderLayer};
use x_planets_core::viewport::{CameraController, Viewport};
use x_planets_math::{GeoCoord, TileCoord, TileUniforms};

const SHADER: &str = r#"
struct ViewportUniforms {
    view_proj: mat4x4<f32>,
    resolution: vec4<f32>,
    camera: vec4<f32>,
};

struct TileUniforms {
    bounds: vec4<f32>,
    tile_meta: vec4<f32>,  // zoom, opacity, layer_id, _pad
    uv_rect: vec4<f32>,
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

fn hash_val(v: f32) -> f32 {
    return fract(sin(v * 127.1) * 43758.5453);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let opacity = tile.tile_meta.y;
    let layer_id = tile.tile_meta.z;
    let seed = tile.bounds.x * 127.1 + tile.bounds.y * 311.7;

    // --- Layer 0: Base (terrain-like checker) ---
    if layer_id < 0.5 {
        let r = hash_val(seed);
        let g = hash_val(seed + 1.0);
        let b = hash_val(seed + 2.0);
        let base = vec3<f32>(r * 0.3 + 0.2, g * 0.4 + 0.3, b * 0.2 + 0.15);

        let checker = (floor(in.uv.x * 4.0) + floor(in.uv.y * 4.0)) % 2.0;
        let color = mix(base, base * 1.3, checker);

        // Border
        let b_w = 0.015;
        let is_border = select(0.0, 1.0,
            in.uv.x < b_w || in.uv.x > (1.0 - b_w) ||
            in.uv.y < b_w || in.uv.y > (1.0 - b_w));

        return vec4<f32>(mix(color, vec3<f32>(0.4, 0.3, 0.2), is_border * 0.5), opacity);
    }

    // --- Layer 1: Overlay (label-like stripes) ---
    let stripe_freq = 12.0;
    let stripe = smoothstep(0.4, 0.5, fract(in.uv.y * stripe_freq));

    // Semi-transparent overlay with colored stripes
    let overlay_r = hash_val(seed + 10.0);
    let overlay_color = vec3<f32>(0.9, 0.8 * overlay_r + 0.2, 0.3);
    let alpha = stripe * opacity * 0.7;

    // Cross pattern for label simulation
    let cross_x = smoothstep(0.45, 0.5, abs(in.uv.x - 0.5));
    let cross_y = smoothstep(0.45, 0.5, abs(in.uv.y - 0.5));
    let cross = max(1.0 - cross_x, 1.0 - cross_y) * 0.3;

    return vec4<f32>(overlay_color, alpha + cross * opacity);
}
"#;

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    viewport: Viewport,
    camera: CameraController,
    layer_stack: LayerStack,
    base_opacity: f32,
    overlay_opacity: f32,
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
}

use wgpu::util::DeviceExt;

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("Step 07: Layer Stack [1=base, 2=overlay, Q/W=opacity]")
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
                println!("✓ Window closed. Step 07 complete.");
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
                        // Toggle layers
                        PhysicalKey::Code(KeyCode::Digit1) => {
                            if let Some(layer) = self.layer_stack.get_layer_mut("base") {
                                layer.visible = !layer.visible;
                                println!("  Base layer: {}", if layer.visible { "ON" } else { "OFF" });
                            }
                        }
                        PhysicalKey::Code(KeyCode::Digit2) => {
                            if let Some(layer) = self.layer_stack.get_layer_mut("overlay") {
                                layer.visible = !layer.visible;
                                println!("  Overlay layer: {}", if layer.visible { "ON" } else { "OFF" });
                            }
                        }
                        // Adjust overlay opacity
                        PhysicalKey::Code(KeyCode::KeyQ) => {
                            self.overlay_opacity = (self.overlay_opacity - 0.1).max(0.0);
                            println!("  Overlay opacity: {:.1}", self.overlay_opacity);
                        }
                        PhysicalKey::Code(KeyCode::KeyW) => {
                            self.overlay_opacity = (self.overlay_opacity + 0.1).min(1.0);
                            println!("  Overlay opacity: {:.1}", self.overlay_opacity);
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
                    render_frame(gpu, &self.layer_stack, self.base_opacity, self.overlay_opacity);
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
            if !tiles.is_empty() {
                let (vertices, indices) = build_tile_mesh(&tiles);
                gpu.vertex_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("step07-vertices"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                gpu.index_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("step07-indices"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
            }
            gpu.visible_tiles = tiles;
        }
        self.window.as_ref().unwrap().request_redraw();
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
        label: Some("step07-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let viewport_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("step07-viewport-bgl"),
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
        label: Some("step07-tile-bgl"),
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

    let vu = viewport_uniforms(viewport);
    let viewport_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step07-viewport-uniforms"),
        contents: bytemuck::bytes_of(&vu),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let tu = TileUniforms { bounds: [0.0; 4], meta: [0.0, 1.0, 0.0, 0.0], uv_rect: [0.0, 0.0, 1.0, 1.0] };
    let tile_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step07-tile-uniforms"),
        contents: bytemuck::bytes_of(&tu),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let viewport_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("step07-viewport-bg"),
        layout: &viewport_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: viewport_uniform_buffer.as_entire_binding(),
        }],
    });
    let tile_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("step07-tile-bg"),
        layout: &tile_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: tile_uniform_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("step07-layout"),
        bind_group_layouts: &[&viewport_bgl, &tile_bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("step07-pipeline"),
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

    let tiles = visible_tiles(viewport);
    let (vertices, indices) = build_tile_mesh(&tiles);

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step07-vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("step07-indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    println!("✓ Layer stack pipeline ready");
    println!("  [1] Toggle base layer  [2] Toggle overlay layer");
    println!("  [Q/W] Decrease/Increase overlay opacity");

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

fn render_frame(
    gpu: &GpuState,
    layer_stack: &LayerStack,
    base_opacity: f32,
    overlay_opacity: f32,
) {
    let frame = match gpu.surface.get_current_texture() {
        Ok(f) => f,
        Err(_) => return,
    };

    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("step07-pass"),
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

        // Render layers bottom-to-top
        for (layer_idx, layer) in layer_stack.visible_layers().enumerate() {
            let layer_opacity = if layer_idx == 0 { base_opacity } else { overlay_opacity };

            for (i, coord) in gpu.visible_tiles.iter().enumerate() {
                let mut tu = tile_uniforms(coord, layer_opacity * layer.opacity);
                tu.meta[2] = layer_idx as f32; // layer_id for shader
                gpu.queue.write_buffer(&gpu.tile_uniform_buffer, 0, bytemuck::bytes_of(&tu));

                pass.set_bind_group(1, &gpu.tile_bind_group, &[]);
                let idx_start = (i * 6) as u32;
                pass.draw_indexed(idx_start..(idx_start + 6), 0, 0..1);
            }
        }
    }

    gpu.queue.submit(std::iter::once(encoder.finish()));
    frame.present();
}

fn main() {
    env_logger::init();
    println!("═══════════════════════════════════════");
    println!("  Step 07: LayerStack Compositing");
    println!("═══════════════════════════════════════");

    let mut viewport = Viewport::new(800, 600);
    viewport.center = GeoCoord::new(0.0, 0.0);
    viewport.zoom = 2.0;

    let mut layer_stack = LayerStack::new();
    layer_stack.add_layer(TileRenderLayer::new("base"));
    layer_stack.add_layer(TileRenderLayer::new("overlay"));

    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        gpu: None,
        viewport,
        camera: CameraController::new(),
        layer_stack,
        base_opacity: 1.0,
        overlay_opacity: 0.6,
    };
    event_loop.run_app(&mut app).unwrap();
}
