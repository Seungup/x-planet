//! Shared GPU resources used by all renderers (tile, terrain, model3d).
//!
//! Eliminates duplication of viewport bind group layouts, depth textures,
//! samplers, and uniform buffers that were previously created independently
//! by each renderer.  A single `SharedRenderResources` is created once and
//! passed to all renderers by reference.

use x_planets_gpu::GpuContext;
use x_planets_math::ViewportUniforms;

// ═══════════════════════════════════════════════════════════════════
// SharedRenderResources
// ═══════════════════════════════════════════════════════════════════

/// GPU resources shared across all renderers.
///
/// Created once at startup and passed to each renderer.  Call
/// [`resize`] on window resize and [`update_viewport`] once per frame
/// before any renderer draws.
pub struct SharedRenderResources {
    // ── Depth buffer (single, shared across all render passes) ──
    pub depth_view: wgpu::TextureView,
    pub depth_format: wgpu::TextureFormat,
    surface_width: u32,
    surface_height: u32,

    // ── Viewport (bind group 0 — identical layout for all renderers) ──
    pub viewport_bgl: wgpu::BindGroupLayout,
    pub viewport_buffer: wgpu::Buffer,
    pub viewport_bg: wgpu::BindGroup,

    // ── Per-tile / per-model bind group layout (group 1) ──
    // uniform buffer + texture2D + sampler — used by tile, terrain, and model3d
    pub tile_bgl: wgpu::BindGroupLayout,

    // ── Shared linear sampler ──
    pub sampler: wgpu::Sampler,
}

impl SharedRenderResources {
    /// Create shared GPU resources.
    ///
    /// Requires a `GpuContext` with a surface (panics if headless).
    pub fn new(gpu: &GpuContext) -> Self {
        // ── Viewport bind group layout (group 0) ──
        let viewport_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("shared-viewport-bgl"),
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

        // ── Per-tile / per-model bind group layout (group 1) ──
        // Uniform buffer (VERTEX|FRAGMENT) + Texture2D (FRAGMENT) + Sampler (FRAGMENT)
        let tile_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("shared-tile-bgl"),
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
            gpu.create_uniform_buffer("shared-viewport-uniforms", &viewport_uniforms);

        let viewport_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shared-viewport-bg"),
            layout: &viewport_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: viewport_buffer.as_entire_binding(),
            }],
        });

        // ── Sampler ──
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shared-sampler"),
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
        let depth_format = Self::depth_format();
        let depth_view =
            Self::create_depth_texture(&gpu.device, surface_width, surface_height, depth_format);

        log::info!(
            "SharedRenderResources created (depth: {:?}, {}x{})",
            depth_format,
            surface_width,
            surface_height,
        );

        Self {
            depth_view,
            depth_format,
            surface_width,
            surface_height,
            viewport_bgl,
            viewport_buffer,
            viewport_bg,
            tile_bgl,
            sampler,
        }
    }

    /// Platform-appropriate depth format.
    /// Depth24Plus for WebGL2 fallback; Depth32Float for native.
    pub fn depth_format() -> wgpu::TextureFormat {
        #[cfg(target_arch = "wasm32")]
        {
            wgpu::TextureFormat::Depth24Plus
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            wgpu::TextureFormat::Depth32Float
        }
    }

    /// Recreate the depth texture after window resize.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width != self.surface_width || height != self.surface_height {
            self.surface_width = width;
            self.surface_height = height;
            self.depth_view =
                Self::create_depth_texture(device, width, height, self.depth_format);
        }
    }

    /// Update the viewport uniform buffer.  Call once per frame before rendering.
    pub fn update_viewport(&self, gpu: &GpuContext, uniforms: &ViewportUniforms) {
        gpu.update_buffer(&self.viewport_buffer, uniforms);
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shared-depth-texture"),
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
}
