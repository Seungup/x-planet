//! GPU context: device, queue, and surface management.

use thiserror::Error;
use wgpu;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("Failed to request adapter")]
    AdapterNotFound,
    #[error("Failed to request device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
    #[error("Surface error: {0}")]
    Surface(#[from] wgpu::CreateSurfaceError),
    #[error("Surface configuration not supported by adapter")]
    SurfaceConfig,
}

/// Core GPU context wrapping wgpu primitives.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: Option<GpuSurface>,
}

/// Surface configuration for presenting to a window.
pub struct GpuSurface {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
}

impl GpuContext {
    /// Create a GPU context with a renderable surface.
    ///
    /// Accepts any window handle that wgpu can create a surface from:
    /// - `Arc<winit::window::Window>` (desktop)
    /// - `Arc<web_sys::HtmlCanvasElement>` (web)
    ///
    /// Karpathy step: "Get pixels on screen before doing anything clever."
    pub async fn new_with_window(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::AdapterNotFound)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("x-planets-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        let config = surface
            .get_default_config(&adapter, width.max(1), height.max(1))
            .ok_or(GpuError::SurfaceConfig)?;
        surface.configure(&device, &config);

        log::info!(
            "GPU context initialized: {} ({}x{}, {:?})",
            adapter.get_info().name,
            config.width,
            config.height,
            config.format,
        );

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface: Some(GpuSurface { surface, config }),
        })
    }

    /// Create a new GPU context without a surface (headless / compute only).
    pub async fn new_headless() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::AdapterNotFound)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("x-planets-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface: None,
        })
    }

    /// Create a uniform buffer from data.
    pub fn create_uniform_buffer<T: bytemuck::Pod>(
        &self,
        label: &str,
        data: &T,
    ) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Create a storage buffer from a slice of data.
    pub fn create_storage_buffer<T: bytemuck::Pod>(
        &self,
        label: &str,
        data: &[T],
    ) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Create a vertex buffer from data.
    pub fn create_vertex_buffer<T: bytemuck::Pod>(
        &self,
        label: &str,
        data: &[T],
    ) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::VERTEX,
            })
    }

    /// Create an index buffer.
    pub fn create_index_buffer(&self, label: &str, indices: &[u32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            })
    }

    /// Update a uniform buffer with new data.
    pub fn update_buffer<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, data: &T) {
        self.queue.write_buffer(buffer, 0, bytemuck::bytes_of(data));
    }

    /// Resize the surface. No-op if headless.
    pub fn resize_surface(&mut self, width: u32, height: u32) {
        if let Some(ref mut surf) = self.surface {
            surf.config.width = width.max(1);
            surf.config.height = height.max(1);
            surf.surface.configure(&self.device, &surf.config);
        }
    }

    /// Get the surface texture format. Returns None if headless.
    pub fn surface_format(&self) -> Option<wgpu::TextureFormat> {
        self.surface.as_ref().map(|s| s.config.format)
    }

    /// Get adapter info for debugging.
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }
}
