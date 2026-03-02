//! Shared utilities for x-planets examples.
//!
//! Each example is a standalone milestone in the Karpathy approach:
//! start with the stupidest thing that works, add one thing at a time.

/// Common GPU initialization for examples.
pub mod common {
    use std::sync::Arc;
    use winit::window::Window;

    pub struct ExampleGpu {
        pub surface: wgpu::Surface<'static>,
        pub device: wgpu::Device,
        pub queue: wgpu::Queue,
        pub config: wgpu::SurfaceConfiguration,
    }

    impl ExampleGpu {
        pub async fn new(window: Arc<Window>) -> Self {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
            let surface = instance.create_surface(window.clone()).unwrap();

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: Some(&surface),
                    ..Default::default()
                })
                .await
                .expect("No suitable GPU adapter found");

            log::info!("GPU adapter: {}", adapter.get_info().name);

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .expect("Failed to create device");

            let size = window.inner_size();
            let config = surface
                .get_default_config(&adapter, size.width.max(1), size.height.max(1))
                .expect("Surface not supported");
            surface.configure(&device, &config);

            Self {
                surface,
                device,
                queue,
                config,
            }
        }

        pub fn resize(&mut self, width: u32, height: u32) {
            self.config.width = width.max(1);
            self.config.height = height.max(1);
            self.surface.configure(&self.device, &self.config);
        }
    }
}
