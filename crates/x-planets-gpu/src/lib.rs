//! x-planets-gpu: GPU abstraction layer for the x-planets rendering engine.
//!
//! Wraps wgpu to provide:
//! - GPU context management (device, queue, surface)
//! - Texture creation and management
//! - Render and compute pipeline builders
//! - Test utilities (CPU reference, pixel diff, snapshot)

pub mod context;
pub mod pipeline;
pub mod test_utils;
pub mod texture;

pub use context::{GpuContext, GpuError, GpuSurface};
pub use pipeline::{ComputePipelineBuilder, RenderPipelineBuilder};
pub use texture::{GpuTexture, TextureManager};
