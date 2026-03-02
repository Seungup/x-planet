//! x-planets-core: Core map rendering engine.
//!
//! Architecture follows the Karpathy principle:
//! - `pipeline`: pure functions (input→output, no side effects, independently testable)
//! - `verify_chain`: step-by-step verification framework
//! - `engine`: orchestrator that wires pipeline stages together
//! - `render`: GPU vertex layouts and layer management

pub mod engine;
pub mod pipeline;
pub mod render;
pub mod tile_renderer;
pub mod verify_chain;
pub mod viewport;

pub use engine::MapEngine;
pub use pipeline::FrameSummary;
pub use tile_renderer::TileRenderer;
pub use viewport::{CameraController, Viewport};
