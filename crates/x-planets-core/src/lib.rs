//! x-planets-core: Core map rendering engine.
//!
//! Architecture follows the Karpathy principle:
//! - `pipeline`: pure functions (input→output, no side effects, independently testable)
//! - `verify_chain`: step-by-step verification framework
//! - `engine`: orchestrator that wires pipeline stages together
//! - `render`: GPU vertex layouts and layer management

pub mod engine;
pub mod model3d_renderer;
pub mod pipeline;
pub mod render;
pub mod terrain_renderer;
pub mod tile_renderer;
pub mod tiles3d_pipeline;
pub mod verify_chain;
pub mod viewport;

pub use engine::{LayerConfig, LayerKind, MapEngine, TileLayer};
pub use model3d_renderer::{GpuModel3d, Model3dRenderer, Model3dVertex, ModelUniforms};
pub use pipeline::FrameSummary;
pub use render::RenderLayerData;
pub use terrain_renderer::{TerrainLayerData, TerrainRenderer, TerrainTileData};
pub use tile_renderer::TileRenderer;
pub use viewport::{CameraController, Viewport};
