//! x-planets-core: Core map rendering engine.
//!
//! Architecture follows the Karpathy principle:
//! - `pipeline`: pure functions (input→output, no side effects, independently testable)
//! - `verify_chain`: step-by-step verification framework
//! - `engine`: orchestrator that wires pipeline stages together
//! - `render`: GPU vertex layouts and layer management

pub mod engine;
pub mod interaction;
pub mod layer_state;
pub mod map_controller;
pub mod tile_load_planner;
#[cfg(feature = "gpu")]
pub mod model3d_renderer;
pub mod pipeline;
pub mod render;
#[cfg(feature = "gpu")]
pub mod shared_render_resources;
pub mod terrain_data;
#[cfg(feature = "gpu")]
pub mod terrain_renderer;
#[cfg(feature = "gpu")]
pub mod tile_renderer;
#[cfg(feature = "gpu")]
pub mod tiles3d_pipeline;
#[cfg(feature = "gpu")]
pub mod tiles3d_state;
pub mod verify_chain;
pub mod viewport;

pub use engine::{LayerConfig, LayerKind, MapEngine, TileLayer};
pub use map_controller::{LayerInfo, LayerStateView, MapController, MapEvent};
#[cfg(feature = "gpu")]
pub use map_controller::RenderOutput;
#[cfg(feature = "gpu")]
pub use model3d_renderer::{GpuModel3d, Model3dRenderer, Model3dVertex, ModelUniforms};
pub use pipeline::FrameSummary;
#[cfg(feature = "gpu")]
pub use render::RenderLayerData;
#[cfg(feature = "gpu")]
pub use shared_render_resources::SharedRenderResources;
pub use terrain_data::TerrainTileData;
#[cfg(feature = "gpu")]
pub use terrain_renderer::{TerrainLayerData, TerrainRenderer};
#[cfg(feature = "gpu")]
pub use tile_renderer::TileRenderer;
#[cfg(feature = "gpu")]
pub use tiles3d_state::{GpuTileContent, Tiles3dGpuState};
pub use viewport::{CameraController, Viewport};
