//! x-planets-render: GPU rendering layer for x-planets.
//!
//! Re-exports renderers and GPU-dependent types from `x-planets-core` (with the
//! `gpu` feature enabled).  This crate serves as the single dependency for
//! platform crates (native, web) that need GPU rendering.
//!
//! Crates that only need viewport/camera/layer control (e.g. FFI bindings) can
//! depend on `x-planets-core` directly *without* the `gpu` feature.

// Re-export renderers from core (gpu feature is enabled via our Cargo.toml dependency)
pub use x_planets_core::tile_renderer::TileRenderer;
pub use x_planets_core::terrain_renderer::{TerrainLayerData, TerrainRenderer};
pub use x_planets_core::model3d_renderer::{GpuModel3d, Model3dRenderer, Model3dVertex, ModelUniforms};

// Re-export GPU-dependent render data types
pub use x_planets_core::render::RenderLayerData;
pub use x_planets_core::map_controller::RenderOutput;

// Re-export terrain data (also available without gpu feature)
pub use x_planets_core::terrain_data::TerrainTileData;
