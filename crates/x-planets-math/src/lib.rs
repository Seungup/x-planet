//! x-planets-math: Geospatial math utilities for the x-planets rendering engine.
//!
//! Provides geographic coordinate types, tile coordinate systems,
//! bounding box calculations, ECEF coordinate transforms, and projection-related math primitives.

pub mod ecef;
mod frustum;
mod geo;
mod polygon;
mod projection;
mod uniforms;

pub use frustum::*;
pub use geo::*;
pub use polygon::*;
pub use projection::*;
pub use uniforms::*;

pub use glam::{DMat3, DMat4, DVec2, DVec3, Mat4, Vec2, Vec3, Vec4};
