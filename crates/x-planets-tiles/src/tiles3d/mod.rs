//! OGC 3D Tiles support (1.0 / 1.1).
//!
//! Provides parsing for tileset.json, bounding volume types,
//! screen-space error calculation, and tile format decoders.
//!
//! Supports both 3D Tiles 1.0 (B3DM/I3DM/PNTS) and 1.1 (glTF/GLB) content.

pub mod b3dm;
pub mod bounding_volume;
pub mod decoder;
pub mod gltf_mesh;
pub mod tileset;
pub mod traversal;
