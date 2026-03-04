//! x-planets-projection: Extensible map projection system.
//!
//! Provides a plugin architecture where projections are defined as WGSL shader
//! snippets with CPU fallbacks. Users can register custom projections at runtime.

pub mod builtins;
pub mod registry;

use glam::DVec3;

/// Core trait for map projection plugins.
///
/// Each projection provides:
/// - A WGSL shader function that transforms world coordinates to projected coordinates
/// - A CPU fallback for testing and tile coordinate calculations
/// - Uniform data that gets passed to the shader
/// - A rendering mode that determines tile mesh construction, camera, and interaction
pub trait ProjectionPlugin: Send + Sync {
    /// Human-readable name of the projection.
    fn name(&self) -> &str;

    /// EPSG code if applicable (e.g., "EPSG:3857" for Web Mercator).
    fn epsg_code(&self) -> Option<&str> {
        None
    }

    /// Rendering mode for tile positioning, camera control, and visible tile selection.
    ///
    /// This determines:
    /// - How tiles are positioned and rendered (flat quad vs sphere tessellation)
    /// - How the camera VP matrix is constructed (perspective in projection space vs orbital)
    /// - How visible tiles are selected (Mercator frustum vs spherical cap)
    /// - How user interaction (pan, zoom) is interpreted
    ///
    /// Defaults to `Mercator` (flat tile rendering with centered Mercator VP).
    fn rendering_mode(&self) -> x_planets_math::ProjectionMode {
        x_planets_math::ProjectionMode::Mercator
    }

    /// WGSL shader source snippet.
    ///
    /// Must define a function with signature:
    /// ```wgsl
    /// fn project(world_pos: vec3<f32>) -> vec3<f32>
    /// ```
    /// where `world_pos` is (latitude_deg, longitude_deg, altitude) and
    /// the return value is normalized projected coordinates.
    fn shader_source(&self) -> &str;

    /// The name of the projection function in the shader (default: "project").
    fn shader_function_name(&self) -> &str {
        "project"
    }

    /// CPU-side projection for testing and tile coordinate calculations.
    /// Input: (lat_deg, lon_deg, altitude) → projected (x, y, z)
    fn project_cpu(&self, world_pos: DVec3) -> DVec3;

    /// CPU-side inverse projection.
    /// Input: projected (x, y, z) → (lat_deg, lon_deg, altitude)
    fn unproject_cpu(&self, projected: DVec3) -> DVec3;

    /// Uniform buffer data to pass to the shader (serialized as bytes).
    /// Override if the projection needs runtime parameters.
    fn uniform_data(&self) -> Vec<u8> {
        Vec::new()
    }

    /// Valid latitude range for this projection.
    fn latitude_range(&self) -> (f64, f64) {
        (-90.0, 90.0)
    }

    /// Valid longitude range for this projection.
    fn longitude_range(&self) -> (f64, f64) {
        (-180.0, 180.0)
    }
}

/// Compose multiple WGSL shader snippets into one module.
///
/// The base projection shader is included first, followed by any transform
/// shaders. The final composed shader calls them in sequence.
pub fn compose_shaders(base: &str, transforms: &[&str]) -> String {
    let mut source = String::new();

    // Include base projection
    source.push_str("// === Base Projection ===\n");
    source.push_str(base);
    source.push('\n');

    // Include transform shaders
    for (i, transform) in transforms.iter().enumerate() {
        source.push_str(&format!("\n// === Transform {} ===\n", i));
        source.push_str(transform);
        source.push('\n');
    }

    source
}

pub use builtins::{Globe, Mercator};
pub use registry::ProjectionRegistry;
