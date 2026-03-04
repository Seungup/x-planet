//! Built-in projection implementations: Mercator and Globe.

use crate::ProjectionPlugin;
use glam::DVec3;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Web Mercator (EPSG:3857)
// ---------------------------------------------------------------------------

/// Web Mercator projection (EPSG:3857).
///
/// Projects geographic coordinates to a square (0..1, 0..1) space
/// suitable for slippy map tile rendering.
pub struct Mercator;

impl ProjectionPlugin for Mercator {
    fn name(&self) -> &str {
        "Web Mercator"
    }

    fn epsg_code(&self) -> Option<&str> {
        Some("EPSG:3857")
    }

    fn rendering_mode(&self) -> x_planets_math::ProjectionMode {
        x_planets_math::ProjectionMode::Mercator
    }

    fn shader_source(&self) -> &str {
        MERCATOR_WGSL
    }

    fn project_cpu(&self, world_pos: DVec3) -> DVec3 {
        let lat = world_pos.x;
        let lon = world_pos.y;
        let alt = world_pos.z;

        let x = (lon + 180.0) / 360.0;
        let lat_rad = lat.to_radians();
        let y = (1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / PI) / 2.0;

        DVec3::new(x, y, alt)
    }

    fn unproject_cpu(&self, projected: DVec3) -> DVec3 {
        let lon = projected.x * 360.0 - 180.0;
        let lat_rad = (PI * (1.0 - 2.0 * projected.y)).sinh().atan();
        let lat = lat_rad.to_degrees();

        DVec3::new(lat, lon, projected.z)
    }

    fn latitude_range(&self) -> (f64, f64) {
        (-85.0511, 85.0511) // Mercator limit
    }
}

const MERCATOR_WGSL: &str = r#"
const PI: f32 = 3.14159265358979323846;

fn project(world_pos: vec3<f32>) -> vec3<f32> {
    let lat = world_pos.x;
    let lon = world_pos.y;
    let alt = world_pos.z;

    let x = (lon + 180.0) / 360.0;
    let lat_rad = radians(lat);
    let sin_lat = sin(lat_rad);
    let y = 0.5 - 0.5 * log((1.0 + sin_lat) / (1.0 - sin_lat)) / (2.0 * PI);

    return vec3<f32>(x, y, alt);
}
"#;

// ---------------------------------------------------------------------------
// Globe (3D sphere rendering)
// ---------------------------------------------------------------------------

/// Globe projection — renders tiles on a 3D sphere (like Google Earth).
///
/// Internally uses equirectangular math for CPU-side coordinate mapping,
/// but renders via the Globe pipeline (orbital camera, spherical mesh).
pub struct Globe;

impl ProjectionPlugin for Globe {
    fn name(&self) -> &str {
        "Globe"
    }

    fn epsg_code(&self) -> Option<&str> {
        Some("EPSG:4326")
    }

    fn rendering_mode(&self) -> x_planets_math::ProjectionMode {
        x_planets_math::ProjectionMode::Globe
    }

    fn shader_source(&self) -> &str {
        EQUIRECTANGULAR_WGSL
    }

    fn project_cpu(&self, world_pos: DVec3) -> DVec3 {
        let lat = world_pos.x;
        let lon = world_pos.y;
        let alt = world_pos.z;

        let x = (lon + 180.0) / 360.0;
        let y = (90.0 - lat) / 180.0;

        DVec3::new(x, y, alt)
    }

    fn unproject_cpu(&self, projected: DVec3) -> DVec3 {
        let lon = projected.x * 360.0 - 180.0;
        let lat = 90.0 - projected.y * 180.0;

        DVec3::new(lat, lon, projected.z)
    }
}

const EQUIRECTANGULAR_WGSL: &str = r#"
fn project(world_pos: vec3<f32>) -> vec3<f32> {
    let lat = world_pos.x;
    let lon = world_pos.y;
    let alt = world_pos.z;

    let x = (lon + 180.0) / 360.0;
    let y = (90.0 - lat) / 180.0;

    return vec3<f32>(x, y, alt);
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mercator_projection_origin() {
        let proj = Mercator;
        let result = proj.project_cpu(DVec3::new(0.0, 0.0, 0.0));
        assert!((result.x - 0.5).abs() < 1e-10);
        assert!((result.y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mercator_roundtrip() {
        let proj = Mercator;
        let original = DVec3::new(48.8566, 2.3522, 0.0); // Paris
        let projected = proj.project_cpu(original);
        let recovered = proj.unproject_cpu(projected);
        assert!((original.x - recovered.x).abs() < 1e-8);
        assert!((original.y - recovered.y).abs() < 1e-8);
    }

    #[test]
    fn test_globe_origin() {
        let proj = Globe;
        let result = proj.project_cpu(DVec3::new(0.0, 0.0, 0.0));
        assert!((result.x - 0.5).abs() < 1e-10);
        assert!((result.y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_globe_roundtrip() {
        let proj = Globe;
        let original = DVec3::new(35.6762, 139.6503, 0.0); // Tokyo
        let projected = proj.project_cpu(original);
        let recovered = proj.unproject_cpu(projected);
        assert!((original.x - recovered.x).abs() < 1e-8);
        assert!((original.y - recovered.y).abs() < 1e-8);
    }

    #[test]
    fn test_mercator_rendering_mode() {
        let proj = Mercator;
        assert_eq!(
            proj.rendering_mode(),
            x_planets_math::ProjectionMode::Mercator,
        );
    }

    #[test]
    fn test_globe_rendering_mode() {
        let proj = Globe;
        assert_eq!(
            proj.rendering_mode(),
            x_planets_math::ProjectionMode::Globe,
        );
    }

    #[test]
    fn test_mercator_corners() {
        let proj = Mercator;

        // Top-left: lat=85.0511, lon=-180
        let tl = proj.project_cpu(DVec3::new(85.0511, -180.0, 0.0));
        assert!(tl.x.abs() < 1e-4);
        assert!(tl.y.abs() < 1e-2); // near 0

        // Bottom-right: lat=-85.0511, lon=180
        let br = proj.project_cpu(DVec3::new(-85.0511, 180.0, 0.0));
        assert!((br.x - 1.0).abs() < 1e-4);
        assert!((br.y - 1.0).abs() < 1e-2); // near 1
    }
}
