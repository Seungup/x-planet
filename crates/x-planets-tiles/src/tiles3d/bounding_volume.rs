//! OGC 3D Tiles bounding volume types and screen-space error.
//!
//! Supports all three bounding volume types from the spec:
//! - **Region**: geographic bounds (west, south, east, north, min_height, max_height) in radians
//! - **OrientedBox**: center + half-axes (12 floats)
//! - **Sphere**: center + radius (4 floats)
//!
//! Provides frustum culling and SSE calculations for LOD traversal.

use glam::{DMat3, DMat4, DVec3, DVec4};
use serde::{Deserialize, Serialize};

use x_planets_math::ecef::{geodetic_to_ecef, WGS84};

// ═══════════════════════════════════════════════════════════════════
// Serde bounding volume (raw JSON representation)
// ═══════════════════════════════════════════════════════════════════

/// Bounding volume as stored in tileset.json.
///
/// Exactly one of `region`, `box`, or `sphere` should be set.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct BoundingVolume {
    /// `[west, south, east, north, minHeight, maxHeight]` in radians + meters.
    #[serde(default)]
    pub region: Option<[f64; 6]>,
    /// `[cx, cy, cz, xx, xy, xz, yx, yy, yz, zx, zy, zz]` — center + 3 half-axis vectors.
    #[serde(rename = "box", default)]
    pub bounding_box: Option<[f64; 12]>,
    /// `[cx, cy, cz, radius]` — center + radius in meters.
    #[serde(default)]
    pub sphere: Option<[f64; 4]>,
}

impl BoundingVolume {
    /// Convert to a resolved `BoundingVolumeKind` for geometric operations.
    pub fn to_kind(&self) -> Option<BoundingVolumeKind> {
        if let Some(region) = &self.region {
            Some(BoundingVolumeKind::Region {
                west: region[0],
                south: region[1],
                east: region[2],
                north: region[3],
                min_height: region[4],
                max_height: region[5],
            })
        } else if let Some(b) = &self.bounding_box {
            Some(BoundingVolumeKind::OrientedBox {
                center: DVec3::new(b[0], b[1], b[2]),
                half_axes: DMat3::from_cols(
                    DVec3::new(b[3], b[4], b[5]),
                    DVec3::new(b[6], b[7], b[8]),
                    DVec3::new(b[9], b[10], b[11]),
                ),
            })
        } else {
            self.sphere.as_ref().map(|s| BoundingVolumeKind::Sphere {
                center: DVec3::new(s[0], s[1], s[2]),
                radius: s[3],
            })
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Resolved bounding volume (for geometric operations)
// ═══════════════════════════════════════════════════════════════════

/// A resolved bounding volume for geometric operations.
#[derive(Debug, Clone)]
pub enum BoundingVolumeKind {
    /// Geographic region in radians + meters.
    Region {
        west: f64,
        south: f64,
        east: f64,
        north: f64,
        min_height: f64,
        max_height: f64,
    },
    /// Oriented bounding box (center + 3 half-axis column vectors).
    OrientedBox {
        center: DVec3,
        half_axes: DMat3,
    },
    /// Bounding sphere.
    Sphere {
        center: DVec3,
        radius: f64,
    },
}

impl BoundingVolumeKind {
    /// Get the center of the bounding volume in ECEF coordinates.
    pub fn center_ecef(&self) -> DVec3 {
        match self {
            BoundingVolumeKind::Region {
                west,
                south,
                east,
                north,
                min_height,
                max_height,
            } => {
                let lat = (south + north) / 2.0;
                let lon = (west + east) / 2.0;
                let height = (min_height + max_height) / 2.0;
                geodetic_to_ecef(lat, lon, height, &WGS84)
            }
            BoundingVolumeKind::OrientedBox { center, .. } => *center,
            BoundingVolumeKind::Sphere { center, .. } => *center,
        }
    }

    /// Approximate bounding radius from the center (for distance-based SSE).
    pub fn bounding_radius(&self) -> f64 {
        match self {
            BoundingVolumeKind::Region {
                west,
                south,
                east,
                north,
                min_height,
                max_height,
            } => {
                // Approximate: compute ECEF corners and find max distance from center.
                let center = self.center_ecef();
                let corners = [
                    geodetic_to_ecef(*south, *west, *min_height, &WGS84),
                    geodetic_to_ecef(*south, *east, *min_height, &WGS84),
                    geodetic_to_ecef(*north, *west, *min_height, &WGS84),
                    geodetic_to_ecef(*north, *east, *min_height, &WGS84),
                    geodetic_to_ecef(*south, *west, *max_height, &WGS84),
                    geodetic_to_ecef(*south, *east, *max_height, &WGS84),
                    geodetic_to_ecef(*north, *west, *max_height, &WGS84),
                    geodetic_to_ecef(*north, *east, *max_height, &WGS84),
                ];
                corners
                    .iter()
                    .map(|c| (*c - center).length())
                    .fold(0.0_f64, f64::max)
            }
            BoundingVolumeKind::OrientedBox { half_axes, .. } => {
                // Sum of half-axis lengths = max distance from center to any corner.
                let ax = half_axes.col(0).length();
                let ay = half_axes.col(1).length();
                let az = half_axes.col(2).length();
                (ax * ax + ay * ay + az * az).sqrt()
            }
            BoundingVolumeKind::Sphere { radius, .. } => *radius,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Frustum culling
// ═══════════════════════════════════════════════════════════════════

/// A frustum plane in Hessian normal form (normal · p + d = 0).
#[derive(Debug, Clone, Copy)]
pub struct FrustumPlane {
    pub normal: DVec3,
    pub d: f64,
}

/// Extract 6 frustum planes from a view-projection matrix.
///
/// Returns `[left, right, bottom, top, near, far]`.
pub fn extract_frustum_planes(view_proj: &DMat4) -> [FrustumPlane; 6] {
    let m = *view_proj;
    let row0 = DVec4::new(m.col(0).x, m.col(1).x, m.col(2).x, m.col(3).x);
    let row1 = DVec4::new(m.col(0).y, m.col(1).y, m.col(2).y, m.col(3).y);
    let row2 = DVec4::new(m.col(0).z, m.col(1).z, m.col(2).z, m.col(3).z);
    let row3 = DVec4::new(m.col(0).w, m.col(1).w, m.col(2).w, m.col(3).w);

    let raw = [
        row3 + row0, // left
        row3 - row0, // right
        row3 + row1, // bottom
        row3 - row1, // top
        row3 + row2, // near
        row3 - row2, // far
    ];

    let mut planes = [FrustumPlane {
        normal: DVec3::ZERO,
        d: 0.0,
    }; 6];

    for (i, r) in raw.iter().enumerate() {
        let len = DVec3::new(r.x, r.y, r.z).length();
        if len > 1e-12 {
            planes[i] = FrustumPlane {
                normal: DVec3::new(r.x / len, r.y / len, r.z / len),
                d: r.w / len,
            };
        }
    }

    planes
}

/// Test if a bounding sphere is visible against the 6 frustum planes.
///
/// Returns `true` if the sphere is at least partially inside the frustum.
pub fn is_sphere_visible(center: DVec3, radius: f64, planes: &[FrustumPlane; 6]) -> bool {
    for plane in planes {
        let dist = plane.normal.dot(center) + plane.d;
        if dist < -radius {
            return false; // Entirely outside this plane.
        }
    }
    true
}

/// Test if a bounding volume is visible within the view frustum.
///
/// Uses a conservative sphere test for all volume types.
pub fn is_visible(volume: &BoundingVolumeKind, view_proj: &DMat4) -> bool {
    let planes = extract_frustum_planes(view_proj);
    let center = volume.center_ecef();
    let radius = volume.bounding_radius();
    is_sphere_visible(center, radius, &planes)
}

// ═══════════════════════════════════════════════════════════════════
// Screen-space error (SSE)
// ═══════════════════════════════════════════════════════════════════

/// Calculate the screen-space error (SSE) in pixels.
///
/// This is the core LOD metric for 3D Tiles traversal.
///
/// # Formula
/// ```text
/// SSE = geometric_error × screen_height / (2 × distance × tan(fov_y / 2))
/// ```
///
/// # Parameters
/// - `geometric_error`: The tile's geometric error in meters.
/// - `distance`: Distance from camera to tile center in meters.
/// - `screen_height`: Viewport height in pixels.
/// - `fov_y`: Vertical field of view in radians.
///
/// # Returns
/// SSE in pixels. Higher values mean the tile's error is visually significant
/// and children should be loaded for more detail.
pub fn screen_space_error(
    geometric_error: f64,
    distance: f64,
    screen_height: f64,
    fov_y: f64,
) -> f64 {
    if distance <= 0.0 {
        return f64::MAX;
    }
    let half_tan = (fov_y / 2.0).tan();
    if half_tan <= 0.0 {
        return f64::MAX;
    }
    geometric_error * screen_height / (2.0 * distance * half_tan)
}

/// Calculate the distance from camera to the closest point on a bounding volume.
///
/// For SSE calculations, we use the distance to the center minus the bounding
/// radius (clamped to a minimum of 1.0 to avoid division by zero).
pub fn distance_to_volume(camera_ecef: DVec3, volume: &BoundingVolumeKind) -> f64 {
    let center = volume.center_ecef();
    let radius = volume.bounding_radius();
    let dist_to_center = (camera_ecef - center).length();
    (dist_to_center - radius).max(1.0)
}

// ═══════════════════════════════════════════════════════════════════
// Transform helpers
// ═══════════════════════════════════════════════════════════════════

/// Apply a column-major 4×4 transform to a bounding volume.
///
/// Used when tiles have a `transform` property in the tileset.
pub fn transform_volume(volume: &BoundingVolumeKind, transform: &[f64; 16]) -> BoundingVolumeKind {
    let m = DMat4::from_cols_array(transform);

    match volume {
        BoundingVolumeKind::OrientedBox { center, half_axes } => {
            let new_center = (m * DVec4::new(center.x, center.y, center.z, 1.0)).truncate();
            let upper3x3 = DMat3::from_cols(
                m.col(0).truncate(),
                m.col(1).truncate(),
                m.col(2).truncate(),
            );
            let new_half_axes = upper3x3 * *half_axes;
            BoundingVolumeKind::OrientedBox {
                center: new_center,
                half_axes: new_half_axes,
            }
        }
        BoundingVolumeKind::Sphere { center, radius } => {
            let new_center = (m * DVec4::new(center.x, center.y, center.z, 1.0)).truncate();
            // Scale radius by the max scale factor of the transform.
            let scale_x = m.col(0).truncate().length();
            let scale_y = m.col(1).truncate().length();
            let scale_z = m.col(2).truncate().length();
            let max_scale = scale_x.max(scale_y).max(scale_z);
            BoundingVolumeKind::Sphere {
                center: new_center,
                radius: radius * max_scale,
            }
        }
        BoundingVolumeKind::Region { .. } => {
            // Region volumes are in geographic coordinates and typically
            // don't have tile-level transforms applied. Return as-is.
            volume.clone()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_4;

    #[test]
    fn test_bounding_volume_region_parse() {
        let bv = BoundingVolume {
            region: Some([-1.3197, 0.6988, -1.3196, 0.6989, 0.0, 100.0]),
            bounding_box: None,
            sphere: None,
        };
        let kind = bv.to_kind().unwrap();
        match kind {
            BoundingVolumeKind::Region {
                west, north, max_height, ..
            } => {
                assert!((west - (-1.3197)).abs() < 1e-10);
                assert!((north - 0.6989).abs() < 1e-10);
                assert!((max_height - 100.0).abs() < 1e-10);
            }
            _ => panic!("Expected Region"),
        }
    }

    #[test]
    fn test_bounding_volume_box_parse() {
        let bv = BoundingVolume {
            region: None,
            bounding_box: Some([
                0.0, 0.0, 0.0, // center
                1.0, 0.0, 0.0, // half-axis X
                0.0, 1.0, 0.0, // half-axis Y
                0.0, 0.0, 1.0, // half-axis Z
            ]),
            sphere: None,
        };
        let kind = bv.to_kind().unwrap();
        match kind {
            BoundingVolumeKind::OrientedBox { center, half_axes } => {
                assert_eq!(center, DVec3::ZERO);
                assert!((half_axes.col(0) - DVec3::X).length() < 1e-10);
                assert!((half_axes.col(1) - DVec3::Y).length() < 1e-10);
                assert!((half_axes.col(2) - DVec3::Z).length() < 1e-10);
            }
            _ => panic!("Expected OrientedBox"),
        }
    }

    #[test]
    fn test_bounding_volume_sphere_parse() {
        let bv = BoundingVolume {
            region: None,
            bounding_box: None,
            sphere: Some([100.0, 200.0, 300.0, 50.0]),
        };
        let kind = bv.to_kind().unwrap();
        match kind {
            BoundingVolumeKind::Sphere { center, radius } => {
                assert_eq!(center, DVec3::new(100.0, 200.0, 300.0));
                assert!((radius - 50.0).abs() < 1e-10);
            }
            _ => panic!("Expected Sphere"),
        }
    }

    #[test]
    fn test_sphere_bounding_radius() {
        let vol = BoundingVolumeKind::Sphere {
            center: DVec3::new(1000.0, 2000.0, 3000.0),
            radius: 42.0,
        };
        assert!((vol.bounding_radius() - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_obb_bounding_radius() {
        // Unit cube: half-axes are (1,0,0), (0,1,0), (0,0,1)
        // bounding radius = sqrt(1²+1²+1²) = sqrt(3)
        let vol = BoundingVolumeKind::OrientedBox {
            center: DVec3::ZERO,
            half_axes: DMat3::IDENTITY,
        };
        let expected = 3.0_f64.sqrt();
        assert!((vol.bounding_radius() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_screen_space_error_basic() {
        // At 1000m distance, with geometric error 10m, 1080p screen, 60° FOV
        let sse = screen_space_error(10.0, 1000.0, 1080.0, 60.0_f64.to_radians());
        // SSE = 10 * 1080 / (2 * 1000 * tan(30°)) = 10800 / (2000 * 0.5774) ≈ 9.35
        assert!(sse > 9.0 && sse < 10.0, "SSE was {sse}");
    }

    #[test]
    fn test_screen_space_error_closer_means_larger() {
        let fov = 60.0_f64.to_radians();
        let sse_far = screen_space_error(10.0, 10000.0, 1080.0, fov);
        let sse_near = screen_space_error(10.0, 100.0, 1080.0, fov);
        assert!(sse_near > sse_far, "Near SSE should be larger");
    }

    #[test]
    fn test_screen_space_error_zero_distance() {
        let sse = screen_space_error(10.0, 0.0, 1080.0, 60.0_f64.to_radians());
        assert_eq!(sse, f64::MAX);
    }

    #[test]
    fn test_screen_space_error_zero_geometric_error() {
        let sse = screen_space_error(0.0, 1000.0, 1080.0, 60.0_f64.to_radians());
        assert!((sse - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_to_volume_sphere() {
        let vol = BoundingVolumeKind::Sphere {
            center: DVec3::new(100.0, 0.0, 0.0),
            radius: 10.0,
        };
        let camera = DVec3::ZERO;
        let dist = distance_to_volume(camera, &vol);
        // 100 - 10 = 90
        assert!((dist - 90.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_to_volume_clamped() {
        // Camera inside the sphere — distance clamped to 1.0
        let vol = BoundingVolumeKind::Sphere {
            center: DVec3::new(5.0, 0.0, 0.0),
            radius: 100.0,
        };
        let camera = DVec3::ZERO;
        let dist = distance_to_volume(camera, &vol);
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_frustum_planes_sphere_visible() {
        // Create a simple perspective-like view-projection matrix
        // Looking down -Z axis, sphere at origin should be visible.
        let view_proj = DMat4::perspective_rh(
            FRAC_PI_4 * 2.0, // 90° FOV
            1.0,
            0.1,
            10000.0,
        );
        let planes = extract_frustum_planes(&view_proj);

        // Sphere at (0, 0, -100) should be visible
        assert!(is_sphere_visible(DVec3::new(0.0, 0.0, -100.0), 10.0, &planes));

        // Sphere behind camera at (0, 0, 100) should NOT be visible
        assert!(!is_sphere_visible(DVec3::new(0.0, 0.0, 100.0), 10.0, &planes));
    }

    #[test]
    fn test_frustum_sphere_far_left() {
        let view_proj = DMat4::perspective_rh(FRAC_PI_4 * 2.0, 1.0, 0.1, 10000.0);
        let planes = extract_frustum_planes(&view_proj);

        // Sphere far to the left should NOT be visible
        assert!(!is_sphere_visible(
            DVec3::new(-1000.0, 0.0, -50.0),
            5.0,
            &planes
        ));
    }

    #[test]
    fn test_transform_sphere() {
        let vol = BoundingVolumeKind::Sphere {
            center: DVec3::new(1.0, 0.0, 0.0),
            radius: 5.0,
        };
        // Translation by (10, 20, 30), uniform scale 2x
        let transform = [
            2.0, 0.0, 0.0, 0.0, // col 0
            0.0, 2.0, 0.0, 0.0, // col 1
            0.0, 0.0, 2.0, 0.0, // col 2
            10.0, 20.0, 30.0, 1.0, // col 3
        ];
        let transformed = transform_volume(&vol, &transform);
        match transformed {
            BoundingVolumeKind::Sphere { center, radius } => {
                // center = scale(2) * (1,0,0) + (10,20,30) = (12, 20, 30)
                assert!((center.x - 12.0).abs() < 1e-10);
                assert!((center.y - 20.0).abs() < 1e-10);
                assert!((center.z - 30.0).abs() < 1e-10);
                assert!((radius - 10.0).abs() < 1e-10); // 5 * 2
            }
            _ => panic!("Expected Sphere"),
        }
    }

    #[test]
    fn test_transform_obb() {
        let vol = BoundingVolumeKind::OrientedBox {
            center: DVec3::ZERO,
            half_axes: DMat3::IDENTITY,
        };
        // Translation by (5, 5, 5)
        let transform = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 5.0, 5.0, 1.0,
        ];
        let transformed = transform_volume(&vol, &transform);
        match transformed {
            BoundingVolumeKind::OrientedBox { center, half_axes } => {
                assert!((center - DVec3::new(5.0, 5.0, 5.0)).length() < 1e-10);
                // Half-axes should remain identity (no rotation/scale)
                assert!((half_axes.col(0) - DVec3::X).length() < 1e-10);
                assert!((half_axes.col(1) - DVec3::Y).length() < 1e-10);
                assert!((half_axes.col(2) - DVec3::Z).length() < 1e-10);
            }
            _ => panic!("Expected OrientedBox"),
        }
    }

    #[test]
    fn test_region_center_ecef() {
        // Region around the equator at prime meridian
        let vol = BoundingVolumeKind::Region {
            west: -0.01,
            south: -0.01,
            east: 0.01,
            north: 0.01,
            min_height: 0.0,
            max_height: 100.0,
        };
        let center = vol.center_ecef();
        // Should be near (WGS84_a + 50, 0, 0) since lat/lon ~= 0
        assert!(center.x > 6_000_000.0);
        assert!(center.y.abs() < 10_000.0);
        assert!(center.z.abs() < 10_000.0);
    }

    #[test]
    fn test_bounding_volume_serde_roundtrip() {
        let original = BoundingVolume {
            region: Some([-1.3, 0.6, -1.2, 0.7, 0.0, 500.0]),
            bounding_box: None,
            sphere: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: BoundingVolume = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.region.unwrap(), original.region.unwrap());
    }

    #[test]
    fn test_default_bounding_volume_has_no_kind() {
        let bv = BoundingVolume::default();
        assert!(bv.to_kind().is_none());
    }
}
