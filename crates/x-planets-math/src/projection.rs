use glam::{DVec2, DVec3};
use std::f64::consts::PI;

use super::{GeoCoord};

// ---------------------------------------------------------------------------
// Web Mercator helpers
// ---------------------------------------------------------------------------

/// Convert latitude/longitude to Web Mercator normalized coordinates (0..1).
pub fn geo_to_mercator(coord: &GeoCoord) -> DVec2 {
    let x = (coord.lon + 180.0) / 360.0;
    let lat_rad = coord.lat.to_radians();
    let y = (1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / PI) / 2.0;
    DVec2::new(x, y)
}

/// Convert Web Mercator normalized coordinates (0..1) to latitude/longitude.
pub fn mercator_to_geo(pos: DVec2) -> GeoCoord {
    let lon = pos.x * 360.0 - 180.0;
    let lat_rad = (PI * (1.0 - 2.0 * pos.y)).sinh().atan();
    GeoCoord::new(lat_rad.to_degrees(), lon)
}

/// Convert a Mercator normalized y [0,1] to latitude in radians.
pub fn mercator_y_to_lat_rad(y: f64) -> f64 {
    (PI * (1.0 - 2.0 * y)).sinh().atan()
}

/// Convert geographic coordinates (radians) to a point on the unit sphere.
///
/// Returns `(cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat))`.
pub fn geo_to_unit_sphere(lat_rad: f64, lon_rad: f64) -> DVec3 {
    DVec3::new(
        lat_rad.cos() * lon_rad.cos(),
        lat_rad.cos() * lon_rad.sin(),
        lat_rad.sin(),
    )
}

/// Oblique Mercator projection centered on a given reference point.
///
/// Rotates the sphere so `(center_lat_rad, center_lon_rad)` maps to the
/// equator/prime-meridian, then applies standard Web Mercator.
/// This minimizes distortion near the viewport center.
///
/// Returns coordinates in [0, 1] x [0, 1] just like standard Mercator,
/// but centered on the given point instead of (0, 0).
pub fn oblique_mercator(
    lat_rad: f64,
    lon_rad: f64,
    center_lat_rad: f64,
    center_lon_rad: f64,
) -> DVec2 {
    // 1. Convert to 3D unit sphere
    let p = geo_to_unit_sphere(lat_rad, lon_rad);

    // 2. Rotate by -center_lon around Z axis (align center longitude to prime meridian)
    let sin_clon = center_lon_rad.sin();
    let cos_clon = center_lon_rad.cos();
    let rx = p.x * cos_clon + p.y * sin_clon;
    let ry = -p.x * sin_clon + p.y * cos_clon;
    let rz = p.z;

    // 3. Rotate by -center_lat around Y axis (align center latitude to equator)
    let sin_clat = center_lat_rad.sin();
    let cos_clat = center_lat_rad.cos();
    let fx = rx * cos_clat + rz * sin_clat;
    let fy = ry;
    let fz = -rx * sin_clat + rz * cos_clat;

    // 4. Convert back to lat/lon in the rotated frame
    let rot_lat = fz.asin();
    let rot_lon = fy.atan2(fx);

    // 5. Standard Mercator of the rotated coordinates
    let x = (rot_lon + PI) / (2.0 * PI);
    let y = (1.0 - (rot_lat.tan() + 1.0 / rot_lat.cos()).ln() / PI) / 2.0;
    DVec2::new(x, y)
}

/// Inverse of [`oblique_mercator`]: convert centered Mercator back to geographic (radians).
pub fn oblique_mercator_inverse(
    merc: DVec2,
    center_lat_rad: f64,
    center_lon_rad: f64,
) -> (f64, f64) {
    // 1. Inverse standard Mercator -> rotated lat/lon
    let rot_lon = merc.x * 2.0 * PI - PI;
    let rot_lat = (PI * (1.0 - 2.0 * merc.y)).sinh().atan();

    // 2. Convert to 3D
    let fx = rot_lat.cos() * rot_lon.cos();
    let fy = rot_lat.cos() * rot_lon.sin();
    let fz = rot_lat.sin();

    // 3. Inverse latitude rotation (+center_lat around Y)
    let sin_clat = center_lat_rad.sin();
    let cos_clat = center_lat_rad.cos();
    let rx = fx * cos_clat - fz * sin_clat;
    let ry = fy;
    let rz = fx * sin_clat + fz * cos_clat;

    // 4. Inverse longitude rotation (+center_lon around Z)
    let sin_clon = center_lon_rad.sin();
    let cos_clon = center_lon_rad.cos();
    let px = rx * cos_clon - ry * sin_clon;
    let py = rx * sin_clon + ry * cos_clon;
    let pz = rz;

    // 5. Convert back to lat/lon
    let lat_rad = pz.asin();
    let lon_rad = py.atan2(px);
    (lat_rad, lon_rad)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mercator_roundtrip() {
        let original = GeoCoord::new(48.8566, 2.3522); // Paris
        let mercator = geo_to_mercator(&original);
        let recovered = mercator_to_geo(mercator);
        assert!((original.lat - recovered.lat).abs() < 1e-10);
        assert!((original.lon - recovered.lon).abs() < 1e-10);
    }

    #[test]
    fn test_mercator_roundtrip_near_poles() {
        // Mercator roundtrip should work near the boundary latitude.
        for &lat in &[80.0, -80.0, 84.0, -84.0, 85.0, -85.0] {
            let original = GeoCoord::new(lat, 30.0);
            let merc = geo_to_mercator(&original);
            let recovered = mercator_to_geo(merc);
            assert!(
                (original.lat - recovered.lat).abs() < 1e-6,
                "Roundtrip failed at lat={}: got {:.6}",
                lat, recovered.lat
            );
            assert!(
                (original.lon - recovered.lon).abs() < 1e-6,
                "Roundtrip failed at lat={}: lon {:.6} != {:.6}",
                lat, original.lon, recovered.lon
            );
        }
    }

    #[test]
    fn test_mercator_y_range_near_poles() {
        // Mercator y should be within [0, 1] for valid latitudes
        let north = geo_to_mercator(&GeoCoord::new(85.0, 0.0));
        let south = geo_to_mercator(&GeoCoord::new(-85.0, 0.0));
        let equator = geo_to_mercator(&GeoCoord::new(0.0, 0.0));

        assert!(north.y > 0.0 && north.y < 0.1, "North pole merc.y={:.4}", north.y);
        assert!(south.y > 0.9 && south.y < 1.0, "South pole merc.y={:.4}", south.y);
        assert!((equator.y - 0.5).abs() < 1e-10, "Equator merc.y={:.4}", equator.y);
    }

    #[test]
    fn test_geo_to_unit_sphere_poles() {
        // North pole should be at (0, 0, 1)
        let north = geo_to_unit_sphere(std::f64::consts::FRAC_PI_2, 0.0);
        assert!((north.x).abs() < 1e-10);
        assert!((north.y).abs() < 1e-10);
        assert!((north.z - 1.0).abs() < 1e-10);

        // South pole should be at (0, 0, -1)
        let south = geo_to_unit_sphere(-std::f64::consts::FRAC_PI_2, 0.0);
        assert!((south.x).abs() < 1e-10);
        assert!((south.y).abs() < 1e-10);
        assert!((south.z + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_geo_to_unit_sphere_equator() {
        // Equator, prime meridian -> (1, 0, 0)
        let point = geo_to_unit_sphere(0.0, 0.0);
        assert!((point.x - 1.0).abs() < 1e-10);
        assert!((point.y).abs() < 1e-10);
        assert!((point.z).abs() < 1e-10);

        // Equator, 90 E -> (0, 1, 0)
        let east = geo_to_unit_sphere(0.0, std::f64::consts::FRAC_PI_2);
        assert!((east.x).abs() < 1e-10);
        assert!((east.y - 1.0).abs() < 1e-10);
        assert!((east.z).abs() < 1e-10);
    }

    // -- Oblique Mercator tests --

    #[test]
    fn test_oblique_mercator_center_maps_to_half() {
        // The center point should map to (0.5, 0.5) in oblique Mercator.
        let center_lat = 37.5_f64.to_radians();
        let center_lon = 127.0_f64.to_radians();
        let result = oblique_mercator(center_lat, center_lon, center_lat, center_lon);
        assert!((result.x - 0.5).abs() < 1e-10, "x={}", result.x);
        assert!((result.y - 0.5).abs() < 1e-10, "y={}", result.y);
    }

    #[test]
    fn test_oblique_mercator_roundtrip() {
        // Forward + inverse should recover the original point.
        let centers = [
            (37.5_f64, 127.0_f64),   // Seoul
            (40.7_f64, -74.0_f64),   // New York
            (0.0_f64, 0.0_f64),      // Equator/prime meridian
            (-33.9_f64, 18.4_f64),   // Cape Town
            (78.0_f64, 15.6_f64),    // Svalbard (high latitude)
        ];
        for (clat, clon) in &centers {
            let clat_r = clat.to_radians();
            let clon_r = clon.to_radians();
            // Test a point offset from center
            let lat_r = (clat + 5.0).to_radians();
            let lon_r = (clon + 5.0).to_radians();
            let merc = oblique_mercator(lat_r, lon_r, clat_r, clon_r);
            let (rlat, rlon) = oblique_mercator_inverse(merc, clat_r, clon_r);
            assert!(
                (lat_r - rlat).abs() < 1e-8,
                "lat roundtrip failed for center ({}, {}): {} vs {}",
                clat, clon, lat_r, rlat
            );
            assert!(
                (lon_r - rlon).abs() < 1e-8,
                "lon roundtrip failed for center ({}, {}): {} vs {}",
                clat, clon, lon_r, rlon
            );
        }
    }

    #[test]
    fn test_oblique_mercator_equator_center_matches_standard() {
        // When centered at (0,0), oblique Mercator should match standard Mercator.
        let lat = 48.8566_f64.to_radians(); // Paris
        let lon = 2.3522_f64.to_radians();
        let oblique = oblique_mercator(lat, lon, 0.0, 0.0);
        let standard = geo_to_mercator(&GeoCoord::new(48.8566, 2.3522));
        assert!(
            (oblique.x - standard.x).abs() < 1e-8,
            "x: {} vs {}", oblique.x, standard.x
        );
        assert!(
            (oblique.y - standard.y).abs() < 1e-8,
            "y: {} vs {}", oblique.y, standard.y
        );
    }
}
