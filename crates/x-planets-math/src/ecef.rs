//! ECEF (Earth-Centered, Earth-Fixed) coordinate transforms.
//!
//! Supports WGS84 (Earth), IAU Moon, and IAU Mars ellipsoids.
//! All angles in radians unless noted otherwise.

use glam::DVec3;

use crate::GeoCoord;

// ═══════════════════════════════════════════════════════════════════
// Ellipsoid parameters
// ═══════════════════════════════════════════════════════════════════

/// Reference ellipsoid parameters.
#[derive(Debug, Clone, Copy)]
pub struct Ellipsoid {
    /// Semi-major axis (equatorial radius) in meters.
    pub a: f64,
    /// Semi-minor axis (polar radius) in meters.
    pub b: f64,
    /// First eccentricity squared: e² = (a² - b²) / a².
    pub e2: f64,
}

impl Ellipsoid {
    /// Create an ellipsoid from semi-major axis and flattening.
    pub const fn from_a_f(a: f64, f: f64) -> Self {
        let b = a * (1.0 - f);
        let e2 = 2.0 * f - f * f;
        Self { a, b, e2 }
    }

    /// Create a spherical body (no flattening).
    pub const fn sphere(radius: f64) -> Self {
        Self {
            a: radius,
            b: radius,
            e2: 0.0,
        }
    }

    /// Radius of curvature in the prime vertical at latitude `lat_rad`.
    fn prime_vertical_radius(&self, sin_lat: f64) -> f64 {
        self.a / (1.0 - self.e2 * sin_lat * sin_lat).sqrt()
    }
}

/// WGS84 ellipsoid (Earth).
pub const WGS84: Ellipsoid = Ellipsoid::from_a_f(6_378_137.0, 1.0 / 298.257223563);

/// IAU Moon reference sphere (no flattening).
pub const MOON: Ellipsoid = Ellipsoid::sphere(1_737_400.0);

/// IAU Mars reference ellipsoid.
pub const MARS: Ellipsoid = Ellipsoid::from_a_f(3_396_190.0, 1.0 / 169.8944472);

// ═══════════════════════════════════════════════════════════════════
// CelestialBody — planet/moon parameters
// ═══════════════════════════════════════════════════════════════════

/// Parameters for a celestial body (planet, moon, etc.).
///
/// Groups the reference ellipsoid with derived values needed by the
/// rendering pipeline (circumference, Mercator latitude limit).
#[derive(Debug, Clone, Copy)]
pub struct CelestialBody {
    pub name: &'static str,
    pub ellipsoid: Ellipsoid,
    /// Equatorial circumference in meters (2 * PI * a).
    pub circumference: f64,
    /// Web Mercator latitude limit in degrees (default: 85.0511).
    pub mercator_lat_limit: f64,
}

impl CelestialBody {
    /// Create a body from an ellipsoid. Circumference is derived automatically.
    pub const fn new(name: &'static str, ellipsoid: Ellipsoid, mercator_lat_limit: f64) -> Self {
        // 2 * PI * a  (const-compatible approximation)
        let circumference = 2.0 * std::f64::consts::PI * ellipsoid.a;
        Self {
            name,
            ellipsoid,
            circumference,
            mercator_lat_limit,
        }
    }
}

/// Earth (WGS84 ellipsoid).
pub const EARTH: CelestialBody = CelestialBody::new("Earth", WGS84, 85.0511);

/// Moon (IAU reference sphere).
pub const MOON_BODY: CelestialBody = CelestialBody::new("Moon", MOON, 85.0511);

/// Mars (IAU reference ellipsoid).
pub const MARS_BODY: CelestialBody = CelestialBody::new("Mars", MARS, 85.0511);

// ═══════════════════════════════════════════════════════════════════
// Coordinate transforms
// ═══════════════════════════════════════════════════════════════════

/// Convert geodetic coordinates (lat_rad, lon_rad, height_m) to ECEF (x, y, z) meters.
///
/// Latitude: -π/2..π/2 (south negative), Longitude: -π..π (west negative).
pub fn geodetic_to_ecef(lat_rad: f64, lon_rad: f64, height: f64, ellipsoid: &Ellipsoid) -> DVec3 {
    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let sin_lon = lon_rad.sin();
    let cos_lon = lon_rad.cos();

    let n = ellipsoid.prime_vertical_radius(sin_lat);

    DVec3::new(
        (n + height) * cos_lat * cos_lon,
        (n + height) * cos_lat * sin_lon,
        (n * (1.0 - ellipsoid.e2) + height) * sin_lat,
    )
}

/// Convert ECEF (x, y, z) meters to geodetic (lat_rad, lon_rad, height_m).
///
/// Uses iterative Bowring method (converges in 2–3 iterations for typical positions).
pub fn ecef_to_geodetic(ecef: DVec3, ellipsoid: &Ellipsoid) -> (f64, f64, f64) {
    let x = ecef.x;
    let y = ecef.y;
    let z = ecef.z;

    let lon = y.atan2(x);
    let p = (x * x + y * y).sqrt();

    // Handle pole case
    if p < 1e-10 {
        let lat = if z >= 0.0 {
            std::f64::consts::FRAC_PI_2
        } else {
            -std::f64::consts::FRAC_PI_2
        };
        let height = z.abs() - ellipsoid.b;
        return (lat, lon, height);
    }

    // Initial estimate using Bowring's method
    let mut lat = (z / p * (1.0 - ellipsoid.e2)).atan();

    // Iterate (typically converges in 2-3 iterations)
    for _ in 0..10 {
        let sin_lat = lat.sin();
        let n = ellipsoid.prime_vertical_radius(sin_lat);

        let new_lat = (z + ellipsoid.e2 * n * sin_lat).atan2(p);

        if (new_lat - lat).abs() < 1e-12 {
            lat = new_lat;
            break;
        }
        lat = new_lat;
    }

    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let n = ellipsoid.prime_vertical_radius(sin_lat);

    let height = if cos_lat.abs() > 1e-10 {
        p / cos_lat - n
    } else {
        z / sin_lat - n * (1.0 - ellipsoid.e2)
    };

    (lat, lon, height)
}

/// Convert a [`GeoCoord`] (degrees) + height to ECEF using WGS84.
pub fn geocoord_to_ecef(coord: &GeoCoord, height: f64) -> DVec3 {
    geodetic_to_ecef(
        coord.lat.to_radians(),
        coord.lon.to_radians(),
        height,
        &WGS84,
    )
}

/// Convert ECEF to [`GeoCoord`] (degrees) + height using WGS84.
pub fn ecef_to_geocoord(ecef: DVec3) -> (GeoCoord, f64) {
    let (lat_rad, lon_rad, height) = ecef_to_geodetic(ecef, &WGS84);
    (GeoCoord::from_radians(lat_rad, lon_rad), height)
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    const TOLERANCE: f64 = 0.01; // 1cm accuracy

    #[test]
    fn test_origin_to_ecef() {
        // (0, 0, 0) on WGS84 → (a, 0, 0)
        let ecef = geodetic_to_ecef(0.0, 0.0, 0.0, &WGS84);
        assert!((ecef.x - WGS84.a).abs() < TOLERANCE);
        assert!(ecef.y.abs() < TOLERANCE);
        assert!(ecef.z.abs() < TOLERANCE);
    }

    #[test]
    fn test_north_pole() {
        // (90°, 0°, 0m) → (0, 0, b)
        let ecef = geodetic_to_ecef(FRAC_PI_2, 0.0, 0.0, &WGS84);
        assert!(ecef.x.abs() < TOLERANCE);
        assert!(ecef.y.abs() < TOLERANCE);
        assert!((ecef.z - WGS84.b).abs() < TOLERANCE);
    }

    #[test]
    fn test_south_pole() {
        let ecef = geodetic_to_ecef(-FRAC_PI_2, 0.0, 0.0, &WGS84);
        assert!(ecef.x.abs() < TOLERANCE);
        assert!(ecef.y.abs() < TOLERANCE);
        assert!((ecef.z + WGS84.b).abs() < TOLERANCE);
    }

    #[test]
    fn test_lon_90_degrees() {
        // (0, π/2, 0) → (0, a, 0)
        let ecef = geodetic_to_ecef(0.0, FRAC_PI_2, 0.0, &WGS84);
        assert!(ecef.x.abs() < TOLERANCE);
        assert!((ecef.y - WGS84.a).abs() < TOLERANCE);
        assert!(ecef.z.abs() < TOLERANCE);
    }

    #[test]
    fn test_wgs84_roundtrip() {
        let test_points = [
            (0.0_f64, 0.0_f64, 0.0_f64),           // Origin
            (48.8566_f64.to_radians(), 2.3522_f64.to_radians(), 35.0),   // Paris
            (37.5665_f64.to_radians(), 126.978_f64.to_radians(), 38.0),  // Seoul
            (-33.8688_f64.to_radians(), 151.2093_f64.to_radians(), 58.0), // Sydney
            (FRAC_PI_2, 0.0, 100.0),                 // North pole + 100m
            (-FRAC_PI_2, PI, 0.0),                    // South pole
        ];

        for (lat, lon, h) in test_points {
            let ecef = geodetic_to_ecef(lat, lon, h, &WGS84);
            let (lat2, lon2, h2) = ecef_to_geodetic(ecef, &WGS84);

            assert!(
                (lat - lat2).abs() < 1e-10,
                "Lat mismatch: {lat} vs {lat2}"
            );
            assert!(
                (lon - lon2).abs() < 1e-10,
                "Lon mismatch: {lon} vs {lon2}"
            );
            assert!((h - h2).abs() < TOLERANCE, "Height mismatch: {h} vs {h2}");
        }
    }

    #[test]
    fn test_geocoord_roundtrip() {
        let coord = GeoCoord::new(37.5665, 126.978);
        let ecef = geocoord_to_ecef(&coord, 38.0);
        let (coord2, h2) = ecef_to_geocoord(ecef);

        assert!((coord.lat - coord2.lat).abs() < 1e-8);
        assert!((coord.lon - coord2.lon).abs() < 1e-8);
        assert!((38.0 - h2).abs() < TOLERANCE);
    }

    #[test]
    fn test_moon_sphere() {
        // Moon is spherical: (0, 0, 0) → (radius, 0, 0)
        let ecef = geodetic_to_ecef(0.0, 0.0, 0.0, &MOON);
        assert!((ecef.x - MOON.a).abs() < TOLERANCE);
        assert!(ecef.y.abs() < TOLERANCE);
        assert!(ecef.z.abs() < TOLERANCE);

        // Roundtrip on Moon
        let (lat, lon, h) = ecef_to_geodetic(ecef, &MOON);
        assert!(lat.abs() < 1e-10);
        assert!(lon.abs() < 1e-10);
        assert!(h.abs() < TOLERANCE);
    }

    #[test]
    fn test_mars_roundtrip() {
        let lat = 18.65_f64.to_radians(); // Olympus Mons area
        let lon = -133.8_f64.to_radians();
        let h = 21_229.0; // ~21km altitude

        let ecef = geodetic_to_ecef(lat, lon, h, &MARS);
        let (lat2, lon2, h2) = ecef_to_geodetic(ecef, &MARS);

        assert!((lat - lat2).abs() < 1e-10);
        assert!((lon - lon2).abs() < 1e-10);
        assert!((h - h2).abs() < 1.0); // 1m accuracy for Mars
    }

    #[test]
    fn test_with_altitude() {
        // Aircraft altitude: 10,000m over origin
        let ecef = geodetic_to_ecef(0.0, 0.0, 10_000.0, &WGS84);
        assert!((ecef.x - (WGS84.a + 10_000.0)).abs() < TOLERANCE);

        let (_, _, h) = ecef_to_geodetic(ecef, &WGS84);
        assert!((h - 10_000.0).abs() < TOLERANCE);
    }
}
