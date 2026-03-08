//! 3D Tiles pipeline utilities.
//!
//! Pure functions for converting between the map engine's viewport
//! coordinate system and the ECEF-based coordinate system used by
//! 3D Tiles traversal and rendering.
//!
//! Follows the Karpathy principle: no GPU, no I/O, independently testable.

use glam::{DMat4, DVec3, DVec4, Mat4};
use x_planets_math::ecef::{geodetic_to_ecef, WGS84};
use x_planets_tiles::tiles3d::traversal::TraversalCamera;

use crate::viewport::Viewport;

// ═══════════════════════════════════════════════════════════════════
// Viewport → TraversalCamera conversion
// ═══════════════════════════════════════════════════════════════════

/// Convert a map viewport to a 3D Tiles traversal camera.
///
/// Maps the 2D map camera (lat/lon/zoom/pitch/bearing) to an ECEF
/// camera position and view-projection matrix suitable for 3D Tiles
/// LOD traversal.
pub fn viewport_to_traversal_camera(viewport: &Viewport) -> TraversalCamera {
    let lat_rad = viewport.center.lat.to_radians();
    let lon_rad = viewport.center.lon.to_radians();

    // Camera altitude from zoom level (approximate).
    let altitude = zoom_to_altitude(viewport.zoom);

    // Camera position in ECEF.
    let position_ecef = geodetic_to_ecef(lat_rad, lon_rad, altitude, &WGS84);

    // Build view-projection matrix.
    let view_proj = build_ecef_view_proj(viewport, position_ecef, lat_rad, lon_rad, altitude);

    TraversalCamera {
        position_ecef,
        view_proj,
    }
}

/// Convert zoom level to approximate camera altitude in meters.
///
/// At zoom 0 the camera sees the whole Earth (~20,000 km altitude).
/// Each zoom level halves the visible area (doubles the resolution).
pub fn zoom_to_altitude(zoom: f64) -> f64 {
    zoom_to_altitude_for(zoom, x_planets_math::ecef::EARTH.circumference)
}

/// Like [`zoom_to_altitude`] but for an arbitrary body circumference.
pub fn zoom_to_altitude_for(zoom: f64, circumference: f64) -> f64 {
    // At zoom 0, roughly half-circumference altitude to see the whole body.
    // Each zoom level halves the distance.
    (circumference / 2.0) / 2.0_f64.powf(zoom)
}

/// Convert camera altitude in meters to approximate zoom level.
pub fn altitude_to_zoom(altitude: f64) -> f64 {
    altitude_to_zoom_for(altitude, x_planets_math::ecef::EARTH.circumference)
}

/// Like [`altitude_to_zoom`] but for an arbitrary body circumference.
pub fn altitude_to_zoom_for(altitude: f64, circumference: f64) -> f64 {
    ((circumference / 2.0) / altitude.max(1.0)).log2()
}

/// Build an ECEF view-projection matrix for 3D Tiles rendering.
///
/// Creates a look-at camera positioned at the given ECEF point,
/// looking toward the Earth's center with appropriate pitch and bearing.
fn build_ecef_view_proj(
    viewport: &Viewport,
    camera_ecef: DVec3,
    lat_rad: f64,
    lon_rad: f64,
    altitude: f64,
) -> DMat4 {
    // ── Local ENU (East-North-Up) basis at camera position ──
    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let sin_lon = lon_rad.sin();
    let cos_lon = lon_rad.cos();

    let up = DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
    let east = DVec3::new(-sin_lon, cos_lon, 0.0);
    let north = up.cross(east).normalize();

    // ── Apply bearing and pitch ──
    let bearing_rad = viewport.bearing.to_radians();
    let pitch_rad = viewport.pitch.to_radians();

    // Forward direction: starts as -up (looking down), rotated by pitch.
    // At pitch=0, looking straight down. At pitch=90, looking at horizon.
    let cos_pitch = pitch_rad.cos();
    let sin_pitch = pitch_rad.sin();

    // Bearing rotates the horizontal component of the forward direction.
    let cos_bearing = bearing_rad.cos();
    let sin_bearing = bearing_rad.sin();

    let forward_horizontal = north * cos_bearing + east * sin_bearing;
    let forward = -up * cos_pitch + forward_horizontal * sin_pitch;
    let forward = forward.normalize();

    // For top-down views (pitch ≈ 0), forward ≈ -up, so we can't use
    // forward.cross(up) to get the right vector. Use north as the
    // camera-up reference instead.
    let camera_up = if pitch_rad.abs() < 0.01 {
        // Looking straight down: use bearing-rotated north as "up".
        let rotated_north = north * cos_bearing + east * sin_bearing;
        rotated_north.normalize()
    } else {
        let right = forward.cross(up).normalize();
        right.cross(forward).normalize()
    };

    // ── View matrix (look-at) ──
    let view = DMat4::look_to_rh(camera_ecef, forward, camera_up);

    // ── Projection matrix ──
    let aspect = viewport.width as f64 / viewport.height.max(1) as f64;
    let fov_y = 60.0_f64.to_radians();
    let near = altitude * 0.01; // 1% of altitude
    let far = altitude * 100.0; // 100x altitude

    let proj = DMat4::perspective_rh(fov_y, aspect, near.max(0.1), far.max(1000.0));

    proj * view
}

// ═══════════════════════════════════════════════════════════════════
// ECEF → World space transform
// ═══════════════════════════════════════════════════════════════════

/// Convert an ECEF model matrix (f64) to a world-space Mat4 (f32)
/// relative to a reference ECEF point.
///
/// Since ECEF coordinates are very large (millions of meters),
/// we subtract a reference point to avoid floating-point precision
/// issues in the f32 GPU matrices.
pub fn ecef_to_relative_world(ecef_transform: DMat4, reference_ecef: DVec3) -> Mat4 {
    // Translate to be relative to reference.
    let translation = ecef_transform.col(3).truncate() - reference_ecef;

    let result = DMat4::from_cols(
        ecef_transform.col(0),
        ecef_transform.col(1),
        ecef_transform.col(2),
        DVec4::new(translation.x, translation.y, translation.z, 1.0),
    );

    // Convert to f32.
    Mat4::from_cols(
        result.col(0).as_vec4(),
        result.col(1).as_vec4(),
        result.col(2).as_vec4(),
        result.col(3).as_vec4(),
    )
}

/// Build a model matrix from an RTC center (relative-to-center)
/// and a tile transform.
///
/// Many 3D Tiles content uses RTC_CENTER to store positions relative
/// to a known ECEF point. This function combines RTC with the tile's
/// hierarchical transform.
pub fn build_model_matrix(
    rtc_center: Option<[f64; 3]>,
    tile_transform: DMat4,
) -> DMat4 {
    if let Some(rtc) = rtc_center {
        // RTC: translate by the center point, then apply tile transform.
        let rtc_translation = DMat4::from_translation(DVec3::new(rtc[0], rtc[1], rtc[2]));
        tile_transform * rtc_translation
    } else {
        tile_transform
    }
}

/// Get the vertical FOV used for 3D Tiles traversal.
pub fn traversal_fov_y() -> f64 {
    60.0_f64.to_radians()
}

// ═══════════════════════════════════════════════════════════════════
// ECEF rendering uniforms
// ═══════════════════════════════════════════════════════════════════

/// Build viewport uniforms for ECEF 3D Tiles rendering.
///
/// Uses a camera-at-origin approach: the view-projection matrix is
/// computed with the camera at (0,0,0) in "relative world space".
/// Model matrices must be offset by `camera_ecef` using
/// [`ecef_to_relative_world`] to match.
///
/// Returns `(ViewportUniforms, camera_ecef_position)`.
pub fn build_tiles3d_uniforms(viewport: &Viewport) -> (x_planets_math::ViewportUniforms, DVec3) {
    let lat_rad = viewport.center.lat.to_radians();
    let lon_rad = viewport.center.lon.to_radians();
    let altitude = zoom_to_altitude(viewport.zoom);

    // Camera ECEF position (returned for model matrix computation).
    let camera_ecef = geodetic_to_ecef(lat_rad, lon_rad, altitude, &WGS84);

    // ── Local ENU basis ──
    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let sin_lon = lon_rad.sin();
    let cos_lon = lon_rad.cos();

    let up = DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
    let east = DVec3::new(-sin_lon, cos_lon, 0.0);
    let north = up.cross(east).normalize();

    // ── Bearing + pitch ──
    let bearing_rad = viewport.bearing.to_radians();
    let pitch_rad = viewport.pitch.to_radians();
    let cos_pitch = pitch_rad.cos();
    let sin_pitch = pitch_rad.sin();
    let cos_bearing = bearing_rad.cos();
    let sin_bearing = bearing_rad.sin();

    let forward_horizontal = north * cos_bearing + east * sin_bearing;
    let forward = (-up * cos_pitch + forward_horizontal * sin_pitch).normalize();

    let camera_up = if pitch_rad.abs() < 0.01 {
        (north * cos_bearing + east * sin_bearing).normalize()
    } else {
        let right = forward.cross(up).normalize();
        right.cross(forward).normalize()
    };

    // ── View matrix with camera at ORIGIN (relative space) ──
    let view = DMat4::look_to_rh(DVec3::ZERO, forward, camera_up);

    // ── Projection ──
    let aspect = viewport.width as f64 / viewport.height.max(1) as f64;
    let fov_y = 60.0_f64.to_radians();
    let near = altitude * 0.01;
    let far = altitude * 100.0;
    let proj = DMat4::perspective_rh(fov_y, aspect, near.max(0.1), far.max(1000.0));

    let view_proj = proj * view;

    // Convert to f32.
    let vp_f32 = Mat4::from_cols(
        view_proj.col(0).as_vec4(),
        view_proj.col(1).as_vec4(),
        view_proj.col(2).as_vec4(),
        view_proj.col(3).as_vec4(),
    );

    let uniforms = x_planets_math::ViewportUniforms {
        view_proj: vp_f32.to_cols_array(),
        resolution: [
            viewport.width as f32,
            viewport.height as f32,
            1.0 / viewport.width as f32,
            1.0 / viewport.height as f32,
        ],
        camera: [0.0, 0.0, viewport.zoom as f32, 0.0],
        clip_sphere: [0.0, 0.0, 1.0, -1.0], // no clipping (cos(-1) = all pass)
    };

    (uniforms, camera_ecef)
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use x_planets_math::GeoCoord;

    fn make_viewport() -> Viewport {
        let mut vp = Viewport::new(1920, 1080);
        vp.center = GeoCoord::new(37.5665, 126.978);
        vp.zoom = 15.0;
        vp.pitch = 0.0;
        vp.bearing = 0.0;
        vp
    }

    #[test]
    fn test_zoom_to_altitude() {
        let half_circ = x_planets_math::ecef::EARTH.circumference / 2.0;

        // Zoom 0 → ~half circumference
        assert!((zoom_to_altitude(0.0) - half_circ).abs() < 1.0);

        // Zoom 1 → ~half of that
        assert!((zoom_to_altitude(1.0) - half_circ / 2.0).abs() < 1.0);

        // Higher zoom → lower altitude
        assert!(zoom_to_altitude(10.0) < zoom_to_altitude(5.0));
        assert!(zoom_to_altitude(15.0) < zoom_to_altitude(10.0));
    }

    #[test]
    fn test_altitude_to_zoom_roundtrip() {
        for zoom in [0.0, 5.0, 10.0, 15.0, 20.0] {
            let alt = zoom_to_altitude(zoom);
            let zoom2 = altitude_to_zoom(alt);
            assert!(
                (zoom - zoom2).abs() < 1e-10,
                "Roundtrip failed: {zoom} → {alt} → {zoom2}"
            );
        }
    }

    #[test]
    fn test_viewport_to_traversal_camera() {
        // Use a moderate zoom level to avoid precision issues.
        let mut vp = Viewport::new(1920, 1080);
        vp.center = GeoCoord::new(37.5665, 126.978);
        vp.zoom = 5.0;
        vp.pitch = 0.0;
        vp.bearing = 0.0;

        let cam = viewport_to_traversal_camera(&vp);

        // Camera should be above Seoul, roughly at ECEF.
        assert!(cam.position_ecef.length() > 6_000_000.0);

        // View-proj matrix should not contain NaN.
        let cols = [
            cam.view_proj.col(0),
            cam.view_proj.col(1),
            cam.view_proj.col(2),
            cam.view_proj.col(3),
        ];
        for col in &cols {
            assert!(!col.x.is_nan(), "View-proj contains NaN");
            assert!(!col.y.is_nan(), "View-proj contains NaN");
            assert!(!col.z.is_nan(), "View-proj contains NaN");
            assert!(!col.w.is_nan(), "View-proj contains NaN");
        }
    }

    #[test]
    fn test_ecef_to_relative_world() {
        let reference = DVec3::new(6_378_137.0, 0.0, 0.0);
        let transform = DMat4::from_translation(DVec3::new(6_378_237.0, 100.0, 50.0));

        let relative = ecef_to_relative_world(transform, reference);

        // Translation should be (100, 100, 50) relative to reference.
        let t = relative.col(3);
        assert!((t.x - 100.0).abs() < 1e-3);
        assert!((t.y - 100.0).abs() < 1e-3);
        assert!((t.z - 50.0).abs() < 1e-3);
    }

    #[test]
    fn test_build_model_matrix_no_rtc() {
        let transform = DMat4::IDENTITY;
        let model = build_model_matrix(None, transform);
        assert!((model - DMat4::IDENTITY).abs_diff_eq(DMat4::ZERO, 1e-10));
    }

    #[test]
    fn test_build_model_matrix_with_rtc() {
        let rtc = [100.0, 200.0, 300.0];
        let model = build_model_matrix(Some(rtc), DMat4::IDENTITY);

        // Translation column should be the RTC center.
        let t = model.col(3);
        assert!((t.x - 100.0).abs() < 1e-10);
        assert!((t.y - 200.0).abs() < 1e-10);
        assert!((t.z - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_traversal_fov_y() {
        let fov = traversal_fov_y();
        assert!((fov - 60.0_f64.to_radians()).abs() < 1e-10);
    }
}
