//! 3D Tiles pipeline utilities.
//!
//! Pure functions for converting between the map engine's viewport
//! coordinate system and the ECEF-based coordinate system used by
//! 3D Tiles traversal and rendering.
//!
//! Follows the Karpathy principle: no GPU, no I/O, independently testable.

use glam::{DMat4, DVec3, DVec4, Mat4};
use x_planets_math::ecef::geodetic_to_ecef;
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
    let altitude = zoom_to_altitude_for(viewport.zoom, viewport.body.circumference);

    // Displace camera position backward for pitch, matching the raster
    // renderer's orbit-style camera model.  At pitch=0 the camera is
    // directly above the center; at pitch>0 it moves backward and lower.
    let pitch_rad = viewport.pitch.to_radians();
    let bearing_rad = viewport.bearing.to_radians();

    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let sin_lon = lon_rad.sin();
    let cos_lon = lon_rad.cos();

    let up = DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
    let east = DVec3::new(-sin_lon, cos_lon, 0.0);
    let north = up.cross(east).normalize();

    // Camera is displaced backward (opposite to bearing direction) by pitch.
    let forward_horizontal = north * bearing_rad.cos() + east * bearing_rad.sin();
    let backward = -forward_horizontal * pitch_rad.sin() + up * pitch_rad.cos();
    let position_ecef = geodetic_to_ecef(lat_rad, lon_rad, 0.0, &viewport.body.ellipsoid)
        + backward.normalize() * altitude;

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

/// Build a model matrix from an RTC center, glTF local transform,
/// and tile hierarchy transform.
///
/// Composes: tile_transform × local_transform × translate(rtc)
///
/// When `local_transform` already contains an ECEF-scale translation
/// (magnitude > 10 km), it is treated as self-positioning: the tile
/// hierarchy transform is skipped to avoid double-counting the ECEF
/// position.  This is the case for Cesium CWT tiles where the glTF
/// node transform includes the full ECEF placement.
///
/// - `rtc_center`: CESIUM_RTC or B3DM feature table offset (tile-local space)
/// - `local_transform`: glTF node hierarchy transform (Y-up → ECEF conversion)
/// - `tile_transform`: 3D Tiles hierarchy transform (tile-local → ECEF)
pub fn build_model_matrix(
    rtc_center: Option<[f64; 3]>,
    local_transform: DMat4,
    tile_transform: DMat4,
) -> DMat4 {
    // If local_transform has an ECEF-scale translation, it already
    // positions the content in ECEF.  Composing with tile_transform
    // would double-count the positioning.
    let local_translation = local_transform.col(3).truncate();
    let effective_tile_transform = if local_translation.length() > 10_000.0 {
        DMat4::IDENTITY
    } else {
        tile_transform
    };

    let mut result = effective_tile_transform * local_transform;
    if let Some(rtc) = rtc_center {
        let rtc_translation = DMat4::from_translation(DVec3::new(rtc[0], rtc[1], rtc[2]));
        result = result * rtc_translation;
    }
    result
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
    let altitude = zoom_to_altitude_for(viewport.zoom, viewport.body.circumference);

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

    // Camera ECEF position: displaced backward for pitch (orbit-style).
    let backward = -forward_horizontal * sin_pitch + up * cos_pitch;
    let camera_ecef = geodetic_to_ecef(lat_rad, lon_rad, 0.0, &viewport.body.ellipsoid)
        + backward.normalize() * altitude;

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
        terrain: [viewport.max_zoom as f32, viewport.hillshade_strength as f32, 0.0, 0.0],
        sun_dir: [
            viewport.sun_direction[0] as f32,
            viewport.sun_direction[1] as f32,
            viewport.sun_direction[2] as f32,
            0.0,
        ],
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

    #[allow(dead_code)]
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
        let model = build_model_matrix(None, DMat4::IDENTITY, DMat4::IDENTITY);
        assert!((model - DMat4::IDENTITY).abs_diff_eq(DMat4::ZERO, 1e-10));
    }

    #[test]
    fn test_build_model_matrix_with_rtc() {
        let rtc = [100.0, 200.0, 300.0];
        let model = build_model_matrix(Some(rtc), DMat4::IDENTITY, DMat4::IDENTITY);

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

    /// End-to-end test: simulate the full 3D tiles transform chain for the
    /// NYC "3D Buildings" preset (zoom 15, pitch 0, bearing 0) and verify
    /// that clip-space positions are valid (not degenerate / NaN / converging).
    #[test]
    fn test_full_transform_chain_nyc_zoom15() {
        // NYC Statue of Liberty area — matches "3D Buildings" preset
        let mut vp = Viewport::new(1920, 1080);
        vp.center = GeoCoord::new(40.6892, -74.0445);
        vp.zoom = 15.0;
        vp.pitch = 0.0;
        vp.bearing = 0.0;

        // Step 1: Build VP uniforms (camera at origin in relative-world space)
        let (uniforms, camera_ecef) = build_tiles3d_uniforms(&vp);

        // Camera should be above NYC in ECEF
        assert!(camera_ecef.length() > 6_000_000.0, "camera should be near Earth surface");
        assert!(camera_ecef.length() < 7_000_000.0, "camera should be near Earth surface");

        // VP matrix should not contain NaN/Inf
        for v in &uniforms.view_proj {
            assert!(v.is_finite(), "VP contains non-finite: {v}");
        }

        // Step 2: Simulate a B3DM tile near camera position
        // Typical RTC center: near camera in ECEF (within ~1km)
        let rtc_center: [f64; 3] = [
            camera_ecef.x - 100.0, // 100m west
            camera_ecef.y + 50.0,  // 50m east
            camera_ecef.z - 200.0, // 200m below camera altitude
        ];

        // No tile transform (identity) — common for leaf B3DM tiles
        let tile_transform = DMat4::IDENTITY;

        // Step 3: Build model matrix
        let model_ecef = build_model_matrix(Some(rtc_center), DMat4::IDENTITY, tile_transform);

        // Translation should be the RTC center
        let t = model_ecef.col(3);
        assert!((t.x - rtc_center[0]).abs() < 1e-6);
        assert!((t.y - rtc_center[1]).abs() < 1e-6);
        assert!((t.z - rtc_center[2]).abs() < 1e-6);

        // Step 4: Convert to relative-world (f32)
        let model_rel = ecef_to_relative_world(model_ecef, camera_ecef);

        // Translation should be small (relative to camera)
        let rel_t = model_rel.col(3);
        assert!(
            rel_t.x.abs() < 1000.0 && rel_t.y.abs() < 1000.0 && rel_t.z.abs() < 1000.0,
            "relative translation too large: ({}, {}, {})",
            rel_t.x, rel_t.y, rel_t.z
        );

        // Step 5: Transform a vertex through the full pipeline
        // Simulate a building vertex: 10m above ground, relative to RTC center
        let vertex = glam::Vec4::new(0.0, 0.0, 10.0, 1.0);

        // model_rel * vertex → world position (relative to camera)
        let world_pos = model_rel * vertex;
        assert!(
            world_pos.w.abs() > 0.5,
            "world w should be ~1: {}",
            world_pos.w
        );

        // VP * world_pos → clip position
        let vp_mat = glam::Mat4::from_cols_array(&uniforms.view_proj);
        let clip_pos = vp_mat * world_pos;

        // clip_pos should be finite
        assert!(clip_pos.x.is_finite(), "clip x non-finite");
        assert!(clip_pos.y.is_finite(), "clip y non-finite");
        assert!(clip_pos.z.is_finite(), "clip z non-finite");
        assert!(clip_pos.w.is_finite(), "clip w non-finite");

        // w should be positive (vertex is in front of camera)
        assert!(
            clip_pos.w > 0.0,
            "clip w should be positive (in front of camera): {}",
            clip_pos.w
        );

        // NDC coordinates should be reasonable (within visible range roughly)
        let ndc_x = clip_pos.x / clip_pos.w;
        let ndc_y = clip_pos.y / clip_pos.w;
        let ndc_z = clip_pos.z / clip_pos.w;

        // Building 100m away at zoom 15 should be within NDC range
        assert!(
            ndc_x.abs() < 100.0 && ndc_y.abs() < 100.0,
            "NDC out of range: ({}, {})",
            ndc_x, ndc_y
        );
        assert!(
            ndc_z >= 0.0 && ndc_z <= 1.0,
            "NDC z out of [0,1] range: {}",
            ndc_z
        );

        // Step 6: Test with a non-identity tile transform that shifts along
        // the local surface (realistic: child tile offset from parent).
        // Use a small translation along the ENU east/north directions,
        // which keeps the tile near the camera (in front, not behind).
        let lat_rad = vp.center.lat.to_radians();
        let lon_rad = vp.center.lon.to_radians();
        let _cos_lat = lat_rad.cos();
        let sin_lon = lon_rad.sin();
        let cos_lon = lon_rad.cos();
        // ENU east direction at camera location (unit vector in ECEF)
        let east = DVec3::new(-sin_lon, cos_lon, 0.0);
        // Shift tile 50m east (still visible from camera overhead)
        let shift = east * 50.0;
        let tile_shift = DMat4::from_translation(shift);
        let model_with_transform = build_model_matrix(Some(rtc_center), DMat4::IDENTITY, tile_shift);
        let rel_transformed = ecef_to_relative_world(model_with_transform, camera_ecef);
        let world_pos2 = rel_transformed * vertex;
        let clip_pos2 = vp_mat * world_pos2;
        assert!(
            clip_pos2.w > 0.0,
            "clip w with surface-aligned transform should be positive: {}",
            clip_pos2.w
        );
    }

    /// Verify that multiple nearby tiles produce DISTINCT clip-space positions
    /// (not converging to a single point — the "radiating lines" bug).
    #[test]
    fn test_distinct_clip_positions_for_nearby_tiles() {
        let mut vp = Viewport::new(1920, 1080);
        vp.center = GeoCoord::new(40.6892, -74.0445);
        vp.zoom = 15.0;
        vp.pitch = 0.0;
        vp.bearing = 0.0;

        let (uniforms, camera_ecef) = build_tiles3d_uniforms(&vp);
        let vp_mat = glam::Mat4::from_cols_array(&uniforms.view_proj);

        // Create 4 tiles at different nearby RTC centers (100m apart)
        let offsets = [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [100.0, 100.0, 0.0],
        ];

        let mut clip_positions = Vec::new();
        for offset in &offsets {
            let rtc = [
                camera_ecef.x + offset[0],
                camera_ecef.y + offset[1],
                camera_ecef.z + offset[2] - 300.0, // on surface
            ];
            let model = build_model_matrix(Some(rtc), DMat4::IDENTITY, DMat4::IDENTITY);
            let rel = ecef_to_relative_world(model, camera_ecef);

            let vertex = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
            let world = rel * vertex;
            let clip = vp_mat * world;

            assert!(clip.w > 0.0, "tile behind camera");

            let ndc = glam::Vec2::new(clip.x / clip.w, clip.y / clip.w);
            clip_positions.push(ndc);
        }

        // All 4 tiles should produce DIFFERENT NDC positions
        for i in 0..clip_positions.len() {
            for j in (i + 1)..clip_positions.len() {
                let dist = (clip_positions[i] - clip_positions[j]).length();
                assert!(
                    dist > 1e-4,
                    "tiles {} and {} converge to same point (dist={}): {:?} vs {:?}",
                    i,
                    j,
                    dist,
                    clip_positions[i],
                    clip_positions[j]
                );
            }
        }
    }
}
