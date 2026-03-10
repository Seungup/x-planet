//! Viewport and camera controller for map navigation.

mod camera;
mod frustum;
mod tile_selection;
mod view_proj;

pub use camera::CameraController;

use std::cell::Cell;

use x_planets_math::ecef::{CelestialBody, EARTH};
use x_planets_math::GeoCoord;

/// LOD mode for quadtree tile selection.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TileLodMode {
    /// Standard flat Mercator: pitch-based perspective LOD, Mercator-distance
    /// priority, seeds from min_z.
    Flat,
    /// Globe: aggressive angular LOD with foreshortening, angular priority,
    /// seeds from z=0.  Suited for the 3D globe where edge tiles are
    /// physically foreshortened.
    Globe,
    /// Centered Mercator: angular priority and z=0 seeding (like Globe) but
    /// **no distance-based zoom reduction**.  Centered Mercator projects the
    /// full spherical cap onto a flat plane, so edge tiles are NOT
    /// foreshortened — they need full-resolution zoom just like center tiles.
    /// Only pitch-based LOD is applied (if any).
    Centered,
}

/// Orbital camera altitude on a unit sphere for globe mode.
///
/// At zoom 0 the camera is ~3.14 radii above the surface (sees whole globe).
/// Each zoom level halves the altitude.
fn globe_unit_altitude(zoom: f64, body: &CelestialBody) -> f64 {
    ((body.circumference / 2.0) / body.ellipsoid.a) / 2.0_f64.powf(zoom)
}

/// Effective visible half-angle for globe tile selection and interaction.
///
/// At low zoom the sphere's **horizon** limits visibility (cap formula).
/// At high zoom the camera is close to the surface and the surface appears
/// flat, so the **camera FOV** limits visibility instead.  Taking the
/// minimum gives the correct visible extent at every zoom level.
fn globe_visible_half_angle(unit_altitude: f64) -> f64 {
    let cap_half = (1.0 / (unit_altitude + 1.0)).acos();
    let fov_half = unit_altitude * (std::f64::consts::FRAC_PI_3 * 0.5).tan();
    cap_half.min(fov_half)
}

/// The viewport represents the visible area of the map.
#[derive(Debug, Clone)]
pub struct Viewport {
    /// Screen width in pixels.
    pub width: u32,
    /// Screen height in pixels.
    pub height: u32,
    /// Geographic center of the viewport.
    pub center: GeoCoord,
    /// Zoom level (fractional, e.g. 5.5).
    pub zoom: f64,
    /// Camera pitch in degrees (0 = top-down, 60 = strongly tilted).
    pub pitch: f64,
    /// Camera bearing in degrees, clockwise from north (0 = north up).
    pub bearing: f64,
    /// Maximum number of tiles rendered per frame.
    pub tile_budget: usize,
    /// Celestial body parameters (affects globe camera altitude, terrain scale, etc.).
    pub body: CelestialBody,
    /// Frustum safety margin (fraction of extent). Larger values include more
    /// off-screen tiles which helps when terrain displacement shifts geometry
    /// into the viewport. Default 0.05 (5%). Terrain mode uses 0.15 (15%).
    pub frustum_margin: f64,
    /// Maximum zoom level (for depth-bias calculation in shaders). Default: 22.0.
    pub max_zoom: f64,
    /// Sun direction for hillshade lighting (normalized xyz). Default: [-0.5, -0.5, 0.7].
    pub sun_direction: [f64; 3],
    /// Hillshade strength (0.0 = flat, 1.0 = full relief). Default: 1.0.
    pub hillshade_strength: f64,
    /// Previous tile zoom level for hysteresis (prevents oscillation at zoom boundaries).
    prev_tile_zoom: Cell<Option<u8>>,
}

impl Viewport {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            center: GeoCoord::default(),
            zoom: 2.0,
            pitch: 0.0,
            bearing: 0.0,
            tile_budget: 150,
            body: EARTH,
            frustum_margin: 0.05,
            max_zoom: 22.0,
            sun_direction: [-0.5, -0.5, 0.7],
            hillshade_strength: 1.0,
            prev_tile_zoom: Cell::new(None),
        }
    }

    /// Integer zoom level for tile fetching.
    ///
    /// Uses hysteresis (±0.4 threshold) to prevent tile oscillation at zoom
    /// boundaries.  The previous tile zoom is sticky: it only changes when the
    /// fractional zoom moves more than 0.4 away from the current integer level.
    pub fn tile_zoom(&self) -> u8 {
        let raw = self.zoom.round().clamp(0.0, 22.0) as u8;
        let result = if let Some(prev) = self.prev_tile_zoom.get() {
            if (self.zoom - prev as f64).abs() < 0.4 {
                prev
            } else {
                raw
            }
        } else {
            raw
        };
        self.prev_tile_zoom.set(Some(result));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use x_planets_math::TileCoord;

    #[test]
    fn test_viewport_visible_tiles() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 2.0;

        let tiles = viewport.visible_tiles();
        assert!(!tiles.is_empty());
    }

    #[test]
    fn test_visible_bounds_covers_screen() {
        let viewport = Viewport::new(800, 600);
        let bounds = viewport.visible_bounds();
        let sw = bounds.south_west;
        let ne = bounds.north_east;
        assert!(ne.lon > sw.lon || ne.lat != sw.lat);
    }

    #[test]
    fn test_camera_zoom_clamp() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.zoom = 20.0;

        ctrl.zoom(&mut viewport, 10.0);
        assert!(viewport.zoom <= 22.0);

        ctrl.zoom(&mut viewport, -100.0);
        assert!(viewport.zoom >= 0.0);
    }

    #[test]
    fn test_bearing_wrap() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);

        ctrl.set_bearing(&mut viewport, 370.0);
        assert!((viewport.bearing - 10.0).abs() < 1e-9);

        ctrl.set_bearing(&mut viewport, -90.0);
        assert!((viewport.bearing - 270.0).abs() < 1e-9);
    }

    #[test]
    fn test_viewport_uniforms() {
        let viewport = Viewport::new(1920, 1080);
        let uniforms = viewport.to_uniforms();
        assert_eq!(uniforms.resolution[0], 1920.0);
        assert_eq!(uniforms.resolution[1], 1080.0);
    }

    #[test]
    fn test_lod_multiple_zoom_levels_at_high_pitch() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 6.0;
        viewport.pitch = 55.0;

        let tiles = viewport.visible_tiles();
        assert!(!tiles.is_empty());

        let zoom_levels: std::collections::HashSet<u8> =
            tiles.iter().map(|t| t.coord.z).collect();
        assert!(
            zoom_levels.len() >= 2,
            "Expected multi-level LOD, got levels: {:?}",
            zoom_levels
        );
    }

    #[test]
    fn test_lod_sorted_coarse_first() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 5.0;
        viewport.pitch = 45.0;

        let tiles = viewport.visible_tiles();
        for pair in tiles.windows(2) {
            assert!(pair[0].coord.z <= pair[1].coord.z, "Tiles not sorted by zoom level");
        }
    }

    #[test]
    fn test_no_lod_at_zero_pitch() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 5.0;
        viewport.pitch = 0.0;

        let tiles = viewport.visible_tiles();
        let zoom_levels: std::collections::HashSet<u8> =
            tiles.iter().map(|t| t.coord.z).collect();
        assert_eq!(zoom_levels.len(), 1, "pitch=0 should use single zoom level");
        assert!(zoom_levels.contains(&5));
    }

    #[test]
    fn test_perspective_bounds_covers_far_tiles() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 6.0;
        viewport.pitch = 55.0;

        let bounds = viewport.visible_bounds();
        assert!(
            bounds.north_east.lat > 20.0,
            "Pitched bounds should extend far north, got NE lat {:.1}",
            bounds.north_east.lat
        );
    }

    #[test]
    fn test_lod_gap_free() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 6.0;
        viewport.pitch = 55.0;

        let base_z = viewport.tile_zoom();
        let tiles = viewport.visible_tiles();
        let result_set: std::collections::HashSet<TileCoord> =
            tiles.iter().map(|vt| vt.coord).collect();

        let frustum = viewport.frustum();
        for base_vt in frustum.visible_tiles(base_z) {
            let mut cur = base_vt.coord;
            let mut found = false;
            loop {
                if result_set.contains(&cur) {
                    found = true;
                    break;
                }
                match cur.parent() {
                    Some(p) => cur = p,
                    None => break,
                }
            }
            assert!(
                found,
                "Base tile {:?} has no covering tile in LOD result",
                base_vt.coord
            );
        }
    }

    #[test]
    fn test_quadtree_budget_respected() {
        let mut viewport = Viewport::new(1920, 1080);
        viewport.center = GeoCoord::new(37.5, 127.0);
        viewport.zoom = 15.0;
        viewport.pitch = 60.0;

        let tiles = viewport.visible_tiles();
        assert!(
            tiles.len() <= 150,
            "Too many tiles: {} (budget hard cap is 150)",
            tiles.len()
        );
    }

    #[test]
    fn test_quadtree_near_tiles_prioritized() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5, 127.0);
        viewport.zoom = 8.0;
        viewport.pitch = 55.0;

        let base_z = viewport.tile_zoom();
        let tiles = viewport.visible_tiles();
        let fine_tiles: Vec<_> = tiles.iter().filter(|t| t.coord.z == base_z).collect();
        assert!(
            !fine_tiles.is_empty(),
            "Should have at least some tiles at the base zoom level {}",
            base_z
        );
    }

    #[test]
    fn test_quadtree_lod_with_bearing() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 6.0;
        viewport.pitch = 45.0;
        viewport.bearing = 45.0;

        let tiles = viewport.visible_tiles();
        assert!(!tiles.is_empty());

        for pair in tiles.windows(2) {
            assert!(pair[0].coord.z <= pair[1].coord.z);
        }
    }

    #[test]
    fn test_frustum_polygon_active_when_pitched() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 5.0;
        viewport.pitch = 45.0;

        let frustum = viewport.frustum();
        assert!(
            frustum.polygon.is_some(),
            "Pitched view should have a convex polygon for precise culling"
        );
    }

    #[test]
    fn test_tile_selection_near_north_pole() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(80.0, 0.0);
        viewport.zoom = 3.0;

        let tiles = viewport.visible_tiles();
        assert!(!tiles.is_empty(), "Should select tiles near the north pole");

        let has_y0 = tiles.iter().any(|t| t.coord.y == 0);
        assert!(has_y0, "Should include northernmost tiles (y=0) at lat=80°");
    }

    #[test]
    fn test_tile_selection_near_south_pole() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(-80.0, 0.0);
        viewport.zoom = 3.0;

        let tiles = viewport.visible_tiles();
        assert!(!tiles.is_empty(), "Should select tiles near the south pole");

        let n = 1u32 << 3;
        let has_max_y = tiles.iter().any(|t| t.coord.y == n - 1);
        assert!(
            has_max_y,
            "Should include southernmost tiles (y={}) at lat=-80°",
            n - 1
        );
    }

    #[test]
    fn test_tile_selection_at_mercator_boundary() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(85.0, 0.0);
        viewport.zoom = 2.0;

        let tiles = viewport.visible_tiles();
        assert!(!tiles.is_empty(), "Should select tiles at Mercator boundary");
    }

    #[test]
    fn test_tile_selection_high_lat_high_pitch() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(70.0, 0.0);
        viewport.zoom = 5.0;
        viewport.pitch = 55.0;

        let tiles = viewport.visible_tiles();
        assert!(!tiles.is_empty());

        let center_tile = x_planets_math::TileCoord::from_geo(&viewport.center, viewport.tile_zoom());
        let northernmost = tiles.iter().map(|t| t.coord.y).min().unwrap();
        assert!(
            northernmost <= center_tile.y,
            "Pitched view should include tiles north of center"
        );
    }

    #[test]
    fn test_globe_vp_north_at_top() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 2.0;

        let vp = viewport.to_globe_view_proj_f64();

        let north_point = x_planets_math::geo_to_unit_sphere(
            10.0_f64.to_radians(),
            0.0_f64.to_radians(),
        );
        let clip = vp * glam::DVec4::new(north_point.x, north_point.y, north_point.z, 1.0);
        let ndc_y = clip.y / clip.w;

        assert!(
            ndc_y > 0.0,
            "North (lat=10°) should map to positive clip Y (top), got ndc_y={:.4}",
            ndc_y
        );

        let south_point = x_planets_math::geo_to_unit_sphere(
            (-10.0_f64).to_radians(),
            0.0_f64.to_radians(),
        );
        let clip_s = vp * glam::DVec4::new(south_point.x, south_point.y, south_point.z, 1.0);
        let ndc_y_s = clip_s.y / clip_s.w;

        assert!(
            ndc_y_s < 0.0,
            "South (lat=-10°) should map to negative clip Y (bottom), got ndc_y={:.4}",
            ndc_y_s
        );
    }

    #[test]
    fn test_globe_vp_east_at_right() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 2.0;

        let vp = viewport.to_globe_view_proj_f64();

        let east_point = x_planets_math::geo_to_unit_sphere(
            0.0_f64.to_radians(),
            10.0_f64.to_radians(),
        );
        let clip = vp * glam::DVec4::new(east_point.x, east_point.y, east_point.z, 1.0);
        let ndc_x = clip.x / clip.w;

        assert!(
            ndc_x > 0.0,
            "East (lon=10°) should map to positive clip X (right), got ndc_x={:.4}",
            ndc_x
        );
    }

    #[test]
    fn test_globe_vp_with_bearing() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 2.0;
        viewport.bearing = 90.0;

        let vp = viewport.to_globe_view_proj_f64();

        let north_point = x_planets_math::geo_to_unit_sphere(
            10.0_f64.to_radians(),
            0.0_f64.to_radians(),
        );
        let clip = vp * glam::DVec4::new(north_point.x, north_point.y, north_point.z, 1.0);
        let ndc_x = clip.x / clip.w;

        assert!(
            ndc_x < 0.0,
            "At bearing=90° (facing east), north should map to left (-X), got ndc_x={:.4}",
            ndc_x
        );

        let east_point = x_planets_math::geo_to_unit_sphere(
            0.0_f64.to_radians(),
            10.0_f64.to_radians(),
        );
        let clip_e = vp * glam::DVec4::new(east_point.x, east_point.y, east_point.z, 1.0);
        let ndc_y_e = clip_e.y / clip_e.w;

        assert!(
            ndc_y_e > 0.0,
            "At bearing=90°, east should map to top (+Y), got ndc_y={:.4}",
            ndc_y_e
        );
    }

    #[test]
    fn test_globe_pan_up_moves_south() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 5.0;

        let lat_before = viewport.center.lat;
        ctrl.pan_globe(&mut viewport, 0.0, 50.0);
        let lat_after = viewport.center.lat;

        assert!(
            lat_after < lat_before,
            "Drag up (dy>0) should move center south: before={:.4}, after={:.4}",
            lat_before, lat_after
        );
    }

    #[test]
    fn test_globe_pan_right_moves_west() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 5.0;

        let lon_before = viewport.center.lon;
        ctrl.pan_globe(&mut viewport, 50.0, 0.0);
        let lon_after = viewport.center.lon;

        assert!(
            lon_after < lon_before,
            "Drag right should move center west: before={:.4}, after={:.4}",
            lon_before, lon_after
        );
    }

    #[test]
    fn test_globe_pan_matches_mercator_direction() {
        let ctrl = CameraController::new();

        let mut vp_merc = Viewport::new(800, 600);
        vp_merc.center = GeoCoord::new(30.0, 50.0);
        vp_merc.zoom = 5.0;
        let lat_before_merc = vp_merc.center.lat;
        ctrl.pan(&mut vp_merc, 0.0, 50.0);
        let merc_dlat = vp_merc.center.lat - lat_before_merc;

        let mut vp_globe = Viewport::new(800, 600);
        vp_globe.center = GeoCoord::new(30.0, 50.0);
        vp_globe.zoom = 5.0;
        let lat_before_globe = vp_globe.center.lat;
        ctrl.pan_globe(&mut vp_globe, 0.0, 50.0);
        let globe_dlat = vp_globe.center.lat - lat_before_globe;

        assert!(
            merc_dlat.signum() == globe_dlat.signum(),
            "Pan direction mismatch: mercator dlat={:.6}, globe dlat={:.6}",
            merc_dlat, globe_dlat
        );
    }

    #[test]
    fn test_globe_pan_with_bearing() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 5.0;
        viewport.bearing = 90.0;

        let lat_before = viewport.center.lat;
        ctrl.pan_globe(&mut viewport, 50.0, 0.0);
        let lat_after = viewport.center.lat;

        assert!(
            lat_after > lat_before,
            "At bearing=90°, drag right should move center north: before={:.4}, after={:.4}",
            lat_before, lat_after
        );
    }

    #[test]
    fn test_globe_zoom_applies_full_delta() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 5.0;

        let zoom_before = viewport.zoom;
        ctrl.zoom_at_globe(&mut viewport, 1.0, 400.0, 300.0);
        let zoom_change = viewport.zoom - zoom_before;

        assert!(
            (zoom_change - 1.0).abs() < 0.001,
            "zoom_at_globe should apply full delta: expected 1.0, got {:.4}",
            zoom_change
        );
    }

    #[test]
    fn test_globe_zoom_at_pointer_center_stable() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5, 127.0);
        viewport.zoom = 5.0;

        let lat_before = viewport.center.lat;
        let lon_before = viewport.center.lon;
        ctrl.zoom_at_globe(&mut viewport, 1.0, 400.0, 300.0);

        assert!(
            (viewport.center.lat - lat_before).abs() < 0.01,
            "Zoom at center should not shift latitude: before={:.4}, after={:.4}",
            lat_before, viewport.center.lat
        );
        assert!(
            (viewport.center.lon - lon_before).abs() < 0.01,
            "Zoom at center should not shift longitude: before={:.4}, after={:.4}",
            lon_before, viewport.center.lon
        );
    }

    #[test]
    fn test_globe_zoom_at_pointer_offset_shifts_center() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 3.0;

        ctrl.zoom_at_globe(&mut viewport, 2.0, 700.0, 300.0);

        assert!(
            viewport.center.lon > 0.0,
            "Zoom at right edge should shift center east, got lon={:.4}",
            viewport.center.lon
        );
    }

    #[test]
    fn test_globe_zoom_full_delta_at_high_zoom() {
        let ctrl = CameraController::new();

        let mut vp = Viewport::new(800, 600);
        vp.center = GeoCoord::new(0.0, 0.0);
        vp.zoom = 18.0;
        let z_before = vp.zoom;
        ctrl.zoom_at_globe(&mut vp, 0.3, 400.0, 300.0);
        let dz = (vp.zoom - z_before).abs();

        assert!(
            (dz - 0.3).abs() < 0.001,
            "zoom_at_globe at high zoom should apply full delta: expected 0.3, got {:.4}",
            dz
        );
    }

    #[test]
    fn test_globe_zoom_consistent_delta_all_levels() {
        let ctrl = CameraController::new();
        let delta = 0.3;

        for z_int in 0..=20 {
            let z = z_int as f64;
            let mut vp = Viewport::new(800, 600);
            vp.center = GeoCoord::new(0.0, 0.0);
            vp.zoom = z;
            let z_before = vp.zoom;
            ctrl.zoom_at_globe(&mut vp, delta, 400.0, 300.0);
            let dz = (vp.zoom - z_before).abs();

            if z + delta <= ctrl.max_zoom {
                assert!(
                    (dz - delta).abs() < 0.001,
                    "At z={}: expected dz={:.3}, got {:.6}",
                    z, delta, dz
                );
            }
        }
    }

    #[test]
    fn test_globe_zoom_round_trip_preserves_center() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(35.0, 120.0);
        viewport.zoom = 10.0;

        let lat_orig = viewport.center.lat;
        let lon_orig = viewport.center.lon;

        ctrl.zoom_at_globe(&mut viewport, 3.0, 400.0, 300.0);
        ctrl.zoom_at_globe(&mut viewport, -3.0, 400.0, 300.0);

        assert!(
            (viewport.center.lat - lat_orig).abs() < 0.1,
            "Round-trip lat: orig={:.4}, after={:.4}",
            lat_orig, viewport.center.lat
        );
        assert!(
            (viewport.center.lon - lon_orig).abs() < 0.1,
            "Round-trip lon: orig={:.4}, after={:.4}",
            lon_orig, viewport.center.lon
        );
    }

    #[test]
    fn test_visible_tiles_for_mode_mercator_centered() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5665, 126.978);
        viewport.zoom = 5.0;

        let centered_tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);

        assert!(
            !centered_tiles.is_empty(),
            "centered should produce tiles",
        );
        assert!(
            centered_tiles.len() >= 4,
            "centered ({}) should have at least 4 tiles",
            centered_tiles.len(),
        );
    }

    #[test]
    fn test_visible_tiles_globe_returns_tiles() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5665, 126.978);
        viewport.zoom = 3.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
        assert!(
            !tiles.is_empty(),
            "Globe mode should return visible tiles"
        );
    }

    #[test]
    fn test_visible_tiles_globe_center_tile_included() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5665, 126.978);
        viewport.zoom = 5.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);

        let has_center = tiles.iter().any(|vt| {
            let tc = TileCoord::from_geo(&viewport.center, vt.coord.z);
            tc == vt.coord
        });
        assert!(
            has_center,
            "No tile covering center ({:.4}, {:.4}) found in globe visible tiles ({} tiles)",
            viewport.center.lat, viewport.center.lon, tiles.len()
        );
    }

    #[test]
    fn test_visible_tiles_globe_sorted_coarse_first() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 4.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
        for pair in tiles.windows(2) {
            assert!(
                pair[0].coord.z <= pair[1].coord.z,
                "Globe tiles not sorted coarse-first"
            );
        }
    }

    #[test]
    fn test_globe_lod_varies_by_distance() {
        for zoom in [1.0, 2.0, 3.0] {
            let mut viewport = Viewport::new(800, 600);
            viewport.center = GeoCoord::new(0.0, 0.0);
            viewport.zoom = zoom;

            let tiles =
                viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);

            let zoom_levels: std::collections::HashSet<u8> =
                tiles.iter().map(|t| t.coord.z).collect();
            assert!(
                zoom_levels.len() > 1,
                "Globe mode at zoom={zoom} should produce multiple zoom levels (got {:?})",
                zoom_levels,
            );
        }
    }

    #[test]
    fn test_visible_tiles_globe_low_zoom_coverage() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 1.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
        assert!(
            !tiles.is_empty(),
            "Globe mode at zoom 1 should produce tiles"
        );
    }

    #[test]
    fn test_visible_tiles_globe_polar_center() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(80.0, 0.0);
        viewport.zoom = 4.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
        assert!(
            !tiles.is_empty(),
            "Globe mode near pole should produce tiles"
        );
        for vt in &tiles {
            let max_y = vt.coord.extent();
            assert!(vt.coord.y < max_y, "Tile y out of range: {:?}", vt.coord);
        }
    }

    #[test]
    fn test_visible_tiles_globe_budget_respected() {
        let mut viewport = Viewport::new(1920, 1080);
        viewport.center = GeoCoord::new(37.5, 127.0);
        viewport.zoom = 8.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
        assert!(
            tiles.len() <= 150,
            "Globe mode: too many tiles {} (budget hard cap is 150)",
            tiles.len()
        );
    }

    #[test]
    fn test_centered_mercator_polar_selects_multiple_longitudes() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(80.0, 0.0);
        viewport.zoom = 2.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);
        let unique_x: std::collections::HashSet<u32> =
            tiles.iter().map(|vt| vt.coord.x).collect();
        assert!(
            unique_x.len() >= 3,
            "Polar centered Mercator should cover multiple longitudes, got {:?}",
            unique_x
        );
    }

    #[test]
    fn test_centered_mercator_polar_selects_sufficient_tiles() {
        let mut viewport = Viewport::new(800, 600);
        viewport.zoom = 3.0;

        viewport.center = GeoCoord::new(0.0, 0.0);
        let equator_tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);

        viewport.center = GeoCoord::new(80.0, 0.0);
        let polar_tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);

        assert!(
            equator_tiles.len() >= 10,
            "Equator should select sufficient tiles, got {}",
            equator_tiles.len(),
        );
        assert!(
            polar_tiles.len() >= 10,
            "Polar should select sufficient tiles, got {}",
            polar_tiles.len(),
        );
    }

    #[test]
    fn test_centered_mercator_south_pole_selects_tiles() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(-80.0, 120.0);
        viewport.zoom = 2.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);
        let unique_x: std::collections::HashSet<u32> =
            tiles.iter().map(|vt| vt.coord.x).collect();
        assert!(
            unique_x.len() >= 3,
            "South-polar centered Mercator should cover multiple longitudes, got {:?}",
            unique_x
        );
    }

    #[test]
    fn test_globe_zoom0_selects_hemisphere() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 0.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
        assert!(
            tiles.len() >= 1,
            "Globe zoom 0 should select at least the root tile"
        );
    }

    #[test]
    fn test_globe_polar_center_sufficient_tiles() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(80.0, 0.0);
        viewport.zoom = 3.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
        assert!(
            tiles.len() >= 8,
            "Globe polar zoom 3 should have >=8 tiles, got {}",
            tiles.len()
        );
    }

    #[test]
    fn test_centered_mercator_budget_reasonable() {
        let mut viewport = Viewport::new(1920, 1080);
        viewport.center = GeoCoord::new(85.0, 0.0);
        viewport.zoom = 5.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);
        assert!(
            tiles.len() <= 150,
            "Centered Mercator at pole: tile count {} exceeds budget 150",
            tiles.len()
        );
    }

    #[test]
    fn test_centered_mercator_high_zoom_reasonable() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5665, 126.978);
        viewport.zoom = 10.0;

        let standard_tiles = viewport.visible_tiles();
        let centered_tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);

        let ratio = centered_tiles.len() as f64 / standard_tiles.len().max(1) as f64;
        assert!(
            ratio < 3.0,
            "At high zoom, centered ({}) should not be much larger than standard ({})",
            centered_tiles.len(),
            standard_tiles.len(),
        );
    }

    #[test]
    fn test_globe_vp_is_orbital() {
        let mut vp = Viewport::new(640, 480);
        vp.center = GeoCoord::new(0.0, 0.0);
        vp.zoom = 0.0;
        let mat = vp.to_view_proj_f64_projected(x_planets_math::ProjectionMode::Globe);
        let expected = vp.to_globe_view_proj_f64();
        for i in 0..16 {
            assert!(
                (mat.to_cols_array()[i] - expected.to_cols_array()[i]).abs() < 1e-10,
                "Globe VP must use orbital camera (element {i} differs)",
            );
        }
    }

    #[test]
    fn test_mercator_vp_differs_from_globe() {
        let mut vp = Viewport::new(640, 480);
        vp.center = GeoCoord::new(35.0, 127.0);
        vp.zoom = 5.0;
        let merc_mat =
            vp.to_view_proj_f64_projected(x_planets_math::ProjectionMode::Mercator);
        let globe_mat =
            vp.to_view_proj_f64_projected(x_planets_math::ProjectionMode::Globe);
        let mut same_count = 0;
        for i in 0..16 {
            if (merc_mat.to_cols_array()[i] - globe_mat.to_cols_array()[i]).abs() < 1e-6 {
                same_count += 1;
            }
        }
        assert!(
            same_count < 14,
            "Mercator VP should differ substantially from Globe VP",
        );
    }

    #[test]
    fn test_globe_tiles_change_with_zoom() {
        let mut vp = Viewport::new(640, 480);
        vp.center = GeoCoord::new(0.0, 0.0);
        let mut last_max_z = 0u8;
        let mut changes = 0;
        for z in 0..=8 {
            vp.zoom = z as f64;
            let tiles =
                vp.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
            let max_z = tiles.iter().map(|t| t.coord.z).max().unwrap_or(0);
            if max_z != last_max_z {
                changes += 1;
            }
            last_max_z = max_z;
        }
        assert!(
            changes >= 3,
            "Globe tiles must change zoom level as viewport zooms \
             (only changed {changes} times across 0..8)",
        );
    }

    #[test]
    fn test_globe_pan_scales_with_zoom() {
        let ctrl = CameraController::new();

        let mut lo = Viewport::new(640, 480);
        lo.center = GeoCoord::new(0.0, 0.0);
        lo.zoom = 2.0;
        let orig_lo = lo.center;
        ctrl.pan_for_mode(&mut lo, 100.0, 0.0, x_planets_math::ProjectionMode::Globe);
        let dlat_lo = (lo.center.lat - orig_lo.lat).abs()
            + (lo.center.lon - orig_lo.lon).abs();

        let mut hi = Viewport::new(640, 480);
        hi.center = GeoCoord::new(0.0, 0.0);
        hi.zoom = 8.0;
        let orig_hi = hi.center;
        ctrl.pan_for_mode(&mut hi, 100.0, 0.0, x_planets_math::ProjectionMode::Globe);
        let dlat_hi = (hi.center.lat - orig_hi.lat).abs()
            + (hi.center.lon - orig_hi.lon).abs();

        assert!(
            dlat_lo > dlat_hi * 2.0,
            "Pan displacement must decrease with zoom \
             (low-zoom: {dlat_lo:.4}°, high-zoom: {dlat_hi:.4}°)",
        );
    }

    #[test]
    fn test_globe_tile_count_within_budget_all_zooms() {
        for height in [480, 720, 1080, 1440, 2160] {
            let width = height * 16 / 9;
            for zoom in 0..=15 {
                let mut vp = Viewport::new(width, height);
                vp.center = GeoCoord::new(0.0, 0.0);
                vp.zoom = zoom as f64;
                let tiles = vp
                    .visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
                assert!(
                    tiles.len() <= 150,
                    "Globe tiles at {width}x{height} zoom={zoom}: {} > 150",
                    tiles.len(),
                );
            }
        }
    }

    #[test]
    fn test_mercator_tile_count_within_budget_all_zooms() {
        for height in [480, 720, 1080, 1440, 2160] {
            let width = height * 16 / 9;
            for zoom in 0..=15 {
                let mut vp = Viewport::new(width, height);
                vp.center = GeoCoord::new(0.0, 0.0);
                vp.zoom = zoom as f64;
                let tiles = vp
                    .visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);
                assert!(
                    tiles.len() <= 150,
                    "Mercator tiles at {width}x{height} zoom={zoom}: {} > 150",
                    tiles.len(),
                );
            }
        }
    }

    #[test]
    fn test_pitched_tile_count_within_budget() {
        for pitch in [30.0, 45.0, 60.0] {
            for zoom in [3.0, 5.0, 8.0, 12.0] {
                let mut vp = Viewport::new(1920, 1080);
                vp.center = GeoCoord::new(37.5665, 126.978);
                vp.zoom = zoom;
                vp.pitch = pitch;
                let globe = vp
                    .visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
                let merc = vp
                    .visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);
                assert!(
                    globe.len() <= 150,
                    "Globe tiles at pitch={pitch} zoom={zoom}: {} > 150",
                    globe.len(),
                );
                assert!(
                    merc.len() <= 150,
                    "Mercator tiles at pitch={pitch} zoom={zoom}: {} > 150",
                    merc.len(),
                );
            }
        }
    }

    #[test]
    fn test_polar_tile_count_within_budget() {
        for lat in [80.0, 85.0, -80.0, -85.0] {
            for zoom in [0.0, 3.0, 5.0, 8.0] {
                let mut vp = Viewport::new(1920, 1080);
                vp.center = GeoCoord::new(lat, 0.0);
                vp.zoom = zoom;
                let globe = vp
                    .visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
                let merc = vp
                    .visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);
                assert!(
                    globe.len() <= 150,
                    "Globe tiles at lat={lat} zoom={zoom}: {} > 150",
                    globe.len(),
                );
                assert!(
                    merc.len() <= 150,
                    "Mercator tiles at lat={lat} zoom={zoom}: {} > 150",
                    merc.len(),
                );
            }
        }
    }

    #[test]
    fn test_pan_moves_center() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 5.0;

        let lat_before = viewport.center.lat;
        let lon_before = viewport.center.lon;
        ctrl.pan(&mut viewport, 50.0, 50.0);

        assert!(
            (viewport.center.lat - lat_before).abs() > 1e-6
                || (viewport.center.lon - lon_before).abs() > 1e-6,
            "Pan should move the center"
        );
    }

    #[test]
    fn test_pan_wraps_antimeridian() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 179.0);
        viewport.zoom = 5.0;

        ctrl.pan(&mut viewport, -500.0, 0.0);

        assert!(
            viewport.center.lon >= -180.0 && viewport.center.lon <= 180.0,
            "Longitude should wrap: got {}",
            viewport.center.lon
        );
    }

    #[test]
    fn test_pan_y_clamped() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(85.0, 0.0);
        viewport.zoom = 3.0;

        ctrl.pan(&mut viewport, 0.0, -5000.0);

        assert!(
            viewport.center.lat <= 85.1,
            "Latitude should be clamped, got {}",
            viewport.center.lat
        );
    }

    #[test]
    fn test_set_pitch_clamped() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);

        ctrl.set_pitch(&mut viewport, 90.0);
        assert!((viewport.pitch - 60.0).abs() < 1e-9, "pitch clamped to 60");

        ctrl.set_pitch(&mut viewport, -10.0);
        assert!((viewport.pitch - 0.0).abs() < 1e-9, "pitch clamped to 0");

        ctrl.set_pitch(&mut viewport, 30.0);
        assert!((viewport.pitch - 30.0).abs() < 1e-9, "pitch set to 30");
    }

    #[test]
    fn test_zoom_at_center_stable() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5, 127.0);
        viewport.zoom = 5.0;

        let lat_before = viewport.center.lat;
        let lon_before = viewport.center.lon;

        ctrl.zoom_at(&mut viewport, 2.0, 400.0, 300.0);

        assert!(
            (viewport.center.lat - lat_before).abs() < 0.01,
            "Zoom at center should not shift lat: {} → {}",
            lat_before, viewport.center.lat
        );
        assert!(
            (viewport.center.lon - lon_before).abs() < 0.01,
            "Zoom at center should not shift lon: {} → {}",
            lon_before, viewport.center.lon
        );
    }

    #[test]
    fn test_zoom_at_offset_shifts_center() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 3.0;

        ctrl.zoom_at(&mut viewport, 3.0, 750.0, 300.0);

        assert!(
            (viewport.center.lon).abs() > 0.01,
            "Zoom at offset should shift center, got lon={}",
            viewport.center.lon
        );
    }

    #[test]
    fn test_tile_zoom() {
        let mut viewport = Viewport::new(800, 600);
        viewport.zoom = 5.4;
        assert_eq!(viewport.tile_zoom(), 5);

        viewport.zoom = 5.6;
        assert_eq!(viewport.tile_zoom(), 6);

        viewport.zoom = 0.0;
        assert_eq!(viewport.tile_zoom(), 0);

        viewport.zoom = 25.0;
        assert_eq!(viewport.tile_zoom(), 22);
    }

    #[test]
    fn test_pan_for_mode_delegates() {
        let ctrl = CameraController::new();

        let mut vp = Viewport::new(800, 600);
        vp.center = GeoCoord::new(0.0, 0.0);
        vp.zoom = 5.0;
        let before = vp.center;
        ctrl.pan_for_mode(&mut vp, 50.0, 0.0, x_planets_math::ProjectionMode::Mercator);
        assert!(
            (vp.center.lon - before.lon).abs() > 1e-6,
            "pan_for_mode Mercator should pan"
        );

        let mut vp = Viewport::new(800, 600);
        vp.center = GeoCoord::new(0.0, 0.0);
        vp.zoom = 5.0;
        let before = vp.center;
        ctrl.pan_for_mode(&mut vp, 50.0, 0.0, x_planets_math::ProjectionMode::Globe);
        assert!(
            (vp.center.lon - before.lon).abs() > 1e-6,
            "pan_for_mode Globe should pan"
        );
    }

    #[test]
    fn test_centered_mercator_angular_filter_pass_rate() {
        use x_planets_math::ProjectionMode;

        let configs: Vec<(f64, f64, f64)> = vec![
            (30.0, 10.0, 5.0),
            (60.0, 10.0, 4.0),
            (-80.0, -60.0, 3.0),
            (0.0, 170.0, 5.0),
            (45.0, -90.0, 5.0),
        ];

        for (lat, lon, zoom) in &configs {
            let mut vp = Viewport::new(800, 600);
            vp.center = GeoCoord::new(*lat, *lon);
            vp.zoom = *zoom;

            let tiles = vp.visible_tiles_for_mode(ProjectionMode::Mercator);

            let center_lat_rad = lat.to_radians();
            let center_lon_rad = lon.to_radians();
            let renderables: Vec<_> = tiles
                .iter()
                .map(|vt| crate::pipeline::RenderableTile {
                    coord: vt.coord,
                    texture_coord: vt.coord,
                    uv_rect: [0.0, 0.0, 1.0, 1.0],
                    display_x: vt.display_x,
                })
                .collect();

            let passing: Vec<_> = renderables
                .iter()
                .filter(|rt| {
                    crate::pipeline::tile_passes_angular_filter(
                        rt,
                        center_lat_rad,
                        center_lon_rad,
                        *zoom,
                    )
                })
                .collect();

            let ratio = passing.len() as f64 / renderables.len().max(1) as f64;
            eprintln!(
                "  ({},{}) zoom {}: {} tiles selected, {} pass filter ({:.1}%)",
                lat, lon, zoom,
                renderables.len(), passing.len(), ratio * 100.0,
            );
        }
    }

    #[test]
    fn test_centered_mercator_full_viewport_coverage() {
        use x_planets_math::ProjectionMode;

        let configs: Vec<(f64, f64, f64, u32, u32)> = vec![
            (30.0, 10.0, 5.0, 800, 600),
            (60.0, 10.0, 4.0, 800, 600),
            (-80.0, -60.0, 3.0, 800, 600),
            (0.0, 170.0, 5.0, 800, 600),
            (45.0, -90.0, 5.0, 1024, 768),
        ];

        for (lat, lon, zoom, w, h) in &configs {
            let mut vp = Viewport::new(*w, *h);
            vp.center = GeoCoord::new(*lat, *lon);
            vp.zoom = *zoom;

            let tiles = vp.visible_tiles_for_mode(ProjectionMode::Mercator);

            let center_lat_rad = lat.to_radians();
            let center_lon_rad = lon.to_radians();
            let filtered_tiles: Vec<_> = tiles
                .iter()
                .filter(|vt| {
                    let rt = crate::pipeline::RenderableTile {
                        coord: vt.coord,
                        texture_coord: vt.coord,
                        uv_rect: [0.0, 0.0, 1.0, 1.0],
                        display_x: vt.display_x,
                    };
                    crate::pipeline::tile_passes_angular_filter(
                        &rt, center_lat_rad, center_lon_rad, *zoom,
                    )
                })
                .collect();

            let tile_set: std::collections::HashSet<(u8, u32, u32)> = filtered_tiles
                .iter()
                .map(|vt| (vt.coord.z, vt.coord.x, vt.coord.y))
                .collect();

            let scale = 2.0_f64.powf(-zoom);
            let aspect = *w as f64 / *h as f64;
            let half_h = scale * 1.0;
            let half_w = scale * aspect * 1.0;
            let center_lat_rad = lat.to_radians();
            let center_lon_rad = lon.to_radians();
            let threshold_cos = 85.0_f64.to_radians().cos();
            let center_sphere =
                x_planets_math::geo_to_unit_sphere(center_lat_rad, center_lon_rad);

            let mut missing = 0;
            let mut total = 0;
            let n_sample = 10;
            for iy in 0..=n_sample {
                for ix in 0..=n_sample {
                    let tx = ix as f64 / n_sample as f64;
                    let ty = iy as f64 / n_sample as f64;
                    let mx = 0.5 - half_w + 2.0 * half_w * tx;
                    let my = 0.5 - half_h + 2.0 * half_h * ty;
                    let (lr, lonr) = x_planets_math::oblique_mercator_inverse(
                        glam::DVec2::new(mx, my),
                        center_lat_rad,
                        center_lon_rad,
                    );
                    if !lr.is_finite() || !lonr.is_finite() {
                        continue;
                    }
                    let pt_sphere = x_planets_math::geo_to_unit_sphere(lr, lonr);
                    let cos_angle = center_sphere.dot(pt_sphere);
                    if cos_angle < threshold_cos {
                        continue;
                    }
                    let geo = GeoCoord::new(lr.to_degrees(), lonr.to_degrees());
                    let target_z = vp.tile_zoom();
                    let tc = x_planets_math::TileCoord::from_geo(&geo, target_z);
                    total += 1;
                    let mut found = false;
                    let mut check = Some(tc);
                    while let Some(c) = check {
                        if tile_set.contains(&(c.z, c.x, c.y)) {
                            found = true;
                            break;
                        }
                        check = c.parent();
                    }
                    if !found {
                        missing += 1;
                    }
                }
            }

            let coverage = 1.0 - missing as f64 / total.max(1) as f64;
            assert!(
                coverage >= 0.95,
                "Centered Mercator coverage at ({},{}) zoom {} is {:.1}% ({} missing / {} total). \
                 Selected {} tiles.",
                lat, lon, zoom, coverage * 100.0, missing, total, tiles.len(),
            );
        }
    }

    /// Diagnostic: zoom in at high zoom then apply pitch — tiles must not vanish.
    #[test]
    fn test_high_zoom_then_pitch_tiles_not_empty() {
        for zoom in [10.0, 13.0, 15.0, 17.0] {
            for pitch in [0.0, 30.0, 45.0, 60.0] {
                let mut vp = Viewport::new(1920, 1080);
                vp.center = GeoCoord::new(37.5665, 126.978); // Seoul
                vp.zoom = zoom;
                vp.pitch = pitch;
                let tiles =
                    vp.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);
                assert!(
                    !tiles.is_empty(),
                    "No tiles at zoom={zoom} pitch={pitch}! Tile selection is broken.",
                );
                // Center tile must be present at base zoom
                let center_merc = x_planets_math::geo_to_mercator(&vp.center);
                let base_z = vp.tile_zoom();
                let n = (1u32 << base_z) as f64;
                let center_tx = (center_merc.x * n).floor() as u32;
                let center_ty = (center_merc.y * n).floor() as u32;
                // Accept center tile at base_z OR one zoom level lower (LOD)
                let has_center_or_parent = tiles.iter().any(|t| {
                    if t.coord.z == base_z {
                        t.coord.x == center_tx && t.coord.y == center_ty
                    } else if t.coord.z == base_z.saturating_sub(1) {
                        let pn = (1u32 << t.coord.z) as f64;
                        let px = (center_merc.x * pn).floor() as u32;
                        let py = (center_merc.y * pn).floor() as u32;
                        t.coord.x == px && t.coord.y == py
                    } else {
                        false
                    }
                });
                // Count tiles by zoom
                let mut zoom_counts = std::collections::BTreeMap::new();
                for t in &tiles {
                    *zoom_counts.entry(t.coord.z).or_insert(0u32) += 1;
                }
                if !has_center_or_parent {
                    // Find z=10 tiles nearest to center
                    let mut z10_tiles: Vec<_> = tiles.iter()
                        .filter(|t| t.coord.z == base_z)
                        .map(|t| {
                            let tn = (1u32 << t.coord.z) as f64;
                            let tx = (t.coord.x as f64 + 0.5) / tn;
                            let ty = (t.coord.y as f64 + 0.5) / tn;
                            let dist = ((tx - center_merc.x).powi(2) + (ty - center_merc.y).powi(2)).sqrt();
                            (t.coord.x, t.coord.y, t.display_x, dist)
                        })
                        .collect();
                    z10_tiles.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());
                    let nearest_5: Vec<_> = z10_tiles.iter().take(5).collect();
                    panic!(
                        "Center tile area missing at zoom={zoom} pitch={pitch}! \
                         center_merc=({:.4},{:.4}), base_z={base_z}, expected tile ({center_tx},{center_ty}). \
                         Got {} tiles, zoom distribution: {:?}. \
                         Nearest z={base_z} tiles to center: {:?}",
                        center_merc.x, center_merc.y, tiles.len(), zoom_counts, nearest_5,
                    );
                }
            }
        }
    }

    /// Diagnostic: verify VP matrix projects center tile to visible clip space.
    #[test]
    fn test_vp_projects_center_tile_visible() {
        for zoom in [10.0, 15.0] {
            for pitch in [0.0, 30.0, 45.0, 60.0] {
                let mut vp = Viewport::new(1920, 1080);
                vp.center = GeoCoord::new(37.5665, 126.978);
                vp.zoom = zoom;
                vp.pitch = pitch;

                let vp_mat = vp.to_view_proj_f64_projected(
                    x_planets_math::ProjectionMode::Mercator,
                );

                // The center should project near (0.5, 0.5) in oblique Mercator
                // which is the camera target.  Check that it lands in clip space.
                let center_pos = glam::DVec4::new(0.5, 0.5, 0.0, 1.0);
                let clip = vp_mat * center_pos;
                let ndc_x = clip.x / clip.w;
                let ndc_y = clip.y / clip.w;
                let ndc_z = clip.z / clip.w;
                assert!(
                    ndc_x.abs() < 2.0 && ndc_y.abs() < 2.0,
                    "Center (0.5,0.5) projects outside NDC at zoom={zoom} pitch={pitch}: \
                     ndc=({ndc_x:.4}, {ndc_y:.4}, {ndc_z:.4}) clip_w={:.6}",
                    clip.w,
                );
                assert!(
                    ndc_z >= 0.0 && ndc_z <= 1.0,
                    "Center (0.5,0.5) depth outside [0,1] at zoom={zoom} pitch={pitch}: \
                     ndc_z={ndc_z:.6}, near/far clipping issue. clip_z={:.6}, clip_w={:.6}",
                    clip.z, clip.w,
                );
            }
        }
    }
}
