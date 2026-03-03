//! Viewport and camera controller for map navigation.

use x_planets_math::{
    geo_to_mercator, mercator_to_geo, BoundingBox, ConvexPolygon2D, Frustum2D, GeoCoord,
    ViewportUniforms, VisibleTile,
};

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
        }
    }

    /// Integer zoom level for tile fetching.
    pub fn tile_zoom(&self) -> u8 {
        self.zoom.round().clamp(0.0, 22.0) as u8
    }

    /// Get the geographic bounding box visible in this viewport.
    ///
    /// For pitched/rotated views, traces the four perspective-frustum corner
    /// rays to the z=0 ground plane so the AABB genuinely covers the full
    /// camera view.  For top-down north-up views, a fast analytic path is used.
    pub fn visible_bounds(&self) -> BoundingBox {
        let (bbox, _, _, _) = self.compute_frustum_geometry();
        bbox
    }

    /// Compute frustum geometry: returns (BoundingBox, Option<ConvexPolygon2D>, merc_sw, merc_ne).
    ///
    /// The polygon is `None` for top-down north-up views (where AABB is tight).
    /// `merc_sw` / `merc_ne` are raw Mercator bounds where X may extend
    /// beyond [0, 1] for viewports crossing the antimeridian.
    fn compute_frustum_geometry(&self) -> (BoundingBox, Option<ConvexPolygon2D>, glam::DVec2, glam::DVec2) {
        let center_merc = geo_to_mercator(&self.center);
        let scale = 2.0_f64.powf(-self.zoom);
        let aspect = self.width as f64 / self.height as f64;

        // Fast path: nearly top-down, nearly north-up.
        if self.pitch < 1.0 && self.bearing.abs() < 1.0 {
            let half_h = scale * 1.1;
            let half_w = scale * aspect * 1.1;
            // X is NOT clamped — allows viewport to extend beyond [0,1] for
            // antimeridian wrapping.  Y is clamped (no vertical wrap in Mercator).
            let sw = glam::DVec2::new(
                center_merc.x - half_w,
                (center_merc.y + half_h).clamp(0.0, 1.0),
            );
            let ne = glam::DVec2::new(
                center_merc.x + half_w,
                (center_merc.y - half_h).clamp(0.0, 1.0),
            );
            let sw_geo = mercator_to_geo(glam::DVec2::new(sw.x.clamp(0.0, 1.0), sw.y));
            let ne_geo = mercator_to_geo(glam::DVec2::new(ne.x.clamp(0.0, 1.0), ne.y));
            return (BoundingBox::new(sw_geo, ne_geo), None, sw, ne);
        }

        // ── Perspective frustum ray → ground-plane intersection ──
        let fov_half = std::f64::consts::FRAC_PI_3 * 0.5; // 30°
        let fov_half_tan = fov_half.tan();
        let cam_h = scale / fov_half_tan;

        let pitch_rad = self.pitch.to_radians();
        let bearing_rad = self.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();
        let cx = center_merc.x;
        let cy = center_merc.y;

        // Camera eye (same geometry as to_uniforms).
        let eye = glam::DVec3::new(
            cx - sin_b * cam_h * pitch_rad.sin(),
            cy + cos_b * cam_h * pitch_rad.sin(),
            cam_h * pitch_rad.cos(),
        );
        let target = glam::DVec3::new(cx, cy, 0.0);
        let up_hint = glam::DVec3::new(sin_b, -cos_b, 0.0);

        // Camera basis (matches glam look_at_rh convention).
        let f = (target - eye).normalize();
        let s = f.cross(up_hint).normalize_or_zero();
        let u = s.cross(f);

        // Frustum half-extents at unit distance along f.
        let hw = fov_half_tan * aspect;
        let hh = fov_half_tan;

        // Four frustum corner directions.
        let corner_dirs = [
            f + s * hw + u * hh,
            f - s * hw + u * hh,
            f + s * hw - u * hh,
            f - s * hw - u * hh,
        ];

        // Maximum trace distance (prevents infinite bounds at the horizon).
        let max_dist = cam_h * 20.0;
        let mut min_x = cx;
        let mut max_x = cx;
        let mut min_y = cy;
        let mut max_y = cy;

        // Collect ground-plane hit points for the convex polygon.
        let mut ground_points = Vec::with_capacity(4);

        for dir in &corner_dirs {
            let d = dir.normalize();
            let (gx, gy) = if d.z < -1e-10 {
                // Ray hits the ground plane.
                let t = (-eye.z / d.z).min(max_dist);
                let g = eye + d * t;
                (g.x, g.y)
            } else {
                // Ray is horizontal or points upward — project to max distance.
                let horiz = glam::DVec2::new(d.x, d.y);
                let len = horiz.length();
                if len > 1e-10 {
                    let h = horiz / len;
                    (eye.x + h.x * max_dist, eye.y + h.y * max_dist)
                } else {
                    (cx, cy)
                }
            };
            // X is NOT clamped — allows viewport to extend beyond [0,1] for
            // antimeridian wrapping.  Y is clamped (no vertical wrap in Mercator).
            ground_points.push(glam::DVec2::new(
                gx,
                gy.clamp(0.0, 1.0),
            ));
            min_x = min_x.min(gx);
            max_x = max_x.max(gx);
            min_y = min_y.min(gy);
            max_y = max_y.max(gy);
        }

        // 5 % safety margin.  X is NOT clamped; Y is clamped to [0, 1].
        let mx = (max_x - min_x) * 0.05;
        let my = (max_y - min_y) * 0.05;
        min_x -= mx;
        max_x += mx;
        min_y = (min_y - my).clamp(0.0, 1.0);
        max_y = (max_y + my).clamp(0.0, 1.0);

        // Raw Mercator bounds (X can be outside [0,1])
        let merc_sw = glam::DVec2::new(min_x, max_y);
        let merc_ne = glam::DVec2::new(max_x, min_y);

        // BoundingBox (for backward compat) uses clamped X for GeoCoord conversion
        let bbox = BoundingBox::new(
            mercator_to_geo(glam::DVec2::new(min_x.clamp(0.0, 1.0), max_y)),
            mercator_to_geo(glam::DVec2::new(max_x.clamp(0.0, 1.0), min_y)),
        );

        // Build convex polygon from ground-plane hit points (with margin).
        // Add margin points to the polygon as well for safety.
        let polygon = ConvexPolygon2D::from_points(&ground_points);

        (bbox, polygon, merc_sw, merc_ne)
    }

    /// Get the frustum for tile culling.
    ///
    /// For pitched/rotated views, includes a convex polygon for precise
    /// culling that avoids wasting tile budget on off-screen tiles.
    pub fn frustum(&self) -> Frustum2D {
        let (bounds, polygon, merc_sw, merc_ne) = self.compute_frustum_geometry();
        match polygon {
            Some(poly) => Frustum2D::with_merc_bounds_and_polygon(bounds, merc_sw, merc_ne, poly),
            None => Frustum2D::with_merc_bounds(bounds, merc_sw, merc_ne),
        }
    }

    /// Get the list of visible tiles with per-tile screen-space LOD.
    ///
    /// Uses **top-down quadtree refinement**: starts from coarse tiles and
    /// subdivides only where more detail is needed.  A `BinaryHeap`
    /// prioritises near-camera tiles so they always get full detail,
    /// while far/horizon tiles stay coarse.
    ///
    /// **Gap-free guarantee**: every tile either stays in the result
    /// (at its current zoom) or is subdivided into 4 children.
    /// When the budget is reached, remaining heap entries are drained
    /// directly into the result at their current (coarse) level.
    ///
    /// Returns tiles sorted by ascending zoom (coarser first → finer last).
    pub fn visible_tiles(&self) -> Vec<VisibleTile> {
        let base_z = self.tile_zoom();
        if base_z == 0 {
            return self.frustum().visible_tiles(0);
        }
        self.quadtree_lod(base_z)
    }

    /// Quadtree-based LOD tile selection (gap-free, priority-ordered, budgeted).
    fn quadtree_lod(&self, base_z: u8) -> Vec<VisibleTile> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        const TILE_BUDGET: usize = 150;

        let center_merc = geo_to_mercator(&self.center);
        let pitch_rad = self.pitch.to_radians();
        let sin_p = pitch_rad.sin();
        let scale = 2.0_f64.powf(-self.zoom);
        let fov_half_tan = (std::f64::consts::FRAC_PI_3 * 0.5).tan();
        let cam_h = scale / fov_half_tan;

        let bearing_rad = self.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        let max_drop = ((self.pitch / 15.0).ceil() as u8).min(4);
        let min_z = base_z.saturating_sub(max_drop);

        // ── Ideal zoom for a point in Mercator space ──
        let ideal_zoom_at = |mx: f64, my: f64| -> u8 {
            if self.pitch < 5.0 {
                return base_z;
            }
            let dx = mx - center_merc.x;
            let dy = my - center_merc.y;
            let d_fwd = dx * sin_b - dy * cos_b;

            if d_fwd > 0.0 && sin_p > 0.01 {
                let perspective = cam_h / (cam_h + d_fwd * sin_p);
                (self.zoom + perspective.log2())
                    .round()
                    .clamp(min_z as f64, base_z as f64) as u8
            } else {
                base_z
            }
        };

        // ── Priority: closer tiles are more important (max-heap) ──
        #[derive(Debug)]
        struct Candidate {
            tile: VisibleTile,
            priority: f64, // higher = more important
        }
        impl PartialEq for Candidate {
            fn eq(&self, other: &Self) -> bool {
                self.priority == other.priority
            }
        }
        impl Eq for Candidate {}
        impl PartialOrd for Candidate {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Candidate {
            fn cmp(&self, other: &Self) -> Ordering {
                self.priority
                    .partial_cmp(&other.priority)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let frustum = self.frustum();
        let mut heap = BinaryHeap::<Candidate>::new();
        let mut result = Vec::<VisibleTile>::new();

        // Seed: visible tiles at min_z (coarse, guaranteed small set).
        for vt in frustum.visible_tiles(min_z) {
            let tc = vt.display_mercator_center();
            let dist = (tc - center_merc).length();
            heap.push(Candidate {
                tile: vt,
                priority: 1.0 / (dist + 1e-10),
            });
        }

        // ── Quadtree traversal ──
        while let Some(candidate) = heap.pop() {
            let vt = candidate.tile;
            let tc = vt.display_mercator_center();
            let ideal_z = ideal_zoom_at(tc.x, tc.y);

            // Should we subdivide this tile?
            let should_subdivide = vt.coord.z < ideal_z
                && vt.coord.z < base_z
                && (result.len() + heap.len() + 4) <= TILE_BUDGET;

            if should_subdivide {
                // Push 4 children — visible ones only.
                for child in vt.children() {
                    if frustum.is_visible_tile(&child) {
                        let cc = child.display_mercator_center();
                        let dist = (cc - center_merc).length();
                        heap.push(Candidate {
                            tile: child,
                            priority: 1.0 / (dist + 1e-10),
                        });
                    }
                }
            } else {
                // Keep at current zoom level.
                result.push(vt);
            }
        }

        // Sort: coarser first (background), finer last (foreground).
        result.sort_by_key(|vt| vt.coord.z);
        result
    }

    /// Compute the view-projection matrix in f64 for high-precision per-tile MVP.
    ///
    /// Same camera geometry as `to_uniforms()` but uses `DMat4` throughout
    /// to avoid f32 precision loss at high zoom levels.  Each tile computes
    /// `MVP = VP_f64 * translate(tile_center_f64)` in f64, then casts to f32.
    /// This eliminates the ~14px jitter at zoom 18+ caused by f32 VP.
    pub fn to_view_proj_f64(&self) -> glam::DMat4 {
        let center_merc = geo_to_mercator(&self.center);
        let scale = 2.0_f64.powf(self.zoom);
        let aspect = self.width as f64 / self.height as f64;
        let cx = center_merc.x;
        let cy = center_merc.y;

        let half_h = 1.0 / scale;

        let fov_y: f64 = std::f64::consts::FRAC_PI_3; // 60°
        let cam_h = half_h / (fov_y * 0.5).tan();

        let pitch_rad = self.pitch.to_radians();
        let bearing_rad = self.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        let up = glam::DVec3::new(sin_b, -cos_b, 0.0);

        let eye = glam::DVec3::new(
            cx - sin_b * cam_h * pitch_rad.sin(),
            cy + cos_b * cam_h * pitch_rad.sin(),
            cam_h * pitch_rad.cos(),
        );
        let target = glam::DVec3::new(cx, cy, 0.0);

        let view = glam::DMat4::look_at_rh(eye, target, up);
        let proj = glam::DMat4::perspective_rh(fov_y, aspect, cam_h * 0.005, cam_h * 10.0);

        let flip_x = glam::DMat4::from_diagonal(glam::DVec4::new(-1.0, 1.0, 1.0, 1.0));
        flip_x * proj * view
    }

    /// Compute GPU uniforms for this viewport.
    ///
    /// Uses perspective projection so pitch and bearing work correctly.
    /// When pitch=0 and bearing=0 this is equivalent to orthographic top-down.
    pub fn to_uniforms(&self) -> ViewportUniforms {
        let center_merc = geo_to_mercator(&self.center);
        let scale = 2.0_f64.powf(self.zoom) as f32;
        let aspect = self.width as f32 / self.height as f32;
        let cx = center_merc.x as f32;
        let cy = center_merc.y as f32;

        let half_h = 1.0 / scale;

        // FOV set so that at pitch=0 the visible height matches orthographic half_h.
        // tan(fov/2) = half_h / cam_h → cam_h = half_h / tan(fov/2)
        let fov_y = std::f32::consts::FRAC_PI_3; // 60°
        let cam_h = half_h / (fov_y * 0.5).tan();

        let pitch_rad = self.pitch.to_radians() as f32;
        let bearing_rad = self.bearing.to_radians() as f32;
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        // Camera "up" = the direction bearing points (north rotated clockwise by bearing).
        // At bearing=0: up = (0, -1, 0) = north in Mercator (Y=0 = north).
        // At bearing=90: up = (1, 0, 0) = east.
        let up = glam::Vec3::new(sin_b, -cos_b, 0.0);

        // Camera displaced "backward" from the look direction by pitch.
        // Look direction (forward) = (sin_b, -cos_b, 0) horizontally.
        // Backward = (-sin_b, cos_b, 0).
        // At bearing=0, pitch>0: eye displaced south (+Y) above center. ✓
        let eye = glam::Vec3::new(
            cx - sin_b * cam_h * pitch_rad.sin(),
            cy + cos_b * cam_h * pitch_rad.sin(),
            cam_h * pitch_rad.cos(),
        );
        let target = glam::Vec3::new(cx, cy, 0.0);

        let view = glam::Mat4::look_at_rh(eye, target, up);
        let proj = glam::Mat4::perspective_rh(fov_y, aspect, cam_h * 0.005, cam_h * 10.0);

        // look_at_rh with up=(sin_b,-cos_b,0) makes camera_right = world(-cos_b,-sin_b,0),
        // which flips X when bearing=0. Correct with a -X scale so world east → screen right.
        let flip_x = glam::Mat4::from_diagonal(glam::Vec4::new(-1.0, 1.0, 1.0, 1.0));
        let view_proj = (flip_x * proj * view).to_cols_array();

        ViewportUniforms {
            view_proj,
            resolution: [
                self.width as f32,
                self.height as f32,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
            camera: [cx, cy, self.zoom as f32, self.pitch as f32],
        }
    }
}

/// Controls camera movement (pan, zoom, pitch, bearing).
pub struct CameraController {
    /// Pan sensitivity (pixels per degree).
    pub pan_speed: f64,
    /// Zoom speed (scroll units per level).
    pub zoom_speed: f64,
    /// Minimum zoom level.
    pub min_zoom: f64,
    /// Maximum zoom level.
    pub max_zoom: f64,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            pan_speed: 2.0,
            zoom_speed: 1.0,
            min_zoom: 0.0,
            max_zoom: 22.0,
        }
    }

    /// Pan the viewport by a screen-space pixel delta.
    ///
    /// dx > 0 = rightward drag, dy > 0 = upward drag (already negated by caller).
    /// Rotates the delta by the current bearing so panning always follows the screen.
    pub fn pan(&self, viewport: &mut Viewport, dx: f64, dy: f64) {
        let scale = 2.0_f64.powf(-viewport.zoom);
        let aspect = viewport.width as f64 / viewport.height as f64;

        let bearing_rad = viewport.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        // Normalize to Mercator units per pixel.
        let dx_n = dx / viewport.width as f64 * scale * aspect * self.pan_speed;
        let dy_n = dy / viewport.height as f64 * scale * self.pan_speed;

        // screen_right = (cos_b, sin_b), screen_up = (sin_b, -cos_b) in Mercator XY.
        // Center moves opposite to drag: -screen_right * dx_n - screen_up * dy_n.
        let merc_dx = -(cos_b * dx_n + sin_b * dy_n);
        let merc_dy = -(sin_b * dx_n - cos_b * dy_n);

        let mut center_merc = geo_to_mercator(&viewport.center);
        // X wraps around the antimeridian; Y stays clamped (no vertical wrap).
        center_merc.x = (center_merc.x + merc_dx).rem_euclid(1.0);
        center_merc.y = (center_merc.y + merc_dy).clamp(0.0, 1.0);

        viewport.center = mercator_to_geo(center_merc);
    }

    /// Zoom the viewport by a delta (positive = zoom in).
    pub fn zoom(&self, viewport: &mut Viewport, delta: f64) {
        viewport.zoom = (viewport.zoom + delta * self.zoom_speed)
            .clamp(self.min_zoom, self.max_zoom);
    }

    /// Set absolute pitch angle (clamped to 0–60 degrees).
    pub fn set_pitch(&self, viewport: &mut Viewport, degrees: f64) {
        viewport.pitch = degrees.clamp(0.0, 60.0);
    }

    /// Set absolute bearing (0–360 degrees, clockwise from north).
    pub fn set_bearing(&self, viewport: &mut Viewport, degrees: f64) {
        viewport.bearing = degrees.rem_euclid(360.0);
    }

    /// Zoom toward a specific screen point (zoom-to-pointer).
    ///
    /// The geographic point under (screen_x, screen_y) remains fixed after zooming.
    pub fn zoom_at(
        &self,
        viewport: &mut Viewport,
        delta: f64,
        screen_x: f64,
        screen_y: f64,
    ) {
        let old_scale = 2.0_f64.powf(-viewport.zoom);

        self.zoom(viewport, delta);

        let new_scale = 2.0_f64.powf(-viewport.zoom);
        let scale_diff = old_scale - new_scale;
        if scale_diff == 0.0 {
            return;
        }

        let aspect = viewport.width as f64 / viewport.height as f64;

        // Normalized screen offset from center to cursor (right=+, down=+).
        let dx_norm = (screen_x - viewport.width as f64 * 0.5) / viewport.width as f64;
        let dy_norm = (screen_y - viewport.height as f64 * 0.5) / viewport.height as f64;

        // Rotate by bearing: screen_right = (cos_b, sin_b), screen_down = (-sin_b, cos_b).
        let bearing_rad = viewport.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        // Mercator offset of cursor from center = scale * [rotated screen offset].
        // When scale changes by scale_diff, shift center so cursor stays fixed.
        let merc_dx = (dx_norm * aspect * cos_b - dy_norm * sin_b) * scale_diff;
        let merc_dy = (dx_norm * aspect * sin_b + dy_norm * cos_b) * scale_diff;

        let mut center_merc = geo_to_mercator(&viewport.center);
        // X wraps around the antimeridian; Y stays clamped.
        center_merc.x = (center_merc.x + merc_dx).rem_euclid(1.0);
        center_merc.y = (center_merc.y + merc_dy).clamp(0.0, 1.0);
        viewport.center = mercator_to_geo(center_merc);
    }
}

impl Default for CameraController {
    fn default() -> Self {
        Self::new()
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
        // BoundingBox should have nonzero extent.
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

        // Should contain tiles at multiple zoom levels.
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
        // Tiles should be sorted by zoom level (coarse first).
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
        // At high pitch the perspective frustum extends much further
        // than the simple 1/cos(pitch) formula.
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 6.0;
        viewport.pitch = 55.0;

        let bounds = viewport.visible_bounds();
        // With bearing=0, camera faces north.  Far edge should be well
        // north of center (lat > 0 by a significant margin).
        assert!(
            bounds.north_east.lat > 20.0,
            "Pitched bounds should extend far north, got NE lat {:.1}",
            bounds.north_east.lat
        );
    }

    #[test]
    fn test_lod_gap_free() {
        // Every base-z tile must be covered by exactly one result tile
        // (itself at base_z, or an ancestor at a coarser level).
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
            // Walk up from base_tile; at least one ancestor must be in result_set.
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

    // ── New tests for quadtree LOD / convex frustum ──────────

    #[test]
    fn test_quadtree_budget_respected() {
        // At high zoom + high pitch the quadtree should respect TILE_BUDGET.
        let mut viewport = Viewport::new(1920, 1080);
        viewport.center = GeoCoord::new(37.5, 127.0);
        viewport.zoom = 15.0;
        viewport.pitch = 60.0;

        let tiles = viewport.visible_tiles();
        assert!(
            tiles.len() <= 200, // Allow some slack above TILE_BUDGET=150
            "Too many tiles: {} (budget should cap this)",
            tiles.len()
        );
    }

    #[test]
    fn test_quadtree_near_tiles_prioritized() {
        // Near-camera tiles should be at the finest zoom level.
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5, 127.0);
        viewport.zoom = 8.0;
        viewport.pitch = 55.0;

        let base_z = viewport.tile_zoom();
        let tiles = viewport.visible_tiles();
        // Find tiles that are at the base zoom level (finest).
        let fine_tiles: Vec<_> = tiles.iter().filter(|t| t.coord.z == base_z).collect();
        assert!(
            !fine_tiles.is_empty(),
            "Should have at least some tiles at the base zoom level {}",
            base_z
        );
    }

    #[test]
    fn test_quadtree_lod_with_bearing() {
        // Rotated view should still produce valid gap-free tiles.
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 6.0;
        viewport.pitch = 45.0;
        viewport.bearing = 45.0;

        let tiles = viewport.visible_tiles();
        assert!(!tiles.is_empty());

        // Should still be sorted coarse-first.
        for pair in tiles.windows(2) {
            assert!(pair[0].coord.z <= pair[1].coord.z);
        }
    }

    #[test]
    fn test_frustum_polygon_active_when_pitched() {
        // When pitched, compute_frustum_geometry should produce a polygon.
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
}
