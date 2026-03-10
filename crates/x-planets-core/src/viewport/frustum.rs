//! Frustum geometry computation for the viewport.

use x_planets_math::{geo_to_mercator, mercator_to_geo, BoundingBox, ConvexPolygon2D, Frustum2D};

impl super::Viewport {
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
    pub(super) fn compute_frustum_geometry(&self) -> (BoundingBox, Option<ConvexPolygon2D>, glam::DVec2, glam::DVec2) {
        let center_merc = geo_to_mercator(&self.center);
        let scale = 2.0_f64.powf(-self.zoom);
        let aspect = self.width as f64 / self.height.max(1) as f64;

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
        // Scale with pitch: at high pitch the horizon is much further than cam_h.
        let pitch_factor = 1.0 + 2.0 * pitch_rad.sin();
        let max_dist = cam_h * 20.0 * pitch_factor;
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

        // Safety margin.  X is NOT clamped; Y is clamped to [0, 1].
        let margin = self.frustum_margin;
        let mx = (max_x - min_x) * margin;
        let my = (max_y - min_y) * margin;
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
}
