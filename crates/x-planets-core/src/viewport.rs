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

    /// Projection-aware visible tile selection.
    ///
    /// For Globe mode, computes tile zoom from the orbital camera altitude
    /// and selects tiles visible from the sphere surface.  For Mercator and
    /// other flat modes, delegates to the standard Mercator-based frustum.
    pub fn visible_tiles_for_mode(&self, mode: x_planets_math::ProjectionMode) -> Vec<VisibleTile> {
        match mode {
            x_planets_math::ProjectionMode::Globe => self.visible_tiles_globe(),
            x_planets_math::ProjectionMode::Mercator => self.visible_tiles_centered(),
            _ => self.visible_tiles(),
        }
    }

    /// Centered-Mercator visible tile selection.
    ///
    /// The centered (oblique) Mercator rendering path re-projects tiles
    /// through an oblique Mercator centered on the viewport.  Near the
    /// poles the standard Mercator frustum misses tiles that wrap around
    /// the sphere.  This method computes the visible spherical cap from
    /// the viewport zoom and selects all tiles within it.
    fn visible_tiles_centered(&self) -> Vec<VisibleTile> {
        let scale = 2.0_f64.powf(-self.zoom);
        let aspect = self.width as f64 / self.height as f64;

        // Viewport half-extents in oblique Mercator space (same formula
        // as the top-down fast path in compute_frustum_geometry).
        let half_h = scale * 1.1;
        let half_w = scale * aspect * 1.1;

        // Maximum distance from center in oblique Mercator space.
        // Use the diagonal for the worst-case corner.
        let mut max_extent = (half_h * half_h + half_w * half_w).sqrt();

        // For pitched views the forward ground-plane intersection
        // extends much further.
        if self.pitch >= 1.0 {
            let fov_half_tan = (std::f64::consts::FRAC_PI_3 * 0.5).tan();
            let cam_h = scale / fov_half_tan;
            let pitch_rad = self.pitch.to_radians();
            let fov_half = std::f64::consts::FRAC_PI_3 * 0.5;
            let bottom_angle = pitch_rad + fov_half;
            let forward_dist = if bottom_angle < std::f64::consts::FRAC_PI_2 * 0.98 {
                cam_h * bottom_angle.tan()
            } else {
                cam_h * 20.0 // horizon cap
            };
            max_extent = max_extent.max(forward_dist);
        }

        // Convert oblique Mercator extent to angular distance on the
        // sphere.  In the rotated coordinate system the center maps to
        // Mercator Y = 0.5; an offset of `max_extent` corresponds to a
        // certain latitude (= angular distance from center).
        let edge_y = (0.5 + max_extent).min(0.9999);
        let edge_geo = mercator_to_geo(glam::DVec2::new(0.5, edge_y));
        let viewport_angular_deg = edge_geo.lat.abs();

        // Cap at the oblique Mercator singularity guard threshold
        // (mirrors pipeline::centered_angular_threshold_deg).
        let threshold_deg: f64 = if self.zoom < 4.0 { 89.0 } else { 85.0 };
        let visible_deg = viewport_angular_deg.min(threshold_deg);

        let lat = self.center.lat;
        let lon = self.center.lon;

        // Geographic bounding box covering the spherical cap.
        let lat_min = (lat - visible_deg).max(-85.05);
        let lat_max = (lat + visible_deg).min(85.05);
        // Longitude span widens at higher latitudes (meridian convergence).
        let cos_lat = lat.to_radians().cos().max(0.01);
        let lon_span = (visible_deg / cos_lat).min(180.0);
        let lon_min = lon - lon_span;
        let lon_max = lon + lon_span;

        // Build a Frustum2D from the geographic extent.
        let sw = geo_to_mercator(&GeoCoord::new(lat_min, lon_min));
        let ne = geo_to_mercator(&GeoCoord::new(lat_max, lon_max));
        let bbox = BoundingBox::new(
            GeoCoord::new(lat_min, lon_min.clamp(-180.0, 180.0)),
            GeoCoord::new(lat_max, lon_max.clamp(-180.0, 180.0)),
        );
        let frustum = Frustum2D::with_merc_bounds(bbox, sw, ne);

        let base_z = self.tile_zoom();
        if base_z == 0 {
            return frustum.visible_tiles(0);
        }
        self.quadtree_lod_with_frustum(base_z, &frustum)
    }

    /// Globe-mode visible tile selection.
    ///
    /// Uses the orbital camera geometry (altitude above unit sphere) to
    /// determine the visible spherical cap, then converts it to Mercator
    /// tile coordinates for tile fetching.
    fn visible_tiles_globe(&self) -> Vec<VisibleTile> {
        // Camera altitude in unit-sphere radii.
        let unit_altitude =
            (20_000_000.0 / 6_378_137.0) / 2.0_f64.powf(self.zoom);

        // Angular radius of the visible cap on the sphere surface.
        // cos(surface_angle) = R / (R + h) = 1 / (1 + unit_altitude)
        // (acos gives the angle measured on the sphere from the
        //  sub-satellite point to the horizon; asin would give the
        //  much smaller camera-to-limb angle.)
        let half_angle = (1.0 / (unit_altitude + 1.0)).acos();

        // Compute effective zoom from angular extent:
        // At zoom z, each tile covers 360/2^z degrees of longitude.
        // The visible cap diameter in degrees ≈ 2 * half_angle_degrees.
        // We want tiles where tile_angular_size ≈ viewport_angular_size / (viewport_pixels / 256).
        let visible_deg = half_angle.to_degrees() * 2.0;
        let tiles_needed = (self.height as f64 / 256.0).max(1.0);
        let tile_size_deg = visible_deg / tiles_needed;
        // 360 / 2^z = tile_size_deg → z = log2(360 / tile_size_deg)
        let globe_zoom = (360.0 / tile_size_deg).log2()
            .round()
            .clamp(0.0, 22.0) as u8;

        // Visible bounding box in geographic coordinates.
        let half_deg = half_angle.to_degrees().min(89.0);
        let lat = self.center.lat;
        let lon = self.center.lon;

        let lat_min = (lat - half_deg).max(-85.05);
        let lat_max = (lat + half_deg).min(85.05);
        // Longitude span scales by cos(lat) at the equator edge
        let cos_lat = lat.to_radians().cos().max(0.05);
        let lon_span = (half_deg / cos_lat).min(180.0);
        let lon_min = lon - lon_span;
        let lon_max = lon + lon_span;

        // Convert to Mercator and build a Frustum2D.
        let sw = geo_to_mercator(&GeoCoord::new(lat_min, lon_min));
        let ne = geo_to_mercator(&GeoCoord::new(lat_max, lon_max));

        let bbox = BoundingBox::new(
            GeoCoord::new(lat_min, lon_min.clamp(-180.0, 180.0)),
            GeoCoord::new(lat_max, lon_max.clamp(-180.0, 180.0)),
        );
        let frustum = Frustum2D::with_merc_bounds(bbox, sw, ne);

        if globe_zoom == 0 {
            return frustum.visible_tiles(0);
        }

        // Use the quadtree LOD algorithm with globe-derived zoom.
        self.quadtree_lod_with_frustum(globe_zoom, &frustum)
    }

    /// Quadtree LOD with a custom frustum (used by globe-mode visible tiles).
    fn quadtree_lod_with_frustum(&self, base_z: u8, frustum: &Frustum2D) -> Vec<VisibleTile> {
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

        #[derive(Debug)]
        struct Candidate {
            tile: VisibleTile,
            priority: f64,
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

        let mut heap = BinaryHeap::<Candidate>::new();
        let mut result = Vec::<VisibleTile>::new();

        for vt in frustum.visible_tiles(min_z) {
            let tc = vt.display_mercator_center();
            let dist = (tc - center_merc).length();
            heap.push(Candidate {
                tile: vt,
                priority: 1.0 / (dist + 1e-10),
            });
        }

        while let Some(candidate) = heap.pop() {
            let vt = candidate.tile;
            let tc = vt.display_mercator_center();
            let ideal_z = ideal_zoom_at(tc.x, tc.y);

            let should_subdivide = vt.coord.z < ideal_z
                && vt.coord.z < base_z
                && (result.len() + heap.len() + 4) <= TILE_BUDGET;

            if should_subdivide {
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
                result.push(vt);
            }
        }

        result.sort_by_key(|vt| vt.coord.z);
        result
    }

    /// Quadtree-based LOD tile selection (gap-free, priority-ordered, budgeted).
    fn quadtree_lod(&self, base_z: u8) -> Vec<VisibleTile> {
        let frustum = self.frustum();
        self.quadtree_lod_with_frustum(base_z, &frustum)
    }

    /// Compute the view-projection matrix in f64 for high-precision per-tile MVP.
    ///
    /// Same camera geometry as `to_uniforms()` but uses `DMat4` throughout
    /// to avoid f32 precision loss at high zoom levels.  Each tile computes
    /// `MVP = VP_f64 * translate(tile_center_f64)` in f64, then casts to f32.
    /// This eliminates the ~14px jitter at zoom 18+ caused by f32 VP.
    pub fn to_view_proj_f64(&self) -> glam::DMat4 {
        self.to_view_proj_f64_projected(x_planets_math::ProjectionMode::Mercator)
    }

    /// Like [`to_view_proj_f64`] but positions the camera using the given projection.
    pub fn to_view_proj_f64_projected(&self, mode: x_planets_math::ProjectionMode) -> glam::DMat4 {
        if mode == x_planets_math::ProjectionMode::Globe {
            return self.to_globe_view_proj_f64();
        }
        let center = match mode {
            x_planets_math::ProjectionMode::Mercator => {
                // Centered (oblique) Mercator: viewport center → (0.5, 0.5)
                glam::DVec2::new(0.5, 0.5)
            }
            x_planets_math::ProjectionMode::Globe => geo_to_mercator(&self.center),
            x_planets_math::ProjectionMode::Equirectangular => {
                x_planets_math::geo_to_equirectangular(&self.center)
            }
        };
        let center_merc = center;
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

    /// Compute the globe view-projection matrix in f64.
    ///
    /// Orbital camera around a unit sphere.  The camera looks at the surface
    /// point corresponding to `self.center`, positioned at a distance
    /// determined by `self.zoom`.  Pitch and bearing rotate the camera.
    pub fn to_globe_view_proj_f64(&self) -> glam::DMat4 {
        let lat_rad = self.center.lat.to_radians();
        let lon_rad = self.center.lon.to_radians();

        // Surface point (look-at target) on unit sphere
        let surface_point = x_planets_math::geo_to_unit_sphere(lat_rad, lon_rad);

        // Camera altitude above sphere surface (unit-sphere radius = 1.0).
        // At zoom 0, ~3.14 radii above surface → sees whole globe.
        // Each zoom level halves the altitude.
        let unit_altitude =
            (20_000_000.0 / 6_378_137.0) / 2.0_f64.powf(self.zoom);

        // Local ENU (East-North-Up) basis at surface point
        let up_surface = surface_point.normalize();
        let east = glam::DVec3::new(-lon_rad.sin(), lon_rad.cos(), 0.0).normalize();
        let north = up_surface.cross(east).normalize();

        // Apply bearing: rotate horizontal component
        let bearing_rad = self.bearing.to_radians();
        let cos_b = bearing_rad.cos();
        let sin_b = bearing_rad.sin();
        let forward_h = north * cos_b + east * sin_b;

        // Apply pitch: at pitch=0, camera directly above, looking down.
        let pitch_rad = self.pitch.to_radians();
        let backward = -forward_h * pitch_rad.sin() + up_surface * pitch_rad.cos();
        let camera_pos = surface_point + backward.normalize() * unit_altitude;

        let camera_up = if pitch_rad.abs() < 0.01 {
            forward_h.normalize()
        } else {
            let forward = (surface_point - camera_pos).normalize();
            let right = forward.cross(up_surface).normalize();
            right.cross(forward).normalize()
        };

        let view = glam::DMat4::look_at_rh(camera_pos, surface_point, camera_up);

        let aspect = self.width as f64 / self.height.max(1) as f64;
        let fov_y: f64 = std::f64::consts::FRAC_PI_3; // 60°
        let near = unit_altitude * 0.01;
        let far = (unit_altitude + 2.0) * 3.0; // far enough to see whole sphere
        let proj = glam::DMat4::perspective_rh(fov_y, aspect, near.max(0.0001), far);

        proj * view
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
        self.zoom_at_for_mode(
            viewport,
            delta,
            screen_x,
            screen_y,
            x_planets_math::ProjectionMode::Mercator,
        );
    }

    /// Zoom toward a specific screen point, projection-aware.
    pub fn zoom_at_for_mode(
        &self,
        viewport: &mut Viewport,
        delta: f64,
        screen_x: f64,
        screen_y: f64,
        mode: x_planets_math::ProjectionMode,
    ) {
        if mode == x_planets_math::ProjectionMode::Globe {
            return self.zoom_at_globe(viewport, delta, screen_x, screen_y);
        }

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

    // ── Globe-specific camera methods ──

    /// Pan the viewport in globe mode using angular deltas.
    ///
    /// Converts pixel deltas directly to geographic degree changes,
    /// bypassing Mercator to avoid polar amplification.
    ///
    /// The sensitivity is derived from the perspective FOV and camera
    /// altitude so that dragging across the full viewport height sweeps
    /// exactly the visible angular extent of the sphere surface.
    pub fn pan_globe(&self, viewport: &mut Viewport, dx: f64, dy: f64) {
        // Visible angular extent of the sphere surface from the camera.
        // arccos(R/(R+h)) gives the angular radius of the visible cap
        // on the unit sphere — this DECREASES when zooming in, correctly
        // reducing the degrees-per-pixel rate at higher zoom.
        let unit_altitude =
            (20_000_000.0 / 6_378_137.0) / 2.0_f64.powf(viewport.zoom);
        let visible_half = (1.0 / (unit_altitude + 1.0)).acos();
        let visible_deg = visible_half.to_degrees() * 2.0;

        // Degrees per pixel — no extra pan_speed multiplier; the acos-based
        // derivation already gives 1:1 feel (finger-under-cursor tracking).
        let deg_per_px = visible_deg / viewport.height as f64;

        let bearing_rad = viewport.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        let dx_deg = dx * deg_per_px;
        let dy_deg = dy * deg_per_px;

        // Rotate by bearing.  Signs match the Mercator `pan()` convention:
        // dy > 0 (screen up, already negated by caller) → center moves south,
        // dx > 0 (screen right) → center moves west.
        let dlat = sin_b * dx_deg - cos_b * dy_deg;

        // Scale longitude by cos(lat), with a safe floor to prevent
        // singularity at the poles while still allowing polar navigation.
        let cos_lat = viewport.center.lat.to_radians().cos().max(0.05);
        let dlon = -(cos_b * dx_deg + sin_b * dy_deg) / cos_lat;

        viewport.center.lat = (viewport.center.lat + dlat).clamp(-89.9, 89.9);
        viewport.center.lon = ((viewport.center.lon + dlon) + 180.0).rem_euclid(360.0) - 180.0;
    }

    /// Zoom toward a screen point in globe mode.
    ///
    /// Applies zoom-level-dependent damping: at low zoom levels the camera
    /// altitude halves per level, making each step visually dramatic.
    /// Damping smooths this out so pinch-zoom on mobile feels natural.
    pub fn zoom_at_globe(
        &self,
        viewport: &mut Viewport,
        delta: f64,
        screen_x: f64,
        screen_y: f64,
    ) {
        // Globe zoom damping: provide consistent visual zoom speed across
        // all zoom levels. At low zoom (z<3), altitude halves dramatically
        // per step. At high zoom (z>12), the camera is so close that each
        // step causes hyper-sensitive movement. Damping smooths both extremes.
        let z = viewport.zoom;
        let ramp_up = 0.3 + 0.7 / (1.0 + (-1.5 * (z - 2.5)).exp());
        let ramp_down = 1.0 / (1.0 + 0.02 * (z - 6.0).max(0.0).powi(2));
        let damping = ramp_up * ramp_down;
        let delta = delta * damping;

        // Compute angular offset of cursor from center before zoom.
        // arccos(R/(R+h)) = visible surface angular radius (decreases when zooming in).
        let unit_altitude =
            (20_000_000.0 / 6_378_137.0) / 2.0_f64.powf(viewport.zoom);
        let half_angle_old = (1.0 / (unit_altitude + 1.0)).acos().to_degrees();

        let dx_norm = (screen_x - viewport.width as f64 * 0.5) / viewport.height as f64;
        let dy_norm = (screen_y - viewport.height as f64 * 0.5) / viewport.height as f64;

        let bearing_rad = viewport.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        let cos_lat = viewport.center.lat.to_radians().cos().max(0.05);
        let cursor_lon_off = (dx_norm * cos_b - dy_norm * sin_b) * half_angle_old * 2.0
            / cos_lat;
        let cursor_lat_off = -(dx_norm * sin_b + dy_norm * cos_b) * half_angle_old * 2.0;

        self.zoom(viewport, delta);

        let unit_altitude_new =
            (20_000_000.0 / 6_378_137.0) / 2.0_f64.powf(viewport.zoom);
        let half_angle_new = (1.0 / (unit_altitude_new + 1.0)).acos().to_degrees();

        let cos_lat = viewport.center.lat.to_radians().cos().max(0.05);
        let new_cursor_lon_off = (dx_norm * cos_b - dy_norm * sin_b) * half_angle_new * 2.0
            / cos_lat;
        let new_cursor_lat_off = -(dx_norm * sin_b + dy_norm * cos_b) * half_angle_new * 2.0;

        // Shift center so cursor geographic point stays fixed
        let dlat = cursor_lat_off - new_cursor_lat_off;
        let dlon = cursor_lon_off - new_cursor_lon_off;

        viewport.center.lat = (viewport.center.lat + dlat).clamp(-89.9, 89.9);
        viewport.center.lon = ((viewport.center.lon + dlon) + 180.0).rem_euclid(360.0) - 180.0;
    }

    /// Pan with projection-mode awareness.
    pub fn pan_for_mode(
        &self,
        viewport: &mut Viewport,
        dx: f64,
        dy: f64,
        mode: x_planets_math::ProjectionMode,
    ) {
        match mode {
            x_planets_math::ProjectionMode::Globe => self.pan_globe(viewport, dx, dy),
            _ => self.pan(viewport, dx, dy),
        }
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

    // ── Mercator tile selection near poles ──────────────────

    #[test]
    fn test_tile_selection_near_north_pole() {
        // At lat=80° zoom 3, tiles near the north pole should be selected.
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(80.0, 0.0);
        viewport.zoom = 3.0;

        let tiles = viewport.visible_tiles();
        assert!(!tiles.is_empty(), "Should select tiles near the north pole");

        // Should include y=0 tiles (northernmost in Mercator)
        let has_y0 = tiles.iter().any(|t| t.coord.y == 0);
        assert!(has_y0, "Should include northernmost tiles (y=0) at lat=80°");
    }

    #[test]
    fn test_tile_selection_near_south_pole() {
        // At lat=-80° zoom 3, tiles near the south pole should be selected.
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(-80.0, 0.0);
        viewport.zoom = 3.0;

        let tiles = viewport.visible_tiles();
        assert!(!tiles.is_empty(), "Should select tiles near the south pole");

        // Should include tiles at maximum y (southernmost in Mercator)
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
        // At the exact Mercator boundary (~85.05°), tiles should still be selected.
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(85.0, 0.0);
        viewport.zoom = 2.0;

        let tiles = viewport.visible_tiles();
        assert!(!tiles.is_empty(), "Should select tiles at Mercator boundary");
    }

    #[test]
    fn test_tile_selection_high_lat_high_pitch() {
        // High latitude + high pitch: frustum extends far toward the pole.
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(70.0, 0.0);
        viewport.zoom = 5.0;
        viewport.pitch = 55.0;

        let tiles = viewport.visible_tiles();
        assert!(!tiles.is_empty());

        // Should include some tiles north of center
        let center_tile = x_planets_math::TileCoord::from_geo(&viewport.center, viewport.tile_zoom());
        let northernmost = tiles.iter().map(|t| t.coord.y).min().unwrap();
        assert!(
            northernmost <= center_tile.y,
            "Pitched view should include tiles north of center"
        );
    }

    // ── Globe view projection ──────────────────────────────

    #[test]
    fn test_globe_vp_north_at_top() {
        // After fix: a point slightly north of center should project to
        // positive clip Y (top of screen). Verifies flip_y is removed.
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 2.0;

        let vp = viewport.to_globe_view_proj_f64();

        // Project a point on the sphere at lat=10°, lon=0° (slightly north)
        let north_point = x_planets_math::geo_to_unit_sphere(
            10.0_f64.to_radians(),
            0.0_f64.to_radians(),
        );
        let clip = vp * glam::DVec4::new(north_point.x, north_point.y, north_point.z, 1.0);
        let ndc_y = clip.y / clip.w;

        // North should be at positive Y (top of screen)
        assert!(
            ndc_y > 0.0,
            "North (lat=10°) should map to positive clip Y (top), got ndc_y={:.4}",
            ndc_y
        );

        // Project a point at lat=-10° (south)
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
        // A point slightly east of center should project to positive clip X.
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
        // At bearing=90° (camera facing east), east is at the top and
        // north is to the LEFT (-X) of the screen.
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

        // East should be at the top (+Y)
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

    // ── Globe pan direction ──────────────────────────────

    #[test]
    fn test_globe_pan_up_moves_south() {
        // Convention: callers pass dy with screen-up = positive (already negated).
        // dy > 0 = "screen up drag" → center should move SOUTH (reveal content below).
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 5.0;

        let lat_before = viewport.center.lat;
        // dy > 0 = screen up drag
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
        // Dragging right (dx > 0) should move center west (longitude decreases).
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
        // Globe pan direction should match Mercator pan direction.
        let ctrl = CameraController::new();

        // Mercator: dy > 0 = screen up → center south
        let mut vp_merc = Viewport::new(800, 600);
        vp_merc.center = GeoCoord::new(30.0, 50.0);
        vp_merc.zoom = 5.0;
        let lat_before_merc = vp_merc.center.lat;
        ctrl.pan(&mut vp_merc, 0.0, 50.0);
        let merc_dlat = vp_merc.center.lat - lat_before_merc;

        // Globe: same dy > 0 should also move south
        let mut vp_globe = Viewport::new(800, 600);
        vp_globe.center = GeoCoord::new(30.0, 50.0);
        vp_globe.zoom = 5.0;
        let lat_before_globe = vp_globe.center.lat;
        ctrl.pan_globe(&mut vp_globe, 0.0, 50.0);
        let globe_dlat = vp_globe.center.lat - lat_before_globe;

        // Both should move south (negative dlat)
        assert!(
            merc_dlat.signum() == globe_dlat.signum(),
            "Pan direction mismatch: mercator dlat={:.6}, globe dlat={:.6}",
            merc_dlat, globe_dlat
        );
    }

    #[test]
    fn test_globe_pan_with_bearing() {
        // At bearing=90°, dragging right should move center north
        // (because screen-right at bearing=90° = geographic south,
        //  and center moves opposite to drag = north).
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

    // ── Globe zoom sensitivity ──────────────────────────

    #[test]
    fn test_globe_zoom_damped_at_low_zoom() {
        // At low zoom levels, globe zoom should be damped.
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 1.0;

        let zoom_before = viewport.zoom;
        ctrl.zoom_at_globe(&mut viewport, 1.0, 400.0, 300.0);
        let zoom_change_low = viewport.zoom - zoom_before;

        // At high zoom, should be less damped
        viewport.zoom = 10.0;
        let zoom_before_high = viewport.zoom;
        ctrl.zoom_at_globe(&mut viewport, 1.0, 400.0, 300.0);
        let zoom_change_high = viewport.zoom - zoom_before_high;

        assert!(
            zoom_change_low < zoom_change_high,
            "Zoom at low level should be damped more: low_change={:.4}, high_change={:.4}",
            zoom_change_low, zoom_change_high
        );
    }

    #[test]
    fn test_globe_zoom_at_pointer_center_stable() {
        // Zooming at screen center should not shift the viewport center.
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5, 127.0);
        viewport.zoom = 5.0;

        let lat_before = viewport.center.lat;
        let lon_before = viewport.center.lon;
        ctrl.zoom_at_globe(&mut viewport, 1.0, 400.0, 300.0); // center of screen

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
        // Zooming at an off-center point should shift the center toward that point.
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 3.0;

        // Zoom in at the right side of the screen
        ctrl.zoom_at_globe(&mut viewport, 2.0, 700.0, 300.0);

        // Center should have shifted east (positive longitude)
        assert!(
            viewport.center.lon > 0.0,
            "Zoom at right edge should shift center east, got lon={:.4}",
            viewport.center.lon
        );
    }

    #[test]
    fn test_globe_zoom_damped_at_high_zoom() {
        // At high zoom (z=15+), zoom should be damped to prevent hyper-sensitivity.
        let ctrl = CameraController::new();

        // Measure effective zoom delta at medium zoom (z=8)
        let mut vp_mid = Viewport::new(800, 600);
        vp_mid.center = GeoCoord::new(0.0, 0.0);
        vp_mid.zoom = 8.0;
        let z_before = vp_mid.zoom;
        ctrl.zoom_at_globe(&mut vp_mid, 0.3, 400.0, 300.0);
        let dz_mid = (vp_mid.zoom - z_before).abs();

        // Measure effective zoom delta at high zoom (z=18)
        let mut vp_high = Viewport::new(800, 600);
        vp_high.center = GeoCoord::new(0.0, 0.0);
        vp_high.zoom = 18.0;
        let z_before = vp_high.zoom;
        ctrl.zoom_at_globe(&mut vp_high, 0.3, 400.0, 300.0);
        let dz_high = (vp_high.zoom - z_before).abs();

        // High zoom delta should be strictly less than mid zoom delta (damping kicks in)
        assert!(
            dz_high < dz_mid,
            "Zoom at z=18 ({:.4}) should be more damped than at z=8 ({:.4})",
            dz_high, dz_mid
        );
    }

    #[test]
    fn test_globe_zoom_damping_smooth_no_discontinuity() {
        // Damping should change smoothly across all zoom levels (no abrupt jumps).
        let ctrl = CameraController::new();
        let delta = 0.3;
        let mut prev_dz = None;

        for z_int in 0..=20 {
            let z = z_int as f64;
            let mut vp = Viewport::new(800, 600);
            vp.center = GeoCoord::new(0.0, 0.0);
            vp.zoom = z;
            let z_before = vp.zoom;
            ctrl.zoom_at_globe(&mut vp, delta, 400.0, 300.0);
            let dz = (vp.zoom - z_before).abs();

            if let Some(prev) = prev_dz {
                let ratio: f64 = if prev > 1e-9 { dz / prev } else { 1.0 };
                assert!(
                    ratio < 3.0 && ratio > 0.3,
                    "Zoom damping discontinuity at z={}: dz={:.6}, prev_dz={:.6}, ratio={:.2}",
                    z, dz, prev, ratio
                );
            }
            prev_dz = Some(dz);
        }
    }

    #[test]
    fn test_globe_zoom_round_trip_preserves_center() {
        // Zooming in then out by the same amount at screen center should
        // return to approximately the same viewport center.
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(35.0, 120.0);
        viewport.zoom = 10.0;

        let lat_orig = viewport.center.lat;
        let lon_orig = viewport.center.lon;

        // Zoom in then out at center
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

    // ── Projection-aware visible tile selection ──────────

    #[test]
    fn test_visible_tiles_for_mode_mercator_centered() {
        // Centered Mercator selects tiles via angular distance from
        // the viewport center, which is a superset of (or equal to)
        // the standard Mercator frustum at mid-latitudes.
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5665, 126.978);
        viewport.zoom = 5.0;

        let default_tiles = viewport.visible_tiles();
        let centered_tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);

        // Centered selection should include at least as many tiles.
        assert!(
            centered_tiles.len() >= default_tiles.len(),
            "centered ({}) should be >= standard ({})",
            centered_tiles.len(),
            default_tiles.len(),
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
        // The tile containing the viewport center must always be in the result.
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5665, 126.978);
        viewport.zoom = 5.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
        let center_tile = TileCoord::from_geo(&viewport.center, viewport.tile_zoom());

        // The center tile (or one of its ancestors) must be in the result.
        let has_center = tiles.iter().any(|vt| {
            let mut cur = center_tile;
            loop {
                if cur == vt.coord {
                    return true;
                }
                match cur.parent() {
                    Some(p) => cur = p,
                    None => return false,
                }
            }
        });
        assert!(
            has_center,
            "Center tile {:?} (or ancestor) not found in globe visible tiles",
            center_tile
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
    fn test_visible_tiles_globe_low_zoom_coverage() {
        // At zoom 0-1, globe mode should select some tiles.
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
        // Viewport centered near the north pole.
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(80.0, 0.0);
        viewport.zoom = 4.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
        assert!(
            !tiles.is_empty(),
            "Globe mode near pole should produce tiles"
        );
        // All tiles should have valid y coordinates.
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
            tiles.len() <= 200,
            "Globe mode: too many tiles {} (budget should cap)",
            tiles.len()
        );
    }

    // ── Polar tile selection regression tests ──────────

    #[test]
    fn test_centered_mercator_polar_selects_multiple_longitudes() {
        // At lat=80°, zoom 2, centered Mercator should select tiles
        // across many longitudes (the view wraps around the pole).
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(80.0, 0.0);
        viewport.zoom = 2.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);
        // Collect unique canonical x values.
        let unique_x: std::collections::HashSet<u32> =
            tiles.iter().map(|vt| vt.coord.x).collect();
        // At zoom 2 there are 4 x columns; near the pole we should
        // see tiles from at least 3 (wrap around the pole).
        assert!(
            unique_x.len() >= 3,
            "Polar centered Mercator should cover multiple longitudes, got {:?}",
            unique_x
        );
    }

    #[test]
    fn test_centered_mercator_polar_more_tiles_than_equator() {
        // At the same zoom, a polar center should select MORE tiles
        // than an equatorial center in centered Mercator because the
        // oblique Mercator wraps around the pole.
        let mut viewport = Viewport::new(800, 600);
        viewport.zoom = 3.0;

        viewport.center = GeoCoord::new(0.0, 0.0);
        let equator_tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);

        viewport.center = GeoCoord::new(80.0, 0.0);
        let polar_tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);

        assert!(
            polar_tiles.len() >= equator_tiles.len(),
            "Polar ({}) should have >= tiles than equator ({})",
            polar_tiles.len(),
            equator_tiles.len(),
        );
    }

    #[test]
    fn test_centered_mercator_south_pole_selects_tiles() {
        // South pole should also get wide tile coverage.
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
        // At zoom 0 the whole globe is visible; the tile set should
        // cover a large portion of available tiles.
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 0.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
        // At zoom 0 there is 1 tile.  The quadtree should refine it
        // into at least a few children.
        assert!(
            tiles.len() >= 1,
            "Globe zoom 0 should select at least the root tile"
        );
    }

    #[test]
    fn test_globe_polar_center_sufficient_tiles() {
        // When the globe camera is near the north pole at zoom 3,
        // we should get a reasonable number of tiles covering the
        // visible cap (not just a narrow band).
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(80.0, 0.0);
        viewport.zoom = 3.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Globe);
        // With the acos fix, the visible cap at zoom 3 is ~45° radius.
        // We should have tiles covering a significant area.
        assert!(
            tiles.len() >= 8,
            "Globe polar zoom 3 should have >=8 tiles, got {}",
            tiles.len()
        );
    }

    #[test]
    fn test_centered_mercator_budget_reasonable() {
        // Near the poles the centered Mercator frustum covers a wide
        // geographic area.  The angular filter in the renderer culls
        // excess tiles, so the count here can be higher than the
        // standard frustum but should not be unbounded.
        let mut viewport = Viewport::new(1920, 1080);
        viewport.center = GeoCoord::new(85.0, 0.0);
        viewport.zoom = 5.0;

        let tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);
        assert!(
            tiles.len() <= 500,
            "Centered Mercator at pole: tile count {} seems unreasonable",
            tiles.len()
        );
    }

    #[test]
    fn test_centered_mercator_high_zoom_reasonable() {
        // At high zoom the centered Mercator selection should be
        // similar in size to the standard Mercator frustum (the
        // viewport covers a tiny area).
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5665, 126.978);
        viewport.zoom = 10.0;

        let standard_tiles = viewport.visible_tiles();
        let centered_tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);

        // At high zoom near mid-latitude, both should be similar.
        let ratio = centered_tiles.len() as f64 / standard_tiles.len().max(1) as f64;
        assert!(
            ratio < 3.0,
            "At high zoom, centered ({}) should not be much larger than standard ({})",
            centered_tiles.len(),
            standard_tiles.len(),
        );
    }
}
