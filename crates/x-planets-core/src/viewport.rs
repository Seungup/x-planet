//! Viewport and camera controller for map navigation.

use x_planets_math::{
    geo_to_mercator, mercator_to_geo, BoundingBox, ConvexPolygon2D, Frustum2D, GeoCoord,
    ViewportUniforms, VisibleTile,
};

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
fn globe_unit_altitude(zoom: f64) -> f64 {
    (20_000_000.0 / 6_378_137.0) / 2.0_f64.powf(zoom)
}

/// Effective visible half-angle for globe tile selection and interaction.
///
/// At low zoom the sphere's **horizon** limits visibility (cap formula).
/// At high zoom the camera is close to the surface and the surface appears
/// flat, so the **camera FOV** limits visibility instead.  Taking the
/// minimum gives the correct visible extent at every zoom level.
///
/// Without this, the cap formula overestimates the visible area by up to
/// 100×+ at high zoom, causing tiles to be selected at far too coarse a
/// level and pan/zoom-to-point to overshoot dramatically.
fn globe_visible_half_angle(unit_altitude: f64) -> f64 {
    // Cap: angular radius from sub-satellite point to horizon on unit sphere.
    let cap_half = (1.0 / (unit_altitude + 1.0)).acos();
    // FOV: angular extent on a flat surface at distance `unit_altitude`,
    // using the same 60° vertical FOV as `to_globe_view_proj_f64`.
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

        // Cap at the oblique Mercator singularity guard threshold.
        let threshold_deg: f64 = crate::pipeline::centered_angular_threshold_deg(self.zoom);
        let visible_deg = viewport_angular_deg.min(threshold_deg);

        let lat = self.center.lat;
        let lon = self.center.lon;

        // Geographic bounding box covering the spherical cap.
        let lat_min = (lat - visible_deg).max(-89.9);
        let lat_max = (lat + visible_deg).min(89.9);
        // Longitude span must cover the widest parallel within the cap.
        // At the equator cos(lat)≈1 so lon_span≈visible_deg; near the
        // poles cos(lat)→0 so lon_span→180°.  Use the highest-latitude
        // edge of the cap (worst case for meridian convergence).
        let worst_lat = if lat_min.abs() > lat_max.abs() {
            lat_min.to_radians()
        } else {
            lat_max.to_radians()
        };
        let cos_worst = worst_lat.cos().max(0.01);
        let lon_span = (visible_deg / cos_worst).min(180.0);
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
        // Use centered mode: angular priority and z=0 seeding (like globe)
        // but no distance-based zoom reduction — centered Mercator projects
        // the spherical cap flat, so edge tiles need full resolution.
        self.quadtree_lod_with_frustum(base_z, &frustum, TileLodMode::Centered)
    }

    /// Globe-mode visible tile selection.
    ///
    /// Uses the orbital camera geometry (altitude above unit sphere) to
    /// determine the visible spherical cap, then converts it to Mercator
    /// tile coordinates for tile fetching.
    fn visible_tiles_globe(&self) -> Vec<VisibleTile> {
        let unit_altitude = globe_unit_altitude(self.zoom);
        let cap_half = (1.0 / (unit_altitude + 1.0)).acos();

        // For tile zoom selection, use a wider effective FOV (3× the
        // camera's actual 60° FOV).  The pure camera FOV underestimates
        // the usable area on the curved sphere surface, producing tiles
        // ~2 levels finer than viewport.zoom.  The 3× factor brings
        // globe_zoom ≈ viewport.zoom, which avoids excessive detail.
        let tile_fov_half =
            unit_altitude * (std::f64::consts::FRAC_PI_3 * 0.5).tan() * 3.0;
        let tile_half = cap_half.min(tile_fov_half);
        let visible_deg = tile_half.to_degrees() * 2.0;
        let tiles_needed = (self.height as f64 / 256.0).max(1.0);
        let tile_size_deg = visible_deg / tiles_needed;
        let globe_zoom = (360.0 / tile_size_deg).log2()
            .floor()
            .clamp(0.0, 22.0) as u8;
        let half_deg = cap_half.to_degrees().min(89.0);
        let lat = self.center.lat;
        let lon = self.center.lon;

        let lat_min = (lat - half_deg).max(-89.9);
        let lat_max = (lat + half_deg).min(89.9);
        // Longitude span must cover the widest parallel within the cap.
        // Use the highest-latitude edge (worst case for meridian convergence).
        let worst_lat = if lat_min.abs() > lat_max.abs() {
            lat_min.to_radians()
        } else {
            lat_max.to_radians()
        };
        let cos_worst = worst_lat.cos().max(0.01);
        let lon_span = (half_deg / cos_worst).min(180.0);
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
        self.quadtree_lod_with_frustum(globe_zoom, &frustum, TileLodMode::Globe)
    }

    /// Quadtree LOD with a custom frustum.
    ///
    /// `mode` controls LOD behaviour:
    /// - `Globe`: angular distance LOD with foreshortening (for 3D globe)
    /// - `Centered`: angular priority + z=0 seed but no distance-based zoom
    ///   reduction (for centered Mercator flat projection)
    /// - `Flat`: pitch-based perspective LOD with Mercator distance priority
    fn quadtree_lod_with_frustum(
        &self,
        base_z: u8,
        frustum: &Frustum2D,
        mode: TileLodMode,
    ) -> Vec<VisibleTile> {
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

        // Pre-compute globe geometry for distance-based LOD on the unit
        // sphere.  d² = 1 + (1+h)² − 2(1+h)cos(θ), nadir distance = h.
        let globe_h = globe_unit_altitude(self.zoom);
        let globe_r2 = 1.0 + (1.0 + globe_h).powi(2);
        let globe_2rh = 2.0 * (1.0 + globe_h);
        // Globe mode allows up to 3 zoom levels of LOD reduction for
        // distant/foreshortened tiles.  Centered Mercator uses pitch-based
        // drop only (edge tiles are not foreshortened in centered projection).
        let max_drop = match mode {
            TileLodMode::Globe => 3_u8,
            _ => ((self.pitch / 15.0).ceil() as u8).min(4),
        };
        let min_z = base_z.saturating_sub(max_drop);

        // Pre-compute center lat/lon in radians for haversine.
        let center_lat_rad = self.center.lat.to_radians();
        let center_lon_rad = self.center.lon.to_radians();
        let cos_center_lat = center_lat_rad.cos();

        let ideal_zoom_at = |mx: f64, my: f64| -> u8 {
            if mode == TileLodMode::Globe {
                // Screen-space size LOD on the unit sphere.
                // factor = sqrt(cos(θ)) × h / d, where:
                //   θ = central angle from nadir to tile
                //   h = camera altitude above unit sphere
                //   d = camera-to-tile distance (law of cosines)
                // h/d captures the reduced apparent size at distance.
                // sqrt(cos(θ)) captures foreshortening: a tile at angle θ
                // is compressed by cos(θ) in the radial direction but not
                // tangentially, so the geometric mean of the two dimensions
                // is sqrt(cos(θ)).  Using cos(θ) directly over-penalises
                // tiles near the visible edge, producing an abrupt LOD
                // boundary in the middle of the view.
                let tile_geo = mercator_to_geo(glam::DVec2::new(mx, my));
                let dlat = tile_geo.lat.to_radians() - center_lat_rad;
                let dlon = tile_geo.lon.to_radians() - center_lon_rad;
                let a = (dlat * 0.5).sin().powi(2)
                    + cos_center_lat
                        * tile_geo.lat.to_radians().cos()
                        * (dlon * 0.5).sin().powi(2);
                let theta = 2.0 * a.sqrt().asin(); // central angle
                let cos_theta = theta.cos();
                let d = (globe_r2 - globe_2rh * cos_theta).sqrt().max(globe_h);
                let factor = (cos_theta.sqrt() * globe_h / d).max(0.01);
                let zoom_adjust = factor.log2(); // ≤ 0
                // Use floor for hysteresis: tiles only drop a zoom level
                // when the adjustment crosses a full integer boundary.
                // round() oscillates at the −0.5 boundary during small
                // camera movements, causing tile flicker.  To compensate
                // for floor's aggressive rounding (even tiles adjacent to
                // center drop a level), add a +0.3 bias so tiles stay at
                // base_z until the adjustment exceeds −0.7.
                return (base_z as f64 + zoom_adjust + 0.3)
                    .floor()
                    .clamp(min_z as f64, base_z as f64) as u8;
            }

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

        // In globe/centered mode, compute priority using angular (great-circle)
        // distance instead of Mercator distance.  Mercator stretches
        // high-latitude tiles, biasing the priority queue so that tiles
        // toward the equator are processed first, exhausting the tile
        // budget before high-latitude tiles get subdivided.
        let use_angular = mode != TileLodMode::Flat;
        let tile_priority = |tc: glam::DVec2| -> f64 {
            if use_angular {
                let g = mercator_to_geo(tc);
                let dlat = g.lat.to_radians() - center_lat_rad;
                let dlon = g.lon.to_radians() - center_lon_rad;
                let a = (dlat * 0.5).sin().powi(2)
                    + cos_center_lat * g.lat.to_radians().cos() * (dlon * 0.5).sin().powi(2);
                let theta = 2.0 * a.sqrt().asin();
                1.0 / (theta + 1e-10)
            } else {
                let dist = (tc - center_merc).length();
                1.0 / (dist + 1e-10)
            }
        };

        // In globe/centered mode, start from z=0 so the quadtree can
        // naturally build the LOD gradient: tiles near the camera get
        // subdivided to base_z, while distant tiles stay coarse.  Starting
        // from min_z would produce seed tiles whose centers are already far
        // from the camera, preventing subdivision.
        let seed_z = if mode != TileLodMode::Flat { 0 } else { min_z };
        for vt in frustum.visible_tiles(seed_z) {
            let tc = vt.display_mercator_center();
            heap.push(Candidate {
                tile: vt,
                priority: tile_priority(tc),
            });
        }

        while let Some(candidate) = heap.pop() {
            let vt = candidate.tile;
            let tc = vt.display_mercator_center();
            let ideal_z = ideal_zoom_at(tc.x, tc.y);

            // In globe mode, force subdivision for the tile that contains
            // the viewport center.  Large low-zoom tiles have their centers
            // far from the viewport center on the sphere, so `ideal_z`
            // alone would prevent them from subdividing.  Only the single
            // tile containing the camera nadir is forced; its siblings use
            // the normal distance-based ideal_z.
            let contains_center = if mode != TileLodMode::Flat {
                let n = (1u32 << vt.coord.z) as f64;
                let tile_min_x = vt.coord.x as f64 / n;
                let tile_max_x = (vt.coord.x + 1) as f64 / n;
                let tile_min_y = vt.coord.y as f64 / n;
                let tile_max_y = (vt.coord.y + 1) as f64 / n;
                tile_min_x <= center_merc.x
                    && center_merc.x < tile_max_x
                    && tile_min_y <= center_merc.y
                    && center_merc.y < tile_max_y
            } else {
                false
            };

            let should_subdivide = (vt.coord.z < ideal_z || contains_center)
                && vt.coord.z < base_z
                && (result.len() + heap.len() + 4) <= TILE_BUDGET;

            if should_subdivide {
                for child in vt.children() {
                    if frustum.is_visible_tile(&child) {
                        let cc = child.display_mercator_center();
                        heap.push(Candidate {
                            tile: child,
                            priority: tile_priority(cc),
                        });
                    }
                }
            } else {
                result.push(vt);
            }
        }

        // Hard cap: if the initial tile set already exceeds the budget
        // (e.g. min_z == base_z with a large frustum), keep only the
        // tiles closest to the viewport center to prevent texture/cache
        // exhaustion.
        if result.len() > TILE_BUDGET {
            if use_angular {
                // Use angular distance for globe/centered modes to avoid
                // Mercator-distance bias at high latitudes.
                result.sort_by(|a, b| {
                    let ang_dist = |tc: glam::DVec2| -> f64 {
                        let g = mercator_to_geo(tc);
                        let dlat = g.lat.to_radians() - center_lat_rad;
                        let dlon = g.lon.to_radians() - center_lon_rad;
                        let a = (dlat * 0.5).sin().powi(2)
                            + cos_center_lat
                                * g.lat.to_radians().cos()
                                * (dlon * 0.5).sin().powi(2);
                        2.0 * a.sqrt().asin()
                    };
                    let da = ang_dist(a.display_mercator_center());
                    let db = ang_dist(b.display_mercator_center());
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
            } else {
                result.sort_by(|a, b| {
                    let da = (a.display_mercator_center() - center_merc).length();
                    let db = (b.display_mercator_center() - center_merc).length();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            result.truncate(TILE_BUDGET);
        }

        result.sort_by_key(|vt| vt.coord.z);
        result
    }

    /// Quadtree-based LOD tile selection (gap-free, priority-ordered, budgeted).
    fn quadtree_lod(&self, base_z: u8) -> Vec<VisibleTile> {
        let frustum = self.frustum();
        self.quadtree_lod_with_frustum(base_z, &frustum, TileLodMode::Flat)
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
        // Centered (oblique) Mercator: viewport center → (0.5, 0.5)
        let center = glam::DVec2::new(0.5, 0.5);
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
        // Adaptive near/far: tighten the ratio at low zoom to preserve
        // depth-buffer precision and prevent jitter.  At zoom 0 cam_h ≈ 1.73;
        // the old 0.005/10.0 gave a 2000:1 ratio which caused heavy shaking.
        let near = cam_h * 0.1;
        let far = cam_h * 4.0;
        let proj = glam::DMat4::perspective_rh(fov_y, aspect, near, far);

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
        // Accelerated descent at high zoom to match Mercator visible area.
        let unit_altitude = globe_unit_altitude(self.zoom);

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
        // At high zoom the camera is extremely close to the surface; use
        // adaptive near/far to preserve depth-buffer precision.
        let horizon_dist = (2.0 * unit_altitude).sqrt();
        let near = (unit_altitude * 0.1).max(1e-7);
        let far = (unit_altitude + horizon_dist) * 2.0 + 0.1;
        let proj = glam::DMat4::perspective_rh(fov_y, aspect, near, far);

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
        let near = cam_h * 0.1;
        let far = cam_h * 4.0;
        let proj = glam::Mat4::perspective_rh(fov_y, aspect, near, far);

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

    /// Zoom with projection-aware minimum.
    ///
    /// Centered (oblique) Mercator has a singularity at ~90° from the
    /// projection center, so zooming out beyond ~2 leaves large gaps.
    /// This enforces a higher minimum zoom for Mercator mode.
    pub fn zoom_for_mode(
        &self,
        viewport: &mut Viewport,
        delta: f64,
        mode: x_planets_math::ProjectionMode,
    ) {
        let effective_min = match mode {
            x_planets_math::ProjectionMode::Mercator => self.min_zoom.max(2.0),
            _ => self.min_zoom,
        };
        viewport.zoom = (viewport.zoom + delta * self.zoom_speed)
            .clamp(effective_min, self.max_zoom);
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

        self.zoom_for_mode(viewport, delta, mode);

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

        // Centered Mercator: shift center in Mercator space.
        let merc_dx = (dx_norm * aspect * cos_b - dy_norm * sin_b) * scale_diff;
        let merc_dy = (dx_norm * aspect * sin_b + dy_norm * cos_b) * scale_diff;

        let mut center_merc = geo_to_mercator(&viewport.center);
        center_merc.x = (center_merc.x + merc_dx).rem_euclid(1.0);
        center_merc.y = (center_merc.y + merc_dy).clamp(0.0, 1.0);
        viewport.center = mercator_to_geo(center_merc);
    }

    // ── Globe-specific camera methods ──

    /// Pan the viewport in globe mode using angular deltas.
    ///
    /// Uses the FOV-based visible extent so pan sensitivity matches the
    /// on-screen tile density.  This gives consistent drag-to-movement
    /// ratio across all zoom levels, similar to Mercator mode.
    pub fn pan_globe(&self, viewport: &mut Viewport, dx: f64, dy: f64) {
        let unit_altitude = globe_unit_altitude(viewport.zoom);
        let visible_half = globe_visible_half_angle(unit_altitude);
        let visible_deg = visible_half.to_degrees() * 2.0;

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
    /// Zoom toward a screen point in globe mode.
    ///
    /// No internal damping — the caller (animation system's `exp_decay`)
    /// already provides smooth interpolation.  Adding damping here would
    /// fight the animation, preventing the zoom level from advancing and
    /// causing tiles to appear stuck at high zoom.
    pub fn zoom_at_globe(
        &self,
        viewport: &mut Viewport,
        delta: f64,
        screen_x: f64,
        screen_y: f64,
    ) {
        // Compute geographic offset of cursor from center before zoom,
        // using the FOV-limited visible extent (not the full cap).
        let unit_altitude = globe_unit_altitude(viewport.zoom);
        let half_angle_old = globe_visible_half_angle(unit_altitude).to_degrees();

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

        let unit_altitude_new = globe_unit_altitude(viewport.zoom);
        let half_angle_new = globe_visible_half_angle(unit_altitude_new).to_degrees();

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
    /// Pan in centered Mercator mode using angular deltas.
    ///
    /// Standard Mercator pan() moves in Mercator coordinates where
    /// near-pole movement is extremely compressed, making drag feel
    /// frozen.  Centered Mercator re-projects around the viewport center,
    /// so angular (degree-based) panning matches the visual.
    pub fn pan_centered(&self, viewport: &mut Viewport, dx: f64, dy: f64) {
        // In centered Mercator the viewport maps the oblique Mercator
        // with the center at (0.5, 0.5).  The scale factor 2^(-zoom)
        // gives the Mercator-space extent visible; convert that to
        // angular extent for panning.
        let scale = 2.0_f64.powf(-viewport.zoom);
        // The viewport height spans `scale` Mercator units.
        // At center Y=0.5, 1 Mercator unit ≈ 360/π ≈ 114.6° near equator,
        // but we need the actual angular extent via inverse Mercator.
        let edge_y = (0.5 + scale * 0.5).min(0.9999);
        let edge_geo = mercator_to_geo(glam::DVec2::new(0.5, edge_y));
        let visible_deg = edge_geo.lat.abs() * 2.0;
        let deg_per_px = visible_deg / viewport.height as f64;

        let bearing_rad = viewport.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        let dx_deg = dx * deg_per_px * self.pan_speed;
        let dy_deg = dy * deg_per_px * self.pan_speed;

        let dlat = sin_b * dx_deg - cos_b * dy_deg;

        let cos_lat = viewport.center.lat.to_radians().cos().max(0.05);
        let dlon = -(cos_b * dx_deg + sin_b * dy_deg) / cos_lat;

        viewport.center.lat = (viewport.center.lat + dlat).clamp(-89.9, 89.9);
        viewport.center.lon = ((viewport.center.lon + dlon) + 180.0).rem_euclid(360.0) - 180.0;
    }

    pub fn pan_for_mode(
        &self,
        viewport: &mut Viewport,
        dx: f64,
        dy: f64,
        mode: x_planets_math::ProjectionMode,
    ) {
        match mode {
            x_planets_math::ProjectionMode::Globe => {
                self.pan_globe(viewport, dx, dy)
            }
            x_planets_math::ProjectionMode::Mercator => {
                self.pan_centered(viewport, dx, dy)
            }
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
            tiles.len() <= 150,
            "Too many tiles: {} (budget hard cap is 150)",
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
    fn test_globe_zoom_applies_full_delta() {
        // zoom_at_globe should apply the full delta without internal damping,
        // because the caller (animation system) handles smoothing.
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
    fn test_globe_zoom_full_delta_at_high_zoom() {
        // zoom_at_globe must apply the full delta even at high zoom levels
        // so that the animation system can reach its target.
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
        // Without internal damping, zoom_at_globe should apply the same
        // delta at all zoom levels (zoom change = delta, clamped at bounds).
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

            // Should be exactly delta unless clamped at max_zoom
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
        // the viewport center with globe-style LOD, so distant tiles
        // may get coarser zoom.  It should still return a reasonable
        // number of tiles covering the viewport.
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5665, 126.978);
        viewport.zoom = 5.0;

        let centered_tiles =
            viewport.visible_tiles_for_mode(x_planets_math::ProjectionMode::Mercator);

        assert!(
            !centered_tiles.is_empty(),
            "centered should produce tiles",
        );
        // At zoom 5, we expect a reasonable tile count.
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
        // Some tile covering the viewport center must always be in the result.
        // Globe mode may derive a different tile zoom than viewport.tile_zoom(),
        // so we check that at least one returned tile contains the center point.
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
        // In globe mode, tiles near the visible edge should use coarser
        // zoom levels than tiles near the center (angular-distance LOD).
        // At low zoom (large visible cap), the effect is most pronounced.
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
            tiles.len() <= 150,
            "Globe mode: too many tiles {} (budget hard cap is 150)",
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
            tiles.len() <= 150,
            "Centered Mercator at pole: tile count {} exceeds budget 150",
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

    // ───────────────────────────────────────────────────────────────
    // Regression: VP / mesh / dispatch consistency per projection mode
    // ───────────────────────────────────────────────────────────────

    /// Verify that Globe mode produces an orbital VP matrix (camera looks at
    /// the unit sphere, not at a 2D plane).  The orbital VP has the eye far
    /// from the origin at zoom 0, producing a view matrix with large
    /// translation components.
    #[test]
    fn test_globe_vp_is_orbital() {
        let mut vp = Viewport::new(640, 480);
        vp.center = GeoCoord::new(0.0, 0.0);
        vp.zoom = 0.0;
        let mat = vp.to_view_proj_f64_projected(x_planets_math::ProjectionMode::Globe);
        // The globe VP at zoom 0 should be the same as to_globe_view_proj_f64.
        let expected = vp.to_globe_view_proj_f64();
        for i in 0..16 {
            assert!(
                (mat.to_cols_array()[i] - expected.to_cols_array()[i]).abs() < 1e-10,
                "Globe VP must use orbital camera (element {i} differs)",
            );
        }
    }

    /// Verify that Mercator VP is NOT the same as Globe VP — they are
    /// fundamentally different camera models.
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

    /// Globe tile selection must produce tiles at every zoom level from 0..8.
    /// This guards against the bug where globe_zoom changed too slowly and
    /// tiles appeared stuck.
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
        // Tiles should change max zoom level at least 3 times across 9 zoom levels.
        assert!(
            changes >= 3,
            "Globe tiles must change zoom level as viewport zooms \
             (only changed {changes} times across 0..8)",
        );
    }

    /// Pan in Globe mode must move the center, and the displacement must
    /// scale with zoom (higher zoom → smaller displacement per pixel).
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


    // ───────────────────────────────────────────────────────────────
    // Regression: tile budget hard cap
    // ───────────────────────────────────────────────────────────────

    /// The quadtree LOD must never return more than TILE_BUDGET tiles,
    /// regardless of viewport size, zoom level, or pitch.
    /// This prevents texture/cache exhaustion that causes black tiles.
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

    /// Same budget test for Mercator mode.
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

    /// Pitched views should also respect the budget.
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

    /// Polar centers at various zoom levels should respect the budget.
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

    // ── CameraController pan tests ──────────────────────────

    #[test]
    fn test_pan_moves_center() {
        let ctrl = CameraController::new();
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(0.0, 0.0);
        viewport.zoom = 5.0;

        let lat_before = viewport.center.lat;
        let lon_before = viewport.center.lon;
        ctrl.pan(&mut viewport, 50.0, 50.0);

        // Center should have moved
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

        // Pan right to cross antimeridian
        ctrl.pan(&mut viewport, -500.0, 0.0);

        // Longitude should wrap (stay in valid range)
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

        // Pan way up past the Mercator limit
        ctrl.pan(&mut viewport, 0.0, -5000.0);

        // Latitude should be clamped (not go above ~85.05)
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

        // Zoom at exact screen center
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

        // Zoom in at the right edge of the screen
        ctrl.zoom_at(&mut viewport, 3.0, 750.0, 300.0);

        // Center should shift east (negative longitude in Mercator convention)
        // The exact direction depends on implementation, but center should have moved
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

        viewport.zoom = 25.0; // beyond max
        assert_eq!(viewport.tile_zoom(), 22);
    }

    #[test]
    fn test_pan_for_mode_delegates() {
        let ctrl = CameraController::new();

        // Mercator mode should use standard pan
        let mut vp = Viewport::new(800, 600);
        vp.center = GeoCoord::new(0.0, 0.0);
        vp.zoom = 5.0;
        let before = vp.center;
        ctrl.pan_for_mode(&mut vp, 50.0, 0.0, x_planets_math::ProjectionMode::Mercator);
        assert!(
            (vp.center.lon - before.lon).abs() > 1e-6,
            "pan_for_mode Mercator should pan"
        );

        // Globe mode should also pan
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
}
