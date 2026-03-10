//! Tile selection algorithms: visible tiles, quadtree LOD, globe/centered selection.

use x_planets_math::{geo_to_mercator, mercator_to_geo, Frustum2D, GeoCoord, VisibleTile};

use super::TileLodMode;

impl super::Viewport {
    /// Select visible tiles for the current viewport using quadtree LOD.
    ///
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
    fn visible_tiles_centered(&self) -> Vec<VisibleTile> {
        let scale = 2.0_f64.powf(-self.zoom);
        let aspect = self.width as f64 / self.height as f64;

        let half_h = scale * 1.1;
        let half_w = scale * aspect * 1.1;

        let center_lat_rad = self.center.lat.to_radians();
        let center_lon_rad = self.center.lon.to_radians();
        let center_sphere = x_planets_math::geo_to_unit_sphere(center_lat_rad, center_lon_rad);
        let threshold_deg: f64 = crate::pipeline::centered_angular_threshold_deg(self.zoom);

        // Compute forward extent in oblique Mercator space.
        // The old method used flat camera geometry (cam_h * 20) which only
        // covered ~13° of angular distance.  Instead, compute the oblique
        // Mercator y-offset for the maximum visible angular distance.
        let forward_ext = if self.pitch >= 1.0 {
            let pitch_rad = self.pitch.to_radians();
            let fov_half = std::f64::consts::FRAC_PI_3 * 0.5;
            // Maximum angular distance from center the camera can see
            let max_visible_angle = (pitch_rad + fov_half).min(threshold_deg.to_radians());
            // Convert to oblique Mercator y-offset:
            // In oblique Mercator, y = 0.5 - atanh(sin(θ))/(2π)
            // so the offset from center (0.5) is atanh(sin(θ))/(2π)
            let sin_vis = max_visible_angle.sin().min(0.998);
            sin_vis.atanh() / (2.0 * std::f64::consts::PI)
        } else {
            0.0
        };

        // Use finer grid for pitched views (oblique Mercator is highly
        // nonlinear at large distances from center).
        let n_grid_y = if self.pitch >= 10.0 { 16_usize } else { 8_usize };
        let n_grid = 8_usize;
        let mut lat_min_deg = self.center.lat;
        let mut lat_max_deg = self.center.lat;
        let mut lon_min_deg = self.center.lon;
        let mut lon_max_deg = self.center.lon;
        let mut hit_singularity = false;

        let rect_left = 0.5 - half_w;
        let rect_right = 0.5 + half_w;
        let rect_top = 0.5 - half_h - forward_ext;
        let rect_bottom = 0.5 + half_h;

        for iy in 0..=n_grid_y {
            let ty = iy as f64 / n_grid_y as f64;
            let y = rect_top + (rect_bottom - rect_top) * ty;
            for ix in 0..=n_grid {
                let tx = ix as f64 / n_grid as f64;
                let x = rect_left + (rect_right - rect_left) * tx;
                let pt = glam::DVec2::new(x, y);
                let (lat_r, lon_r) = x_planets_math::oblique_mercator_inverse(
                    pt,
                    center_lat_rad,
                    center_lon_rad,
                );
                if !lat_r.is_finite() || !lon_r.is_finite() {
                    hit_singularity = true;
                    continue;
                }
                let pt_sphere = x_planets_math::geo_to_unit_sphere(lat_r, lon_r);
                let cos_angle = center_sphere.dot(pt_sphere).clamp(-1.0, 1.0);
                let angle_deg = cos_angle.acos().to_degrees();
                if angle_deg > threshold_deg {
                    continue;
                }
                let lat_d = lat_r.to_degrees();
                let lon_d = lon_r.to_degrees();
                if lat_d < lat_min_deg { lat_min_deg = lat_d; }
                if lat_d > lat_max_deg { lat_max_deg = lat_d; }
                if lon_d < lon_min_deg { lon_min_deg = lon_d; }
                if lon_d > lon_max_deg { lon_max_deg = lon_d; }
            }
        }

        if hit_singularity {
            let lat = self.center.lat;
            let lon = self.center.lon;
            lat_min_deg = (lat - threshold_deg).max(-89.9);
            lat_max_deg = (lat + threshold_deg).min(89.9);
            let cos_worst = lat_max_deg.abs().max(lat_min_deg.abs()).to_radians().cos().max(0.01);
            let lon_span = (threshold_deg / cos_worst).min(180.0);
            lon_min_deg = lon - lon_span;
            lon_max_deg = lon + lon_span;
        }

        let lat_range = lat_max_deg - lat_min_deg;
        let lon_range = lon_max_deg - lon_min_deg;
        lat_min_deg = (lat_min_deg - lat_range * 0.15).max(-89.9);
        lat_max_deg = (lat_max_deg + lat_range * 0.15).min(89.9);
        lon_min_deg -= lon_range * 0.15;
        lon_max_deg += lon_range * 0.15;

        let lat_min = lat_min_deg;
        let lat_max = lat_max_deg;
        let lon_min = lon_min_deg;
        let lon_max = lon_max_deg;

        let sw = geo_to_mercator(&GeoCoord::new(lat_min, lon_min));
        let ne = geo_to_mercator(&GeoCoord::new(lat_max, lon_max));
        let bbox = x_planets_math::BoundingBox::new(
            GeoCoord::new(lat_min, lon_min.clamp(-180.0, 180.0)),
            GeoCoord::new(lat_max, lon_max.clamp(-180.0, 180.0)),
        );
        let frustum = Frustum2D::with_merc_bounds(bbox, sw, ne);

        let base_z = self.tile_zoom();
        if base_z == 0 {
            return frustum.visible_tiles(0);
        }
        self.quadtree_lod_with_frustum(base_z, &frustum, TileLodMode::Centered)
    }

    /// Globe-mode visible tile selection.
    fn visible_tiles_globe(&self) -> Vec<VisibleTile> {
        let unit_altitude = super::globe_unit_altitude(self.zoom, &self.body);
        let cap_half = (1.0 / (unit_altitude + 1.0)).acos();

        let tile_fov_half =
            unit_altitude * (std::f64::consts::FRAC_PI_3 * 0.5).tan() * 3.0;
        let tile_half = cap_half.min(tile_fov_half);
        let visible_deg = tile_half.to_degrees() * 2.0;
        let tiles_needed = (self.height as f64 / 256.0).max(1.0);
        let tile_size_deg = visible_deg / tiles_needed;
        let globe_zoom = (360.0 / tile_size_deg).log2()
            .floor()
            .clamp(0.0, 22.0) as u8;

        // When pitched, the camera sees much further toward the horizon
        // in the forward (bearing) direction.  Extend the tile selection
        // toward the geometric horizon proportionally to pitch.
        let effective_half = if self.pitch >= 1.0 {
            let pitch_factor = (self.pitch / 60.0).clamp(0.0, 1.0);
            tile_half + (cap_half - tile_half) * pitch_factor
        } else {
            tile_half
        };
        let half_deg = effective_half.to_degrees().min(89.0);

        // For pitched views, offset the bounding box center in the bearing
        // direction so coverage favours the forward (horizon) side over the
        // area behind the camera.
        let (lat_offset, lon_offset) = if self.pitch >= 5.0 {
            let bearing_rad = self.bearing.to_radians();
            let pitch_factor = (self.pitch / 60.0).clamp(0.0, 1.0);
            let fwd_deg = (half_deg - tile_half.to_degrees()) * pitch_factor * 0.5;
            // Forward in bearing direction: north component + east component
            let dlat = fwd_deg * bearing_rad.cos();
            let cos_lat = self.center.lat.to_radians().cos().max(0.05);
            let dlon = fwd_deg * bearing_rad.sin() / cos_lat;
            (dlat, dlon)
        } else {
            (0.0, 0.0)
        };

        let lat = self.center.lat + lat_offset;
        let lon = self.center.lon + lon_offset;

        let lat_min = (lat - half_deg).max(-89.9);
        let lat_max = (lat + half_deg).min(89.9);
        let worst_lat = if lat_min.abs() > lat_max.abs() {
            lat_min.to_radians()
        } else {
            lat_max.to_radians()
        };
        let cos_worst = worst_lat.cos().max(0.01);
        let lon_span = (half_deg / cos_worst).min(180.0);
        let lon_min = lon - lon_span;
        let lon_max = lon + lon_span;

        let sw = geo_to_mercator(&GeoCoord::new(lat_min, lon_min));
        let ne = geo_to_mercator(&GeoCoord::new(lat_max, lon_max));

        let bbox = x_planets_math::BoundingBox::new(
            GeoCoord::new(lat_min, lon_min.clamp(-180.0, 180.0)),
            GeoCoord::new(lat_max, lon_max.clamp(-180.0, 180.0)),
        );
        let frustum = Frustum2D::with_merc_bounds(bbox, sw, ne);

        if globe_zoom == 0 {
            return frustum.visible_tiles(0);
        }

        self.quadtree_lod_with_frustum(globe_zoom, &frustum, TileLodMode::Globe)
    }

    /// Quadtree LOD with a custom frustum.
    fn quadtree_lod_with_frustum(
        &self,
        base_z: u8,
        frustum: &Frustum2D,
        mode: TileLodMode,
    ) -> Vec<VisibleTile> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        let tile_budget = self.tile_budget;

        let center_merc = geo_to_mercator(&self.center);
        let pitch_rad = self.pitch.to_radians();
        let sin_p = pitch_rad.sin();
        let scale = 2.0_f64.powf(-self.zoom);
        let fov_half_tan = (std::f64::consts::FRAC_PI_3 * 0.5).tan();
        let cam_h = scale / fov_half_tan;

        let bearing_rad = self.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        let globe_h = super::globe_unit_altitude(self.zoom, &self.body);
        let globe_r2 = 1.0 + (1.0 + globe_h).powi(2);
        let globe_2rh = 2.0 * (1.0 + globe_h);
        let max_drop = match mode {
            TileLodMode::Globe => {
                // At high pitch, the LOD range between near (fine) and far
                // (coarse) tiles is larger.  Allow up to 4 zoom-level drop.
                if self.pitch >= 30.0 { 4_u8 }
                else if self.pitch >= 10.0 { 3_u8 }
                else { 2_u8 }
            }
            _ => ((self.pitch / 15.0).ceil() as u8).min(4),
        };
        let min_z = base_z.saturating_sub(max_drop);

        let center_lat_rad = self.center.lat.to_radians();
        let center_lon_rad = self.center.lon.to_radians();
        let cos_center_lat = center_lat_rad.cos();

        let ideal_zoom_at = |mx: f64, my: f64| -> u8 {
            if mode == TileLodMode::Globe {
                let tile_geo = mercator_to_geo(glam::DVec2::new(mx, my));
                let dlat = tile_geo.lat.to_radians() - center_lat_rad;
                let dlon = tile_geo.lon.to_radians() - center_lon_rad;
                let a = (dlat * 0.5).sin().powi(2)
                    + cos_center_lat
                        * tile_geo.lat.to_radians().cos()
                        * (dlon * 0.5).sin().powi(2);
                let theta = 2.0 * a.sqrt().asin();
                let cos_theta = theta.cos();
                let d = (globe_r2 - globe_2rh * cos_theta).sqrt().max(globe_h);
                let factor = (cos_theta.sqrt() * globe_h / d).max(0.01);
                let zoom_adjust = factor.log2();
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
                // Geometric perspective factor: ratio of apparent tile size
                // at distance d_fwd vs directly below the camera.
                // The 3D distance from eye to a ground point at forward
                // offset d_fwd is sqrt(cam_h² + (d_fwd * sin_p)²).
                let eye_z = cam_h * pitch_rad.cos();
                let d_3d = (eye_z * eye_z + d_fwd * d_fwd).sqrt();
                let perspective = cam_h / d_3d;
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

            // For pitched views, allow exploration beyond the final budget
            // so the post-loop truncation can pick the best distribution.
            // A tighter check starves fine tiles near the camera when the
            // frustum covers a large area.
            let explore_budget = if self.pitch >= 10.0 {
                tile_budget + tile_budget / 2
            } else {
                tile_budget
            };
            let should_subdivide = (vt.coord.z < ideal_z || contains_center)
                && vt.coord.z < base_z
                && (result.len() + heap.len() + 4) <= explore_budget;

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

        if result.len() > tile_budget {
            // Separate coarse background tiles from fine foreground tiles.
            // Coarse tiles (below the ideal base zoom) are few and provide
            // essential coverage for distant areas in pitched views.  Always
            // keep them; only truncate fine tiles when the budget is exceeded.
            // Without this, pitched views at high zoom drop distant coarse
            // tiles, leaving visible dark gaps at the horizon.
            let coarse_threshold = min_z.saturating_add(1);
            let (coarse, mut fine): (Vec<_>, Vec<_>) = result
                .into_iter()
                .partition(|vt| vt.coord.z <= coarse_threshold);

            if use_angular {
                fine.sort_by(|a, b| {
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
                fine.sort_by(|a, b| {
                    let da = (a.display_mercator_center() - center_merc).length();
                    let db = (b.display_mercator_center() - center_merc).length();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            let fine_budget = tile_budget.saturating_sub(coarse.len());
            fine.truncate(fine_budget);

            result = coarse;
            result.extend(fine);

            // Hard cap: if coarse tiles alone exceed the budget (can happen
            // at polar latitudes or extreme views), truncate to budget.
            if result.len() > tile_budget {
                result.truncate(tile_budget);
            }
        }

        result.sort_by_key(|vt| vt.coord.z);
        result
    }

    /// Quadtree-based LOD tile selection (gap-free, priority-ordered, budgeted).
    fn quadtree_lod(&self, base_z: u8) -> Vec<VisibleTile> {
        let frustum = self.frustum();
        self.quadtree_lod_with_frustum(base_z, &frustum, TileLodMode::Flat)
    }
}
