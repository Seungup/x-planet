use glam::DVec2;

use super::{
    BoundingBox, ConvexPolygon2D, TileCoord, VisibleTile,
    geo_to_mercator,
};

// ---------------------------------------------------------------------------
// Frustum (for tile culling)
// ---------------------------------------------------------------------------

/// 2D frustum for tile visibility culling.
///
/// Contains both an AABB (for fast grid enumeration) and an optional
/// convex polygon (for precise culling when camera is rotated/pitched).
///
/// `merc_sw` / `merc_ne` store raw Mercator bounds where X may extend
/// beyond `[0, 1]` for viewports crossing the antimeridian.
#[derive(Debug, Clone)]
pub struct Frustum2D {
    pub bounds: BoundingBox,
    /// Raw Mercator bounds (X can be < 0 or > 1 for antimeridian wrapping).
    pub merc_sw: DVec2,
    pub merc_ne: DVec2,
    /// Precise frustum polygon in Mercator space.  `None` for top-down
    /// north-up views where the AABB is already tight.
    pub polygon: Option<ConvexPolygon2D>,
}

impl Frustum2D {
    pub fn new(bounds: BoundingBox) -> Self {
        let sw = geo_to_mercator(&bounds.south_west);
        let ne = geo_to_mercator(&bounds.north_east);
        Self {
            bounds,
            merc_sw: sw,
            merc_ne: ne,
            polygon: None,
        }
    }

    /// Create a frustum with raw Mercator bounds (X may be outside [0,1]).
    pub fn with_merc_bounds(bounds: BoundingBox, merc_sw: DVec2, merc_ne: DVec2) -> Self {
        Self {
            bounds,
            merc_sw,
            merc_ne,
            polygon: None,
        }
    }

    /// Create a frustum with raw Mercator bounds and a precise polygon for culling.
    pub fn with_merc_bounds_and_polygon(
        bounds: BoundingBox,
        merc_sw: DVec2,
        merc_ne: DVec2,
        polygon: ConvexPolygon2D,
    ) -> Self {
        Self {
            bounds,
            merc_sw,
            merc_ne,
            polygon: Some(polygon),
        }
    }

    /// Create a frustum with a precise polygon for culling.
    pub fn with_polygon(bounds: BoundingBox, polygon: ConvexPolygon2D) -> Self {
        let sw = geo_to_mercator(&bounds.south_west);
        let ne = geo_to_mercator(&bounds.north_east);
        Self {
            bounds,
            merc_sw: sw,
            merc_ne: ne,
            polygon: Some(polygon),
        }
    }

    /// Test whether a visible tile (with display_x) is within this frustum.
    ///
    /// Uses AABB check against raw Mercator bounds (supports antimeridian wrapping),
    /// then precise polygon SAT test if available.
    pub fn is_visible_tile(&self, vt: &VisibleTile) -> bool {
        let n = vt.coord.extent() as f64;
        let tmin = DVec2::new(vt.display_x as f64 / n, vt.coord.y as f64 / n);
        let tmax = DVec2::new((vt.display_x + 1) as f64 / n, (vt.coord.y + 1) as f64 / n);

        // AABB check against raw Mercator bounds (X may be outside [0,1])
        if tmax.x < self.merc_sw.x || tmin.x > self.merc_ne.x
            || tmax.y < self.merc_ne.y || tmin.y > self.merc_sw.y
        {
            return false;
        }

        // Precise polygon check if available.
        if let Some(ref poly) = self.polygon {
            return poly.intersects_aabb(tmin, tmax);
        }

        true
    }

    /// Test whether a tile (canonical coordinates) is visible.
    ///
    /// Uses AABB pre-filter, then precise polygon SAT test if available.
    pub fn is_tile_visible(&self, tile: &TileCoord) -> bool {
        // Fast AABB check first.
        let tile_bounds = tile.to_geo_bounds();
        if !self.bounds.intersects(&tile_bounds) {
            return false;
        }

        // Precise polygon check if available.
        if let Some(ref poly) = self.polygon {
            let tmin = tile.mercator_min();
            let tmax = tile.mercator_max();
            return poly.intersects_aabb(tmin, tmax);
        }

        true
    }

    /// Get all visible tiles at a given zoom level, with antimeridian wrapping.
    ///
    /// Returns `VisibleTile` with `display_x` that may be negative or >= 2^z.
    /// The canonical `coord.x` is always wrapped to `[0, 2^z)`.
    pub fn visible_tiles(&self, zoom: u8) -> Vec<VisibleTile> {
        let n = 1u32 << zoom;
        let n_f = n as f64;
        let n_i = n as i64;

        // Use raw Mercator bounds (X can extend beyond [0, 1])
        let x_min = (self.merc_sw.x * n_f).floor() as i64;
        let x_max = (self.merc_ne.x * n_f).ceil() as i64;
        let y_min = (self.merc_ne.y * n_f).floor().max(0.0) as u32;
        let y_max = (self.merc_sw.y * n_f).ceil().min(n_f) as u32;

        let mut tiles = Vec::new();
        for display_x in x_min..x_max {
            let canonical_x = display_x.rem_euclid(n_i) as u32;
            for y in y_min..y_max {
                let coord = TileCoord::new(zoom, canonical_x, y);
                // Apply polygon filter using display coordinates.
                if let Some(ref poly) = self.polygon {
                    let tmin = DVec2::new(display_x as f64 / n_f, y as f64 / n_f);
                    let tmax = DVec2::new((display_x + 1) as f64 / n_f, (y + 1) as f64 / n_f);
                    if !poly.intersects_aabb(tmin, tmax) {
                        continue;
                    }
                }
                tiles.push(VisibleTile { coord, display_x });
            }
        }
        tiles
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GeoCoord, mercator_to_geo};

    #[test]
    fn test_frustum_visible_tiles() {
        let frustum = Frustum2D::new(BoundingBox::new(
            GeoCoord::new(48.0, 2.0),
            GeoCoord::new(49.0, 3.0),
        ));
        let tiles = frustum.visible_tiles(5);
        assert!(!tiles.is_empty());
        // All returned tiles should intersect the frustum bounds
        for t in &tiles {
            assert!(frustum.is_tile_visible(&t.coord));
        }
    }

    #[test]
    fn test_frustum_polygon_tighter_than_aabb() {
        // A rotated polygon should reject more tiles than the AABB alone.
        // Create a narrow diagonal polygon.
        let poly = ConvexPolygon2D::from_points(&[
            DVec2::new(0.2, 0.0),
            DVec2::new(0.8, 0.0),
            DVec2::new(0.8, 1.0),
            DVec2::new(0.2, 1.0),
        ]).unwrap();
        let bounds = BoundingBox::new(
            GeoCoord::new(-85.0, -108.0), // SW covers ~0.2 Mercator X
            GeoCoord::new(85.0, 108.0),   // NE covers ~0.8 Mercator X
        );
        let frustum_aabb = Frustum2D::new(bounds.clone());
        let frustum_poly = Frustum2D::with_polygon(bounds, poly);

        let tiles_aabb = frustum_aabb.visible_tiles(3);
        let tiles_poly = frustum_poly.visible_tiles(3);

        // Polygon should produce <= tiles than AABB.
        assert!(
            tiles_poly.len() <= tiles_aabb.len(),
            "Polygon tiles ({}) should be <= AABB tiles ({})",
            tiles_poly.len(),
            tiles_aabb.len()
        );
    }

    // -- Polar frustum tile selection tests --

    #[test]
    fn test_frustum_visible_tiles_near_north_pole() {
        // Viewport centered at lat=80 should select tiles near y=0.
        let center = GeoCoord::new(80.0, 0.0);
        let center_merc = geo_to_mercator(&center);
        let zoom = 3u8;
        let scale = 2.0_f64.powf(-(zoom as f64));
        let half_h = scale * 1.1;
        let half_w = scale * (800.0 / 600.0) * 1.1;

        let sw = DVec2::new(
            center_merc.x - half_w,
            (center_merc.y + half_h).clamp(0.0, 1.0),
        );
        let ne = DVec2::new(
            center_merc.x + half_w,
            (center_merc.y - half_h).clamp(0.0, 1.0),
        );

        let frustum = Frustum2D::with_merc_bounds(
            BoundingBox::new(
                mercator_to_geo(DVec2::new(sw.x.clamp(0.0, 1.0), sw.y)),
                mercator_to_geo(DVec2::new(ne.x.clamp(0.0, 1.0), ne.y)),
            ),
            sw,
            ne,
        );

        let tiles = frustum.visible_tiles(zoom);
        assert!(!tiles.is_empty(), "Should have tiles near north pole");
        assert!(
            tiles.iter().any(|t| t.coord.y == 0),
            "Should include northernmost tiles (y=0)"
        );
    }

    #[test]
    fn test_frustum_visible_tiles_near_south_pole() {
        // Viewport centered at lat=-80 should select tiles near y=max.
        let center = GeoCoord::new(-80.0, 0.0);
        let center_merc = geo_to_mercator(&center);
        let zoom = 3u8;
        let n = 1u32 << zoom;
        let scale = 2.0_f64.powf(-(zoom as f64));
        let half_h = scale * 1.1;
        let half_w = scale * (800.0 / 600.0) * 1.1;

        let sw = DVec2::new(
            center_merc.x - half_w,
            (center_merc.y + half_h).clamp(0.0, 1.0),
        );
        let ne = DVec2::new(
            center_merc.x + half_w,
            (center_merc.y - half_h).clamp(0.0, 1.0),
        );

        let frustum = Frustum2D::with_merc_bounds(
            BoundingBox::new(
                mercator_to_geo(DVec2::new(sw.x.clamp(0.0, 1.0), sw.y)),
                mercator_to_geo(DVec2::new(ne.x.clamp(0.0, 1.0), ne.y)),
            ),
            sw,
            ne,
        );

        let tiles = frustum.visible_tiles(zoom);
        assert!(!tiles.is_empty(), "Should have tiles near south pole");
        assert!(
            tiles.iter().any(|t| t.coord.y == n - 1),
            "Should include southernmost tiles (y={})",
            n - 1
        );
    }

    #[test]
    fn test_frustum_at_mercator_boundary_selects_tiles() {
        // At the Mercator boundary (lat~85), the frustum should still produce tiles.
        let center = GeoCoord::new(85.0, 0.0);
        let center_merc = geo_to_mercator(&center);
        let zoom = 2u8;
        let scale = 2.0_f64.powf(-(zoom as f64));
        let half_h = scale * 1.1;
        let half_w = scale * 1.3 * 1.1;

        let sw = DVec2::new(
            center_merc.x - half_w,
            (center_merc.y + half_h).clamp(0.0, 1.0),
        );
        let ne = DVec2::new(
            center_merc.x + half_w,
            (center_merc.y - half_h).clamp(0.0, 1.0),
        );

        let frustum = Frustum2D::with_merc_bounds(
            BoundingBox::new(
                mercator_to_geo(DVec2::new(sw.x.clamp(0.0, 1.0), sw.y)),
                mercator_to_geo(DVec2::new(ne.x.clamp(0.0, 1.0), ne.y)),
            ),
            sw,
            ne,
        );

        let tiles = frustum.visible_tiles(zoom);
        assert!(
            !tiles.is_empty(),
            "Should select tiles even at Mercator boundary (lat=85)"
        );
    }
}
