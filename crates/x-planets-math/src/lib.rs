//! x-planets-math: Geospatial math utilities for the x-planets rendering engine.
//!
//! Provides geographic coordinate types, tile coordinate systems,
//! bounding box calculations, ECEF coordinate transforms, and projection-related math primitives.

pub mod ecef;

pub use glam::{DMat3, DMat4, DVec2, DVec3, Mat4, Vec2, Vec3, Vec4};

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Geographic Coordinate
// ---------------------------------------------------------------------------

/// A geographic coordinate in degrees (WGS84).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeoCoord {
    /// Latitude in degrees (-90..90)
    pub lat: f64,
    /// Longitude in degrees (-180..180)
    pub lon: f64,
}

impl GeoCoord {
    pub const fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// Convert to radians.
    pub fn to_radians(&self) -> DVec2 {
        DVec2::new(self.lat.to_radians(), self.lon.to_radians())
    }

    /// Create from radians.
    pub fn from_radians(lat_rad: f64, lon_rad: f64) -> Self {
        Self {
            lat: lat_rad.to_degrees(),
            lon: lon_rad.to_degrees(),
        }
    }

    /// Clamp latitude to valid range and normalize longitude.
    pub fn normalize(&self) -> Self {
        let lat = self.lat.clamp(-90.0, 90.0);
        let mut lon = self.lon % 360.0;
        if lon > 180.0 {
            lon -= 360.0;
        } else if lon < -180.0 {
            lon += 360.0;
        }
        Self { lat, lon }
    }
}

impl Default for GeoCoord {
    fn default() -> Self {
        Self { lat: 0.0, lon: 0.0 }
    }
}

// ---------------------------------------------------------------------------
// Tile Coordinate
// ---------------------------------------------------------------------------

/// A tile coordinate in the XYZ tile scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileCoord {
    /// Zoom level (0 = world in one tile)
    pub z: u8,
    /// Column index
    pub x: u32,
    /// Row index
    pub y: u32,
}

impl TileCoord {
    pub const fn new(z: u8, x: u32, y: u32) -> Self {
        Self { z, x, y }
    }

    /// Total number of tiles at this zoom level (per axis).
    pub fn extent(&self) -> u32 {
        1 << self.z
    }

    /// Return the parent tile (one zoom level up).
    pub fn parent(&self) -> Option<Self> {
        if self.z == 0 {
            return None;
        }
        Some(Self {
            z: self.z - 1,
            x: self.x / 2,
            y: self.y / 2,
        })
    }

    /// Return the four children tiles (one zoom level down).
    pub fn children(&self) -> [Self; 4] {
        let z = self.z + 1;
        let x = self.x * 2;
        let y = self.y * 2;
        [
            Self::new(z, x, y),
            Self::new(z, x + 1, y),
            Self::new(z, x, y + 1),
            Self::new(z, x + 1, y + 1),
        ]
    }

    /// Convert tile coordinate to geographic bounds (Web Mercator).
    pub fn to_geo_bounds(&self) -> BoundingBox {
        let n = self.extent() as f64;
        let lon_min = (self.x as f64) / n * 360.0 - 180.0;
        let lon_max = ((self.x + 1) as f64) / n * 360.0 - 180.0;

        let lat_max = tile_y_to_lat(self.y as f64, n);
        let lat_min = tile_y_to_lat((self.y + 1) as f64, n);

        BoundingBox {
            south_west: GeoCoord::new(lat_min, lon_min),
            north_east: GeoCoord::new(lat_max, lon_max),
        }
    }

    /// Center of the tile in Mercator normalized coordinates [0..1].
    pub fn mercator_center(&self) -> DVec2 {
        let n = self.extent() as f64;
        DVec2::new((self.x as f64 + 0.5) / n, (self.y as f64 + 0.5) / n)
    }

    /// Min corner (top-left) of the tile in Mercator normalized coordinates.
    pub fn mercator_min(&self) -> DVec2 {
        let n = self.extent() as f64;
        DVec2::new(self.x as f64 / n, self.y as f64 / n)
    }

    /// Max corner (bottom-right) of the tile in Mercator normalized coordinates.
    pub fn mercator_max(&self) -> DVec2 {
        let n = self.extent() as f64;
        DVec2::new((self.x + 1) as f64 / n, (self.y + 1) as f64 / n)
    }

    /// Create a TileCoord from a geographic position and zoom level (Web Mercator).
    pub fn from_geo(coord: &GeoCoord, zoom: u8) -> Self {
        let n = (1u32 << zoom) as f64;
        let x = ((coord.lon + 180.0) / 360.0 * n).floor() as u32;
        let lat_rad = coord.lat.to_radians();
        let y = ((1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / PI) / 2.0 * n).floor()
            as u32;
        Self {
            z: zoom,
            x: x.min((n as u32) - 1),
            y: y.min((n as u32) - 1),
        }
    }
}

impl std::fmt::Display for TileCoord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.z, self.x, self.y)
    }
}

fn tile_y_to_lat(y: f64, n: f64) -> f64 {
    let lat_rad = (PI * (1.0 - 2.0 * y / n)).sinh().atan();
    lat_rad.to_degrees()
}

// ---------------------------------------------------------------------------
// Bounding Box
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box in geographic coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub south_west: GeoCoord,
    pub north_east: GeoCoord,
}

impl BoundingBox {
    pub fn new(south_west: GeoCoord, north_east: GeoCoord) -> Self {
        Self {
            south_west,
            north_east,
        }
    }

    /// Check if a geographic coordinate falls within this bounding box.
    pub fn contains(&self, coord: &GeoCoord) -> bool {
        coord.lat >= self.south_west.lat
            && coord.lat <= self.north_east.lat
            && coord.lon >= self.south_west.lon
            && coord.lon <= self.north_east.lon
    }

    /// Check if this bounding box intersects another.
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.south_west.lat <= other.north_east.lat
            && self.north_east.lat >= other.south_west.lat
            && self.south_west.lon <= other.north_east.lon
            && self.north_east.lon >= other.south_west.lon
    }

    /// Get the center of the bounding box.
    pub fn center(&self) -> GeoCoord {
        GeoCoord {
            lat: (self.south_west.lat + self.north_east.lat) / 2.0,
            lon: (self.south_west.lon + self.north_east.lon) / 2.0,
        }
    }

    /// Width in degrees.
    pub fn width(&self) -> f64 {
        self.north_east.lon - self.south_west.lon
    }

    /// Height in degrees.
    pub fn height(&self) -> f64 {
        self.north_east.lat - self.south_west.lat
    }

    /// Whole world bounding box.
    pub const WORLD: Self = Self {
        south_west: GeoCoord::new(-85.0511, -180.0),
        north_east: GeoCoord::new(85.0511, 180.0),
    };
}

// ---------------------------------------------------------------------------
// GPU-friendly Uniforms
// ---------------------------------------------------------------------------

/// Viewport uniforms that get uploaded to the GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ViewportUniforms {
    /// View-projection matrix
    pub view_proj: [f32; 16],
    /// Viewport resolution (width, height, 1/width, 1/height)
    pub resolution: [f32; 4],
    /// Camera center in world coordinates (x, y, zoom, _padding)
    pub camera: [f32; 4],
}

/// Per-tile uniforms uploaded to GPU.
///
/// Includes a per-tile model-view-projection matrix computed in f64 on the
/// CPU.  This eliminates f32 jitter at high zoom levels by baking the
/// tile-center translation into the matrix while still in f64.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TileUniforms {
    /// Per-tile model-view-projection matrix.
    /// `mvp = VP_f64 * translate(tile_center_f64)`, then cast to f32.
    /// The shader multiplies this by the RTE vertex position directly.
    pub mvp: [f32; 16],
    /// Tile world-space bounds (min_x, min_y, max_x, max_y)
    pub bounds: [f32; 4],
    /// Tile metadata (zoom_level, opacity, _pad, _pad)
    pub meta: [f32; 4],
    /// UV sub-rectangle within the texture (u_min, v_min, u_max, v_max).
    /// Default is [0, 0, 1, 1] for full texture; sub-rects are used
    /// when a parent tile's texture is used as fallback.
    pub uv_rect: [f32; 4],
}

// ---------------------------------------------------------------------------
// Web Mercator helpers
// ---------------------------------------------------------------------------

/// Convert latitude/longitude to Web Mercator normalized coordinates (0..1).
pub fn geo_to_mercator(coord: &GeoCoord) -> DVec2 {
    let x = (coord.lon + 180.0) / 360.0;
    let lat_rad = coord.lat.to_radians();
    let y = (1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / PI) / 2.0;
    DVec2::new(x, y)
}

/// Convert Web Mercator normalized coordinates (0..1) to latitude/longitude.
pub fn mercator_to_geo(pos: DVec2) -> GeoCoord {
    let lon = pos.x * 360.0 - 180.0;
    let lat_rad = (PI * (1.0 - 2.0 * pos.y)).sinh().atan();
    GeoCoord::new(lat_rad.to_degrees(), lon)
}

// ---------------------------------------------------------------------------
// Convex Polygon 2D (for precise frustum culling)
// ---------------------------------------------------------------------------

/// A convex polygon in Mercator [0,1]×[0,1] space.
///
/// Used for precise frustum-tile intersection tests via the Separating
/// Axis Theorem (SAT).  Much tighter than an AABB when the camera is
/// rotated (bearing ≠ 0).
#[derive(Debug, Clone)]
pub struct ConvexPolygon2D {
    /// Vertices in counter-clockwise order, in Mercator coordinates.
    pub vertices: Vec<DVec2>,
    /// Outward-facing edge normals (pre-computed for SAT).
    edge_normals: Vec<DVec2>,
}

impl ConvexPolygon2D {
    /// Create from a set of points.  Computes convex hull and edge normals.
    ///
    /// For a frustum quadrilateral the 4 ground-plane hit-points are
    /// already convex, but we sort them into CCW order anyway for safety.
    pub fn from_points(points: &[DVec2]) -> Option<Self> {
        if points.len() < 3 {
            return None;
        }

        // Convex hull via gift-wrapping (for 4–6 points this is fine).
        let hull = convex_hull_ccw(points);
        if hull.len() < 3 {
            return None;
        }

        let edge_normals = compute_edge_normals(&hull);

        Some(Self {
            vertices: hull,
            edge_normals,
        })
    }

    /// Test if this convex polygon intersects an axis-aligned bounding box.
    ///
    /// Uses SAT with the polygon's edge normals + the 2 AABB axes (X, Y).
    pub fn intersects_aabb(&self, min: DVec2, max: DVec2) -> bool {
        // AABB as 4 corners.
        let box_corners = [
            DVec2::new(min.x, min.y),
            DVec2::new(max.x, min.y),
            DVec2::new(max.x, max.y),
            DVec2::new(min.x, max.y),
        ];

        // Test polygon's edge normals.
        for normal in &self.edge_normals {
            let (poly_min, poly_max) = project_polygon(&self.vertices, normal);
            let (box_min, box_max) = project_polygon(&box_corners, normal);
            if poly_max < box_min || box_max < poly_min {
                return false; // Separating axis found.
            }
        }

        // Test AABB's 2 axes (X and Y).
        // X-axis: normal = (1, 0)
        {
            let (poly_min, poly_max) = project_x(&self.vertices);
            if poly_max < min.x || max.x < poly_min {
                return false;
            }
        }
        // Y-axis: normal = (0, 1)
        {
            let (poly_min, poly_max) = project_y(&self.vertices);
            if poly_max < min.y || max.y < poly_min {
                return false;
            }
        }

        true
    }
}

/// Project polygon vertices onto an axis and return (min, max).
fn project_polygon(verts: &[DVec2], axis: &DVec2) -> (f64, f64) {
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for v in verts {
        let d = v.dot(*axis);
        lo = lo.min(d);
        hi = hi.max(d);
    }
    (lo, hi)
}

/// Fast X-axis projection.
fn project_x(verts: &[DVec2]) -> (f64, f64) {
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for v in verts {
        lo = lo.min(v.x);
        hi = hi.max(v.x);
    }
    (lo, hi)
}

/// Fast Y-axis projection.
fn project_y(verts: &[DVec2]) -> (f64, f64) {
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for v in verts {
        lo = lo.min(v.y);
        hi = hi.max(v.y);
    }
    (lo, hi)
}

/// Compute outward-facing normals for a CCW polygon.
fn compute_edge_normals(verts: &[DVec2]) -> Vec<DVec2> {
    let n = verts.len();
    let mut normals = Vec::with_capacity(n);
    for i in 0..n {
        let j = (i + 1) % n;
        let edge = verts[j] - verts[i];
        // CCW polygon: outward normal is (edge.y, -edge.x)
        let normal = DVec2::new(edge.y, -edge.x);
        let len = normal.length();
        if len > 1e-12 {
            normals.push(normal / len);
        }
    }
    normals
}

/// Gift-wrapping convex hull, returns vertices in CCW order.
fn convex_hull_ccw(points: &[DVec2]) -> Vec<DVec2> {
    if points.len() < 3 {
        return points.to_vec();
    }

    // Find leftmost point.
    let mut start = 0;
    for (i, p) in points.iter().enumerate() {
        if p.x < points[start].x || (p.x == points[start].x && p.y < points[start].y) {
            start = i;
        }
    }

    let mut hull = Vec::new();
    let mut current = start;
    loop {
        hull.push(points[current]);
        let mut next = 0;
        for i in 0..points.len() {
            if i == current {
                continue;
            }
            if next == current {
                next = i;
                continue;
            }
            let cross = (points[i] - points[current]).perp_dot(points[next] - points[current]);
            if cross > 0.0
                || (cross == 0.0
                    && (points[i] - points[current]).length_squared()
                        > (points[next] - points[current]).length_squared())
            {
                next = i;
            }
        }
        current = next;
        if current == start {
            break;
        }
        if hull.len() > points.len() {
            break; // Safety.
        }
    }
    hull
}

// ---------------------------------------------------------------------------
// Frustum (for tile culling)
// ---------------------------------------------------------------------------

/// 2D frustum for tile visibility culling.
///
/// Contains both an AABB (for fast grid enumeration) and an optional
/// convex polygon (for precise culling when camera is rotated/pitched).
#[derive(Debug, Clone)]
pub struct Frustum2D {
    pub bounds: BoundingBox,
    /// Precise frustum polygon in Mercator space.  `None` for top-down
    /// north-up views where the AABB is already tight.
    pub polygon: Option<ConvexPolygon2D>,
}

impl Frustum2D {
    pub fn new(bounds: BoundingBox) -> Self {
        Self {
            bounds,
            polygon: None,
        }
    }

    /// Create a frustum with a precise polygon for culling.
    pub fn with_polygon(bounds: BoundingBox, polygon: ConvexPolygon2D) -> Self {
        Self {
            bounds,
            polygon: Some(polygon),
        }
    }

    /// Test whether a tile is visible within this frustum.
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

    /// Get all visible tiles at a given zoom level.
    pub fn visible_tiles(&self, zoom: u8) -> Vec<TileCoord> {
        let n = 1u32 << zoom;
        let sw = geo_to_mercator(&self.bounds.south_west);
        let ne = geo_to_mercator(&self.bounds.north_east);

        let x_min = (sw.x * n as f64).floor().max(0.0) as u32;
        let x_max = (ne.x * n as f64).ceil().min(n as f64) as u32;
        let y_min = (ne.y * n as f64).floor().max(0.0) as u32;
        let y_max = (sw.y * n as f64).ceil().min(n as f64) as u32;

        let mut tiles = Vec::new();
        for x in x_min..x_max {
            for y in y_min..y_max {
                let tile = TileCoord::new(zoom, x, y);
                // Apply polygon filter if available.
                if let Some(ref poly) = self.polygon {
                    let tmin = tile.mercator_min();
                    let tmax = tile.mercator_max();
                    if !poly.intersects_aabb(tmin, tmax) {
                        continue;
                    }
                }
                tiles.push(tile);
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

    #[test]
    fn test_geo_coord_normalize() {
        let coord = GeoCoord::new(100.0, 200.0);
        let normalized = coord.normalize();
        assert!((normalized.lat - 90.0).abs() < f64::EPSILON);
        assert!((normalized.lon - (-160.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tile_coord_parent_children() {
        let tile = TileCoord::new(2, 1, 1);
        let parent = tile.parent().unwrap();
        assert_eq!(parent, TileCoord::new(1, 0, 0));

        let children = parent.children();
        assert!(children.contains(&tile));
    }

    #[test]
    fn test_tile_coord_zero_has_no_parent() {
        let tile = TileCoord::new(0, 0, 0);
        assert!(tile.parent().is_none());
    }

    #[test]
    fn test_tile_geo_bounds_roundtrip() {
        let coord = GeoCoord::new(51.5074, -0.1278); // London
        let tile = TileCoord::from_geo(&coord, 10);
        let bounds = tile.to_geo_bounds();
        assert!(bounds.contains(&coord));
    }

    #[test]
    fn test_mercator_roundtrip() {
        let original = GeoCoord::new(48.8566, 2.3522); // Paris
        let mercator = geo_to_mercator(&original);
        let recovered = mercator_to_geo(mercator);
        assert!((original.lat - recovered.lat).abs() < 1e-10);
        assert!((original.lon - recovered.lon).abs() < 1e-10);
    }

    #[test]
    fn test_bounding_box_contains() {
        let bbox = BoundingBox::new(
            GeoCoord::new(30.0, -10.0),
            GeoCoord::new(60.0, 30.0),
        );
        assert!(bbox.contains(&GeoCoord::new(45.0, 10.0)));
        assert!(!bbox.contains(&GeoCoord::new(70.0, 10.0)));
    }

    #[test]
    fn test_bounding_box_intersects() {
        let a = BoundingBox::new(GeoCoord::new(0.0, 0.0), GeoCoord::new(10.0, 10.0));
        let b = BoundingBox::new(GeoCoord::new(5.0, 5.0), GeoCoord::new(15.0, 15.0));
        let c = BoundingBox::new(GeoCoord::new(20.0, 20.0), GeoCoord::new(30.0, 30.0));
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

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
            assert!(frustum.is_tile_visible(t));
        }
    }

    // ── ConvexPolygon2D / SAT tests ───────────────────────────

    #[test]
    fn test_polygon_sat_basic() {
        // A unit-square polygon [0,0]-[1,1] should intersect an AABB fully inside it.
        let poly = ConvexPolygon2D::from_points(&[
            DVec2::new(0.0, 0.0),
            DVec2::new(1.0, 0.0),
            DVec2::new(1.0, 1.0),
            DVec2::new(0.0, 1.0),
        ]).unwrap();
        // Fully inside.
        assert!(poly.intersects_aabb(DVec2::new(0.2, 0.2), DVec2::new(0.8, 0.8)));
        // Overlapping edge.
        assert!(poly.intersects_aabb(DVec2::new(0.5, 0.5), DVec2::new(1.5, 1.5)));
        // Completely outside.
        assert!(!poly.intersects_aabb(DVec2::new(2.0, 2.0), DVec2::new(3.0, 3.0)));
    }

    #[test]
    fn test_polygon_sat_rotated_diamond() {
        // Diamond shape: tilted 45 degrees.
        let poly = ConvexPolygon2D::from_points(&[
            DVec2::new(0.5, 0.0),
            DVec2::new(1.0, 0.5),
            DVec2::new(0.5, 1.0),
            DVec2::new(0.0, 0.5),
        ]).unwrap();
        // Inside the diamond.
        assert!(poly.intersects_aabb(DVec2::new(0.4, 0.4), DVec2::new(0.6, 0.6)));
        // Corner region: AABB overlaps the diamond's extent but not the actual polygon.
        // Box at top-left corner [0,0]-[0.1,0.1] — outside the diamond.
        assert!(!poly.intersects_aabb(DVec2::new(0.0, 0.0), DVec2::new(0.1, 0.1)));
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

    #[test]
    fn test_tile_coord_mercator_helpers() {
        let tile = TileCoord::new(1, 0, 0); // top-left quadrant
        let min = tile.mercator_min();
        let max = tile.mercator_max();
        let center = tile.mercator_center();

        assert!((min.x - 0.0).abs() < 1e-10);
        assert!((min.y - 0.0).abs() < 1e-10);
        assert!((max.x - 0.5).abs() < 1e-10);
        assert!((max.y - 0.5).abs() < 1e-10);
        assert!((center.x - 0.25).abs() < 1e-10);
        assert!((center.y - 0.25).abs() < 1e-10);
    }
}
