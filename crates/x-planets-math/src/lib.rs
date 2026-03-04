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

    /// Clamp this tile to `max_zoom` by computing its ancestor at that zoom level.
    ///
    /// Returns `self` unchanged if already at or below `max_zoom`.
    pub fn clamp_to_zoom(&self, max_zoom: u8) -> Self {
        if self.z <= max_zoom {
            return *self;
        }
        let dz = self.z - max_zoom;
        Self::new(max_zoom, self.x >> dz, self.y >> dz)
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

// ---------------------------------------------------------------------------
// Visible Tile (canonical coord + unwrapped display X for antimeridian)
// ---------------------------------------------------------------------------

/// A tile visible in the viewport with unwrapped display coordinate.
///
/// `coord` uses canonical x in `[0, 2^z)` — for cache, fetch, and texture lookup.
/// `display_x` is the unwrapped x position in tile units — used for MVP computation
/// so tiles render at the correct screen position across the antimeridian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisibleTile {
    pub coord: TileCoord,
    pub display_x: i64,
}

impl VisibleTile {
    /// Create a VisibleTile with no wrapping offset (display_x == coord.x).
    pub fn canonical(coord: TileCoord) -> Self {
        Self {
            display_x: coord.x as i64,
            coord,
        }
    }

    /// Mercator center using the unwrapped display_x for correct screen placement.
    pub fn display_mercator_center(&self) -> DVec2 {
        let n = self.coord.extent() as f64;
        DVec2::new(
            (self.display_x as f64 + 0.5) / n,
            (self.coord.y as f64 + 0.5) / n,
        )
    }

    /// Return the four children tiles, propagating the display_x offset.
    pub fn children(&self) -> [VisibleTile; 4] {
        let z = self.coord.z + 1;
        let n = (1u32 << z) as i64;
        let dx = self.display_x * 2;
        [
            VisibleTile {
                coord: TileCoord::new(z, (dx).rem_euclid(n) as u32, self.coord.y * 2),
                display_x: dx,
            },
            VisibleTile {
                coord: TileCoord::new(z, (dx + 1).rem_euclid(n) as u32, self.coord.y * 2),
                display_x: dx + 1,
            },
            VisibleTile {
                coord: TileCoord::new(z, (dx).rem_euclid(n) as u32, self.coord.y * 2 + 1),
                display_x: dx,
            },
            VisibleTile {
                coord: TileCoord::new(z, (dx + 1).rem_euclid(n) as u32, self.coord.y * 2 + 1),
                display_x: dx + 1,
            },
        ]
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
    /// Small-circle clipping: (center_x, center_y, center_z, cos_clip_angle)
    /// on the unit sphere.  The fragment shader discards pixels where
    /// dot(sphere_pos, clip_center.xyz) < clip_center.w.
    pub clip_sphere: [f32; 4],
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

/// Convert a Mercator normalized y [0,1] to latitude in radians.
pub fn mercator_y_to_lat_rad(y: f64) -> f64 {
    (PI * (1.0 - 2.0 * y)).sinh().atan()
}

// ---------------------------------------------------------------------------
// Projection mode
// ---------------------------------------------------------------------------

/// Which map projection to use for tile positioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProjectionMode {
    /// Web Mercator (conformal, area-distorting at poles).
    #[default]
    Mercator,
    /// 3D Globe — tiles rendered on a unit sphere surface.
    Globe,
}

/// Convert geographic coordinates (radians) to a point on the unit sphere.
///
/// Returns `(cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat))`.
pub fn geo_to_unit_sphere(lat_rad: f64, lon_rad: f64) -> DVec3 {
    DVec3::new(
        lat_rad.cos() * lon_rad.cos(),
        lat_rad.cos() * lon_rad.sin(),
        lat_rad.sin(),
    )
}

/// Oblique Mercator projection centered on a given reference point.
///
/// Rotates the sphere so `(center_lat_rad, center_lon_rad)` maps to the
/// equator/prime-meridian, then applies standard Web Mercator.
/// This minimizes distortion near the viewport center.
///
/// Returns coordinates in [0, 1] × [0, 1] just like standard Mercator,
/// but centered on the given point instead of (0°, 0°).
pub fn oblique_mercator(
    lat_rad: f64,
    lon_rad: f64,
    center_lat_rad: f64,
    center_lon_rad: f64,
) -> DVec2 {
    // 1. Convert to 3D unit sphere
    let p = geo_to_unit_sphere(lat_rad, lon_rad);

    // 2. Rotate by -center_lon around Z axis (align center longitude to prime meridian)
    let sin_clon = center_lon_rad.sin();
    let cos_clon = center_lon_rad.cos();
    let rx = p.x * cos_clon + p.y * sin_clon;
    let ry = -p.x * sin_clon + p.y * cos_clon;
    let rz = p.z;

    // 3. Rotate by -center_lat around Y axis (align center latitude to equator)
    let sin_clat = center_lat_rad.sin();
    let cos_clat = center_lat_rad.cos();
    let fx = rx * cos_clat + rz * sin_clat;
    let fy = ry;
    let fz = -rx * sin_clat + rz * cos_clat;

    // 4. Convert back to lat/lon in the rotated frame
    let rot_lat = fz.asin();
    let rot_lon = fy.atan2(fx);

    // 5. Standard Mercator of the rotated coordinates
    let x = (rot_lon + PI) / (2.0 * PI);
    let y = (1.0 - (rot_lat.tan() + 1.0 / rot_lat.cos()).ln() / PI) / 2.0;
    DVec2::new(x, y)
}

/// Inverse of [`oblique_mercator`]: convert centered Mercator back to geographic (radians).
pub fn oblique_mercator_inverse(
    merc: DVec2,
    center_lat_rad: f64,
    center_lon_rad: f64,
) -> (f64, f64) {
    // 1. Inverse standard Mercator → rotated lat/lon
    let rot_lon = merc.x * 2.0 * PI - PI;
    let rot_lat = (PI * (1.0 - 2.0 * merc.y)).sinh().atan();

    // 2. Convert to 3D
    let fx = rot_lat.cos() * rot_lon.cos();
    let fy = rot_lat.cos() * rot_lon.sin();
    let fz = rot_lat.sin();

    // 3. Inverse latitude rotation (+center_lat around Y)
    let sin_clat = center_lat_rad.sin();
    let cos_clat = center_lat_rad.cos();
    let rx = fx * cos_clat - fz * sin_clat;
    let ry = fy;
    let rz = fx * sin_clat + fz * cos_clat;

    // 4. Inverse longitude rotation (+center_lon around Z)
    let sin_clon = center_lon_rad.sin();
    let cos_clon = center_lon_rad.cos();
    let px = rx * cos_clon - ry * sin_clon;
    let py = rx * sin_clon + ry * cos_clon;
    let pz = rz;

    // 5. Convert back to lat/lon
    let lat_rad = pz.asin();
    let lon_rad = py.atan2(px);
    (lat_rad, lon_rad)
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
            assert!(frustum.is_tile_visible(&t.coord));
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

    // ── Polar frustum tile selection tests ─────────────────

    #[test]
    fn test_frustum_visible_tiles_near_north_pole() {
        // Viewport centered at lat=80° should select tiles near y=0.
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
        // Viewport centered at lat=-80° should select tiles near y=max.
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
        // At the Mercator boundary (lat≈85°), the frustum should still produce tiles.
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
            "Should select tiles even at Mercator boundary (lat=85°)"
        );
    }

    #[test]
    fn test_geo_to_unit_sphere_poles() {
        // North pole should be at (0, 0, 1)
        let north = geo_to_unit_sphere(std::f64::consts::FRAC_PI_2, 0.0);
        assert!((north.x).abs() < 1e-10);
        assert!((north.y).abs() < 1e-10);
        assert!((north.z - 1.0).abs() < 1e-10);

        // South pole should be at (0, 0, -1)
        let south = geo_to_unit_sphere(-std::f64::consts::FRAC_PI_2, 0.0);
        assert!((south.x).abs() < 1e-10);
        assert!((south.y).abs() < 1e-10);
        assert!((south.z + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_geo_to_unit_sphere_equator() {
        // Equator, prime meridian → (1, 0, 0)
        let point = geo_to_unit_sphere(0.0, 0.0);
        assert!((point.x - 1.0).abs() < 1e-10);
        assert!((point.y).abs() < 1e-10);
        assert!((point.z).abs() < 1e-10);

        // Equator, 90°E → (0, 1, 0)
        let east = geo_to_unit_sphere(0.0, std::f64::consts::FRAC_PI_2);
        assert!((east.x).abs() < 1e-10);
        assert!((east.y - 1.0).abs() < 1e-10);
        assert!((east.z).abs() < 1e-10);
    }

    #[test]
    fn test_mercator_roundtrip_near_poles() {
        // Mercator roundtrip should work near the boundary latitude.
        for &lat in &[80.0, -80.0, 84.0, -84.0, 85.0, -85.0] {
            let original = GeoCoord::new(lat, 30.0);
            let merc = geo_to_mercator(&original);
            let recovered = mercator_to_geo(merc);
            assert!(
                (original.lat - recovered.lat).abs() < 1e-6,
                "Roundtrip failed at lat={}: got {:.6}",
                lat, recovered.lat
            );
            assert!(
                (original.lon - recovered.lon).abs() < 1e-6,
                "Roundtrip failed at lat={}: lon {:.6} != {:.6}",
                lat, original.lon, recovered.lon
            );
        }
    }

    #[test]
    fn test_mercator_y_range_near_poles() {
        // Mercator y should be within [0, 1] for valid latitudes
        let north = geo_to_mercator(&GeoCoord::new(85.0, 0.0));
        let south = geo_to_mercator(&GeoCoord::new(-85.0, 0.0));
        let equator = geo_to_mercator(&GeoCoord::new(0.0, 0.0));

        assert!(north.y > 0.0 && north.y < 0.1, "North pole merc.y={:.4}", north.y);
        assert!(south.y > 0.9 && south.y < 1.0, "South pole merc.y={:.4}", south.y);
        assert!((equator.y - 0.5).abs() < 1e-10, "Equator merc.y={:.4}", equator.y);
    }

    // ── Oblique Mercator tests ──────────────────────────────

    #[test]
    fn test_oblique_mercator_center_maps_to_half() {
        // The center point should map to (0.5, 0.5) in oblique Mercator.
        let center_lat = 37.5_f64.to_radians();
        let center_lon = 127.0_f64.to_radians();
        let result = oblique_mercator(center_lat, center_lon, center_lat, center_lon);
        assert!((result.x - 0.5).abs() < 1e-10, "x={}", result.x);
        assert!((result.y - 0.5).abs() < 1e-10, "y={}", result.y);
    }

    #[test]
    fn test_oblique_mercator_roundtrip() {
        // Forward + inverse should recover the original point.
        let centers = [
            (37.5_f64, 127.0_f64),   // Seoul
            (40.7_f64, -74.0_f64),   // New York
            (0.0_f64, 0.0_f64),      // Equator/prime meridian
            (-33.9_f64, 18.4_f64),   // Cape Town
            (78.0_f64, 15.6_f64),    // Svalbard (high latitude)
        ];
        for (clat, clon) in &centers {
            let clat_r = clat.to_radians();
            let clon_r = clon.to_radians();
            // Test a point offset from center
            let lat_r = (clat + 5.0).to_radians();
            let lon_r = (clon + 5.0).to_radians();
            let merc = oblique_mercator(lat_r, lon_r, clat_r, clon_r);
            let (rlat, rlon) = oblique_mercator_inverse(merc, clat_r, clon_r);
            assert!(
                (lat_r - rlat).abs() < 1e-8,
                "lat roundtrip failed for center ({}, {}): {} vs {}",
                clat, clon, lat_r, rlat
            );
            assert!(
                (lon_r - rlon).abs() < 1e-8,
                "lon roundtrip failed for center ({}, {}): {} vs {}",
                clat, clon, lon_r, rlon
            );
        }
    }

    #[test]
    fn test_oblique_mercator_equator_center_matches_standard() {
        // When centered at (0,0), oblique Mercator should match standard Mercator.
        let lat = 48.8566_f64.to_radians(); // Paris
        let lon = 2.3522_f64.to_radians();
        let oblique = oblique_mercator(lat, lon, 0.0, 0.0);
        let standard = geo_to_mercator(&GeoCoord::new(48.8566, 2.3522));
        assert!(
            (oblique.x - standard.x).abs() < 1e-8,
            "x: {} vs {}", oblique.x, standard.x
        );
        assert!(
            (oblique.y - standard.y).abs() < 1e-8,
            "y: {} vs {}", oblique.y, standard.y
        );
    }

    // ── VisibleTile tests ───────────────────────────────────

    #[test]
    fn test_visible_tile_canonical() {
        let coord = TileCoord::new(3, 5, 2);
        let vt = VisibleTile::canonical(coord);
        assert_eq!(vt.coord, coord);
        assert_eq!(vt.display_x, 5);
    }

    #[test]
    fn test_visible_tile_display_mercator_center() {
        let vt = VisibleTile::canonical(TileCoord::new(1, 0, 0));
        let center = vt.display_mercator_center();
        assert!((center.x - 0.25).abs() < 1e-10);
        assert!((center.y - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_visible_tile_children() {
        let vt = VisibleTile::canonical(TileCoord::new(1, 0, 0));
        let children = vt.children();
        assert_eq!(children.len(), 4);
        // Children should be at zoom 2
        for child in &children {
            assert_eq!(child.coord.z, 2);
        }
        // display_x should be 0 and 1 (parent display_x=0 → children 0,1)
        assert_eq!(children[0].display_x, 0);
        assert_eq!(children[1].display_x, 1);
        assert_eq!(children[2].display_x, 0);
        assert_eq!(children[3].display_x, 1);
    }

    #[test]
    fn test_visible_tile_wrapped_negative_display_x() {
        // A tile with negative display_x (across antimeridian)
        let vt = VisibleTile {
            coord: TileCoord::new(1, 1, 0), // canonical x=1
            display_x: -1,                   // displayed at x=-1 (wrapped)
        };
        let center = vt.display_mercator_center();
        // display_x=-1, n=2: (-1+0.5)/2 = -0.25
        assert!((center.x - (-0.25)).abs() < 1e-10);

        let children = vt.children();
        // display_x=-1 → children at display_x=-2 and -1
        assert_eq!(children[0].display_x, -2);
        assert_eq!(children[1].display_x, -1);
        // Canonical coord.x wraps via rem_euclid: (-2).rem_euclid(4)=2, (-1).rem_euclid(4)=3
        assert_eq!(children[0].coord.x, 2);
        assert_eq!(children[1].coord.x, 3);
    }

    // ── GeoCoord additional tests ───────────────────────────

    #[test]
    fn test_geo_coord_to_from_radians() {
        let coord = GeoCoord::new(45.0, 90.0);
        let rad = coord.to_radians();
        let recovered = GeoCoord::from_radians(rad.x, rad.y);
        assert!((coord.lat - recovered.lat).abs() < 1e-10);
        assert!((coord.lon - recovered.lon).abs() < 1e-10);
    }

    #[test]
    fn test_geo_coord_default() {
        let coord = GeoCoord::default();
        assert_eq!(coord.lat, 0.0);
        assert_eq!(coord.lon, 0.0);
    }

    #[test]
    fn test_geo_coord_normalize_boundary() {
        // Exact boundary values
        let coord = GeoCoord::new(90.0, 180.0);
        let n = coord.normalize();
        assert_eq!(n.lat, 90.0);
        assert_eq!(n.lon, 180.0);

        let coord = GeoCoord::new(-90.0, -180.0);
        let n = coord.normalize();
        assert_eq!(n.lat, -90.0);
        assert_eq!(n.lon, -180.0);
    }

    // ── BoundingBox additional tests ────────────────────────

    #[test]
    fn test_bounding_box_center() {
        let bbox = BoundingBox::new(
            GeoCoord::new(10.0, 20.0),
            GeoCoord::new(30.0, 40.0),
        );
        let c = bbox.center();
        assert!((c.lat - 20.0).abs() < 1e-10);
        assert!((c.lon - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounding_box_width_height() {
        let bbox = BoundingBox::new(
            GeoCoord::new(10.0, 20.0),
            GeoCoord::new(30.0, 50.0),
        );
        assert!((bbox.width() - 30.0).abs() < 1e-10);
        assert!((bbox.height() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounding_box_world_constant() {
        let w = BoundingBox::WORLD;
        assert!(w.contains(&GeoCoord::new(0.0, 0.0)));
        assert!(w.contains(&GeoCoord::new(85.0, 180.0)));
        assert!(!w.contains(&GeoCoord::new(86.0, 0.0)));
    }

    #[test]
    fn test_bounding_box_contains_on_edge() {
        let bbox = BoundingBox::new(
            GeoCoord::new(0.0, 0.0),
            GeoCoord::new(10.0, 10.0),
        );
        // Points exactly on edges should be contained
        assert!(bbox.contains(&GeoCoord::new(0.0, 0.0)));
        assert!(bbox.contains(&GeoCoord::new(10.0, 10.0)));
        assert!(bbox.contains(&GeoCoord::new(5.0, 0.0)));
    }

    #[test]
    fn test_bounding_box_self_intersects() {
        let bbox = BoundingBox::new(
            GeoCoord::new(0.0, 0.0),
            GeoCoord::new(10.0, 10.0),
        );
        assert!(bbox.intersects(&bbox));
    }

    // ── TileCoord additional tests ──────────────────────────

    #[test]
    fn test_tile_coord_display() {
        let tile = TileCoord::new(5, 10, 15);
        assert_eq!(format!("{}", tile), "5/10/15");
    }

    #[test]
    fn test_tile_coord_extent() {
        assert_eq!(TileCoord::new(0, 0, 0).extent(), 1);
        assert_eq!(TileCoord::new(1, 0, 0).extent(), 2);
        assert_eq!(TileCoord::new(4, 0, 0).extent(), 16);
        assert_eq!(TileCoord::new(10, 0, 0).extent(), 1024);
    }

    #[test]
    fn test_tile_coord_from_geo_at_poles() {
        // Near north pole
        let north = TileCoord::from_geo(&GeoCoord::new(85.0, 0.0), 3);
        assert_eq!(north.z, 3);
        assert_eq!(north.y, 0); // Northernmost row

        // Near south pole
        let south = TileCoord::from_geo(&GeoCoord::new(-85.0, 0.0), 3);
        assert_eq!(south.z, 3);
        assert_eq!(south.y, 7); // Southernmost row at z=3
    }

    // ── ConvexPolygon2D additional tests ────────────────────

    #[test]
    fn test_polygon_from_too_few_points() {
        assert!(ConvexPolygon2D::from_points(&[DVec2::ZERO, DVec2::ONE]).is_none());
        assert!(ConvexPolygon2D::from_points(&[]).is_none());
    }
}
