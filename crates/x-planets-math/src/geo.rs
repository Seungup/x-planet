use glam::DVec2;
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

    // -- VisibleTile tests --

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
        for child in &children {
            assert_eq!(child.coord.z, 2);
        }
        assert_eq!(children[0].display_x, 0);
        assert_eq!(children[1].display_x, 1);
        assert_eq!(children[2].display_x, 0);
        assert_eq!(children[3].display_x, 1);
    }

    #[test]
    fn test_visible_tile_wrapped_negative_display_x() {
        let vt = VisibleTile {
            coord: TileCoord::new(1, 1, 0),
            display_x: -1,
        };
        let center = vt.display_mercator_center();
        assert!((center.x - (-0.25)).abs() < 1e-10);

        let children = vt.children();
        assert_eq!(children[0].display_x, -2);
        assert_eq!(children[1].display_x, -1);
        assert_eq!(children[0].coord.x, 2);
        assert_eq!(children[1].coord.x, 3);
    }

    // -- GeoCoord additional tests --

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
        let coord = GeoCoord::new(90.0, 180.0);
        let n = coord.normalize();
        assert_eq!(n.lat, 90.0);
        assert_eq!(n.lon, 180.0);

        let coord = GeoCoord::new(-90.0, -180.0);
        let n = coord.normalize();
        assert_eq!(n.lat, -90.0);
        assert_eq!(n.lon, -180.0);
    }

    // -- BoundingBox additional tests --

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

    // -- TileCoord additional tests --

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
        let north = TileCoord::from_geo(&GeoCoord::new(85.0, 0.0), 3);
        assert_eq!(north.z, 3);
        assert_eq!(north.y, 0);

        let south = TileCoord::from_geo(&GeoCoord::new(-85.0, 0.0), 3);
        assert_eq!(south.z, 3);
        assert_eq!(south.y, 7);
    }
}
