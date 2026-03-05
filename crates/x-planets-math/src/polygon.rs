use glam::DVec2;

// ---------------------------------------------------------------------------
// Convex Polygon 2D (for precise frustum culling)
// ---------------------------------------------------------------------------

/// A convex polygon in Mercator [0,1]x[0,1] space.
///
/// Used for precise frustum-tile intersection tests via the Separating
/// Axis Theorem (SAT).  Much tighter than an AABB when the camera is
/// rotated (bearing != 0).
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

        // Convex hull via gift-wrapping (for 4-6 points this is fine).
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
        // Box at top-left corner [0,0]-[0.1,0.1] -- outside the diamond.
        assert!(!poly.intersects_aabb(DVec2::new(0.0, 0.0), DVec2::new(0.1, 0.1)));
    }

    #[test]
    fn test_polygon_from_too_few_points() {
        assert!(ConvexPolygon2D::from_points(&[DVec2::ZERO, DVec2::ONE]).is_none());
        assert!(ConvexPolygon2D::from_points(&[]).is_none());
    }
}
