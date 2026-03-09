//! Stage 4b–4e: Terrain mesh generation, QM conversion, heightmap rasterization,
//! and EPSG:4326 → EPSG:3857 resampling.

use x_planets_math::TileCoord;
use crate::render::TerrainVertex;

/// Terrain mesh grid resolution (vertices per axis = GRID_SIZE + 1).
/// 32 → 33×33 = 1089 verts, 2048 triangles per tile.
pub const TERRAIN_GRID_SIZE: u32 = 32;

/// Build a displaced terrain mesh for a single tile.
///
/// Generates a `(TERRAIN_GRID_SIZE+1)²` vertex grid with per-vertex normals
/// for hillshade lighting.  Vertex Z is sampled from `elevation` via bilinear
/// interpolation and scaled by `height_scale`.
///
/// `elev_uv_rect` remaps sampling coordinates into the elevation grid:
/// `[0, 0, 1, 1]` = use the full grid (own tile data available).
/// When using a parent's elevation data as fallback, pass the sub-rect
/// computed by `fallback_uv_rect(tile, parent)` so only the relevant
/// quadrant is sampled.
///
/// Pure function.
pub fn build_terrain_mesh(
    coord: &TileCoord,
    elevation: &[f32],
    src_width: u32,
    src_height: u32,
    height_scale: f32,
    elev_uv_rect: [f32; 4],
) -> (Vec<TerrainVertex>, Vec<u32>) {
    let grid = TERRAIN_GRID_SIZE;
    let verts_per_side = grid + 1;
    let vert_count = (verts_per_side * verts_per_side) as usize;
    let mut indices = Vec::with_capacity((grid * grid * 6) as usize);

    // Use f64 for tile bounds to avoid precision loss at high zoom,
    // then store vertex positions *relative to tile center* in f32.
    let n = coord.extent() as f64;
    let tile_size_f64 = 1.0 / n;
    let tile_w = tile_size_f64 as f32;
    let tile_h = tile_w; // square tiles

    // Elevation UV sub-rect: remap [0,1] → [eu_min, eu_max] for parent fallback.
    let eu_min = elev_uv_rect[0];
    let ev_min = elev_uv_rect[1];
    let eu_range = elev_uv_rect[2] - eu_min;
    let ev_range = elev_uv_rect[3] - ev_min;

    // ── Pass 1: Sample elevation at grid points ──
    // Positions are RELATIVE TO TILE CENTER for f32 precision.
    // The shader reconstructs world position using tile.bounds.
    let mut positions = Vec::with_capacity(vert_count);
    let mut tex_coords = Vec::with_capacity(vert_count);

    for gy in 0..verts_per_side {
        for gx in 0..verts_per_side {
            let u = gx as f32 / grid as f32;
            let v = gy as f32 / grid as f32;

            // Remap u,v into the elevation grid's sub-rect
            let eu = eu_min + u * eu_range;
            let ev = ev_min + v * ev_range;
            let h = sample_elevation_bilinear(elevation, src_width, src_height, eu, ev);

            // Relative to tile center: (u - 0.5) * tile_w, (v - 0.5) * tile_h
            positions.push([(u - 0.5) * tile_w, (v - 0.5) * tile_h, h * height_scale]);
            tex_coords.push([u, v]);
        }
    }

    // ── Pass 2: Compute per-vertex normals from neighboring positions ──
    //
    // At tile edges, the naive approach clamps the missing neighbor to `self`,
    // producing a one-sided difference.  Adjacent tiles compute the opposite
    // one-sided difference → normal discontinuity → visible hillshade seam.
    //
    // Fix: use a **reflected sample** at edges.  For example, at the left edge
    // (gx=0), the missing "left" neighbor is approximated as the mirror of
    // the "right" neighbor across the current vertex:
    //     left_virtual = 2 * p - right
    // This produces a central-difference-like normal that's symmetric and
    // continuous.  At interior vertices, standard central differences are used.
    let mut normals = vec![[0.0f32, 0.0, 1.0]; vert_count];
    let vs = verts_per_side as usize;

    for gy in 0..vs {
        for gx in 0..vs {
            let idx = gy * vs + gx;
            let p = positions[idx];

            // Reflected-sample neighbors at edges for smooth boundary normals.
            let right = if gx + 1 < vs { positions[idx + 1] } else {
                // Right edge: reflect left neighbor across p
                let l = positions[idx - 1];
                [2.0 * p[0] - l[0], 2.0 * p[1] - l[1], 2.0 * p[2] - l[2]]
            };
            let left = if gx > 0 { positions[idx - 1] } else {
                // Left edge: reflect right neighbor across p
                let r = positions[idx + 1];
                [2.0 * p[0] - r[0], 2.0 * p[1] - r[1], 2.0 * p[2] - r[2]]
            };
            let down = if gy + 1 < vs { positions[idx + vs] } else {
                // Bottom edge: reflect up neighbor across p
                let u = positions[idx - vs];
                [2.0 * p[0] - u[0], 2.0 * p[1] - u[1], 2.0 * p[2] - u[2]]
            };
            let up = if gy > 0 { positions[idx - vs] } else {
                // Top edge: reflect down neighbor across p
                let d = positions[idx + vs];
                [2.0 * p[0] - d[0], 2.0 * p[1] - d[1], 2.0 * p[2] - d[2]]
            };

            // Tangent vectors (central difference or reflected central difference)
            let dx = [right[0] - left[0], right[1] - left[1], right[2] - left[2]];
            let dy = [down[0] - up[0], down[1] - up[1], down[2] - up[2]];

            // Cross product (dx × dy) → surface normal
            let nx = dx[1] * dy[2] - dx[2] * dy[1];
            let ny = dx[2] * dy[0] - dx[0] * dy[2];
            let nz = dx[0] * dy[1] - dx[1] * dy[0];

            let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-10);
            normals[idx] = [nx / len, ny / len, nz / len];
        }
    }

    // ── Assemble surface vertices ──
    let mut vertices: Vec<TerrainVertex> = (0..vert_count)
        .map(|i| TerrainVertex {
            position: positions[i],
            normal: normals[i],
            tex_coord: tex_coords[i],
        })
        .collect();

    // ── Generate surface triangle indices ──
    // Winding: CW in tile-local (y-down) space.
    // After VP flip_x + y-inversion, this becomes CCW in clip space,
    // matching the terrain pipeline's FrontFace::Ccw setting.
    // (Same winding convention as QM PrebuiltMesh indices.)
    for gy in 0..grid {
        for gx in 0..grid {
            let tl = gy * verts_per_side + gx;
            let tr = tl + 1;
            let bl = tl + verts_per_side;
            let br = bl + 1;

            indices.push(tl);
            indices.push(bl);
            indices.push(tr);
            indices.push(tr);
            indices.push(bl);
            indices.push(br);
        }
    }

    // ── Skirt geometry ──
    // Extend vertical "walls" below each edge to hide gaps between tiles.
    let skirt_depth = tile_w * 0.05; // 5% of tile width
    let down_normal = [0.0f32, 0.0, -1.0];

    // Collect edge vertex indices: bottom, top, right, left edges
    let mut edge_strips: Vec<Vec<u32>> = Vec::new();

    // Bottom edge (gy=last, left to right)
    let mut strip = Vec::new();
    for gx in 0..verts_per_side {
        strip.push(grid * verts_per_side + gx);
    }
    edge_strips.push(strip);

    // Top edge (gy=0, right to left)
    // Reversed traversal: outward normal is -Y, reversing direction
    // makes the skirt triangles CW when viewed from outside.
    let mut strip = Vec::new();
    for gx in (0..verts_per_side).rev() {
        strip.push(gx);
    }
    edge_strips.push(strip);

    // Right edge (gx=last, bottom to top)
    // Reversed traversal: outward normal is +X, reversing direction
    // makes the skirt triangles CW when viewed from outside.
    let mut strip = Vec::new();
    for gy in (0..verts_per_side).rev() {
        strip.push(gy * verts_per_side + grid);
    }
    edge_strips.push(strip);

    // Left edge (gx=0, top to bottom)
    let mut strip = Vec::new();
    for gy in 0..verts_per_side {
        strip.push(gy * verts_per_side);
    }
    edge_strips.push(strip);

    for edge in &edge_strips {
        for i in 0..edge.len() - 1 {
            let top_a = edge[i] as usize;
            let top_b = edge[i + 1] as usize;

            // Add two skirt vertices (same xy, lowered z)
            let skirt_a = vertices.len() as u32;
            let mut pa = positions[top_a];
            pa[2] -= skirt_depth;
            vertices.push(TerrainVertex {
                position: pa,
                normal: down_normal,
                tex_coord: tex_coords[top_a],
            });

            let skirt_b = vertices.len() as u32;
            let mut pb = positions[top_b];
            pb[2] -= skirt_depth;
            vertices.push(TerrainVertex {
                position: pb,
                normal: down_normal,
                tex_coord: tex_coords[top_b],
            });

            // Two triangles (reversed winding to match surface):
            // top_a, skirt_a, top_b  +  skirt_a, skirt_b, top_b
            indices.push(edge[i]);
            indices.push(skirt_a);
            indices.push(edge[i + 1]);
            indices.push(skirt_a);
            indices.push(skirt_b);
            indices.push(edge[i + 1]);
        }
    }

    (vertices, indices)
}

/// Bilinear sample from elevation grid.
pub(crate) fn sample_elevation_bilinear(
    elevation: &[f32],
    src_width: u32,
    src_height: u32,
    u: f32,
    v: f32,
) -> f32 {
    let sx = u * (src_width - 1) as f32;
    let sy = v * (src_height - 1) as f32;
    let ix = (sx as u32).min(src_width.saturating_sub(2));
    let iy = (sy as u32).min(src_height.saturating_sub(2));
    let fx = sx - ix as f32;
    let fy = sy - iy as f32;

    let idx00 = (iy * src_width + ix) as usize;
    let idx10 = idx00 + 1;
    let idx01 = idx00 + src_width as usize;
    let idx11 = idx01 + 1;

    if idx11 < elevation.len() {
        elevation[idx00] * (1.0 - fx) * (1.0 - fy)
            + elevation[idx10] * fx * (1.0 - fy)
            + elevation[idx01] * (1.0 - fx) * fy
            + elevation[idx11] * fx * fy
    } else if !elevation.is_empty() {
        elevation[idx00.min(elevation.len() - 1)]
    } else {
        0.0
    }
}

/// Compute height scale: converts meters of elevation to Mercator [0,1] world units.
///
/// `exaggeration` controls visual amplification (1.0 = real scale, 2.0 = 2× taller).
/// At the equator, 1 Mercator unit ≈ 40,075,000 m.
///
/// A base exaggeration of 1.0 would produce realistic proportions but
/// the heights are nearly invisible at global zoom levels.
///
/// Pure function.
pub fn compute_height_scale(exaggeration: f64) -> f32 {
    compute_height_scale_for(exaggeration, x_planets_math::ecef::EARTH.circumference)
}

/// Like [`compute_height_scale`] but accepts an explicit equatorial circumference
/// so it works for any celestial body (Moon, Mars, etc.).
pub fn compute_height_scale_for(exaggeration: f64, circumference: f64) -> f32 {
    (exaggeration / circumference) as f32
}

// ───────────────────────────────────────────────────────────────────
// Stage 4c: Quantized Mesh → TerrainVertex conversion
// ───────────────────────────────────────────────────────────────────

/// Convert a decoded Quantized Mesh tile into `TerrainVertex` + index lists.
///
/// Heights in the returned vertices are in **metres** (not scaled).
/// The caller (`TerrainRenderer::get_or_build_mesh`) applies `height_scale`
/// when uploading to the GPU so that exaggeration changes work without
/// re-fetching tiles.
///
/// ## Coordinate mapping
/// - `u`: 0-32767 → 0-1 (west → east across tile)
/// - `v`: 0-32767 → 0-1 QM south-north; **flipped** here so that v_tile = 0 = north
///   (matches the existing XYZ / web-Mercator orientation used by `build_terrain_mesh`).
/// - `height`: 0-32767 → [`header.min_height`, `header.max_height`] metres.
///
/// ## Normals
/// If the tile contains an oct-encoded normals extension, those are decoded and used.
/// Otherwise, per-vertex normals are computed from the mesh geometry (face-normal
/// accumulation, area-weighted).
///
/// ## Skirts
/// Edge vertex lists (`west_indices`, `south_indices`, `east_indices`, `north_indices`)
/// are used to generate skirt quads that prevent cracks between adjacent tiles.
///
/// Pure function.
pub fn build_terrain_mesh_from_qm(
    coord: &TileCoord,
    qm: &x_planets_tiles::DecodedQuantizedMesh,
) -> (Vec<TerrainVertex>, Vec<u32>) {
    build_terrain_mesh_from_qm_with(coord, qm, x_planets_math::ecef::EARTH.circumference)
}

/// Like [`build_terrain_mesh_from_qm`] but accepts an explicit equatorial
/// circumference for multi-planet support.
pub fn build_terrain_mesh_from_qm_with(
    coord: &TileCoord,
    qm: &x_planets_tiles::DecodedQuantizedMesh,
    circumference: f64,
) -> (Vec<TerrainVertex>, Vec<u32>) {
    use x_planets_tiles::quantized_mesh::decode_oct_normal;

    let n = coord.extent() as f64;
    let tile_w = (1.0 / n) as f32;
    let tile_h = tile_w;

    let min_h = qm.header.min_height;
    let max_h = qm.header.max_height;
    let h_range = max_h - min_h;
    let vertex_count = qm.u.len();

    // ── Pass 1: positions + tex_coords ────────────────────────────
    let mut positions: Vec<[f32; 3]>   = Vec::with_capacity(vertex_count);
    let mut tex_coords: Vec<[f32; 2]>  = Vec::with_capacity(vertex_count);

    for i in 0..vertex_count {
        let u_norm = qm.u[i] as f32 / 32767.0;       // 0-1, west→east
        let v_norm = qm.v[i] as f32 / 32767.0;       // 0-1, south→north (QM convention)
        let h_norm = qm.height[i] as f32 / 32767.0;

        // QM v=0 is south, v=1 is north.
        // The existing pipeline uses v=0 at the top (north in XYZ TMS=false tiles).
        let v_tile = 1.0 - v_norm; // flip to match existing convention

        let height_m = min_h + h_norm * h_range; // metres (not scaled)

        positions.push([
            (u_norm - 0.5) * tile_w,
            (v_tile - 0.5) * tile_h,
            height_m,            // raw metres; height_scale applied in renderer
        ]);
        tex_coords.push([u_norm, v_tile]);
    }

    // ── Pass 2: normals ────────────────────────────────────────────
    let normals: Vec<[f32; 3]> = if let Some(oct) = &qm.oct_normals {
        oct.iter()
            .map(|&[x, y]| decode_oct_normal(x, y))
            .collect()
    } else {
        compute_normals_from_triangles(&positions, &qm.indices)
    };

    // ── Assemble surface vertices ─────────────────────────────────
    let mut vertices: Vec<TerrainVertex> = (0..vertex_count)
        .map(|i| TerrainVertex {
            position:  positions[i],
            normal:    normals[i],
            tex_coord: tex_coords[i],
        })
        .collect();

    let mut indices = qm.indices.clone();

    // ── Skirts (edge vertices → downward quads) ───────────────────
    // Prevents gaps/cracks between adjacent tiles at different detail.
    //
    // QM positions store z in **metres**, not in scaled tile-local space
    // (height_scale is applied later in the renderer).  The skirt depth
    // must therefore also be in metres so it remains proportional after
    // scaling.  We use ~2 % of the tile's equatorial width in metres,
    // which gives a consistent 2-3 % depth relative to tile_w in the
    // final coordinate space regardless of exaggeration.
    let tile_extent_m = circumference / n;
    let skirt_depth = (tile_extent_m * 0.02) as f32;
    let down_normal = [0.0f32, 0.0, -1.0];

    for edge_indices in [
        &qm.west_indices,
        &qm.south_indices,
        &qm.east_indices,
        &qm.north_indices,
    ] {
        let edge_count = edge_indices.len();
        if edge_count < 2 {
            continue;
        }
        for i in 0..edge_count - 1 {
            let top_a = edge_indices[i] as usize;
            let top_b = edge_indices[i + 1] as usize;
            if top_a >= positions.len() || top_b >= positions.len() {
                continue;
            }

            let skirt_a_idx = vertices.len() as u32;
            let mut pa = positions[top_a];
            pa[2] -= skirt_depth;
            vertices.push(TerrainVertex {
                position:  pa,
                normal:    down_normal,
                tex_coord: tex_coords[top_a],
            });

            let skirt_b_idx = vertices.len() as u32;
            let mut pb = positions[top_b];
            pb[2] -= skirt_depth;
            vertices.push(TerrainVertex {
                position:  pb,
                normal:    down_normal,
                tex_coord: tex_coords[top_b],
            });

            // CW winding in tile-local space → CCW in clip space after VP flip_x
            // (matches terrain_renderer pipeline: FrontFace::Ccw)
            let ia = edge_indices[i];
            let ib = edge_indices[i + 1];
            indices.push(ia);
            indices.push(skirt_a_idx);
            indices.push(ib);
            indices.push(ib);
            indices.push(skirt_a_idx);
            indices.push(skirt_b_idx);
        }
    }

    (vertices, indices)
}

// ───────────────────────────────────────────────────────────────────
// Stage 4b-centered: Terrain mesh in oblique (centered) Mercator space
// ───────────────────────────────────────────────────────────────────

/// Build a displaced terrain mesh in oblique (viewport-centered) Mercator space.
///
/// Same elevation sampling as [`build_terrain_mesh`], but vertex XY positions are
/// projected through `oblique_mercator(lat, lon, center)` — matching the raster
/// renderer's `tile_centered_mesh`.  The tile center in oblique Mercator space
/// is subtracted (RTE) so positions stay near the origin for f32 precision.
///
/// Z = elevation × `height_scale` (same as standard terrain mesh).
///
/// This ensures terrain meshes align with the raster tiles in centered Mercator
/// projection mode.
///
/// Pure function.
pub fn build_terrain_mesh_centered(
    coord: &TileCoord,
    elevation: &[f32],
    src_width: u32,
    src_height: u32,
    height_scale: f32,
    elev_uv_rect: [f32; 4],
    center_lat_rad: f64,
    center_lon_rad: f64,
    tile_center_2d: glam::DVec2,
) -> (Vec<TerrainVertex>, Vec<u32>) {
    use std::f64::consts::PI;

    let grid = TERRAIN_GRID_SIZE;
    let verts_per_side = grid + 1;
    let vert_count = (verts_per_side * verts_per_side) as usize;
    let mut indices = Vec::with_capacity((grid * grid * 6) as usize);

    let n = coord.extent() as f64;

    let eu_min = elev_uv_rect[0];
    let ev_min = elev_uv_rect[1];
    let eu_range = elev_uv_rect[2] - eu_min;
    let ev_range = elev_uv_rect[3] - ev_min;

    let tile_w = (1.0 / n) as f32;
    let tile_h = tile_w;

    let mut positions = Vec::with_capacity(vert_count);
    // Standard Mercator positions — used only for normal computation.
    // The oblique projection distorts XY distances, making normals
    // artificially steep for distant tiles and darkening the hillshade.
    // By computing normals from undistorted Mercator coordinates we get
    // consistent, geographically correct relief shading everywhere.
    let mut merc_positions = Vec::with_capacity(vert_count);
    let mut tex_coords = Vec::with_capacity(vert_count);

    for gy in 0..verts_per_side {
        for gx in 0..verts_per_side {
            let u = gx as f64 / grid as f64;
            let v = gy as f64 / grid as f64;

            // Global Mercator position [0,1]
            let mx = (coord.x as f64 + u) / n;
            let my = (coord.y as f64 + v) / n;

            // Standard Mercator → lat/lon
            let lon_rad = (mx * 2.0 - 1.0) * PI;
            let lat_rad = x_planets_math::mercator_y_to_lat_rad(my);

            // Oblique (centered) Mercator
            let centered = x_planets_math::oblique_mercator(
                lat_rad, lon_rad, center_lat_rad, center_lon_rad,
            );
            let cx = if centered.x.is_finite() { centered.x } else { 0.5 };
            let cy = if centered.y.is_finite() { centered.y } else { 0.5 };

            // RTE: subtract tile center in 2D
            let rx = (cx - tile_center_2d.x) as f32;
            let ry = (cy - tile_center_2d.y) as f32;

            // Sample elevation
            let eu = eu_min + u as f32 * eu_range;
            let ev = ev_min + v as f32 * ev_range;
            let h = sample_elevation_bilinear(elevation, src_width, src_height, eu, ev);

            positions.push([rx, ry, h * height_scale]);
            // Standard Mercator position for normal calc (undistorted)
            merc_positions.push([
                (u as f32 - 0.5) * tile_w,
                (v as f32 - 0.5) * tile_h,
                h * height_scale,
            ]);
            tex_coords.push([u as f32, v as f32]);
        }
    }

    // Normals — computed from standard Mercator positions (merc_positions)
    // to avoid oblique projection distortion darkening the hillshade.
    let mut normals = vec![[0.0f32, 0.0, 1.0]; vert_count];
    let vs = verts_per_side as usize;
    for gy in 0..vs {
        for gx in 0..vs {
            let idx = gy * vs + gx;
            let p = merc_positions[idx];
            let right = if gx + 1 < vs { merc_positions[idx + 1] } else {
                let l = merc_positions[idx - 1];
                [2.0 * p[0] - l[0], 2.0 * p[1] - l[1], 2.0 * p[2] - l[2]]
            };
            let left = if gx > 0 { merc_positions[idx - 1] } else {
                let r = merc_positions[idx + 1];
                [2.0 * p[0] - r[0], 2.0 * p[1] - r[1], 2.0 * p[2] - r[2]]
            };
            let down = if gy + 1 < vs { merc_positions[idx + vs] } else {
                let u = merc_positions[idx - vs];
                [2.0 * p[0] - u[0], 2.0 * p[1] - u[1], 2.0 * p[2] - u[2]]
            };
            let up = if gy > 0 { merc_positions[idx - vs] } else {
                let d = merc_positions[idx + vs];
                [2.0 * p[0] - d[0], 2.0 * p[1] - d[1], 2.0 * p[2] - d[2]]
            };
            let dx = [right[0] - left[0], right[1] - left[1], right[2] - left[2]];
            let dy = [down[0] - up[0], down[1] - up[1], down[2] - up[2]];
            let nx = dx[1] * dy[2] - dx[2] * dy[1];
            let ny = dx[2] * dy[0] - dx[0] * dy[2];
            let nz = dx[0] * dy[1] - dx[1] * dy[0];
            let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-10);
            normals[idx] = [nx / len, ny / len, nz / len];
        }
    }

    // Detect whether the oblique Mercator projection has flipped the
    // spatial orientation of this tile.  The standard winding (tl→bl→tr)
    // is CW in y-down space.  After oblique reprojection, tiles far from
    // the center can have inverted orientation, which flips the winding.
    // Check the 2D cross product of a center quad to decide.
    let mid = grid / 2;
    let mid_tl = (mid * verts_per_side + mid) as usize;
    let mid_tr = mid_tl + 1;
    let mid_bl = mid_tl + verts_per_side as usize;
    let winding_cross = {
        let a = positions[mid_tl];
        let b = positions[mid_bl];
        let c = positions[mid_tr];
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    // winding_cross > 0 → CW in y-down (normal), < 0 → inverted by projection.
    let flipped = winding_cross < 0.0;

    let mut vertices: Vec<TerrainVertex> = (0..vert_count)
        .map(|i| TerrainVertex {
            position: positions[i],
            normal: normals[i],
            tex_coord: tex_coords[i],
        })
        .collect();

    // Surface triangle indices.
    // When the oblique projection inverts orientation, swap the winding
    // so back-face culling doesn't discard the visible faces.
    for gy in 0..grid {
        for gx in 0..grid {
            let tl = gy * verts_per_side + gx;
            let tr = tl + 1;
            let bl = tl + verts_per_side;
            let br = bl + 1;
            if flipped {
                // Swap to CCW in y-down (= CW after flip_x = correct)
                indices.push(tl);
                indices.push(tr);
                indices.push(bl);
                indices.push(tr);
                indices.push(br);
                indices.push(bl);
            } else {
                indices.push(tl);
                indices.push(bl);
                indices.push(tr);
                indices.push(tr);
                indices.push(bl);
                indices.push(br);
            }
        }
    }

    // Skirt geometry — use 5% of the centered-space tile extent.
    let skirt_depth = {
        let extent_x = (positions[verts_per_side as usize - 1][0] - positions[0][0]).abs();
        let extent_y = (positions[(verts_per_side * (verts_per_side - 1)) as usize][1] - positions[0][1]).abs();
        (extent_x.max(extent_y) * 0.05).max(1e-8)
    };
    let down_normal = [0.0f32, 0.0, -1.0];

    let mut edge_strips: Vec<Vec<u32>> = Vec::new();
    // Bottom edge
    let mut strip = Vec::new();
    for gx in 0..verts_per_side { strip.push(grid * verts_per_side + gx); }
    edge_strips.push(strip);
    // Top edge (reversed)
    let mut strip = Vec::new();
    for gx in (0..verts_per_side).rev() { strip.push(gx); }
    edge_strips.push(strip);
    // Right edge (reversed)
    let mut strip = Vec::new();
    for gy in (0..verts_per_side).rev() { strip.push(gy * verts_per_side + grid); }
    edge_strips.push(strip);
    // Left edge
    let mut strip = Vec::new();
    for gy in 0..verts_per_side { strip.push(gy * verts_per_side); }
    edge_strips.push(strip);

    for edge in &edge_strips {
        for i in 0..edge.len() - 1 {
            let top_a = edge[i] as usize;
            let top_b = edge[i + 1] as usize;
            let skirt_a = vertices.len() as u32;
            let mut pa = positions[top_a];
            pa[2] -= skirt_depth;
            vertices.push(TerrainVertex { position: pa, normal: down_normal, tex_coord: tex_coords[top_a] });
            let skirt_b = vertices.len() as u32;
            let mut pb = positions[top_b];
            pb[2] -= skirt_depth;
            vertices.push(TerrainVertex { position: pb, normal: down_normal, tex_coord: tex_coords[top_b] });
            if flipped {
                indices.push(edge[i]);
                indices.push(edge[i + 1]);
                indices.push(skirt_a);
                indices.push(skirt_a);
                indices.push(edge[i + 1]);
                indices.push(skirt_b);
            } else {
                indices.push(edge[i]);
                indices.push(skirt_a);
                indices.push(edge[i + 1]);
                indices.push(skirt_a);
                indices.push(skirt_b);
                indices.push(edge[i + 1]);
            }
        }
    }

    (vertices, indices)
}

// ───────────────────────────────────────────────────────────────────
// Stage 4b-globe: Terrain mesh on the unit sphere
// ───────────────────────────────────────────────────────────────────

/// Build a displaced terrain mesh on the unit sphere.
///
/// Same elevation sampling as [`build_terrain_mesh`], but vertex positions are
/// placed on the unit sphere (like `tile_globe_mesh`) with radial displacement
/// proportional to elevation.  The tile center in 3D sphere space is subtracted
/// (RTE) so positions stay near the origin for f32 precision.
///
/// Elevation is converted to sphere-radius offset:
///   `radius = 1.0 + height_metres * height_scale`
/// where `height_scale` is from `compute_height_scale()`.
///
/// Pure function.
pub fn build_terrain_mesh_globe(
    coord: &TileCoord,
    elevation: &[f32],
    src_width: u32,
    src_height: u32,
    height_scale: f32,
    elev_uv_rect: [f32; 4],
    tile_center_3d: glam::DVec3,
) -> (Vec<TerrainVertex>, Vec<u32>) {
    use std::f64::consts::PI;

    let grid = TERRAIN_GRID_SIZE;
    let verts_per_side = grid + 1;
    let vert_count = (verts_per_side * verts_per_side) as usize;

    let n = coord.extent() as f64;

    let eu_min = elev_uv_rect[0];
    let ev_min = elev_uv_rect[1];
    let eu_range = elev_uv_rect[2] - eu_min;
    let ev_range = elev_uv_rect[3] - ev_min;

    let mut positions = Vec::with_capacity(vert_count);
    let mut tex_coords = Vec::with_capacity(vert_count);

    for gy in 0..verts_per_side {
        for gx in 0..verts_per_side {
            let u = gx as f64 / grid as f64;
            let v = gy as f64 / grid as f64;

            // Global Mercator position [0,1]
            let mx = (coord.x as f64 + u) / n;
            let my = (coord.y as f64 + v) / n;

            // Mercator → lat/lon (radians)
            let lon_rad = (mx * 2.0 - 1.0) * PI;
            let lat_rad = x_planets_math::mercator_y_to_lat_rad(my);

            // Sample elevation
            let eu = eu_min + u as f32 * eu_range;
            let ev = ev_min + v as f32 * ev_range;
            let h = sample_elevation_bilinear(elevation, src_width, src_height, eu, ev);

            // Position on unit sphere with radial elevation displacement.
            // height_scale is calibrated for Mercator [0,1] space (circumference = 1).
            // The unit sphere has circumference = 2π, so multiply by TAU to convert.
            let surface_pos = x_planets_math::geo_to_unit_sphere(lat_rad, lon_rad);
            let radius = 1.0 + h as f64 * height_scale as f64 * std::f64::consts::TAU;
            let pos_3d = surface_pos * radius;

            // RTE: subtract tile center
            let rte = pos_3d - tile_center_3d;

            positions.push([rte.x as f32, rte.y as f32, rte.z as f32]);
            tex_coords.push([u as f32, v as f32]);
        }
    }

    // Normals — compute from mesh geometry (area-weighted face normals)
    // since the sphere curvature makes the flat-grid approach inappropriate.
    // First build surface indices, then compute normals from those triangles.
    let mut surface_indices = Vec::with_capacity((grid * grid * 6) as usize);
    for gy in 0..grid {
        for gx in 0..grid {
            let tl = gy * verts_per_side + gx;
            let tr = tl + 1;
            let bl = tl + verts_per_side;
            let br = bl + 1;
            // CCW winding from outside the sphere (matching tile_globe_mesh)
            surface_indices.push(tl);
            surface_indices.push(bl);
            surface_indices.push(tr);
            surface_indices.push(tr);
            surface_indices.push(bl);
            surface_indices.push(br);
        }
    }

    let normals = compute_normals_from_triangles(&positions, &surface_indices);

    let mut vertices: Vec<TerrainVertex> = (0..vert_count)
        .map(|i| TerrainVertex {
            position: positions[i],
            normal: normals[i],
            tex_coord: tex_coords[i],
        })
        .collect();

    let mut indices = surface_indices;

    // Skirt geometry: push vertices toward the sphere center.
    // Positions are RTE (tile_center_3d subtracted), so we must reconstruct
    // absolute positions, shrink toward origin, then re-subtract the center.
    let skirt_depth: f64 = 0.005; // fraction of radius to push inward

    let edge_strips: [Vec<u32>; 4] = [
        (0..verts_per_side).collect(),
        (0..verts_per_side).map(|i| grid * verts_per_side + i).collect(),
        (0..verts_per_side).map(|j| j * verts_per_side).collect(),
        (0..verts_per_side).map(|j| j * verts_per_side + grid).collect(),
    ];

    // Helper: given an RTE position, reconstruct absolute, shrink toward
    // the sphere center by `skirt_depth` fraction, then convert back to RTE.
    let skirt_pos = |p: [f32; 3]| -> [f32; 3] {
        let abs_x = p[0] as f64 + tile_center_3d.x;
        let abs_y = p[1] as f64 + tile_center_3d.y;
        let abs_z = p[2] as f64 + tile_center_3d.z;
        [
            (abs_x * (1.0 - skirt_depth) - tile_center_3d.x) as f32,
            (abs_y * (1.0 - skirt_depth) - tile_center_3d.y) as f32,
            (abs_z * (1.0 - skirt_depth) - tile_center_3d.z) as f32,
        ]
    };

    for strip in &edge_strips {
        for k in 0..strip.len() - 1 {
            let top_a = strip[k];
            let top_b = strip[k + 1];

            let skirt_a = vertices.len() as u32;
            let va = &vertices[top_a as usize];
            vertices.push(TerrainVertex {
                position: skirt_pos(va.position),
                normal: va.normal,
                tex_coord: va.tex_coord,
            });

            let skirt_b = vertices.len() as u32;
            let vb = &vertices[top_b as usize];
            vertices.push(TerrainVertex {
                position: skirt_pos(vb.position),
                normal: vb.normal,
                tex_coord: vb.tex_coord,
            });

            indices.extend_from_slice(&[
                top_a, skirt_a, top_b,
                top_b, skirt_a, skirt_b,
            ]);
        }
    }

    (vertices, indices)
}

// ───────────────────────────────────────────────────────────────────
// Stage 4c-geo: Project QM mesh from EPSG:4326 to EPSG:3857
// ───────────────────────────────────────────────────────────────────

/// Re-project a QM mesh from EPSG:4326 tile-local space to EPSG:3857
/// tile-local space **in place**.
///
/// After `build_terrain_mesh_from_qm` the vertex tex_coords are in the
/// 4326 tile's UV space: u ∈ [0,1] west→east, v ∈ [0,1] north→south.
/// This function converts every vertex's position and tex_coord so
/// that they live in the target 3857 tile's local space instead.
///
/// ## Why this matters
/// The rasterize → resample pipeline loses edge continuity:
/// adjacent 3857 tiles that straddle a 4326 latitude boundary
/// rasterize DIFFERENT QM TIN meshes, producing different
/// interpolated heights at the shared seam → cliff walls.
///
/// By projecting the QM mesh directly, edge vertices that are
/// shared between adjacent 4326 tiles (guaranteed by the QM spec)
/// map to the same 3857 positions, preserving continuity.
///
/// ## Vertex transformation
/// - `tex_coord` → geographic (lon, lat) → 3857 normalised → 3857 tile-local UV
/// - `position[0..2]` → recalculated from the new UV + original height
/// - `position[2]` (height in metres) is **unchanged**
///
/// Pure function (modifies `vertices` in place).
pub fn project_qm_vertices_4326_to_3857(
    vertices: &mut [crate::render::TerrainVertex],
    merc_coord: &TileCoord,
    geo_west: f64,
    geo_east: f64,
    geo_north: f64,
    geo_south: f64,
) {
    let pi = std::f64::consts::PI;
    let n = (1u64 << merc_coord.z) as f64;
    let tile_w = (1.0 / n) as f32;

    let geo_lon_range = geo_east - geo_west;
    let geo_lat_range = geo_north - geo_south;

    for v in vertices.iter_mut() {
        let u_4326 = v.tex_coord[0] as f64;
        let v_4326 = v.tex_coord[1] as f64;

        // 4326 UV → geographic degrees
        let lon_deg = geo_west + u_4326 * geo_lon_range;
        let lat_deg = geo_north - v_4326 * geo_lat_range;

        // Geographic → 3857 normalised [0,1]×[0,1]
        let merc_nx = (lon_deg + 180.0) / 360.0;
        let lat_rad = lat_deg.to_radians();
        let merc_ny = 0.5 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / (2.0 * pi);

        // 3857 normalised → tile-local UV
        let u_3857 = merc_nx * n - merc_coord.x as f64;
        let v_3857 = merc_ny * n - merc_coord.y as f64;

        // Update position (tile-local coords in 3857 grid).
        // position[2] (height in metres) stays unchanged.
        v.position[0] = (u_3857 as f32 - 0.5) * tile_w;
        v.position[1] = (v_3857 as f32 - 0.5) * tile_w;

        // Update tex_coord to 3857 tile-local UV (for imagery draping)
        v.tex_coord = [u_3857 as f32, v_3857 as f32];
    }
}

/// Compute per-vertex normals from a triangle mesh.
///
/// Uses area-weighted face normal accumulation.
/// Fallback when oct-encoded normals are not available in the QM tile.
///
/// Pure function.
fn compute_normals_from_triangles(
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32, 0.0, 0.0]; positions.len()];

    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if ia >= positions.len() || ib >= positions.len() || ic >= positions.len() {
            continue;
        }
        let a = positions[ia];
        let b = positions[ib];
        let c = positions[ic];

        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];

        // Cross product (magnitude = 2× triangle area → area-weighted)
        let nx = ab[1] * ac[2] - ab[2] * ac[1];
        let ny = ab[2] * ac[0] - ab[0] * ac[2];
        let nz = ab[0] * ac[1] - ab[1] * ac[0];

        for &idx in &[ia, ib, ic] {
            normals[idx][0] += nx;
            normals[idx][1] += ny;
            normals[idx][2] += nz;
        }
    }

    for n in &mut normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-10);
        n[0] /= len;
        n[1] /= len;
        n[2] /= len;
    }

    normals
}

// ───────────────────────────────────────────────────────────────────
// Stage 4d: QM mesh → regular heightmap rasterization (over-zoom fallback)
// ───────────────────────────────────────────────────────────────────

/// Rasterize a Quantized Mesh triangle mesh into a regular grid heightmap.
///
/// Used to create the `fallback_heightmap` stored alongside `PrebuiltMesh`.
/// When a child tile beyond `max_zoom` needs elevation from a parent QM tile,
/// it sub-samples this heightmap using `build_terrain_mesh()` with the appropriate
/// `elev_uv_rect`, just like heightmap-based terrain (Terrain RGB / Terrarium).
///
/// The grid is `grid_size × grid_size` and covers the full [0,1]² UV space of the
/// tile.  Heights are in metres (matching `TerrainVertex::position[2]`).
///
/// Pure function.
pub fn rasterize_qm_to_heightmap(
    vertices: &[crate::render::TerrainVertex],
    indices: &[u32],
    grid_size: u32,
) -> Vec<f32> {
    let gs = grid_size as usize;
    let mut heightmap = vec![0.0f32; gs * gs];
    // Track which cells have been written for gap-filling later.
    let mut filled = vec![false; gs * gs];

    let inv = 1.0 / (grid_size - 1) as f32;

    // Rasterize each triangle: for each grid cell whose center falls inside
    // the triangle (in tex_coord / UV space), compute height via barycentric
    // interpolation.
    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let (ia, ib, ic) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if ia >= vertices.len() || ib >= vertices.len() || ic >= vertices.len() {
            continue;
        }

        let a_uv = vertices[ia].tex_coord;
        let b_uv = vertices[ib].tex_coord;
        let c_uv = vertices[ic].tex_coord;
        let a_h = vertices[ia].position[2];
        let b_h = vertices[ib].position[2];
        let c_h = vertices[ic].position[2];

        // Bounding box of the triangle in grid coordinates
        let min_u = a_uv[0].min(b_uv[0]).min(c_uv[0]);
        let max_u = a_uv[0].max(b_uv[0]).max(c_uv[0]);
        let min_v = a_uv[1].min(b_uv[1]).min(c_uv[1]);
        let max_v = a_uv[1].max(b_uv[1]).max(c_uv[1]);

        let col_min = ((min_u / inv).floor() as usize).min(gs - 1);
        let col_max = ((max_u / inv).ceil() as usize).min(gs - 1);
        let row_min = ((min_v / inv).floor() as usize).min(gs - 1);
        let row_max = ((max_v / inv).ceil() as usize).min(gs - 1);

        for row in row_min..=row_max {
            for col in col_min..=col_max {
                let pu = col as f32 * inv;
                let pv = row as f32 * inv;

                // Barycentric coordinates
                let (w0, w1, w2) = barycentric(
                    pu, pv,
                    a_uv[0], a_uv[1],
                    b_uv[0], b_uv[1],
                    c_uv[0], c_uv[1],
                );

                if w0 >= -1e-4 && w1 >= -1e-4 && w2 >= -1e-4 {
                    let idx = row * gs + col;
                    let h = w0 * a_h + w1 * b_h + w2 * c_h;
                    heightmap[idx] = h;
                    filled[idx] = true;
                }
            }
        }
    }

    // Fill unfilled cells with nearest filled neighbor (simple flood fill).
    // This handles tiny gaps due to floating-point precision.
    fill_gaps(&mut heightmap, &filled, gs);

    heightmap
}

/// Barycentric coordinates of point (px, py) with respect to triangle (ax,ay)-(bx,by)-(cx,cy).
fn barycentric(
    px: f32, py: f32,
    ax: f32, ay: f32,
    bx: f32, by: f32,
    cx: f32, cy: f32,
) -> (f32, f32, f32) {
    let v0x = bx - ax;
    let v0y = by - ay;
    let v1x = cx - ax;
    let v1y = cy - ay;
    let v2x = px - ax;
    let v2y = py - ay;

    let d00 = v0x * v0x + v0y * v0y;
    let d01 = v0x * v1x + v0y * v1y;
    let d11 = v1x * v1x + v1y * v1y;
    let d20 = v2x * v0x + v2y * v0y;
    let d21 = v2x * v1x + v2y * v1y;

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        return (-1.0, -1.0, -1.0); // Degenerate triangle
    }
    let inv_denom = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;
    let u = 1.0 - v - w;

    (u, v, w)
}

/// Fill unfilled cells with the nearest filled cell's value.
/// Simple iterative spreading — runs at most `gs` passes.
fn fill_gaps(heightmap: &mut [f32], filled: &[bool], gs: usize) {
    let unfilled_count = filled.iter().filter(|&&f| !f).count();
    if unfilled_count == 0 {
        return;
    }

    let mut current_filled = filled.to_vec();
    let offsets: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for _ in 0..gs {
        let mut any_changed = false;
        let prev_filled = current_filled.clone();
        for row in 0..gs {
            for col in 0..gs {
                let idx = row * gs + col;
                if prev_filled[idx] {
                    continue;
                }
                // Find any filled neighbor
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for &(dr, dc) in &offsets {
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    if nr >= 0 && nr < gs as i32 && nc >= 0 && nc < gs as i32 {
                        let ni = nr as usize * gs + nc as usize;
                        if prev_filled[ni] {
                            sum += heightmap[ni];
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    heightmap[idx] = sum / count as f32;
                    current_filled[idx] = true;
                    any_changed = true;
                }
            }
        }
        if !any_changed {
            break;
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Stage 4e: EPSG:4326 → EPSG:3857 heightmap resampling
// ───────────────────────────────────────────────────────────────────

/// Resample a heightmap from EPSG:4326 (Geographic) tile UV space to
/// EPSG:3857 (Web Mercator) tile UV space.
///
/// The input heightmap lives in the UV space of a 4326 tile with the given
/// geographic bounds.  This function creates a new heightmap for a 3857 tile
/// by mapping each output grid point to geographic (lon, lat) coordinates
/// (using the Mercator projection), then sampling the input heightmap.
///
/// This is necessary because the two projections use fundamentally different
/// tile grids:
/// - EPSG:3857 tiles are square in Mercator space (latitude varies non-linearly)
/// - EPSG:4326 tiles are rectangular in lat/lon space (uniform degree spacing)
///
/// ## Parameters
/// - `src_heightmap`: input heightmap in 4326 UV space (row-major, north→south)
/// - `src_grid_size`: side length of the square source grid
/// - `geo_west/east/north/south`: geographic bounds of the 4326 tile (degrees)
/// - `merc_coord`: the 3857 tile coordinate to produce the output for
/// - `out_grid_size`: side length of the square output grid
///
/// ## Returns
/// A `Vec<f32>` heightmap in 3857 UV space, `out_grid_size × out_grid_size`,
/// with heights in metres.
///
/// Pure function.
pub fn resample_geographic_to_mercator(
    src_heightmap: &[f32],
    src_grid_size: u32,
    geo_west: f64,
    geo_east: f64,
    geo_north: f64,
    geo_south: f64,
    merc_coord: &x_planets_math::TileCoord,
    out_grid_size: u32,
) -> Vec<f32> {
    let ogs = out_grid_size as usize;
    let mut out = vec![0.0f32; ogs * ogs];

    let n_3857 = (1u64 << merc_coord.z) as f64;
    let inv = 1.0 / (out_grid_size - 1) as f64;

    let geo_lon_range = geo_east - geo_west;
    let geo_lat_range = geo_north - geo_south; // positive (north > south)

    for row in 0..ogs {
        for col in 0..ogs {
            let u_merc = col as f64 * inv;
            let v_merc = row as f64 * inv;

            // Convert (u_merc, v_merc) to geographic (lon, lat).
            // Longitude is linear within a Mercator tile:
            let lon = (merc_coord.x as f64 + u_merc) / n_3857 * 360.0 - 180.0;

            // Latitude requires the Mercator Y → latitude conversion:
            let merc_y = std::f64::consts::PI
                * (1.0 - 2.0 * (merc_coord.y as f64 + v_merc) / n_3857);
            let lat = merc_y.sinh().atan().to_degrees();

            // Map (lon, lat) to the source 4326 tile's UV space.
            let u_4326 = if geo_lon_range.abs() > 1e-12 {
                (lon - geo_west) / geo_lon_range
            } else {
                0.5
            };
            // 4326 heightmap: row 0 = north, row (gs-1) = south
            let v_4326 = if geo_lat_range.abs() > 1e-12 {
                (geo_north - lat) / geo_lat_range
            } else {
                0.5
            };

            // Clamp to [0, 1] — edge samples for areas outside the 4326 tile.
            let u_clamped = u_4326.clamp(0.0, 1.0) as f32;
            let v_clamped = v_4326.clamp(0.0, 1.0) as f32;

            let h = sample_elevation_bilinear(
                src_heightmap,
                src_grid_size,
                src_grid_size,
                u_clamped,
                v_clamped,
            );
            out[row * ogs + col] = h;
        }
    }

    out
}

/// A single EPSG:4326 heightmap source for multi-source resampling.
pub struct GeoHeightmapSource<'a> {
    pub heightmap: &'a [f32],
    pub grid_size: u32,
    pub west: f64,
    pub east: f64,
    pub north: f64,
    pub south: f64,
}

/// Resample from **multiple** EPSG:4326 heightmaps to a single EPSG:3857 tile.
///
/// For each output pixel, the function finds the 4326 source whose geographic
/// bounds contain the pixel's (lon, lat) and samples from that source.  If no
/// source covers the point, the nearest edge of the nearest source is used
/// (clamping, same as the single-source variant).
///
/// This eliminates cliff walls at 4326 tile boundaries: when a 3857 tile
/// straddles two 4326 tiles, both are provided as sources and the correct
/// one is chosen per-pixel.
///
/// Pure function.
pub fn resample_geographic_to_mercator_multi(
    sources: &[GeoHeightmapSource<'_>],
    merc_coord: &x_planets_math::TileCoord,
    out_grid_size: u32,
) -> Vec<f32> {
    if sources.is_empty() {
        return vec![0.0f32; (out_grid_size * out_grid_size) as usize];
    }
    // Fast path: single source → delegate to avoid overhead.
    if sources.len() == 1 {
        let s = &sources[0];
        return resample_geographic_to_mercator(
            s.heightmap, s.grid_size,
            s.west, s.east, s.north, s.south,
            merc_coord, out_grid_size,
        );
    }

    let ogs = out_grid_size as usize;
    let mut out = vec![0.0f32; ogs * ogs];

    let n_3857 = (1u64 << merc_coord.z) as f64;
    let inv = 1.0 / (out_grid_size - 1) as f64;

    for row in 0..ogs {
        for col in 0..ogs {
            let u_merc = col as f64 * inv;
            let v_merc = row as f64 * inv;

            // Geographic coordinates of this output pixel.
            let lon = (merc_coord.x as f64 + u_merc) / n_3857 * 360.0 - 180.0;
            let merc_y = std::f64::consts::PI
                * (1.0 - 2.0 * (merc_coord.y as f64 + v_merc) / n_3857);
            let lat = merc_y.sinh().atan().to_degrees();

            // Find the source that contains (lon, lat).
            let mut best_h = 0.0f32;
            let mut found = false;
            for s in sources {
                let lon_range = s.east - s.west;
                let lat_range = s.north - s.south;

                let u_4326 = if lon_range.abs() > 1e-12 {
                    (lon - s.west) / lon_range
                } else { 0.5 };
                let v_4326 = if lat_range.abs() > 1e-12 {
                    (s.north - lat) / lat_range
                } else { 0.5 };

                // Check if this source covers the point (within [0,1]).
                if u_4326 >= -1e-6 && u_4326 <= 1.0 + 1e-6
                    && v_4326 >= -1e-6 && v_4326 <= 1.0 + 1e-6
                {
                    let u_c = u_4326.clamp(0.0, 1.0) as f32;
                    let v_c = v_4326.clamp(0.0, 1.0) as f32;
                    best_h = sample_elevation_bilinear(
                        s.heightmap, s.grid_size, s.grid_size, u_c, v_c,
                    );
                    found = true;
                    break;
                }
            }

            if !found {
                // No source covers this point — clamp to nearest source edge.
                // Use the first source (primary).
                let s = &sources[0];
                let lon_range = s.east - s.west;
                let lat_range = s.north - s.south;
                let u_4326 = if lon_range.abs() > 1e-12 {
                    (lon - s.west) / lon_range
                } else { 0.5 };
                let v_4326 = if lat_range.abs() > 1e-12 {
                    (s.north - lat) / lat_range
                } else { 0.5 };
                let u_c = u_4326.clamp(0.0, 1.0) as f32;
                let v_c = v_4326.clamp(0.0, 1.0) as f32;
                best_h = sample_elevation_bilinear(
                    s.heightmap, s.grid_size, s.grid_size, u_c, v_c,
                );
            }

            out[row * ogs + col] = best_h;
        }
    }

    out
}
