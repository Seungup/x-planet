//! Pure-function pipeline stages.
//!
//! Karpathy principle: every stage is a pure function.
//! Input → Output. No side effects. Independently testable.
//!
//! The MapEngine orchestrates these stages, but each stage
//! knows nothing about the engine or GPU.

mod frame_summary;
mod projection;
pub(crate) mod terrain_mesh;
pub(crate) mod tile_mesh;
mod tile_uniforms;
mod visible_tiles;

pub use frame_summary::*;
pub use projection::*;
pub use terrain_mesh::*;
pub use tile_mesh::*;
pub use tile_uniforms::*;
pub use visible_tiles::*;

// ═══════════════════════════════════════════════════════════════════
// Tests — 카파시 원칙: 모든 순수 함수는 즉시 테스트
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use x_planets_math::{GeoCoord, TileCoord, VisibleTile};
    use x_planets_projection::{Globe, Mercator, ProjectionPlugin};
    use crate::render::TerrainVertex;
    use crate::viewport::Viewport;

    // ── Stage 1 ────────────────────────────────────────────────

    #[test]
    fn test_visible_tiles_zoom_0_covers_world() {
        let mut vp = Viewport::new(800, 600);
        vp.center = GeoCoord::new(0.0, 0.0);
        vp.zoom = 0.0;

        let tiles = visible_tiles(&vp);
        assert!(
            tiles.iter().any(|vt| vt.coord == TileCoord::new(0, 0, 0)),
            "zoom 0 must include the single world tile, got: {:?}",
            tiles
        );
    }

    #[test]
    fn test_visible_tiles_zoom_1_has_4() {
        let mut vp = Viewport::new(800, 600);
        vp.center = GeoCoord::new(0.0, 0.0);
        vp.zoom = 1.0;

        let tiles = visible_tiles(&vp);
        assert!(tiles.len() >= 2, "zoom 1 center=(0,0) should see multiple tiles");
    }

    #[test]
    fn test_visible_tiles_increases_with_zoom() {
        let mut vp = Viewport::new(800, 600);
        vp.center = GeoCoord::new(37.5, 127.0);

        let count_z2 = { vp.zoom = 2.0; visible_tiles(&vp).len() };
        let count_z5 = { vp.zoom = 5.0; visible_tiles(&vp).len() };

        assert!(
            count_z5 >= count_z2,
            "z5 ({}) should have >= tiles than z2 ({})",
            count_z5,
            count_z2
        );
    }

    // ── Stage 2 ────────────────────────────────────────────────

    #[test]
    fn test_load_requests_excludes_cached() {
        let visible = vec![
            TileCoord::new(2, 0, 0),
            TileCoord::new(2, 1, 0),
            TileCoord::new(2, 2, 0),
        ];
        let mut cached = HashSet::new();
        cached.insert(TileCoord::new(2, 1, 0));

        let requests = compute_load_requests(&visible, &cached, &GeoCoord::new(0.0, 0.0));

        assert_eq!(requests.len(), 2);
        assert!(requests.iter().all(|r| r.coord != TileCoord::new(2, 1, 0)));
    }

    #[test]
    fn test_load_requests_sorted_by_distance() {
        let visible = vec![
            TileCoord::new(3, 0, 0),
            TileCoord::new(3, 4, 4),
        ];
        let cached = HashSet::new();
        let center = GeoCoord::new(0.0, 0.0);

        let requests = compute_load_requests(&visible, &cached, &center);

        assert_eq!(requests.len(), 2);
        assert!(
            requests[0].priority <= requests[1].priority,
            "should be sorted by distance: {:.2} <= {:.2}",
            requests[0].priority,
            requests[1].priority
        );
    }

    #[test]
    fn test_load_requests_empty_when_all_cached() {
        let visible = vec![TileCoord::new(1, 0, 0), TileCoord::new(1, 1, 0)];
        let cached: HashSet<TileCoord> = visible.iter().copied().collect();

        let requests = compute_load_requests(&visible, &cached, &GeoCoord::new(0.0, 0.0));
        assert!(requests.is_empty());
    }

    // ── Stage 3 ────────────────────────────────────────────────

    #[test]
    fn test_build_tile_mesh_single_quad() {
        let tiles = vec![TileCoord::new(0, 0, 0)];
        let (verts, indices) = build_tile_mesh(&tiles);

        assert_eq!(verts.len(), 4, "1 tile = 4 vertices");
        assert_eq!(indices.len(), 6, "1 tile = 6 indices");
    }

    #[test]
    fn test_build_tile_mesh_multiple() {
        let tiles = vec![
            TileCoord::new(1, 0, 0),
            TileCoord::new(1, 1, 0),
            TileCoord::new(1, 0, 1),
            TileCoord::new(1, 1, 1),
        ];
        let (verts, indices) = build_tile_mesh(&tiles);

        assert_eq!(verts.len(), 16, "4 tiles × 4 vertices");
        assert_eq!(indices.len(), 24, "4 tiles × 6 indices");
    }

    #[test]
    fn test_build_tile_mesh_no_degenerate_triangles() {
        for z in 0..=4u8 {
            let n = 1u32 << z;
            for x in 0..n {
                for y in 0..n {
                    let tiles = vec![TileCoord::new(z, x, y)];
                    let (verts, _) = build_tile_mesh(&tiles);

                    let w = verts[1].position[0] - verts[0].position[0];
                    let h = verts[2].position[1] - verts[0].position[1];
                    let area = w * h;

                    assert!(
                        area.abs() > 1e-10,
                        "degenerate quad at z={} x={} y={}: area={}",
                        z, x, y, area
                    );
                }
            }
        }
    }

    #[test]
    fn test_build_tile_mesh_rte_centered() {
        let tiles = vec![TileCoord::new(0, 0, 0)];
        let (verts, _) = build_tile_mesh(&tiles);

        let min_x = verts.iter().map(|v| v.position[0]).fold(f32::MAX, f32::min);
        let max_x = verts.iter().map(|v| v.position[0]).fold(f32::MIN, f32::max);
        let min_y = verts.iter().map(|v| v.position[1]).fold(f32::MAX, f32::min);
        let max_y = verts.iter().map(|v| v.position[1]).fold(f32::MIN, f32::max);

        assert!((min_x - (-0.5)).abs() < 1e-6, "min_x should be -0.5, got {}", min_x);
        assert!((max_x - 0.5).abs() < 1e-6, "max_x should be 0.5, got {}", max_x);
        assert!((min_y - (-0.5)).abs() < 1e-6, "min_y should be -0.5, got {}", min_y);
        assert!((max_y - 0.5).abs() < 1e-6, "max_y should be 0.5, got {}", max_y);
    }

    #[test]
    fn test_build_tile_mesh_rte_all_same_size_per_zoom() {
        let tiles = vec![
            TileCoord::new(1, 0, 0),
            TileCoord::new(1, 1, 0),
            TileCoord::new(1, 0, 1),
            TileCoord::new(1, 1, 1),
        ];
        let (verts, _) = build_tile_mesh(&tiles);

        for chunk in verts.chunks(4) {
            let min_x = chunk.iter().map(|v| v.position[0]).fold(f32::MAX, f32::min);
            let max_x = chunk.iter().map(|v| v.position[0]).fold(f32::MIN, f32::max);
            assert!((min_x - (-0.25)).abs() < 1e-6, "min_x should be -0.25, got {}", min_x);
            assert!((max_x - 0.25).abs() < 1e-6, "max_x should be 0.25, got {}", max_x);
        }
    }

    // ── Stage 4 ────────────────────────────────────────────────

    #[test]
    fn test_tile_uniforms_bounds() {
        let vp = glam::DMat4::IDENTITY;
        let u = tile_uniforms(&TileCoord::new(1, 0, 0), 1.0, &vp);
        assert!((u.bounds[0] - 0.0).abs() < 1e-6);
        assert!((u.bounds[1] - 0.0).abs() < 1e-6);
        assert!((u.bounds[2] - 0.5).abs() < 1e-6);
        assert!((u.bounds[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tile_uniforms_opacity() {
        let vp = glam::DMat4::IDENTITY;
        let u = tile_uniforms(&TileCoord::new(0, 0, 0), 0.75, &vp);
        assert!((u.meta[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_tile_uniforms_mvp_precision() {
        let vp_f64 = {
            let mut v = Viewport::new(800, 600);
            v.center = GeoCoord::new(0.0, 0.0);
            v.zoom = 18.0;
            v.to_view_proj_f64()
        };
        let coord = TileCoord::new(18, 131072, 131072);
        let u = tile_uniforms(&coord, 1.0, &vp_f64);
        let mvp_sum: f32 = u.mvp.iter().map(|v| v.abs()).sum();
        assert!(mvp_sum > 1.0, "MVP should be a valid transform");
    }

    #[test]
    fn test_viewport_uniforms_resolution() {
        let vp = Viewport::new(1920, 1080);
        let u = viewport_uniforms(&vp);
        assert_eq!(u.resolution[0], 1920.0);
        assert_eq!(u.resolution[1], 1080.0);
        assert!((u.resolution[2] - 1.0 / 1920.0).abs() < 1e-6);
    }

    // ── Stage 5: Projection ────────────────────────────────────

    #[test]
    fn test_mercator_roundtrip_1000_points() {
        let proj = Mercator;
        let points = generate_test_grid(20, 50);

        let max_err = verify_projection_roundtrip(&proj, &points);
        assert!(
            max_err < 1e-8,
            "Mercator roundtrip max error: {:.2e} (should be < 1e-8)",
            max_err
        );
    }

    #[test]
    fn test_globe_roundtrip_1000_points() {
        let proj = Globe;
        let points = generate_test_grid(20, 50);

        let max_err = verify_projection_roundtrip(&proj, &points);
        assert!(
            max_err < 1e-10,
            "Globe roundtrip max error: {:.2e} (should be < 1e-10)",
            max_err
        );
    }

    #[test]
    fn test_mercator_known_values() {
        let proj = Mercator;

        let origin = proj.project_cpu(glam::DVec3::new(0.0, 0.0, 0.0));
        assert!((origin.x - 0.5).abs() < 1e-10, "origin.x = {}", origin.x);
        assert!((origin.y - 0.5).abs() < 1e-10, "origin.y = {}", origin.y);

        let left = proj.project_cpu(glam::DVec3::new(0.0, -180.0, 0.0));
        assert!(left.x.abs() < 1e-10, "left.x = {}", left.x);

        let right = proj.project_cpu(glam::DVec3::new(0.0, 180.0, 0.0));
        assert!((right.x - 1.0).abs() < 1e-10, "right.x = {}", right.x);
    }

    #[test]
    fn test_project_positions_cpu_batch() {
        let proj = Mercator;
        let positions = vec![
            glam::DVec3::new(0.0, 0.0, 0.0),
            glam::DVec3::new(45.0, 90.0, 0.0),
            glam::DVec3::new(-30.0, -60.0, 0.0),
        ];

        let results = project_positions_cpu(&proj, &positions);

        assert_eq!(results.len(), 3);
        for (pos, result) in positions.iter().zip(results.iter()) {
            let expected = proj.project_cpu(*pos);
            assert!(
                (*result - expected).length() < 1e-15,
                "batch should match individual"
            );
        }
    }

    // ── Stage 6: FrameSummary ──────────────────────────────────

    #[test]
    fn test_frame_summary_display() {
        let summary = FrameSummary {
            visible_tile_count: 16,
            cached_tile_count: 12,
            load_requests: 4,
            zoom: 3.5,
            center: GeoCoord::new(37.57, 126.98),
        };

        let s = format!("{}", summary);
        assert!(s.contains("z=3.5"));
        assert!(s.contains("visible=16"));
        assert!(s.contains("pending=4"));
    }

    // ── Fallback UV / resolve_fallbacks tests ────────────────

    #[test]
    fn test_fallback_uv_same_tile() {
        let tile = TileCoord::new(5, 10, 15);
        let uv = fallback_uv_rect(&tile, &tile);
        assert!((uv[0] - 0.0).abs() < 1e-6);
        assert!((uv[1] - 0.0).abs() < 1e-6);
        assert!((uv[2] - 1.0).abs() < 1e-6);
        assert!((uv[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_fallback_uv_one_level_up() {
        let parent = TileCoord::new(2, 0, 0);
        let child = TileCoord::new(3, 1, 0);

        let uv = fallback_uv_rect(&child, &parent);
        assert!((uv[0] - 0.5).abs() < 1e-5, "u_min = {}", uv[0]);
        assert!((uv[1] - 0.0).abs() < 1e-5, "v_min = {}", uv[1]);
        assert!((uv[2] - 1.0).abs() < 1e-5, "u_max = {}", uv[2]);
        assert!((uv[3] - 0.5).abs() < 1e-5, "v_max = {}", uv[3]);
    }

    #[test]
    fn test_fallback_uv_two_levels_up() {
        let ancestor = TileCoord::new(1, 0, 0);
        let tile = TileCoord::new(3, 1, 1);

        let uv = fallback_uv_rect(&tile, &ancestor);
        assert!((uv[0] - 0.25).abs() < 1e-5, "u_min = {}", uv[0]);
        assert!((uv[1] - 0.25).abs() < 1e-5, "v_min = {}", uv[1]);
        assert!((uv[2] - 0.5).abs() < 1e-5, "u_max = {}", uv[2]);
        assert!((uv[3] - 0.5).abs() < 1e-5, "v_max = {}", uv[3]);
    }

    #[test]
    fn test_resolve_fallbacks_complete_coverage() {
        let visible: Vec<VisibleTile> = vec![
            VisibleTile::canonical(TileCoord::new(2, 0, 0)),
            VisibleTile::canonical(TileCoord::new(2, 1, 0)),
            VisibleTile::canonical(TileCoord::new(2, 0, 1)),
        ];
        let available: HashSet<TileCoord> = visible.iter().map(|vt| vt.coord).collect();
        let result = resolve_fallbacks(&visible, &available);

        assert_eq!(result.len(), 3);
        for r in &result {
            assert_eq!(r.coord, r.texture_coord);
            assert!((r.uv_rect[0] - 0.0).abs() < 1e-6);
            assert!((r.uv_rect[2] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_resolve_fallbacks_uses_parent() {
        let child = TileCoord::new(3, 2, 3);
        let parent = child.parent().unwrap();

        let visible = vec![VisibleTile::canonical(child)];
        let mut available = HashSet::new();
        available.insert(parent);

        let result = resolve_fallbacks(&visible, &available);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].coord, child);
        assert_eq!(result[0].texture_coord, parent);
        assert!(
            result[0].uv_rect[0] != 0.0 || result[0].uv_rect[1] != 0.0
                || result[0].uv_rect[2] != 1.0 || result[0].uv_rect[3] != 1.0,
            "Fallback UV should differ from identity"
        );
    }

    #[test]
    fn test_resolve_fallbacks_missing_all_returns_empty() {
        let visible = vec![VisibleTile::canonical(TileCoord::new(5, 10, 10))];
        let available = HashSet::new();
        let result = resolve_fallbacks(&visible, &available);
        assert!(result.is_empty());
    }

    // ── Terrain mesh tests ──────────────────────────────────────

    #[test]
    fn test_build_terrain_mesh_dimensions() {
        let coord = TileCoord::new(2, 1, 1);
        let elevation = vec![0.0f32; 256 * 256];
        let identity_uv = [0.0, 0.0, 1.0, 1.0];
        let (verts, indices) = build_terrain_mesh(&coord, &elevation, 256, 256, 1e-5, identity_uv);
        let g = TERRAIN_GRID_SIZE;
        let surface_verts = (g + 1) * (g + 1);
        let skirt_verts = 4 * g * 2;
        assert!(
            verts.len() == (surface_verts + skirt_verts) as usize,
            "expected {} verts ({}+{}), got {}",
            surface_verts + skirt_verts, surface_verts, skirt_verts, verts.len()
        );
        let surface_indices = g * g * 6;
        let skirt_indices = 4 * g * 6;
        assert!(
            indices.len() == (surface_indices + skirt_indices) as usize,
            "expected {} indices, got {}",
            surface_indices + skirt_indices, indices.len()
        );
    }

    #[test]
    fn test_build_terrain_mesh_flat_has_zero_z() {
        let coord = TileCoord::new(0, 0, 0);
        let elevation = vec![0.0f32; 4];
        let identity_uv = [0.0, 0.0, 1.0, 1.0];
        let (verts, _) = build_terrain_mesh(&coord, &elevation, 2, 2, 1e-5, identity_uv);
        let g = TERRAIN_GRID_SIZE;
        let surface_count = ((g + 1) * (g + 1)) as usize;
        for v in &verts[..surface_count] {
            assert!(
                v.position[2].abs() < 1e-10,
                "flat terrain should have z≈0, got {}",
                v.position[2]
            );
        }
    }

    #[test]
    fn test_build_terrain_mesh_elevated() {
        let coord = TileCoord::new(0, 0, 0);
        let elevation = vec![1000.0f32; 4];
        let scale = 1e-5;
        let identity_uv = [0.0, 0.0, 1.0, 1.0];
        let (verts, _) = build_terrain_mesh(&coord, &elevation, 2, 2, scale, identity_uv);
        let g = TERRAIN_GRID_SIZE;
        let surface_count = ((g + 1) * (g + 1)) as usize;
        let expected_z = 1000.0 * scale;
        for v in &verts[..surface_count] {
            assert!(
                (v.position[2] - expected_z).abs() < 1e-6,
                "expected z={}, got {}",
                expected_z,
                v.position[2]
            );
        }
    }

    #[test]
    fn test_build_terrain_mesh_parent_fallback_uv() {
        let parent = TileCoord::new(1, 0, 0);
        let child = TileCoord::new(2, 1, 1);
        let elevation = vec![0.0, 1000.0, 2000.0, 3000.0];

        let scale = 1e-5;
        let elev_uv = fallback_uv_rect(&child, &parent);
        assert!((elev_uv[0] - 0.5).abs() < 1e-4, "u_min={}", elev_uv[0]);
        assert!((elev_uv[1] - 0.5).abs() < 1e-4, "v_min={}", elev_uv[1]);

        let (verts, _) = build_terrain_mesh(&child, &elevation, 2, 2, scale, elev_uv);

        let g = TERRAIN_GRID_SIZE;
        let br_idx = g as usize * (g as usize + 1) + g as usize;
        let br_elev = verts[br_idx].position[2] / scale;
        assert!(
            (br_elev - 3000.0).abs() < 50.0,
            "bottom-right should be ~3000m (parent's BR corner), got {}m",
            br_elev
        );

        let tl_elev = verts[0].position[2] / scale;
        assert!(
            (tl_elev - 1500.0).abs() < 100.0,
            "top-left should be ~1500m (parent center), got {}m",
            tl_elev
        );
    }

    #[test]
    fn test_build_terrain_mesh_edge_normals_smooth() {
        let coord = TileCoord::new(5, 16, 16);
        let grid_size = 33u32;
        let mut elevation = vec![0.0f32; (grid_size * grid_size) as usize];
        for row in 0..grid_size {
            for col in 0..grid_size {
                let v = row as f32 / (grid_size - 1) as f32;
                elevation[(row * grid_size + col) as usize] = v * 1000.0;
            }
        }
        let scale = 1e-5;
        let identity_uv = [0.0, 0.0, 1.0, 1.0];
        let (verts, _) = build_terrain_mesh(
            &coord, &elevation, grid_size, grid_size, scale, identity_uv,
        );
        let vs = (TERRAIN_GRID_SIZE + 1) as usize;

        let mid_x = vs / 2;
        let edge_top = verts[0 * vs + mid_x].normal;
        let interior_1 = verts[1 * vs + mid_x].normal;

        let dot_top = edge_top[0] * interior_1[0]
            + edge_top[1] * interior_1[1]
            + edge_top[2] * interior_1[2];
        assert!(
            dot_top > 0.99,
            "Top edge normal should match interior (dot={:.4}), edge={:?}, interior={:?}",
            dot_top, edge_top, interior_1,
        );

        let last = vs - 1;
        let edge_bottom = verts[last * vs + mid_x].normal;
        let interior_last = verts[(last - 1) * vs + mid_x].normal;

        let dot_bottom = edge_bottom[0] * interior_last[0]
            + edge_bottom[1] * interior_last[1]
            + edge_bottom[2] * interior_last[2];
        assert!(
            dot_bottom > 0.99,
            "Bottom edge normal should match interior (dot={:.4}), edge={:?}, interior={:?}",
            dot_bottom, edge_bottom, interior_last,
        );
    }

    #[test]
    fn test_compute_height_scale() {
        let scale = compute_height_scale(1.0);
        let expected = (1.0 / 40_075_000.0_f64) as f32;
        assert!((scale - expected).abs() < 1e-12);

        let scale_2x = compute_height_scale(2.0);
        assert!((scale_2x - 2.0 * scale).abs() < 1e-12);
    }

    // ── Stage 4d: rasterize_qm_to_heightmap ────────────────────

    #[test]
    fn test_rasterize_flat_triangle() {
        let height = 500.0f32;
        let vertices = vec![
            TerrainVertex { position: [-0.5, -0.5, height], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 0.0] },
            TerrainVertex { position: [0.5, -0.5, height], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 0.0] },
            TerrainVertex { position: [-0.5, 0.5, height], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 1.0] },
        ];
        let indices = vec![0, 1, 2];
        let hm = rasterize_qm_to_heightmap(&vertices, &indices, 5);
        let inside_count = hm.iter().filter(|&&h| (h - height).abs() < 1.0).count();
        assert!(
            inside_count >= 6,
            "at least 6 of 25 cells should be inside triangle, got {}",
            inside_count
        );
    }

    #[test]
    fn test_rasterize_two_triangles_full_coverage() {
        let h = 1000.0f32;
        let vertices = vec![
            TerrainVertex { position: [-0.5, -0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 0.0] },
            TerrainVertex { position: [0.5, -0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 0.0] },
            TerrainVertex { position: [-0.5, 0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 1.0] },
            TerrainVertex { position: [0.5, 0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 1.0] },
        ];
        let indices = vec![0, 1, 2, 1, 3, 2];
        let hm = rasterize_qm_to_heightmap(&vertices, &indices, 9);
        for (i, &val) in hm.iter().enumerate() {
            assert!(
                (val - h).abs() < 1.0,
                "cell {} should be {}m, got {}m",
                i, h, val
            );
        }
    }

    #[test]
    fn test_rasterize_sloped_surface() {
        let vertices = vec![
            TerrainVertex { position: [-0.5, -0.5, 0.0], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 0.0] },
            TerrainVertex { position: [0.5, -0.5, 1000.0], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 0.0] },
            TerrainVertex { position: [-0.5, 0.5, 0.0], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 1.0] },
            TerrainVertex { position: [0.5, 0.5, 1000.0], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 1.0] },
        ];
        let indices = vec![0, 1, 2, 1, 3, 2];
        let gs = 5u32;
        let hm = rasterize_qm_to_heightmap(&vertices, &indices, gs);

        for col in 0..gs as usize {
            let u = col as f32 / (gs - 1) as f32;
            let expected = u * 1000.0;
            let actual = hm[col];
            assert!(
                (actual - expected).abs() < 50.0,
                "col {} (u={:.2}): expected ~{:.0}m, got {:.0}m",
                col, u, expected, actual
            );
        }
    }

    #[test]
    fn test_rasterize_grid_size_matches() {
        let h = 100.0f32;
        let vertices = vec![
            TerrainVertex { position: [-0.5, -0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 0.0] },
            TerrainVertex { position: [0.5, -0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 0.0] },
            TerrainVertex { position: [0.5, 0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 1.0] },
            TerrainVertex { position: [-0.5, 0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 1.0] },
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];

        for gs in [5u32, 17, 33, 65] {
            let hm = rasterize_qm_to_heightmap(&vertices, &indices, gs);
            assert_eq!(
                hm.len(),
                (gs * gs) as usize,
                "grid_size={}: expected {} cells, got {}",
                gs, gs * gs, hm.len()
            );
        }
    }

    // ── Stage 4e: resample_geographic_to_mercator ───────────────

    #[test]
    fn test_resample_constant_height() {
        let h = 2500.0f32;
        let gs = 9u32;
        let src = vec![h; (gs * gs) as usize];
        let (west, east, north, south) = (0.0, 5.625, 5.625, 0.0);
        let merc = TileCoord::new(6, 33, 31);
        let out = resample_geographic_to_mercator(
            &src, gs, west, east, north, south, &merc, gs,
        );
        assert_eq!(out.len(), (gs * gs) as usize);
        for (i, &val) in out.iter().enumerate() {
            assert!(
                (val - h).abs() < 1.0,
                "cell {}: expected {}m, got {}m",
                i, h, val
            );
        }
    }

    #[test]
    fn test_resample_output_size() {
        let gs = 17u32;
        let src = vec![0.0f32; (gs * gs) as usize];
        let merc = TileCoord::new(5, 16, 15);
        for out_gs in [5u32, 17, 33, 65] {
            let out = resample_geographic_to_mercator(
                &src, gs, -5.625, 0.0, 5.625, 0.0, &merc, out_gs,
            );
            assert_eq!(
                out.len(),
                (out_gs * out_gs) as usize,
                "out_grid_size={}: expected {} cells",
                out_gs, out_gs * out_gs
            );
        }
    }

    #[test]
    fn test_resample_north_south_gradient() {
        let gs = 33u32;
        let mut src = vec![0.0f32; (gs * gs) as usize];
        for row in 0..gs {
            for col in 0..gs {
                src[(row * gs + col) as usize] = row as f32 / (gs - 1) as f32 * 1000.0;
            }
        }
        let west = -5.625;
        let east = 0.0;
        let north = 5.625;
        let south = 0.0;
        let merc = TileCoord::new(6, 31, 31);
        let out_gs = 17u32;
        let out = resample_geographic_to_mercator(
            &src, gs, west, east, north, south, &merc, out_gs,
        );
        let top_avg: f32 = (0..out_gs).map(|c| out[c as usize]).sum::<f32>() / out_gs as f32;
        let bot_avg: f32 = (0..out_gs)
            .map(|c| out[((out_gs - 1) * out_gs + c) as usize])
            .sum::<f32>()
            / out_gs as f32;
        assert!(
            bot_avg > top_avg,
            "bottom average ({:.1}) should be > top average ({:.1})",
            bot_avg, top_avg
        );
    }

    #[test]
    fn test_resample_equatorial_symmetry() {
        let gs = 33u32;
        let mut src = vec![0.0f32; (gs * gs) as usize];
        let mid = (gs - 1) as f32 / 2.0;
        for row in 0..gs {
            for col in 0..gs {
                let v = (row as f32 - mid).abs() / mid;
                src[(row * gs + col) as usize] = (1.0 - v) * 1000.0;
            }
        }
        let merc = TileCoord::new(7, 64, 63);
        let out = resample_geographic_to_mercator(
            &src, gs, -2.8125, 2.8125, 2.8125, -2.8125, &merc, 17,
        );
        let mid_row = 8;
        let center_h = out[mid_row * 17 + 8];
        let edge_h = out[0 * 17 + 8];
        assert!(
            center_h > edge_h,
            "center ({:.1}) should be higher than edge ({:.1})",
            center_h, edge_h
        );
    }

    // ── sample_elevation_bilinear ───────────────────────────────

    #[test]
    fn test_bilinear_corners() {
        let elev = vec![0.0, 100.0, 200.0, 300.0];
        assert!((terrain_mesh::sample_elevation_bilinear(&elev, 2, 2, 0.0, 0.0) - 0.0).abs() < 1e-3);
        assert!((terrain_mesh::sample_elevation_bilinear(&elev, 2, 2, 1.0, 0.0) - 100.0).abs() < 1e-3);
        assert!((terrain_mesh::sample_elevation_bilinear(&elev, 2, 2, 0.0, 1.0) - 200.0).abs() < 1e-3);
        assert!((terrain_mesh::sample_elevation_bilinear(&elev, 2, 2, 1.0, 1.0) - 300.0).abs() < 1e-3);
    }

    #[test]
    fn test_bilinear_center() {
        let elev = vec![0.0, 100.0, 200.0, 300.0];
        let center = terrain_mesh::sample_elevation_bilinear(&elev, 2, 2, 0.5, 0.5);
        let expected = (0.0 + 100.0 + 200.0 + 300.0) / 4.0;
        assert!(
            (center - expected).abs() < 1e-3,
            "center should be {}, got {}",
            expected, center
        );
    }

    #[test]
    fn test_bilinear_edge_midpoint() {
        let elev = vec![0.0, 100.0, 200.0, 300.0];
        let mid_top = terrain_mesh::sample_elevation_bilinear(&elev, 2, 2, 0.5, 0.0);
        assert!(
            (mid_top - 50.0).abs() < 1e-3,
            "top edge midpoint should be 50, got {}",
            mid_top
        );
    }

    #[test]
    fn test_bilinear_constant_surface() {
        let h = 777.0f32;
        let elev = vec![h; 65 * 65];
        for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let val = terrain_mesh::sample_elevation_bilinear(&elev, 65, 65, u, v);
                assert!(
                    (val - h).abs() < 1e-3,
                    "constant surface at ({}, {}): expected {}, got {}",
                    u, v, h, val
                );
            }
        }
    }

    #[test]
    fn test_bilinear_larger_grid() {
        let elev = vec![
            0.0, 100.0, 200.0,
            300.0, 400.0, 500.0,
            600.0, 700.0, 800.0,
        ];
        let center = terrain_mesh::sample_elevation_bilinear(&elev, 3, 3, 0.5, 0.5);
        assert!(
            (center - 400.0).abs() < 1e-3,
            "3×3 center should be 400, got {}",
            center
        );
    }

    // ── Over-zoom fallback integration ──────────────────────────

    #[test]
    fn test_overzoom_fallback_heightmap_workflow() {
        let parent_gs = 33u32;
        let mut parent_hm = vec![0.0f32; (parent_gs * parent_gs) as usize];
        for row in 0..parent_gs {
            for col in 0..parent_gs {
                let u = col as f32 / (parent_gs - 1) as f32;
                let v = row as f32 / (parent_gs - 1) as f32;
                parent_hm[(row * parent_gs + col) as usize] = (u + v) * 500.0;
            }
        }

        let parent = TileCoord::new(13, 4096, 3072);
        let child = TileCoord::new(14, 8192, 6144);

        let elev_uv = fallback_uv_rect(&child, &parent);
        assert!((elev_uv[0] - 0.0).abs() < 1e-4, "u_min={}", elev_uv[0]);
        assert!((elev_uv[1] - 0.0).abs() < 1e-4, "v_min={}", elev_uv[1]);
        assert!((elev_uv[2] - 0.5).abs() < 1e-4, "u_max={}", elev_uv[2]);
        assert!((elev_uv[3] - 0.5).abs() < 1e-4, "v_max={}", elev_uv[3]);

        let scale = compute_height_scale(1.5);
        let (verts, _) = build_terrain_mesh(
            &child, &parent_hm, parent_gs, parent_gs, scale, elev_uv,
        );

        let tl_h = verts[0].position[2] / scale;
        let g = TERRAIN_GRID_SIZE;
        let br_idx = g as usize * (g as usize + 1) + g as usize;
        let br_h = verts[br_idx].position[2] / scale;

        assert!(tl_h.abs() < 20.0, "child TL should be ~0m, got {}m", tl_h);
        assert!(
            (br_h - 500.0).abs() < 50.0,
            "child BR should be ~500m, got {}m",
            br_h
        );
    }

    #[test]
    fn test_overzoom_multiple_levels() {
        let ancestor = TileCoord::new(13, 4096, 3072);
        let grandchild = TileCoord::new(15, 16384, 12288);

        let elev_uv = fallback_uv_rect(&grandchild, &ancestor);
        assert!((elev_uv[0] - 0.0).abs() < 1e-4, "u_min={}", elev_uv[0]);
        assert!((elev_uv[1] - 0.0).abs() < 1e-4, "v_min={}", elev_uv[1]);
        assert!((elev_uv[2] - 0.25).abs() < 1e-4, "u_max={}", elev_uv[2]);
        assert!((elev_uv[3] - 0.25).abs() < 1e-4, "v_max={}", elev_uv[3]);
    }

    // ── Skirt exclusion regression ────────────

    #[test]
    fn test_rasterize_skirt_excluded_preserves_edge_heights() {
        let h = 500.0f32;
        let skirt_depth = 300.0f32;

        let surface_verts = vec![
            TerrainVertex { position: [-0.5, -0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 0.0] },
            TerrainVertex { position: [ 0.5, -0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 0.0] },
            TerrainVertex { position: [ 0.5,  0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 1.0] },
            TerrainVertex { position: [-0.5,  0.5, h], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 1.0] },
        ];
        let surface_indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3];
        let surface_idx_count = surface_indices.len();

        let mut all_verts = surface_verts.clone();
        let skirt_a_idx = all_verts.len() as u32;
        all_verts.push(TerrainVertex {
            position: [-0.5, -0.5, h - skirt_depth],
            normal: [0.0, 0.0, -1.0],
            tex_coord: [0.3, 0.0],
        });
        let skirt_b_idx = all_verts.len() as u32;
        all_verts.push(TerrainVertex {
            position: [-0.5, 0.5, h - skirt_depth],
            normal: [0.0, 0.0, -1.0],
            tex_coord: [0.3, 1.0],
        });
        let mut all_indices = surface_indices.clone();
        all_indices.extend_from_slice(&[0, skirt_a_idx, 3, 3, skirt_a_idx, skirt_b_idx]);

        let gs = 17u32;

        let without_skirts = rasterize_qm_to_heightmap(
            &all_verts, &all_indices[..surface_idx_count], gs,
        );
        for row in 0..gs {
            for col in 0..gs {
                let idx = (row * gs + col) as usize;
                assert!(
                    (without_skirts[idx] - h).abs() < 1.0,
                    "skirt-excluded: ({},{}) should be ~{}m, got {}m",
                    row, col, h, without_skirts[idx]
                );
            }
        }

        let with_skirts = rasterize_qm_to_heightmap(&all_verts, &all_indices, gs);
        let corrupted_col = 2usize;
        let mut found_corruption = false;
        for row in 0..gs {
            let idx = (row as usize) * (gs as usize) + corrupted_col;
            if (with_skirts[idx] - h).abs() > 10.0 {
                found_corruption = true;
                break;
            }
        }
        assert!(
            found_corruption,
            "inset-UV skirts (u=0.3) should corrupt cells near col=2 (u=0.125)",
        );
    }

    #[test]
    fn test_build_terrain_mesh_from_qm_surface_vs_total_indices() {
        let coord = TileCoord::new(5, 16, 12);
        let qm = x_planets_tiles::DecodedQuantizedMesh {
            coord,
            header: x_planets_tiles::quantized_mesh::QmHeader {
                center_x: 0.0, center_y: 0.0, center_z: 0.0,
                min_height: 0.0, max_height: 1000.0,
                bounding_sphere_radius: 1.0,
                horizon_occlusion_point_x: 0.0,
                horizon_occlusion_point_y: 0.0,
                horizon_occlusion_point_z: 0.0,
            },
            u: vec![0, 32767, 32767, 0],
            v: vec![0, 0, 32767, 32767],
            height: vec![16383, 16383, 16383, 16383],
            indices: vec![0, 1, 2, 0, 2, 3],
            west_indices: vec![0, 3],
            south_indices: vec![0, 1],
            east_indices: vec![1, 2],
            north_indices: vec![3, 2],
            oct_normals: None,
        };

        let surface_idx_count = qm.indices.len();
        assert_eq!(surface_idx_count, 6, "surface should have 6 indices (2 triangles)");

        let (_vertices, indices) = build_terrain_mesh_from_qm(&coord, &qm);
        assert!(
            indices.len() > surface_idx_count,
            "total indices ({}) should be > surface-only ({}) due to skirts",
            indices.len(), surface_idx_count
        );

        let surface_only = &indices[..surface_idx_count];
        assert_eq!(surface_only.len(), 6);
        for &idx in surface_only {
            assert!(idx < 4, "surface index {} should be < 4", idx);
        }
    }

    // ── project_qm_vertices_4326_to_3857 ──────────────────────

    fn geo_tile_bounds(gx: u32, gy: u32, gz: u8) -> (f64, f64, f64, f64) {
        let n_x = (1u32 << (gz + 1)) as f64;
        let n_y = (1u32 << gz) as f64;
        let west  = gx as f64 / n_x * 360.0 - 180.0;
        let east  = (gx + 1) as f64 / n_x * 360.0 - 180.0;
        let north = 90.0 - gy as f64 / n_y * 180.0;
        let south = 90.0 - (gy + 1) as f64 / n_y * 180.0;
        (west, east, north, south)
    }

    #[test]
    fn test_project_4326_to_3857_center_maps_correctly() {
        let merc = TileCoord::new(6, 33, 31);
        let (geo_west, geo_east, geo_north, geo_south) = geo_tile_bounds(33, 15, 5);

        let mut verts = vec![
            TerrainVertex {
                position: [0.0, 0.0, 1000.0],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.5, 0.5],
            },
        ];

        project_qm_vertices_4326_to_3857(
            &mut verts, &merc,
            geo_west, geo_east, geo_north, geo_south,
        );

        assert!(
            (verts[0].tex_coord[0] - 0.5).abs() < 0.01,
            "projected u should be 0.5, got {}",
            verts[0].tex_coord[0]
        );
        assert!(
            verts[0].tex_coord[1] > -0.5 && verts[0].tex_coord[1] < 1.5,
            "projected v should be approximately in tile range, got {}",
            verts[0].tex_coord[1]
        );
        assert_eq!(verts[0].position[2], 1000.0);
    }

    #[test]
    fn test_project_4326_to_3857_preserves_height() {
        let merc = TileCoord::new(5, 16, 16);
        let (geo_west, geo_east, geo_north, geo_south) = geo_tile_bounds(16, 7, 4);

        let heights = [0.0, 100.0, 500.0, 8848.0, -420.0];
        for &h in &heights {
            let mut verts = vec![TerrainVertex {
                position: [0.0, 0.0, h],
                normal: [0.0, 0.0, 1.0],
                tex_coord: [0.5, 0.5],
            }];
            project_qm_vertices_4326_to_3857(
                &mut verts, &merc,
                geo_west, geo_east, geo_north, geo_south,
            );
            assert_eq!(
                verts[0].position[2], h,
                "height {}m should be preserved after projection", h
            );
        }
    }

    #[test]
    fn test_project_4326_to_3857_edge_continuity() {
        let (west_a, east_a, north_a, south_a) = geo_tile_bounds(16, 7, 4);
        let (west_b, east_b, north_b, south_b) = geo_tile_bounds(16, 8, 4);

        assert!(
            (south_a - north_b).abs() < 1e-10,
            "tiles should share edge: A.south={}, B.north={}",
            south_a, north_b
        );

        let merc = TileCoord::new(5, 32, 16);
        let h = 750.0f32;
        let u_shared = 0.3;

        let mut vert_a = vec![TerrainVertex {
            position: [0.0, 0.0, h],
            normal: [0.0, 0.0, 1.0],
            tex_coord: [u_shared, 1.0],
        }];
        project_qm_vertices_4326_to_3857(
            &mut vert_a, &merc,
            west_a, east_a, north_a, south_a,
        );

        let mut vert_b = vec![TerrainVertex {
            position: [0.0, 0.0, h],
            normal: [0.0, 0.0, 1.0],
            tex_coord: [u_shared, 0.0],
        }];
        project_qm_vertices_4326_to_3857(
            &mut vert_b, &merc,
            west_b, east_b, north_b, south_b,
        );

        let dx = (vert_a[0].position[0] - vert_b[0].position[0]).abs();
        let dy = (vert_a[0].position[1] - vert_b[0].position[1]).abs();
        assert!(
            dx < 1e-6 && dy < 1e-6,
            "shared edge vertices should project to same position: \
             A=({}, {}), B=({}, {}), diff=({}, {})",
            vert_a[0].position[0], vert_a[0].position[1],
            vert_b[0].position[0], vert_b[0].position[1],
            dx, dy,
        );
        assert_eq!(vert_a[0].position[2], vert_b[0].position[2]);
    }

    #[test]
    fn test_project_4326_to_3857_full_tile_coverage() {
        let merc = TileCoord::new(6, 33, 31);
        let (geo_west, geo_east, geo_north, geo_south) = geo_tile_bounds(33, 15, 5);

        let mut verts = vec![
            TerrainVertex { position: [0.0, 0.0, 0.0], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 0.0] },
            TerrainVertex { position: [0.0, 0.0, 0.0], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 0.0] },
            TerrainVertex { position: [0.0, 0.0, 0.0], normal: [0.0, 0.0, 1.0], tex_coord: [0.0, 1.0] },
            TerrainVertex { position: [0.0, 0.0, 0.0], normal: [0.0, 0.0, 1.0], tex_coord: [1.0, 1.0] },
        ];

        project_qm_vertices_4326_to_3857(
            &mut verts, &merc,
            geo_west, geo_east, geo_north, geo_south,
        );

        assert!((verts[0].tex_coord[0] - 0.0).abs() < 0.01, "NW u={}", verts[0].tex_coord[0]);
        assert!((verts[1].tex_coord[0] - 1.0).abs() < 0.01, "NE u={}", verts[1].tex_coord[0]);

        let v_min = verts.iter().map(|v| v.tex_coord[1]).fold(f32::MAX, f32::min);
        let v_max = verts.iter().map(|v| v.tex_coord[1]).fold(f32::MIN, f32::max);
        assert!(
            v_min < 0.1 && v_max > 0.9,
            "projected v range [{}, {}] should approximately cover [0, 1]",
            v_min, v_max
        );
    }

    // ── Multi-source resampling tests ─────────────────────────

    #[test]
    fn test_multi_source_single_source_matches_original() {
        let grid_size = 5u32;
        let heightmap: Vec<f32> = (0..(grid_size * grid_size))
            .map(|i| 100.0 + i as f32 * 10.0)
            .collect();
        let merc_coord = TileCoord::new(3, 4, 3);
        let (west, east, north, south) = (-45.0, -22.5, 45.0, 33.75);
        let out_grid_size = 5u32;

        let single = resample_geographic_to_mercator(
            &heightmap, grid_size,
            west, east, north, south,
            &merc_coord, out_grid_size,
        );
        let multi = resample_geographic_to_mercator_multi(
            &[GeoHeightmapSource {
                heightmap: &heightmap,
                grid_size,
                west, east, north, south,
            }],
            &merc_coord, out_grid_size,
        );

        for (i, (s, m)) in single.iter().zip(multi.iter()).enumerate() {
            assert!(
                (s - m).abs() < 1e-4,
                "Mismatch at index {}: single={}, multi={}",
                i, s, m,
            );
        }
    }

    #[test]
    fn test_multi_source_two_tiles_no_clamping() {
        let grid_size = 5u32;
        let heightmap_a: Vec<f32> = vec![1000.0; (grid_size * grid_size) as usize];
        let heightmap_b: Vec<f32> = vec![500.0; (grid_size * grid_size) as usize];

        let source_a = GeoHeightmapSource {
            heightmap: &heightmap_a, grid_size,
            west: 123.75, east: 135.0, north: 45.0, south: 33.75,
        };
        let source_b = GeoHeightmapSource {
            heightmap: &heightmap_b, grid_size,
            west: 123.75, east: 135.0, north: 33.75, south: 22.5,
        };

        let merc_coord = TileCoord::new(5, 27, 12);
        let out_grid_size = 9u32;

        let multi = resample_geographic_to_mercator_multi(
            &[source_a, source_b],
            &merc_coord, out_grid_size,
        );

        let single = resample_geographic_to_mercator(
            &heightmap_a, grid_size,
            123.75, 135.0, 45.0, 33.75,
            &merc_coord, out_grid_size,
        );

        let has_500 = multi.iter().any(|&h| (h - 500.0).abs() < 1.0);
        let single_all_1000 = single.iter().all(|&h| (h - 1000.0).abs() < 1.0);

        assert!(has_500, "Multi-source should sample from tile B (500m) for southern pixels");
        assert!(single_all_1000, "Single-source should clamp all to tile A (1000m)");
    }

    #[test]
    fn test_multi_source_smooth_boundary() {
        let grid_size = 33u32;
        let gs = grid_size as usize;

        let heightmap_a: Vec<f32> = (0..gs * gs)
            .map(|i| {
                let row = i / gs;
                let t = row as f32 / (gs - 1) as f32;
                1000.0 - 500.0 * t
            })
            .collect();

        let heightmap_b: Vec<f32> = (0..gs * gs)
            .map(|i| {
                let row = i / gs;
                let t = row as f32 / (gs - 1) as f32;
                500.0 - 500.0 * t
            })
            .collect();

        assert!(
            (heightmap_a[gs * (gs - 1)] - heightmap_b[0]).abs() < 1.0,
            "Tile edge heights should match"
        );

        let source_a = GeoHeightmapSource {
            heightmap: &heightmap_a, grid_size,
            west: 0.0, east: 11.25, north: 45.0, south: 33.75,
        };
        let source_b = GeoHeightmapSource {
            heightmap: &heightmap_b, grid_size,
            west: 0.0, east: 11.25, north: 33.75, south: 22.5,
        };

        let merc_coord = TileCoord::new(5, 16, 12);
        let out_grid_size = 33u32;
        let multi = resample_geographic_to_mercator_multi(
            &[source_a, source_b],
            &merc_coord, out_grid_size,
        );

        let ogs = out_grid_size as usize;
        let mut max_jump = 0.0f32;
        for row in 1..ogs {
            for col in 0..ogs {
                let h_prev = multi[(row - 1) * ogs + col];
                let h_curr = multi[row * ogs + col];
                let jump = (h_curr - h_prev).abs();
                if jump > max_jump {
                    max_jump = jump;
                }
            }
        }
        assert!(
            max_jump < 100.0,
            "Max row-to-row height jump {:.1}m exceeds 100m — cliff wall detected!",
            max_jump,
        );
    }

    // ── Helpers ────────────────────────────────────────────────

    fn generate_test_grid(lat_steps: usize, lon_steps: usize) -> Vec<glam::DVec3> {
        let mut points = Vec::with_capacity(lat_steps * lon_steps);
        for i in 0..lat_steps {
            let lat = -80.0 + (160.0 / lat_steps as f64) * i as f64;
            for j in 0..lon_steps {
                let lon = -179.0 + (358.0 / lon_steps as f64) * j as f64;
                points.push(glam::DVec3::new(lat, lon, 0.0));
            }
        }
        points
    }

    // ── Globe/centered mesh alignment regression tests ─────────

    #[test]
    fn test_build_globe_tile_mesh_count_matches_tiles() {
        let tiles = vec![
            RenderableTile { coord: TileCoord::new(2, 0, 0), texture_coord: TileCoord::new(2, 0, 0), uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 0 },
            RenderableTile { coord: TileCoord::new(2, 1, 1), texture_coord: TileCoord::new(2, 1, 1), uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 1 },
            RenderableTile { coord: TileCoord::new(2, 3, 2), texture_coord: TileCoord::new(2, 3, 2), uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 3 },
        ];

        let (_verts, _idxs, tile_idx_counts) = build_globe_tile_mesh(&tiles);
        assert_eq!(tile_idx_counts.len(), tiles.len());
        for (i, &count) in tile_idx_counts.iter().enumerate() {
            assert!(count > 0, "tile {} must produce non-zero index count, got 0", i);
        }
    }

    #[test]
    fn test_build_centered_tile_mesh_count_matches_tiles() {
        let center_lat = 37.5_f64.to_radians();
        let center_lon = 127.0_f64.to_radians();
        let tiles = vec![
            RenderableTile { coord: TileCoord::new(3, 7, 3), texture_coord: TileCoord::new(3, 7, 3), uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 7 },
            RenderableTile { coord: TileCoord::new(3, 7, 4), texture_coord: TileCoord::new(3, 7, 4), uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 7 },
        ];

        let (_verts, _idxs, tile_idx_counts) =
            build_centered_tile_mesh(&tiles, center_lat, center_lon);
        assert_eq!(tile_idx_counts.len(), tiles.len());
        for (i, &count) in tile_idx_counts.iter().enumerate() {
            assert!(count > 0, "tile {} must produce non-zero index count, got 0", i);
        }
    }

    #[test]
    fn test_globe_mesh_index_offsets_are_contiguous() {
        let tiles: Vec<RenderableTile> = (0..4)
            .map(|i| RenderableTile {
                coord: TileCoord::new(2, i, 0), texture_coord: TileCoord::new(2, i, 0),
                uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: i as i64,
            })
            .collect();

        let (_verts, all_idxs, tile_idx_counts) = build_globe_tile_mesh(&tiles);
        let total: u32 = tile_idx_counts.iter().sum();
        assert_eq!(total, all_idxs.len() as u32);
    }

    #[test]
    fn test_centered_mesh_index_offsets_are_contiguous() {
        let center_lat = 0.0_f64.to_radians();
        let center_lon = 0.0_f64.to_radians();
        let tiles: Vec<RenderableTile> = (0..4)
            .map(|i| RenderableTile {
                coord: TileCoord::new(2, i, 1), texture_coord: TileCoord::new(2, i, 1),
                uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: i as i64,
            })
            .collect();

        let (_verts, all_idxs, tile_idx_counts) =
            build_centered_tile_mesh(&tiles, center_lat, center_lon);
        let total: u32 = tile_idx_counts.iter().sum();
        assert_eq!(total, all_idxs.len() as u32);
    }

    // ── Angular filter tests ──────

    #[test]
    fn test_centered_mesh_center_tile_always_has_indices() {
        let center_lat = 37.5_f64.to_radians();
        let center_lon = 127.0_f64.to_radians();
        let tile = RenderableTile {
            coord: TileCoord::new(5, 27, 12), texture_coord: TileCoord::new(5, 27, 12),
            uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 27,
        };
        let (_verts, _idxs, counts) = build_centered_tile_mesh(&[tile], center_lat, center_lon);
        assert_eq!(counts.len(), 1);
        assert!(counts[0] > 0);
    }

    #[test]
    fn test_centered_mesh_no_degenerate_triangles_near_center() {
        let center_lat = 0.0_f64;
        let center_lon = 0.0_f64;
        let tile = RenderableTile {
            coord: TileCoord::new(2, 2, 2), texture_coord: TileCoord::new(2, 2, 2),
            uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 2,
        };
        let (verts, idxs, _counts) = build_centered_tile_mesh(&[tile], center_lat, center_lon);
        for tri in idxs.chunks(3) {
            if tri.len() < 3 { continue; }
            let p0 = verts[tri[0] as usize].position;
            let p1 = verts[tri[1] as usize].position;
            let p2 = verts[tri[2] as usize].position;
            let cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
            assert!(cross.abs() > 1e-12, "degenerate triangle found: area ≈ {:.2e}", cross.abs());
        }
    }

    #[test]
    fn test_centered_mesh_far_tile_produces_indices() {
        let center_lat = 37.5_f64.to_radians();
        let center_lon = 127.0_f64.to_radians();

        let near = RenderableTile {
            coord: TileCoord::new(2, 3, 1), texture_coord: TileCoord::new(2, 3, 1),
            uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 3,
        };
        let far = RenderableTile {
            coord: TileCoord::new(2, 1, 1), texture_coord: TileCoord::new(2, 1, 1),
            uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 1,
        };

        let (_v1, _i1, counts_near) = build_centered_tile_mesh(&[near], center_lat, center_lon);
        let (_v2, _i2, counts_far) = build_centered_tile_mesh(&[far], center_lat, center_lon);

        assert!(counts_near[0] > 0, "near tile must produce indices");
        assert!(counts_far[0] > 0, "far tile must produce indices (shader handles clipping)");
    }

    // ── End-to-end tests ──────

    fn renderable(vt: &VisibleTile) -> RenderableTile {
        RenderableTile {
            coord: vt.coord, texture_coord: vt.coord,
            uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: vt.display_x,
        }
    }

    #[test]
    fn test_e2e_zoom5_all_visible_tiles_have_mesh() {
        let mut viewport = Viewport::new(1920, 1080);
        viewport.center = GeoCoord::new(37.5, 127.0);
        viewport.zoom = 5.0;

        let visible = viewport.visible_tiles();
        let center_lat_rad = viewport.center.lat.to_radians();
        let center_lon_rad = viewport.center.lon.to_radians();

        let tiles: Vec<RenderableTile> = visible.iter().map(|vt| renderable(vt))
            .filter(|rt| tile_passes_angular_filter(rt, center_lat_rad, center_lon_rad, viewport.zoom))
            .collect();

        assert_eq!(tiles.len(), visible.len(), "at zoom 5 all visible tiles must pass angular filter");

        let (_verts, _idxs, counts) = build_centered_tile_mesh(&tiles, center_lat_rad, center_lon_rad);
        for (i, &count) in counts.iter().enumerate() {
            assert!(count > 0, "tile {} ({:?}) at zoom 5 must produce mesh indices, got 0", i, tiles[i].coord);
        }
    }

    #[test]
    fn test_e2e_zoom3_almost_all_visible_tiles_pass_filter() {
        let mut viewport = Viewport::new(800, 600);
        viewport.center = GeoCoord::new(37.5, 127.0);
        viewport.zoom = 3.0;

        let visible = viewport.visible_tiles();
        let center_lat_rad = viewport.center.lat.to_radians();
        let center_lon_rad = viewport.center.lon.to_radians();

        let filtered_count = visible.iter().map(|vt| renderable(vt))
            .filter(|rt| tile_passes_angular_filter(rt, center_lat_rad, center_lon_rad, viewport.zoom))
            .count();

        let pass_ratio = filtered_count as f64 / visible.len() as f64;
        assert!(
            pass_ratio >= 0.9,
            "at zoom 3 at least 90% of visible tiles must pass, got {}/{} ({:.0}%)",
            filtered_count, visible.len(), pass_ratio * 100.0,
        );
    }

    #[test]
    fn test_e2e_mobile_zoom10_all_visible_tiles_have_mesh() {
        let mut viewport = Viewport::new(375, 812);
        viewport.center = GeoCoord::new(37.5, 127.0);
        viewport.zoom = 10.0;

        let visible = viewport.visible_tiles();
        let center_lat_rad = viewport.center.lat.to_radians();
        let center_lon_rad = viewport.center.lon.to_radians();

        let tiles: Vec<RenderableTile> = visible.iter().map(|vt| renderable(vt))
            .filter(|rt| tile_passes_angular_filter(rt, center_lat_rad, center_lon_rad, viewport.zoom))
            .collect();

        assert_eq!(tiles.len(), visible.len());

        let (_verts, _idxs, counts) = build_centered_tile_mesh(&tiles, center_lat_rad, center_lon_rad);
        for (i, &count) in counts.iter().enumerate() {
            assert!(count > 0, "mobile tile {} ({:?}) at zoom 10 must produce mesh indices, got 0", i, tiles[i].coord);
        }
    }

    #[test]
    fn test_e2e_zoom8_various_centers_all_tiles_have_mesh() {
        let centers: [(f64, f64); 4] = [
            (0.0, 0.0), (37.5, 127.0), (-33.9, 18.4), (60.0, -120.0),
        ];

        for (lat, lon) in centers {
            let mut viewport = Viewport::new(800, 600);
            viewport.center = GeoCoord::new(lat, lon);
            viewport.zoom = 8.0;

            let visible = viewport.visible_tiles();
            let center_lat_rad = lat.to_radians();
            let center_lon_rad = lon.to_radians();

            let tiles: Vec<RenderableTile> = visible.iter().map(|vt| renderable(vt))
                .filter(|rt| tile_passes_angular_filter(rt, center_lat_rad, center_lon_rad, viewport.zoom))
                .collect();

            assert_eq!(tiles.len(), visible.len(), "zoom 8 center ({},{}) all tiles must pass filter", lat, lon);

            let (_verts, _idxs, counts) = build_centered_tile_mesh(&tiles, center_lat_rad, center_lon_rad);
            for (i, &count) in counts.iter().enumerate() {
                assert!(count > 0, "zoom 8 center ({},{}) tile {} must produce mesh", lat, lon, i);
            }
        }
    }

    #[test]
    fn test_angular_threshold_floor() {
        assert!(centered_angular_threshold_deg(0.0) >= 85.0);
        assert!(centered_angular_threshold_deg(10.0) >= 85.0);
        assert!(centered_angular_threshold_deg(0.0) < 90.0);
    }

    // ── Globe tile mesh tests ─────────────────────────────

    #[test]
    fn test_build_globe_tile_mesh_produces_geometry() {
        let tiles = vec![RenderableTile {
            coord: TileCoord::new(2, 1, 1), texture_coord: TileCoord::new(2, 1, 1),
            uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 1,
        }];
        let (verts, idxs, counts) = build_globe_tile_mesh(&tiles);
        assert!(!verts.is_empty());
        assert!(!idxs.is_empty());
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[0] as usize, idxs.len());
    }

    #[test]
    fn test_build_globe_tile_mesh_vertices_reasonable_size() {
        let tiles = vec![RenderableTile {
            coord: TileCoord::new(4, 8, 5), texture_coord: TileCoord::new(4, 8, 5),
            uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 8,
        }];
        let (verts, _, _) = build_globe_tile_mesh(&tiles);
        for v in &verts {
            let pos_len = (v.position[0].powi(2) + v.position[1].powi(2) + v.position[2].powi(2)).sqrt();
            assert!(pos_len < 0.5, "RTE vertex at zoom 4 should be small, got length {}", pos_len);
        }
    }

    #[test]
    fn test_build_polar_caps_has_geometry() {
        let (verts, idxs) = build_polar_caps();
        assert!(!verts.is_empty());
        assert!(!idxs.is_empty());
        assert!(idxs.len() > 10);
    }

    #[test]
    fn test_tile_uniforms_for_globe_valid_mvp() {
        let rt = RenderableTile {
            coord: TileCoord::new(2, 1, 1), texture_coord: TileCoord::new(2, 1, 1),
            uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 1,
        };
        let vp = glam::DMat4::IDENTITY;
        let uniforms = tile_uniforms_for_globe(&rt, 1.0, &vp);
        for v in &uniforms.mvp {
            assert!(!v.is_nan());
        }
        assert_eq!(uniforms.meta[1], 1.0);
        assert_eq!(uniforms.meta[0], 2.0);
    }

    // ── Centered tile mesh tests ──────────────────────────

    #[test]
    fn test_build_centered_tile_mesh_produces_geometry() {
        let center_lat = 37.5_f64.to_radians();
        let center_lon = 127.0_f64.to_radians();
        let tiles = vec![RenderableTile {
            coord: TileCoord::new(3, 4, 3), texture_coord: TileCoord::new(3, 4, 3),
            uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 4,
        }];
        let (verts, idxs, counts) = build_centered_tile_mesh(&tiles, center_lat, center_lon);
        assert!(!verts.is_empty());
        assert!(!idxs.is_empty());
        assert_eq!(counts.len(), 1);
    }

    #[test]
    fn test_tile_uniforms_for_centered_valid() {
        let rt = RenderableTile {
            coord: TileCoord::new(3, 4, 3), texture_coord: TileCoord::new(3, 4, 3),
            uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: 4,
        };
        let center_lat = 37.5_f64.to_radians();
        let center_lon = 127.0_f64.to_radians();
        let vp = glam::DMat4::IDENTITY;
        let uniforms = tile_uniforms_for_centered(&rt, 0.8, &vp, center_lat, center_lon);
        for v in &uniforms.mvp {
            assert!(!v.is_nan());
        }
        assert_eq!(uniforms.meta[1], 0.8);
    }

    #[test]
    fn test_tile_passes_angular_filter_near_center() {
        let center_lat = 37.5_f64.to_radians();
        let center_lon = 127.0_f64.to_radians();
        let center_tile = TileCoord::from_geo(&x_planets_math::GeoCoord::new(37.5, 127.0), 5);
        let rt = RenderableTile {
            coord: center_tile, texture_coord: center_tile,
            uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: center_tile.x as i64,
        };
        assert!(tile_passes_angular_filter(&rt, center_lat, center_lon, 5.0));
    }

    #[test]
    fn test_tile_passes_angular_filter_far_away() {
        let center_lat = 37.5_f64.to_radians();
        let center_lon = 127.0_f64.to_radians();
        let far_tile = TileCoord::from_geo(&x_planets_math::GeoCoord::new(-37.5, -53.0), 5);
        let rt = RenderableTile {
            coord: far_tile, texture_coord: far_tile,
            uv_rect: [0.0, 0.0, 1.0, 1.0], display_x: far_tile.x as i64,
        };
        assert!(!tile_passes_angular_filter(&rt, center_lat, center_lon, 5.0));
    }
}
