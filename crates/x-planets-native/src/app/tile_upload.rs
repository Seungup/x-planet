//! Poll completed tile fetch results and upload to GPU.

use std::time::Instant;

use x_planets_core::TerrainTileData;

use crate::tile_source::TileResult;

use super::NativeApp;

impl NativeApp {
    /// Poll the tile result channel and create GPU textures for completed tiles.
    ///
    /// Caps uploads per frame to avoid frame-time spikes when many tiles
    /// arrive at once (each raster tile ≈ 256 KB GPU upload, each terrain
    /// mesh = CPU build).  Remaining tiles stay in the channel for the
    /// next frame.
    pub(super) fn poll_tile_results(&mut self, now: Instant) {
        const MAX_TILES_PER_FRAME: usize = 4;
        let mut tiles_this_frame = 0;
        let now_secs = now.duration_since(self.start_time).as_secs_f64();

        // Collect raster tile coords to register fade-in after the loop
        // (avoids borrow conflicts between layer_states and controller).
        let mut raster_loaded_coords: Vec<x_planets_math::TileCoord> = Vec::new();

        let gpu = self.gpu.as_ref().unwrap();
        let tex_mgr = self.tex_manager.as_ref().unwrap();

        while tiles_this_frame < MAX_TILES_PER_FRAME {
            let msg = match self.tile_rx.try_recv() {
                Ok(m) => m,
                Err(_) => break,
            };
            tiles_this_frame += 1;
            if let Some(ls) = self.layer_states.iter_mut().find(|s| s.name == msg.layer_name) {
                match msg.result {
                    Ok(TileResult::Raster(decoded)) => {
                        if ls.pending_coords.remove(&decoded.coord) {
                            ls.tile_loader.complete();
                        }
                        log::debug!(
                            "[{}] Raster tile loaded: z={} x={} y={} ({}×{})",
                            ls.name,
                            decoded.coord.z, decoded.coord.x, decoded.coord.y,
                            decoded.width, decoded.height,
                        );
                        raster_loaded_coords.push(decoded.coord);
                        let tex = tex_mgr.create_rgba_texture(
                            &gpu.device, &gpu.queue,
                            &format!("{}-tile-{}-{}-{}", ls.name,
                                decoded.coord.z, decoded.coord.x, decoded.coord.y),
                            decoded.width, decoded.height, &decoded.pixels,
                        );
                        ls.tile_textures.insert(decoded.coord, tex);
                    }
                    Ok(TileResult::Terrain(decoded)) => {
                        // Could be from config-file terrain layer (pending_coords)
                        // or runtime elevation loading (pending_elevation_coords)
                        if ls.pending_coords.remove(&decoded.coord) {
                            ls.tile_loader.complete();
                        }
                        ls.pending_elevation_coords.remove(&decoded.coord);
                        log::debug!(
                            "[{}] Terrain tile loaded: z={} x={} y={} elev=[{:.0}..{:.0}]m",
                            ls.name,
                            decoded.coord.z, decoded.coord.x, decoded.coord.y,
                            decoded.min_elevation, decoded.max_elevation,
                        );
                        // Note: we do NOT insert terrain tile_fade_start here.
                        // Terrain imagery comes from the companion raster layer,
                        // whose fade_start is recorded when the raster tile loads.
                        // Inserting here would overwrite the imagery fade_start,
                        // causing incorrect cross-fade timing.

                        // Store elevation data on CPU for mesh generation
                        ls.terrain_data.insert(
                            decoded.coord,
                            TerrainTileData::Heightmap {
                                elevation: decoded.elevation,
                                width: decoded.width,
                                height: decoded.height,
                            },
                        );
                        // Also insert a placeholder texture so the tile is considered "loaded"
                        // (the actual imagery texture comes from the companion raster layer)
                        let tex = tex_mgr.create_rgba_texture(
                            &gpu.device, &gpu.queue,
                            &format!("{}-terrain-{}-{}-{}", ls.name,
                                decoded.coord.z, decoded.coord.x, decoded.coord.y),
                            1, 1, &[128, 128, 128, 255], // 1×1 gray placeholder
                        );
                        ls.tile_textures.insert(decoded.coord, tex);
                    }
                    Ok(TileResult::QuantizedMesh(qm)) => {
                        if ls.pending_coords.remove(&qm.coord) {
                            ls.tile_loader.complete();
                        }
                        log::info!(
                            "[{}] QM tile loaded: z={} x={} y={} ({} verts, {} tris, h=[{:.0}..{:.0}]m, geo={})",
                            ls.name,
                            qm.coord.z, qm.coord.x, qm.coord.y,
                            qm.u.len(),
                            qm.indices.len() / 3,
                            qm.header.min_height, qm.header.max_height,
                            ls.geographic,
                        );

                        if ls.geographic {
                            // ── Geographic (EPSG:4326) QM → rasterize → multi-source resample → 3857 Heightmap ──
                            //
                            // Pipeline:
                            // 1. Build QM mesh from raw data (4326 tile-local UV space)
                            // 2. Rasterize TIN mesh → regular grid heightmap (4326 UV)
                            // 3. Cache the 4326 heightmap for neighbor resampling
                            // 4. Resample from ALL available 4326 sources → 3857 heightmap
                            // 5. Re-resample any previously loaded adjacent tiles that
                            //    now have additional 4326 data available

                            let surface_idx_count = qm.indices.len();
                            let (vertices, indices) =
                                x_planets_core::pipeline::build_terrain_mesh_from_qm(
                                    &qm.coord, &qm,
                                );

                            // Step 1-2: Rasterize QM TIN → 4326 heightmap.
                            let raster_grid_size = 65u32;
                            let heightmap_4326 =
                                x_planets_core::pipeline::rasterize_qm_to_heightmap(
                                    &vertices, &indices[..surface_idx_count], raster_grid_size,
                                );

                            // Step 3: Cache the 4326 heightmap.
                            // Evict old entries when cache exceeds budget
                            // (each 65×65 f32 heightmap ≈ 17 KB, 256 entries ≈ 4 MB).
                            const GEO_CACHE_MAX: usize = 256;
                            if ls.geo_heightmap_cache.len() >= GEO_CACHE_MAX {
                                // Simple eviction: remove entries for zoom levels
                                // far from the current tile's zoom.
                                let current_gz = if qm.coord.z > 0 { qm.coord.z - 1 } else { 0 };
                                ls.geo_heightmap_cache.retain(|&(_, _, gz), _| {
                                    (gz as i16 - current_gz as i16).unsigned_abs() <= 2
                                });
                            }

                            let (gx, gy, gz) =
                                crate::tile_source::mercator_to_geographic_tile(&qm.coord);
                            let (geo_west, geo_east, geo_north, geo_south) =
                                crate::tile_source::geographic_tile_bounds(gx, gy, gz);

                            ls.geo_heightmap_cache.insert(
                                (gx, gy, gz),
                                crate::tile_source::GeoHeightmapEntry {
                                    heightmap: heightmap_4326,
                                    grid_size: raster_grid_size,
                                    west: geo_west,
                                    east: geo_east,
                                    north: geo_north,
                                    south: geo_south,
                                },
                            );

                            // Step 4: Collect all available 4326 sources for this 3857 tile.
                            let needed =
                                crate::tile_source::overlapping_geographic_tiles(&qm.coord);
                            let sources: Vec<x_planets_core::pipeline::GeoHeightmapSource<'_>> =
                                needed.iter()
                                    .filter_map(|key| ls.geo_heightmap_cache.get(key))
                                    .map(|e| x_planets_core::pipeline::GeoHeightmapSource {
                                        heightmap: &e.heightmap,
                                        grid_size: e.grid_size,
                                        west: e.west,
                                        east: e.east,
                                        north: e.north,
                                        south: e.south,
                                    })
                                    .collect();
                            let out_grid_size = 33u32;
                            let heightmap_3857 =
                                x_planets_core::pipeline::resample_geographic_to_mercator_multi(
                                    &sources,
                                    &qm.coord,
                                    out_grid_size,
                                );

                            log::info!(
                                "[{}] QM→3857 resampled: z={} x={} y={} (4326 tile z={} x={} y={}, \
                                 bounds=[{:.2}°..{:.2}°, {:.2}°..{:.2}°], sources={}/{})",
                                ls.name,
                                qm.coord.z, qm.coord.x, qm.coord.y,
                                gz, gx, gy,
                                geo_west, geo_east, geo_south, geo_north,
                                sources.len(), needed.len(),
                            );

                            ls.terrain_data.insert(
                                qm.coord,
                                TerrainTileData::Heightmap {
                                    elevation: heightmap_3857,
                                    width: out_grid_size,
                                    height: out_grid_size,
                                },
                            );

                            // Step 5: Check adjacent 3857 tiles that might benefit
                            // from this new 4326 data.
                            // An adjacent tile needs re-resampling if:
                            //   (a) it already has terrain_data (was loaded before), AND
                            //   (b) it needs the 4326 tile we just cached as a secondary source
                            let geo_key = (gx, gy, gz);
                            let neighbors = [
                                if qm.coord.y > 0 {
                                    Some(x_planets_math::TileCoord::new(
                                        qm.coord.z, qm.coord.x, qm.coord.y - 1,
                                    ))
                                } else { None },
                                Some(x_planets_math::TileCoord::new(
                                    qm.coord.z, qm.coord.x, qm.coord.y + 1,
                                )),
                            ];
                            for neighbor_coord in neighbors.into_iter().flatten() {
                                // Does this neighbor need the 4326 tile we just loaded?
                                let neighbor_needed =
                                    crate::tile_source::overlapping_geographic_tiles(&neighbor_coord);
                                if !neighbor_needed.contains(&geo_key) {
                                    continue;
                                }
                                // Does the neighbor already have terrain data?
                                if ls.terrain_data.peek(&neighbor_coord).is_none() {
                                    continue;
                                }
                                // Re-resample the neighbor using ALL available 4326 sources.
                                let n_sources: Vec<x_planets_core::pipeline::GeoHeightmapSource<'_>> =
                                    neighbor_needed.iter()
                                        .filter_map(|key| ls.geo_heightmap_cache.get(key))
                                        .map(|e| x_planets_core::pipeline::GeoHeightmapSource {
                                            heightmap: &e.heightmap,
                                            grid_size: e.grid_size,
                                            west: e.west,
                                            east: e.east,
                                            north: e.north,
                                            south: e.south,
                                        })
                                        .collect();
                                // Only re-resample if we now have MORE sources than the
                                // neighbor originally had (otherwise it's already optimal).
                                if n_sources.len() <= 1 {
                                    continue;
                                }
                                let n_heightmap =
                                    x_planets_core::pipeline::resample_geographic_to_mercator_multi(
                                        &n_sources,
                                        &neighbor_coord,
                                        out_grid_size,
                                    );
                                log::info!(
                                    "[{}] Re-resampled neighbor z={} x={} y={} with {} sources (cliff wall fix)",
                                    ls.name,
                                    neighbor_coord.z, neighbor_coord.x, neighbor_coord.y,
                                    n_sources.len(),
                                );
                                ls.terrain_data.insert(
                                    neighbor_coord,
                                    TerrainTileData::Heightmap {
                                        elevation: n_heightmap,
                                        width: out_grid_size,
                                        height: out_grid_size,
                                    },
                                );
                                // Invalidate the neighbor's cached mesh so it's rebuilt
                                // with the updated heightmap.
                                if let Some(tr) = self.terrain_renderer.as_mut() {
                                    tr.invalidate_mesh(&neighbor_coord);
                                }
                            }
                        } else {
                            // ── Mercator QM → direct PrebuiltMesh ──
                            // Convert raw QM data to pre-built vertices/indices.
                            // Heights stay in metres; height_scale applied in renderer
                            // so exaggeration changes work without re-fetching.
                            let surface_idx_count = qm.indices.len();
                            let (vertices, indices) =
                                x_planets_core::pipeline::build_terrain_mesh_from_qm(
                                    &qm.coord, &qm,
                                );
                            // Rasterize QM mesh into a regular grid heightmap for
                            // over-zoom fallback: child tiles beyond max_zoom can
                            // sub-sample this heightmap instead of being flat.
                            // Only use surface triangles — exclude skirt geometry
                            // to prevent edge height corruption (see geographic path).
                            let fallback_grid_size = 33u32;
                            let fallback_heightmap =
                                x_planets_core::pipeline::rasterize_qm_to_heightmap(
                                    &vertices, &indices[..surface_idx_count], fallback_grid_size,
                                );
                            ls.terrain_data.insert(
                                qm.coord,
                                TerrainTileData::PrebuiltMesh {
                                    vertices,
                                    indices,
                                    fallback_heightmap,
                                    fallback_grid_size,
                                },
                            );
                        }

                        // Placeholder texture (imagery from companion raster layer)
                        let tex = tex_mgr.create_rgba_texture(
                            &gpu.device, &gpu.queue,
                            &format!("{}-qm-{}-{}-{}", ls.name,
                                qm.coord.z, qm.coord.x, qm.coord.y),
                            1, 1, &[128, 128, 128, 255],
                        );
                        ls.tile_textures.insert(qm.coord, tex);
                    }
                    Err((coord, err_msg)) => {
                        if ls.pending_coords.remove(&coord) {
                            ls.tile_loader.complete();
                        }
                        ls.pending_elevation_coords.remove(&coord);
                        // Backoff cooldown:
                        //   429 (rate limit) → 30s
                        //   400 (bad request / tile doesn't exist at this zoom) → 300s
                        //   other errors → 5s
                        let cooldown_secs = if err_msg.contains("429") {
                            30
                        } else if err_msg.contains("400") {
                            300
                        } else {
                            5
                        };
                        ls.failed_cooldowns.insert(
                            coord,
                            now + std::time::Duration::from_secs(cooldown_secs),
                        );
                        // Always print tile failures to stderr so the user can
                        // see errors even without RUST_LOG=debug.
                        eprintln!(
                            "[x-planets] TILE FAIL [{}] z={} x={} y={} (retry {}s): {}",
                            ls.name, coord.z, coord.x, coord.y, cooldown_secs, err_msg,
                        );
                        log::warn!(
                            "[{}] Tile load failed {} (retry in {}s): {}",
                            ls.name, coord, cooldown_secs, err_msg,
                        );
                    }
                }
            }
        }

        // Register fade-in for loaded raster tiles via MapController
        if !raster_loaded_coords.is_empty() {
            if let Some(ctrl) = &mut self.controller {
                for coord in raster_loaded_coords {
                    ctrl.register_tile_loaded(coord, now_secs);
                }
            }
        }
    }
}
