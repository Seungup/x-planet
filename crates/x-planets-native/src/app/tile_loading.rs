//! Per-frame tile loading pipeline: abort stale requests, parent-first
//! loading, enqueue visible tiles, and spawn async fetch tasks.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use glam::DVec2;
use x_planets_core::engine::LayerKind;
use x_planets_math::{TileCoord, VisibleTile};
use x_planets_tiles::{
    RasterTileDecoder, TerrainEncoding, TerrainRgbDecoder, TerrariumDecoder, TileDecoder,
    TileRequest, TileSource,
};

use crate::tile_source::{LayerTileResult, TileResult};

use super::NativeApp;

impl NativeApp {
    pub(super) fn run_tile_loading(
        &mut self,
        visible: &[VisibleTile],
        visible_set: &HashSet<TileCoord>,
        camera_center: DVec2,
        now: Instant,
    ) {
        for ls in &mut self.layer_states {
            // ── Abort stale requests ──
            // Clear the priority queue every frame.  Tiles that were queued
            // but never dequeued (max_concurrent reached) are NOT in
            // pending_coords, so they'll be naturally re-enqueued below
            // with fresh priorities.
            ls.tile_loader.clear();

            // Build the "needed" set: visible tiles + uncached ancestors
            // that serve as fallback coverage.  Only abort in-flight
            // tiles NOT in this set.  This prevents the old zoom_diff
            // heuristic from killing ancestor tiles spawned by
            // parent-first loading (which caused infinite re-spawn loops).
            //
            // For over-zoomed tiles (z > max_zoom), the max_zoom ancestor
            // is the deepest tile we can fetch, so include it in the needed set.
            let mut needed_coords: HashSet<TileCoord> = visible_set.clone();
            // Always include base tiles (z=0, z=1) — they must never be
            // aborted because they provide global fallback coverage.
            {
                let base_max = 1u8.min(ls.max_zoom);
                for z in ls.min_zoom..=base_max {
                    let n = 1u32 << z;
                    for y in 0..n {
                        for x in 0..n {
                            needed_coords.insert(TileCoord::new(z, x, y));
                        }
                    }
                }
            }
            for vt in visible {
                let coord = vt.coord;
                // If tile exceeds max_zoom, start the ancestor chain
                // from the corresponding tile AT max_zoom.
                let start = if coord.z > ls.max_zoom {
                    let clamped = coord.clamp_to_zoom(ls.max_zoom);
                    needed_coords.insert(clamped);
                    clamped.parent()
                } else {
                    coord.parent()
                };
                let mut cur = start;
                while let Some(p) = cur {
                    if ls.tile_textures.contains(&p) {
                        // Cached ancestor found — it and everything
                        // above it are already available.
                        needed_coords.insert(p);
                        break;
                    }
                    needed_coords.insert(p);
                    cur = p.parent();
                }
            }

            let stale_coords: Vec<TileCoord> = ls.pending_coords
                .iter()
                .filter(|c| !needed_coords.contains(c))
                .copied()
                .collect();
            for coord in stale_coords {
                ls.pending_coords.remove(&coord);
                ls.tile_loader.complete(); // free concurrency slot
            }

            // GC expired cooldowns (once per frame is cheap).
            ls.failed_cooldowns.retain(|_, expire| now < *expire);

            // ── Parent-first loading ──
            // For each visible tile missing a cached ancestor, enqueue
            // the NEAREST uncached ancestor (one level at a time).
            // Once that ancestor loads, next frame discovers the next
            // one.  This avoids flooding the queue with deep ancestor
            // chains (z=0..z=14) that block visible tile loading.
            //
            // Budget: limit ancestor enqueues to at most
            // `max_concurrent_loads` per frame to leave headroom for
            // visible-tile loading.
            {
                let mut budget = ls.tile_loader.max_concurrent();
                let mut ancestor_enqueued: HashSet<TileCoord> = HashSet::new();
                for vt in visible {
                    let coord = vt.coord;
                    if budget == 0 { break; }
                    // Already have a cached texture? No ancestor needed.
                    if ls.tile_textures.contains(&coord) { continue; }

                    // Start from the closest fetchable ancestor
                    // (skip children beyond max_zoom).
                    let start = if coord.z > ls.max_zoom {
                        Some(coord.clamp_to_zoom(ls.max_zoom))
                    } else {
                        coord.parent()
                    };
                    let mut cur = start;
                    while let Some(p) = cur {
                        if p.z < ls.min_zoom { break; }
                        if ls.tile_textures.contains(&p) {
                            break; // ancestor cached, chain OK
                        }
                        if !ls.pending_coords.contains(&p)
                            && !ls.failed_cooldowns.contains_key(&p)
                            && ancestor_enqueued.insert(p)
                            && !visible_set.contains(&p)
                        {
                            // Enqueue the nearest uncached ancestor.
                            // Priority: slightly better than the
                            // worst visible tile so it loads soon
                            // but doesn't starve visible tiles.
                            let p_center = p.mercator_center();
                            let p_dist = (p_center - camera_center)
                                .length() as f32;
                            ls.tile_loader.enqueue(TileRequest {
                                coord: p,
                                priority: p_dist * 0.8,
                            });
                            budget = budget.saturating_sub(1);
                            break; // only nearest ancestor per visible tile
                        }
                        cur = p.parent();
                    }
                }
            }

            // ── Base tile loading ──
            // Always eagerly load z=0 and z=1 tiles (5 total) so that
            // resolve_fallbacks() always finds a cached ancestor.
            // Without this, panning to a new area shows black gaps because
            // no ancestor texture is available for newly visible tiles.
            {
                let base_max = 1u8.min(ls.max_zoom);
                for z in ls.min_zoom..=base_max {
                    let n = 1u32 << z;
                    for y in 0..n {
                        for x in 0..n {
                            let coord = TileCoord::new(z, x, y);
                            if ls.tile_textures.contains(&coord)
                                || ls.pending_coords.contains(&coord)
                                || ls.failed_cooldowns.contains_key(&coord)
                            {
                                continue;
                            }
                            // Highest priority (0.0) — these are tiny and
                            // critical for fallback coverage.
                            ls.tile_loader.enqueue(TileRequest {
                                coord,
                                priority: 0.0,
                            });
                        }
                    }
                }
            }

            // Enqueue visible tiles that are not yet loaded or in-flight.
            // Priority: distance from camera × fallback penalty.
            // Tiles with no/distant fallback texture are prioritized
            // (lower value = higher priority in the min-heap).
            //
            // Zoom clamping: tiles beyond `max_zoom` are never requested.
            // The fallback system renders them with parent tiles at `max_zoom`.
            // Tiles below `min_zoom` are also skipped (rare edge case).
            //
            // Over-zoom: for visible tiles at z > max_zoom, we enqueue the
            // corresponding tile at max_zoom so the fallback system can use
            // it.  Multiple over-zoomed children may map to the SAME max_zoom
            // tile, so we deduplicate.
            let mut overzoom_enqueued: HashSet<TileCoord> = HashSet::new();
            for vt in visible {
                let coord = vt.coord;
                if coord.z < ls.min_zoom {
                    continue;
                }
                // Clamp over-zoomed tiles: enqueue the deepest fetchable tile.
                let fetch_coord = if coord.z > ls.max_zoom {
                    let clamped = coord.clamp_to_zoom(ls.max_zoom);
                    if !overzoom_enqueued.insert(clamped) {
                        continue; // already enqueued this max_zoom tile
                    }
                    clamped
                } else {
                    coord
                };

                if ls.tile_textures.contains(&fetch_coord)
                    || ls.pending_coords.contains(&fetch_coord)
                    || ls.failed_cooldowns.contains_key(&fetch_coord)
                {
                    continue;
                }
                let tile_center = fetch_coord.mercator_center();
                let dist = (tile_center - camera_center).length() as f32;

                // Fallback depth: how many zoom levels up to the nearest
                // cached ancestor?  0 = no ancestor at all (blank tile!).
                let fallback_depth = {
                    let mut depth = 0u32;
                    let mut cur = fetch_coord.parent();
                    loop {
                        match cur {
                            Some(c) if ls.tile_textures.contains(&c) => {
                                depth += 1;
                                break;
                            }
                            Some(c) => {
                                depth += 1;
                                cur = c.parent();
                            }
                            None => {
                                depth = 0; // no ancestor found
                                break;
                            }
                        }
                    }
                    depth
                };
                // No fallback (depth=0) → factor=0.5 (boost priority)
                // Close fallback (depth=1) → factor=1.5 (deprioritize)
                // Distant fallback (depth≥3) → factor=1.0 (normal)
                let fallback_factor = match fallback_depth {
                    0 => 0.5,
                    1 => 1.5,
                    2 => 1.2,
                    _ => 1.0,
                };
                ls.tile_loader.enqueue(TileRequest {
                    coord: fetch_coord,
                    priority: dist * fallback_factor,
                });
                // NOTE: Do NOT insert into pending_coords here!
                // pending_coords tracks only truly in-flight tasks (spawned).
                // Tiles that stay in the queue are dropped by clear() next
                // frame and re-enqueued with fresh priorities.
            }

            // Dequeue & spawn (route decoder by layer kind).
            // Insert into pending_coords ONLY when a task is actually spawned.
            let terrain_encoding = match &ls.kind {
                LayerKind::Terrain { encoding, .. } => Some(*encoding),
                _ => None,
            };
            while let Some(req) = ls.tile_loader.dequeue() {
                ls.pending_coords.insert(req.coord);
                let source = Arc::clone(&ls.tile_source);
                let tx = self.tile_tx.clone();
                let layer_name = ls.name.clone();
                if let Some(enc) = terrain_encoding {
                    self.rt.spawn(async move {
                        let result = match source.fetch(req.coord).await {
                            Ok(bytes) => match enc {
                                TerrainEncoding::MapboxRgb => {
                                    match TerrainRgbDecoder.decode(req.coord, &bytes).await {
                                        Ok(d) => Ok(TileResult::Terrain(d)),
                                        Err(e) => Err((req.coord, e.to_string())),
                                    }
                                }
                                TerrainEncoding::Terrarium => {
                                    match TerrariumDecoder.decode(req.coord, &bytes).await {
                                        Ok(d) => Ok(TileResult::Terrain(d)),
                                        Err(e) => Err((req.coord, e.to_string())),
                                    }
                                }
                                TerrainEncoding::QuantizedMesh => {
                                    match x_planets_tiles::parse_quantized_mesh(req.coord, &bytes) {
                                        Ok(qm) => Ok(TileResult::QuantizedMesh(Box::new(qm))),
                                        Err(e) => Err((req.coord, e.to_string())),
                                    }
                                }
                            },
                            Err(e) => Err((req.coord, e.to_string())),
                        };
                        let _ = tx.send(LayerTileResult { layer_name, result });
                    });
                } else {
                    self.rt.spawn(async move {
                        let result = match source.fetch(req.coord).await {
                            Ok(bytes) => {
                                let decoder = RasterTileDecoder::default();
                                match decoder.decode(req.coord, &bytes).await {
                                    Ok(decoded) => Ok(TileResult::Raster(decoded)),
                                    Err(e) => Err((req.coord, e.to_string())),
                                }
                            }
                            Err(e) => Err((req.coord, e.to_string())),
                        };
                        let _ = tx.send(LayerTileResult { layer_name, result });
                    });
                }
            }
        }
    }
}
