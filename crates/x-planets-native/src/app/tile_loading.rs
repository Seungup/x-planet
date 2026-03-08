//! Per-frame tile loading pipeline using the shared `plan_tile_loads()` planner.
//!
//! The pure-function planner (in `x-planets-core`) decides **which** tiles to load
//! and in what priority.  This module handles the platform-specific parts:
//! aborting stale requests, GC'ing cooldowns, and spawning tokio tasks.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use glam::DVec2;
use x_planets_core::engine::LayerKind;
use x_planets_core::tile_load_planner::{plan_tile_loads, PlannedRequestKind};
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
            // Clear the priority queue every frame.
            ls.tile_loader.clear();

            // ── Core planner: what to load and what's still needed ──
            let (needed, requests) = plan_tile_loads(ls, visible, visible_set, camera_center);

            // ── Abort stale raster/terrain requests ──
            let stale_coords: Vec<TileCoord> = ls
                .pending_coords
                .iter()
                .filter(|c| !needed.raster.contains(c))
                .copied()
                .collect();
            for coord in stale_coords {
                ls.pending_coords.remove(&coord);
                ls.tile_loader.complete();
            }

            // ── Abort stale elevation requests ──
            ls.pending_elevation_coords
                .retain(|c| needed.elevation.contains(c));

            // ── GC expired cooldowns ──
            ls.failed_cooldowns.retain(|_, expire| now < *expire);

            // ── Enqueue planned requests into the TileLoader ──
            // Only enqueue raster/terrain requests (not elevation — those bypass TileLoader).
            let terrain_encoding = match &ls.kind {
                LayerKind::Terrain { encoding, .. } => Some(*encoding),
                _ => None,
            };

            for req in &requests {
                match req.kind {
                    PlannedRequestKind::Raster | PlannedRequestKind::Terrain => {
                        ls.tile_loader.enqueue(TileRequest {
                            coord: req.coord,
                            priority: req.priority,
                        });
                    }
                    PlannedRequestKind::Elevation => {
                        // Elevation tiles are spawned directly below.
                    }
                }
            }

            // ── Dequeue & spawn raster/terrain tasks ──
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

            // ── Spawn elevation tasks (bypass TileLoader, use direct slots) ──
            if let Some(elev_source) = &ls.elevation_source {
                let elev_slots = ls
                    .max_elevation_concurrent
                    .saturating_sub(ls.pending_elevation_coords.len());
                let mut elev_count = 0usize;
                for req in &requests {
                    if req.kind != PlannedRequestKind::Elevation {
                        continue;
                    }
                    if elev_count >= elev_slots {
                        break;
                    }
                    // Double-check not already pending (planner doesn't insert).
                    if ls.pending_elevation_coords.contains(&req.coord) {
                        continue;
                    }
                    ls.pending_elevation_coords.insert(req.coord);
                    let source = Arc::clone(elev_source);
                    let tx = self.tile_tx.clone();
                    let layer_name = ls.name.clone();
                    let coord = req.coord;
                    self.rt.spawn(async move {
                        let result = match source.fetch(coord).await {
                            Ok(bytes) => {
                                match TerrariumDecoder.decode(coord, &bytes).await {
                                    Ok(d) => Ok(TileResult::Terrain(d)),
                                    Err(e) => Err((coord, e.to_string())),
                                }
                            }
                            Err(e) => Err((coord, e.to_string())),
                        };
                        let _ = tx.send(LayerTileResult { layer_name, result });
                    });
                    elev_count += 1;
                }
            }
        }
    }
}
