//! Layer and 3D Tiles state initialization (called from `resumed()`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use x_planets_core::engine::LayerKind;
use x_planets_core::MapEngine;
use x_planets_tiles::{TileCache, TileLoader};

use crate::tile_source::{NativeLayerState, NativeTileSource};
use crate::tilejson::{is_tilejson_url, resolve_tilejson, TileJsonMeta};
use crate::tiles3d_native::{Tiles3dAuthKind, Tiles3dLayerState};

/// Resolve TileJSON endpoints and create `NativeLayerState` for every
/// non-3D-Tiles layer.
pub(super) fn init_layer_states(
    rt: &tokio::runtime::Runtime,
    engine: &MapEngine,
) -> Vec<NativeLayerState> {
    let tilejson_client = crate::http_client();

    engine
        .layers
        .iter()
        .filter(|layer| !matches!(layer.config.kind, LayerKind::Tiles3d))
        .map(|layer| {
            let cfg = &layer.config;
            let meta = if is_tilejson_url(&cfg.tile_source_url) {
                log::info!(
                    "Layer '{}': resolving TileJSON → {}",
                    cfg.name, cfg.tile_source_url,
                );
                match rt.block_on(resolve_tilejson(&tilejson_client, &cfg.tile_source_url)) {
                    Ok(m) => {
                        log::info!(
                            "  → resolved to: {}",
                            if m.tile_url.len() > 80 { format!("{}…", &m.tile_url[..80]) } else { m.tile_url.clone() },
                        );
                        m
                    }
                    Err(e) => {
                        log::warn!("  → TileJSON resolution failed, using URL as-is: {}", e);
                        TileJsonMeta::passthrough(cfg.tile_source_url.clone())
                    }
                }
            } else {
                TileJsonMeta::passthrough(cfg.tile_source_url.clone())
            };

            // If TileJSON detected a terrain encoding (e.g. "quantized-mesh-1.0"),
            // override the encoding in the layer kind — but only when the user
            // did NOT explicitly set `terrain_encoding` in the config file.
            let kind = if let (LayerKind::Terrain { ref imagery_layer, .. }, Some(enc)) =
                (&cfg.kind, meta.detected_encoding)
            {
                if cfg.terrain_encoding_explicit {
                    log::info!(
                        "[x-planets] Layer '{}': TileJSON detected {:?} but config \
                         explicitly set encoding — keeping config value",
                        cfg.name, enc
                    );
                    cfg.kind.clone()
                } else {
                    eprintln!(
                        "[x-planets] Layer '{}': TileJSON auto-detected encoding → {:?}",
                        cfg.name, enc
                    );
                    log::info!(
                        "  → auto-detected terrain encoding: {:?}",
                        enc
                    );
                    LayerKind::Terrain {
                        imagery_layer: imagery_layer.clone(),
                        encoding: enc,
                    }
                }
            } else {
                cfg.kind.clone()
            };

            // For geographic (EPSG:4326) sources, the TileJSON maxzoom refers to
            // the geographic zoom level.  The engine uses EPSG:3857 zoom internally,
            // and `mercator_to_geographic_tile()` converts with gz = z_3857 − 1.
            // Therefore a geographic maxzoom of N can serve 3857 requests up to
            // z = N + 1.  Adjust the stored max_zoom accordingly.
            let max_zoom = if meta.geographic {
                meta.max_zoom.map(|z| z.saturating_add(1)).unwrap_or(22)
            } else {
                meta.max_zoom.unwrap_or(22)
            };

            log::info!(
                "Creating layer '{}' → {} (zoom {}-{}{}, scale={}x, geographic={}, max_concurrent={}, max_cached={})",
                cfg.name, meta.tile_url,
                meta.min_zoom.unwrap_or(0), max_zoom,
                if meta.geographic { format!(" (4326 maxzoom={})", meta.max_zoom.unwrap_or(22)) } else { String::new() },
                meta.scale, meta.geographic,
                cfg.max_concurrent_loads, cfg.max_cached_tiles,
            );
            let source = NativeTileSource::new(&meta.tile_url)
                .with_tms(meta.tms)
                .with_geographic(meta.geographic);
            NativeLayerState {
                name: cfg.name.clone(),
                kind,
                tile_source: Arc::new(source),
                tile_textures: TileCache::new(cfg.max_cached_tiles),
                tile_loader: TileLoader::new(cfg.max_concurrent_loads),
                pending_coords: HashSet::new(),
                terrain_data: TileCache::new(cfg.max_cached_tiles),
                failed_cooldowns: HashMap::new(),
                min_zoom: meta.min_zoom.unwrap_or(0),
                max_zoom,
                tile_scale: meta.scale,
                geographic: meta.geographic,
                geo_heightmap_cache: HashMap::new(),
            }
        })
        .collect()
}

/// Create `Tiles3dLayerState` for every 3D Tiles layer.
pub(super) fn init_tiles3d_states(engine: &MapEngine) -> Vec<Tiles3dLayerState> {
    engine
        .layers
        .iter()
        .filter(|layer| matches!(layer.config.kind, LayerKind::Tiles3d))
        .map(|layer| {
            let cfg = &layer.config;
            let auth = if let Some(token) = &cfg.cesium_ion_token {
                let asset_id = cfg.cesium_ion_asset_id.unwrap_or(1);
                log::info!(
                    "Creating 3D Tiles layer '{}' → Cesium Ion asset {}",
                    cfg.name, asset_id,
                );
                Tiles3dAuthKind::CesiumIon {
                    account_token: token.clone(),
                    asset_id,
                }
            } else if let Some(key) = &cfg.google_api_key {
                log::info!(
                    "Creating 3D Tiles layer '{}' → Google 3D Tiles",
                    cfg.name,
                );
                Tiles3dAuthKind::Google {
                    api_key: key.clone(),
                }
            } else {
                log::warn!(
                    "3D Tiles layer '{}' has no auth config, skipping",
                    cfg.name,
                );
                // Use a dummy — will fail on init
                Tiles3dAuthKind::CesiumIon {
                    account_token: String::new(),
                    asset_id: 0,
                }
            };
            Tiles3dLayerState::new(cfg.name.clone(), auth)
        })
        .collect()
}
