//! Build `RenderLayerData` and `TerrainLayerData` for each visible layer,
//! including cross-fade overlay logic.
//!
//! Uses shared crossfade functions from `x_planets_core::interaction` to avoid
//! duplicating fade logic between native and web.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use x_planets_core::engine::LayerKind;
use x_planets_core::interaction::{
    build_crossfade_overlay, compute_crossfade, compute_fade_overrides,
};
use x_planets_core::render::RenderLayerData;
use x_planets_core::{MapEngine, TerrainLayerData, TerrainTileData};
use x_planets_math::{TileCoord, VisibleTile};

use crate::animation::AnimationState;
use crate::tile_source::NativeLayerState;

/// Build render data for all visible layers.
///
/// Returns `(raster_layers, terrain_base_layers, terrain_overlay_layers)`.
pub(super) fn build_all_layers<'a>(
    engine: &'a MapEngine,
    layer_states: &'a [NativeLayerState],
    anim: &AnimationState,
    visible: &[VisibleTile],
    now: Instant,
) -> (
    Vec<RenderLayerData<'a>>,
    Vec<TerrainLayerData<'a>>,
    Vec<TerrainLayerData<'a>>,
) {
    let mut render_layers: Vec<RenderLayerData> = Vec::new();
    let mut terrain_layers: Vec<TerrainLayerData> = Vec::new();
    let mut terrain_overlay_layers: Vec<TerrainLayerData> = Vec::new();

    // Collect raster layer names that are used as imagery for terrain layers.
    let terrain_imagery_names: HashSet<&str> = engine
        .visible_layers()
        .filter_map(|l| match &l.config.kind {
            LayerKind::Terrain { imagery_layer, .. } => Some(imagery_layer.as_str()),
            _ => None,
        })
        .collect();

    for layer in engine.visible_layers() {
        if let Some(ls) = layer_states.iter().find(|s| s.name == layer.config.name) {
            match &layer.config.kind {
                LayerKind::Raster => {
                    if terrain_imagery_names.contains(layer.config.name.as_str()) {
                        continue;
                    }

                    let (base, overlay) =
                        build_raster_layer(layer, ls, anim, visible, now);
                    render_layers.push(base);
                    if let Some(ovl) = overlay {
                        render_layers.push(ovl);
                    }
                }
                LayerKind::Tiles3d => {
                    // 3D Tiles layers are handled separately.
                }
                LayerKind::Terrain { imagery_layer, .. } => {
                    let imagery_ls = layer_states.iter().find(|s| s.name == *imagery_layer);

                    if imagery_ls.is_none() {
                        eprintln!(
                            "[x-planets] TERRAIN WARN: layer '{}' references imagery_layer '{}' \
                             which was not found — terrain cannot render without it.",
                            layer.config.name, imagery_layer
                        );
                    }
                    if let Some(img_ls) = imagery_ls {
                        let (base, overlay) =
                            build_terrain_layer(layer, ls, img_ls, anim, visible, now);
                        terrain_layers.push(base);
                        if let Some(ovl) = overlay {
                            terrain_overlay_layers.push(ovl);
                        }
                    }
                }
            }
        }
    }

    (render_layers, terrain_layers, terrain_overlay_layers)
}

/// Build raster layer render data with cross-fade overlay.
fn build_raster_layer<'a>(
    layer: &'a x_planets_core::engine::TileLayer,
    ls: &'a NativeLayerState,
    anim: &AnimationState,
    visible: &[VisibleTile],
    now: Instant,
) -> (RenderLayerData<'a>, Option<RenderLayerData<'a>>) {
    let available: HashSet<TileCoord> = ls.tile_textures.keys().copied().collect();

    let texture_views: HashMap<TileCoord, &wgpu::TextureView> = ls
        .tile_textures
        .iter()
        .map(|(k, v)| (*k, &v.view))
        .collect();

    // Use shared crossfade computation (convert Instant → elapsed f64)
    let (available_for_base, crossfade_tiles) =
        compute_crossfade(visible, &available, |coord| {
            anim.tile_fade_elapsed(coord, now)
        });

    let renderable =
        x_planets_core::pipeline::resolve_fallbacks(visible, &available_for_base);

    // Use shared fade override computation
    let tile_opacity_overrides = compute_fade_overrides(
        &renderable,
        layer.config.opacity,
        |coord| anim.tile_fade_elapsed(coord, now),
    );

    let base = RenderLayerData {
        name: &layer.config.name,
        opacity: layer.config.opacity,
        tiles: renderable,
        texture_views: texture_views.clone(),
        tile_opacity_overrides,
    };

    // Use shared overlay builder
    let overlay = if !crossfade_tiles.is_empty() {
        let (overlay_tiles, overlay_opacity) =
            build_crossfade_overlay(&crossfade_tiles, layer.config.opacity);
        Some(RenderLayerData {
            name: "crossfade-overlay",
            opacity: layer.config.opacity,
            tiles: overlay_tiles,
            texture_views,
            tile_opacity_overrides: overlay_opacity,
        })
    } else {
        None
    };

    (base, overlay)
}

/// Build terrain layer render data with cross-fade overlay for imagery.
fn build_terrain_layer<'a>(
    layer: &'a x_planets_core::engine::TileLayer,
    terrain_ls: &'a NativeLayerState,
    imagery_ls: &'a NativeLayerState,
    anim: &AnimationState,
    visible: &[VisibleTile],
    now: Instant,
) -> (TerrainLayerData<'a>, Option<TerrainLayerData<'a>>) {
    let available: HashSet<TileCoord> = imagery_ls.tile_textures.keys().copied().collect();

    let imagery_views: HashMap<TileCoord, &wgpu::TextureView> = imagery_ls
        .tile_textures
        .iter()
        .map(|(k, v)| (*k, &v.view))
        .collect();

    // Use shared crossfade computation
    let (available_for_base, crossfade_tiles) =
        compute_crossfade(visible, &available, |coord| {
            anim.tile_fade_elapsed(coord, now)
        });

    let renderable = x_planets_core::pipeline::resolve_fallbacks(
        visible, &available_for_base,
    );

    // Elevation data with parent fallback
    let mut elevation_data: HashMap<TileCoord, (&TerrainTileData, TileCoord)> = HashMap::new();
    let all_needed: HashSet<TileCoord> = renderable
        .iter()
        .map(|rt| rt.coord)
        .chain(crossfade_tiles.iter().map(|&(c, _, _)| c))
        .collect();
    for &coord in &all_needed {
        let mut c = Some(coord);
        while let Some(candidate) = c {
            if let Some(data) = terrain_ls.terrain_data.peek(&candidate) {
                elevation_data.insert(coord, (data, candidate));
                break;
            }
            c = candidate.parent();
        }
    }

    let base = TerrainLayerData {
        name: &layer.config.name,
        opacity: layer.config.opacity,
        tiles: renderable,
        imagery_views: imagery_views.clone(),
        elevation_data: elevation_data.clone(),
        tile_opacity_overrides: HashMap::new(),
    };

    let overlay = if !crossfade_tiles.is_empty() {
        let (overlay_tiles, overlay_opacity) =
            build_crossfade_overlay(&crossfade_tiles, layer.config.opacity);
        let overlay_elev: HashMap<TileCoord, (&TerrainTileData, TileCoord)> =
            crossfade_tiles
                .iter()
                .filter_map(|&(coord, _, _)| {
                    elevation_data.get(&coord).map(|&v| (coord, v))
                })
                .collect();
        Some(TerrainLayerData {
            name: "terrain-crossfade",
            opacity: layer.config.opacity,
            tiles: overlay_tiles,
            imagery_views,
            elevation_data: overlay_elev,
            tile_opacity_overrides: overlay_opacity,
        })
    } else {
        None
    };

    (base, overlay)
}
