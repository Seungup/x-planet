//! Build `RenderLayerData` and `TerrainLayerData` for each visible layer,
//! including cross-fade overlay logic.
//!
//! These are free functions (not methods) to avoid borrow conflicts: the
//! caller destructures `NativeApp` fields and passes immutable references.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use x_planets_core::engine::LayerKind;
use x_planets_core::render::RenderLayerData;
use x_planets_core::{MapEngine, TerrainLayerData, TerrainTileData};
use x_planets_math::TileCoord;

use crate::animation::{AnimationState, FADE_DURATION};
use crate::tile_source::NativeLayerState;

/// Compute cross-fade tiles: identify tiles transitioning parent → child.
///
/// During the fade-in period, exclude child tiles from the `available` set
/// so `resolve_fallbacks` picks the parent texture as the base. Returns
/// the modified available set and a list of `(coord, fade_t)` pairs for
/// the overlay pass.
fn compute_crossfade(
    visible: &[TileCoord],
    available: &HashSet<TileCoord>,
    anim: &AnimationState,
    now: Instant,
) -> (HashSet<TileCoord>, Vec<(TileCoord, f32)>) {
    let mut available_for_base = available.clone();
    let mut crossfade_tiles: Vec<(TileCoord, f32)> = Vec::new();

    for &coord in visible {
        if !available.contains(&coord) { continue; }
        if let Some(&start) = anim.tile_fade_start.get(&coord) {
            let elapsed = now.duration_since(start).as_secs_f64();
            if elapsed < FADE_DURATION {
                let has_parent = {
                    let mut c = coord.parent();
                    let mut found = false;
                    while let Some(p) = c {
                        if available.contains(&p) {
                            found = true;
                            break;
                        }
                        c = p.parent();
                    }
                    found
                };
                if has_parent {
                    available_for_base.remove(&coord);
                    let fade_t = ((elapsed / FADE_DURATION) as f32)
                        .clamp(1.0 / 60.0, 1.0);
                    crossfade_tiles.push((coord, fade_t));
                }
            }
        }
    }

    (available_for_base, crossfade_tiles)
}

/// Build render data for all visible layers.
///
/// Returns `(raster_layers, terrain_base_layers, terrain_overlay_layers)`.
pub(super) fn build_all_layers<'a>(
    engine: &'a MapEngine,
    layer_states: &'a [NativeLayerState],
    anim: &AnimationState,
    visible: &[TileCoord],
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
    // These will be skipped in the flat raster render pass — they're already
    // draped onto the 3D terrain mesh.
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
                    // Skip raster layers that serve as terrain imagery —
                    // they're already draped onto the 3D terrain mesh.
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
                    // Find companion imagery layer's texture views
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
    visible: &[TileCoord],
    now: Instant,
) -> (RenderLayerData<'a>, Option<RenderLayerData<'a>>) {
    let available: HashSet<TileCoord> = ls.tile_textures.keys().copied().collect();

    // Build texture view map
    let texture_views: HashMap<TileCoord, &wgpu::TextureView> = ls
        .tile_textures
        .iter()
        .map(|(k, v)| (*k, &v.view))
        .collect();

    let (available_for_base, crossfade_tiles) =
        compute_crossfade(visible, &available, anim, now);

    let renderable =
        x_planets_core::pipeline::resolve_fallbacks(visible, &available_for_base);

    // Opacity overrides: only for tiles with NO parent
    // coverage (first-time appearance, fade from zero).
    let mut tile_opacity_overrides = HashMap::new();
    for rt in &renderable {
        if rt.texture_coord != rt.coord {
            continue; // using parent fallback → full opacity
        }
        if let Some(&start) = anim.tile_fade_start.get(&rt.coord) {
            let elapsed = now.duration_since(start).as_secs_f64();
            if elapsed < FADE_DURATION {
                // No parent coverage → fade from near-zero
                let t = ((elapsed / FADE_DURATION) as f32)
                    .clamp(1.0 / 60.0, 1.0);
                tile_opacity_overrides.insert(
                    rt.coord,
                    layer.config.opacity * t,
                );
            }
        }
    }

    // Base layer: parent fallbacks for crossfading tiles,
    // own textures for tiles that finished fading or have
    // no parent coverage.
    let base = RenderLayerData {
        name: &layer.config.name,
        opacity: layer.config.opacity,
        tiles: renderable,
        texture_views: texture_views.clone(),
        tile_opacity_overrides,
    };

    // Cross-fade overlay: child tiles fading in over parent.
    // Rendered as a separate layer — each layer gets its own
    // render pass with cleared depth, so the overlay composites
    // correctly via alpha blending.
    let overlay = if !crossfade_tiles.is_empty() {
        let mut overlay_tiles = Vec::new();
        let mut overlay_opacity = HashMap::new();
        for &(coord, fade_t) in &crossfade_tiles {
            overlay_tiles.push(
                x_planets_core::pipeline::RenderableTile {
                    coord,
                    texture_coord: coord,
                    uv_rect: [0.0, 0.0, 1.0, 1.0],
                },
            );
            overlay_opacity.insert(
                coord,
                layer.config.opacity * fade_t,
            );
        }
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
    visible: &[TileCoord],
    now: Instant,
) -> (TerrainLayerData<'a>, Option<TerrainLayerData<'a>>) {
    let available: HashSet<TileCoord> = imagery_ls.tile_textures.keys().copied().collect();

    // Imagery texture views from companion layer
    let imagery_views: HashMap<TileCoord, &wgpu::TextureView> = imagery_ls
        .tile_textures
        .iter()
        .map(|(k, v)| (*k, &v.view))
        .collect();

    let (available_for_base, crossfade_tiles) =
        compute_crossfade(visible, &available, anim, now);

    let renderable = x_planets_core::pipeline::resolve_fallbacks(
        visible, &available_for_base,
    );

    // Elevation data with parent fallback.
    // Include coords for both base and overlay tiles.
    let mut elevation_data: HashMap<TileCoord, (&TerrainTileData, TileCoord)> = HashMap::new();
    let all_needed: HashSet<TileCoord> = renderable
        .iter()
        .map(|rt| rt.coord)
        .chain(crossfade_tiles.iter().map(|&(c, _)| c))
        .collect();
    for &coord in &all_needed {
        let mut c = Some(coord);
        while let Some(candidate) = c {
            if let Some(data) = terrain_ls.terrain_data.peek(&candidate) {
                // Accept any elevation data including parent
                // PrebuiltMesh tiles.  When a parent QM mesh is
                // used for a child coord the renderer generates a
                // flat placeholder so the imagery is visible
                // immediately during the parent-first loading phase.
                elevation_data.insert(coord, (data, candidate));
                break;
            }
            c = candidate.parent();
        }
    }

    // Base terrain layer (parent fallback imagery for
    // crossfading tiles, own imagery for stable tiles).
    let base = TerrainLayerData {
        name: &layer.config.name,
        opacity: layer.config.opacity,
        tiles: renderable,
        imagery_views: imagery_views.clone(),
        elevation_data: elevation_data.clone(),
        tile_opacity_overrides: HashMap::new(),
    };

    // Cross-fade overlay: child imagery fading in.
    // Must be rendered in a SEPARATE render_terrain_layered
    // call because the mesh cache shares uniform buffers
    // per coord — a single call would overwrite the base
    // pass uniforms before submission.
    let overlay = if !crossfade_tiles.is_empty() {
        let overlay_tiles: Vec<_> = crossfade_tiles
            .iter()
            .map(|&(coord, _)| {
                x_planets_core::pipeline::RenderableTile {
                    coord,
                    texture_coord: coord,
                    uv_rect: [0.0, 0.0, 1.0, 1.0],
                }
            })
            .collect();
        let mut overlay_opacity = HashMap::new();
        for &(coord, fade_t) in &crossfade_tiles {
            overlay_opacity.insert(
                coord,
                layer.config.opacity * fade_t,
            );
        }
        let overlay_elev: HashMap<TileCoord, (&TerrainTileData, TileCoord)> =
            crossfade_tiles
                .iter()
                .filter_map(|&(coord, _)| {
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
