//! LOD traversal engine for 3D Tiles.
//!
//! Implements the screen-space error (SSE) based traversal algorithm
//! from the OGC 3D Tiles specification. Each frame, the traversal
//! determines which tiles to render, which to load, and which to unload.
//!
//! Pure functions (Karpathy principle) — no GPU, no I/O, fully testable.

use std::collections::HashSet;

use glam::{DMat4, DVec3};

use super::bounding_volume::{
    distance_to_volume, extract_frustum_planes, is_sphere_visible, screen_space_error,
    transform_volume,
};
use super::tileset::{effective_refine, resolve_content_uri, Refine, Tile, Tileset};

// ═══════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════

/// Configuration for LOD traversal.
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Maximum screen-space error in pixels before refining to children.
    /// Lower values = higher quality, more tiles loaded.
    /// Typical values: 8–16 pixels.
    pub max_sse: f64,
    /// Maximum number of tiles to render per frame (budget).
    pub tile_budget: usize,
    /// Screen height in pixels (for SSE calculation).
    pub screen_height: f64,
    /// Vertical field of view in radians (for SSE calculation).
    pub fov_y: f64,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_sse: 16.0,
            tile_budget: 256,
            screen_height: 1080.0,
            fov_y: 60.0_f64.to_radians(),
        }
    }
}

/// Camera state for traversal (in ECEF coordinates).
#[derive(Debug, Clone)]
pub struct TraversalCamera {
    /// Camera position in ECEF coordinates (meters).
    pub position_ecef: DVec3,
    /// View-projection matrix (ECEF → clip space).
    pub view_proj: DMat4,
}

// ═══════════════════════════════════════════════════════════════════
// Traversal results
// ═══════════════════════════════════════════════════════════════════

/// Result of a single frame's LOD traversal.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    /// Tiles to render this frame.
    pub render_set: Vec<TraversalTile>,
    /// Content URIs to load (sorted by priority, highest first).
    pub load_requests: Vec<LoadRequest3d>,
    /// Content URIs that are no longer needed and can be unloaded.
    pub unload_set: Vec<String>,
}

/// A tile selected for rendering.
#[derive(Debug, Clone)]
pub struct TraversalTile {
    /// Resolved content URI (absolute URL).
    pub content_uri: String,
    /// Combined model matrix (product of ancestor transforms).
    pub transform: DMat4,
    /// Tile's geometric error.
    pub geometric_error: f64,
    /// Computed screen-space error for this tile.
    pub sse: f64,
}

/// A request to load tile content.
#[derive(Debug, Clone)]
pub struct LoadRequest3d {
    /// Resolved content URI (absolute URL).
    pub content_uri: String,
    /// Priority score (higher = more important = should load first).
    pub priority: f64,
}

// ═══════════════════════════════════════════════════════════════════
// Main traversal function
// ═══════════════════════════════════════════════════════════════════

/// Traverse the tileset tree and determine which tiles to render/load/unload.
///
/// This is the core LOD algorithm, called every frame.
///
/// # Arguments
/// - `tileset`: The parsed tileset.json.
/// - `base_url`: Base URL for resolving relative content URIs.
/// - `camera`: Camera state in ECEF.
/// - `loaded_uris`: Set of content URIs that are already loaded and available for rendering.
/// - `config`: Traversal parameters (max SSE, budget, etc.).
///
/// # Returns
/// `TraversalResult` with render set, load requests, and unload set.
pub fn traverse_tileset(
    tileset: &Tileset,
    base_url: &str,
    camera: &TraversalCamera,
    loaded_uris: &HashSet<String>,
    config: &TraversalConfig,
) -> TraversalResult {
    let mut result = TraversalResult {
        render_set: Vec::new(),
        load_requests: Vec::new(),
        unload_set: Vec::new(),
    };

    let frustum_planes = extract_frustum_planes(&camera.view_proj);
    let parent_refine = tileset.root.refine.unwrap_or(Refine::Replace);

    // Start traversal from root.
    traverse_tile(
        &tileset.root,
        base_url,
        DMat4::IDENTITY,
        parent_refine,
        camera,
        &frustum_planes,
        loaded_uris,
        config,
        &mut result,
    );

    // Sort load requests by priority (highest first).
    result
        .load_requests
        .sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

    // Trim to budget.
    if result.render_set.len() > config.tile_budget {
        // Keep only the most important tiles (lowest SSE = most refined).
        result.render_set.sort_by(|a, b| {
            a.sse
                .partial_cmp(&b.sse)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result.render_set.truncate(config.tile_budget);
    }

    // Build unload set: loaded URIs that aren't in the render or load set.
    let active_uris: HashSet<&str> = result
        .render_set
        .iter()
        .map(|t| t.content_uri.as_str())
        .chain(result.load_requests.iter().map(|r| r.content_uri.as_str()))
        .collect();

    for uri in loaded_uris {
        if !active_uris.contains(uri.as_str()) {
            result.unload_set.push(uri.clone());
        }
    }

    result
}

/// Recursively traverse a tile node.
#[allow(clippy::too_many_arguments)]
fn traverse_tile(
    tile: &Tile,
    base_url: &str,
    parent_transform: DMat4,
    parent_refine: Refine,
    camera: &TraversalCamera,
    frustum_planes: &[super::bounding_volume::FrustumPlane; 6],
    loaded_uris: &HashSet<String>,
    config: &TraversalConfig,
    result: &mut TraversalResult,
) {
    // ── Compute effective transform ──
    let tile_transform = if let Some(t) = &tile.transform {
        parent_transform * DMat4::from_cols_array(t)
    } else {
        parent_transform
    };

    // ── Resolve bounding volume in world space ──
    // Use the accumulated tile_transform (parent × local), not just the
    // tile's own transform, so that parent translations/rotations are
    // correctly applied to children's bounding volumes.
    let bv_kind = match tile.bounding_volume.to_kind() {
        Some(kind) => {
            if tile_transform != DMat4::IDENTITY {
                transform_volume(&kind, &tile_transform.to_cols_array())
            } else {
                kind
            }
        }
        None => return, // No valid bounding volume — skip.
    };

    // ── Frustum culling ──
    let center = bv_kind.center_ecef();
    let radius = bv_kind.bounding_radius();
    if !is_sphere_visible(center, radius, frustum_planes) {
        return; // Not visible — skip entire subtree.
    }

    // ── Compute SSE ──
    let distance = distance_to_volume(camera.position_ecef, &bv_kind);
    let sse = screen_space_error(
        tile.geometric_error,
        distance,
        config.screen_height,
        config.fov_y,
    );

    let refine = effective_refine(tile, parent_refine);
    let content_uri = resolve_content_uri(base_url, tile);

    // ── Decision: render this tile or refine to children? ──
    if sse <= config.max_sse || tile.children.is_empty() {
        // SSE is small enough OR leaf tile → render this tile.
        if let Some(uri) = &content_uri {
            if loaded_uris.contains(uri) {
                result.render_set.push(TraversalTile {
                    content_uri: uri.clone(),
                    transform: tile_transform,
                    geometric_error: tile.geometric_error,
                    sse,
                });
            } else {
                // Not loaded yet — request it.
                result.load_requests.push(LoadRequest3d {
                    content_uri: uri.clone(),
                    priority: sse, // Higher SSE = higher priority.
                });
            }
        }
    } else {
        // SSE too large → refine (recurse into children).
        match refine {
            Refine::Replace => {
                // REPLACE: children replace the parent.
                // Check if all children with content are loaded.
                let children_ready = tile.children.iter().all(|child| {
                    resolve_content_uri(base_url, child)
                        .map(|uri| loaded_uris.contains(&uri))
                        .unwrap_or(true) // Children without content are "ready".
                });

                if children_ready {
                    // All children loaded → render children, not parent.
                    for child in &tile.children {
                        traverse_tile(
                            child,
                            base_url,
                            tile_transform,
                            refine,
                            camera,
                            frustum_planes,
                            loaded_uris,
                            config,
                            result,
                        );
                    }
                } else {
                    // Not all children loaded → render parent as placeholder.
                    if let Some(uri) = &content_uri {
                        if loaded_uris.contains(uri) {
                            result.render_set.push(TraversalTile {
                                content_uri: uri.clone(),
                                transform: tile_transform,
                                geometric_error: tile.geometric_error,
                                sse,
                            });
                        }
                    }
                    // Request missing children with per-child SSE priority.
                    for child in &tile.children {
                        if let Some(child_uri) = resolve_content_uri(base_url, child) {
                            if !loaded_uris.contains(&child_uri) {
                                // Compute child's own SSE for more accurate prioritization.
                                let child_transform = if let Some(t) = &child.transform {
                                    tile_transform * DMat4::from_cols_array(t)
                                } else {
                                    tile_transform
                                };
                                let child_priority = child
                                    .bounding_volume
                                    .to_kind()
                                    .map(|bv| {
                                        let bv = if child_transform != DMat4::IDENTITY {
                                            transform_volume(&bv, &child_transform.to_cols_array())
                                        } else {
                                            bv
                                        };
                                        let d = distance_to_volume(camera.position_ecef, &bv);
                                        screen_space_error(
                                            child.geometric_error,
                                            d,
                                            config.screen_height,
                                            config.fov_y,
                                        )
                                    })
                                    .unwrap_or(sse); // Fall back to parent SSE if no bounding volume.
                                result.load_requests.push(LoadRequest3d {
                                    content_uri: child_uri,
                                    priority: child_priority,
                                });
                            }
                        }
                        // Recurse into children that are loaded (they may have further children).
                        traverse_tile(
                            child,
                            base_url,
                            tile_transform,
                            refine,
                            camera,
                            frustum_planes,
                            loaded_uris,
                            config,
                            result,
                        );
                    }
                }
            }
            Refine::Add => {
                // ADD: render parent AND recurse into children.
                if let Some(uri) = &content_uri {
                    if loaded_uris.contains(uri) {
                        result.render_set.push(TraversalTile {
                            content_uri: uri.clone(),
                            transform: tile_transform,
                            geometric_error: tile.geometric_error,
                            sse,
                        });
                    } else {
                        result.load_requests.push(LoadRequest3d {
                            content_uri: uri.clone(),
                            priority: sse,
                        });
                    }
                }
                // Always recurse into children for ADD.
                for child in &tile.children {
                    traverse_tile(
                        child,
                        base_url,
                        tile_transform,
                        refine,
                        camera,
                        frustum_planes,
                        loaded_uris,
                        config,
                        result,
                    );
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiles3d::tileset::parse_tileset;

    fn make_camera_at_origin() -> TraversalCamera {
        let camera_pos = DVec3::new(6_378_137.0 + 100_000.0, 0.0, 0.0); // 100km above equator

        // Look toward Earth's center (inward along -X at equator/prime meridian).
        let target = DVec3::ZERO;
        let up = DVec3::new(0.0, 0.0, 1.0); // Z-up for ENU at equator

        let view = DMat4::look_at_rh(camera_pos, target, up);
        let proj = DMat4::perspective_rh(
            60.0_f64.to_radians(),
            1.0,
            1.0,
            100_000_000.0,
        );

        TraversalCamera {
            position_ecef: camera_pos,
            view_proj: proj * view,
        }
    }

    fn make_config() -> TraversalConfig {
        TraversalConfig {
            max_sse: 16.0,
            tile_budget: 256,
            screen_height: 1080.0,
            fov_y: 60.0_f64.to_radians(),
        }
    }

    // Helper: build a simple 2-level tileset for testing.
    fn two_level_tileset_json() -> &'static str {
        r#"{
            "asset": { "version": "1.1" },
            "geometricError": 500.0,
            "root": {
                "boundingVolume": {
                    "sphere": [6378137.0, 0.0, 0.0, 10000.0]
                },
                "geometricError": 200.0,
                "refine": "REPLACE",
                "content": { "uri": "root.glb" },
                "children": [
                    {
                        "boundingVolume": {
                            "sphere": [6378137.0, 0.0, 0.0, 5000.0]
                        },
                        "geometricError": 50.0,
                        "content": { "uri": "child0.glb" }
                    },
                    {
                        "boundingVolume": {
                            "sphere": [6378137.0, 0.0, 0.0, 5000.0]
                        },
                        "geometricError": 50.0,
                        "content": { "uri": "child1.glb" }
                    }
                ]
            }
        }"#
    }

    #[test]
    fn test_traverse_root_only() {
        let tileset = parse_tileset(two_level_tileset_json().as_bytes()).unwrap();
        let camera = make_camera_at_origin();
        let config = TraversalConfig {
            max_sse: 10000.0, // Very high → never refine.
            ..make_config()
        };
        let loaded = HashSet::from(["https://example.com/root.glb".to_string()]);

        let result = traverse_tileset(
            &tileset,
            "https://example.com/tileset.json",
            &camera,
            &loaded,
            &config,
        );

        // Should render only root (SSE threshold too high to refine).
        assert_eq!(result.render_set.len(), 1);
        assert_eq!(
            result.render_set[0].content_uri,
            "https://example.com/root.glb"
        );
        assert!(result.load_requests.is_empty());
    }

    #[test]
    fn test_traverse_refine_to_children() {
        let tileset = parse_tileset(two_level_tileset_json().as_bytes()).unwrap();
        let camera = make_camera_at_origin();
        let config = TraversalConfig {
            max_sse: 0.001, // Very low → always refine.
            ..make_config()
        };
        let loaded = HashSet::from([
            "https://example.com/root.glb".to_string(),
            "https://example.com/child0.glb".to_string(),
            "https://example.com/child1.glb".to_string(),
        ]);

        let result = traverse_tileset(
            &tileset,
            "https://example.com/tileset.json",
            &camera,
            &loaded,
            &config,
        );

        // Both children should be in render set, not the root.
        let uris: HashSet<&str> = result.render_set.iter().map(|t| t.content_uri.as_str()).collect();
        assert!(uris.contains("https://example.com/child0.glb"));
        assert!(uris.contains("https://example.com/child1.glb"));
        assert!(!uris.contains("https://example.com/root.glb"));
    }

    #[test]
    fn test_traverse_replace_pending_children() {
        // When children are NOT loaded, REPLACE should render parent + request children.
        let tileset = parse_tileset(two_level_tileset_json().as_bytes()).unwrap();
        let camera = make_camera_at_origin();
        let config = TraversalConfig {
            max_sse: 0.001,
            ..make_config()
        };
        let loaded = HashSet::from(["https://example.com/root.glb".to_string()]);

        let result = traverse_tileset(
            &tileset,
            "https://example.com/tileset.json",
            &camera,
            &loaded,
            &config,
        );

        // Parent should be in render set (placeholder).
        let render_uris: HashSet<&str> =
            result.render_set.iter().map(|t| t.content_uri.as_str()).collect();
        assert!(render_uris.contains("https://example.com/root.glb"));

        // Children should be requested.
        let load_uris: HashSet<&str> = result
            .load_requests
            .iter()
            .map(|r| r.content_uri.as_str())
            .collect();
        assert!(load_uris.contains("https://example.com/child0.glb"));
        assert!(load_uris.contains("https://example.com/child1.glb"));
    }

    #[test]
    fn test_traverse_add_refinement() {
        // ADD refinement: parent AND children should both be rendered.
        let json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 500.0,
            "root": {
                "boundingVolume": {
                    "sphere": [6378137.0, 0.0, 0.0, 10000.0]
                },
                "geometricError": 200.0,
                "refine": "ADD",
                "content": { "uri": "root.glb" },
                "children": [
                    {
                        "boundingVolume": {
                            "sphere": [6378137.0, 0.0, 0.0, 5000.0]
                        },
                        "geometricError": 10.0,
                        "content": { "uri": "child.glb" }
                    }
                ]
            }
        }"#;
        let tileset = parse_tileset(json.as_bytes()).unwrap();
        let camera = make_camera_at_origin();
        let config = TraversalConfig {
            max_sse: 0.001,
            ..make_config()
        };
        let loaded = HashSet::from([
            "https://example.com/root.glb".to_string(),
            "https://example.com/child.glb".to_string(),
        ]);

        let result = traverse_tileset(
            &tileset,
            "https://example.com/tileset.json",
            &camera,
            &loaded,
            &config,
        );

        // Both parent AND child should be rendered (ADD).
        let uris: HashSet<&str> = result.render_set.iter().map(|t| t.content_uri.as_str()).collect();
        assert!(uris.contains("https://example.com/root.glb"));
        assert!(uris.contains("https://example.com/child.glb"));
    }

    #[test]
    fn test_traverse_unloaded_root_requests_load() {
        let tileset = parse_tileset(two_level_tileset_json().as_bytes()).unwrap();
        let camera = make_camera_at_origin();
        let config = make_config();
        let loaded = HashSet::new(); // Nothing loaded.

        let result = traverse_tileset(
            &tileset,
            "https://example.com/tileset.json",
            &camera,
            &loaded,
            &config,
        );

        // Root should be requested for loading.
        assert!(result.render_set.is_empty());
        assert!(!result.load_requests.is_empty());
    }

    #[test]
    fn test_traverse_unload_unused() {
        let tileset = parse_tileset(two_level_tileset_json().as_bytes()).unwrap();
        let camera = make_camera_at_origin();
        let config = TraversalConfig {
            max_sse: 10000.0, // Never refine → only root needed.
            ..make_config()
        };
        let loaded = HashSet::from([
            "https://example.com/root.glb".to_string(),
            "https://example.com/old_tile.glb".to_string(), // Not in tileset.
        ]);

        let result = traverse_tileset(
            &tileset,
            "https://example.com/tileset.json",
            &camera,
            &loaded,
            &config,
        );

        // old_tile should be in unload set.
        assert!(result.unload_set.contains(&"https://example.com/old_tile.glb".to_string()));
    }

    #[test]
    fn test_traverse_tile_budget() {
        let tileset = parse_tileset(two_level_tileset_json().as_bytes()).unwrap();
        let camera = make_camera_at_origin();
        let config = TraversalConfig {
            max_sse: 0.001,
            tile_budget: 1, // Only 1 tile allowed.
            ..make_config()
        };
        let loaded = HashSet::from([
            "https://example.com/root.glb".to_string(),
            "https://example.com/child0.glb".to_string(),
            "https://example.com/child1.glb".to_string(),
        ]);

        let result = traverse_tileset(
            &tileset,
            "https://example.com/tileset.json",
            &camera,
            &loaded,
            &config,
        );

        // Render set should be trimmed to budget.
        assert!(result.render_set.len() <= 1);
    }

    #[test]
    fn test_load_requests_sorted_by_priority() {
        let tileset = parse_tileset(two_level_tileset_json().as_bytes()).unwrap();
        let camera = make_camera_at_origin();
        let config = TraversalConfig {
            max_sse: 0.001,
            ..make_config()
        };
        let loaded = HashSet::from(["https://example.com/root.glb".to_string()]);

        let result = traverse_tileset(
            &tileset,
            "https://example.com/tileset.json",
            &camera,
            &loaded,
            &config,
        );

        // Load requests should be sorted by priority (descending).
        for i in 1..result.load_requests.len() {
            assert!(
                result.load_requests[i - 1].priority >= result.load_requests[i].priority,
                "Load requests not sorted by priority"
            );
        }
    }

    #[test]
    fn test_leaf_tile_always_renders() {
        // A leaf tile (no children) should always render regardless of SSE.
        let json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 500.0,
            "root": {
                "boundingVolume": {
                    "sphere": [6378137.0, 0.0, 0.0, 10000.0]
                },
                "geometricError": 99999.0,
                "refine": "REPLACE",
                "content": { "uri": "root.glb" }
            }
        }"#;
        let tileset = parse_tileset(json.as_bytes()).unwrap();
        let camera = make_camera_at_origin();
        let config = TraversalConfig {
            max_sse: 0.001, // Very low but root is a leaf.
            ..make_config()
        };
        let loaded = HashSet::from(["https://example.com/root.glb".to_string()]);

        let result = traverse_tileset(
            &tileset,
            "https://example.com/tileset.json",
            &camera,
            &loaded,
            &config,
        );

        // Leaf should render even with huge SSE.
        assert_eq!(result.render_set.len(), 1);
    }

    #[test]
    fn test_zero_geometric_error_never_refines() {
        let json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 0.0,
            "root": {
                "boundingVolume": {
                    "sphere": [6378137.0, 0.0, 0.0, 10000.0]
                },
                "geometricError": 0.0,
                "refine": "REPLACE",
                "content": { "uri": "root.glb" },
                "children": [
                    {
                        "boundingVolume": {
                            "sphere": [6378137.0, 0.0, 0.0, 5000.0]
                        },
                        "geometricError": 0.0,
                        "content": { "uri": "child.glb" }
                    }
                ]
            }
        }"#;
        let tileset = parse_tileset(json.as_bytes()).unwrap();
        let camera = make_camera_at_origin();
        let config = TraversalConfig {
            max_sse: 0.0,
            ..make_config()
        };
        let loaded = HashSet::from([
            "https://example.com/root.glb".to_string(),
            "https://example.com/child.glb".to_string(),
        ]);

        let result = traverse_tileset(
            &tileset,
            "https://example.com/tileset.json",
            &camera,
            &loaded,
            &config,
        );

        // With geometric_error=0, SSE is 0, which is <= max_sse=0, so no refinement.
        let uris: HashSet<&str> = result.render_set.iter().map(|t| t.content_uri.as_str()).collect();
        assert!(uris.contains("https://example.com/root.glb"));
    }

    #[test]
    fn test_tile_with_transform_inherits_parent() {
        let json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 500.0,
            "root": {
                "boundingVolume": {
                    "sphere": [6378137.0, 0.0, 0.0, 10000.0]
                },
                "geometricError": 200.0,
                "refine": "REPLACE",
                "transform": [
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    100, 0, 0, 1
                ],
                "content": { "uri": "root.glb" },
                "children": [
                    {
                        "boundingVolume": {
                            "sphere": [6378137.0, 0.0, 0.0, 5000.0]
                        },
                        "geometricError": 50.0,
                        "content": { "uri": "child.glb" }
                    }
                ]
            }
        }"#;
        let tileset = parse_tileset(json.as_bytes()).unwrap();
        let camera = make_camera_at_origin();
        let config = TraversalConfig {
            max_sse: 0.001,
            ..make_config()
        };
        let loaded = HashSet::from([
            "https://example.com/root.glb".to_string(),
            "https://example.com/child.glb".to_string(),
        ]);

        let result = traverse_tileset(
            &tileset,
            "https://example.com/tileset.json",
            &camera,
            &loaded,
            &config,
        );

        // Child should inherit root's transform.
        let child_tile = result.render_set.iter()
            .find(|t| t.content_uri == "https://example.com/child.glb");
        assert!(child_tile.is_some(), "child tile should be in render set");
        // The child should have the parent's transform applied.
        let t = child_tile.unwrap().transform;
        // Column-major: m[3][0] should be 100 (translation X from parent).
        assert!((t.col(3).x - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_traversal_config_default() {
        let config = TraversalConfig::default();
        assert_eq!(config.max_sse, 16.0);
        assert_eq!(config.tile_budget, 256);
        assert_eq!(config.screen_height, 1080.0);
        assert!((config.fov_y - 60.0_f64.to_radians()).abs() < 1e-10);
    }
}
