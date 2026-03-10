//! OGC 3D Tiles tileset.json parser.
//!
//! Supports both 3D Tiles 1.0 (B3DM/I3DM/PNTS) and 1.1 (glTF/GLB).
//! Spec: <https://docs.ogc.org/cs/22-025r4/22-025r4.html>

use serde::{Deserialize, Serialize};

use super::bounding_volume::BoundingVolume;

// ═══════════════════════════════════════════════════════════════════
// Tileset root
// ═══════════════════════════════════════════════════════════════════

/// Root of a 3D Tiles tileset.json.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tileset {
    pub asset: Asset,
    pub geometric_error: f64,
    pub root: Tile,
    #[serde(default)]
    pub properties: Option<serde_json::Value>,
    #[serde(default)]
    pub extensions_used: Vec<String>,
    #[serde(default)]
    pub extensions_required: Vec<String>,
}

/// Asset metadata.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Asset {
    pub version: String,
    #[serde(default)]
    pub tileset_version: Option<String>,
    #[serde(default)]
    pub generator: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════
// Tile node
// ═══════════════════════════════════════════════════════════════════

/// A tile in the 3D Tiles tree.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tile {
    pub bounding_volume: BoundingVolume,
    pub geometric_error: f64,
    #[serde(default)]
    pub refine: Option<Refine>,
    #[serde(default)]
    pub content: Option<TileContent>,
    #[serde(default)]
    pub children: Vec<Tile>,
    /// Column-major 4x4 transform matrix (16 doubles).
    #[serde(default)]
    pub transform: Option<[f64; 16]>,
    #[serde(default)]
    pub viewer_request_volume: Option<BoundingVolume>,
    /// Implicit tiling extension (3D Tiles 1.1).
    #[serde(default)]
    pub implicit_tiling: Option<ImplicitTiling>,
}

/// Refinement strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Refine {
    /// Children replace the parent when refined.
    Replace,
    /// Children are added alongside the parent.
    Add,
}

/// Tile content reference.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TileContent {
    pub uri: String,
    #[serde(default)]
    pub bounding_volume: Option<BoundingVolume>,
}

// ═══════════════════════════════════════════════════════════════════
// Implicit tiling (3D Tiles 1.1)
// ═══════════════════════════════════════════════════════════════════

/// Implicit tiling extension for quadtree/octree subdivision.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImplicitTiling {
    pub subdivision_scheme: SubdivisionScheme,
    pub subtree_levels: u32,
    pub available_levels: u32,
    pub subtrees: TemplateUri,
    #[serde(default)]
    pub content: Option<TemplateUri>,
}

/// Subdivision scheme for implicit tiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SubdivisionScheme {
    Quadtree,
    Octree,
}

/// URI template with placeholders like `{level}`, `{x}`, `{y}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TemplateUri {
    pub uri: String,
}

// ═══════════════════════════════════════════════════════════════════
// Parsing functions
// ═══════════════════════════════════════════════════════════════════

/// Parse a tileset.json from bytes.
pub fn parse_tileset(json: &[u8]) -> Result<Tileset, serde_json::Error> {
    serde_json::from_slice(json)
}

/// Resolve a tile content URI relative to a base URL.
///
/// Returns `None` if the tile has no content.
pub fn resolve_content_uri(base_url: &str, tile: &Tile) -> Option<String> {
    let content = tile.content.as_ref()?;
    let uri = &content.uri;

    if uri.starts_with("http://") || uri.starts_with("https://") {
        // Absolute URL
        Some(uri.clone())
    } else if uri.starts_with('/') {
        // Absolute path — extract origin from base_url
        if let Some(origin_end) = base_url.find("://").map(|i| {
            base_url[i + 3..]
                .find('/')
                .map(|j| i + 3 + j)
                .unwrap_or(base_url.len())
        }) {
            Some(format!("{}{}", &base_url[..origin_end], uri))
        } else {
            Some(uri.clone())
        }
    } else {
        // Relative URI — combine with base
        let base = if let Some(slash_pos) = base_url.rfind('/') {
            &base_url[..=slash_pos]
        } else {
            base_url
        };
        Some(format!("{}{}", base, uri))
    }
}

/// Recursively count all tiles in the tree (including the root).
pub fn tile_count(tile: &Tile) -> usize {
    1 + tile.children.iter().map(tile_count).sum::<usize>()
}

/// Splice an external tileset into the main tileset tree.
///
/// Finds the tile whose **resolved** content URI matches `resolved_content_uri`
/// and replaces it with the external tileset's root. The external root's
/// children become children of the matched tile, and the external root's
/// content (if any) replaces the original `.json` content reference.
///
/// `external_base_url` is the base URL for the external tileset (derived from
/// the .json file's URL). All relative URIs in the spliced subtree are resolved
/// to absolute URLs using this base.
///
/// Returns `true` if the splice was performed.
pub fn splice_external_tileset(
    tile: &mut Tile,
    base_url: &str,
    resolved_content_uri: &str,
    external: &Tileset,
    external_base_url: &str,
) -> bool {
    // Check if this tile's resolved content URI matches.
    if let Some(resolved) = resolve_content_uri(base_url, tile) {
        if resolved == resolved_content_uri {
            let ext_root = &external.root;

            // Replace content with external root's content.
            tile.content = ext_root.content.clone();

            // Replace children with external root's children.
            tile.children = ext_root.children.clone();

            // Compose transforms: if both the referencing tile and the
            // external root have transforms, they must be composed so that
            // traversal applies both.  Per the 3D Tiles spec, the external
            // tileset root is positioned relative to the referencing tile's
            // coordinate frame.
            match (tile.transform, ext_root.transform) {
                (Some(parent_t), Some(ext_t)) => {
                    let parent_mat = glam::DMat4::from_cols_array(&parent_t);
                    let ext_mat = glam::DMat4::from_cols_array(&ext_t);
                    tile.transform = Some((parent_mat * ext_mat).to_cols_array());
                }
                (None, Some(_)) => {
                    tile.transform = ext_root.transform;
                }
                _ => {} // Keep existing transform (or both None)
            }

            // Use external root's bounding volume.
            tile.bounding_volume = ext_root.bounding_volume.clone();

            // Inherit geometric error from external root.
            tile.geometric_error = ext_root.geometric_error;

            // Inherit refine strategy if not set.
            if tile.refine.is_none() {
                tile.refine = ext_root.refine;
            }

            // Resolve all relative URIs in the spliced subtree to absolute.
            resolve_all_uris(tile, external_base_url);

            return true;
        }
    }

    // Recurse into children.
    for child in &mut tile.children {
        if splice_external_tileset(child, base_url, resolved_content_uri, external, external_base_url) {
            return true;
        }
    }

    false
}

/// Get the effective refine strategy for a tile (inherits from parent if not set).
pub fn effective_refine(tile: &Tile, parent_refine: Refine) -> Refine {
    tile.refine.unwrap_or(parent_refine)
}

/// Resolve all relative content URIs in a tile tree to absolute URLs.
///
/// This is used after splicing an external tileset so that all URIs
/// in the subtree are absolute and don't depend on the base URL.
pub fn resolve_all_uris(tile: &mut Tile, base_url: &str) {
    if let Some(ref mut content) = tile.content {
        if !content.uri.starts_with("http://") && !content.uri.starts_with("https://") {
            // Resolve relative URI to absolute.
            if content.uri.starts_with('/') {
                // Absolute path — extract origin from base_url.
                if let Some(origin_end) = base_url.find("://").map(|i| {
                    base_url[i + 3..]
                        .find('/')
                        .map(|j| i + 3 + j)
                        .unwrap_or(base_url.len())
                }) {
                    content.uri = format!("{}{}", &base_url[..origin_end], content.uri);
                }
            } else {
                // Relative URI.
                let base = if let Some(slash_pos) = base_url.rfind('/') {
                    &base_url[..=slash_pos]
                } else {
                    base_url
                };
                content.uri = format!("{}{}", base, content.uri);
            }
        }
    }
    for child in &mut tile.children {
        resolve_all_uris(child, base_url);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TILESET: &str = r#"{
        "asset": { "version": "1.1", "generator": "test" },
        "geometricError": 240.0,
        "root": {
            "boundingVolume": {
                "region": [-1.3197, 0.6988, -1.3196, 0.6989, 0.0, 100.0]
            },
            "geometricError": 70.0,
            "refine": "REPLACE",
            "content": { "uri": "root.glb" },
            "children": [
                {
                    "boundingVolume": {
                        "box": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                    },
                    "geometricError": 10.0,
                    "content": { "uri": "child0.b3dm" }
                },
                {
                    "boundingVolume": {
                        "sphere": [0.0, 0.0, 0.0, 100.0]
                    },
                    "geometricError": 5.0,
                    "content": { "uri": "child1.glb" },
                    "refine": "ADD"
                }
            ]
        }
    }"#;

    #[test]
    fn test_parse_tileset() {
        let tileset = parse_tileset(SAMPLE_TILESET.as_bytes()).unwrap();
        assert_eq!(tileset.asset.version, "1.1");
        assert_eq!(tileset.geometric_error, 240.0);
        assert_eq!(tileset.root.geometric_error, 70.0);
        assert_eq!(tileset.root.refine, Some(Refine::Replace));
        assert_eq!(tileset.root.children.len(), 2);
    }

    #[test]
    fn test_parse_content_uri() {
        let tileset = parse_tileset(SAMPLE_TILESET.as_bytes()).unwrap();
        assert_eq!(
            tileset.root.content.as_ref().unwrap().uri,
            "root.glb"
        );
    }

    #[test]
    fn test_refine_strategies() {
        let tileset = parse_tileset(SAMPLE_TILESET.as_bytes()).unwrap();
        assert_eq!(tileset.root.refine, Some(Refine::Replace));
        // First child: no refine set → inherits
        assert_eq!(tileset.root.children[0].refine, None);
        assert_eq!(
            effective_refine(&tileset.root.children[0], Refine::Replace),
            Refine::Replace
        );
        // Second child: explicit ADD
        assert_eq!(tileset.root.children[1].refine, Some(Refine::Add));
    }

    #[test]
    fn test_tile_count() {
        let tileset = parse_tileset(SAMPLE_TILESET.as_bytes()).unwrap();
        assert_eq!(tile_count(&tileset.root), 3);
    }

    #[test]
    fn test_resolve_content_uri_relative() {
        let tileset = parse_tileset(SAMPLE_TILESET.as_bytes()).unwrap();
        let base = "https://assets.cesium.com/12345/tileset.json";
        let uri = resolve_content_uri(base, &tileset.root).unwrap();
        assert_eq!(uri, "https://assets.cesium.com/12345/root.glb");
    }

    #[test]
    fn test_resolve_content_uri_absolute() {
        let tile = Tile {
            bounding_volume: BoundingVolume::default(),
            geometric_error: 1.0,
            refine: None,
            content: Some(TileContent {
                uri: "https://example.com/tile.glb".to_string(),
                bounding_volume: None,
            }),
            children: vec![],
            transform: None,
            viewer_request_volume: None,
            implicit_tiling: None,
        };
        let uri = resolve_content_uri("https://other.com/tileset.json", &tile).unwrap();
        assert_eq!(uri, "https://example.com/tile.glb");
    }

    #[test]
    fn test_resolve_content_uri_absolute_path() {
        let tile = Tile {
            bounding_volume: BoundingVolume::default(),
            geometric_error: 1.0,
            refine: None,
            content: Some(TileContent {
                uri: "/v1/3dtiles/files/abc.json".to_string(),
                bounding_volume: None,
            }),
            children: vec![],
            transform: None,
            viewer_request_volume: None,
            implicit_tiling: None,
        };
        let uri =
            resolve_content_uri("https://tile.googleapis.com/v1/3dtiles/root.json", &tile)
                .unwrap();
        assert_eq!(
            uri,
            "https://tile.googleapis.com/v1/3dtiles/files/abc.json"
        );
    }

    #[test]
    fn test_no_content_returns_none() {
        let tile = Tile {
            bounding_volume: BoundingVolume::default(),
            geometric_error: 1.0,
            refine: None,
            content: None,
            children: vec![],
            transform: None,
            viewer_request_volume: None,
            implicit_tiling: None,
        };
        assert!(resolve_content_uri("https://example.com/tileset.json", &tile).is_none());
    }

    #[test]
    fn test_splice_external_tileset() {
        // Main tileset has a child referencing an external JSON.
        let main_json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 500.0,
            "root": {
                "boundingVolume": { "sphere": [0.0, 0.0, 0.0, 10000.0] },
                "geometricError": 200.0,
                "refine": "ADD",
                "content": { "uri": "root.b3dm" },
                "children": [
                    {
                        "boundingVolume": { "sphere": [0.0, 0.0, 0.0, 5000.0] },
                        "geometricError": 100.0,
                        "content": { "uri": "0-0-0.json" }
                    }
                ]
            }
        }"#;
        let external_json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 100.0,
            "root": {
                "boundingVolume": { "sphere": [0.0, 0.0, 0.0, 5000.0] },
                "geometricError": 50.0,
                "refine": "REPLACE",
                "content": { "uri": "tile.b3dm" },
                "children": [
                    {
                        "boundingVolume": { "sphere": [0.0, 0.0, 0.0, 2500.0] },
                        "geometricError": 25.0,
                        "content": { "uri": "child0.b3dm" }
                    }
                ]
            }
        }"#;

        let mut main_tileset = parse_tileset(main_json.as_bytes()).unwrap();
        let external = parse_tileset(external_json.as_bytes()).unwrap();
        let base_url = "https://example.com/tiles/";

        // The external tileset's base URL is derived from the .json file's URL.
        let external_base_url = "https://example.com/tiles/";
        let spliced = splice_external_tileset(
            &mut main_tileset.root,
            base_url,
            "https://example.com/tiles/0-0-0.json",
            &external,
            external_base_url,
        );

        assert!(spliced);
        // The child should now have the external root's content (resolved to absolute).
        let child = &main_tileset.root.children[0];
        assert_eq!(child.content.as_ref().unwrap().uri, "https://example.com/tiles/tile.b3dm");
        assert_eq!(child.geometric_error, 50.0);
        assert_eq!(child.refine, Some(Refine::Replace));
        // And the external root's children (also resolved).
        assert_eq!(child.children.len(), 1);
        assert_eq!(
            child.children[0].content.as_ref().unwrap().uri,
            "https://example.com/tiles/child0.b3dm"
        );
    }

    #[test]
    fn test_splice_composes_transforms() {
        // Parent tile has a transform, external root also has a transform.
        // Both should be composed (parent * external).
        let main_json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 500.0,
            "root": {
                "boundingVolume": { "sphere": [0.0, 0.0, 0.0, 10000.0] },
                "geometricError": 200.0,
                "refine": "ADD",
                "children": [
                    {
                        "boundingVolume": { "sphere": [0.0, 0.0, 0.0, 5000.0] },
                        "geometricError": 100.0,
                        "content": { "uri": "sub.json" },
                        "transform": [
                            2.0, 0.0, 0.0, 0.0,
                            0.0, 2.0, 0.0, 0.0,
                            0.0, 0.0, 2.0, 0.0,
                            100.0, 200.0, 300.0, 1.0
                        ]
                    }
                ]
            }
        }"#;
        let external_json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 100.0,
            "root": {
                "boundingVolume": { "sphere": [0.0, 0.0, 0.0, 5000.0] },
                "geometricError": 50.0,
                "refine": "REPLACE",
                "content": { "uri": "tile.b3dm" },
                "transform": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    10.0, 20.0, 30.0, 1.0
                ]
            }
        }"#;

        let mut main_tileset = parse_tileset(main_json.as_bytes()).unwrap();
        let external = parse_tileset(external_json.as_bytes()).unwrap();
        let base_url = "https://example.com/";
        let external_base_url = "https://example.com/";

        let spliced = splice_external_tileset(
            &mut main_tileset.root,
            base_url,
            "https://example.com/sub.json",
            &external,
            external_base_url,
        );

        assert!(spliced);
        let child = &main_tileset.root.children[0];
        let t = child.transform.expect("transform should be composed");

        // Expected: parent_scale(2) * ext_translation(10,20,30) + parent_translation(100,200,300)
        // Column 3 (translation) of (parent * ext):
        //   parent * [10, 20, 30, 1] = [2*10+100, 2*20+200, 2*30+300, 1] = [120, 240, 360, 1]
        let composed = glam::DMat4::from_cols_array(&t);
        let col3 = composed.col(3);
        assert!((col3.x - 120.0).abs() < 1e-6, "translation x: {}", col3.x);
        assert!((col3.y - 240.0).abs() < 1e-6, "translation y: {}", col3.y);
        assert!((col3.z - 360.0).abs() < 1e-6, "translation z: {}", col3.z);

        // Scale columns should be 2x (from parent).
        let col0 = composed.col(0);
        assert!((col0.x - 2.0).abs() < 1e-6, "scale x: {}", col0.x);
    }

    #[test]
    fn test_parse_implicit_tiling() {
        let json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 500.0,
            "root": {
                "boundingVolume": { "region": [-1.3, 0.6, -1.2, 0.7, 0.0, 500.0] },
                "geometricError": 250.0,
                "refine": "REPLACE",
                "implicitTiling": {
                    "subdivisionScheme": "QUADTREE",
                    "subtreeLevels": 7,
                    "availableLevels": 21,
                    "subtrees": { "uri": "subtrees/{level}/{x}/{y}.subtree" },
                    "content": { "uri": "content/{level}/{x}/{y}.glb" }
                }
            }
        }"#;
        let tileset = parse_tileset(json.as_bytes()).unwrap();
        let it = tileset.root.implicit_tiling.as_ref().unwrap();
        assert_eq!(it.subdivision_scheme, SubdivisionScheme::Quadtree);
        assert_eq!(it.subtree_levels, 7);
        assert_eq!(it.available_levels, 21);
        assert!(it.content.is_some());
    }
}
