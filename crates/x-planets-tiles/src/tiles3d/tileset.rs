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

/// Get the effective refine strategy for a tile (inherits from parent if not set).
pub fn effective_refine(tile: &Tile, parent_refine: Refine) -> Refine {
    tile.refine.unwrap_or(parent_refine)
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
