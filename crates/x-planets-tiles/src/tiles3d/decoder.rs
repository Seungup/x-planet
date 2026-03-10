//! 3D Tiles content decoder.
//!
//! Auto-detects the tile format (B3DM, GLB, or external tileset JSON)
//! from magic bytes / content and extracts mesh data or sub-tileset.

use super::b3dm::{is_b3dm, parse_b3dm};
use super::gltf_mesh::{extract_meshes_from_glb, is_glb, ExtractedMesh, GltfExtractError};
use super::tileset::Tileset;

use thiserror::Error;

// ═══════════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════════

/// Errors from 3D tile content decoding.
#[derive(Debug, Error)]
pub enum Tiles3dDecodeError {
    #[error("B3DM parse error: {0}")]
    B3dm(#[from] super::b3dm::B3dmError),

    #[error("glTF extraction error: {0}")]
    Gltf(#[from] GltfExtractError),

    #[error("external tileset JSON parse error: {0}")]
    TilesetJson(#[from] serde_json::Error),

    #[error("unknown tile format (magic bytes: {0:02x} {1:02x} {2:02x} {3:02x})")]
    UnknownFormat(u8, u8, u8, u8),

    #[error("tile data too short to identify format")]
    TooShort,
}

// ═══════════════════════════════════════════════════════════════════
// Decoded tile
// ═══════════════════════════════════════════════════════════════════

/// A decoded 3D tile ready for GPU upload.
#[derive(Debug)]
pub struct Decoded3dTile {
    /// Extracted meshes from the tile content.
    pub meshes: Vec<ExtractedMesh>,
    /// The content URI this tile was loaded from.
    pub content_uri: String,
}

/// Result of decoding 3D tile content: either mesh data or a sub-tileset.
#[derive(Debug)]
pub enum Tiles3dContent {
    /// B3DM or GLB mesh data ready for GPU upload.
    Mesh(Decoded3dTile),
    /// External tileset JSON that needs to be spliced into the tile tree.
    ExternalTileset {
        tileset: Tileset,
        /// Base URL for resolving relative URIs within this sub-tileset.
        base_url: String,
        /// The content URI that referenced this external tileset.
        content_uri: String,
    },
}

// ═══════════════════════════════════════════════════════════════════
// Decoder
// ═══════════════════════════════════════════════════════════════════

/// Decode 3D tile content from raw bytes, auto-detecting the format.
///
/// Supports:
/// - **B3DM** (magic: `b3dm`) — extracts embedded GLB + RTC_CENTER
/// - **GLB** (magic: `glTF`) — direct glTF Binary
/// - **JSON** — external tileset reference (sub-tileset)
///
/// Returns `Tiles3dContent` — either mesh data or an external tileset.
pub fn decode_3d_tile(
    data: &[u8],
    content_uri: &str,
) -> Result<Tiles3dContent, Tiles3dDecodeError> {
    if data.len() < 4 {
        return Err(Tiles3dDecodeError::TooShort);
    }

    if is_b3dm(data) {
        let b3dm = parse_b3dm(data)?;
        let rtc_center = b3dm.rtc_center();
        let meshes = extract_meshes_from_glb(&b3dm.glb, rtc_center)?;
        return Ok(Tiles3dContent::Mesh(Decoded3dTile {
            meshes,
            content_uri: content_uri.to_string(),
        }));
    }

    if is_glb(data) {
        let meshes = extract_meshes_from_glb(data, None)?;
        return Ok(Tiles3dContent::Mesh(Decoded3dTile {
            meshes,
            content_uri: content_uri.to_string(),
        }));
    }

    // Check if this is JSON (external tileset reference).
    if is_json(data) {
        let tileset: Tileset = serde_json::from_slice(data)?;
        // Derive base URL from the content URI (everything up to and including last '/').
        let base_url = content_uri
            .rfind('/')
            .map(|i| &content_uri[..=i])
            .unwrap_or(content_uri)
            .to_string();
        return Ok(Tiles3dContent::ExternalTileset {
            tileset,
            base_url,
            content_uri: content_uri.to_string(),
        });
    }

    Err(Tiles3dDecodeError::UnknownFormat(
        data[0], data[1], data[2], data[3],
    ))
}

/// Check if data is a recognized 3D tile format (B3DM, GLB, or JSON tileset).
pub fn is_3d_tile(data: &[u8]) -> bool {
    is_b3dm(data) || is_glb(data) || is_json(data)
}

/// Check if data looks like JSON (starts with `{` after optional whitespace).
fn is_json(data: &[u8]) -> bool {
    data.iter()
        .find(|&&b| !b.is_ascii_whitespace())
        .map_or(false, |&b| b == b'{')
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_3d_tile() {
        assert!(is_3d_tile(b"b3dm\x01\x00\x00\x00"));
        assert!(is_3d_tile(b"glTF\x02\x00\x00\x00"));
        assert!(!is_3d_tile(b"\x89PNG\x50\x4e\x47\x0a"));
        assert!(!is_3d_tile(b"abc"));
    }

    #[test]
    fn test_is_3d_tile_json() {
        assert!(is_3d_tile(b"{\"asset\":{\"version\":\"1.1\"}}"));
        assert!(is_3d_tile(b"  \n{\"asset\":{}}"));
    }

    #[test]
    fn test_decode_too_short() {
        let err = decode_3d_tile(b"ab", "test.glb").unwrap_err();
        assert!(matches!(err, Tiles3dDecodeError::TooShort));
    }

    #[test]
    fn test_decode_unknown_format() {
        let err = decode_3d_tile(b"\x89PNG\r\n\x1a\n", "test.png").unwrap_err();
        assert!(matches!(err, Tiles3dDecodeError::UnknownFormat(..)));
    }

    #[test]
    fn test_decode_external_tileset_json() {
        let json = br#"{
            "asset": { "version": "1.1" },
            "geometricError": 100.0,
            "root": {
                "boundingVolume": { "sphere": [0.0, 0.0, 0.0, 1000.0] },
                "geometricError": 50.0,
                "refine": "ADD",
                "content": { "uri": "tile.b3dm" }
            }
        }"#;
        let result = decode_3d_tile(json, "https://example.com/tiles/0-0-0.json").unwrap();
        match result {
            Tiles3dContent::ExternalTileset { tileset, base_url, content_uri } => {
                assert_eq!(tileset.asset.version, "1.1");
                assert_eq!(base_url, "https://example.com/tiles/");
                assert_eq!(content_uri, "https://example.com/tiles/0-0-0.json");
            }
            Tiles3dContent::Mesh(_) => panic!("expected ExternalTileset"),
        }
    }

    #[test]
    fn test_is_json() {
        assert!(is_json(b"{\"hello\": true}"));
        assert!(is_json(b"  \t\n{\"hello\": true}"));
        assert!(!is_json(b"b3dm"));
        assert!(!is_json(b""));
        assert!(!is_json(b"   "));
    }
}
