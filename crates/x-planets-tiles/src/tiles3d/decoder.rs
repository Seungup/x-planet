//! 3D Tiles content decoder.
//!
//! Auto-detects the tile format (B3DM or GLB) from magic bytes
//! and extracts mesh data suitable for GPU upload.

use super::b3dm::{is_b3dm, parse_b3dm};
use super::gltf_mesh::{extract_meshes_from_glb, is_glb, ExtractedMesh, GltfExtractError};

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

// ═══════════════════════════════════════════════════════════════════
// Decoder
// ═══════════════════════════════════════════════════════════════════

/// Decode a 3D tile from raw bytes, auto-detecting the format.
///
/// Supports:
/// - **B3DM** (magic: `b3dm`) — extracts embedded GLB + RTC_CENTER
/// - **GLB** (magic: `glTF`) — direct glTF Binary
///
/// Returns `Decoded3dTile` with extracted meshes.
pub fn decode_3d_tile(
    data: &[u8],
    content_uri: &str,
) -> Result<Decoded3dTile, Tiles3dDecodeError> {
    if data.len() < 4 {
        return Err(Tiles3dDecodeError::TooShort);
    }

    let meshes = if is_b3dm(data) {
        // B3DM: parse container, extract GLB, pass RTC_CENTER.
        let b3dm = parse_b3dm(data)?;
        let rtc_center = b3dm.rtc_center();
        extract_meshes_from_glb(&b3dm.glb, rtc_center)?
    } else if is_glb(data) {
        // GLB: direct extraction.
        extract_meshes_from_glb(data, None)?
    } else {
        return Err(Tiles3dDecodeError::UnknownFormat(
            data[0], data[1], data[2], data[3],
        ));
    };

    Ok(Decoded3dTile {
        meshes,
        content_uri: content_uri.to_string(),
    })
}

/// Check if data is a recognized 3D tile format (B3DM or GLB).
pub fn is_3d_tile(data: &[u8]) -> bool {
    is_b3dm(data) || is_glb(data)
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
        assert!(!is_3d_tile(b"PNG\x89\x50\x4e\x47"));
        assert!(!is_3d_tile(b"abc"));
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
}
