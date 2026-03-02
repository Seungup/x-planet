//! Quantized Mesh 1.0 terrain tile parser.
//!
//! Quantized Mesh is a binary terrain format developed by Cesium.
//! Unlike heightmap formats (Terrain RGB / Terrarium), QM tiles contain
//! **pre-built triangle meshes** with adaptive vertex density —
//! more vertices where terrain is rough, fewer on flat areas.
//!
//! Spec: <https://github.com/CesiumGS/quantized-mesh>
//!
//! ## Binary layout
//! ```text
//! Header          88 bytes (doubles + floats, LE)
//! VertexCount     u32
//! u[]             VertexCount × u16  (zigzag-delta encoded, 0-32767 = west-east)
//! v[]             VertexCount × u16  (zigzag-delta encoded, 0-32767 = south-north)
//! height[]        VertexCount × u16  (zigzag-delta encoded, 0-32767 = min_h-max_h)
//! [align to 2 or 4 bytes]
//! TriangleCount   u32
//! indices[]       TriangleCount×3 × u16 or u32  (HWM encoded)
//! westCount       u32 + indices[]   (raw, not HWM)
//! southCount      u32 + indices[]
//! eastCount       u32 + indices[]
//! northCount      u32 + indices[]
//! [extensions…]
//! ```

use x_planets_math::TileCoord;

use crate::decoder::DecodeError;

// ═══════════════════════════════════════════════════════════════════
// Data types
// ═══════════════════════════════════════════════════════════════════

/// Header from the Quantized Mesh binary (88 bytes).
///
/// Center XYZ and bounding sphere are in ECEF coordinates.
/// Heights are in meters above the WGS84 ellipsoid.
#[derive(Debug, Clone)]
pub struct QmHeader {
    pub center_x: f64,
    pub center_y: f64,
    pub center_z: f64,
    pub min_height: f32,
    pub max_height: f32,
    pub bounding_sphere_radius: f64,
    pub horizon_occlusion_point_x: f64,
    pub horizon_occlusion_point_y: f64,
    pub horizon_occlusion_point_z: f64,
}

/// Decoded Quantized Mesh tile.
///
/// Vertex coordinates are stored in their raw quantized form (u16 0-32767).
/// The conversion to normalised 0-1 range and then to world coordinates
/// is deferred to `build_terrain_mesh_from_qm` in `x-planets-core`.
#[derive(Debug, Clone)]
pub struct DecodedQuantizedMesh {
    /// Tile coordinate this data belongs to.
    pub coord: TileCoord,
    /// Parsed header metadata.
    pub header: QmHeader,
    /// Quantized u values (0-32767, west → east).
    pub u: Vec<u16>,
    /// Quantized v values (0-32767, south → north).
    pub v: Vec<u16>,
    /// Quantized height values (0-32767, min_height → max_height).
    pub height: Vec<u16>,
    /// Triangle indices (already HWM-decoded, three indices per triangle).
    pub indices: Vec<u32>,
    /// Edge indices for skirt generation (west edge vertex indices).
    pub west_indices: Vec<u32>,
    /// Edge indices for skirt generation (south edge vertex indices).
    pub south_indices: Vec<u32>,
    /// Edge indices for skirt generation (east edge vertex indices).
    pub east_indices: Vec<u32>,
    /// Edge indices for skirt generation (north edge vertex indices).
    pub north_indices: Vec<u32>,
    /// Oct-encoded per-vertex normals, 2 bytes each, if extension present.
    pub oct_normals: Option<Vec<[u8; 2]>>,
}

// ═══════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════

/// Parse a Quantized Mesh 1.0 binary blob.
///
/// `data` may have been transparently decompressed by reqwest (gzip).
/// Returns a structured [`DecodedQuantizedMesh`] or a [`DecodeError`].
pub fn parse_quantized_mesh(
    coord: TileCoord,
    data: &[u8],
) -> Result<DecodedQuantizedMesh, DecodeError> {
    let mut cur = 0usize;

    // ── Header (88 bytes) ──────────────────────────────────────────
    if data.len() < 88 {
        return Err(DecodeError::InvalidData(format!(
            "QM tile too short for header: {} bytes (need 88)",
            data.len()
        )));
    }

    let center_x             = read_f64(data, &mut cur);
    let center_y             = read_f64(data, &mut cur);
    let center_z             = read_f64(data, &mut cur);
    let min_height           = read_f32(data, &mut cur);
    let max_height           = read_f32(data, &mut cur);
    let _bsph_cx             = read_f64(data, &mut cur); // bounding sphere center (unused)
    let _bsph_cy             = read_f64(data, &mut cur);
    let _bsph_cz             = read_f64(data, &mut cur);
    let bounding_sphere_radius = read_f64(data, &mut cur);
    let horizon_occlusion_point_x = read_f64(data, &mut cur);
    let horizon_occlusion_point_y = read_f64(data, &mut cur);
    let horizon_occlusion_point_z = read_f64(data, &mut cur);

    debug_assert_eq!(cur, 88);

    let header = QmHeader {
        center_x,
        center_y,
        center_z,
        min_height,
        max_height,
        bounding_sphere_radius,
        horizon_occlusion_point_x,
        horizon_occlusion_point_y,
        horizon_occlusion_point_z,
    };

    // ── Vertex count ───────────────────────────────────────────────
    if cur + 4 > data.len() {
        return Err(DecodeError::InvalidData("QM: truncated at vertexCount".into()));
    }
    let vertex_count = read_u32(data, &mut cur) as usize;

    // ── Three quantised vertex arrays (u, v, height) ───────────────
    // Each is `vertex_count` u16 values, zigzag-delta encoded.
    let u_enc      = read_u16_slice(data, &mut cur, vertex_count)?;
    let v_enc      = read_u16_slice(data, &mut cur, vertex_count)?;
    let height_enc = read_u16_slice(data, &mut cur, vertex_count)?;

    let u      = zigzag_delta_decode(&u_enc);
    let v      = zigzag_delta_decode(&v_enc);
    let height = zigzag_delta_decode(&height_enc);

    // ── Index alignment ────────────────────────────────────────────
    // Indices are stored as u32 if vertex_count ≥ 65536, else u16.
    // The spec requires 2-byte alignment for u16 indices,
    // 4-byte alignment for u32 indices.
    let use_u32_indices = vertex_count >= 65536;
    let align = if use_u32_indices { 4 } else { 2 };
    if cur % align != 0 {
        cur += align - (cur % align);
    }

    // ── Triangle count + indices ───────────────────────────────────
    if cur + 4 > data.len() {
        return Err(DecodeError::InvalidData("QM: truncated at triangleCount".into()));
    }
    let triangle_count = read_u32(data, &mut cur) as usize;
    let index_count = triangle_count * 3;

    let raw_indices: Vec<u32> = if use_u32_indices {
        read_u32_indices(data, &mut cur, index_count)?
    } else {
        read_u16_indices_as_u32(data, &mut cur, index_count)?
    };
    let indices = decode_hwm(&raw_indices);

    // ── Edge indices (raw, not HWM) ────────────────────────────────
    let west_indices  = read_edge_indices(data, &mut cur, use_u32_indices)?;
    let south_indices = read_edge_indices(data, &mut cur, use_u32_indices)?;
    let east_indices  = read_edge_indices(data, &mut cur, use_u32_indices)?;
    let north_indices = read_edge_indices(data, &mut cur, use_u32_indices)?;

    // ── Extensions ────────────────────────────────────────────────
    let oct_normals = parse_extensions(data, &mut cur, vertex_count);

    Ok(DecodedQuantizedMesh {
        coord,
        header,
        u,
        v,
        height,
        indices,
        west_indices,
        south_indices,
        east_indices,
        north_indices,
        oct_normals,
    })
}

/// Decode a pair of oct-encoded normal bytes to a unit vector.
///
/// Each normal is stored as 2 bytes (x, y) in oct-encoded form.
/// Z is derived: `z = 1 − |x| − |y|`, then the result is normalised.
///
/// Algorithm from "A Survey of Efficient Representations for Independent
/// Unit Vectors" (Cigolle et al. 2014).
pub fn decode_oct_normal(x_byte: u8, y_byte: u8) -> [f32; 3] {
    // Map [0, 255] → [−1, 1]
    let x = (x_byte as f32 / 255.0) * 2.0 - 1.0;
    let y = (y_byte as f32 / 255.0) * 2.0 - 1.0;
    let z = 1.0 - x.abs() - y.abs();

    // Reflect the lower octahedron hemisphere
    let (x, y) = if z < 0.0 {
        let sx = if x >= 0.0 { 1.0 } else { -1.0 };
        let sy = if y >= 0.0 { 1.0 } else { -1.0 };
        ((1.0 - y.abs()) * sx, (1.0 - x.abs()) * sy)
    } else {
        (x, y)
    };

    let len = (x * x + y * y + z * z).sqrt().max(1e-10);
    [x / len, y / len, z / len]
}

// ═══════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════

// ── Primitive readers ──────────────────────────────────────────────

#[inline]
fn read_f64(data: &[u8], cur: &mut usize) -> f64 {
    let v = f64::from_le_bytes(data[*cur..*cur + 8].try_into().unwrap());
    *cur += 8;
    v
}

#[inline]
fn read_f32(data: &[u8], cur: &mut usize) -> f32 {
    let v = f32::from_le_bytes(data[*cur..*cur + 4].try_into().unwrap());
    *cur += 4;
    v
}

#[inline]
fn read_u32(data: &[u8], cur: &mut usize) -> u32 {
    let v = u32::from_le_bytes(data[*cur..*cur + 4].try_into().unwrap());
    *cur += 4;
    v
}

// ── Slice readers ──────────────────────────────────────────────────

fn read_u16_slice(
    data: &[u8],
    cur: &mut usize,
    count: usize,
) -> Result<Vec<u16>, DecodeError> {
    let byte_len = count * 2;
    if *cur + byte_len > data.len() {
        return Err(DecodeError::InvalidData(format!(
            "QM: truncated reading {} u16 values at offset {}",
            count, *cur
        )));
    }
    let slice = &data[*cur..*cur + byte_len];
    *cur += byte_len;
    Ok(slice
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect())
}

fn read_u16_indices_as_u32(
    data: &[u8],
    cur: &mut usize,
    count: usize,
) -> Result<Vec<u32>, DecodeError> {
    let byte_len = count * 2;
    if *cur + byte_len > data.len() {
        return Err(DecodeError::InvalidData(format!(
            "QM: truncated reading {} u16 indices at offset {}",
            count, *cur
        )));
    }
    let slice = &data[*cur..*cur + byte_len];
    *cur += byte_len;
    Ok(slice
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]) as u32)
        .collect())
}

fn read_u32_indices(
    data: &[u8],
    cur: &mut usize,
    count: usize,
) -> Result<Vec<u32>, DecodeError> {
    let byte_len = count * 4;
    if *cur + byte_len > data.len() {
        return Err(DecodeError::InvalidData(format!(
            "QM: truncated reading {} u32 indices at offset {}",
            count, *cur
        )));
    }
    let slice = &data[*cur..*cur + byte_len];
    *cur += byte_len;
    Ok(slice
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn read_edge_indices(
    data: &[u8],
    cur: &mut usize,
    use_u32: bool,
) -> Result<Vec<u32>, DecodeError> {
    if *cur + 4 > data.len() {
        return Err(DecodeError::InvalidData(
            "QM: truncated at edge index count".into(),
        ));
    }
    let count = read_u32(data, cur) as usize;
    if use_u32 {
        read_u32_indices(data, cur, count)
    } else {
        read_u16_indices_as_u32(data, cur, count)
    }
}

// ── Encoding algorithms ────────────────────────────────────────────

/// Decode a zigzag-delta-encoded u16 array.
///
/// The QM spec stores consecutive deltas as zigzag-encoded unsigned values:
/// - Zigzag maps signed → unsigned: `encode(n) = (n << 1) ^ (n >> 31)`
/// - Decode: `decode(v) = (v >> 1) ^ -(v & 1)`
/// - Then accumulate a running sum to recover the original values.
fn zigzag_delta_decode(encoded: &[u16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(encoded.len());
    let mut acc: i32 = 0;
    for &val in encoded {
        let delta = ((val as i32) >> 1) ^ -((val as i32) & 1);
        acc = acc.wrapping_add(delta);
        // Values must be in [0, 32767]; clamp against wrapping artifacts.
        out.push(acc.clamp(0, 32767) as u16);
    }
    out
}

/// Decode high-water-mark (HWM) encoded triangle indices.
///
/// The HWM encoding stores each index as an offset from a running maximum:
/// - `code == 0`: the index equals the current high-water-mark; advance it.
/// - otherwise: `index = highest - code`.
///
/// This exploits the typical mesh property that triangles tend to reference
/// recently introduced vertices (good compression for strip-like meshes).
fn decode_hwm(encoded: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(encoded.len());
    let mut highest: u32 = 0;
    for &code in encoded {
        let index = highest.wrapping_sub(code);
        out.push(index);
        if code == 0 {
            highest += 1;
        }
    }
    out
}

/// Parse optional extensions after the edge indices.
///
/// Returns oct-encoded normals if extension ID 1 is present.
/// Unknown extension IDs are skipped using the stored length field.
fn parse_extensions(data: &[u8], cur: &mut usize, vertex_count: usize) -> Option<Vec<[u8; 2]>> {
    let mut oct_normals: Option<Vec<[u8; 2]>> = None;

    while *cur + 5 <= data.len() {
        // Extension header: 1 byte ID + 4 bytes length
        let ext_id = data[*cur];
        *cur += 1;
        let ext_len = u32::from_le_bytes(
            data[*cur..*cur + 4].try_into().unwrap_or([0; 4]),
        ) as usize;
        *cur += 4;

        if *cur + ext_len > data.len() {
            break; // Corrupt / truncated extension — stop parsing
        }

        match ext_id {
            1 => {
                // Oct-encoded per-vertex normals: 2 bytes per vertex
                if ext_len == vertex_count * 2 {
                    let normals: Vec<[u8; 2]> = data[*cur..*cur + ext_len]
                        .chunks_exact(2)
                        .map(|b| [b[0], b[1]])
                        .collect();
                    oct_normals = Some(normals);
                }
            }
            _ => {} // Water mask (2), metadata (4), etc. — skip
        }

        *cur += ext_len;
    }

    oct_normals
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── zigzag_delta_decode ────────────────────────────────────────

    #[test]
    fn test_zigzag_delta_decode_zero_stays_zero() {
        // Encoded 0 → delta 0 → cumulative 0
        assert_eq!(zigzag_delta_decode(&[0, 0, 0]), vec![0, 0, 0]);
    }

    #[test]
    fn test_zigzag_delta_decode_constant_increase() {
        // Encoded 2 → delta +1 each step → [1, 2, 3]
        assert_eq!(zigzag_delta_decode(&[2, 2, 2]), vec![1, 2, 3]);
    }

    #[test]
    fn test_zigzag_delta_decode_known_values() {
        // Encoded sequence that decodes to [0, 100, 200]
        // Delta: 0 (+0), 200 (+100), 200 (+100)  →  zigzag: 0, 200, 200
        let encoded: Vec<u16> = vec![0, 200, 200];
        let decoded = zigzag_delta_decode(&encoded);
        assert_eq!(decoded, vec![0, 100, 200]);
    }

    // ── decode_hwm ────────────────────────────────────────────────

    #[test]
    fn test_decode_hwm_ascending() {
        // All zeros → indices 0, 1, 2, 3 (HWM advances each time)
        assert_eq!(decode_hwm(&[0, 0, 0, 0]), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_decode_hwm_back_reference() {
        // 0, 0, 1 → [0, 1, 1]   (last is highest(2) - 1 = 1)
        assert_eq!(decode_hwm(&[0, 0, 1]), vec![0, 1, 1]);
    }

    // ── decode_oct_normal ─────────────────────────────────────────

    #[test]
    fn test_oct_normal_up() {
        // (128, 128) ≈ centre of octahedron → should be close to [0, 0, 1]
        let n = decode_oct_normal(128, 128);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 0.02, "not unit: len={len}");
        assert!(n[2] > 0.9, "z should be near 1, got {}", n[2]);
    }

    #[test]
    fn test_oct_normal_unit_length() {
        for x in [0u8, 64, 128, 192, 255] {
            for y in [0u8, 64, 128, 192, 255] {
                let n = decode_oct_normal(x, y);
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                assert!((len - 1.0).abs() < 0.02, "({x},{y}): len={len}");
            }
        }
    }

    // ── parse_quantized_mesh ───────────────────────────────────────

    #[test]
    fn test_parse_too_short() {
        let coord = TileCoord::new(5, 0, 0);
        let err = parse_quantized_mesh(coord, &[0u8; 10]).unwrap_err();
        assert!(
            matches!(err, DecodeError::InvalidData(_)),
            "expected InvalidData, got {err:?}"
        );
    }

    #[test]
    fn test_parse_minimal_tile() {
        // Build a minimal valid QM binary:
        //   88-byte header + 4-byte vertexCount=3 +
        //   3 u16 arrays (3×2 = 6 bytes each) +
        //   padding (cur=88+4+18=110, align to 2 → none needed) +
        //   triangleCount=1 + 3 u16 indices (0,0,0 HWM → triangle 0,1,2) +
        //   4 edge lists (each count=0)
        let mut data: Vec<u8> = vec![0u8; 88];

        // header: center xyz f64 (0,0,0), min/max height f32 (0,100)
        // Just zeros for center, write min=0, max=100 at bytes 24-32
        data[24..28].copy_from_slice(&0.0f32.to_le_bytes());     // min_height
        data[28..32].copy_from_slice(&100.0f32.to_le_bytes());   // max_height
        // Bounding sphere radius at byte 56: 1.0
        data[56..64].copy_from_slice(&1.0f64.to_le_bytes());

        // vertexCount = 3
        data.extend_from_slice(&3u32.to_le_bytes());

        // u, v, height arrays: 3 × u16 each, all zero-encoded
        // zigzag(delta 0) = 0
        data.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // u
        data.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // v
        data.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // height

        // triangleCount = 1
        data.extend_from_slice(&1u32.to_le_bytes());
        // 3 u16 indices: 0,0,0 → HWM decodes to [0, 1, 2] (0→0 advance, 0→1 advance, 0→2 advance)
        data.extend_from_slice(&[0, 0, 0, 0, 0, 0]);

        // 4 empty edge lists
        for _ in 0..4 {
            data.extend_from_slice(&0u32.to_le_bytes());
        }

        let coord = TileCoord::new(5, 0, 0);
        let qm = parse_quantized_mesh(coord, &data).expect("parse failed");

        assert_eq!(qm.u.len(), 3);
        assert_eq!(qm.v.len(), 3);
        assert_eq!(qm.height.len(), 3);
        assert_eq!(qm.indices.len(), 3); // 1 triangle × 3 vertices
        assert_eq!(qm.header.max_height, 100.0);
        assert!(qm.oct_normals.is_none());
    }
}
