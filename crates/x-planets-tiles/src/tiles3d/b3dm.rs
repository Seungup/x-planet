//! B3DM (Batched 3D Model) parser.
//!
//! B3DM is a legacy 3D Tiles 1.0 container format wrapping a glTF/GLB payload
//! with optional feature table and batch table metadata.
//!
//! Binary layout:
//! ```text
//! ┌────────────────────────────────────────────────┐
//! │ Header (28 bytes)                              │
//! │   magic:                    "b3dm" (4 bytes)   │
//! │   version:                  u32 LE             │
//! │   byteLength:               u32 LE             │
//! │   featureTableJSONByteLength: u32 LE           │
//! │   featureTableBinaryByteLength: u32 LE         │
//! │   batchTableJSONByteLength: u32 LE             │
//! │   batchTableBinaryByteLength: u32 LE           │
//! ├────────────────────────────────────────────────┤
//! │ Feature Table JSON (padded to 8-byte boundary) │
//! │ Feature Table Binary                           │
//! │ Batch Table JSON                               │
//! │ Batch Table Binary                             │
//! │ GLB payload (rest of data)                     │
//! └────────────────────────────────────────────────┘
//! ```
//!
//! Spec: <https://github.com/CesiumGS/3d-tiles/blob/main/specification/TileFormats/Batched3DModel/README.adoc>

use thiserror::Error;

/// B3DM magic bytes: `b"b3dm"`.
const B3DM_MAGIC: [u8; 4] = *b"b3dm";

/// B3DM header size in bytes.
const HEADER_SIZE: usize = 28;

// ═══════════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════════

/// Errors that can occur when parsing a B3DM file.
#[derive(Debug, Error)]
pub enum B3dmError {
    #[error("data too short for B3DM header (need {HEADER_SIZE} bytes, got {0})")]
    TooShort(usize),

    #[error("invalid magic: expected b3dm, got {0:?}")]
    InvalidMagic([u8; 4]),

    #[error("unsupported B3DM version: {0} (expected 1)")]
    UnsupportedVersion(u32),

    #[error("byte length mismatch: header says {header} but data has {actual} bytes")]
    LengthMismatch { header: u32, actual: usize },

    #[error("table offsets exceed data length")]
    OffsetOverflow,
}

// ═══════════════════════════════════════════════════════════════════
// B3DM Header
// ═══════════════════════════════════════════════════════════════════

/// Parsed B3DM header (28 bytes).
#[derive(Debug, Clone, Copy)]
pub struct B3dmHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub byte_length: u32,
    pub feature_table_json_length: u32,
    pub feature_table_binary_length: u32,
    pub batch_table_json_length: u32,
    pub batch_table_binary_length: u32,
}

impl B3dmHeader {
    fn parse(data: &[u8]) -> Result<Self, B3dmError> {
        if data.len() < HEADER_SIZE {
            return Err(B3dmError::TooShort(data.len()));
        }

        let magic = [data[0], data[1], data[2], data[3]];
        if magic != B3DM_MAGIC {
            return Err(B3dmError::InvalidMagic(magic));
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != 1 {
            return Err(B3dmError::UnsupportedVersion(version));
        }

        let byte_length = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let feature_table_json_length =
            u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let feature_table_binary_length =
            u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let batch_table_json_length = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
        let batch_table_binary_length =
            u32::from_le_bytes([data[24], data[25], data[26], data[27]]);

        Ok(Self {
            magic,
            version,
            byte_length,
            feature_table_json_length,
            feature_table_binary_length,
            batch_table_json_length,
            batch_table_binary_length,
        })
    }

    /// Total size of feature + batch tables.
    fn tables_size(&self) -> usize {
        (self.feature_table_json_length
            + self.feature_table_binary_length
            + self.batch_table_json_length
            + self.batch_table_binary_length) as usize
    }
}

// ═══════════════════════════════════════════════════════════════════
// B3DM container
// ═══════════════════════════════════════════════════════════════════

/// A parsed B3DM container.
#[derive(Debug)]
pub struct B3dm {
    pub header: B3dmHeader,
    /// Feature table JSON (UTF-8, may be empty).
    pub feature_table_json: Vec<u8>,
    /// Feature table binary payload.
    pub feature_table_binary: Vec<u8>,
    /// Batch table JSON (UTF-8, may be empty).
    pub batch_table_json: Vec<u8>,
    /// Batch table binary payload.
    pub batch_table_binary: Vec<u8>,
    /// The embedded GLB (glTF Binary) payload.
    pub glb: Vec<u8>,
}

impl B3dm {
    /// Get the RTC_CENTER from the feature table JSON, if present.
    ///
    /// Many B3DM files use the `CESIUM_RTC` extension or embed `RTC_CENTER`
    /// in the feature table to store a relative-to-center offset.
    pub fn rtc_center(&self) -> Option<[f64; 3]> {
        if self.feature_table_json.is_empty() {
            return None;
        }
        let json: serde_json::Value =
            serde_json::from_slice(&self.feature_table_json).ok()?;
        let rtc_val = json.get("RTC_CENTER")?;

        // Format A: inline JSON array — "RTC_CENTER": [x, y, z]
        if let Some(arr) = rtc_val.as_array() {
            if arr.len() != 3 {
                return None;
            }
            return Some([
                arr[0].as_f64()?,
                arr[1].as_f64()?,
                arr[2].as_f64()?,
            ]);
        }

        // Format B: byteOffset into feature table binary —
        // "RTC_CENTER": {"byteOffset": N}
        // Values are 3 × f64 little-endian (24 bytes) in feature_table_binary.
        if let Some(obj) = rtc_val.as_object() {
            let offset = obj.get("byteOffset")?.as_u64()? as usize;
            let end = offset.checked_add(24)?;
            if end > self.feature_table_binary.len() {
                return None;
            }
            let bin = &self.feature_table_binary[offset..end];
            return Some([
                f64::from_le_bytes(bin[0..8].try_into().ok()?),
                f64::from_le_bytes(bin[8..16].try_into().ok()?),
                f64::from_le_bytes(bin[16..24].try_into().ok()?),
            ]);
        }

        None
    }
}

/// Parse a B3DM file from raw bytes.
pub fn parse_b3dm(data: &[u8]) -> Result<B3dm, B3dmError> {
    let header = B3dmHeader::parse(data)?;

    // Validate byte length.
    if (header.byte_length as usize) != data.len() {
        return Err(B3dmError::LengthMismatch {
            header: header.byte_length,
            actual: data.len(),
        });
    }

    // Calculate offsets.
    let mut offset = HEADER_SIZE;
    let tables_end = offset
        .checked_add(header.tables_size())
        .ok_or(B3dmError::OffsetOverflow)?;

    if tables_end > data.len() {
        return Err(B3dmError::OffsetOverflow);
    }

    // Feature table JSON.
    let ft_json_end = offset + header.feature_table_json_length as usize;
    let feature_table_json = data[offset..ft_json_end].to_vec();
    offset = ft_json_end;

    // Feature table binary.
    let ft_bin_end = offset + header.feature_table_binary_length as usize;
    let feature_table_binary = data[offset..ft_bin_end].to_vec();
    offset = ft_bin_end;

    // Batch table JSON.
    let bt_json_end = offset + header.batch_table_json_length as usize;
    let batch_table_json = data[offset..bt_json_end].to_vec();
    offset = bt_json_end;

    // Batch table binary.
    let bt_bin_end = offset + header.batch_table_binary_length as usize;
    let batch_table_binary = data[offset..bt_bin_end].to_vec();
    offset = bt_bin_end;

    // Remaining data is the GLB payload.
    let glb = data[offset..].to_vec();

    Ok(B3dm {
        header,
        feature_table_json,
        feature_table_binary,
        batch_table_json,
        batch_table_binary,
        glb,
    })
}

/// Check if data begins with the B3DM magic bytes.
pub fn is_b3dm(data: &[u8]) -> bool {
    data.len() >= 4 && data[..4] == B3DM_MAGIC
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid B3DM with a fake GLB payload.
    fn make_test_b3dm(
        feature_json: &[u8],
        feature_bin: &[u8],
        batch_json: &[u8],
        batch_bin: &[u8],
        glb: &[u8],
    ) -> Vec<u8> {
        let total = HEADER_SIZE
            + feature_json.len()
            + feature_bin.len()
            + batch_json.len()
            + batch_bin.len()
            + glb.len();

        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(b"b3dm"); // magic
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&(total as u32).to_le_bytes()); // byteLength
        buf.extend_from_slice(&(feature_json.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(feature_bin.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(batch_json.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(batch_bin.len() as u32).to_le_bytes());
        buf.extend_from_slice(feature_json);
        buf.extend_from_slice(feature_bin);
        buf.extend_from_slice(batch_json);
        buf.extend_from_slice(batch_bin);
        buf.extend_from_slice(glb);
        buf
    }

    #[test]
    fn test_parse_minimal_b3dm() {
        let glb = b"fake_glb_data";
        let data = make_test_b3dm(b"", b"", b"", b"", glb);
        let b3dm = parse_b3dm(&data).unwrap();

        assert_eq!(b3dm.header.version, 1);
        assert_eq!(b3dm.header.byte_length as usize, data.len());
        assert!(b3dm.feature_table_json.is_empty());
        assert!(b3dm.feature_table_binary.is_empty());
        assert!(b3dm.batch_table_json.is_empty());
        assert!(b3dm.batch_table_binary.is_empty());
        assert_eq!(b3dm.glb, b"fake_glb_data");
    }

    #[test]
    fn test_parse_with_feature_table() {
        let ft_json = br#"{"BATCH_LENGTH":10,"RTC_CENTER":[1.0,2.0,3.0]}"#;
        let glb = b"glb";
        let data = make_test_b3dm(ft_json, b"", b"", b"", glb);
        let b3dm = parse_b3dm(&data).unwrap();

        assert_eq!(b3dm.feature_table_json, ft_json);
        assert_eq!(b3dm.glb, b"glb");

        let rtc = b3dm.rtc_center().unwrap();
        assert!((rtc[0] - 1.0).abs() < 1e-10);
        assert!((rtc[1] - 2.0).abs() < 1e-10);
        assert!((rtc[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_with_all_tables() {
        let ft_json = br#"{"BATCH_LENGTH":5}"#;
        let ft_bin = b"\x00\x01\x02\x03";
        let bt_json = br#"{"name":["a","b","c","d","e"]}"#;
        let bt_bin = b"\x10\x20";
        let glb = b"GLB_PAYLOAD";

        let data = make_test_b3dm(ft_json, ft_bin, bt_json, bt_bin, glb);
        let b3dm = parse_b3dm(&data).unwrap();

        assert_eq!(b3dm.feature_table_json, ft_json);
        assert_eq!(b3dm.feature_table_binary, ft_bin);
        assert_eq!(b3dm.batch_table_json, bt_json);
        assert_eq!(b3dm.batch_table_binary, bt_bin);
        assert_eq!(b3dm.glb, b"GLB_PAYLOAD");
    }

    #[test]
    fn test_too_short() {
        let data = b"b3dm";
        let err = parse_b3dm(data).unwrap_err();
        assert!(matches!(err, B3dmError::TooShort(4)));
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = make_test_b3dm(b"", b"", b"", b"", b"glb");
        data[0] = b'x';
        let err = parse_b3dm(&data).unwrap_err();
        assert!(matches!(err, B3dmError::InvalidMagic(_)));
    }

    #[test]
    fn test_wrong_version() {
        let mut data = make_test_b3dm(b"", b"", b"", b"", b"glb");
        // Set version to 2.
        data[4..8].copy_from_slice(&2u32.to_le_bytes());
        let err = parse_b3dm(&data).unwrap_err();
        assert!(matches!(err, B3dmError::UnsupportedVersion(2)));
    }

    #[test]
    fn test_length_mismatch() {
        let mut data = make_test_b3dm(b"", b"", b"", b"", b"glb");
        // Corrupt byte length.
        data[8..12].copy_from_slice(&9999u32.to_le_bytes());
        let err = parse_b3dm(&data).unwrap_err();
        assert!(matches!(err, B3dmError::LengthMismatch { .. }));
    }

    #[test]
    fn test_is_b3dm() {
        assert!(is_b3dm(b"b3dm\x01\x00\x00\x00"));
        assert!(!is_b3dm(b"glTF\x02\x00\x00\x00"));
        assert!(!is_b3dm(b"abc"));
    }

    #[test]
    fn test_rtc_center_none_when_no_feature_table() {
        let data = make_test_b3dm(b"", b"", b"", b"", b"glb");
        let b3dm = parse_b3dm(&data).unwrap();
        assert!(b3dm.rtc_center().is_none());
    }

    #[test]
    fn test_rtc_center_none_when_no_rtc_in_json() {
        let ft_json = br#"{"BATCH_LENGTH":10}"#;
        let data = make_test_b3dm(ft_json, b"", b"", b"", b"glb");
        let b3dm = parse_b3dm(&data).unwrap();
        assert!(b3dm.rtc_center().is_none());
    }

    #[test]
    fn test_rtc_center_from_byte_offset() {
        // Cesium ION CWT tiles store RTC_CENTER as {"byteOffset": N}
        // with 3 × f64 LE values in the feature table binary.
        let ft_json = br#"{"BATCH_LENGTH":0,"RTC_CENTER":{"byteOffset":0}}"#;
        let x: f64 = -3_058_211.5;
        let y: f64 = 4_052_013.25;
        let z: f64 = 3_863_471.75;
        let mut ft_bin = Vec::with_capacity(24);
        ft_bin.extend_from_slice(&x.to_le_bytes());
        ft_bin.extend_from_slice(&y.to_le_bytes());
        ft_bin.extend_from_slice(&z.to_le_bytes());

        let data = make_test_b3dm(ft_json, &ft_bin, b"", b"", b"glb");
        let b3dm = parse_b3dm(&data).unwrap();

        let rtc = b3dm.rtc_center().unwrap();
        assert!((rtc[0] - x).abs() < 1e-10);
        assert!((rtc[1] - y).abs() < 1e-10);
        assert!((rtc[2] - z).abs() < 1e-10);
    }

    #[test]
    fn test_rtc_center_byte_offset_nonzero() {
        // byteOffset can be > 0 when other data precedes RTC_CENTER.
        let ft_json = br#"{"BATCH_LENGTH":0,"RTC_CENTER":{"byteOffset":8}}"#;
        let x: f64 = 100.0;
        let y: f64 = 200.0;
        let z: f64 = 300.0;
        let mut ft_bin = vec![0u8; 8]; // 8 bytes of padding before RTC
        ft_bin.extend_from_slice(&x.to_le_bytes());
        ft_bin.extend_from_slice(&y.to_le_bytes());
        ft_bin.extend_from_slice(&z.to_le_bytes());

        let data = make_test_b3dm(ft_json, &ft_bin, b"", b"", b"glb");
        let b3dm = parse_b3dm(&data).unwrap();

        let rtc = b3dm.rtc_center().unwrap();
        assert!((rtc[0] - 100.0).abs() < 1e-10);
        assert!((rtc[1] - 200.0).abs() < 1e-10);
        assert!((rtc[2] - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_rtc_center_byte_offset_out_of_bounds() {
        // byteOffset + 24 exceeds binary length → graceful None.
        let ft_json = br#"{"BATCH_LENGTH":0,"RTC_CENTER":{"byteOffset":0}}"#;
        let ft_bin = [0u8; 16]; // Only 16 bytes, need 24

        let data = make_test_b3dm(ft_json, &ft_bin, b"", b"", b"glb");
        let b3dm = parse_b3dm(&data).unwrap();
        assert!(b3dm.rtc_center().is_none());
    }
}
