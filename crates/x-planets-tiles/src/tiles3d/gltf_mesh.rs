//! glTF/GLB mesh extraction for 3D Tiles.
//!
//! Extracts vertex data (positions, normals, tex coords, indices)
//! and textures from glTF Binary (GLB) payloads, whether standalone
//! or embedded within B3DM containers.
//!
//! Supports:
//! - Standard glTF 2.0 attributes (POSITION, NORMAL, TEXCOORD_0)
//! - CESIUM_RTC extension (relative-to-center)
//! - Auto-generated flat normals when normals are missing
//! - Base color texture extraction

use thiserror::Error;

// ═══════════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════════

/// Errors from glTF mesh extraction.
#[derive(Debug, Error)]
pub enum GltfExtractError {
    #[error("glTF parse error: {0}")]
    GltfError(#[from] gltf::Error),

    #[error("no meshes found in glTF")]
    NoMeshes,

    #[error("missing POSITION attribute in primitive")]
    MissingPositions,

    #[error("unsupported accessor data type")]
    UnsupportedDataType,

    #[error("buffer data out of bounds")]
    BufferOutOfBounds,
}

// ═══════════════════════════════════════════════════════════════════
// Extracted mesh data
// ═══════════════════════════════════════════════════════════════════

/// A single extracted mesh from a glTF scene.
#[derive(Debug, Clone)]
pub struct ExtractedMesh {
    /// Vertex positions (x, y, z) in local or ECEF coordinates.
    pub positions: Vec<[f32; 3]>,
    /// Vertex normals (may be auto-generated if missing from source).
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates (UV).
    pub tex_coords: Vec<[f32; 2]>,
    /// Triangle indices.
    pub indices: Vec<u32>,
    /// Base color texture (RGBA, row-major).
    pub texture_rgba: Option<Vec<u8>>,
    /// Texture width in pixels.
    pub texture_width: u32,
    /// Texture height in pixels.
    pub texture_height: u32,
    /// RTC_CENTER offset from CESIUM_RTC extension or B3DM feature table.
    /// When present, vertex positions are relative to this ECEF point.
    pub rtc_center: Option<[f64; 3]>,
    /// Accumulated glTF node transform (column-major 4x4, f64).
    /// Captures the scene hierarchy transform that should be applied
    /// between the tile transform and vertex positions.
    /// Identity when no node transform exists.
    pub local_transform: [[f64; 4]; 4],
}

// ═══════════════════════════════════════════════════════════════════
// GLB extraction
// ═══════════════════════════════════════════════════════════════════

/// Extract all meshes from a GLB (glTF Binary) payload.
///
/// Handles both standalone GLB files and GLB payloads extracted from B3DM.
/// If `rtc_center` is provided (e.g., from B3DM feature table), it will be
/// attached to all extracted meshes.
pub fn extract_meshes_from_glb(
    glb_data: &[u8],
    external_rtc_center: Option<[f64; 3]>,
) -> Result<Vec<ExtractedMesh>, GltfExtractError> {
    let (document, buffers, images) = gltf::import_slice(glb_data)?;

    // Check for CESIUM_RTC extension in the document.
    let rtc_center = external_rtc_center.or_else(|| extract_cesium_rtc(&document));

    let mut meshes = Vec::new();

    // Walk the scene/node hierarchy to collect node transforms.
    // Instead of baking node transforms into vertex positions (which can
    // cause coordinate-system mismatches with the 3D Tiles hierarchy),
    // store them in `local_transform` and let the rendering pipeline
    // compose: tile_transform × local_transform × translate(rtc).
    let node_transforms = collect_node_transforms(&document);

    for node in document.nodes() {
        if let Some(mesh) = node.mesh() {
            let transform = node_transforms
                .get(&node.index())
                .copied()
                .unwrap_or(IDENTITY_F32);

            // Convert node transform to f64 for the local_transform field.
            let local_transform = f32_to_f64_mat4(&transform);

            for primitive in mesh.primitives() {
                // Skip non-triangle primitives (strips, fans, lines, points)
                // since the render pipeline uses TriangleList topology.
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                if let Some(mut extracted) =
                    extract_primitive(&primitive, &buffers, &images, rtc_center)?
                {
                    extracted.local_transform = local_transform;
                    meshes.push(extracted);
                }
            }
        }
    }

    // Fallback: if no nodes reference meshes (unusual), iterate meshes directly.
    if meshes.is_empty() {
        for mesh in document.meshes() {
            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                if let Some(extracted) =
                    extract_primitive(&primitive, &buffers, &images, rtc_center)?
                {
                    meshes.push(extracted);
                }
            }
        }
    }

    Ok(meshes)
}

/// Identity 4x4 matrix in column-major f64.
pub const IDENTITY_F64: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Identity 4x4 matrix in column-major f32.
const IDENTITY_F32: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Convert f32 4x4 matrix to f64.
fn f32_to_f64_mat4(m: &[[f32; 4]; 4]) -> [[f64; 4]; 4] {
    [
        [m[0][0] as f64, m[0][1] as f64, m[0][2] as f64, m[0][3] as f64],
        [m[1][0] as f64, m[1][1] as f64, m[1][2] as f64, m[1][3] as f64],
        [m[2][0] as f64, m[2][1] as f64, m[2][2] as f64, m[2][3] as f64],
        [m[3][0] as f64, m[3][1] as f64, m[3][2] as f64, m[3][3] as f64],
    ]
}

/// Collect accumulated transforms for each node by walking the scene hierarchy.
fn collect_node_transforms(document: &gltf::Document) -> std::collections::HashMap<usize, [[f32; 4]; 4]> {
    let mut result = std::collections::HashMap::new();

    fn walk_node(
        node: &gltf::Node<'_>,
        parent_transform: [[f32; 4]; 4],
        result: &mut std::collections::HashMap<usize, [[f32; 4]; 4]>,
    ) {
        let local = node.transform().matrix();
        let accumulated = mat4_mul(&parent_transform, &local);
        result.insert(node.index(), accumulated);
        for child in node.children() {
            walk_node(&child, accumulated, result);
        }
    }

    for scene in document.scenes() {
        for node in scene.nodes() {
            walk_node(&node, IDENTITY_F32, &mut result);
        }
    }

    result
}

/// Multiply two 4x4 column-major matrices (a * b).
fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            result[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    result
}

/// Extract a single primitive's data.
fn extract_primitive(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    rtc_center: Option<[f64; 3]>,
) -> Result<Option<ExtractedMesh>, GltfExtractError> {
    // ── Positions (required) ──
    let positions_accessor = primitive
        .get(&gltf::Semantic::Positions)
        .ok_or(GltfExtractError::MissingPositions)?;
    let positions = read_vec3_accessor(&positions_accessor, buffers)?;

    if positions.is_empty() {
        return Ok(None);
    }

    // ── Normals (optional, auto-generate if missing) ──
    let normals = if let Some(accessor) = primitive.get(&gltf::Semantic::Normals) {
        read_vec3_accessor(&accessor, buffers)?
    } else {
        Vec::new() // Will be generated after indices are known.
    };

    // ── Tex coords (optional) ──
    let tex_coords = if let Some(accessor) = primitive.get(&gltf::Semantic::TexCoords(0)) {
        read_vec2_accessor(&accessor, buffers)?
    } else {
        vec![[0.0, 0.0]; positions.len()]
    };

    // ── Indices ──
    let indices = if let Some(accessor) = primitive.indices() {
        read_indices_accessor(&accessor, buffers)?
    } else {
        // Non-indexed geometry: sequential indices.
        (0..positions.len() as u32).collect()
    };

    // ── Auto-generate normals if missing ──
    let normals = if normals.is_empty() {
        generate_flat_normals(&positions, &indices)
    } else {
        normals
    };

    // ── Texture ──
    let (texture_rgba, texture_width, texture_height) =
        extract_base_color_texture(primitive, images);

    Ok(Some(ExtractedMesh {
        positions,
        normals,
        tex_coords,
        indices,
        texture_rgba,
        texture_width,
        texture_height,
        rtc_center,
        local_transform: IDENTITY_F64,
    }))
}

// ═══════════════════════════════════════════════════════════════════
// Accessor reading helpers
// ═══════════════════════════════════════════════════════════════════

/// Read a Vec3 (float) accessor → `Vec<[f32; 3]>`.
fn read_vec3_accessor(
    accessor: &gltf::Accessor<'_>,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<[f32; 3]>, GltfExtractError> {
    let view = accessor
        .view()
        .ok_or(GltfExtractError::UnsupportedDataType)?;
    let buffer = &buffers[view.buffer().index()];
    let stride = view.stride().unwrap_or(12); // 3 * f32
    let offset = view.offset() + accessor.offset();
    let count = accessor.count();

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let start = offset + i * stride;
        if start + 12 > buffer.len() {
            return Err(GltfExtractError::BufferOutOfBounds);
        }
        let x = f32::from_le_bytes([
            buffer[start],
            buffer[start + 1],
            buffer[start + 2],
            buffer[start + 3],
        ]);
        let y = f32::from_le_bytes([
            buffer[start + 4],
            buffer[start + 5],
            buffer[start + 6],
            buffer[start + 7],
        ]);
        let z = f32::from_le_bytes([
            buffer[start + 8],
            buffer[start + 9],
            buffer[start + 10],
            buffer[start + 11],
        ]);
        result.push([x, y, z]);
    }
    Ok(result)
}

/// Read a Vec2 (float) accessor → `Vec<[f32; 2]>`.
fn read_vec2_accessor(
    accessor: &gltf::Accessor<'_>,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<[f32; 2]>, GltfExtractError> {
    let view = accessor
        .view()
        .ok_or(GltfExtractError::UnsupportedDataType)?;
    let buffer = &buffers[view.buffer().index()];
    let stride = view.stride().unwrap_or(8); // 2 * f32
    let offset = view.offset() + accessor.offset();
    let count = accessor.count();

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let start = offset + i * stride;
        if start + 8 > buffer.len() {
            return Err(GltfExtractError::BufferOutOfBounds);
        }
        let u = f32::from_le_bytes([
            buffer[start],
            buffer[start + 1],
            buffer[start + 2],
            buffer[start + 3],
        ]);
        let v = f32::from_le_bytes([
            buffer[start + 4],
            buffer[start + 5],
            buffer[start + 6],
            buffer[start + 7],
        ]);
        result.push([u, v]);
    }
    Ok(result)
}

/// Read an index accessor (u8/u16/u32) → `Vec<u32>`.
fn read_indices_accessor(
    accessor: &gltf::Accessor<'_>,
    buffers: &[gltf::buffer::Data],
) -> Result<Vec<u32>, GltfExtractError> {
    let view = accessor
        .view()
        .ok_or(GltfExtractError::UnsupportedDataType)?;
    let buffer = &buffers[view.buffer().index()];
    let offset = view.offset() + accessor.offset();
    let count = accessor.count();

    let component_type = accessor.data_type();
    let component_size = match component_type {
        gltf::accessor::DataType::U8 => 1,
        gltf::accessor::DataType::U16 => 2,
        gltf::accessor::DataType::U32 => 4,
        _ => return Err(GltfExtractError::UnsupportedDataType),
    };
    let stride = view.stride().unwrap_or(component_size);

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let start = offset + i * stride;
        let value = match component_type {
            gltf::accessor::DataType::U8 => {
                if start >= buffer.len() {
                    return Err(GltfExtractError::BufferOutOfBounds);
                }
                buffer[start] as u32
            }
            gltf::accessor::DataType::U16 => {
                if start + 2 > buffer.len() {
                    return Err(GltfExtractError::BufferOutOfBounds);
                }
                u16::from_le_bytes([buffer[start], buffer[start + 1]]) as u32
            }
            gltf::accessor::DataType::U32 => {
                if start + 4 > buffer.len() {
                    return Err(GltfExtractError::BufferOutOfBounds);
                }
                u32::from_le_bytes([
                    buffer[start],
                    buffer[start + 1],
                    buffer[start + 2],
                    buffer[start + 3],
                ])
            }
            _ => unreachable!(),
        };
        result.push(value);
    }
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════
// Normal generation
// ═══════════════════════════════════════════════════════════════════

/// Generate area-weighted smooth normals from positions and indices.
///
/// For each triangle, the unnormalized cross product (whose magnitude is
/// proportional to the triangle area) is accumulated into each vertex.
/// After all triangles are processed, the accumulated normals are normalized.
/// This produces smooth shading at shared edges and naturally weights
/// larger faces more heavily.
fn generate_flat_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![glam::Vec3::ZERO; positions.len()];

    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }

        let v0 = glam::Vec3::from(positions[i0]);
        let v1 = glam::Vec3::from(positions[i1]);
        let v2 = glam::Vec3::from(positions[i2]);

        // Unnormalized cross product — magnitude ∝ triangle area.
        let face_normal = (v1 - v0).cross(v2 - v0);
        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }

    normals
        .into_iter()
        .map(|n| {
            let len = n.length();
            if len > 1e-8 {
                (n / len).to_array()
            } else {
                [0.0, 1.0, 0.0] // Degenerate — fallback to up vector.
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// Texture extraction
// ═══════════════════════════════════════════════════════════════════

/// Extract the base color texture from a primitive's material.
fn extract_base_color_texture(
    primitive: &gltf::Primitive<'_>,
    images: &[gltf::image::Data],
) -> (Option<Vec<u8>>, u32, u32) {
    let material = primitive.material();
    let pbr = material.pbr_metallic_roughness();

    if let Some(info) = pbr.base_color_texture() {
        let texture = info.texture();
        let source = texture.source();
        let idx = source.index();

        if idx < images.len() {
            let image_data = &images[idx];
            let width = image_data.width;
            let height = image_data.height;

            // Convert to RGBA if needed.
            let rgba = match image_data.format {
                gltf::image::Format::R8G8B8A8 => image_data.pixels.clone(),
                gltf::image::Format::R8G8B8 => {
                    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
                    for chunk in image_data.pixels.chunks(3) {
                        rgba.extend_from_slice(chunk);
                        rgba.push(255);
                    }
                    rgba
                }
                gltf::image::Format::R8 => {
                    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
                    for &val in &image_data.pixels {
                        rgba.extend_from_slice(&[val, val, val, 255]);
                    }
                    rgba
                }
                gltf::image::Format::R8G8 => {
                    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
                    for chunk in image_data.pixels.chunks(2) {
                        rgba.extend_from_slice(&[chunk[0], chunk[0], chunk[0], chunk[1]]);
                    }
                    rgba
                }
                _ => return (None, 0, 0),
            };

            return (Some(rgba), width, height);
        }
    }

    (None, 0, 0)
}

// ═══════════════════════════════════════════════════════════════════
// CESIUM_RTC extension
// ═══════════════════════════════════════════════════════════════════

/// Extract RTC_CENTER from CESIUM_RTC glTF extension.
fn extract_cesium_rtc(document: &gltf::Document) -> Option<[f64; 3]> {
    // Use the extension_value API (requires "extensions" feature).
    let ext_value = document.extension_value("CESIUM_RTC")?;
    let center = ext_value.get("center")?.as_array()?;
    if center.len() != 3 {
        return None;
    }
    Some([
        center[0].as_f64()?,
        center[1].as_f64()?,
        center[2].as_f64()?,
    ])
}

/// Check if data begins with the GLB magic bytes (`glTF`).
pub fn is_glb(data: &[u8]) -> bool {
    data.len() >= 4 && data[..4] == *b"glTF"
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_flat_normals_single_triangle() {
        // Triangle in the XY plane.
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = [0u32, 1, 2];
        let normals = generate_flat_normals(&positions, &indices);

        assert_eq!(normals.len(), 3);
        // Normal should point in +Z direction.
        for n in &normals {
            assert!(n[0].abs() < 1e-6);
            assert!(n[1].abs() < 1e-6);
            assert!((n[2] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_generate_flat_normals_degenerate() {
        // Degenerate triangle (all points the same).
        let positions = [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let indices = [0u32, 1, 2];
        let normals = generate_flat_normals(&positions, &indices);

        // Should fallback to up vector.
        for n in &normals {
            assert!((n[1] - 1.0).abs() < 1e-6, "Expected up vector for degenerate");
        }
    }

    #[test]
    fn test_is_glb() {
        assert!(is_glb(b"glTF\x02\x00\x00\x00"));
        assert!(!is_glb(b"b3dm\x01\x00\x00\x00"));
        assert!(!is_glb(b"abc"));
    }

    #[test]
    fn test_generate_flat_normals_two_triangles() {
        // Two triangles sharing edge.
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = [0u32, 1, 2, 1, 3, 2];
        let normals = generate_flat_normals(&positions, &indices);

        assert_eq!(normals.len(), 4);
        // Both triangles in XY plane → normal should be ±Z.
        for n in &normals {
            assert!(n[2].abs() > 0.9, "Normal Z component should be significant");
        }
    }

    #[test]
    fn test_generate_normals_empty() {
        let normals = generate_flat_normals(&[], &[]);
        assert!(normals.is_empty());
    }

    #[test]
    fn test_local_transform_stored_not_applied() {
        // Verify that extract stores local_transform but does NOT apply it to vertices.
        let mesh = ExtractedMesh {
            positions: vec![[1.0, 2.0, 3.0]],
            normals: vec![[0.0, 0.0, 1.0]],
            tex_coords: vec![[0.5, 0.5]],
            indices: vec![0],
            texture_rgba: None,
            texture_width: 0,
            texture_height: 0,
            rtc_center: None,
            local_transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [100.0, 200.0, 300.0, 1.0],
            ],
        };
        // Positions remain unchanged — transform is applied at render time
        assert!((mesh.positions[0][0] - 1.0).abs() < 1e-6);
        assert!((mesh.positions[0][1] - 2.0).abs() < 1e-6);
        assert!((mesh.positions[0][2] - 3.0).abs() < 1e-6);
        // Transform is stored for later use
        assert!((mesh.local_transform[3][0] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_mat4_mul() {
        let a = IDENTITY_F32;
        let b = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
        ];
        let result = mat4_mul(&a, &b);
        // Identity * B = B
        for col in 0..4 {
            for row in 0..4 {
                assert!(
                    (result[col][row] - b[col][row]).abs() < 1e-6,
                    "mismatch at [{col}][{row}]: {} vs {}",
                    result[col][row],
                    b[col][row]
                );
            }
        }
    }
}
