//! Platform-agnostic GPU state for 3D Tiles layers.
//!
//! Contains:
//! - `GpuTileContent`: per-tile GPU models + RTC centers
//! - `Tiles3dGpuState`: GPU-side state (uploaded tiles, loaded/pending URIs)
//! - Upload, transform update, and render collection logic
//!
//! Platform-specific concerns (HTTP, auth, async tasks) remain in
//! the native/web crates.

use std::collections::{HashMap, HashSet};

use x_planets_gpu::GpuContext;

use crate::model3d_renderer::{GpuModel3d, Model3dRenderer, Model3dVertex};
use crate::shared_render_resources::SharedRenderResources;

/// A single GPU-uploaded tile content (one or more meshes).
pub struct GpuTileContent {
    /// GPU models (one per mesh in the decoded tile).
    pub models: Vec<GpuModel3d>,
    /// RTC centers for each mesh (needed for model matrix computation).
    pub rtc_centers: Vec<Option<[f64; 3]>>,
}

/// Platform-agnostic GPU state for a 3D Tiles layer.
///
/// Owns GPU-uploaded tile models and tracks which content URIs
/// have been loaded or are pending. Does NOT own HTTP clients
/// or async runtimes — those belong to the platform crate.
pub struct Tiles3dGpuState {
    /// GPU-uploaded tile contents, keyed by content URI.
    pub gpu_tiles: HashMap<String, GpuTileContent>,
    /// Content URIs that have been loaded and have GPU models.
    pub loaded_uris: HashSet<String>,
    /// Content URIs currently being fetched.
    pub pending_uris: HashSet<String>,
    /// Maximum concurrent tile content loads.
    pub max_concurrent: usize,
}

impl Tiles3dGpuState {
    pub fn new() -> Self {
        Self {
            gpu_tiles: HashMap::new(),
            loaded_uris: HashSet::new(),
            pending_uris: HashSet::new(),
            max_concurrent: 6,
        }
    }

    /// Upload a decoded 3D tile to the GPU.
    pub fn upload_decoded_tile(
        &mut self,
        gpu: &GpuContext,
        shared: &SharedRenderResources,
        renderer: &Model3dRenderer,
        content_uri: &str,
        decoded: &x_planets_tiles::tiles3d::decoder::Decoded3dTile,
    ) {
        let mut models = Vec::new();
        let mut rtc_centers = Vec::new();

        for (i, mesh) in decoded.meshes.iter().enumerate() {
            let vertices: Vec<Model3dVertex> = (0..mesh.positions.len())
                .map(|j| Model3dVertex {
                    position: mesh.positions[j],
                    normal: if j < mesh.normals.len() {
                        mesh.normals[j]
                    } else {
                        [0.0, 1.0, 0.0]
                    },
                    tex_coord: if j < mesh.tex_coords.len() {
                        mesh.tex_coords[j]
                    } else {
                        [0.0, 0.0]
                    },
                })
                .collect();

            if vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }

            let texture_view = mesh.texture_rgba.as_ref().and_then(|rgba| {
                let w = mesh.texture_width;
                let h = mesh.texture_height;
                if w > 0 && h > 0 && rgba.len() == (w * h * 4) as usize {
                    Some(Model3dRenderer::create_texture(
                        gpu,
                        &format!("{}-tex-{}", content_uri, i),
                        w,
                        h,
                        rgba,
                    ))
                } else {
                    None
                }
            });

            let model = renderer.upload_mesh(
                gpu,
                shared,
                &format!("{}-mesh-{}", content_uri, i),
                &vertices,
                &mesh.indices,
                glam::Mat4::IDENTITY.to_cols_array(),
                1.0,
                texture_view.as_ref(),
            );

            models.push(model);
            rtc_centers.push(mesh.rtc_center);
        }

        if !models.is_empty() {
            self.gpu_tiles.insert(
                content_uri.to_string(),
                GpuTileContent {
                    models,
                    rtc_centers,
                },
            );
            self.loaded_uris.insert(content_uri.to_string());
        }
    }

    /// Update model transforms for all tiles in the render set.
    pub fn update_render_transforms(
        &self,
        queue: &wgpu::Queue,
        render_set: &[x_planets_tiles::tiles3d::traversal::TraversalTile],
        camera_ecef: glam::DVec3,
        opacity: f32,
    ) {
        for tile in render_set {
            if let Some(content) = self.gpu_tiles.get(&tile.content_uri) {
                for (model, rtc_center) in
                    content.models.iter().zip(content.rtc_centers.iter())
                {
                    let model_matrix =
                        crate::tiles3d_pipeline::build_model_matrix(*rtc_center, tile.transform);
                    let relative =
                        crate::tiles3d_pipeline::ecef_to_relative_world(model_matrix, camera_ecef);
                    model.update_transform(queue, relative.to_cols_array(), opacity);
                }
            }
        }
    }

    /// Collect all GPU model references for tiles in the render set.
    pub fn collect_render_models(
        &self,
        render_set: &[x_planets_tiles::tiles3d::traversal::TraversalTile],
    ) -> Vec<&GpuModel3d> {
        let mut models = Vec::new();
        for tile in render_set {
            if let Some(content) = self.gpu_tiles.get(&tile.content_uri) {
                for model in &content.models {
                    models.push(model);
                }
            }
        }
        models
    }

    /// Remove GPU resources for tiles no longer needed.
    pub fn unload_tiles<'a, I>(&mut self, uris: I)
    where
        I: IntoIterator<Item = &'a String>,
    {
        for uri in uris {
            self.gpu_tiles.remove(uri);
            self.loaded_uris.remove(uri);
        }
    }
}

impl Default for Tiles3dGpuState {
    fn default() -> Self {
        Self::new()
    }
}
