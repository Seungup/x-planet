//! 3D Tiles authentication, loading, and per-layer state for native platform.
//!
//! Handles:
//! - Cesium Ion 2-token authentication workflow
//! - Google 3D Tiles session-based authentication
//! - Per-layer state management (tileset, GPU models, load queue)

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use x_planets_render::{GpuModel3d, Model3dRenderer, Model3dVertex};
use x_planets_gpu::GpuContext;
use x_planets_tiles::tiles3d::decoder::Decoded3dTile;
use x_planets_tiles::tiles3d::tileset::Tileset;

// ═══════════════════════════════════════════════════════════════════
// Authentication
// ═══════════════════════════════════════════════════════════════════

/// Cesium Ion endpoint resolution response.
#[derive(Deserialize)]
struct CesiumEndpointResponse {
    url: String,
    #[serde(rename = "accessToken")]
    access_token: String,
}

/// Which 3D Tiles authentication provider to use.
pub enum Tiles3dAuthKind {
    /// Cesium Ion: 2-token workflow (account token → access token).
    CesiumIon {
        account_token: String,
        asset_id: u64,
    },
    /// Google Maps Platform: API key + session token.
    Google {
        api_key: String,
    },
}

/// Resolve a Cesium Ion asset endpoint.
///
/// Returns `(tileset_base_url, access_token)`.
pub async fn cesium_resolve_endpoint(
    client: &reqwest::Client,
    account_token: &str,
    asset_id: u64,
) -> Result<(String, String), String> {
    let url = format!(
        "https://api.cesium.com/v1/assets/{}/endpoint?access_token={}",
        asset_id, account_token
    );
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Cesium endpoint request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Cesium endpoint error: HTTP {}", resp.status()));
    }

    let body: CesiumEndpointResponse = resp
        .json()
        .await
        .map_err(|e| format!("Cesium endpoint parse failed: {}", e))?;

    Ok((body.url, body.access_token))
}

/// Fetch and parse a tileset.json.
///
/// Returns `(Tileset, base_url)`.
pub async fn fetch_tileset(
    client: &reqwest::Client,
    url: &str,
    bearer_token: Option<&str>,
) -> Result<(Tileset, String), String> {
    let mut req = client.get(url);
    if let Some(token) = bearer_token {
        req = req.bearer_auth(token);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("Tileset fetch failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Tileset fetch error: HTTP {}", resp.status()));
    }

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Tileset read failed: {}", e))?;

    let tileset = x_planets_tiles::tiles3d::tileset::parse_tileset(&bytes)
        .map_err(|e| format!("Tileset parse failed: {}", e))?;

    Ok((tileset, url.to_string()))
}

/// Fetch raw tile content bytes (B3DM/GLB).
pub async fn fetch_tile_content(
    client: &reqwest::Client,
    url: &str,
    bearer_token: Option<&str>,
) -> Result<Vec<u8>, String> {
    let mut req = client.get(url);
    if let Some(token) = bearer_token {
        req = req.bearer_auth(token);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("Tile content fetch failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Tile content error: HTTP {} for {}",
            resp.status(),
            url
        ));
    }

    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| format!("Tile content read failed: {}", e))
}

// ═══════════════════════════════════════════════════════════════════
// Messages (async → main thread)
// ═══════════════════════════════════════════════════════════════════

/// Result of 3D Tiles initialization: (tileset, base_url, access_token).
pub type Tiles3dInitResult = Result<(Tileset, String, Option<String>), String>;

/// Messages sent from async tasks back to the main render loop.
pub enum Tiles3dMessage {
    /// Auth resolved + tileset.json fetched and parsed.
    Initialized {
        layer_name: String,
        result: Box<Tiles3dInitResult>,
    },
    /// A tile's content (B3DM/GLB) has been fetched and decoded.
    ContentLoaded {
        layer_name: String,
        content_uri: String,
        result: Result<Decoded3dTile, String>,
        /// Generation at which this load was requested.
        generation: u64,
    },
    /// An external tileset JSON was loaded and needs to be spliced into the tree.
    ExternalTilesetLoaded {
        layer_name: String,
        content_uri: String,
        tileset: Tileset,
        base_url: String,
        generation: u64,
    },
}

/// Runtime statistics for a 3D Tiles layer (for DX/debugging).
#[derive(Debug, Clone, Default)]
pub struct Tiles3dStats {
    pub rendered_tiles: usize,
    pub pending_loads: usize,
    pub loaded_tiles: usize,
    pub gpu_bytes: usize,
    pub stale_loads_skipped: usize,
    pub total_triangles: u64,
}

impl std::fmt::Display for Tiles3dStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rendered={} pending={} loaded={} gpu={:.1}MB stale_skipped={} tris={}",
            self.rendered_tiles,
            self.pending_loads,
            self.loaded_tiles,
            self.gpu_bytes as f64 / (1024.0 * 1024.0),
            self.stale_loads_skipped,
            self.total_triangles,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
// Per-layer state
// ═══════════════════════════════════════════════════════════════════

/// A single GPU-uploaded tile content (one or more meshes).
pub struct GpuTileContent {
    /// GPU models (one per mesh in the decoded tile).
    pub models: Vec<GpuModel3d>,
    /// RTC centers for each mesh (needed for model matrix computation).
    pub rtc_centers: Vec<Option<[f64; 3]>>,
    /// Total GPU memory used by this tile (vertex + index + texture bytes).
    pub gpu_bytes: usize,
    /// Last-access generation for LRU eviction.
    pub last_access: u64,
}

/// Per-layer state for a 3D Tiles layer.
pub struct Tiles3dLayerState {
    pub name: String,
    pub auth: Tiles3dAuthKind,
    pub client: reqwest::Client,
    /// Parsed tileset (set after initialization).
    pub tileset: Option<Tileset>,
    /// Base URL for resolving relative content URIs.
    pub base_url: String,
    /// Access token for tile requests (Cesium Ion bearer token).
    pub access_token: Option<String>,
    /// GPU-uploaded tile contents, keyed by content URI.
    pub gpu_tiles: HashMap<String, GpuTileContent>,
    /// Content URIs that have been loaded and have GPU models.
    pub loaded_uris: HashSet<String>,
    /// Content URIs currently being fetched.
    pub pending_uris: HashSet<String>,
    /// Whether the initialization task has been spawned.
    pub init_spawned: bool,
    /// Maximum concurrent tile content loads (configurable).
    pub max_concurrent: usize,
    /// Maximum screen-space error for LOD traversal (configurable, default 16.0).
    pub max_sse: f64,
    /// Maximum tiles to render per frame (configurable, default 256).
    pub tile_budget: usize,
    /// Generation counter — incremented each traversal for stale request detection.
    pub generation: u64,
    /// Total GPU memory used by this layer (bytes).
    pub total_gpu_bytes: usize,
    /// Maximum GPU memory budget (bytes, default 512 MB).
    pub max_gpu_bytes: usize,
    /// Stale loads skipped since last stats reset.
    pub stale_loads_skipped: u64,
}

impl Tiles3dLayerState {
    pub fn new(name: String, auth: Tiles3dAuthKind) -> Self {
        Self {
            name,
            auth,
            client: crate::http_client(),
            tileset: None,
            base_url: String::new(),
            access_token: None,
            gpu_tiles: HashMap::new(),
            loaded_uris: HashSet::new(),
            pending_uris: HashSet::new(),
            init_spawned: false,
            max_concurrent: 6,
            max_sse: 16.0,
            tile_budget: 256,
            generation: 0,
            total_gpu_bytes: 0,
            max_gpu_bytes: 512 * 1024 * 1024,
            stale_loads_skipped: 0,
        }
    }

    /// Apply config overrides from LayerConfig.
    pub fn with_config(mut self, max_sse: Option<f64>, tile_budget: Option<usize>, max_concurrent: usize) -> Self {
        if let Some(sse) = max_sse {
            if sse > 0.0 {
                self.max_sse = sse;
            } else {
                log::warn!("[{}] ignoring invalid max_sse={}, using default {}", self.name, sse, self.max_sse);
            }
        }
        if let Some(budget) = tile_budget {
            if budget > 0 {
                self.tile_budget = budget;
            } else {
                log::warn!("[{}] ignoring invalid tile_budget={}, using default {}", self.name, budget, self.tile_budget);
            }
        }
        self.max_concurrent = max_concurrent;
        self
    }

    /// Get current statistics for this layer.
    pub fn stats(&self) -> Tiles3dStats {
        Tiles3dStats {
            rendered_tiles: 0, // Set by caller after traversal
            pending_loads: self.pending_uris.len(),
            loaded_tiles: self.loaded_uris.len(),
            gpu_bytes: self.total_gpu_bytes,
            stale_loads_skipped: self.stale_loads_skipped as usize,
            total_triangles: 0, // Set by caller
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.tileset.is_some()
    }

    /// Upload a decoded 3D tile to the GPU.
    ///
    /// Tracks GPU memory usage and evicts LRU tiles if the budget is exceeded.
    pub fn upload_decoded_tile(
        &mut self,
        gpu: &GpuContext,
        renderer: &Model3dRenderer,
        content_uri: &str,
        decoded: &Decoded3dTile,
    ) {
        let mut models = Vec::new();
        let mut rtc_centers = Vec::new();
        let mut tile_gpu_bytes: usize = 0;

        for (i, mesh) in decoded.meshes.iter().enumerate() {
            // Build vertex data.
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

            // Track GPU memory: vertex buffer + index buffer + texture.
            let vertex_bytes = vertices.len() * std::mem::size_of::<Model3dVertex>();
            let index_bytes = mesh.indices.len() * std::mem::size_of::<u32>();
            let texture_bytes = mesh.texture_rgba.as_ref().map_or(0, |rgba| rgba.len());
            tile_gpu_bytes += vertex_bytes + index_bytes + texture_bytes;

            // Create texture if available.
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

            // Upload with identity matrix (updated each frame).
            let model = renderer.upload_mesh(
                gpu,
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
            self.total_gpu_bytes += tile_gpu_bytes;
            self.gpu_tiles.insert(
                content_uri.to_string(),
                GpuTileContent {
                    models,
                    rtc_centers,
                    gpu_bytes: tile_gpu_bytes,
                    last_access: self.generation,
                },
            );
            self.loaded_uris.insert(content_uri.to_string());

            // Evict LRU tiles if over budget.
            self.evict_over_budget();
        }
    }

    /// Evict least-recently-used tiles until GPU memory is within budget.
    fn evict_over_budget(&mut self) {
        while self.total_gpu_bytes > self.max_gpu_bytes && !self.gpu_tiles.is_empty() {
            let oldest_uri = self
                .gpu_tiles
                .iter()
                .min_by_key(|(_, content)| content.last_access)
                .map(|(uri, _)| uri.clone());

            if let Some(uri) = oldest_uri {
                if let Some(evicted) = self.gpu_tiles.remove(&uri) {
                    self.total_gpu_bytes = self.total_gpu_bytes.saturating_sub(evicted.gpu_bytes);
                    self.loaded_uris.remove(&uri);
                    log::debug!(
                        "[{}] evicted tile {} ({:.1} KB) to stay under {:.0} MB budget",
                        self.name,
                        uri,
                        evicted.gpu_bytes as f64 / 1024.0,
                        self.max_gpu_bytes as f64 / (1024.0 * 1024.0),
                    );
                }
            } else {
                break;
            }
        }
    }

    /// Update model transforms for all tiles in the render set.
    ///
    /// Computes ECEF-relative model matrices for each mesh.
    /// Also updates LRU `last_access` for GPU memory eviction.
    pub fn update_render_transforms(
        &mut self,
        queue: &wgpu::Queue,
        render_set: &[x_planets_tiles::tiles3d::traversal::TraversalTile],
        camera_ecef: glam::DVec3,
        opacity: f32,
    ) {
        let current_gen = self.generation;
        for tile in render_set {
            if let Some(content) = self.gpu_tiles.get_mut(&tile.content_uri) {
                content.last_access = current_gen;
                for (model, rtc_center) in
                    content.models.iter().zip(content.rtc_centers.iter())
                {
                    let model_matrix =
                        x_planets_core::tiles3d_pipeline::build_model_matrix(*rtc_center, tile.transform);
                    let relative =
                        x_planets_core::tiles3d_pipeline::ecef_to_relative_world(model_matrix, camera_ecef);
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
}
