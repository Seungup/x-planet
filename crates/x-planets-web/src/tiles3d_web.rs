//! 3D Tiles state and async loading for WASM.
//!
//! Single-threaded equivalent of `tiles3d_native.rs`: uses `spawn_local()`
//! and `Rc<RefCell<>>` instead of tokio channels.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use x_planets_core::model3d_renderer::{GpuModel3d, Model3dRenderer, Model3dVertex};
use x_planets_gpu::GpuContext;
use x_planets_tiles::tiles3d::decoder::{decode_3d_tile, Decoded3dTile};
use x_planets_tiles::tiles3d::tileset::Tileset;
use x_planets_tiles::tiles3d::traversal::{
    TraversalCamera, TraversalConfig, TraversalTile, traverse_tileset,
};

// ═══════════════════════════════════════════════════════════════════
// Authentication
// ═══════════════════════════════════════════════════════════════════

/// How to authenticate with a 3D Tiles provider.
#[derive(Debug, Clone)]
pub enum Tiles3dAuthKind {
    CesiumIon {
        account_token: String,
        asset_id: u64,
    },
    Google {
        api_key: String,
    },
}

// ═══════════════════════════════════════════════════════════════════
// Messages (via Rc<RefCell<Vec<...>>>)
// ═══════════════════════════════════════════════════════════════════

enum Tiles3dMsg {
    Initialized {
        tileset: Tileset,
        base_url: String,
        access_token: Option<String>,
    },
    InitFailed(String),
    ContentLoaded {
        content_uri: String,
        decoded: Decoded3dTile,
        generation: u64,
    },
    ContentFailed {
        content_uri: String,
        error: String,
    },
}

// ═══════════════════════════════════════════════════════════════════
// GPU tile content
// ═══════════════════════════════════════════════════════════════════

pub struct GpuTileContent {
    pub models: Vec<GpuModel3d>,
    pub rtc_centers: Vec<Option<[f64; 3]>>,
    pub gpu_bytes: usize,
    pub last_access: u64,
}

// ═══════════════════════════════════════════════════════════════════
// Per-layer state
// ═══════════════════════════════════════════════════════════════════

pub struct Tiles3dWebState {
    pub name: String,
    pub auth: Tiles3dAuthKind,
    pub tileset: Option<Tileset>,
    pub base_url: String,
    pub access_token: Option<String>,
    pub gpu_tiles: HashMap<String, GpuTileContent>,
    pub loaded_uris: HashSet<String>,
    pub pending_uris: HashSet<String>,
    pub init_spawned: bool,
    pub max_concurrent: usize,
    pub max_sse: f64,
    pub tile_budget: usize,
    pub generation: u64,
    pub total_gpu_bytes: usize,
    pub max_gpu_bytes: usize,
    pub stale_loads_skipped: u64,

    /// Message queue (filled by spawn_local, drained each frame).
    msg_queue: Rc<RefCell<Vec<Tiles3dMsg>>>,
}

impl Tiles3dWebState {
    pub fn new(name: String, auth: Tiles3dAuthKind) -> Self {
        Self {
            name,
            auth,
            tileset: None,
            base_url: String::new(),
            access_token: None,
            gpu_tiles: HashMap::new(),
            loaded_uris: HashSet::new(),
            pending_uris: HashSet::new(),
            init_spawned: false,
            max_concurrent: 4,
            max_sse: 16.0,
            tile_budget: 200,
            generation: 0,
            total_gpu_bytes: 0,
            max_gpu_bytes: 256 * 1024 * 1024, // 256 MB for mobile/web
            stale_loads_skipped: 0,
            msg_queue: Rc::new(RefCell::new(Vec::new())),
        }
    }

    pub fn with_config(mut self, max_sse: Option<f64>, tile_budget: Option<usize>) -> Self {
        if let Some(sse) = max_sse {
            if sse > 0.0 { self.max_sse = sse; }
        }
        if let Some(budget) = tile_budget {
            if budget > 0 { self.tile_budget = budget; }
        }
        self
    }

    pub fn is_initialized(&self) -> bool {
        self.tileset.is_some()
    }

    // ── Init ──

    pub fn spawn_init(&mut self) {
        if self.init_spawned {
            return;
        }
        self.init_spawned = true;
        let queue = Rc::clone(&self.msg_queue);

        match &self.auth {
            Tiles3dAuthKind::CesiumIon { account_token, asset_id } => {
                let token = account_token.clone();
                let aid = *asset_id;
                wasm_bindgen_futures::spawn_local(async move {
                    match cesium_init_web(&token, aid).await {
                        Ok((tileset, base_url, access_token)) => {
                            queue.borrow_mut().push(Tiles3dMsg::Initialized {
                                tileset, base_url, access_token: Some(access_token),
                            });
                        }
                        Err(e) => {
                            queue.borrow_mut().push(Tiles3dMsg::InitFailed(e));
                        }
                    }
                });
            }
            Tiles3dAuthKind::Google { api_key } => {
                let key = api_key.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let url = format!(
                        "https://tile.googleapis.com/v1/3dtiles/root.json?key={}", key
                    );
                    match fetch_and_parse_tileset(&url, None).await {
                        Ok((tileset, base_url)) => {
                            queue.borrow_mut().push(Tiles3dMsg::Initialized {
                                tileset, base_url, access_token: None,
                            });
                        }
                        Err(e) => {
                            queue.borrow_mut().push(Tiles3dMsg::InitFailed(e));
                        }
                    }
                });
            }
        }
    }

    // ── Poll messages ──

    pub fn poll_messages(
        &mut self,
        gpu: &GpuContext,
        renderer: &Model3dRenderer,
    ) {
        let msgs: Vec<Tiles3dMsg> = self.msg_queue.borrow_mut().drain(..).collect();
        for msg in msgs {
            match msg {
                Tiles3dMsg::Initialized { tileset, base_url, access_token } => {
                    let tile_count = x_planets_tiles::tiles3d::tileset::tile_count(&tileset.root);
                    log::info!("[{}] 3D Tiles initialized: {} tiles", self.name, tile_count);
                    self.tileset = Some(tileset);
                    self.base_url = base_url;
                    self.access_token = access_token;
                }
                Tiles3dMsg::InitFailed(e) => {
                    log::error!("[{}] 3D Tiles init failed: {}", self.name, e);
                }
                Tiles3dMsg::ContentLoaded { content_uri, decoded, generation } => {
                    self.pending_uris.remove(&content_uri);
                    if generation < self.generation.saturating_sub(2) {
                        self.stale_loads_skipped += 1;
                        return;
                    }
                    self.upload_decoded_tile(gpu, renderer, &content_uri, &decoded);
                }
                Tiles3dMsg::ContentFailed { content_uri, error } => {
                    self.pending_uris.remove(&content_uri);
                    log::warn!("[{}] 3D tile failed: {} - {}", self.name, content_uri, error);
                }
            }
        }
    }

    // ── Traversal + load spawning ──

    pub fn traverse_and_spawn_loads(
        &mut self,
        camera: &TraversalCamera,
        screen_height: f64,
        fov_y: f64,
    ) -> Vec<TraversalTile> {
        let tileset = match &self.tileset {
            Some(ts) => ts,
            None => return Vec::new(),
        };

        self.generation += 1;
        let current_gen = self.generation;

        let config = TraversalConfig {
            max_sse: self.max_sse,
            tile_budget: self.tile_budget,
            screen_height,
            fov_y,
        };

        let traversal = traverse_tileset(
            tileset, &self.base_url, camera, &self.loaded_uris, &config,
        );

        // Spawn loads
        for req in &traversal.load_requests {
            if self.pending_uris.contains(&req.content_uri) {
                continue;
            }
            if self.pending_uris.len() >= self.max_concurrent {
                break;
            }
            self.pending_uris.insert(req.content_uri.clone());

            let queue = Rc::clone(&self.msg_queue);
            let content_uri = req.content_uri.clone();
            let access_token = self.access_token.clone();

            wasm_bindgen_futures::spawn_local(async move {
                match fetch_tile_bytes(&content_uri, access_token.as_deref()).await {
                    Ok(bytes) => {
                        match decode_3d_tile(&bytes, &content_uri) {
                            Ok(decoded) => {
                                queue.borrow_mut().push(Tiles3dMsg::ContentLoaded {
                                    content_uri, decoded, generation: current_gen,
                                });
                            }
                            Err(e) => {
                                queue.borrow_mut().push(Tiles3dMsg::ContentFailed {
                                    content_uri, error: e.to_string(),
                                });
                            }
                        }
                    }
                    Err(e) => {
                        queue.borrow_mut().push(Tiles3dMsg::ContentFailed {
                            content_uri, error: e,
                        });
                    }
                }
            });
        }

        // Unload
        for uri in &traversal.unload_set {
            if let Some(evicted) = self.gpu_tiles.remove(uri) {
                self.total_gpu_bytes = self.total_gpu_bytes.saturating_sub(evicted.gpu_bytes);
            }
            self.loaded_uris.remove(uri);
        }

        traversal.render_set
    }

    // ── GPU upload ──

    fn upload_decoded_tile(
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
            let vertices: Vec<Model3dVertex> = (0..mesh.positions.len())
                .map(|j| Model3dVertex {
                    position: mesh.positions[j],
                    normal: if j < mesh.normals.len() { mesh.normals[j] } else { [0.0, 1.0, 0.0] },
                    tex_coord: if j < mesh.tex_coords.len() { mesh.tex_coords[j] } else { [0.0, 0.0] },
                })
                .collect();

            if vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }

            let vertex_bytes = vertices.len() * std::mem::size_of::<Model3dVertex>();
            let index_bytes = mesh.indices.len() * std::mem::size_of::<u32>();
            let texture_bytes = mesh.texture_rgba.as_ref().map_or(0, |rgba| rgba.len());
            tile_gpu_bytes += vertex_bytes + index_bytes + texture_bytes;

            let texture_view = mesh.texture_rgba.as_ref().and_then(|rgba| {
                let w = mesh.texture_width;
                let h = mesh.texture_height;
                if w > 0 && h > 0 && rgba.len() == (w * h * 4) as usize {
                    Some(Model3dRenderer::create_texture(
                        gpu, &format!("{}-tex-{}", content_uri, i), w, h, rgba,
                    ))
                } else {
                    None
                }
            });

            let model = renderer.upload_mesh(
                gpu,
                &format!("{}-mesh-{}", content_uri, i),
                &vertices, &mesh.indices,
                glam::Mat4::IDENTITY.to_cols_array(), 1.0,
                texture_view.as_ref(),
            );

            models.push(model);
            rtc_centers.push(mesh.rtc_center);
        }

        if !models.is_empty() {
            self.total_gpu_bytes += tile_gpu_bytes;
            self.gpu_tiles.insert(content_uri.to_string(), GpuTileContent {
                models, rtc_centers,
                gpu_bytes: tile_gpu_bytes,
                last_access: self.generation,
            });
            self.loaded_uris.insert(content_uri.to_string());
            self.evict_over_budget();
        }
    }

    fn evict_over_budget(&mut self) {
        while self.total_gpu_bytes > self.max_gpu_bytes && !self.gpu_tiles.is_empty() {
            let oldest = self.gpu_tiles.iter()
                .min_by_key(|(_, c)| c.last_access)
                .map(|(uri, _)| uri.clone());
            if let Some(uri) = oldest {
                if let Some(evicted) = self.gpu_tiles.remove(&uri) {
                    self.total_gpu_bytes = self.total_gpu_bytes.saturating_sub(evicted.gpu_bytes);
                    self.loaded_uris.remove(&uri);
                }
            } else {
                break;
            }
        }
    }

    // ── Transform + render helpers ──

    pub fn update_render_transforms(
        &mut self,
        queue: &wgpu::Queue,
        render_set: &[TraversalTile],
        camera_ecef: glam::DVec3,
        opacity: f32,
    ) {
        let gen = self.generation;
        for tile in render_set {
            if let Some(content) = self.gpu_tiles.get_mut(&tile.content_uri) {
                content.last_access = gen;
                for (model, rtc) in content.models.iter().zip(content.rtc_centers.iter()) {
                    let mm = x_planets_core::tiles3d_pipeline::build_model_matrix(*rtc, tile.transform);
                    let rel = x_planets_core::tiles3d_pipeline::ecef_to_relative_world(mm, camera_ecef);
                    model.update_transform(queue, rel.to_cols_array(), opacity);
                }
            }
        }
    }

    pub fn collect_render_models<'a>(
        &'a self,
        render_set: &[TraversalTile],
    ) -> Vec<&'a GpuModel3d> {
        let mut out = Vec::new();
        for tile in render_set {
            if let Some(content) = self.gpu_tiles.get(&tile.content_uri) {
                for model in &content.models {
                    out.push(model);
                }
            }
        }
        out
    }
}

// ═══════════════════════════════════════════════════════════════════
// Async helpers (Fetch API)
// ═══════════════════════════════════════════════════════════════════

async fn fetch_json(url: &str, bearer: Option<&str>) -> Result<String, String> {
    let window = web_sys::window().ok_or("No window")?;

    let request = if let Some(token) = bearer {
        let opts = web_sys::RequestInit::new();
        let headers = web_sys::Headers::new().map_err(|e| format!("{:?}", e))?;
        headers.set("Authorization", &format!("Bearer {}", token))
            .map_err(|e| format!("{:?}", e))?;
        opts.set_headers(&headers);
        web_sys::Request::new_with_str_and_init(url, &opts)
            .map_err(|e| format!("{:?}", e))?
    } else {
        web_sys::Request::new_with_str(url).map_err(|e| format!("{:?}", e))?
    };

    let resp = JsFuture::from(window.fetch_with_request(&request))
        .await.map_err(|e| format!("{:?}", e))?;
    let resp: web_sys::Response = resp.dyn_into().map_err(|_| "cast failed")?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let text = JsFuture::from(resp.text().map_err(|e| format!("{:?}", e))?)
        .await.map_err(|e| format!("{:?}", e))?;
    text.as_string().ok_or_else(|| "not a string".to_string())
}

async fn fetch_tile_bytes(url: &str, bearer: Option<&str>) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or("No window")?;

    let request = if let Some(token) = bearer {
        let opts = web_sys::RequestInit::new();
        let headers = web_sys::Headers::new().map_err(|e| format!("{:?}", e))?;
        headers.set("Authorization", &format!("Bearer {}", token))
            .map_err(|e| format!("{:?}", e))?;
        opts.set_headers(&headers);
        web_sys::Request::new_with_str_and_init(url, &opts)
            .map_err(|e| format!("{:?}", e))?
    } else {
        web_sys::Request::new_with_str(url).map_err(|e| format!("{:?}", e))?
    };

    let resp = JsFuture::from(window.fetch_with_request(&request))
        .await.map_err(|e| format!("{:?}", e))?;
    let resp: web_sys::Response = resp.dyn_into().map_err(|_| "cast failed")?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let buf = JsFuture::from(resp.array_buffer().map_err(|e| format!("{:?}", e))?)
        .await.map_err(|e| format!("{:?}", e))?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

async fn fetch_and_parse_tileset(url: &str, bearer: Option<&str>) -> Result<(Tileset, String), String> {
    let json = fetch_json(url, bearer).await?;
    let tileset = x_planets_tiles::tiles3d::tileset::parse_tileset(json.as_bytes())
        .map_err(|e| e.to_string())?;
    // Base URL = everything before the last '/'
    let base_url = url.rfind('/').map(|i| &url[..=i]).unwrap_or(url).to_string();
    Ok((tileset, base_url))
}

async fn cesium_init_web(account_token: &str, asset_id: u64) -> Result<(Tileset, String, String), String> {
    // Step 1: resolve Cesium Ion endpoint
    let endpoint_url = format!(
        "https://api.cesium.com/v1/assets/{}/endpoint", asset_id
    );
    let json = fetch_json(&endpoint_url, Some(account_token)).await?;
    let ep: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
    let tileset_url = ep["url"].as_str().ok_or("no url in endpoint response")?.to_string();
    let access_token = ep["accessToken"].as_str().ok_or("no accessToken")?.to_string();

    // Step 2: fetch tileset.json
    let (tileset, base_url) = fetch_and_parse_tileset(&tileset_url, Some(&access_token)).await?;
    Ok((tileset, base_url, access_token))
}
