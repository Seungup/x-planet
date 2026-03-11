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
use x_planets_tiles::tiles3d::decoder::{decode_3d_tile, Decoded3dTile, Tiles3dContent};
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
    ExternalTilesetLoaded {
        content_uri: String,
        tileset: Tileset,
        base_url: String,
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
    pub local_transforms: Vec<glam::DMat4>,
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
                    let has_implicit = tileset.root.implicit_tiling.is_some();
                    let root_has_content = tileset.root.content.is_some();
                    let root_children = tileset.root.children.len();
                    let root_refine = tileset.root.refine;
                    log::info!(
                        "[{}] 3D Tiles initialized: {} tiles, root: children={}, content={}, implicit={}, refine={:?}, base_url={}",
                        self.name, tile_count, root_children, root_has_content, has_implicit, root_refine, base_url,
                    );
                    if has_implicit {
                        let it = tileset.root.implicit_tiling.as_ref().unwrap();
                        log::warn!(
                            "[{}] implicit tiling detected: scheme={:?}, subtree_levels={}, available_levels={}, subtrees={}, content={:?}",
                            self.name, it.subdivision_scheme, it.subtree_levels, it.available_levels,
                            it.subtrees.uri,
                            it.content.as_ref().map(|c| &c.uri),
                        );
                    }
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
                        continue;
                    }
                    self.upload_decoded_tile(gpu, renderer, &content_uri, &decoded);
                }
                Tiles3dMsg::ExternalTilesetLoaded { content_uri, tileset, base_url, generation } => {
                    self.pending_uris.remove(&content_uri);
                    if generation < self.generation.saturating_sub(2) {
                        self.stale_loads_skipped += 1;
                        continue;
                    }
                    self.splice_external_tileset(&content_uri, &tileset, &base_url);
                }
                Tiles3dMsg::ContentFailed { content_uri, error } => {
                    self.pending_uris.remove(&content_uri);
                    log::warn!("[{}] 3D tile failed: {} - {}", self.name, content_uri, error);
                    // Mark as loaded to prevent infinite re-fetch.
                    self.loaded_uris.insert(content_uri);
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
                            Ok(Tiles3dContent::Mesh(decoded)) => {
                                queue.borrow_mut().push(Tiles3dMsg::ContentLoaded {
                                    content_uri, decoded, generation: current_gen,
                                });
                            }
                            Ok(Tiles3dContent::ExternalTileset { tileset, base_url, content_uri }) => {
                                queue.borrow_mut().push(Tiles3dMsg::ExternalTilesetLoaded {
                                    content_uri, tileset, base_url, generation: current_gen,
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
        if !traversal.unload_set.is_empty() && self.generation % 300 == 1 {
            log::info!(
                "[3dtiles] unloading {} tiles, load_requests={}",
                traversal.unload_set.len(),
                traversal.load_requests.len(),
            );
        }
        for uri in &traversal.unload_set {
            if self.gpu_tiles.contains_key(uri) {
                log::info!("[3dtiles] unloading GPU tile: {}", uri);
            }
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
        let mut local_transforms = Vec::new();
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

            // Log first mesh stats for debugging
            if i == 0 {
                let pos_range = mesh.positions.iter().fold(
                    ([f32::MAX; 3], [f32::MIN; 3]),
                    |(min, max), p| {
                        ([min[0].min(p[0]), min[1].min(p[1]), min[2].min(p[2])],
                         [max[0].max(p[0]), max[1].max(p[1]), max[2].max(p[2])])
                    },
                );
                log::info!(
                    "[3dtiles] upload mesh: {} verts, {} indices, rtc={:?}, pos_range=[({:.1},{:.1},{:.1})..({:.1},{:.1},{:.1})]",
                    vertices.len(), mesh.indices.len(), mesh.rtc_center,
                    pos_range.0[0], pos_range.0[1], pos_range.0[2],
                    pos_range.1[0], pos_range.1[1], pos_range.1[2],
                );
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
            local_transforms.push(glam::DMat4::from_cols_array_2d(&mesh.local_transform));
        }

        // Always mark as loaded to prevent infinite re-fetch of empty tiles.
        self.loaded_uris.insert(content_uri.to_string());

        if !models.is_empty() {
            self.total_gpu_bytes += tile_gpu_bytes;
            self.gpu_tiles.insert(content_uri.to_string(), GpuTileContent {
                models, rtc_centers, local_transforms,
                gpu_bytes: tile_gpu_bytes,
                last_access: self.generation,
            });
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

    // ── External tileset splicing ──

    fn splice_external_tileset(
        &mut self,
        content_uri: &str,
        external: &x_planets_tiles::tiles3d::tileset::Tileset,
        new_base_url: &str,
    ) {
        if let Some(tileset) = &mut self.tileset {
            let spliced = x_planets_tiles::tiles3d::tileset::splice_external_tileset(
                &mut tileset.root,
                &self.base_url,
                content_uri,
                external,
                new_base_url,
            );
            if spliced {
                // Mark as "loaded" so traversal doesn't re-request the .json URI.
                self.loaded_uris.insert(content_uri.to_string());
                let new_count = x_planets_tiles::tiles3d::tileset::tile_count(&tileset.root);
                log::info!(
                    "[{}] spliced external tileset from {} (tree now {} tiles)",
                    self.name, content_uri, new_count,
                );
            } else {
                log::warn!(
                    "[{}] failed to splice external tileset: {} not found in tree",
                    self.name, content_uri,
                );
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
        let mut logged_first = gen % 300 != 0; // Log first tile every ~300 frames
        for tile in render_set {
            if let Some(content) = self.gpu_tiles.get_mut(&tile.content_uri) {
                content.last_access = gen;
                for ((model, rtc), local_tr) in content.models.iter().zip(content.rtc_centers.iter()).zip(content.local_transforms.iter()) {
                    let mm = x_planets_core::tiles3d_pipeline::build_model_matrix(*rtc, *local_tr, tile.transform);
                    let rel = x_planets_core::tiles3d_pipeline::ecef_to_relative_world(mm, camera_ecef);

                    if !logged_first {
                        logged_first = true;
                        let t = rel.col(3);
                        let local_t = local_tr.col(3).truncate();
                        let self_pos = local_t.length() > 10_000.0;
                        log::info!(
                            "[3dtiles] render: {} tiles, camera=({:.0},{:.0},{:.0}), rel=({:.1},{:.1},{:.1}), rtc={:?}, self_pos={}, local_t_len={:.0}",
                            render_set.len(),
                            camera_ecef.x, camera_ecef.y, camera_ecef.z,
                            t.x, t.y, t.z,
                            rtc, self_pos, local_t.length(),
                        );
                    }

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
