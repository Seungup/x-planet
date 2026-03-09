//! 3D Tiles authentication, loading, and per-layer state for native platform.
//!
//! Handles:
//! - Cesium Ion 2-token authentication workflow
//! - Google 3D Tiles session-based authentication
//! - Per-layer state management (delegates GPU ops to core's Tiles3dGpuState)

use serde::Deserialize;

use x_planets_render::Tiles3dGpuState;
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
    },
}

// ═══════════════════════════════════════════════════════════════════
// Per-layer state (wraps core Tiles3dGpuState)
// ═══════════════════════════════════════════════════════════════════

/// Per-layer state for a 3D Tiles layer on native platform.
///
/// Combines platform-specific HTTP client with the core GPU state.
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
    /// Whether the initialization task has been spawned.
    pub init_spawned: bool,
    /// GPU state: uploaded tiles, loaded/pending URIs.
    pub gpu: Tiles3dGpuState,
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
            init_spawned: false,
            gpu: Tiles3dGpuState::new(),
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.tileset.is_some()
    }
}
