//! TOML-based configuration for x-planets.
//!
//! Supports `${ENV_VAR}` substitution in string values so that API keys
//! never need to be hard-coded in the config file.
//!
//! # Example `config.toml`
//!
//! ```toml
//! [map]
//! center = [37.5665, 126.9780]
//! zoom = 5.0
//!
//! [[layers]]
//! name = "imagery"
//! kind = "raster"
//! url  = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
//!
//! [[layers]]
//! name = "terrain"
//! kind = "terrain"
//! imagery_layer = "imagery"
//! url  = "https://api.maptiler.com/tiles/terrain-rgb-v2/{z}/{x}/{y}.webp?key=${MAPTILER_KEY}"
//! ```

use serde::Deserialize;
use std::path::Path;

use x_planets_core::engine::{LayerConfig, LayerKind, MapConfig};
use x_planets_math::GeoCoord;

// ═══════════════════════════════════════════════════════════════════
// TOML schema
// ═══════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
pub struct FileConfig {
    #[serde(default)]
    pub map: MapSection,
    #[serde(default)]
    pub layers: Vec<LayerSection>,
}

#[derive(Deserialize)]
pub struct MapSection {
    /// `[lat, lon]` in degrees.
    #[serde(default = "default_center")]
    pub center: [f64; 2],
    #[serde(default = "default_zoom")]
    pub zoom: f64,
    #[serde(default = "default_projection")]
    pub projection: String,
}

impl Default for MapSection {
    fn default() -> Self {
        Self {
            center: default_center(),
            zoom: default_zoom(),
            projection: default_projection(),
        }
    }
}

fn default_center() -> [f64; 2] {
    [37.5665, 126.9780]
}
fn default_zoom() -> f64 {
    5.0
}
fn default_projection() -> String {
    "Web Mercator".to_string()
}

#[derive(Deserialize)]
pub struct LayerSection {
    pub name: String,
    /// `"raster"` | `"terrain"` | `"3dtiles"`
    #[serde(default = "default_kind_str")]
    pub kind: String,
    /// Tile URL template with `{z}/{x}/{y}`.  Supports `${ENV_VAR}`.
    #[serde(default)]
    pub url: String,
    #[serde(default = "default_opacity")]
    pub opacity: f32,
    #[serde(default = "default_visible")]
    pub visible: bool,
    #[serde(default)]
    pub z_order: i32,
    #[serde(default = "default_max_cached")]
    pub max_cached_tiles: usize,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_loads: usize,

    // ── Terrain-specific ──
    /// For `kind = "terrain"`: the raster layer whose imagery is draped.
    pub imagery_layer: Option<String>,

    // ── 3D Tiles-specific ──
    pub cesium_ion_token: Option<String>,
    pub cesium_ion_asset: Option<u64>,
    pub google_api_key: Option<String>,
}

fn default_kind_str() -> String {
    "raster".to_string()
}
fn default_opacity() -> f32 {
    1.0
}
fn default_visible() -> bool {
    true
}
fn default_max_cached() -> usize {
    512
}
fn default_max_concurrent() -> usize {
    6
}

// ═══════════════════════════════════════════════════════════════════
// Environment variable substitution
// ═══════════════════════════════════════════════════════════════════

/// Replace all `${VAR_NAME}` occurrences with the corresponding env value.
///
/// Returns the substituted string, or `Err` listing any missing variables.
fn expand_env(input: &str) -> Result<String, String> {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    let mut missing: Vec<String> = Vec::new();

    while let Some(ch) = chars.next() {
        if ch == '$' && chars.peek() == Some(&'{') {
            chars.next(); // consume '{'
            let mut var_name = String::new();
            for c in chars.by_ref() {
                if c == '}' {
                    break;
                }
                var_name.push(c);
            }
            match std::env::var(&var_name) {
                Ok(val) => result.push_str(&val),
                Err(_) => missing.push(var_name),
            }
        } else {
            result.push(ch);
        }
    }

    if missing.is_empty() {
        Ok(result)
    } else {
        Err(format!(
            "Missing environment variable(s): {}",
            missing.join(", ")
        ))
    }
}

// ═══════════════════════════════════════════════════════════════════
// Conversion to engine types
// ═══════════════════════════════════════════════════════════════════

impl FileConfig {
    /// Load and parse a TOML config file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        toml::from_str(&text).map_err(|e| format!("Failed to parse {}: {}", path.display(), e))
    }

    /// Convert into engine [`MapConfig`], expanding `${ENV}` in all string fields.
    ///
    /// Layers whose env vars are missing are **skipped** with a warning,
    /// so the app still starts even if some providers are unconfigured.
    pub fn into_map_config(self) -> MapConfig {
        let mut layers = Vec::new();

        for (i, layer) in self.layers.into_iter().enumerate() {
            match convert_layer(layer, i) {
                Ok(lc) => {
                    eprintln!("  ✓ layer {:>2}: \"{}\" ({})", i, lc.name, kind_label(&lc.kind));
                    layers.push(lc);
                }
                Err(e) => {
                    eprintln!("  ✗ layer {:>2}: skipped — {}", i, e);
                }
            }
        }

        MapConfig {
            center: GeoCoord::new(self.map.center[0], self.map.center[1]),
            zoom: self.map.zoom,
            projection: self.map.projection,
            layers,
            ..Default::default()
        }
    }
}

fn kind_label(kind: &LayerKind) -> &'static str {
    match kind {
        LayerKind::Raster => "raster",
        LayerKind::Terrain { .. } => "terrain",
        LayerKind::Tiles3d => "3dtiles",
    }
}

fn convert_layer(section: LayerSection, index: usize) -> Result<LayerConfig, String> {
    let url = expand_env(&section.url)?;

    let kind = match section.kind.to_ascii_lowercase().as_str() {
        "raster" => LayerKind::Raster,
        "terrain" => LayerKind::Terrain {
            imagery_layer: section
                .imagery_layer
                .ok_or_else(|| format!("layer \"{}\" (terrain) requires `imagery_layer`", section.name))?,
        },
        "3dtiles" => LayerKind::Tiles3d,
        other => return Err(format!("layer \"{}\": unknown kind \"{}\"", section.name, other)),
    };

    // Expand env vars in optional token fields.
    let cesium_ion_token = section
        .cesium_ion_token
        .map(|s| expand_env(&s))
        .transpose()?;
    let google_api_key = section
        .google_api_key
        .map(|s| expand_env(&s))
        .transpose()?;

    Ok(LayerConfig {
        name: section.name,
        tile_source_url: url,
        opacity: section.opacity,
        visible: section.visible,
        z_order: if section.z_order != 0 {
            section.z_order
        } else {
            index as i32 * 10
        },
        max_cached_tiles: section.max_cached_tiles,
        max_concurrent_loads: section.max_concurrent_loads,
        kind,
        cesium_ion_token,
        cesium_ion_asset_id: section.cesium_ion_asset,
        google_api_key,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_env_no_vars() {
        let s = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
        assert_eq!(expand_env(s).unwrap(), s);
    }

    #[test]
    fn test_expand_env_with_var() {
        std::env::set_var("TEST_XPLANETS_KEY", "abc123");
        let s = "https://api.example.com/tiles?key=${TEST_XPLANETS_KEY}";
        assert_eq!(
            expand_env(s).unwrap(),
            "https://api.example.com/tiles?key=abc123"
        );
        std::env::remove_var("TEST_XPLANETS_KEY");
    }

    #[test]
    fn test_expand_env_missing_var() {
        let s = "https://api.example.com/tiles?key=${NONEXISTENT_VAR_12345}";
        let err = expand_env(s).unwrap_err();
        assert!(err.contains("NONEXISTENT_VAR_12345"));
    }

    #[test]
    fn test_parse_minimal_config() {
        let toml_str = r#"
[[layers]]
name = "base"
url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
"#;
        let cfg: FileConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.layers.len(), 1);
        assert_eq!(cfg.layers[0].name, "base");
        assert_eq!(cfg.layers[0].kind, "raster");
    }

    #[test]
    fn test_parse_full_config() {
        let toml_str = r#"
[map]
center = [35.0, 127.0]
zoom = 8.0

[[layers]]
name = "imagery"
kind = "raster"
url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

[[layers]]
name = "terrain"
kind = "terrain"
imagery_layer = "imagery"
url = "https://example.com/terrain/{z}/{x}/{y}.webp"
z_order = 10

[[layers]]
name = "buildings"
kind = "3dtiles"
cesium_ion_token = "test_token"
cesium_ion_asset = 96188
z_order = 20
"#;
        let cfg: FileConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.map.center, [35.0, 127.0]);
        assert_eq!(cfg.map.zoom, 8.0);
        assert_eq!(cfg.layers.len(), 3);

        let mc = cfg.into_map_config();
        assert_eq!(mc.layers.len(), 3);
        assert_eq!(mc.layers[2].kind, LayerKind::Tiles3d);
        assert_eq!(
            mc.layers[2].cesium_ion_token.as_deref(),
            Some("test_token")
        );
    }

    #[test]
    fn test_missing_env_skips_layer() {
        let toml_str = r#"
[[layers]]
name = "imagery"
url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

[[layers]]
name = "terrain"
kind = "terrain"
imagery_layer = "imagery"
url = "https://api.example.com/{z}/{x}/{y}.webp?key=${TOTALLY_MISSING_KEY_999}"
"#;
        let cfg: FileConfig = toml::from_str(toml_str).unwrap();
        let mc = cfg.into_map_config();
        // terrain layer skipped because env var is missing
        assert_eq!(mc.layers.len(), 1);
        assert_eq!(mc.layers[0].name, "imagery");
    }
}
