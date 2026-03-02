//! x-planets native viewer binary.
//!
//! Loads `.env` for API keys, then reads `config.toml` for the layer stack.
//! String values in config support `${ENV_VAR}` substitution — layers with
//! missing env vars are silently skipped so the app still launches.
//!
//! Usage:
//!   cargo run -p x-planets-native
//!
//! With a custom config:
//!   cargo run -p x-planets-native -- --config my_config.toml

use std::path::PathBuf;

fn main() {
    // ── Load .env file (if present) ──
    match dotenvy::dotenv() {
        Ok(path) => eprintln!("Loaded env from: {}", path.display()),
        Err(dotenvy::Error::Io(_)) => {} // .env not found — fine
        Err(e) => eprintln!("Warning: .env parse error: {}", e),
    }

    // ── Determine config path ──
    let config_path = std::env::args()
        .skip_while(|a| a != "--config")
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config.toml"));

    eprintln!("Loading config from: {}", config_path.display());

    let map_config = match x_planets_native::config::FileConfig::load(&config_path) {
        Ok(fc) => fc.into_map_config(),
        Err(e) => {
            eprintln!("Warning: {} — using defaults", e);
            Default::default()
        }
    };

    if let Err(e) = x_planets_native::run_native(map_config) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
