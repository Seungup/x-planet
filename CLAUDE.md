# CLAUDE.md

## Project Overview

**x-planet** is a cross-platform (native + WASM) map rendering engine written in Rust. It renders raster tiles, 3D terrain meshes, and OGC 3D Tiles (glTF/B3DM) using wgpu. The project follows the **Karpathy Method**: start with the simplest working end-to-end, verify at each step, add one thing at a time.

- **License**: MIT OR Apache-2.0
- **Rust edition**: 2021 (stable toolchain)
- **GPU API**: wgpu 23

## Quick Reference

```bash
# Build (native)
cargo build -p x-planets-native

# Run the native viewer
cargo run -p x-planets-native

# Run with custom config
cargo run -p x-planets-native -- --config my_config.toml

# Run tests (all crates)
cargo test

# Run tests for a specific crate
cargo test -p x-planets-math
cargo test -p x-planets-tiles
cargo test -p x-planets-core

# Check all targets compile (including WASM)
cargo check --target wasm32-unknown-unknown -p x-planets-web

# Run the verification chain
cargo run --example verify -p x-planets-examples

# Run step-by-step examples
cargo run --example step00_triangle -p x-planets-examples
cargo run --example step01_colored_quad -p x-planets-examples
cargo run --example step02_textured_quad -p x-planets-examples
```

## Workspace Architecture

```
x-planet/
├── Cargo.toml              # Workspace root (resolver = "2")
├── config.toml             # Runtime config (map center, layers, API keys)
├── rust-toolchain.toml     # Stable toolchain + wasm32 target
├── .cargo/config.toml      # Build parallelism (jobs = 8)
├── shaders/                # WGSL shaders
│   ├── projections/        # mercator.wgsl, equirectangular.wgsl
│   └── rendering/          # raster_tile.wgsl, terrain_tile.wgsl, model3d.wgsl
├── examples/               # Symlinked example binaries
└── crates/                 # 8 workspace members
```

### Crate Dependency Graph

```
x-planets-math              ← No internal deps (foundation)
    ↑
x-planets-gpu               ← wgpu, bytemuck, naga
    ↑
x-planets-projection        ← math, gpu
    ↑
x-planets-tiles             ← math, image, gltf, async-trait
    ↑
x-planets-core              ← math, gpu, projection, tiles
    ↑               ↑
x-planets-native    x-planets-web
(winit, tokio,      (wasm-bindgen,
 reqwest)            web-sys)

x-planets-examples          ← all above (step-by-step tutorials)
```

### Crate Responsibilities

| Crate | Purpose |
|-------|---------|
| `x-planets-math` | Geographic types (`GeoCoord`, `TileCoord`, `BoundingBox`), Mercator math, ECEF transforms, frustum culling (SAT), GPU uniform structs |
| `x-planets-gpu` | wgpu wrapper: `GpuContext`, `TextureManager`, `RenderPipelineBuilder`, `ComputePipelineBuilder`, WGSL validation (naga) |
| `x-planets-projection` | `ProjectionPlugin` trait, built-in Mercator/Equirectangular, `ProjectionRegistry` for runtime registration |
| `x-planets-tiles` | `TileSource`/`TileDecoder` traits, raster/terrain decoders, LRU `TileCache`, Quantized Mesh parser, OGC 3D Tiles (B3DM, tileset, glTF) |
| `x-planets-core` | `MapEngine`, `Viewport`, `CameraController`, renderers (`TileRenderer`, `TerrainRenderer`, `Model3dRenderer`), pure-function pipeline stages, verification chain |
| `x-planets-native` | Desktop app: winit event loop, tokio runtime, reqwest HTTP, `config.toml` parsing with `${ENV_VAR}` substitution |
| `x-planets-web` | WASM entry point: `wasm_bindgen`, Fetch API tile source, canvas integration (WIP) |
| `x-planets-examples` | Incremental step-by-step examples (step00–step09) following the Karpathy method |

## Key Traits and Interfaces

### `ProjectionPlugin` (`x-planets-projection`)
```rust
pub trait ProjectionPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn shader_source(&self) -> &str;           // WGSL snippet with fn project(vec3<f32>) -> vec3<f32>
    fn project_cpu(&self, world_pos: DVec3) -> DVec3;   // CPU fallback
    fn unproject_cpu(&self, projected: DVec3) -> DVec3;
}
```

### `TileSource` (`x-planets-tiles`)
```rust
// Native: Send + Sync; WASM: no Send/Sync
pub trait TileSource: Send + Sync {
    async fn fetch(&self, coord: TileCoord) -> Result<Vec<u8>, LoadError>;
    fn tile_url(&self, coord: &TileCoord) -> String;
}
```

### `TileDecoder` (`x-planets-tiles`)
```rust
pub trait TileDecoder: Send + Sync {
    type Output: Send + Sync;
    async fn decode(&self, coord: TileCoord, data: &[u8]) -> Result<Self::Output, DecodeError>;
    fn extension(&self) -> &str;
}
```

## Coding Conventions

### Architecture Principles
- **Pure-function pipeline**: `pipeline.rs` stages are input-output with no side effects, independently testable
- **Platform-agnostic core**: `x-planets-core` has no platform dependencies; platform code lives in `x-planets-native` / `x-planets-web`
- **Verify at each step**: the verification chain (`verify_chain.rs`) validates each pipeline stage sequentially

### Naming
- `GeoCoord` = WGS84 lat/lon in degrees
- `TileCoord` = XYZ tile scheme (z/x/y)
- `Mercator` = Web Mercator normalized coordinates [0..1]
- `RTE` = Relative-To-Eye / Relative-To-Tile-center (vertex positions baked with tile offset in f64 to avoid f32 jitter)
- `MVP` = Model-View-Projection matrix
- `VP` = View-Projection (viewport-level)
- `vp_f64` = f64 view-projection computed on CPU, per-tile MVP = `VP_f64 * translate(tile_center_f64)` then cast to f32
- `bgl` = bind group layout

### Error Handling
- Use `thiserror` for custom error enums with `#[from]` derives
- Error types: `GpuError`, `LoadError`, `DecodeError`
- Layers with missing env vars are silently skipped (app still launches)

### Async Patterns
- `async-trait` with conditional bounds: `Send + Sync` on native, `(?Send)` on wasm32
- Native: `tokio` multi-threaded runtime + `mpsc` channels for tile loading
- WASM: single-threaded browser event loop + `wasm-bindgen-futures`

### Conditional Compilation
```rust
#[cfg(not(target_arch = "wasm32"))]  // Native-only code
#[cfg(target_arch = "wasm32")]        // WASM-only code
```

### GPU Data
- GPU-safe structs: `#[repr(C)]` + `bytemuck::Pod + Zeroable`
- Uniform structs use `[f32; N]` arrays (not glam types directly)
- Per-frame updates via `queue.write_buffer()` — no reallocation
- Per-tile bind groups (one texture + uniforms per tile, not atlased)
- Depth bias for LOD: `(22 - zoom) * 0.0001` so finer tiles render on top

### High-Precision Math
- Viewport matrix computed in f64 on CPU
- Per-tile MVP: bake tile-center translation in f64 before f32 cast — eliminates jitter at zoom 18+
- All geographic math uses f64; only final GPU uploads use f32

### Testing
- Unit tests colocated in source files under `#[cfg(test)] mod tests`
- Async decoder tests use `#[tokio::test]`
- Verification chain (`verify_chain.rs`) runs 9 sequential pipeline validation steps
- GPU test utilities in `x-planets-gpu/src/test_utils.rs` for snapshot testing

## Configuration

### `config.toml` Format
```toml
[map]
center = [37.5665, 126.9780]   # [lat, lon] in degrees
zoom   = 5.0
# projection = "Web Mercator"  # default
# terrain_exaggeration = 1.5   # default

[[layers]]
name = "imagery"
kind = "raster"                 # "raster" | "terrain" | "3dtiles"
url  = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

[[layers]]
name             = "terrain"
kind             = "terrain"
imagery_layer    = "imagery"    # raster layer draped onto terrain
terrain_encoding = "quantized-mesh"  # "mapbox" | "terrarium" | "quantized-mesh"
url              = "https://api.maptiler.com/tiles/terrain-quantized-mesh-v2/tiles.json?key=${MAPTILER_KEY}"
```

### Environment Variables
API keys are injected via `${ENV_VAR}` substitution in config strings. Use a `.env` file (loaded via `dotenvy`) or set them in the shell:
- `MAPTILER_KEY` — MapTiler terrain/imagery tiles
- `MAPBOX_TOKEN` — Mapbox terrain RGB tiles
- `CESIUM_ION_TOKEN` — Cesium Ion 3D Tiles (e.g., OSM Buildings)
- `GOOGLE_3DTILES_KEY` — Google Photorealistic 3D Tiles

## Rendering Pipeline

Each frame follows this flow:

1. **Viewport** → camera position (center, zoom, pitch, bearing)
2. **Frustum culling** → visible tile set (AABB fast path, SAT polygon for rotated views)
3. **Quadtree LOD** → multi-zoom tile selection (near=fine, far=coarse, budget=150 tiles)
4. **Load requests** → priority queue (closer tiles first), async fetch + decode
5. **GPU upload** → decoded pixels/meshes → wgpu textures/buffers
6. **Render** → per-layer: bind viewport uniforms (group 0), iterate tiles, bind per-tile uniforms + texture (group 1), draw

### Layer Kinds
- **Raster**: flat textured quads (raster_tile.wgsl)
- **Terrain**: displaced 33x33 vertex grid or Quantized Mesh + draped imagery + hillshade (terrain_tile.wgsl)
- **3D Tiles**: ECEF-positioned glTF/B3DM models with directional lighting (model3d.wgsl)

## WGSL Shaders

All shaders live in `shaders/` and use two bind groups:
- **Group 0**: `ViewportUniforms` (view_proj matrix, resolution, camera)
- **Group 1**: per-tile data (TileUniforms/ModelUniforms + texture + sampler)

| Shader | Purpose |
|--------|---------|
| `projections/mercator.wgsl` | Web Mercator projection |
| `projections/equirectangular.wgsl` | Plate Carree projection |
| `rendering/raster_tile.wgsl` | Textured quad with UV sub-rect and opacity |
| `rendering/terrain_tile.wgsl` | Displaced mesh with hillshade lighting |
| `rendering/model3d.wgsl` | glTF/B3DM with directional lighting |

## Important Implementation Details

- The `TileLoader` uses a `BinaryHeap` priority queue with configurable max concurrent loads (default 6 per layer)
- `TileCache<T>` is an LRU cache using access counters for eviction
- Terrain supports three encodings: Mapbox RGB, AWS Terrarium, Quantized Mesh 1.0
- Quantized Mesh tiles provide pre-built triangle meshes (variable density), decoded in `quantized_mesh.rs`
- 3D Tiles integration supports both Cesium Ion and Google 3D Tiles endpoints
- The native app uses `dotenvy` to load `.env` files; layers with unresolved `${ENV}` vars are skipped gracefully
- Build parallelism is capped at 8 jobs (`.cargo/config.toml`) to avoid OOM on release builds

## What NOT To Do

- Do not add `Cargo.lock` to git (it is gitignored — this is a library-heavy workspace)
- Do not hard-code API keys in source or config files; always use `${ENV_VAR}` substitution
- Do not break the crate dependency DAG (math → gpu → projection → tiles → core → native/web)
- Do not add `Send + Sync` bounds to WASM-targeted trait impls
- Do not use f32 for geographic/viewport math on the CPU side — use f64 to avoid precision loss
- Do not commit `.env` files (gitignored)
