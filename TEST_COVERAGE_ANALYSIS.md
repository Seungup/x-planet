# Test Coverage Analysis

**Date**: 2026-03-04
**Total tests**: 324 (all passing)
**Test modules**: 26 files with `#[cfg(test)]`

## Test Distribution by Crate

| Crate | Tests | Files with Tests | Files Without Tests | Assessment |
|-------|-------|-----------------|--------------------|----|
| `x-planets-core` | 168 | 7/10 | 3 (GPU renderers) | Strong core logic, no GPU tests |
| `x-planets-tiles` | 81 | 10/12 | 2 (mod files only) | Good structural tests, weak integration |
| `x-planets-math` | 28 | 2/2 | 0 | Decent but missing edge cases |
| `x-planets-native` | 27 | 2/14 | 12 | Very weak — app logic untested |
| `x-planets-projection` | 13 | 2/3 | 1 | Basic coverage only |
| `x-planets-gpu` | 6 | 1/5 | 4 | Nearly untested |
| `x-planets-web` | 0 | 0/3 | 3 | Completely untested |
| `x-planets-examples` | 1 | — | — | Verify chain only |

---

## Priority 1: Critical Gaps (High Impact, Feasible to Test)

### 1.1 Camera & Interaction — `x-planets-core` viewport/interaction/engine

**Current state**: Camera pan/zoom/pitch/rotate methods have zero unit tests. The `AnimationController::tick()` method (which drives all inertia and zoom animation) is completely untested.

**What to add**:
- `CameraController::pan()` — verify lat/lon delta, antimeridian wrapping via `rem_euclid`
- `CameraController::zoom()` / `zoom_at()` — verify zoom clamping to `[0, 22]`, anchor point behavior
- `CameraController::pan_globe()` — verify quaternion-based pan on the sphere
- `AnimationController::tick()` — verify inertia decay converges, zoom animation reaches target
- `AnimationController::is_animating()` — verify returns false when idle
- `MapEngine::pan()`, `zoom()`, `pitch()`, `rotate()` — verify they delegate and set redraw flag

**Why it matters**: These are the most user-facing functions. A regression in pan/zoom math would be immediately visible and hard to diagnose without tests.

**Files**: `crates/x-planets-core/src/viewport.rs` (lines 676-900), `crates/x-planets-core/src/interaction.rs` (lines 110-200), `crates/x-planets-core/src/engine.rs` (lines 201-250)

---

### 1.2 Oblique Mercator Projection — `x-planets-math`

**Current state**: `oblique_mercator()` and `oblique_mercator_inverse()` have zero tests. These are used for the "centered" rendering mode.

**What to add**:
- Roundtrip test: `oblique_mercator_inverse(oblique_mercator(p, center), center) ≈ p`
- Known values at various center points (Seoul, New York, equator, near-polar)
- Edge cases: center at poles, center at antimeridian (±180°)

**Files**: `crates/x-planets-math/src/lib.rs` (lines 398-464)

---

### 1.3 WGSL Shader Validation — `x-planets-gpu`

**Current state**: `validate_wgsl()` is completely untested. The pipeline builders (`RenderPipelineBuilder`, `ComputePipelineBuilder`) have zero tests.

**What to add**:
- `validate_wgsl()` with valid shader → Ok
- `validate_wgsl()` with syntax error → Err
- `validate_wgsl()` with semantic error → Err
- Validate all shipped WGSL shaders in `shaders/` compile without error (prevents shader regressions)

**Why it matters**: Shader validation uses `naga` and doesn't need a GPU device — purely CPU testable. A broken shader would crash the app at startup.

**Files**: `crates/x-planets-gpu/src/pipeline.rs` (lines 260-273)

---

### 1.4 Tile Loading Pipeline — `x-planets-native`

**Current state**: The entire application layer (`app/mod.rs`, `app/tile_loading.rs`, `app/tile_upload.rs`, `app/render_layers.rs`, `app/input.rs`) has zero tests. This is 12 files with no test modules.

**What to add (non-GPU, pure logic)**:
- `AnimationState` inertia physics — verify drag velocity estimation, zoom damping
- Config `expand_env()` edge cases — multiple `${VAR}` in one string, unclosed `${`
- `NativeTileSource::tile_url()` — verify URL template substitution with all tile coordinate systems
- `TileJSON` resolution — mock HTTP responses, verify parsing of 2.x/3.x formats
- Geographic tile coordinate conversion edge cases — tiles at poles, antimeridian

**Files**: `crates/x-planets-native/src/animation.rs`, `crates/x-planets-native/src/config.rs`, `crates/x-planets-native/src/tilejson.rs`

---

## Priority 2: Important Gaps (Medium Impact)

### 2.1 Globe/Centered Rendering Pipeline — `x-planets-core` pipeline.rs

**Current state**: Pipeline functions `tile_uniforms_for_globe()`, `tile_uniforms_for_centered()`, `build_globe_tile_mesh()`, `build_centered_tile_mesh()` lack direct unit tests. The existing 60+ pipeline tests focus almost entirely on the Mercator projection mode.

**What to add**:
- `build_globe_tile_mesh()` — verify vertices lie on the unit sphere, correct winding
- `build_centered_tile_mesh()` — verify oblique Mercator positions, vertex centering
- `tile_uniforms_for_globe()` — verify ECEF→VP matrix, depth bias computation
- `tile_passes_angular_filter()` — verify tiles beyond 90° from center are culled
- `centered_angular_threshold_deg()` — verify threshold scales with zoom

**Files**: `crates/x-planets-core/src/pipeline.rs` (lines 237-420)

---

### 2.2 Quantized Mesh Parsing Edge Cases — `x-planets-tiles`

**Current state**: Basic parsing tested but critical code paths are untested:
- Extension parsing (oct-encoded normals) at lines 374-406 has no coverage
- u32 index path (65536+ vertices) at lines 152-169 is untested
- Edge indices (west/south/east/north skirts) are parsed but never verified

**What to add**:
- Test with a tile containing >65535 vertices (u32 indices)
- Test extension block parsing (type=1 oct-encoded normals)
- Test edge index arrays are correctly extracted
- Test truncation errors at various points in the binary format

**Files**: `crates/x-planets-tiles/src/quantized_mesh.rs` (lines 152-406)

---

### 2.3 glTF/B3DM Mesh Extraction — `x-planets-tiles`

**Current state**: `extract_meshes_from_glb()` — the core function for 3D Tiles rendering — is completely untested. All accessor reading functions (`read_vec3_accessor`, `read_indices_accessor`, etc.) lack tests.

**What to add**:
- Create a minimal valid GLB binary in test data and verify mesh extraction
- Test all four image format conversions (RGBA, RGB, R8, RG8)
- Test `extract_cesium_rtc()` center extraction
- Test primitives without indices (sequential index generation)
- Test primitives without normals (auto-generation path)

**Files**: `crates/x-planets-tiles/src/tiles3d/gltf_mesh.rs` (lines 1-425)

---

### 2.4 `VisibleTile` and Antimeridian Wrapping — `x-planets-math`

**Current state**: `VisibleTile` has zero dedicated tests. Its `canonical()`, `display_mercator_center()`, and `children()` methods are never directly tested. The antimeridian wrapping logic (negative `display_x`, `rem_euclid`) has no coverage.

**What to add**:
- `VisibleTile::canonical()` — verify `display_x == coord.x` for non-wrapped tiles
- `VisibleTile::children()` — verify 4 children with correct `display_x` offsets
- Wrapping: tiles with `display_x = -1`, `display_x = extent` (across world boundary)
- `display_mercator_center()` — verify centers for wrapped tiles differ from canonical

**Files**: `crates/x-planets-math/src/lib.rs` (lines 183-236)

---

### 2.5 `TileCache` Missing API Coverage — `x-planets-tiles`

**Current state**: `remove()`, `contains()`, `len()`, `is_empty()` are all untested. Updating an existing entry (inserting same key twice) has no test.

**What to add**:
- `remove()` — verify removes entry and reduces len
- `contains()` — verify true/false cases
- `is_empty()` — verify before/after insert
- Duplicate insert — verify value is updated, not duplicated
- Cache with capacity=1 — verify single-entry behavior
- Cache with capacity=0 — verify edge case handling

**Files**: `crates/x-planets-tiles/src/cache.rs` (lines 75-110)

---

### 2.6 Projection Registry Gaps — `x-planets-projection`

**Current state**: `compose_shaders()` is completely untested. `ProjectionRegistry::empty()`, `list()`, `is_empty()` lack tests. `CustomProjection` builder methods (`with_epsg()`, `with_function_name()`) are untested.

**What to add**:
- `compose_shaders()` — base only, base + transforms, empty transforms
- `ProjectionRegistry::empty()` — verify no built-ins
- `ProjectionRegistry::list()` — verify names returned
- `CustomProjection::with_epsg().with_function_name()` — verify builder chain
- Register duplicate name — verify overwrite behavior

**Files**: `crates/x-planets-projection/src/lib.rs` (lines 84-100), `crates/x-planets-projection/src/registry.rs` (lines 25-60)

---

## Priority 3: Lower Impact / Harder to Test

### 3.1 GPU Renderer Tests — `x-planets-core` renderers

The three renderer files (`tile_renderer.rs`, `terrain_renderer.rs`, `model3d_renderer.rs`) have zero tests. These require a GPU context and are inherently harder to unit test.

**Recommendation**: Use `GpuContext::new_headless()` (already exists) to create GPU snapshot tests. The `test_utils.rs` module already provides `checkerboard_rgba()`, `PixelDiff`, and `save_rgba_png()` infrastructure for this. A single integration test that renders one tile with a checkerboard texture and validates the output would catch pipeline setup regressions.

---

### 3.2 Web Crate — `x-planets-web`

Zero tests across all 3 files. The WASM target makes standard unit testing difficult, but the pure logic (URL building, tile coordinate math) could be tested under `#[cfg(not(target_arch = "wasm32"))]` or with `wasm-pack test`.

---

### 3.3 Native App Integration — `x-planets-native`

No end-to-end tests exist. Input handling (keyboard, mouse, touch) is completely untested. The tile loading pipeline orchestration (`tile_loading.rs`, `tile_upload.rs`) involves complex state machines with zero coverage.

---

## Summary of Recommendations

| Priority | Area | Effort | Impact |
|----------|------|--------|--------|
| **P1** | Camera pan/zoom/pitch unit tests | Low | High |
| **P1** | Oblique Mercator roundtrip tests | Low | High |
| **P1** | WGSL shader validation + shipped shader tests | Low | High |
| **P1** | Native animation/config edge cases | Medium | High |
| **P2** | Globe/Centered pipeline unit tests | Medium | Medium |
| **P2** | Quantized Mesh edge cases (u32, extensions) | Medium | Medium |
| **P2** | glTF mesh extraction with test data | High | Medium |
| **P2** | VisibleTile antimeridian wrapping | Low | Medium |
| **P2** | TileCache missing API methods | Low | Low |
| **P2** | Projection registry + compose_shaders | Low | Low |
| **P3** | GPU snapshot rendering tests | High | Medium |
| **P3** | WASM/web crate tests | High | Low |
| **P3** | Native app integration tests | High | Low |

The most impactful improvements with the least effort are:
1. **Camera/interaction unit tests** — pure math, no GPU needed, high regression risk
2. **Shader validation tests** — CPU-only via naga, catches build-breaking regressions
3. **Oblique Mercator tests** — simple roundtrip, protects centered rendering mode
4. **VisibleTile wrapping tests** — small surface area, protects antimeridian correctness
