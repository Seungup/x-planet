# Renderer Interface Refactoring Plan

## 현재 상태 분석: 잘 된 코드 vs 문제 코드

---

### GOOD: `traversal.rs` — 참조 품질 코드

```
traversal.rs (346줄, 테스트 345줄 추가)
├── 명확한 타입 정의 (TraversalConfig, TraversalCamera, TraversalResult, TraversalTile, LoadRequest3d)
├── 순수 함수 설계: traverse_tileset() — no side effects, no GPU, no I/O
├── 단일 책임: LOD traversal만 담당
├── 10개 유닛 테스트: edge case 모두 커버
└── 결과 타입이 명확: render_set, load_requests, unload_set
```

**왜 좋은가:**
- Input → Output이 완전히 예측 가능
- GPU/IO에 의존하지 않아 테스트가 쉬움
- `TraversalResult` 하나로 caller가 필요한 모든 정보 전달
- `Refine::Replace` vs `Refine::Add` 분기가 깔끔하게 분리

---

### GOOD: `TerrainRenderer` 메시 캐싱 전략

```
mesh_cache: HashMap<TileCoord, CachedMesh>
├── 메시 빌드: elevation source 변경 시에만
├── uniform buffer: 매 프레임 write_buffer() (재할당 없음)
├── bind group: imagery texture 변경 시에만 재생성
└── cache invalidation: exaggeration/projection/center 변경 시 전체 clear
```

**왜 좋은가:**
- GPU 리소스 재활용이 명확
- 캐시 무효화 조건이 코드에 명시적
- phase 분리: build → update → render

---

### BAD: 3개 렌더러의 보일러플레이트 중복

| 중복 항목 | TileRenderer | TerrainRenderer | Model3dRenderer |
|-----------|:---:|:---:|:---:|
| `depth_format()` | 동일 | 동일 | 동일 |
| `create_depth_texture()` | 동일 | 동일 | 동일 |
| `resize()` | 동일 | 동일 | 동일 |
| viewport BGL 생성 | 동일 | 동일 | 거의 동일 |
| viewport uniform buffer | 동일 | 동일 | 동일 |
| viewport bind group | 동일 | 동일 | 동일 |
| tile/model BGL | 동일 | 동일 | 유사 |
| sampler | 동일 | 동일 | 동일 |

**정확히 동일한 코드가 3번 반복됨:**
```rust
// 이 코드가 tile_renderer.rs, terrain_renderer.rs, model3d_renderer.rs에 각각 존재
fn depth_format() -> wgpu::TextureFormat {
    #[cfg(target_arch = "wasm32")]
    { wgpu::TextureFormat::Depth24Plus }
    #[cfg(not(target_arch = "wasm32"))]
    { wgpu::TextureFormat::Depth32Float }
}

fn create_depth_texture(device, width, height, format) -> TextureView { ... } // 동일
pub fn resize(&mut self, device, width, height) { ... } // 동일
```

**문제:**
- 한 렌더러에서 depth format을 바꾸면 다른 렌더러도 바꿔야 함 (동기화 실패 위험)
- 3개 렌더러가 각자 depth texture를 생성 → **depth buffer가 공유되지 않음**
- 3개 viewport uniform buffer가 독립적으로 업데이트됨 → 매 프레임 3번 `write_buffer()`

---

### BAD: `TileRenderer::render_frame_layered_projected()` — 330줄 모놀리식 메서드

```
render_frame_layered_projected() — 330줄
├── viewport uniform 업데이트
├── if layers.is_empty() → clear pass
├── for layer in layers
│   ├── if layer.tiles.is_empty() && first → clear pass
│   ├── if is_globe
│   │   ├── filter tiles (texture available)
│   │   ├── build_globe_tile_mesh()
│   │   ├── create vertex/index buffers
│   │   ├── prepare_tile_globe() × N
│   │   ├── build polar caps (cached)
│   │   ├── create polar cap bind group
│   │   ├── begin_render_pass()
│   │   ├── draw tiles + draw polar caps
│   │   └── end pass
│   └── else (centered)
│       ├── filter tiles (texture + angular)
│       ├── build_centered_tile_mesh()
│       ├── create vertex/index buffers
│       ├── prepare_tile_centered() × N
│       ├── begin_render_pass()
│       ├── draw tiles
│       └── end pass
└── submit
```

**문제:**
- Globe path와 Centered path가 **90% 동일**하지만 전부 복사됨
- "prepare" 단계와 "render" 단계가 혼재
- polar cap 로직이 렌더 메서드 안에 인라인됨
- 매 프레임 vertex/index buffer를 새로 생성 (TerrainRenderer의 캐싱과 대비)

---

### BAD: 각 렌더러가 독립 depth buffer → z-fighting 위험

```
Frame:
  TileRenderer   → depth clear(1.0) → draw raster tiles
  TerrainRenderer → depth clear(1.0) → draw terrain tiles   ← 자체 depth!
  Model3dRenderer → depth clear(1.0) → draw 3D models       ← 자체 depth!
```

**문제:**
- Raster 위에 Terrain이 올바르게 깊이 테스트를 할 수 없음 (각자 depth buffer)
- 3D 모델이 terrain에 매몰되거나 뜨는 현상 발생 가능

---

### BAD: `tiles3d_frame.rs` — 플랫폼 코드에 코어 로직 혼재

```
tiles3d_traverse_and_render() — native 전용
├── viewport → camera 변환 (코어 로직)
├── traverse_tileset() (코어 로직)
├── load 결정 (코어 로직)
├── tokio::spawn (플랫폼 로직)
├── unload 결정 (코어 로직)
├── transform 업데이트 (코어 로직)
└── render (코어 로직)
```

**문제:**
- 이 로직의 80%가 웹에서도 필요하지만 native에 하드코딩됨
- `tiles3d_pipeline.rs`(코어)에 일부 코어 로직이 있지만 나머지가 native에 분산

---

## 구체적 수정 계획

### Phase 1: SharedRenderResources 추출

**목표:** 3개 렌더러에서 반복되는 GPU 리소스를 하나로 통합

**새 파일:** `crates/x-planets-core/src/shared_render_resources.rs`

```rust
/// 모든 렌더러가 공유하는 GPU 리소스.
/// 프레임당 한 번만 업데이트하면 됨.
pub struct SharedRenderResources {
    // --- Depth ---
    pub depth_view: wgpu::TextureView,
    pub depth_format: wgpu::TextureFormat,
    surface_width: u32,
    surface_height: u32,

    // --- Viewport (Group 0) ---
    pub viewport_bgl: wgpu::BindGroupLayout,
    pub viewport_buffer: wgpu::Buffer,
    pub viewport_bg: wgpu::BindGroup,

    // --- Shared sampler ---
    pub sampler: wgpu::Sampler,
}

impl SharedRenderResources {
    pub fn new(gpu: &GpuContext) -> Self;
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32);
    pub fn update_viewport(&self, gpu: &GpuContext, uniforms: &ViewportUniforms);
    pub fn depth_format() -> wgpu::TextureFormat;
}
```

**변경 대상:**
- `tile_renderer.rs`: `_viewport_bgl`, `viewport_buffer`, `viewport_bg`, `sampler`, `depth_view`, `depth_format`, `surface_width/height` 제거 → `&SharedRenderResources` 참조
- `terrain_renderer.rs`: 동일하게 제거
- `model3d_renderer.rs`: 동일하게 제거

**검증:** `cargo test -p x-planets-core` + 네이티브 앱 실행하여 동일 렌더링 확인

---

### Phase 2: Tile BGL 통합 (TileRenderer + TerrainRenderer)

**현황:** TileRenderer와 TerrainRenderer의 tile_bgl이 100% 동일

```rust
// tile_bgl entries (두 렌더러에서 동일):
// binding 0: uniform buffer (VERTEX | FRAGMENT)
// binding 1: texture 2D (FRAGMENT)
// binding 2: sampler (FRAGMENT)
```

**계획:** `SharedRenderResources`에 `tile_bgl` 추가

```rust
impl SharedRenderResources {
    pub fn tile_bgl(&self) -> &wgpu::BindGroupLayout;  // uniform + texture + sampler
    pub fn model_bgl(&self) -> &wgpu::BindGroupLayout; // uniform + texture + sampler (동일 구조)
}
```

**실제로 tile_bgl과 model_bgl은 동일 구조** → 하나로 통합 가능

---

### Phase 3: TileRenderer 리팩토링 — Globe/Centered 분기 통합

**현황:** `render_frame_layered_projected()` 330줄, Globe/Centered 경로 90% 중복

**계획:** prepare/render 분리 + 공통 경로 추출

```rust
/// 프로젝션별 타일 메시 생성을 추상화
struct PreparedLayer {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    tile_bind_groups: Vec<wgpu::BindGroup>,
    tile_idx_ranges: Vec<Range<u32>>,
    // Globe 전용
    polar_cap: Option<PolarCapData>,
}

impl TileRenderer {
    /// 1. Prepare: 프로젝션에 따라 메시 + bind group 생성
    fn prepare_layer(
        &mut self, gpu, viewport, layer, mode, vp_f64
    ) -> Option<PreparedLayer>;

    /// 2. Render: PreparedLayer를 그리기
    fn render_prepared(
        &self, encoder, target, shared, prepared, is_first, use_globe_pipeline
    );

    /// 3. 공개 API (기존 시그니처 유지)
    pub fn render_frame_layered_projected(
        &mut self, gpu, target, viewport, layers, mode
    ) {
        let shared = ...;
        let prepared_layers = layers.map(|l| self.prepare_layer(...));
        for prepared in prepared_layers {
            self.render_prepared(...);
        }
        submit;
    }
}
```

**효과:** 330줄 → ~150줄, Globe/Centered 공통 로직 50% 감소

---

### Phase 4: TileRenderer에 메시 캐싱 도입

**현황:** TileRenderer는 매 프레임 vertex/index buffer를 재생성
**참조:** TerrainRenderer는 메시를 캐싱하여 elevation 변경 시에만 재빌드

```rust
// TileRenderer 현재 (매 프레임):
let (globe_verts, globe_idxs, tile_idx_counts) = build_globe_tile_mesh(&renderable_refs);
let vertex_buffer = gpu.create_vertex_buffer(..., &globe_verts);  // 매 프레임 할당!

// TerrainRenderer (캐시):
fn get_or_build_mesh(&mut self, ...) -> &CachedMesh;  // 변경 시에만 재빌드
```

**계획:**
```rust
struct TileRenderer {
    // 기존 필드...
    /// 캐시된 타일 메시 (projection + tile set이 변경되지 않으면 재사용)
    mesh_cache: HashMap<TileCoord, CachedTileMesh>,
    cached_projection_mode: ProjectionMode,
    cached_center: Option<(f64, f64)>,
}

struct CachedTileMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    last_texture_coord: Option<TileCoord>,
}
```

**효과:**
- 정적 카메라에서 GPU 버퍼 할당 0 (현재: 매 프레임 N개)
- TerrainRenderer와 동일한 캐싱 패턴 → 코드 일관성

---

### Phase 5: 3D Tiles 코어 로직 추출

**현황:** `tiles3d_frame.rs` (native 전용)에 코어 로직이 하드코딩

**계획:** `crates/x-planets-core/src/tiles3d_manager.rs` 신규 생성

```rust
/// 플랫폼 독립적 3D Tiles 상태 관리.
/// traversal, load/unload 결정, transform 계산을 담당.
pub struct Tiles3dManager {
    tileset: Option<Tileset>,
    base_url: String,
    loaded_models: HashMap<String, GpuModel3d>,
    loaded_uris: HashSet<String>,
    pending_uris: HashSet<String>,
    config: TraversalConfig,
    max_concurrent: usize,
}

/// 프레임별 결정 결과 (플랫폼이 실행할 명령)
pub struct Tiles3dFrameCommands {
    /// 로드해야 할 URI들 (플랫폼이 fetch 수행)
    pub load: Vec<LoadRequest3d>,
    /// 언로드할 URI들
    pub unload: Vec<String>,
    /// 렌더할 모델 + transform
    pub render: Vec<(String, [f32; 16])>,
}

impl Tiles3dManager {
    /// 순수 함수: 현재 상태 + viewport → 프레임 명령
    pub fn frame_update(
        &mut self,
        viewport: &Viewport,
        body: &CelestialBody,
    ) -> Tiles3dFrameCommands;

    /// 로드 완료 콜백
    pub fn on_content_loaded(&mut self, uri: String, model: GpuModel3d);

    /// 언로드 실행
    pub fn execute_unloads(&mut self, uris: &[String]);
}
```

**변경 대상:**
- `tiles3d_frame.rs`: `tiles3d_traverse_and_render()` 분해
  - 코어 로직 → `Tiles3dManager::frame_update()`
  - 플랫폼 로직 (tokio spawn, fetch) → 기존 파일에 유지

**효과:**
- 웹에서도 동일 로직 재사용 가능
- 코어 로직을 유닛 테스트 가능 (현재 불가능)

---

### Phase 6: RendererTrait 도입 (Optional — Phase 1-5 완료 후)

```rust
/// 렌더러가 공유 리소스와 상호작용하는 최소 인터페이스.
pub trait Renderer {
    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32);
}
```

현재는 3개 렌더러의 시그니처가 너무 다르므로 (raster는 RenderLayerData, terrain은 TerrainLayerData, 3D tiles는 자체 관리) 강제적인 trait 통합은 오히려 해로울 수 있음. Phase 1-5로 충분한 개선이 달성되면 trait은 선택적으로 도입.

---

## 수정 순서 및 검증

| Phase | 파일 변경 | 검증 방법 | 예상 효과 |
|-------|----------|----------|----------|
| 1 | +shared_render_resources.rs, ~tile_renderer.rs, ~terrain_renderer.rs, ~model3d_renderer.rs | `cargo test` + 앱 실행 | 중복 ~200줄 제거, depth buffer 공유 |
| 2 | ~shared_render_resources.rs, ~tile_renderer.rs, ~terrain_renderer.rs, ~model3d_renderer.rs | `cargo test` + 앱 실행 | BGL 중복 제거 |
| 3 | ~tile_renderer.rs | `cargo test` + Globe/Centered 모드 전환 테스트 | 330줄 → ~150줄 |
| 4 | ~tile_renderer.rs | 프로파일링 (GPU 버퍼 할당 수 비교) | 매 프레임 할당 → 캐시 히트 |
| 5 | +tiles3d_manager.rs, ~tiles3d_frame.rs, ~mod.rs | `cargo test -p x-planets-core` | 코어 로직 테스트 가능, 웹 재사용 |
| 6 | ~all renderers | `cargo test` | 인터페이스 통일 (선택적) |

---

## 기술 면접 대비 포인트

### Q: "왜 trait으로 추상화하지 않았나?"
A: 3개 렌더러의 render 시그니처가 근본적으로 다름 (raster: tile coords + textures, terrain: elevation + imagery, 3D tiles: ECEF models). 무리한 trait 통합은 타입 안전성을 해치고 dynamic dispatch 오버헤드만 추가. 대신 **공유 리소스를 composition으로 추출** (SharedRenderResources)하여 중복을 제거하되 각 렌더러의 전문성은 유지.

### Q: "depth buffer를 왜 공유해야 하나?"
A: 현재 3개 렌더러가 각자 depth buffer를 소유하면 raster → terrain → 3D 간 깊이 비교가 불가능. 예: terrain이 raster 위에 올바르게 그려지려면 같은 depth buffer를 써야 함. 공유하면 `LoadOp::Load`로 이전 패스의 depth를 유지하면서 정확한 occlusion이 가능.

### Q: "TileRenderer는 왜 매 프레임 버퍼를 재생성하나? TerrainRenderer처럼 캐싱하면?"
A: 맞음. 현재 TileRenderer는 visible tile set이 매 프레임 달라질 수 있어서 단순하게 재생성했지만, 실제로 정적 카메라에서는 99%가 같은 타일 → 캐시 히트. TerrainRenderer의 `get_or_build_mesh()` 패턴을 TileRenderer에도 적용하면 GPU 메모리 할당이 크게 줄어듦.

### Q: "tiles3d_frame.rs의 코어 로직을 왜 분리해야 하나?"
A: platform-agnostic core 원칙 위반. traversal 결정, load/unload 판단은 플랫폼에 무관한 순수 로직. native에 하드코딩하면 web에서 동일 코드를 재작성해야 함. `Tiles3dManager`로 추출하면 native/web 모두 `frame_update()` → 플랫폼별 fetch만 구현.

### Q: "ProjectionMode별 분기가 왜 문제인가?"
A: `render_frame_layered_projected()`에서 Globe/Centered의 코드가 90% 동일한데 전체가 if-else로 복사됨. 새 프로젝션 추가 시 또 다시 복사해야 함. prepare/render 분리 후 프로젝션별 차이는 mesh builder만 교체하면 됨 (Strategy 패턴).

### Q: "bind group layout이 왜 중복되면 안 되나?"
A: wgpu에서 bind group layout은 pipeline layout에 바인딩됨. 렌더러 간 동일 layout이지만 별도 객체면 pipeline compatibility가 깨져 bind group을 공유할 수 없음. 통합하면 이론적으로 bind group 재사용도 가능해짐.
