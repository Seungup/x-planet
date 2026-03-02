# x-planets: 카파시 방법론으로 고도화하기

## 카파시의 원칙

Andrej Karpathy가 micrograd, nanoGPT, minbpe 등을 만들 때 일관되게 쓰는 패턴이 있다:

1. **가장 작은 동작하는 단위부터 시작** — "Hello World"가 아니라 "가장 간단한 end-to-end"
2. **매 단계마다 눈으로 확인** — printf 디버깅, 시각화, assert
3. **CPU에서 먼저 검증하고, 그 다음 GPU** — CPU 레퍼런스가 truth
4. **복잡성을 하나씩만 추가** — 두 가지를 동시에 바꾸지 않음
5. **수치적으로 검증** — "대충 맞는 것 같다"가 아니라 오차 범위를 명시

이 프로젝트에 적용하면: **각 기능을 순수 함수로 분리하고, CPU 레퍼런스 구현을 먼저 만들고, GPU 결과를 CPU와 비교 검증하는 구조**가 필요하다.

---

## 현재 아키텍처의 문제점

### 문제 1: 테스트 불가능한 GPU 코드

현재 `GpuContext`는 실제 GPU가 필요함. CI/headless 환경에서 테스트 불가.

```rust
// 현재 — GPU 없으면 테스트 불가
pub async fn new_headless() -> Result<Self, GpuError> {
    let adapter = instance.request_adapter(...).await  // GPU 필요!
        .ok_or(GpuError::AdapterNotFound)?;
}
```

### 문제 2: CPU-GPU 동일성 보장 없음

`ProjectionPlugin`에 `project_cpu()`와 `shader_source()`가 있지만, 둘이 같은 결과를 내는지 검증하는 구조가 없음.

### 문제 3: 큰 단위의 통합

`MapEngine`이 너무 많은 것을 한꺼번에 엮고 있음. 개별 파이프라인 단계를 독립적으로 테스트하기 어려움.

### 문제 4: 시각적 회귀 테스트 없음

타일 렌더링 결과가 "맞는지" 확인할 방법이 없음. 사람이 눈으로 봐야 함.

---

## 아키텍처 변경안

### 변경 1: 테스트 가능한 GPU 추상화 (TestGpuContext)

```
변경 파일: crates/x-planets-gpu/src/context.rs
새 파일:   crates/x-planets-gpu/src/test_utils.rs
```

핵심: GPU 연산을 trait으로 추상화하고, 테스트용 CPU 폴백을 제공.

```rust
// trait으로 추상화
pub trait GpuBackend: Send + Sync {
    fn create_texture(&self, width: u32, height: u32, data: &[u8]) -> TextureHandle;
    fn read_texture(&self, handle: &TextureHandle) -> Vec<u8>;
    fn dispatch_compute(&self, pipeline: &ComputeTask) -> Vec<u8>;
    fn render_to_texture(&self, task: &RenderTask) -> Vec<u8>;
}

// 실제 GPU
pub struct WgpuBackend { device: wgpu::Device, queue: wgpu::Queue }

impl GpuBackend for WgpuBackend {
    fn dispatch_compute(&self, task: &ComputeTask) -> Vec<u8> {
        // 실제 wgpu compute dispatch
    }
}

// 테스트용 CPU 폴백
pub struct CpuBackend;

impl GpuBackend for CpuBackend {
    fn dispatch_compute(&self, task: &ComputeTask) -> Vec<u8> {
        // CPU에서 동일한 연산 수행
        // 이것이 "ground truth"
    }
}
```

이러면 GPU 없이도 모든 로직을 테스트할 수 있고, GPU 결과를 CPU와 비교할 수도 있음.

### 변경 2: CPU-GPU 동일성 테스트 프레임워크

```
새 파일: crates/x-planets-gpu/src/verify.rs
```

```rust
/// CPU 결과와 GPU 결과를 비교하여 동일성 검증
pub struct CpuGpuVerifier {
    cpu: CpuBackend,
    gpu: WgpuBackend,
    tolerance: f32,       // 허용 오차 (float 정밀도 차이)
}

impl CpuGpuVerifier {
    /// 프로젝션 함수의 CPU-GPU 동일성 검증
    pub async fn verify_projection(
        &self,
        plugin: &dyn ProjectionPlugin,
        test_points: &[DVec3],
    ) -> VerifyResult {
        // 1. CPU로 계산
        let cpu_results: Vec<DVec3> = test_points.iter()
            .map(|p| plugin.project_cpu(*p))
            .collect();

        // 2. GPU compute shader로 계산
        let gpu_results = self.gpu.dispatch_compute(&ComputeTask {
            shader: plugin.shader_source(),
            input: test_points,
        });

        // 3. 비교
        let max_error = cpu_results.iter().zip(gpu_results.iter())
            .map(|(c, g)| (c - g).length())
            .fold(0.0f64, f64::max);

        VerifyResult {
            passed: max_error < self.tolerance as f64,
            max_error,
            num_points: test_points.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_mercator_cpu_gpu_match() {
        let verifier = CpuGpuVerifier::new_or_skip(); // GPU 없으면 skip
        let proj = Mercator;

        // 전 세계 격자점에서 테스트
        let test_points = generate_grid_points(-85.0, 85.0, -180.0, 180.0, 100);

        let result = verifier.verify_projection(&proj, &test_points).await;
        assert!(result.passed, "Max error: {}", result.max_error);
    }
}
```

### 변경 3: 파이프라인을 순수 함수 단계로 분해

현재 `MapEngine::update()`가 하는 일을 각각 독립 함수로 분리.

```
변경 파일: crates/x-planets-core/src/engine.rs
새 파일:   crates/x-planets-core/src/pipeline.rs
```

```rust
// 현재: MapEngine 안에 모든 것이 뭉쳐있음
// 변경: 각 단계를 순수 함수로 분리

/// Stage 1: 뷰포트 → 필요한 타일 목록 결정
/// 순수 함수. 부수효과 없음. 독립 테스트 가능.
pub fn determine_visible_tiles(
    viewport: &Viewport,
    zoom: u8,
) -> Vec<TileCoord> {
    viewport.frustum().visible_tiles(zoom)
}

/// Stage 2: 필요한 타일 vs 캐시된 타일 → 로딩 필요한 타일
/// 순수 함수.
pub fn compute_tile_requests(
    visible: &[TileCoord],
    cached: &HashSet<TileCoord>,
    center: &GeoCoord,
) -> Vec<TileRequest> {
    visible.iter()
        .filter(|t| !cached.contains(t))
        .map(|t| {
            let bounds = t.to_geo_bounds();
            let c = bounds.center();
            let dx = c.lon - center.lon;
            let dy = c.lat - center.lat;
            TileRequest {
                coord: *t,
                priority: (dx * dx + dy * dy).sqrt() as f32,
            }
        })
        .collect()
}

/// Stage 3: 타일 좌표 → GPU 버텍스 데이터
/// 순수 함수.
pub fn build_tile_geometry(
    tiles: &[TileCoord],
) -> (Vec<TileVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for (i, tile) in tiles.iter().enumerate() {
        let base = (i * 4) as u32;
        let quad = tile_quad_vertices(tile);
        vertices.extend_from_slice(&quad);
        for idx in TILE_QUAD_INDICES {
            indices.push(base + idx);
        }
    }

    (vertices, indices)
}

/// Stage 4: 뷰포트 → 유니폼 버퍼 데이터
/// 순수 함수.
pub fn compute_viewport_uniforms(
    viewport: &Viewport,
) -> ViewportUniforms {
    viewport.to_uniforms()
}
```

각 함수가 순수하므로 **입력만 주면 출력이 결정적**. 테스트가 쉬움.

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_determine_visible_tiles_zoom_0() {
        let vp = Viewport::new(800, 600);
        let tiles = determine_visible_tiles(&vp, 0);
        assert_eq!(tiles.len(), 1); // zoom 0 = 전 세계 1타일
        assert_eq!(tiles[0], TileCoord::new(0, 0, 0));
    }

    #[test]
    fn test_compute_tile_requests_filters_cached() {
        let visible = vec![
            TileCoord::new(2, 0, 0),
            TileCoord::new(2, 1, 0),
            TileCoord::new(2, 1, 1),
        ];
        let mut cached = HashSet::new();
        cached.insert(TileCoord::new(2, 1, 0)); // 이미 캐시됨

        let requests = compute_tile_requests(
            &visible, &cached, &GeoCoord::new(0.0, 0.0)
        );

        assert_eq!(requests.len(), 2); // 캐시된 1개 제외
        assert!(requests.iter().all(|r| r.coord != TileCoord::new(2, 1, 0)));
    }

    #[test]
    fn test_build_tile_geometry_correctness() {
        let tiles = vec![TileCoord::new(1, 0, 0)];
        let (verts, indices) = build_tile_geometry(&tiles);

        assert_eq!(verts.len(), 4);   // 1 quad = 4 vertices
        assert_eq!(indices.len(), 6); // 1 quad = 2 triangles = 6 indices

        // 버텍스가 올바른 범위에 있는지 확인
        for v in &verts {
            assert!(v.position[0] >= 0.0 && v.position[0] <= 1.0);
            assert!(v.position[1] >= 0.0 && v.position[1] <= 1.0);
            assert!(v.tex_coord[0] >= 0.0 && v.tex_coord[0] <= 1.0);
            assert!(v.tex_coord[1] >= 0.0 && v.tex_coord[1] <= 1.0);
        }
    }

    #[test]
    fn test_tile_requests_sorted_by_distance() {
        let visible = vec![
            TileCoord::new(5, 0, 0),   // 멀리 있음
            TileCoord::new(5, 16, 16), // 가운데 근처
        ];
        let cached = HashSet::new();
        let center = GeoCoord::new(0.0, 0.0); // 적도/본초자오선

        let mut requests = compute_tile_requests(&visible, &cached, &center);
        requests.sort_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap());

        // 가운데에 가까운 타일이 더 높은 우선순위
        // (priority 숫자가 낮을수록 우선)
    }
}
```

### 변경 4: 시각적 회귀 테스트 (Snapshot Testing)

```
새 파일:   crates/x-planets-gpu/src/snapshot.rs
새 폴더:   tests/snapshots/
```

```rust
/// GPU 렌더링 결과를 PNG로 저장하고, 기준 이미지와 비교
pub struct SnapshotTest {
    gpu: WgpuBackend,
    snapshot_dir: PathBuf,
}

impl SnapshotTest {
    /// 오프스크린 렌더링 → PNG 저장 → 기준과 비교
    pub async fn assert_snapshot(
        &self,
        name: &str,
        render_fn: impl FnOnce(&WgpuBackend) -> Vec<u8>,  // RGBA pixels
        width: u32,
        height: u32,
    ) {
        let pixels = render_fn(&self.gpu);

        let expected_path = self.snapshot_dir.join(format!("{}.png", name));
        let actual_path = self.snapshot_dir.join(format!("{}.actual.png", name));

        // 실제 결과 저장
        save_png(&actual_path, &pixels, width, height);

        if expected_path.exists() {
            // 기준 이미지와 비교
            let expected = load_png(&expected_path);
            let diff = pixel_diff(&expected, &pixels);

            if diff > THRESHOLD {
                let diff_path = self.snapshot_dir.join(format!("{}.diff.png", name));
                save_diff_image(&diff_path, &expected, &pixels, width, height);
                panic!(
                    "Snapshot mismatch: {} (diff: {:.4})\n\
                     Expected: {}\n\
                     Actual:   {}\n\
                     Diff:     {}",
                    name, diff,
                    expected_path.display(),
                    actual_path.display(),
                    diff_path.display(),
                );
            }
        } else {
            // 기준 이미지 없음 → 최초 실행, 기준으로 저장
            std::fs::copy(&actual_path, &expected_path).unwrap();
            println!("Created new snapshot: {}", expected_path.display());
        }
    }
}

// 사용 예시:
#[tokio::test]
async fn test_mercator_single_tile_render() {
    let snap = SnapshotTest::new_or_skip();

    snap.assert_snapshot("mercator_tile_0_0_0", |gpu| {
        // 줌 0, 단일 타일을 Mercator로 렌더링
        let tile = create_checkerboard_tile(256, 256);
        let viewport = Viewport { zoom: 0.0, .. };
        render_single_tile(gpu, &tile, &viewport, &Mercator)
    }, 512, 512).await;
}
```

### 변경 5: 단계별 검증 체인 (Karpathy의 "gradient checking")

카파시가 신경망에서 gradient를 수치 미분으로 검증하듯,
우리도 **각 변환 단계의 출력을 독립적으로 검증**.

```
새 파일: crates/x-planets-core/src/verify_chain.rs
```

```rust
/// 파이프라인의 각 단계를 독립적으로 검증하는 체인
pub struct VerifyChain {
    steps: Vec<VerifyStep>,
}

pub struct VerifyStep {
    name: String,
    verify_fn: Box<dyn Fn() -> StepResult>,
}

pub struct StepResult {
    passed: bool,
    details: String,
    metrics: HashMap<String, f64>,
}

impl VerifyChain {
    pub fn new() -> Self { Self { steps: Vec::new() } }

    pub fn add_step(&mut self, name: &str, f: impl Fn() -> StepResult + 'static) {
        self.steps.push(VerifyStep {
            name: name.to_string(),
            verify_fn: Box::new(f),
        });
    }

    /// 전체 체인 실행 — 하나라도 실패하면 즉시 중단
    pub fn run(&self) -> ChainResult {
        println!("═══════════════════════════════════════");
        println!(" Verification Chain ({} steps)", self.steps.len());
        println!("═══════════════════════════════════════");

        for (i, step) in self.steps.iter().enumerate() {
            print!(" [{}/{}] {} ... ", i + 1, self.steps.len(), step.name);
            let result = (step.verify_fn)();

            if result.passed {
                println!("✓ PASS");
                for (k, v) in &result.metrics {
                    println!("        {} = {:.6}", k, v);
                }
            } else {
                println!("✗ FAIL");
                println!("        {}", result.details);
                return ChainResult::Failed { step: i, name: step.name.clone() };
            }
        }

        println!("═══════════════════════════════════════");
        println!(" All {} steps passed", self.steps.len());
        println!("═══════════════════════════════════════");
        ChainResult::AllPassed
    }
}
```

사용 예시 (Phase 1 전체 검증):

```rust
fn verify_phase1() {
    let mut chain = VerifyChain::new();

    // Step 1: 수학 기초
    chain.add_step("GeoCoord ↔ Mercator roundtrip", || {
        let cities = vec![
            ("Seoul", GeoCoord::new(37.5665, 126.9780)),
            ("NYC", GeoCoord::new(40.7128, -74.0060)),
            ("Sydney", GeoCoord::new(-33.8688, 151.2093)),
            ("North Pole edge", GeoCoord::new(85.05, 0.0)),
        ];

        let mut max_err = 0.0f64;
        for (name, coord) in &cities {
            let merc = geo_to_mercator(coord);
            let back = mercator_to_geo(merc);
            let err = ((coord.lat - back.lat).powi(2)
                     + (coord.lon - back.lon).powi(2)).sqrt();
            max_err = max_err.max(err);
        }

        StepResult {
            passed: max_err < 1e-10,
            details: format!("max roundtrip error: {:.2e}", max_err),
            metrics: [("max_error".into(), max_err)].into(),
        }
    });

    // Step 2: TileCoord 정합성
    chain.add_step("TileCoord::from_geo → to_geo_bounds containment", || {
        let test_points = generate_random_geo_coords(1000);
        let mut failures = 0;

        for coord in &test_points {
            for zoom in 0..=18 {
                let tile = TileCoord::from_geo(coord, zoom);
                let bounds = tile.to_geo_bounds();
                if !bounds.contains(coord) {
                    failures += 1;
                }
            }
        }

        StepResult {
            passed: failures == 0,
            details: format!("{} containment failures out of 19000", failures),
            metrics: [("failures".into(), failures as f64)].into(),
        }
    });

    // Step 3: Frustum 컬링 정확도
    chain.add_step("Frustum culling: no false negatives", || {
        // 뷰포트 안에 확실히 보이는 타일이 컬링되지 않는지 검증
        let viewport = Viewport::new(800, 600);
        let frustum = viewport.frustum();
        let tiles = frustum.visible_tiles(5);

        // 모든 반환된 타일이 실제로 교차하는지
        let false_positives = tiles.iter()
            .filter(|t| !frustum.is_tile_visible(t))
            .count();

        StepResult {
            passed: false_positives == 0,
            details: format!("{} tiles, {} false positives", tiles.len(), false_positives),
            metrics: [
                ("tile_count".into(), tiles.len() as f64),
                ("false_positives".into(), false_positives as f64),
            ].into(),
        }
    });

    // Step 4: 프로젝션 CPU 정확도
    chain.add_step("Mercator projection: known values", || {
        let proj = Mercator;

        // (0, 0) → (0.5, 0.5)
        let origin = proj.project_cpu(DVec3::new(0.0, 0.0, 0.0));
        let origin_err = (origin - DVec3::new(0.5, 0.5, 0.0)).length();

        // (-180, -85.0511) → (0, ~1)  top-left
        let sw = proj.project_cpu(DVec3::new(-85.0511, -180.0, 0.0));
        let sw_err_x = sw.x.abs();
        let sw_err_y = (sw.y - 1.0).abs();

        let max_err = origin_err.max(sw_err_x).max(sw_err_y);

        StepResult {
            passed: max_err < 0.01,
            details: format!("origin_err={:.6}, sw=({:.4},{:.4})", origin_err, sw.x, sw.y),
            metrics: [("max_error".into(), max_err)].into(),
        }
    });

    // Step 5: Quad geometry 무결성
    chain.add_step("Tile quad: no degenerate triangles", || {
        let mut degenerate = 0;
        for z in 0..=5u8 {
            let n = 1u32 << z;
            for x in 0..n {
                for y in 0..n {
                    let coord = TileCoord::new(z, x, y);
                    let verts = tile_quad_vertices(&coord);

                    // 넓이가 0이 아닌지 확인
                    let area = (verts[1].position[0] - verts[0].position[0])
                             * (verts[2].position[1] - verts[0].position[1]);
                    if area.abs() < 1e-10 {
                        degenerate += 1;
                    }
                }
            }
        }

        StepResult {
            passed: degenerate == 0,
            details: format!("{} degenerate quads", degenerate),
            metrics: [("degenerate".into(), degenerate as f64)].into(),
        }
    });

    // Step 6: (GPU 있을 때만) CPU-GPU 프로젝션 일치
    chain.add_step("CPU ↔ GPU projection match (if GPU available)", || {
        // GPU 없으면 스킵
        // 있으면: 1000개 점에 대해 CPU vs GPU compute shader 비교
        // 오차 < 1e-4 (f32 정밀도)
        StepResult {
            passed: true,
            details: "GPU not available, skipped".into(),
            metrics: [].into(),
        }
    });

    chain.run();
}
```

---

## 변경된 전체 아키텍처 다이어그램

```
현재:
  MapEngine { viewport, camera, tile_loader, tile_cache, ... }
     └── update()  ← 모든 것을 한 번에

카파시 스타일:

  [순수 함수 레이어]                  [부수효과 레이어]
  ─────────────────                  ─────────────────
  determine_visible_tiles()     ←──  Viewport (상태)
         │
  compute_tile_requests()       ←──  TileCache (상태)
         │
  build_tile_geometry()              (순수)
         │
  compute_viewport_uniforms()        (순수)
         │
  ┌──────┴──────┐
  │ CpuBackend  │  GpuBackend       (trait 추상화)
  │ (테스트용)   │  (실제 렌더)
  └──────┬──────┘
         │
  SnapshotTest                       (시각적 회귀 테스트)
         │
  VerifyChain                        (단계별 검증)
```

---

## 실행 순서: 카파시라면 이렇게 했을 것

### Milestone 0: "가장 바보같은 것부터"

```
목표: 단색 사각형 하나를 화면에 띄운다.
파일: examples/step00_triangle.rs

테스트:
  ✓ wgpu 초기화 성공
  ✓ 윈도우 생성 성공
  ✓ 단색 clear → 파란 화면
  ✓ 삼각형 1개 렌더링
  ✓ 스크린샷 저장 → snapshots/step00.png

검증:
  "파란 화면에 하얀 삼각형이 보이면 성공"
```

### Milestone 1: "단일 컬러 쿼드"

```
목표: TileVertex 4개 → 컬러 사각형
파일: examples/step01_colored_quad.rs

테스트:
  ✓ TileVertex 레이아웃이 wgpu에 맞는지
  ✓ 인덱스 버퍼 올바른지
  ✓ 쿼드가 화면의 올바른 위치에 표시되는지

검증:
  tile_quad_vertices(TileCoord(0,0,0)) → 화면 전체를 덮는 쿼드
  tile_quad_vertices(TileCoord(1,0,0)) → 화면 왼쪽 위 1/4
```

### Milestone 2: "체커보드 텍스처 쿼드"

```
목표: 프로그래밍 방식으로 생성한 텍스처를 쿼드에 매핑
파일: examples/step02_textured_quad.rs

테스트:
  ✓ TextureManager.create_rgba_texture() 동작
  ✓ 바인드그룹 생성 + 셰이더 바인딩
  ✓ 텍스처 샘플링 결과 확인 (readback)

검증:
  8x8 체커보드 패턴 생성 → 텍스처 업로드 → 렌더 → readback
  → 픽셀값이 기대와 일치하는지 assert
```

### Milestone 3: "뷰 매트릭스로 팬/줌"

```
목표: 유니폼 버퍼로 뷰 매트릭스 전달, 마우스로 팬/줌
파일: examples/step03_pan_zoom.rs

테스트:
  ✓ ViewportUniforms 바이트 레이아웃 (bytemuck 정합성)
  ✓ 줌 0에서 쿼드 1개 = 전체 화면
  ✓ 줌 1에서 쿼드 1개 = 화면 1/4
  ✓ 팬 후 뷰포트 센터 이동 확인

검증:
  CPU에서 계산한 view_proj * vertex_position과
  셰이더의 output이 같은지 readback으로 검증
```

### Milestone 4: "실제 타일 1장"

```
목표: OSM에서 타일 1장 다운로드 → 디코드 → 텍스처 업로드 → 렌더
파일: examples/step04_single_real_tile.rs

테스트:
  ✓ HTTP fetch → 200 OK, bytes > 0
  ✓ RasterTileDecoder → 256x256 RGBA
  ✓ 텍스처 업로드 → readback → 디코딩 전과 일치
  ✓ 렌더 결과 스냅샷

검증:
  다운로드한 PNG를 image 크레이트로 디코딩한 결과와
  GPU readback 결과의 PSNR > 40dB
```

### Milestone 5: "4타일 (줌 1)"

```
목표: 줌 1의 4개 타일을 올바른 위치에 배치
파일: examples/step05_four_tiles.rs

테스트:
  ✓ 4개 타일이 빈틈/겹침 없이 배치
  ✓ 각 타일의 텍스처 좌표가 올바름
  ✓ 타일 경계에서 이음매 없음

검증:
  타일 경계 픽셀의 색상 연속성 체크
  (인접 타일의 경계 픽셀 차이 < threshold)
```

### Milestone 6: "동적 타일 로딩"

```
목표: 뷰포트 이동 시 필요한 타일만 로드/해제
파일: examples/step06_dynamic_loading.rs

테스트:
  ✓ determine_visible_tiles() 결과와 실제 로드된 타일 일치
  ✓ 뷰포트 밖 타일은 캐시에서 해제
  ✓ 로딩 우선순위: 중심부 먼저

검증 (카파시 스타일 로그):
  Frame 1: visible=[0/0/0] loaded=[0/0/0] pending=[] cache_hit=0%
  Frame 2: zoom→1, visible=[1/0/0,1/1/0,1/0/1,1/1/1] loaded=[] pending=[4]
  Frame 5: visible=[1/0/0,...] loaded=[1/0/0,1/1/0,...] pending=[0] cache_hit=100%
```

### Milestone 7: "프로젝션 전환"

```
목표: Mercator ↔ Equirectangular 실시간 전환
파일: examples/step07_projection_switch.rs

테스트:
  ✓ 프로젝션 전환 시 타일 위치가 올바르게 변경됨
  ✓ CPU와 GPU 프로젝션 결과 일치 (VerifyChain)
  ✓ 전환 중 아티팩트 없음

검증:
  동일 뷰포트에서 Mercator→Equirectangular 전환 시
  적도 근처 타일은 거의 동일, 극지방 타일은 크게 변형
  → 스냅샷 비교
```

### 이후 Milestone (각각 동일 패턴):

```
M8:  벡터 타일 파싱 (CPU only, 렌더링 없이 geometry 추출 검증)
M9:  벡터 타일 테셀레이션 (CPU ear-cut → GPU 결과 비교)
M10: 벡터 타일 렌더링 (래스터 위에 오버레이)
M11: 터레인 높이맵 파싱 + CPU 시각화
M12: 터레인 GPU 렌더링 + 힐셰이드
M13: 컴퓨트 셰이더 프로젝션 (CPU 레퍼런스와 비교)
M14: 텍스처 아틀라스 (draw call 수 측정)
M15: WASM 빌드 (동일 테스트 스위트 브라우저에서 실행)
...
```

---

## 새로 필요한 크레이트/파일 요약

```
새로 추가해야 할 것:

crates/x-planets-gpu/src/test_utils.rs    — CpuBackend, test helpers
crates/x-planets-gpu/src/verify.rs        — CPU-GPU 동일성 검증기
crates/x-planets-gpu/src/snapshot.rs      — 시각적 회귀 테스트
crates/x-planets-core/src/pipeline.rs     — 순수 함수 파이프라인 단계
crates/x-planets-core/src/verify_chain.rs — 단계별 검증 체인

tests/                                     — 통합 테스트
tests/snapshots/                           — 기준 스크린샷
tests/fixtures/                            — 테스트용 타일 데이터

examples/step00_triangle.rs
examples/step01_colored_quad.rs
examples/step02_textured_quad.rs
examples/step03_pan_zoom.rs
examples/step04_single_real_tile.rs
examples/step05_four_tiles.rs
examples/step06_dynamic_loading.rs
examples/step07_projection_switch.rs

변경해야 할 것:

crates/x-planets-gpu/src/context.rs       — GpuBackend trait 추가
crates/x-planets-core/src/engine.rs       — 순수 함수로 분해
crates/x-planets-core/src/render.rs       — 렌더러와 geometry 생성 분리
```

---

## 핵심 원칙 요약

| 카파시 원칙 | x-planets 적용 |
|---|---|
| "가장 바보같은 것부터" | Step 0: 단색 삼각형 → Step 4: 실제 타일 1장 |
| "매번 눈으로 확인" | SnapshotTest: 매 단계 PNG 저장 + diff |
| "CPU가 truth" | CpuBackend가 ground truth, GPU 결과를 이에 비교 |
| "한 번에 하나만" | 각 Milestone에서 딱 1가지만 새로 추가 |
| "수치적 검증" | VerifyChain: 오차 범위 명시, PSNR, pixel diff |
| "gradient checking" | CPU-GPU 프로젝션 일치 검증 (1000개 랜덤 포인트) |
| "training loop 먼저" | 렌더 루프부터 확보, 그 위에 기능을 쌓기 |
