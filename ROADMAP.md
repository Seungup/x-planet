# x-planets 고도화 로드맵

## 현재 상태 (Phase 0 — Scaffold)

완성된 것:
- Cargo workspace 구조 (7 crates)
- 핵심 트레이트: `ProjectionPlugin`, `TileDecoder`, `TileSource`
- 수학 라이브러리: GeoCoord, TileCoord, BoundingBox, Mercator 변환, Frustum 컬링
- GPU 추상화: GpuContext, TextureManager, Pipeline Builder
- 프로젝션 시스템: Registry + Mercator/Equirectangular 빌트인 + CustomProjection
- 타일 시스템: RasterTileDecoder, LRU TileCache, Priority TileLoader
- 엔진: MapEngine, Viewport, CameraController, LayerStack
- WGSL 셰이더: mercator.wgsl, equirectangular.wgsl, raster_tile.wgsl

아직 없는 것:
- 실제 윈도우 생성 및 GPU 서피스 연결
- 실시간 렌더 루프
- async 타일 네트워크 로딩 통합
- 벡터/터레인 타일 디코더
- WASM 빌드 파이프라인

---

## Phase 1 — 실제 렌더링 파이프라인 완성 (핵심)

### 1.1 윈도우 + GPU 서피스 연결

현재 `GpuContext`에 서피스가 없음. winit 윈도우와 연결해야 함.

```
변경 파일:
  crates/x-planets-gpu/src/context.rs
    - new_with_window() 메서드 추가
    - Surface 생성 및 configure
    - resize 핸들링

  crates/x-planets-native/src/lib.rs
    - winit EventLoop + Window 생성
    - GpuContext::new_with_window() 호출
    - 이벤트 핸들링 (resize, mouse, keyboard, close)
```

핵심 구현:
```rust
// context.rs에 추가
impl GpuContext {
    pub async fn new_with_window(window: Arc<Window>) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(...);
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance.request_adapter(&RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..
        }).await?;
        let (device, queue) = adapter.request_device(...).await?;

        let config = surface.get_default_config(&adapter, width, height)?;
        surface.configure(&device, &config);

        Ok(Self { instance, adapter, device, queue,
            surface: Some(GpuSurface { surface, config }) })
    }

    pub fn resize_surface(&mut self, width: u32, height: u32) {
        if let Some(ref mut surf) = self.surface {
            surf.config.width = width;
            surf.config.height = height;
            surf.surface.configure(&self.device, &surf.config);
        }
    }
}
```

### 1.2 렌더 루프 구현

```
변경 파일:
  crates/x-planets-core/src/engine.rs
    - render() 메서드 추가: 프레임 획득 → 타일 그리기 → present
  crates/x-planets-core/src/render.rs
    - RasterTileRenderer 구조체: 파이프라인 + 바인드그룹 생성/관리
    - draw_tiles() 메서드
```

렌더 루프 흐름:
```
매 프레임:
  1. surface.get_current_texture()
  2. create CommandEncoder
  3. begin_render_pass (clear color)
  4. for each visible tile in LayerStack:
       - bind tile texture + sampler
       - bind viewport uniforms
       - bind tile uniforms (bounds, opacity)
       - draw 6 indices (quad)
  5. queue.submit()
  6. frame.present()
```

### 1.3 Async 타일 로딩 통합

현재 TileLoader는 큐만 있고 실제 fetch가 없음.

```
변경 파일:
  crates/x-planets-tiles/src/loader.rs
    - spawn_fetch_tasks() 추가: dequeue → TileSource.fetch() → decode → cache
  crates/x-planets-core/src/engine.rs
    - update()에서 로딩 완료된 타일을 GPU 텍스처로 업로드
    - channel 기반 비동기 통신 (로딩 스레드 → 메인 스레드)
```

```rust
// engine.rs 변경안
pub struct MapEngine {
    // ... 기존 필드
    tile_rx: mpsc::Receiver<(TileCoord, DecodedRasterTile)>,
    tile_tx: mpsc::Sender<(TileCoord, DecodedRasterTile)>,
    gpu_tiles: HashMap<TileCoord, GpuTexture>,  // GPU에 업로드된 타일
}

impl MapEngine {
    pub fn update(&mut self, gpu: &GpuContext) {
        // 1. 완료된 타일 수신 → GPU 텍스처 생성
        while let Ok((coord, decoded)) = self.tile_rx.try_recv() {
            let texture = gpu.texture_manager.create_rgba_texture(
                &gpu.device, &gpu.queue,
                &format!("tile-{}", coord),
                decoded.width, decoded.height,
                &decoded.pixels,
            );
            self.gpu_tiles.insert(coord, texture);
            self.tile_loader.complete();
            self.needs_redraw = true;
        }

        // 2. 새 타일 로딩 요청
        let visible = self.viewport.visible_tiles();
        for coord in &visible {
            if !self.gpu_tiles.contains_key(coord) {
                self.tile_loader.enqueue(...);
            }
        }

        // 3. 큐에서 꺼내서 fetch 시작
        while let Some(req) = self.tile_loader.dequeue() {
            let tx = self.tile_tx.clone();
            let source = self.tile_source.clone();
            tokio::spawn(async move {
                if let Ok(bytes) = source.fetch(req.coord).await {
                    if let Ok(decoded) = decoder.decode(req.coord, &bytes).await {
                        let _ = tx.send((req.coord, decoded)).await;
                    }
                }
            });
        }
    }
}
```

### 1.4 Phase 1 완료 기준

- 데스크톱 윈도우에 OSM 래스터 타일이 실시간으로 표시됨
- 마우스 드래그로 팬, 스크롤로 줌
- 줌 레벨 변경 시 적절한 타일 로딩/해제
- 기본 Mercator 프로젝션으로 렌더링

---

## Phase 2 — 멀티 레이어 + 벡터/터레인 타일

### 2.1 벡터 타일 (MVT/PBF) 지원

```
새 파일:
  crates/x-planets-tiles/src/vector_decoder.rs
신규 의존성: prost (protobuf 파싱)

구현 사항:
  - MVT protobuf 스키마 정의 (.proto → prost codegen)
  - VectorTileDecoder: PBF bytes → VectorLayer[Feature{geometry, properties}]
  - Feature geometry → GPU 버텍스/인덱스 버퍼 변환
    - Point: 인스턴싱으로 아이콘/원 렌더링
    - LineString: 밀링(miter join, round cap) → 삼각형 스트립
    - Polygon: ear-cut 테셀레이션 → 삼각형 메시
  - 스타일 시스템: JSON 기반 레이어 스타일 (MapLibre GL Style spec 호환)
```

벡터 타일 셰이더:
```
새 파일:
  shaders/rendering/vector_line.wgsl     — SDF 기반 라인 렌더링
  shaders/rendering/vector_polygon.wgsl  — 채우기 + 아웃라인
  shaders/rendering/vector_point.wgsl    — 인스턴스 포인트/아이콘
```

### 2.2 터레인 타일 (DEM / Quantized Mesh)

```
새 파일:
  crates/x-planets-tiles/src/terrain_decoder.rs
신규 의존성: geo-tiff (선택), quantized-mesh-decoder

구현 사항:
  - GeoTIFF/Mapbox Terrain RGB → f32 높이맵 텍스처
  - Quantized Mesh → 삼각형 메시 (버텍스 + 인덱스 버퍼)
  - 노멀 맵 자동 생성 (높이맵 → Sobel 필터)
  - 힐셰이드 컴퓨트 셰이더
```

터레인 셰이더:
```
새 파일:
  shaders/rendering/terrain.wgsl
    - 버텍스: 높이맵 샘플링 → Y축 디스플레이스먼트
    - 프래그먼트: 힐셰이드 + 등고색 + 텍스처 블렌딩
  shaders/compute/hillshade.wgsl
    - 컴퓨트: 높이맵 → 음영 텍스처 (라이트 방향 기반)
  shaders/compute/normal_map.wgsl
    - 컴퓨트: 높이맵 → 노멀맵 (RGB)
```

### 2.3 레이어 합성 고도화

```
변경 파일:
  crates/x-planets-core/src/render.rs
    - BlendMode enum: Normal, Multiply, Screen, Overlay
    - 레이어별 독립 렌더 타겟 (오프스크린 텍스처)
    - 최종 합성 패스: 모든 레이어를 블렌딩하여 스크린에 출력
```

### 2.4 WASM 빌드 + 웹 뷰어

```
변경 파일:
  crates/x-planets-web/src/lib.rs
    - wasm-bindgen 바인딩 완성
    - Canvas → wgpu Surface 연결
    - fetch API 기반 TileSource 구현
    - requestAnimationFrame 렌더 루프
    - 마우스/터치 이벤트 핸들링

새 파일:
  examples/web_viewer/index.html
  examples/web_viewer/index.js
  examples/web_viewer/build.sh  (wasm-pack build)
```

### 2.5 Phase 2 완료 기준

- 래스터 + 벡터 + 터레인 레이어 동시 표시
- 벡터 타일 스타일링 (색상, 선 두께, 채우기)
- 3D 터레인 힐셰이드
- 웹 브라우저에서 동일 기능 동작

---

## Phase 3 — GPU 최적화 + 고급 프로젝션

### 3.1 Compute Shader 프로젝션

현재 프로젝션은 버텍스 셰이더에서 인라인 실행.
→ 컴퓨트 셰이더로 분리하여 일괄 변환 + 캐싱.

```
새 파일:
  shaders/compute/projection_transform.wgsl
    - Storage buffer input: 월드 좌표 배열
    - Storage buffer output: 프로젝트된 좌표 배열
    - 워크그룹 256, 한 번에 수천 개 버텍스 변환

  crates/x-planets-gpu/src/compute.rs
    - ProjectionCompute 구조체
    - dispatch() → GPU에 변환 요청
    - 결과 버퍼를 렌더 파이프라인에 바인딩
```

이점:
- 프로젝션 변경 시 한 번만 재계산
- 줌/팬 시 뷰 매트릭스만 업데이트 (프로젝션 재계산 불필요)
- 벡터 타일의 수만 개 버텍스를 GPU에서 병렬 변환

### 3.2 타일 텍스처 아틀라스

개별 타일마다 텍스처/바인드그룹 생성 → draw call 폭발.
→ 텍스처 아틀라스로 통합.

```
새 파일:
  crates/x-planets-gpu/src/atlas.rs
    - TextureAtlas: 큰 텍스처(4096x4096)에 타일을 패킹
    - 타일 슬롯 할당/해제 (bin packing)
    - UV 오프셋 계산: 타일 → 아틀라스 좌표
    - 인스턴스 렌더링: 모든 타일을 1회 draw call로 렌더
```

### 3.3 LOD (Level of Detail)

```
변경 파일:
  crates/x-planets-core/src/viewport.rs
    - fractional_zoom 기반 LOD 판단
    - 상위 줌 타일을 placeholder로 표시 (하위 로딩 중일 때)
    - 타일 페이드인/페이드아웃 애니메이션

  crates/x-planets-tiles/src/cache.rs
    - 계층적 캐시: zoom_level별 LRU
    - 타일 참조 카운팅 (placeholder 사용 추적)
```

### 3.4 프러스텀 컬링 GPU화

```
새 파일:
  shaders/compute/frustum_cull.wgsl
    - 입력: 모든 로드된 타일 바운딩박스
    - 출력: visible 플래그 배열
    - CPU에서 가시성 판단 → GPU indirect draw로 대체
```

### 3.5 고급 프로젝션 추가

```
새 파일:
  shaders/projections/lambert_conformal.wgsl
  shaders/projections/albers_equal_area.wgsl
  shaders/projections/polar_stereographic.wgsl
  shaders/projections/robinson.wgsl
  shaders/projections/orthographic_globe.wgsl  (3D 지구본)

  crates/x-planets-projection/src/builtins.rs
    - 각 프로젝션의 CPU 폴백 구현
    - uniform 파라미터 (중심경위도, 표준평행선 등)
```

### 3.6 프로젝션 핫 리로딩

```
변경 파일:
  crates/x-planets-projection/src/registry.rs
    - load_from_file(path) → WGSL 읽기 → CustomProjection 등록
    - watch_directory(path) → notify 크레이트로 파일 변경 감지
    - 셰이더 변경 시 파이프라인 자동 재생성
```

### 3.7 Phase 3 완료 기준

- 컴퓨트 셰이더 프로젝션 (프로젝션 변경 시 1프레임 내 완료)
- 텍스처 아틀라스 → draw call 10배 감소
- LOD: 줌 중 부모 타일 placeholder
- 10+ 프로젝션 빌트인
- .wgsl 파일 드롭으로 커스텀 프로젝션 추가

---

## Phase 4 — 프로덕션 + 고급 기능

### 4.1 3D Globe 렌더링

```
새 파일:
  crates/x-planets-core/src/globe.rs
    - GlobeMesh: 구체 메시 생성 (icosphere subdivision)
    - 타일을 구체 표면에 매핑
    - 대기 산란 셰이더 (Rayleigh/Mie)

  shaders/rendering/globe.wgsl
    - 구체 렌더링 + 타일 텍스처 샘플링
    - 대기 산란 효과

  shaders/rendering/atmosphere.wgsl
    - 레이마칭 기반 대기 렌더링
```

### 4.2 라벨 / 텍스트 렌더링

```
새 파일:
  crates/x-planets-core/src/text.rs
    - SDF(Signed Distance Field) 폰트 아틀라스 생성
    - 라벨 배치 알고리즘 (충돌 회피)
    - 줌 레벨별 라벨 가시성

  shaders/rendering/text_sdf.wgsl
    - SDF 텍스트 렌더링 (안티앨리어싱, 아웃라인, 그림자)
```

### 4.3 인터랙션 / 피처 피킹

```
새 파일:
  crates/x-planets-core/src/picking.rs
    - GPU 기반 피킹: 별도 렌더 타겟에 feature ID 인코딩
    - 마우스 좌표 → readback → feature 식별
    - 팝업/툴팁 시스템
```

### 4.4 데이터 시각화 레이어

```
새 파일:
  crates/x-planets-core/src/layers/
    - heatmap.rs    — 가중치 기반 히트맵 (컴퓨트 셰이더)
    - choropleth.rs — 영역별 색상 매핑
    - particle.rs   — 바람/해류 파티클 시뮬레이션
    - point_cloud.rs — 대량 포인트 렌더링 (인스턴싱)

  shaders/compute/heatmap.wgsl
  shaders/compute/particle_sim.wgsl
```

### 4.5 오프라인 / 타일 번들

```
새 파일:
  crates/x-planets-cli/src/
    - bundle.rs   — 영역 지정 → 타일 일괄 다운로드 → .xpkg 아카이브
    - convert.rs  — GeoTIFF/Shapefile → 내부 포맷 변환
    - validate.rs — 셰이더 검증 + 타일 무결성 체크
```

### 4.6 플러그인 시스템

```
새 파일:
  crates/x-planets-plugin/
    - Plugin trait: on_load(), on_frame(), on_event()
    - WASI 기반 샌드박스 플러그인 실행 (선택)
    - 또는 Rust dylib 기반 네이티브 플러그인
```

### 4.7 Phase 4 완료 기준

- 3D 지구본 + 대기 렌더링
- 벡터 피처 클릭 → 속성 팝업
- 히트맵, 파티클 시각화
- 오프라인 타일 번들
- 서드파티 플러그인 로딩

---

## 크레이트 의존성 그래프 (최종)

```
x-planets-math           (의존성 없음 — 기초)
    ↑
x-planets-gpu            (wgpu, bytemuck, naga)
    ↑
x-planets-projection     (math, gpu)
    ↑
x-planets-tiles          (math, image, prost)
    ↑
x-planets-core           (math, gpu, projection, tiles)
    ↑               ↑
x-planets-native    x-planets-web
(winit, tokio,      (wasm-bindgen,
 reqwest)            web-sys)
```

## 기술적 난이도 순서

| 난이도 | 항목 |
|--------|------|
| ★☆☆ | Phase 1.1: 윈도우 + 서피스 연결 |
| ★☆☆ | Phase 1.3: async 타일 로딩 |
| ★★☆ | Phase 1.2: 렌더 루프 + 타일 렌더링 |
| ★★☆ | Phase 2.4: WASM 빌드 |
| ★★★ | Phase 2.1: 벡터 타일 테셀레이션 |
| ★★★ | Phase 2.2: 터레인 메시 + 힐셰이드 |
| ★★★ | Phase 3.1: 컴퓨트 셰이더 프로젝션 |
| ★★★ | Phase 3.2: 텍스처 아틀라스 |
| ★★★★ | Phase 4.1: 3D Globe + 대기 |
| ★★★★ | Phase 4.4: 파티클 시뮬레이션 |
| ★★★★★ | Phase 4.6: 플러그인 시스템 (WASI) |

## 권장 실행 순서

1. **지금 바로**: Phase 1.1 → 1.2 → 1.3 (실제 화면에 타일 띄우기)
2. **1~2주 후**: Phase 2.4 (WASM) — 웹에서도 동작 확인
3. **병렬 진행**: Phase 2.1 (벡터) + Phase 3.5 (추가 프로젝션)
4. **성능 이슈 발생 시**: Phase 3.1~3.4 (GPU 최적화)
5. **장기**: Phase 4 (3D Globe, 시각화, 플러그인)
