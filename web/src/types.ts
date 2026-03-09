/** Typed interface for the XPlanetsMap WASM API exposed at `window.xplanets`. */
export interface XPlanetsMap {
  // ── Map Control ──
  panBy(dx: number, dy: number): void;
  zoomTo(zoom: number): void;
  setCenter(lat: number, lon: number): void;
  getCenter(): [number, number];
  getZoom(): number;
  getBearing(): number;
  getPitch(): number;

  // ── Camera Animation ──
  setBearing(degrees: number): void;
  setPitch(degrees: number): void;
  flyTo(
    lat: number,
    lon: number,
    zoom: number,
    duration?: number,
    bearing?: number,
    pitch?: number,
  ): void;
  easeTo(
    lat: number,
    lon: number,
    zoom: number,
    duration?: number,
    bearing?: number,
    pitch?: number,
  ): void;
  jumpTo(
    lat: number,
    lon: number,
    zoom: number,
    bearing?: number,
    pitch?: number,
  ): void;
  stopAnimation(): void;

  // ── Coordinate Conversion ──
  project(lat: number, lon: number): [number, number] | null;
  unproject(x: number, y: number): [number, number] | null;

  // ── Projection ──
  setProjection(name: string): boolean;
  cycleProjection(): string;
  getProjection(): string;

  // ── Terrain ──
  /**
   * Toggle terrain on/off for the base raster layer.
   * Uses the terrain URL/encoding from the config or `setTerrainSource()`.
   */
  toggleTerrain(): boolean;
  terrainEnabled(): boolean;
  /** Set terrain elevation source at runtime. */
  setTerrainSource(url: string, encoding?: TerrainEncoding): void;
  setTerrainExaggeration(value: number): void;
  getTerrainExaggeration(): number;

  // ── Layer Management ──
  addLayer(name: string, url: string, options?: AddLayerOptions | number): number;
  getLayer(name: string): LayerInfo | null;
  getLayers(): string[];
  setLayerVisible(name: string, visible: boolean): boolean;
  setLayerOpacity(name: string, opacity: number): boolean;
  removeLayer(name: string): boolean;
  layerCount(): number;

  // ── Viewport ──
  resize(width: number, height: number): void;

  // ── Camera Limits ──
  getMinZoom(): number;
  setMinZoom(zoom: number): void;
  getMaxZoom(): number;
  setMaxZoom(zoom: number): void;
  getMaxPitch(): number;
  setMaxPitch(degrees: number): void;
  getTileBudget(): number;
  setTileBudget(budget: number): void;

  // ── Events ──
  on(event: "move", callback: (data: { lat: number; lon: number }) => void): void;
  on(event: "zoom", callback: (data: { zoom: number }) => void): void;
  on(event: "pitch", callback: (data: { pitch: number }) => void): void;
  on(event: "bearing", callback: (data: { bearing: number }) => void): void;
  on(event: "moveend", callback: () => void): void;
  on(event: "zoomend", callback: () => void): void;
  on(
    event: "click",
    callback: (data: { lat: number; lon: number; x: number; y: number }) => void,
  ): void;
  on(event: string, callback: (data: any) => void): void;
  off(event: string, callback: (data: any) => void): void;

  // ── Lifecycle ──
  destroy(): void;
}

/** Layer kind identifiers. */
export type LayerKind = "raster" | "raster-dem" | "terrain" | "3dtiles";

/** Terrain elevation encoding format. */
export type TerrainEncoding = "terrarium" | "mapbox" | "mapbox-rgb" | "quantized-mesh" | "qm";

/** Options for `addLayer()`. */
export interface AddLayerOptions {
  /** Layer kind: "raster" (default), "raster-dem" / "terrain", "3dtiles". */
  kind?: LayerKind;
  /** Terrain encoding: "terrarium" (default), "mapbox", "quantized-mesh". */
  encoding?: TerrainEncoding;
  /** For terrain layers: name of the raster layer to drape imagery from. */
  imageryLayer?: string;
  /** Stacking order (lower = drawn first). */
  zOrder?: number;
  /** Opacity 0.0–1.0 (default: 1.0). */
  opacity?: number;
}

/** Layer metadata returned by `getLayer()`. */
export interface LayerInfo {
  name: string;
  url: string;
  opacity: number;
  visible: boolean;
  zOrder: number;
  kind: string;
}

/** Layer configuration for initial setup. */
export interface LayerConfig {
  name: string;
  url: string;
  /** Layer kind: "raster" (default), "raster-dem" / "terrain", "3dtiles". */
  kind?: LayerKind;
  /** Terrain encoding: "terrarium" (default), "mapbox", "quantized-mesh". */
  encoding?: TerrainEncoding;
  /** For terrain layers: name of the raster layer to drape imagery from. */
  imageryLayer?: string;
  zOrder?: number;
  opacity?: number;
}

/** Terrain configuration for `XPlanetsConfig`. */
export interface TerrainConfig {
  /** Elevation tile URL template (e.g. `https://…/{z}/{x}/{y}.png`). */
  url: string;
  /** Encoding format. Default: "terrarium". */
  encoding?: TerrainEncoding;
  /** Sun direction [x, y, z] for hillshade. Default: [-0.5, -0.5, 0.7]. */
  sunDirection?: [number, number, number];
  /** Hillshade strength 0–1. Default: 1.0. */
  hillshadeStrength?: number;
}

/** Configuration for `XPlanets.create()`. */
export interface XPlanetsConfig {
  center?: [number, number];
  zoom?: number;
  projection?: string;
  minZoom?: number;
  maxZoom?: number;
  maxPitch?: number;
  tileBudget?: number;
  /** Celestial body: "Earth" (default), "Moon", or "Mars". */
  body?: "Earth" | "Moon" | "Mars";
  layers?: LayerConfig[];
  /** Terrain elevation source configuration. */
  terrain?: TerrainConfig;
}

/** Factory for creating x-planets map instances via Promise-based init. */
export interface XPlanetsFactory {
  create(canvasId: string, config?: XPlanetsConfig): Promise<XPlanetsMap>;
}

declare global {
  interface Window {
    xplanets?: XPlanetsMap;
  }

  interface WindowEventMap {
    "xplanets-ready": CustomEvent;
  }

  /** Available after WASM module loads. */
  const XPlanets: XPlanetsFactory;
}
