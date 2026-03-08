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
  toggleTerrain(url?: string, encoding?: string): boolean;
  terrainEnabled(): boolean;
  setTerrainExaggeration(value: number): void;
  getTerrainExaggeration(): number;

  // ── Layer Management ──
  addLayer(name: string, url: string, zOrder?: number): number;
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

/** Layer metadata returned by `getLayer()`. */
export interface LayerInfo {
  name: string;
  url: string;
  opacity: number;
  visible: boolean;
  zOrder: number;
  kind: string;
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
  layers?: Array<{
    name: string;
    url: string;
    kind?: string;
    zOrder?: number;
    opacity?: number;
  }>;
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
