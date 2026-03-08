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

declare global {
  interface Window {
    xplanets?: XPlanetsMap;
  }

  interface WindowEventMap {
    "xplanets-ready": CustomEvent;
  }
}
