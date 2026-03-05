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

  // ── Projection ──
  setProjection(name: string): boolean;
  cycleProjection(): string;
  getProjection(): string;

  // ── Terrain ──
  toggleTerrain(): boolean;
  terrainEnabled(): boolean;
  setTerrainExaggeration(value: number): void;
  getTerrainExaggeration(): number;

  // ── Layer Management ──
  setLayerVisible(name: string, visible: boolean): boolean;
  setLayerOpacity(name: string, opacity: number): boolean;
  removeLayer(name: string): boolean;
  layerCount(): number;

  // ── Viewport ──
  resize(width: number, height: number): void;
}

declare global {
  interface Window {
    xplanets?: XPlanetsMap;
  }

  interface WindowEventMap {
    "xplanets-ready": CustomEvent;
  }
}
