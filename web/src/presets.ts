import type { ExamplePreset } from "./types.ts";

/**
 * Cesium Ion access token — read from environment variable at build time.
 *
 * Set `VITE_CESIUM_ION_TOKEN` in your environment or `.env` file.
 * Get a free token at https://ion.cesium.com/tokens
 */
const CESIUM_ION_TOKEN: string | undefined =
  (import.meta as any).env?.VITE_CESIUM_ION_TOKEN || undefined;

/** Cesium Ion well-known asset IDs. */
const CESIUM_ASSETS = {
  OSM_BUILDINGS: 96188,
  WORLD_TERRAIN: 1,
} as const;

/** Whether a Cesium Ion token is available. */
export function hasCesiumToken(): boolean {
  return !!CESIUM_ION_TOKEN;
}

/** Built-in example presets. */
export const PRESETS: ExamplePreset[] = [
  {
    id: "basic",
    name: "Basic Map",
    description: "OpenStreetMap raster tiles",
    config: {
      center: [37.5665, 126.978],
      zoom: 5,
      projection: "Web Mercator",
      layers: [
        {
          name: "osm",
          url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
          kind: "raster",
        },
      ],
    },
  },
  {
    id: "terrain",
    name: "Terrain",
    description: "3D terrain with hillshade (Grand Canyon)",
    config: {
      center: [36.1069, -112.1126],
      zoom: 11,
      projection: "Web Mercator",
      layers: [
        {
          name: "osm",
          url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
          kind: "raster",
        },
      ],
      terrain: {
        url: "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
        encoding: "terrarium",
        hillshadeStrength: 1.0,
      },
    },
  },
  {
    id: "cesium-osm-buildings",
    name: "3D Buildings",
    description: CESIUM_ION_TOKEN
      ? "Cesium OSM Buildings (3D Tiles)"
      : "Cesium OSM Buildings — set VITE_CESIUM_ION_TOKEN",
    config: {
      center: [40.6892, -74.0445],
      zoom: 15,
      projection: "Web Mercator",
      layers: [
        {
          name: "osm",
          url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
          kind: "raster",
        },
        ...(CESIUM_ION_TOKEN
          ? [
              {
                name: "buildings",
                url: "",
                kind: "3dtiles" as const,
                cesiumIonToken: CESIUM_ION_TOKEN,
                cesiumIonAsset: CESIUM_ASSETS.OSM_BUILDINGS,
              },
            ]
          : []),
      ],
      terrain: {
        url: "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
        encoding: "terrarium",
      },
    },
  },
  {
    id: "globe",
    name: "Globe",
    description: "Globe projection view of Earth",
    config: {
      center: [37.5665, 126.978],
      zoom: 3,
      projection: "Globe",
      layers: [
        {
          name: "osm",
          url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
          kind: "raster",
        },
      ],
    },
  },
  {
    id: "seoul-terrain",
    name: "Seoul",
    description: "Seoul with terrain and pitch",
    config: {
      center: [37.5665, 126.978],
      zoom: 12,
      projection: "Web Mercator",
      layers: [
        {
          name: "osm",
          url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
          kind: "raster",
        },
      ],
      terrain: {
        url: "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
        encoding: "terrarium",
        hillshadeStrength: 0.8,
      },
    },
  },
];
