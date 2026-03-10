import type { ExamplePreset } from "./types.ts";

/**
 * Cesium Ion default access token for community assets.
 *
 * This token provides access to Cesium's default assets (World Terrain,
 * OSM Buildings, etc.) for demo/evaluation purposes.
 * For production use, replace with your own token from https://ion.cesium.com
 */
const CESIUM_ION_DEFAULT_TOKEN =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI3NjhhMGZhZC0yMjhiLTRhMjMtYTFlMy01NzBjNjI0NWJjMjgiLCJpZCI6MjU5LCJpYXQiOjE3MzA5Nzg1MjN9.KYdGljmp7CzVbLyhzEuDzJnWezCj7GNvN5SGHIEJQAQ";

/** Cesium Ion well-known asset IDs. */
const CESIUM_ASSETS = {
  OSM_BUILDINGS: 96188,
  WORLD_TERRAIN: 1,
} as const;

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
    description: "Cesium OSM Buildings (3D Tiles)",
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
        {
          name: "buildings",
          url: "",
          kind: "3dtiles",
          cesiumIonToken: CESIUM_ION_DEFAULT_TOKEN,
          cesiumIonAsset: CESIUM_ASSETS.OSM_BUILDINGS,
        },
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
