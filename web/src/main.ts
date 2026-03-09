import "./style.css";
import { setupControls } from "./controls.ts";
import type { XPlanetsMap, XPlanetsConfig } from "./types.ts";

/**
 * Map configuration — all layer setup lives here in TypeScript.
 *
 * Edit this object to change the base imagery, add terrain sources,
 * switch projections, etc. No Rust recompilation needed.
 */
const MAP_CONFIG: XPlanetsConfig = {
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
  terrain: {
    url: "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
    encoding: "terrarium",
  },
};

async function main(): Promise<void> {
  // Load the WASM module and import the XPlanets factory.
  const { default: init, XPlanets } = await import("../pkg/x_planets_web.js");
  await init();

  // Create the map with our TS-defined config.
  const map: XPlanetsMap = await XPlanets.create(
    "x-planets-canvas",
    MAP_CONFIG,
  );

  // Create UI controls driven by the typed map API.
  setupControls(map);
}

main().catch(console.error);
