import "./style.css";
import { setupControls } from "./controls.ts";
import { PRESETS } from "./presets.ts";
import type { XPlanetsMap, ExamplePreset } from "./types.ts";

/** Current map instance — destroyed and re-created on preset switch. */
let currentMap: XPlanetsMap | null = null;

/** Persist selected preset across reloads via URL hash. */
function getPresetId(): string {
  const hash = location.hash.slice(1);
  const found = PRESETS.find((p) => p.id === hash);
  return found ? found.id : PRESETS[0].id;
}

function setPresetId(id: string): void {
  history.replaceState(null, "", `#${id}`);
}

async function createMap(preset: ExamplePreset): Promise<XPlanetsMap> {
  const { default: init, XPlanets } = await import("../pkg/x_planets_web.js");
  await init();

  const map: XPlanetsMap = await XPlanets.create(
    "x-planets-canvas",
    preset.config,
  );
  return map;
}

/** Remove all UI controls created by setupControls. */
function clearControls(): void {
  for (const id of ["proj-btn", "alt-btn", "example-selector"]) {
    document.getElementById(id)?.remove();
  }
}

async function switchPreset(preset: ExamplePreset): Promise<void> {
  setPresetId(preset.id);

  // Destroy existing map
  if (currentMap) {
    currentMap.destroy();
    currentMap = null;
  }
  clearControls();

  // Create new map with selected config
  currentMap = await createMap(preset);

  // Set up UI with the new map
  setupControls(currentMap, PRESETS, preset.id, (newPreset) => {
    switchPreset(newPreset).catch(console.error);
  });

  // Apply terrain if configured (terrain starts enabled when config has terrain)
  if (preset.config.terrain) {
    currentMap.toggleTerrain();
  }
}

async function main(): Promise<void> {
  const presetId = getPresetId();
  const preset = PRESETS.find((p) => p.id === presetId) ?? PRESETS[0];
  await switchPreset(preset);
}

main().catch(console.error);
