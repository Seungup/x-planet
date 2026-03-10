import "./style.css";
import { setupControls } from "./controls.ts";
import { PRESETS, hasCesiumToken } from "./presets.ts";
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
  for (const id of ["proj-btn", "alt-btn", "example-selector", "token-banner"]) {
    document.getElementById(id)?.remove();
  }
}

/** Show a banner prompting the user to set their Cesium Ion token. */
function showTokenBanner(): void {
  if (document.getElementById("token-banner")) return;
  const banner = document.createElement("div");
  banner.id = "token-banner";
  banner.style.cssText = `
    position:fixed; bottom:20px; left:50%; transform:translateX(-50%);
    background:rgba(30,30,30,0.95); color:#fff; padding:12px 20px;
    border-radius:8px; font-family:system-ui,sans-serif; font-size:14px;
    z-index:1000; max-width:500px; text-align:center;
    border:1px solid rgba(255,255,255,0.15);
  `;
  banner.innerHTML = `
    <strong>Cesium Ion token required</strong><br>
    <span style="opacity:0.8">
      3D Buildings needs a Cesium Ion access token.<br>
      Get one free at <a href="https://ion.cesium.com/tokens" target="_blank"
        style="color:#6df">ion.cesium.com/tokens</a>,
      then set <code style="background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:3px">VITE_CESIUM_ION_TOKEN</code> in your <code style="background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:3px">.env</code> file.
    </span>
  `;
  document.body.appendChild(banner);
}

async function switchPreset(preset: ExamplePreset): Promise<void> {
  setPresetId(preset.id);

  // Destroy existing map
  if (currentMap) {
    currentMap.destroy();
    currentMap = null;
  }
  clearControls();

  // Show token banner for 3D Buildings preset when no token is configured
  if (preset.id === "cesium-osm-buildings" && !hasCesiumToken()) {
    showTokenBanner();
  }

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
