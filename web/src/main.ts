import "./style.css";
import { setupControls } from "./controls.ts";
import type { XPlanetsMap } from "./types.ts";

async function main(): Promise<void> {
  // Load and initialize the WASM module.
  // wasm-pack --target web generates an init() that fetches the .wasm file
  // and calls #[wasm_bindgen(start)] which bootstraps GPU + rendering.
  const { default: init } = await import("../pkg/x_planets_web.js");
  await init();

  // The WASM start function spawns an async task that sets window.xplanets
  // once GPU initialization completes. Wait for the ready event.
  const map = await waitForReady();

  // Create UI controls (buttons) driven by the typed map API.
  setupControls(map);
}

function waitForReady(): Promise<XPlanetsMap> {
  return new Promise((resolve) => {
    // Already initialized (fast path).
    if (window.xplanets) {
      resolve(window.xplanets);
      return;
    }
    // Wait for the Rust side to signal readiness.
    window.addEventListener(
      "xplanets-ready",
      () => resolve(window.xplanets!),
      { once: true },
    );
  });
}

main().catch(console.error);
