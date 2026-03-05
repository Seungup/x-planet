import type { XPlanetsMap } from "./types.ts";

/** Create the UI control buttons and wire them to the map API. */
export function setupControls(map: XPlanetsMap): void {
  createProjectionButton(map);
  createAltitudeButton(map);
}

function createButton(
  id: string,
  label: string,
  ariaLabel: string,
): HTMLButtonElement {
  const btn = document.createElement("button");
  btn.id = id;
  btn.className = "xp-btn";
  btn.textContent = label;
  btn.setAttribute("aria-label", ariaLabel);
  document.body.appendChild(btn);
  return btn;
}

function createProjectionButton(map: XPlanetsMap): void {
  const btn = createButton(
    "proj-btn",
    map.getProjection(),
    "Switch map projection",
  );

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const name = map.cycleProjection();
    btn.textContent = name;
  });
}

function createAltitudeButton(map: XPlanetsMap): void {
  const btn = createButton(
    "alt-btn",
    "Terrain OFF",
    "Toggle terrain altitude",
  );

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const enabled = map.toggleTerrain();
    updateAltButton(btn, enabled);
  });
}

function updateAltButton(btn: HTMLButtonElement, enabled: boolean): void {
  btn.textContent = enabled ? "Terrain ON" : "Terrain OFF";
  btn.classList.toggle("active", enabled);
}
