import type { XPlanetsMap, ExamplePreset } from "./types.ts";

/** Create the UI control buttons and wire them to the map API. */
export function setupControls(
  map: XPlanetsMap,
  presets: ExamplePreset[],
  currentPresetId: string,
  onPresetSelect: (preset: ExamplePreset) => void,
): void {
  createProjectionButton(map);
  createAltitudeButton(map);
  createExampleSelector(presets, currentPresetId, onPresetSelect);
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

function createExampleSelector(
  presets: ExamplePreset[],
  currentId: string,
  onSelect: (preset: ExamplePreset) => void,
): void {
  const container = document.createElement("div");
  container.id = "example-selector";
  container.className = "xp-selector";

  const label = document.createElement("span");
  label.className = "xp-selector-label";
  label.textContent = "Examples";
  container.appendChild(label);

  const list = document.createElement("div");
  list.className = "xp-selector-list";
  list.style.display = "none";

  for (const preset of presets) {
    const item = document.createElement("button");
    item.className = "xp-selector-item";
    if (preset.id === currentId) {
      item.classList.add("active");
    }
    item.dataset.presetId = preset.id;

    const name = document.createElement("span");
    name.className = "xp-selector-item-name";
    name.textContent = preset.name;
    item.appendChild(name);

    const desc = document.createElement("span");
    desc.className = "xp-selector-item-desc";
    desc.textContent = preset.description;
    item.appendChild(desc);

    item.addEventListener("click", (e) => {
      e.stopPropagation();
      onSelect(preset);
    });

    list.appendChild(item);
  }

  container.appendChild(list);
  document.body.appendChild(container);

  // Toggle dropdown
  let open = false;
  label.addEventListener("click", (e) => {
    e.stopPropagation();
    open = !open;
    list.style.display = open ? "flex" : "none";
    container.classList.toggle("open", open);
  });

  // Close on outside click
  document.addEventListener("click", () => {
    if (open) {
      open = false;
      list.style.display = "none";
      container.classList.remove("open");
    }
  });

  container.addEventListener("click", (e) => e.stopPropagation());
}
