import { loadViewerBundle, type ViewerBundle } from "./bundle-loader";
import { dataCandidates } from "./data-candidates";
import { navigationState, stepVisibleId } from "./navigation";
import { buildEvidenceView, canFocusGlobalId, entryHasGeometry, formatFormalMetric, listGlobalIds } from "./presentation";
import { createViewerScene, type ViewerSceneController } from "./scene";

interface BootstrapDependencies {
  readonly href: string;
  readonly load: typeof loadViewerBundle;
  readonly mount: (root: HTMLElement, bundle: ViewerBundle) => void;
}

if (typeof document !== "undefined") {
  const app = document.querySelector<HTMLElement>("#app");
  if (app === null) throw new Error("Viewer app root is missing");
  void bootstrap(app);
}

export async function bootstrap(root: HTMLElement, dependencies?: BootstrapDependencies): Promise<void> {
  root.replaceChildren(statusMessage("Loading validated ViewerBundle…"));
  try {
    const href = dependencies?.href ?? window.location.href;
    const load = dependencies?.load ?? loadViewerBundle;
    const mount = dependencies?.mount ?? mountViewer;
    let bundle: ViewerBundle | null = null;
    const attempts: Array<string> = [];
    const errors: Array<string> = [];
    for (const baseUrl of dataCandidates(href)) {
      attempts.push(baseUrl);
      try {
        bundle = await load(baseUrl);
        break;
      } catch (error) {
        const cause = error instanceof Error ? error.message : String(error);
        errors.push(`${baseUrl} => ${cause}`);
      }
    }
    if (bundle === null) throw new Error(`Tried ${attempts.length} data roots but none succeeded.\n${errors.join("\n")}`);
    mount(root, bundle);
  } catch (error) {
    const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    const failure = document.createElement("section");
    failure.className = "load-error";
    failure.append(title("Viewer bundle failed to load"), text("p", "The validated data bundle could not be opened."));
    const detail = document.createElement("pre");
    detail.textContent = message;
    failure.append(detail);
    root.replaceChildren(failure);
  }
}

function mountViewer(root: HTMLElement, bundle: ViewerBundle): void {
  const ids = listGlobalIds(bundle.objects);
  const shell = document.createElement("main");
  shell.className = "viewer-shell";

  const header = document.createElement("header");
  header.className = "status-bar";
  header.append(
    createStatusBadge("Dataset", `DA3 cache · ${bundle.manifest.source.da3_cache.frame_count} frames`),
    createStatusBadge("Backend", bundle.manifest.source.da3_cache.source_model),
    createStatusBadge("Points", bundle.manifest.point_count.toLocaleString()),
    createStatusBadge("Footprint", `${formatFormalMetric(bundle.footprints)} · ${bundle.footprints.status}`),
  );

  const listPanel = document.createElement("aside");
  listPanel.className = "object-panel";
  listPanel.append(title("Global IDs", "h2"));
  const searchArea = document.createElement("div");
  searchArea.className = "search-area";
  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = "Search global ID";
  search.setAttribute("aria-label", "Search global ID");
  const clearSearch = button("Clear");
  clearSearch.className = "search-clear";
  searchArea.append(search, clearSearch);
  const count = document.createElement("p");
  count.className = "object-count";
  const objectList = document.createElement("div");
  objectList.className = "object-list";
  const nav = document.createElement("div");
  nav.className = "object-nav";
  const previous = button("Prev");
  const next = button("Next");
  const clear = button("Clear");
  const focus = button("Focus");
  nav.append(previous, next, clear, focus);
  listPanel.append(searchArea, count, objectList, nav);

  const sceneWrap = document.createElement("section");
  sceneWrap.className = "scene-stage";
  const canvasHost = document.createElement("div");
  canvasHost.className = "canvas-host";
  const hint = text("p", "Drag to orbit · right-drag to pan · scroll to zoom");
  hint.className = "orbit-hint";
  const sceneControls = document.createElement("div");
  sceneControls.className = "scene-controls";
  sceneControls.innerHTML = `
    <div class=\"scene-controls-title\">View controls</div>
    <div class=\"preset-buttons\">
      <button class=\"toolbar-btn\" data-preset=\"fit\" type=\"button\">Fit</button>
      <button class=\"toolbar-btn\" data-preset=\"top\" type=\"button\">Top</button>
      <button class=\"toolbar-btn\" data-preset=\"isometric\" type=\"button\">Iso</button>
    </div>
    <label class=\"control-item\">Point size
      <input type=\"range\" min=\"0.004\" max=\"0.07\" step=\"0.002\" value=\"0.015\" data-control=\"point-size\" />
      <span class=\"control-value\">0.015</span>
    </label>
    <label class=\"control-item\">Footprint opacity
      <input type=\"range\" min=\"0.2\" max=\"1\" step=\"0.05\" value=\"1\" data-control=\"footprint-opacity\" />
      <span class=\"control-value\">1.00</span>
    </label>
  `;
  sceneWrap.append(canvasHost, hint, sceneControls);

  const drawer = document.createElement("aside");
  drawer.className = "evidence-drawer";
  const drawerTitle = title("Evidence");
  const panelCard = document.createElement("div");
  panelCard.className = "evidence-summary";
  const details = document.createElement("div");
  details.className = "evidence-content";
  drawer.append(drawerTitle, panelCard, details);

  shell.append(header, listPanel, sceneWrap, drawer);
  root.replaceChildren(shell);

  const controller = createViewerScene(canvasHost, bundle);
  let selectedGlobalId: string | null = null;
  let searchQuery = "";
  let drawerHidden = false;
  const setHidden = (hidden: boolean) => {
    drawerHidden = hidden;
    drawer.classList.toggle("evidence-drawer--hidden", hidden);
    sceneControls.classList.toggle("scene-controls--compact", hidden);
  };
  const visibleIds = () => ids.filter((globalId) => globalId.includes(searchQuery));
  const renderSummary = () => {
    const totalActive = ids.length ? ids.reduce((acc, id) => acc + bundle.objects[id].active_count, 0) : 0;
    const totalRemoved = ids.length ? ids.reduce((acc, id) => acc + bundle.objects[id].removed_count, 0) : 0;
    const metric = bundle.footprints.status === "accepted" ? formatFormalMetric(bundle.footprints) : "—";
    panelCard.replaceChildren(
      title("Footprint overview", "h3"),
      detailLine("metric", metric),
      detailLine("unit", "m²"),
      detailLine("objects", String(ids.length)),
      detailLine("active", String(totalActive)),
      detailLine("removed", String(totalRemoved)),
    );
    if (bundle.footprints.rejection_reason !== null) panelCard.append(detailLine("reason", bundle.footprints.rejection_reason));
  };

  const renderList = () => {
    const shown = visibleIds();
    count.textContent = `Showing ${shown.length} of ${ids.length}`;
    objectList.replaceChildren(
      ...shown.map((globalId) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = globalId === selectedGlobalId ? "object-item selected" : "object-item";
        if (!entryHasGeometry(bundle.objects[globalId])) {
          item.classList.add("object-item--no-geometry");
          const badge = document.createElement("span");
          badge.className = "no-geometry-badge";
          badge.textContent = "no geometry";
          item.append(document.createTextNode(globalId), badge);
        } else {
          item.textContent = globalId;
        }
        item.addEventListener("click", () => select(globalId, false));
        return item;
      }),
    );
    if (shown.length === 0) objectList.append(text("p", "No matching global ID."));
    const state = navigationState(shown, selectedGlobalId);
    previous.disabled = state.previousDisabled;
    next.disabled = state.nextDisabled;
    focus.disabled = selectedGlobalId === null || !canFocusGlobalId(bundle.objects[selectedGlobalId], bundle.footprints, selectedGlobalId);
    clear.disabled = selectedGlobalId === null;
  };

  const renderEvidence = () => {
    if (selectedGlobalId === null) {
      details.replaceChildren(
        title("Select a global ID", "h3"),
        text("p", "Choose one ID or click footprint mesh in the scene to inspect formal evidence."),
      );
      return;
    }
    const view = buildEvidenceView(bundle, selectedGlobalId);
    if (view === null) {
      details.replaceChildren(title("Not available", "h3"), text("p", "This ID has no evidence payload."));
      return;
    }
    const coverage = view.footprint.available ? `${(view.footprint.areaM2 ?? 0).toFixed(2)} m²` : "—";
    const observations = view.footprint.available ? String(view.footprint.observationsUsed) : "—";
    const fields: Array<readonly [string, string]> = [
      ["Global ID", view.globalId],
      ["Per-ID area", coverage],
      ["Observations", observations],
    ];
    const detailsList = document.createElement("dl");
    for (const [label, value] of fields) {
      detailsList.append(text("dt", label), text("dd", value));
    }
    const noGeometryNote = view.hasGeometry
      ? null
      : text(
          "p",
          "No geometry: every instance of this ID has an empty point range, so no 3D points are rendered or picked for it. The observations below still show what the detector saw.",
        );
    const observationsTitle = title("Observations", "h3");
    const thumbGrid = document.createElement("div");
    thumbGrid.className = "thumb-grid";
    for (const instance of view.instances) {
      const figure = document.createElement("figure");
      figure.className = instance.removed ? "thumb removed" : "thumb";
      const image = document.createElement("img");
      image.src = instance.thumbnailUrl;
      image.alt = `image ${instance.imageId} object ${instance.objectId}`;
      image.loading = "lazy";
      const caption = document.createElement("figcaption");
      caption.textContent = `image ${instance.imageId} · object ${instance.objectId}`;
      figure.append(image, caption);
      thumbGrid.append(figure);
    }
    details.replaceChildren(
      title("Object evidence", "h3"),
      detailsList,
      ...(noGeometryNote === null ? [] : [noGeometryNote]),
      observationsTitle,
      thumbGrid,
    );
  };

  const select = (globalId: string | null, focusCamera: boolean) => {
    selectedGlobalId = globalId;
    controller.selectGlobalId(globalId);
    if (focusCamera && globalId !== null) controller.focusGlobalId(globalId);
    renderList();
    renderEvidence();
  };

  const onSliderInput = (event: Event) => {
    const target = event.target as HTMLInputElement | null;
    if (target === null) return;
    const value = Number(target.value);
    const valueBadge = target.nextElementSibling;
    const precision = target.dataset.control === "footprint-opacity" ? 2 : 3;
    if (valueBadge !== null) valueBadge.textContent = value.toFixed(precision);
    if (target.dataset.control === "point-size") controller.setPointSize(value);
    if (target.dataset.control === "footprint-opacity") controller.setFootprintOpacity(value);
  };

  search.addEventListener("input", () => {
    searchQuery = search.value.trim();
    renderList();
  });
  clearSearch.addEventListener("click", () => {
    search.value = "";
    searchQuery = "";
    renderList();
    search.focus();
  });
  previous.addEventListener("click", () => {
    const nextId = stepVisibleId(visibleIds(), selectedGlobalId, -1);
    if (nextId !== null) select(nextId, true);
  });
  next.addEventListener("click", () => {
    const nextId = stepVisibleId(visibleIds(), selectedGlobalId, 1);
    if (nextId !== null) select(nextId, true);
  });
  clear.addEventListener("click", () => select(null, false));
  focus.addEventListener("click", () => {
    if (selectedGlobalId !== null) controller.focusGlobalId(selectedGlobalId);
  });

  const pointSize = sceneControls.querySelector<HTMLInputElement>('[data-control="point-size"]');
  const footprintOpacity = sceneControls.querySelector<HTMLInputElement>('[data-control="footprint-opacity"]');
  pointSize?.addEventListener("input", onSliderInput);
  footprintOpacity?.addEventListener("input", onSliderInput);
  for (const preset of sceneControls.querySelectorAll<HTMLButtonElement>("[data-preset]")) {
    preset.addEventListener("click", () => {
      controller.setViewPreset(preset.dataset.preset as "fit" | "top" | "isometric");
    });
  }

  const onKeyDown = (event: KeyboardEvent) => {
    if ((event.target instanceof HTMLInputElement) || (event.target instanceof HTMLTextAreaElement)) return;
    if (event.key === "Escape") select(null, false);
    if (event.key.toLowerCase() === "h") setHidden(!drawerHidden);
  };
  window.addEventListener("keydown", onKeyDown);
  controller.setFootprintPickHandler((globalId) => select(globalId, false));
  window.addEventListener("pagehide", () => {
    window.removeEventListener("keydown", onKeyDown);
    controller.dispose();
  }, { once: true });
  renderSummary();
  renderList();
  renderEvidence();
}

function createStatusBadge(label: string, value: string): HTMLElement {
  const item = document.createElement("div");
  item.className = "status-item";
  item.append(text("span", label), text("strong", value));
  return item;
}

function statusMessage(message: string): HTMLElement {
  const element = document.createElement("p");
  element.className = "loading-message";
  element.textContent = message;
  return element;
}

function title(value: string, level: "h2" | "h3" = "h2"): HTMLElement {
  return text(level, value);
}

function detailLine(label: string, value: string): HTMLElement {
  const row = document.createElement("p");
  const key = document.createElement("span");
  const val = document.createElement("strong");
  key.textContent = `${label}:`;
  val.textContent = value;
  row.append(key, val);
  return row;
}

function text(tag: keyof HTMLElementTagNameMap, value: string): HTMLElement {
  const element = document.createElement(tag);
  element.textContent = value;
  return element;
}

function button(label: string): HTMLButtonElement {
  const element = document.createElement("button");
  element.type = "button";
  element.textContent = label;
  return element;
}
