import { loadViewerBundle, type ViewerBundle } from "./bundle-loader";
import { buildEvidenceView, formatFormalMetric, listGlobalIds } from "./presentation";
import { createViewerScene, type ViewerSceneController } from "./scene";

const app = document.querySelector<HTMLElement>("#app");
if (app === null) throw new Error("Viewer app root is missing");

void bootstrap(app);

async function bootstrap(root: HTMLElement): Promise<void> {
  root.replaceChildren(statusMessage("Loading validated ViewerBundle…"));
  try {
    const bundle = await loadViewerBundle("/data/");
    mountViewer(root, bundle);
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
    statusItem("Dataset", `DA3 cache · ${bundle.manifest.source.da3_cache.frame_count} frames`),
    statusItem("Backend", bundle.manifest.source.da3_cache.source_model),
    statusItem("Points", bundle.manifest.point_count.toLocaleString()),
    statusItem("Formal footprint", `${formatFormalMetric(bundle.footprints)} · ${bundle.footprints.status}`),
  );

  const listPanel = document.createElement("aside");
  listPanel.className = "object-panel";
  listPanel.append(title("Global IDs"));
  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = "Search global ID";
  search.setAttribute("aria-label", "Search global ID");
  const objectList = document.createElement("div");
  objectList.className = "object-list";
  const nav = document.createElement("div");
  nav.className = "object-nav";
  const previous = button("Prev");
  const next = button("Next");
  const clear = button("Clear");
  nav.append(previous, next, clear);
  listPanel.append(search, objectList, nav);

  const stage = document.createElement("section");
  stage.className = "scene-stage";
  const canvasHost = document.createElement("div");
  canvasHost.className = "canvas-host";
  const hint = text("p", "Drag to orbit · right-drag to pan · scroll to zoom");
  hint.className = "orbit-hint";
  stage.append(canvasHost, hint);

  const drawer = document.createElement("aside");
  drawer.className = "evidence-drawer";
  drawer.append(title("Evidence"));
  const evidence = document.createElement("div");
  evidence.className = "evidence-content";
  drawer.append(evidence);
  shell.append(header, listPanel, stage, drawer);
  root.replaceChildren(shell);

  const controller = createViewerScene(canvasHost, bundle);
  let selectedGlobalId: string | null = null;
  let searchQuery = "";
  const visibleIds = () => ids.filter((globalId) => globalId.includes(searchQuery));
  const select = (globalId: string | null, focus: boolean) => {
    selectedGlobalId = globalId;
    controller.selectGlobalId(globalId);
    if (globalId !== null && focus && bundle.footprints.per_global_id[globalId] !== undefined) controller.focusGlobalId(globalId);
    renderList();
    renderEvidence();
  };
  const renderList = () => {
    const shown = visibleIds();
    objectList.replaceChildren(...shown.map((globalId) => {
      const item = button(globalId);
      item.className = globalId === selectedGlobalId ? "object-item selected" : "object-item";
      item.addEventListener("click", () => select(globalId, true));
      return item;
    }));
    if (shown.length === 0) objectList.append(text("p", "No matching global ID."));
    const selectedIndex = selectedGlobalId === null ? -1 : ids.indexOf(selectedGlobalId);
    previous.disabled = selectedIndex <= 0;
    next.disabled = selectedIndex < 0 || selectedIndex >= ids.length - 1;
  };
  const renderEvidence = () => {
    const view = selectedGlobalId === null ? null : buildEvidenceView(bundle, selectedGlobalId);
    if (view === null) {
      evidence.replaceChildren(text("p", "Select a global ID to inspect formal evidence and observations."));
      return;
    }
    const footprint = view.footprint.available
      ? `${view.footprint.areaM2?.toFixed(2)} m²`
      : "—";
    const observations = view.footprint.available ? String(view.footprint.observationsUsed) : "—";
    const fields: Array<readonly [string, string]> = [
      ["Metric", bundle.footprints.metric], ["Unit", "m²"], ["Formal status", bundle.footprints.status],
      ["Run", bundle.footprints.run_id], ["Global ID", view.globalId], ["Per-ID area", footprint],
      ["Observations", observations], ["Active", String(view.object.active_count)], ["Removed", String(view.object.removed_count)],
    ];
    if (bundle.footprints.rejection_reason !== null) fields.push(["Rejection reason", bundle.footprints.rejection_reason]);
    const details = document.createElement("dl");
    for (const [label, value] of fields) {
      const term = text("dt", label);
      const definition = text("dd", value);
      details.append(term, definition);
    }
    const observationsTitle = title("Image / object / bbox", "h3");
    const observationsList = document.createElement("ul");
    for (const instance of view.object.instances) {
      const state = instance.removed ? "removed" : "active";
      observationsList.append(text("li", `image ${instance.image_id} · object ${instance.object_id} · bbox [${instance.bbox.join(", ")}] · ${state}`));
    }
    evidence.replaceChildren(details, observationsTitle, observationsList);
  };

  search.addEventListener("input", () => {
    searchQuery = search.value.trim();
    renderList();
  });
  previous.addEventListener("click", () => {
    const index = selectedGlobalId === null ? -1 : ids.indexOf(selectedGlobalId);
    if (index > 0) select(ids[index - 1], true);
  });
  next.addEventListener("click", () => {
    const index = selectedGlobalId === null ? -1 : ids.indexOf(selectedGlobalId);
    if (index >= 0 && index < ids.length - 1) select(ids[index + 1], true);
  });
  clear.addEventListener("click", () => select(null, false));
  controller.setFootprintPickHandler((globalId) => select(globalId, true));
  window.addEventListener("pagehide", () => controller.dispose(), { once: true });
  renderList();
  renderEvidence();
}

function statusMessage(message: string): HTMLElement {
  const element = document.createElement("p");
  element.className = "loading-message";
  element.textContent = message;
  return element;
}

function statusItem(label: string, value: string): HTMLElement {
  const item = document.createElement("div");
  item.className = "status-item";
  item.append(text("span", label), text("strong", value));
  return item;
}

function title(value: string, level: "h2" | "h3" = "h2"): HTMLElement {
  return text(level, value);
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
