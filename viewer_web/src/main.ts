import type { ViewerBundle } from "./bundle-loader";
import { loadDockerViewerBundle } from "./docker-zip-loader";
import { navigationState, stepVisibleId } from "./navigation";
import { buildSelectedObjectView, canFocusGlobalId, entryHasGeometry, formatDatasetSummary, listGlobalIds, summarizeObjectCounts, summarizeObservationCounts } from "./presentation";
import { createViewerScene } from "./scene";
import { buildSkuFacets, filterGlobalIdsBySku } from "./sku-filters";
import type { ObjectIndex, OrderedSku } from "./contracts";

interface BootstrapDependencies {
  readonly mount: (root: HTMLElement, bundle: ViewerBundle) => void;
  readonly loadZip?: typeof loadDockerViewerBundle;
}

export type SelectionMode = "sku" | "global";

export interface SelectionModeTransition {
  readonly mode: SelectionMode;
  readonly searchQuery: string;
  readonly clearSelection: boolean;
}

export interface CanvasPickState {
  readonly mode: "global";
  readonly selectedSkuId: null;
  readonly selectedGlobalId: string;
}

export const disabledFacetLabels = [
  "厂商：主数据待接入",
  "品牌：主数据待接入",
  "品类：主数据待接入",
  "POSM：检测能力待接入",
  "价签：检测能力待接入",
  "空缺位：检测能力待接入",
] as const;

const VIEW_CONTROLS_PANEL_ID = "scene-controls-panel";

export function configureViewControlsState(
  toggle: Pick<HTMLElement, "setAttribute">,
  panel: Pick<HTMLElement, "id" | "hidden">,
  expanded: boolean,
): void {
  panel.id = VIEW_CONTROLS_PANEL_ID;
  toggle.setAttribute("aria-controls", panel.id);
  toggle.setAttribute("aria-expanded", String(expanded));
  panel.hidden = !expanded;
}

export function selectionModeTransition(
  currentMode: SelectionMode,
  nextMode: SelectionMode,
  searchQuery: string,
): SelectionModeTransition {
  return {
    mode: nextMode,
    searchQuery: nextMode === "sku" ? "" : searchQuery,
    clearSelection: currentMode !== nextMode,
  };
}

export function selectionStateAfterCanvasPick(
  _mode: SelectionMode,
  _selectedSkuId: string | null,
  globalId: string,
): CanvasPickState {
  return { mode: "global", selectedSkuId: null, selectedGlobalId: globalId };
}

export function candidateLabel(candidate: OrderedSku): string {
  return `${candidate.sku_id} · ${candidate.sku_name}`;
}

export function visibleGlobalIdsForFilters(
  ids: readonly string[],
  objects: ObjectIndex,
  searchQuery: string,
  skuId: string | null,
): readonly string[] {
  return ids.filter(
    (globalId) => globalId.includes(searchQuery)
      && (skuId === null || objects[globalId]?.ordered_skus[0]?.sku_id === skuId),
  );
}

if (typeof document !== "undefined") {
  const app = document.querySelector<HTMLElement>("#app");
  if (app === null) throw new Error("Viewer app root is missing");
  void bootstrap(app);
}

export async function bootstrap(root: HTMLElement, dependencies?: BootstrapDependencies): Promise<void> {
  const loadZip = dependencies?.loadZip ?? loadDockerViewerBundle;
  const mount = dependencies?.mount ?? mountViewer;
  const pickerPanel = document.createElement("section");
  pickerPanel.className = "zip-picker";
  pickerPanel.append(title("Open Viewer Bundle"), text("p", "Select Docker's viewer_bundle.zip to open it locally."));
  const detail = document.createElement("pre");
  const picker = document.createElement("input");
  picker.type = "file";
  picker.accept = ".zip,application/zip";
  picker.setAttribute("aria-label", "Open viewer_bundle.zip");
  picker.onchange = async () => {
    const selected = picker.files?.[0];
    if (selected === undefined) return;
    picker.disabled = true;
    try {
      mount(root, await loadZip(selected));
    } catch (error) {
      detail.textContent = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    } finally {
      picker.value = "";
      picker.disabled = false;
    }
  };
  pickerPanel.append(detail, picker);
  root.replaceChildren(pickerPanel);
}

export function mountViewer(root: HTMLElement, bundle: ViewerBundle): void {
  const ids = listGlobalIds(bundle.objects);
  const shell = document.createElement("main");
  shell.className = "viewer-shell";

  const badges = document.createElement("div");
  badges.className = "dataset-badges";
  badges.append(
    createBadge("Dataset", formatDatasetSummary(bundle.manifest.dataset_name, bundle.manifest.frame_count)),
    createBadge("Backend", bundle.manifest.backend),
    createBadge("Points", bundle.pointCount.toLocaleString()),
  );

  const listPanel = document.createElement("aside");
  listPanel.className = "object-panel";
  listPanel.append(title("Selection", "h2"));
  const objectStats = document.createElement("div");
  objectStats.className = "object-stats";
  const totalValue = text("strong", String(ids.length));
  const visibleValue = text("strong", String(ids.length));
  objectStats.append(createStat("Total", totalValue), createStat("Visible", visibleValue));
  listPanel.append(objectStats);
  const globalObservationCounts = summarizeObservationCounts(
    Object.values(bundle.objects).flatMap((object) => object.observations),
  );
  const observationStats = document.createElement("div");
  observationStats.className = "object-stats observation-stats";
  observationStats.append(
    createStat("Observations", text("strong", String(globalObservationCounts.total))),
    createStat("Active", text("strong", String(globalObservationCounts.active))),
    createStat("Removed", text("strong", String(globalObservationCounts.removed))),
  );
  listPanel.append(observationStats);
  const modeButtons = document.createElement("div");
  modeButtons.className = "selection-mode-buttons";
  modeButtons.setAttribute("aria-label", "Selection mode");
  const skuModeButton = button("Select by SKU");
  const globalModeButton = button("Select by Global ID");
  modeButtons.append(skuModeButton, globalModeButton);

  const disabledFacets = document.createElement("div");
  disabledFacets.className = "disabled-facets";
  for (const label of disabledFacetLabels) {
    const item = button(label);
    item.className = "facet-placeholder";
    item.disabled = true;
    disabledFacets.append(item);
  }

  const skuPane = document.createElement("section");
  skuPane.className = "selection-pane";
  const skuButtons = document.createElement("div");
  skuButtons.className = "sku-facet-scroll";
  skuPane.append(skuButtons);

  const globalPane = document.createElement("section");
  globalPane.className = "selection-pane global-pane";
  globalPane.hidden = true;
  const searchArea = document.createElement("div");
  searchArea.className = "search-area";
  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = "Search global ID";
  search.setAttribute("aria-label", "Search global ID");
  const clearSearch = button("Clear");
  searchArea.append(search, clearSearch);
  const count = text("p", "");
  count.className = "object-count";
  const objectList = document.createElement("div");
  objectList.className = "object-list";
  const navigation = document.createElement("div");
  navigation.className = "object-nav";
  const previous = button("Prev");
  const next = button("Next");
  const clear = button("Clear");
  const focus = button("Focus");
  navigation.append(previous, next, clear, focus);
  globalPane.append(searchArea, count, objectList, navigation);
  listPanel.append(modeButtons, disabledFacets, skuPane, globalPane);

  const sceneStage = document.createElement("section");
  sceneStage.className = "scene-stage";
  const canvasHost = document.createElement("div");
  canvasHost.className = "canvas-host";
  const hint = text("p", "Drag to orbit · right-drag to pan · scroll to zoom");
  hint.className = "orbit-hint";
  const viewControls = document.createElement("div");
  viewControls.className = "scene-controls";
  let viewControlsExpanded = false;
  const controlsToggle = button("View controls");
  const controlsPanel = document.createElement("div");
  controlsPanel.className = "scene-controls-panel";
  configureViewControlsState(controlsToggle, controlsPanel, viewControlsExpanded);
  controlsPanel.innerHTML = `
    <div class="preset-buttons"><button data-preset="fit" type="button">Fit</button><button data-preset="top" type="button">Top</button><button data-preset="isometric" type="button">Iso</button></div>
    <label>Point size <input type="range" min="0.004" max="0.07" step="0.002" value="0.015" data-control="point-size" /><span class="control-value">0.015</span></label>`;
  viewControls.append(controlsToggle, controlsPanel);
  sceneStage.append(canvasHost, hint, viewControls);

  const selectedPanel = document.createElement("aside");
  selectedPanel.className = "selected-panel";
  selectedPanel.append(title("Selected Object", "h2"));
  const selectedContent = document.createElement("div");
  selectedContent.className = "selected-content";
  selectedPanel.append(selectedContent);
  shell.append(badges, listPanel, sceneStage, selectedPanel);
  root.replaceChildren(shell);

  const controller = createViewerScene(canvasHost, bundle);
  let selectionMode: SelectionMode = "sku";
  let selectedGlobalId: string | null = null;
  let selectedSkuId: string | null = null;
  let searchQuery = "";

  const renderSelectionMode = () => {
    const skuActive = selectionMode === "sku";
    skuModeButton.setAttribute("aria-pressed", String(skuActive));
    globalModeButton.setAttribute("aria-pressed", String(!skuActive));
    skuModeButton.classList.toggle("selected", skuActive);
    globalModeButton.classList.toggle("selected", !skuActive);
    skuPane.hidden = !skuActive;
    globalPane.hidden = skuActive;
  };
  const renderSelectedObject = () => {
    if (selectedGlobalId === null) {
      selectedContent.replaceChildren(text("p", "Choose a global ID or pick a point in the scene."));
      return;
    }
    const selected = buildSelectedObjectView(bundle.objects, selectedGlobalId, bundle.resolveAssetUrl);
    if (selected === null) return;
    const fields = document.createElement("dl");
    fields.append(text("dt", "Global ID"), text("dd", selected.globalId));
    const selectedObservationCounts = summarizeObservationCounts(selected.observations);
    fields.append(
      text("dt", "Observations"), text("dd", String(selectedObservationCounts.total)),
      text("dt", "Active"), text("dd", String(selectedObservationCounts.active)),
      text("dt", "Removed"), text("dd", String(selectedObservationCounts.removed)),
    );
    const skuList = document.createElement("ul");
    for (const sku of selected.orderedSkus) skuList.append(text("li", candidateLabel(sku)));
    const observationsTitle = title("Observations", "h3");
    const thumbGrid = document.createElement("div");
    thumbGrid.className = "thumb-grid";
    for (const observation of selected.observations) {
      const figure = document.createElement("figure");
      figure.className = observation.removed ? "thumb removed" : "thumb";
      const image = document.createElement("img");
      image.src = observation.thumbnailUrl;
      image.alt = `image ${observation.imageId} object ${observation.objectId}`;
      image.loading = "lazy";
      const caption = document.createElement("figcaption");
      caption.textContent = `image ${observation.imageId} · object ${observation.objectId}${observation.removed ? " · removed" : ""}`;
      figure.append(image, caption);
      thumbGrid.append(figure);
    }
    const observationContent = selected.observations.length === 0
      ? text("p", "No observations.")
      : thumbGrid;
    selectedContent.replaceChildren(fields, title("SKU", "h3"), skuList, observationsTitle, observationContent);
  };
  const globalIds = () => visibleGlobalIdsForFilters(ids, bundle.objects, searchQuery, null);
  const renderGlobalList = () => {
    const shown = globalIds();
    if (selectedGlobalId !== null && !shown.includes(selectedGlobalId)) {
      selectGlobal(null, false);
    }
    controller.setVisibleGlobalIds(new Set(shown));
    const stats = summarizeObjectCounts(bundle.objects, new Set(shown));
    totalValue.textContent = String(stats.total);
    visibleValue.textContent = String(stats.visible);
    count.textContent = `Matching ${shown.length} of ${ids.length}`;
    objectList.replaceChildren(...shown.map((globalId) => {
      const item = button(globalId);
      item.className = globalId === selectedGlobalId ? "object-item selected" : "object-item";
      if (!entryHasGeometry(bundle.objects[globalId])) item.classList.add("object-item--no-geometry");
      item.addEventListener("click", () => selectGlobal(globalId, false));
      return item;
    }));
    if (shown.length === 0) {
      objectList.append(text("p", "No matching global ID."));
    }
    const state = navigationState(shown, selectedGlobalId);
    previous.disabled = state.previousDisabled;
    next.disabled = state.nextDisabled;
    clear.disabled = selectedGlobalId === null;
    focus.disabled = selectedGlobalId === null
      || !canFocusGlobalId(bundle.objects[selectedGlobalId]);
  };
  const renderSkuButtons = () => {
    const all = button("All SKUs");
    all.className = selectedSkuId === null ? "facet selected" : "facet";
    all.addEventListener("click", () => selectSku(null));
    const facets = buildSkuFacets(bundle.objects).map((facet) => {
      const item = button(`${facet.skuName} (${facet.count})`);
      item.className = selectedSkuId === facet.skuId ? "facet selected" : "facet";
      item.title = facet.skuId;
      item.addEventListener("click", () => selectSku(facet.skuId));
      return item;
    });
    skuButtons.replaceChildren(all, ...facets);
  };
  const selectSku = (skuId: string | null) => {
    selectedSkuId = skuId;
    selectedGlobalId = null;
    controller.setVisibleGlobalIds(new Set(ids));
    controller.selectGlobalIds(new Set(skuId === null ? [] : filterGlobalIdsBySku(bundle.objects, skuId)));
    renderSkuButtons();
    renderGlobalList();
    renderSelectedObject();
  };
  const selectGlobal = (globalId: string | null, focusCamera: boolean) => {
    selectedGlobalId = globalId;
    controller.selectGlobalId(globalId);
    if (focusCamera && globalId !== null) controller.focusGlobalId(globalId);
    renderGlobalList();
    renderSelectedObject();
  };
  const switchMode = (nextMode: SelectionMode) => {
    const transition = selectionModeTransition(selectionMode, nextMode, searchQuery);
    selectionMode = transition.mode;
    searchQuery = transition.searchQuery;
    search.value = searchQuery;
    if (transition.clearSelection) {
      selectedGlobalId = null;
      selectedSkuId = null;
      controller.selectGlobalIds(new Set());
      controller.setVisibleGlobalIds(new Set(ids));
    }
    renderSelectionMode();
    renderSkuButtons();
    renderGlobalList();
    renderSelectedObject();
  };

  skuModeButton.addEventListener("click", () => switchMode("sku"));
  globalModeButton.addEventListener("click", () => switchMode("global"));
  search.addEventListener("input", () => {
    searchQuery = search.value;
    renderGlobalList();
  });
  clearSearch.addEventListener("click", () => {
    searchQuery = "";
    search.value = "";
    renderGlobalList();
  });
  previous.addEventListener("click", () => {
    const nextId = stepVisibleId(globalIds(), selectedGlobalId, -1);
    if (nextId !== null) {
      selectGlobal(nextId, false);
    }
  });
  next.addEventListener("click", () => {
    const nextId = stepVisibleId(globalIds(), selectedGlobalId, 1);
    if (nextId !== null) {
      selectGlobal(nextId, false);
    }
  });
  clear.addEventListener("click", () => selectGlobal(null, false));
  focus.addEventListener("click", () => {
    if (selectedGlobalId !== null) {
      controller.focusGlobalId(selectedGlobalId);
    }
  });
  controlsToggle.addEventListener("click", () => {
    viewControlsExpanded = !viewControlsExpanded;
    configureViewControlsState(controlsToggle, controlsPanel, viewControlsExpanded);
  });
  controlsPanel.addEventListener("click", (event) => {
    const target = event.target as HTMLElement;
    const preset = target.dataset.preset as "fit" | "top" | "isometric" | undefined;
    if (preset !== undefined) controller.setViewPreset(preset);
  });
  controlsPanel.querySelector<HTMLInputElement>('[data-control="point-size"]')?.addEventListener("input", (event) => {
    const target = event.target as HTMLInputElement;
    controller.setPointSize(Number(target.value));
    const value = controlsPanel.querySelector<HTMLElement>(".control-value");
    if (value !== null) value.textContent = Number(target.value).toFixed(3);
  });
  controller.setPointPickHandler((globalId) => {
    const next = selectionStateAfterCanvasPick(selectionMode, selectedSkuId, globalId);
    selectionMode = next.mode;
    selectedSkuId = next.selectedSkuId;
    selectedGlobalId = next.selectedGlobalId;
    searchQuery = "";
    search.value = "";
    controller.setVisibleGlobalIds(new Set(ids));
    controller.selectGlobalId(globalId);
    renderSelectionMode();
    renderSkuButtons();
    renderGlobalList();
    renderSelectedObject();
  });
  renderSelectionMode();
  renderSkuButtons();
  renderGlobalList();
  renderSelectedObject();
}

function createBadge(label: string, value: string): HTMLElement {
  const badge = document.createElement("div");
  badge.className = "dataset-badge";
  badge.append(text("span", label), text("strong", value));
  return badge;
}

function createStat(label: string, value: HTMLElement): HTMLElement {
  const item = document.createElement("p");
  item.append(text("span", label), value);
  return item;
}

function loadingMessage(message: string): HTMLElement {
  const element = text("p", message);
  element.className = "loading-message";
  return element;
}

function title(value: string, tag = "h2"): HTMLElement {
  return text(tag, value);
}

function text(tag: string, value: string): HTMLElement {
  const element = document.createElement(tag);
  element.textContent = value;
  return element;
}

function button(label: string): HTMLButtonElement {
  const item = document.createElement("button");
  item.type = "button";
  item.textContent = label;
  return item;
}
