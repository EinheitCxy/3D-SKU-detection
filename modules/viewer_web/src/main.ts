import { loadViewerBundle, type ViewerBundle } from "./bundle-loader";
import { dataCandidates } from "./data-candidates";
import { navigationState, stepVisibleId } from "./navigation";
import { buildSelectedObjectView, canFocusGlobalId, entryHasGeometry, listGlobalIds, summarizeObservationCounts } from "./presentation";
import { DEFAULT_POINT_SIZE, createViewerScene } from "./scene";
import { buildMasterDataFacets, filterGlobalIdsByMasterData, type MasterDataFacetKey } from "./masterdata-filters";
import { buildSkuFacets, buildSkuFacetsForGlobalIds, filterGlobalIdsBySku, type SkuFacet } from "./sku-filters";
import type { ObjectIndex, OrderedSku } from "./contracts";

interface BootstrapDependencies {
  readonly href: string;
  readonly load: typeof loadViewerBundle;
  readonly mount: (root: HTMLElement, bundle: ViewerBundle) => void;
}

export type SelectionMode = "manufacturer" | "brand" | "category" | "sku" | "global";

export interface SelectionModeTransition {
  readonly mode: SelectionMode;
  readonly searchQuery: string;
  readonly clearSelection: boolean;
}

export interface CanvasPickState {
  readonly mode: "global";
  readonly selectedFacetId: null;
  readonly selectedGlobalId: string;
}

export const selectionModeLabels = [
  "Manufacturer",
  "Brand",
  "Category",
  "SKU",
  "Global ID",
] as const;

export const selectionSummaryLabels = ["Total", "Removed"] as const;

const selectionModes = [
  { mode: "manufacturer", label: selectionModeLabels[0] },
  { mode: "brand", label: selectionModeLabels[1] },
  { mode: "category", label: selectionModeLabels[2] },
  { mode: "sku", label: selectionModeLabels[3] },
  { mode: "global", label: selectionModeLabels[4] },
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
    searchQuery: nextMode === "global" ? searchQuery : "",
    clearSelection: currentMode !== nextMode,
  };
}

export function selectionStateAfterCanvasPick(
  _mode: SelectionMode,
  _selectedFacetId: string | null,
  globalId: string,
): CanvasPickState {
  return { mode: "global", selectedFacetId: null, selectedGlobalId: globalId };
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
  root.replaceChildren(loadingMessage("Loading ViewerBundle…"));
  try {
    const href = dependencies?.href ?? window.location.href;
    const load = dependencies?.load ?? loadViewerBundle;
    const mount = dependencies?.mount ?? mountViewer;
    let bundle: ViewerBundle | null = null;
    const attempts: string[] = [];
    const errors: string[] = [];
    for (const baseUrl of dataCandidates(href)) {
      attempts.push(baseUrl);
      try {
        bundle = await load(baseUrl);
        break;
      } catch (error) {
        errors.push(`${baseUrl} => ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    if (bundle === null) {
      throw new Error(`Tried ${attempts.length} data roots but none succeeded.\n${errors.join("\n")}`);
    }
    mount(root, bundle);
  } catch (error) {
    const failure = document.createElement("section");
    failure.className = "load-error";
    failure.append(title("Viewer bundle failed to load"), text("p", "The data bundle could not be opened."));
    const detail = document.createElement("pre");
    detail.textContent = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    failure.append(detail);
    root.replaceChildren(failure);
  }
}

export function mountViewer(root: HTMLElement, bundle: ViewerBundle): void {
  const ids = listGlobalIds(bundle.objects);
  const shell = document.createElement("main");
  shell.className = "viewer-shell";

  const listPanel = document.createElement("aside");
  listPanel.className = "object-panel";
  listPanel.append(title("Selection", "h2"));
  const objectStats = document.createElement("div");
  objectStats.className = "object-stats";
  const totalValue = text("strong", String(ids.length));
  const globalObservationCounts = summarizeObservationCounts(
    Object.values(bundle.objects).flatMap((object) => object.observations),
  );
  objectStats.append(
    createStat(selectionSummaryLabels[0], totalValue),
    createStat(selectionSummaryLabels[1], text("strong", String(globalObservationCounts.removed))),
  );
  listPanel.append(objectStats);
  const modeButtons = document.createElement("div");
  modeButtons.className = "selection-mode-buttons";
  modeButtons.setAttribute("aria-label", "Selection mode");
  const modeButtonsByMode = new Map<SelectionMode, HTMLButtonElement>();
  for (const definition of selectionModes) {
    const item = button(definition.label);
    modeButtonsByMode.set(definition.mode, item);
    modeButtons.append(item);
  }

  const facetPane = document.createElement("section");
  facetPane.className = "selection-pane facet-pane";
  const facetHeading = title("", "h3");
  facetHeading.className = "facet-heading";
  const facetButtons = document.createElement("div");
  facetButtons.className = "sku-facet-scroll";
  facetPane.append(facetHeading, facetButtons);

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
  listPanel.append(modeButtons, facetPane, globalPane);

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
    <label>Point size <input type="range" min="0.004" max="0.07" step="0.001" value="${DEFAULT_POINT_SIZE.toFixed(3)}" data-control="point-size" /><span class="control-value">${DEFAULT_POINT_SIZE.toFixed(3)}</span></label>`;
  viewControls.append(controlsToggle, controlsPanel);
  sceneStage.append(canvasHost, hint, viewControls);

  const selectedPanel = document.createElement("aside");
  selectedPanel.className = "selected-panel";
  const selectedPanelHeading = title("Selected Object", "h2");
  selectedPanel.append(selectedPanelHeading);
  const selectedContent = document.createElement("div");
  selectedContent.className = "selected-content";
  selectedPanel.append(selectedContent);
  shell.append(listPanel, sceneStage, selectedPanel);
  root.replaceChildren(shell);

  const controller = createViewerScene(canvasHost, bundle);
  let selectionMode: SelectionMode = "sku";
  let selectedGlobalId: string | null = null;
  let selectedFacetId: string | null = null;
  let expandedSkuId: string | null = null;
  let searchQuery = "";

  const renderSelectionMode = () => {
    for (const definition of selectionModes) {
      const item = modeButtonsByMode.get(definition.mode);
      if (item === undefined) continue;
      const active = selectionMode === definition.mode;
      item.setAttribute("aria-pressed", String(active));
      item.classList.toggle("selected", active);
    }
    facetPane.hidden = selectionMode === "global";
    globalPane.hidden = selectionMode !== "global";
  };
  const renderSelectedObject = () => {
    if (selectedGlobalId === null || (selectedFacetId !== null && selectionMode !== "global")) {
      if (selectedFacetId !== null && selectionMode !== "global") {
        const matchedGlobalIds = globalIdsForFacet(selectedFacetId);
        const skuFacets = buildSkuFacetsForGlobalIds(bundle.objects, matchedGlobalIds);
        const selectionLabel = selectionModes.find((definition) => definition.mode === selectionMode)!.label;
        const context = text("p", selectionLabel);
        context.className = "facet-result-context";
        const selectedFacetHeading = title(selectedFacetId, "h3");
        selectedFacetHeading.className = "facet-result-heading";
        const summary = text("p", `${skuFacets.length} SKUs · ${matchedGlobalIds.length} Global IDs`);
        summary.className = "facet-result-summary";
        const skuHeading = title("SKU breakdown", "h3");
        skuHeading.className = "facet-result-sku-heading";
        const skuResults = document.createElement("ul");
        skuResults.className = "facet-sku-results";
        for (const facet of skuFacets) {
          const item = document.createElement("li");
          item.className = "facet-sku-result";
          const identity = document.createElement("div");
          identity.className = "sku-identity";
          const name = text("span", facet.skuName);
          name.className = "sku-name";
          const id = text("span", facet.skuId);
          id.className = "sku-id";
          identity.append(name, id);
          const skuButton = button("");
          skuButton.className = "sku-breakdown-button";
          skuButton.setAttribute("aria-expanded", String(expandedSkuId === facet.skuId));
          skuButton.append(identity, text("strong", String(facet.count)));
          const skuGlobalIds = filterGlobalIdsBySku(bundle.objects, facet.skuId)
            .filter((globalId) => matchedGlobalIds.includes(globalId));
          skuButton.addEventListener("click", () => {
            selectedGlobalId = null;
            expandedSkuId = expandedSkuId === facet.skuId ? null : facet.skuId;
            controller.selectGlobalIds(new Set(expandedSkuId === null ? matchedGlobalIds : skuGlobalIds));
            renderSelectedObject();
          });
          item.append(skuButton);
          if (expandedSkuId === facet.skuId) {
            const idList = document.createElement("div");
            idList.className = "sku-global-ids";
            idList.setAttribute("aria-label", "Global IDs");
            for (const globalId of skuGlobalIds) {
              const idButton = button(globalId);
              idButton.className = globalId === selectedGlobalId ? "object-item selected" : "object-item";
              idButton.setAttribute("aria-pressed", String(globalId === selectedGlobalId));
              idButton.addEventListener("click", () => selectGlobal(globalId, false));
              idList.append(idButton);
            }
            item.append(idList);
          }
          skuResults.append(item);
        }
        selectedPanelHeading.textContent = "Selection Details";
        selectedContent.replaceChildren(context, selectedFacetHeading, summary, skuHeading, skuResults);
        return;
      }
      selectedPanelHeading.textContent = "Selected Object";
      selectedContent.replaceChildren(text("p", "Choose a global ID or pick a point in the scene."));
      return;
    }
    selectedPanelHeading.textContent = "Selected Object";
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
    const primarySku = selected.orderedSkus[0];
    const masterData = primarySku === undefined ? undefined : bundle.skuMasterData[primarySku.sku_id];
    const masterDataFields = document.createElement("dl");
    if (masterData !== undefined) {
      masterDataFields.append(
        text("dt", "Manufacturer"), text("dd", displayMasterDataValue(masterData.manufacturer)),
        text("dt", "Brand"), text("dd", displayMasterDataValue(masterData.brand)),
        text("dt", "Category"), text("dd", displayMasterDataValue(masterData.category)),
        text("dt", "POSM"), text("dd", displayPosmValue(masterData.is_posm)),
      );
    }
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
    const content: HTMLElement[] = [fields, title("SKU", "h3"), skuList];
    if (masterData !== undefined) {
      content.push(title("Master Data", "h3"), masterDataFields);
    }
    content.push(observationsTitle, observationContent);
    selectedContent.replaceChildren(...content);
  };
  const globalIds = () => visibleGlobalIdsForFilters(ids, bundle.objects, searchQuery, null);
  const renderGlobalList = () => {
    const shown = globalIds();
    if (selectedGlobalId !== null && !shown.includes(selectedGlobalId)) {
      selectGlobal(null, false);
    }
    controller.setVisibleGlobalIds(new Set(shown));
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
  const facetsForMode = () => {
    switch (selectionMode) {
      case "sku":
        return buildSkuFacets(bundle.objects).map((facet) => ({
          id: facet.skuId,
          label: facet.skuName,
          count: facet.count,
        }));
      case "manufacturer":
      case "brand":
      case "category":
        return buildMasterDataFacets(bundle.objects, bundle.skuMasterData, selectionMode);
      case "global":
        return [];
    }
  };
  const globalIdsForFacet = (facetId: string) => {
    if (selectionMode === "sku") {
      return filterGlobalIdsBySku(bundle.objects, facetId);
    }
    if (selectionMode === "global") return [];
    return filterGlobalIdsByMasterData(
      bundle.objects,
      bundle.skuMasterData,
      selectionMode as MasterDataFacetKey,
      facetId,
    );
  };
  const renderFacetButtons = () => {
    const definition = selectionModes.find((item) => item.mode === selectionMode);
    if (definition === undefined || selectionMode === "global") return;
    facetHeading.textContent = definition.label;
    const all = button(`All ${definition.label}s`);
    all.className = selectedFacetId === null ? "facet selected" : "facet";
    all.addEventListener("click", () => selectFacet(null));
    const facets = facetsForMode().map((facet) => {
      const item = button("");
      const name = text("span", facet.label);
      name.className = "facet-name";
      const count = text("span", String(facet.count));
      count.className = "facet-count";
      item.append(name, count);
      item.className = selectedFacetId === facet.id ? "facet selected" : "facet";
      item.addEventListener("click", () => selectFacet(facet.id));
      return item;
    });
    facetButtons.replaceChildren(all, ...facets);
  };
  const selectFacet = (facetId: string | null) => {
    expandedSkuId = null;
    selectedFacetId = facetId;
    selectedGlobalId = null;
    controller.setVisibleGlobalIds(new Set(ids));
    controller.selectGlobalIds(new Set(facetId === null ? [] : globalIdsForFacet(facetId)));
    renderFacetButtons();
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
      selectedFacetId = null;
      expandedSkuId = null;
      controller.selectGlobalIds(new Set());
      controller.setVisibleGlobalIds(new Set(ids));
    }
    renderSelectionMode();
    renderFacetButtons();
    renderGlobalList();
    renderSelectedObject();
  };

  for (const definition of selectionModes) {
    modeButtonsByMode.get(definition.mode)?.addEventListener("click", () => switchMode(definition.mode));
  }
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
    const next = selectionStateAfterCanvasPick(selectionMode, selectedFacetId, globalId);
    selectionMode = next.mode;
    selectedFacetId = next.selectedFacetId;
    selectedGlobalId = next.selectedGlobalId;
    searchQuery = "";
    search.value = "";
    controller.setVisibleGlobalIds(new Set(ids));
    controller.selectGlobalId(globalId);
    renderSelectionMode();
    renderFacetButtons();
    renderGlobalList();
    renderSelectedObject();
  });
  renderSelectionMode();
  renderFacetButtons();
  renderGlobalList();
  renderSelectedObject();
}

function createStat(label: string, value: HTMLElement): HTMLElement {
  const item = document.createElement("p");
  item.append(text("span", label), value);
  return item;
}

function displayMasterDataValue(value: string | null): string {
  return value ?? "Not specified";
}

export function displayPosmValue(_isPosm: boolean): string {
  return "N/A";
}

export function skuFacetResultLabel(facet: Pick<SkuFacet, "skuId" | "skuName">): string {
  return `${facet.skuId} · ${facet.skuName}`;
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
