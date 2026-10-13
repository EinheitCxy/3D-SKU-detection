import type { AreaMeasurementState } from "./area-measurement";

import type { FootprintState } from "./footprint-comparison";

interface AreaMeasurementController {
  getFootprintState(): FootprintState;
  setFootprintHandler(handler: ((state: FootprintState) => void) | null): void;
  saveAreaMeasurement(): void;
  clearSavedAreaMeasurements(): void;
  importOutlineJson(text: string, kind: "manual" | "candidate"): void;
  exportManualOutlines(): string;
  startAreaMeasurement(): void;
  stopAreaMeasurement(): void;
  closeAreaMeasurement(): void;
  undoAreaMeasurement(): void;
  clearAreaMeasurement(): void;
  getAreaMeasurementAvailability(): string | null;
  setAreaMeasurementHandler(handler: ((state: AreaMeasurementState) => void) | null): void;
}

export function createAreaMeasurementPanel(
  controller: AreaMeasurementController,
  onActiveChange: (active: boolean) => void,
): { element: HTMLElement; dispose(): void } {
  const element = document.createElement("section");
  element.className = "area-measurement";
  element.setAttribute("aria-label", "手工圈选水平面积");
  const toggle = makeButton("圈选面积");
  const panel = document.createElement("div");
  panel.id = "area-measurement-panel";
  panel.className = "area-measurement-panel";
  panel.hidden = true;
  toggle.setAttribute("aria-controls", panel.id);
  toggle.setAttribute("aria-expanded", "false");
  toggle.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
    toggle.setAttribute("aria-expanded", String(!panel.hidden));
  });
  const instructions = document.createElement("p");
  instructions.textContent = "俯视图中逐点圈选；点击起点或‘闭合区域’完成。右键拖动平移，滚轮缩放。";
  const availability = controller.getAreaMeasurementAvailability();
  const unavailable = document.createElement("p");
  unavailable.className = "area-measurement-error";
  unavailable.textContent = availability;
  unavailable.hidden = availability === null;
  const summary = document.createElement("p");
  summary.className = "area-measurement-summary";
  summary.setAttribute("aria-live", "polite");
  const error = document.createElement("p");
  error.className = "area-measurement-error";
  error.setAttribute("role", "alert");
  const actions = document.createElement("div");
  actions.className = "area-measurement-actions";
  const start = makeButton("开始圈选");
  const close = makeButton("闭合区域");
  const undo = makeButton("撤销");
  const redraw = makeButton("重画");
  const save = makeButton("保存当前货堆");
  const clearSaved = makeButton("清空已保存区域");
  const exportManual = makeButton("导出人工 JSON");
  const savedSummary = document.createElement("div");
  savedSummary.className = "footprint-comparison-summary";
  savedSummary.setAttribute("aria-live", "polite");
  const showError = (cause: unknown) => { error.textContent = cause instanceof Error ? cause.message : String(cause); error.hidden = false; };
  const perform = (action: () => void) => { try { action(); error.textContent = ""; error.hidden = true; } catch (cause) { showError(cause); } };
  save.addEventListener("click", () => perform(() => controller.saveAreaMeasurement()));
  clearSaved.addEventListener("click", () => perform(() => controller.clearSavedAreaMeasurements()));
  exportManual.addEventListener("click", () => perform(() => {
    const url = URL.createObjectURL(new Blob([controller.exportManualOutlines()], { type: "application/json" }));
    try {
      const link = document.createElement("a"); link.href = url; link.download = "manual-horizontal-outlines.json";
      document.body.append(link); link.click(); link.remove();
    } finally { URL.revokeObjectURL(url); }
  }));
  const imports = document.createElement("div");
  for (const [kind, labelText] of [["manual", "导入人工 JSON"], ["candidate", "导入候选 JSON"]] as const) {
    const label = document.createElement("label"); label.textContent = labelText;
    const input = document.createElement("input"); input.type = "file"; input.accept = ".json,application/json";
    input.setAttribute("aria-label", labelText); input.disabled = availability !== null;
    input.addEventListener("change", async () => {
      const file = input.files?.[0]; if (!file) return;
      try { const text = await file.text(); controller.importOutlineJson(text, kind); error.textContent = ""; error.hidden = true; }
      catch (cause) { showError(cause); }
      finally { input.value = ""; }
    });
    label.append(input); imports.append(label);
  }
  const renderSaved = (state: FootprintState) => {
    savedSummary.replaceChildren();
    const addLine = (text: string, className: string) => { const line = document.createElement("p"); line.className = className; line.textContent = text; savedSummary.append(line); };
    addLine(`人工已保存：${state.manual.regions.length} 区 · 合计 ${state.manual.total_area_m2.toFixed(4)} m²`, "manual-outline-summary");
    for (const region of state.manual.regions) addLine(`人工 ${region.id}：${region.area_m2.toFixed(4)} m²`, "manual-outline-summary");
    const candidate = state.candidate;
    if (candidate) {
      addLine(candidate.regions.length === 0 ? "自动候选：无有效区域（needs_review）" : `自动候选 · ${candidate.complete ? "完整候选合计" : "部分区域小计（needs_review）"}：${candidate.total_area_m2.toFixed(4)} m²`, "candidate-outline-summary");
      for (const region of candidate.regions) addLine(`候选 ${region.id}：${region.area_m2.toFixed(4)} m²`, "candidate-outline-summary");
      const diagnostics = candidate.diagnostics as Record<string, unknown>;
      const issues = Array.isArray(diagnostics.issues) ? diagnostics.issues : [];
      const details = document.createElement("details");
      const heading = document.createElement("summary"); heading.textContent = `候选参数与诊断（${issues.length} 条 issues）`;
      const content = document.createElement("pre"); content.textContent = JSON.stringify({ parameters: candidate.parameters, diagnostics }, null, 2);
      details.append(heading, content); savedSummary.append(details);
    }
    clearSaved.disabled = availability !== null || state.manual.regions.length === 0;
    exportManual.disabled = availability !== null;
  };
  controller.setFootprintHandler(renderSaved);
  const stop = makeButton("退出测量");
  start.addEventListener("click", () => controller.startAreaMeasurement());
  close.addEventListener("click", () => controller.closeAreaMeasurement());
  undo.addEventListener("click", () => controller.undoAreaMeasurement());
  redraw.addEventListener("click", () => {
    controller.clearAreaMeasurement();
    controller.startAreaMeasurement();
  });
  stop.addEventListener("click", () => controller.stopAreaMeasurement());
  actions.append(start, close, save, undo, redraw, stop, clearSaved, exportManual);
  panel.append(instructions, unavailable, summary, savedSummary, error, actions, imports);
  element.append(toggle, panel);
  const render = (state: AreaMeasurementState) => {
    const enabled = availability === null;
    toggle.textContent = state.active ? "圈选面积 · 测量中" : "圈选面积";
    start.disabled = !enabled;
    start.textContent = state.active || state.vertices.length > 0 ? "新圈选" : "开始圈选";
    save.disabled = !enabled || !state.active || !state.closed;
    close.disabled = !enabled || !state.active || state.closed || state.vertices.length < 3;
    undo.disabled = !enabled || !state.active || state.vertices.length === 0;
    redraw.disabled = !enabled || (!state.active && state.vertices.length === 0);
    stop.disabled = !state.active;
    summary.textContent = `顶点：${state.vertices.length}`;
    if (enabled && state.closed && state.areaM2 !== null) {
      summary.textContent += ` · 草稿水平面积：${state.areaM2.toFixed(4)} m²`;
    } else if (enabled) {
      summary.textContent += " · 闭合后显示面积";
    }
    error.textContent = state.error;
    error.hidden = state.error === null;
    onActiveChange(state.active);
  };
  render({ active: false, vertices: [], closed: false, areaM2: null, error: null });
  controller.setAreaMeasurementHandler(render);
  return { element, dispose: () => { controller.setAreaMeasurementHandler(null); controller.setFootprintHandler(null); } };
}

function makeButton(label: string): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  return button;
}
