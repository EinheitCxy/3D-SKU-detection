import type { AreaMeasurementState } from "./area-measurement";

interface AreaMeasurementController {
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
  const stop = makeButton("退出测量");
  start.addEventListener("click", () => controller.startAreaMeasurement());
  close.addEventListener("click", () => controller.closeAreaMeasurement());
  undo.addEventListener("click", () => controller.undoAreaMeasurement());
  redraw.addEventListener("click", () => {
    controller.clearAreaMeasurement();
    controller.startAreaMeasurement();
  });
  stop.addEventListener("click", () => controller.stopAreaMeasurement());
  actions.append(start, close, undo, redraw, stop);
  panel.append(instructions, unavailable, summary, error, actions);
  element.append(toggle, panel);
  const render = (state: AreaMeasurementState) => {
    const enabled = availability === null;
    toggle.textContent = state.active ? "圈选面积 · 测量中" : "圈选面积";
    start.disabled = !enabled || state.active;
    start.textContent = state.vertices.length > 0 ? "开始新测量" : "开始圈选";
    close.disabled = !enabled || !state.active || state.closed || state.vertices.length < 3;
    undo.disabled = !enabled || !state.active || state.vertices.length === 0;
    redraw.disabled = !enabled || (!state.active && state.vertices.length === 0);
    stop.disabled = !state.active;
    summary.textContent = `顶点：${state.vertices.length}`;
    if (enabled && state.closed && state.areaM2 !== null) {
      summary.textContent += ` · 水平面积：${state.areaM2.toFixed(4)} m²`;
    } else if (enabled) {
      summary.textContent += " · 闭合后显示面积";
    }
    error.textContent = state.error;
    error.hidden = state.error === null;
    onActiveChange(state.active);
  };
  render({ active: false, vertices: [], closed: false, areaM2: null, error: null });
  controller.setAreaMeasurementHandler(render);
  return { element, dispose: () => controller.setAreaMeasurementHandler(null) };
}

function makeButton(label: string): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  return button;
}
