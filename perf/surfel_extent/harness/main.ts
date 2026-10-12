import {
  Box3, Color, Matrix4, NoToneMapping, PerspectiveCamera, Quaternion, SRGBColorSpace,
  Uint8BufferAttribute, Vector2, Vector3, WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { loadDockerViewerBundle } from "@viewer/docker-zip-loader";
import { productViewBox } from "@viewer/product-fit";
import { loadSurfels, type SurfelFrame } from "@viewer/surfel-loader";

type Arm = "baseline" | "uniform" | "candidate";
type View = "overview" | "close" | "source";
type VisibilityMode = "all" | "source-frame";
type OutputMode = "old" | "opaque";
type Coverage = {
  width: number; height: number; any: number; contributes: number; saturated: number; total: number;
  compositor: { startsAboveAlpha: number; reachesFullWeightAtAlpha: number };
};
type SurfelController = {
  render(): void;
  setSize(width: number, height: number): void;
  setRadius(size: number): void;
  readCoverage?(): Coverage;
  showCoverageMask?(): void;
  setOpaqueOutput?(enabled: boolean): void;
  setSourceSelection?(enabled: boolean): void;
  setSourceDepthTolerance?(value: number): void;
  dispose(): void;
};
type RendererFactory = (
  renderer: WebGLRenderer,
  camera: PerspectiveCamera,
  bundle: Awaited<ReturnType<typeof loadDockerViewerBundle>>,
  world: Matrix4,
  visibility: Uint8BufferAttribute,
) => SurfelController;

const SCENES = {
  video1: { label: "video1 full · 31 frames · Surfel v2", zip: "/bundles/video1.zip" },
  video2: { label: "video2 full · 12 frames · Surfel v3", zip: "/bundles/video2.zip" },
  video3: { label: "legacy video3 crop", zip: "/bundles/video3.zip" },
  fd6: { label: "legacy fd6 crop", zip: "/bundles/fd6.zip" },
} as const;
type SceneName = keyof typeof SCENES;

const params = new URLSearchParams(location.search);
const diagnosticKind = params.get("diagnostic");
const diagnosticEnabled = diagnosticKind === "source-frame" || diagnosticKind === "source-selection";
const requestedScene = params.get("scene") ?? "video1";
if (!(requestedScene in SCENES)) throw new Error("scene must be video1, video2, video3, or fd6");
const sceneName = requestedScene as SceneName;
const zip = params.get("zip") ?? SCENES[sceneName].zip;
const canvas = document.querySelector<HTMLCanvasElement>("#view")!;
const stage = document.querySelector<HTMLElement>("#stage")!;
const status = document.querySelector<HTMLElement>("#status")!;
const label = document.querySelector<HTMLElement>("#label")!;
const radiusInput = document.querySelector<HTMLInputElement>("#radius")!;
const sceneSelect = document.querySelector<HTMLSelectElement>("#scene")!;
const frameSelect = document.querySelector<HTMLSelectElement>("#frame")!;

sceneSelect.value = sceneName;
sceneSelect.addEventListener("change", () => {
  const url = new URL(location.href);
  url.searchParams.set("scene", sceneSelect.value);
  url.searchParams.delete("zip");
  url.searchParams.delete("frame");
  location.href = url.href;
});

const renderer = new WebGLRenderer({ canvas, antialias: false, alpha: false, preserveDrawingBuffer: true });
renderer.outputColorSpace = SRGBColorSpace;
renderer.toneMapping = NoToneMapping;
renderer.setPixelRatio(1);
renderer.setClearColor(new Color("#ffffff"), 1);
const camera = new PerspectiveCamera(50, 1, 0.01, 10000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = false;
controls.screenSpacePanning = true;

let arm: Arm = params.get("arm") === "baseline" ? "baseline" : params.get("arm") === "uniform" ? "uniform" : "candidate";
let view: View = params.get("view") === "close" ? "close" : params.get("view") === "source" ? "source" : "overview";
let radiusSize = Number(params.get("radiusSize") ?? "0.004");
let sourceFrameIndex = Math.max(0, Number.parseInt(params.get("frame") ?? "0", 10) || 0);
let visibilityMode: VisibilityMode = params.get("visibility") === "source-frame" ? "source-frame" : "all";
let outputMode: OutputMode = params.get("output") === "opaque" ? "opaque" : "old";
let sourceSelectionEnabled = params.get("sourceSelection") === "on";
let sourceDepthTolerance = Number(params.get("sourceDepthTolerance") ?? "0.005");
radiusInput.value = String(radiusSize);

const controllers = new Map<Arm, SurfelController>();
let bundle: Awaited<ReturnType<typeof loadDockerViewerBundle>> & { surfels: Awaited<ReturnType<typeof loadSurfels>> };
let visibility: Uint8BufferAttribute;
let world = new Matrix4();
let sceneCenter = new Vector3();
let productCenter = new Vector3();
let productSize = new Vector3(1, 1, 1);
let sceneSpan = 1;
let physicalWidth = 1;
let physicalHeight = 1;
let ready = false;
let disposed = false;
let suppressControlsRender = false;

function updateButtons() {
  document.querySelectorAll<HTMLButtonElement>("[data-arm]").forEach(button => button.ariaPressed = String(button.dataset.arm === arm));
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach(button => button.ariaPressed = String(button.dataset.view === view));
  const radius = Math.min(2, Math.max(0.6, 1.05 * radiusSize / 0.004));
  const armLabel = arm === "baseline" ? "Before" : arm === "uniform" ? "Uniform snapshot" : "After";
  const source = bundle?.surfels.metadata.frames[sourceFrameIndex];
  const diagnostic = diagnosticKind === "source-selection" ? ` · sourceSelection=${sourceSelectionEnabled ? "on" : "off"} · sourceDepthTolerance=${sourceDepthTolerance}`
    : diagnosticEnabled ? ` · visibility=${visibilityMode} · output=${outputMode}` : "";
  label.textContent = `${SCENES[sceneName].label} · ${armLabel} · ${view} · source frame ${source?.image_id ?? "-"} · pointSize=${radiusSize} · radius=${radius.toFixed(2)}${diagnostic}`;
}

async function factoryFor(next: Arm): Promise<RendererFactory> {
  if (next === "baseline") return (await import("@baseline-renderer")).createSurfelRenderer;
  if (next === "uniform") return (await import("@uniform-renderer")).createSurfelRenderer;
  if (diagnosticKind === "source-selection") return (await import("@source-selection-renderer")).createSurfelRenderer;
  return diagnosticKind === "source-frame" ? (await import("@diagnostic-renderer")).createSurfelRenderer
    : (await import("@candidate-renderer")).createSurfelRenderer;
}

async function controllerFor(next: Arm): Promise<SurfelController> {
  const existing = controllers.get(next);
  if (existing) return existing;
  status.textContent = `initializing ${next}`;
  const create = await factoryFor(next);
  const controller = create(renderer, camera, bundle, world, visibility);
  if (diagnosticEnabled) controller.setOpaqueOutput?.(outputMode === "opaque");
  if (diagnosticKind === "source-selection") controller.setSourceSelection?.(sourceSelectionEnabled);
  if (diagnosticKind === "source-selection") controller.setSourceDepthTolerance?.(sourceDepthTolerance);
  controller.setRadius(radiusSize);
  controller.setSize(physicalWidth, physicalHeight);
  controllers.set(next, controller);
  status.textContent = `${bundle.pointCount.toLocaleString()} surfels · ${controllers.size} renderer${controllers.size === 1 ? "" : "s"} ready`;
  return controller;
}

async function render() {
  if (!ready || disposed) return;
  updateButtons();
  (await controllerFor(arm)).render();
}

async function setArm(next: Arm) {
  arm = next;
  await render();
}

function sourcePose(frame: SurfelFrame) {
  const e = frame.extrinsic;
  const extrinsic = new Matrix4().set(
    ...e[0], ...e[1], ...e[2], 0, 0, 0, 1,
  );
  const cameraToWorld = extrinsic.invert();
  const position = new Vector3().setFromMatrixPosition(cameraToWorld).applyMatrix4(world);
  const rotation = new Quaternion().setFromRotationMatrix(cameraToWorld);
  const worldRotation = new Quaternion().setFromRotationMatrix(world);
  const direction = new Vector3(0, 0, 1).applyQuaternion(rotation).applyQuaternion(worldRotation).normalize();
  const up = new Vector3(0, -1, 0).applyQuaternion(rotation).applyQuaternion(worldRotation).normalize();
  return { position, direction, up };
}

function sourceFov(frame: SurfelFrame): number {
  const fy = frame.intrinsic[1][1];
  const gridHeight = bundle.surfels.metadata.grid_size[1];
  const fov = 2 * Math.atan(gridHeight / (2 * fy)) * 180 / Math.PI;
  if (!Number.isFinite(fov) || fov <= 10 || fov >= 120) {
    throw new Error(`Invalid source camera FOV for frame ${frame.image_id}: ${fov}`);
  }
  return fov;
}

function fittedProductDistance(direction: Vector3, up: Vector3, margin = 1.1): number {
  const right = direction.clone().cross(up).normalize();
  const cameraUp = right.clone().cross(direction).normalize();
  const tanVertical = Math.tan((camera.fov * Math.PI) / 360);
  const tanHorizontal = tanVertical * camera.aspect;
  let distance = 0;
  for (const x of [-0.5, 0.5]) for (const y of [-0.5, 0.5]) for (const z of [-0.5, 0.5]) {
    const offset = new Vector3(productSize.x * x, productSize.y * y, productSize.z * z);
    const along = offset.dot(direction);
    distance = Math.max(
      distance,
      margin * Math.abs(offset.dot(right)) / tanHorizontal - along,
      margin * Math.abs(offset.dot(cameraUp)) / tanVertical - along,
    );
  }
  return Math.max(distance, 0.1);
}

async function setView(next: View) {
  view = next;
  const frame = bundle.surfels.metadata.frames[sourceFrameIndex];
  const pose = sourcePose(frame);
  camera.fov = sourceFov(frame);
  camera.up.copy(pose.up);
  if (next === "source") {
    const depth = Math.max(productCenter.clone().sub(pose.position).dot(pose.direction), sceneSpan * 0.2, 0.1);
    camera.position.copy(pose.position);
    controls.target.copy(pose.position).addScaledVector(pose.direction, depth);
  } else {
    const distance = fittedProductDistance(pose.direction, pose.up) * (next === "close" ? 0.72 : 1);
    camera.position.copy(productCenter).addScaledVector(pose.direction, -distance);
    controls.target.copy(productCenter);
  }
  const distance = Math.max(camera.position.distanceTo(controls.target), 0.01);
  camera.near = Math.max(distance / 1000, 0.001);
  camera.far = Math.max(distance + sceneSpan * 8, 100);
  camera.lookAt(controls.target);
  camera.updateProjectionMatrix();
  suppressControlsRender = true;
  controls.update();
  suppressControlsRender = false;
  await render();
}

async function setSourceFrame(index: number) {
  sourceFrameIndex = Math.min(Math.max(Math.floor(index), 0), bundle.surfels.metadata.frames.length - 1);
  frameSelect.value = String(sourceFrameIndex);
  if (diagnosticEnabled) applyVisibility(visibilityMode);
  await setView(view);
}

function applyVisibility(next: VisibilityMode): number {
  if (!diagnosticEnabled && next !== "all") throw new Error("source-frame visibility is diagnostic-only");
  visibilityMode = next;
  const values = visibility.array as Uint8Array;
  if (next === "all") values.fill(1);
  else for (let index = 0; index < values.length; index += 1) values[index] = bundle.surfels.frames[index] === sourceFrameIndex ? 1 : 0;
  visibility.needsUpdate = true;
  let visible = 0;
  for (const value of values) visible += value;
  return visible;
}

async function setDiagnosticVisibility(next: VisibilityMode) {
  const visible = applyVisibility(next);
  await render();
  return visible;
}

async function setDiagnosticOutput(next: OutputMode) {
  if (!diagnosticEnabled) throw new Error("output mode is diagnostic-only");
  outputMode = next;
  const controller = await controllerFor(arm);
  if (!controller.setOpaqueOutput) throw new Error("active renderer has no output diagnostic");
  controller.setOpaqueOutput(next === "opaque");
  await render();
}

async function setDiagnosticSourceSelection(enabled: boolean) {
  if (diagnosticKind !== "source-selection") throw new Error("source selection is diagnostic-only");
  sourceSelectionEnabled = enabled;
  const controller = await controllerFor(arm);
  if (!controller.setSourceSelection) throw new Error("active renderer has no source-selection control");
  controller.setSourceSelection(enabled);
  await render();
}

async function setDiagnosticSourceDepthTolerance(value: number) {
  if (diagnosticKind !== "source-selection" || !Number.isFinite(value) || value <= 0) throw new Error("source-depth tolerance is diagnostic-only and must be positive");
  sourceDepthTolerance = value;
  const controller = await controllerFor(arm);
  if (!controller.setSourceDepthTolerance) throw new Error("active renderer has no source-depth tolerance control");
  controller.setSourceDepthTolerance(value);
  await render();
}

async function readDiagnosticCoverage() {
  if (!diagnosticEnabled) throw new Error("coverage readback is diagnostic-only");
  const controller = await controllerFor(arm);
  if (!controller.readCoverage) throw new Error("active renderer has no coverage readback");
  return controller.readCoverage();
}

async function showDiagnosticCoverageMask() {
  if (!diagnosticEnabled) throw new Error("coverage mask is diagnostic-only");
  const controller = await controllerFor(arm);
  if (!controller.showCoverageMask) throw new Error("active renderer has no coverage mask");
  controller.showCoverageMask();
}

async function setRadius(size: number) {
  radiusSize = size;
  radiusInput.value = String(size);
  for (const controller of controllers.values()) controller.setRadius(size);
  await render();
}

async function setState(nextArm: Arm, nextView: View, nextRadius = radiusSize, nextFrame = sourceFrameIndex) {
  arm = nextArm;
  radiusSize = nextRadius;
  radiusInput.value = String(nextRadius);
  for (const controller of controllers.values()) controller.setRadius(nextRadius);
  sourceFrameIndex = Math.min(Math.max(Math.floor(nextFrame), 0), bundle.surfels.metadata.frames.length - 1);
  frameSelect.value = String(sourceFrameIndex);
  await controllerFor(nextArm);
  await setView(nextView);
}

function resize() {
  const width = Math.max(stage.clientWidth, 1);
  const height = Math.max(stage.clientHeight, 1);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  const physical = renderer.getDrawingBufferSize(new Vector2());
  physicalWidth = physical.x;
  physicalHeight = physical.y;
  for (const controller of controllers.values()) controller.setSize(physicalWidth, physicalHeight);
  if (ready) void render();
}

async function run() {
  const blob = await (await fetch(zip)).blob();
  const base = await loadDockerViewerBundle(blob, "surfel");
  const surfels = await loadSurfels(base, fetch);
  bundle = { ...base, surfels };
  visibility = new Uint8BufferAttribute(new Uint8Array(bundle.pointCount).fill(1), 1);
  world = new Matrix4().fromArray(bundle.manifest.world_to_view).transpose();
  const bounds = bundle.manifest.display_bounds;
  const worldBox = new Box3(
    new Vector3(bounds[0], bounds[1], bounds[2]),
    new Vector3(bounds[3], bounds[4], bounds[5]),
  ).applyMatrix4(world);
  sceneCenter = worldBox.getCenter(new Vector3());
  sceneSpan = Math.max(worldBox.getSize(new Vector3()).length(), 0.1);
  const products = productViewBox(bundle.positions, bundle.objects, world);
  productCenter = products?.center.clone() ?? sceneCenter.clone();
  productSize = products?.size.clone() ?? worldBox.getSize(new Vector3());
  sourceFrameIndex = Math.min(sourceFrameIndex, surfels.metadata.frames.length - 1);
  if (diagnosticEnabled) applyVisibility(visibilityMode);
  for (let index = 0; index < surfels.metadata.frames.length; index += 1) {
    const frame = surfels.metadata.frames[index];
    frameSelect.add(new Option(`frame ${frame.image_id}`, String(index)));
  }
  frameSelect.value = String(sourceFrameIndex);
  document.querySelectorAll<HTMLButtonElement>("[data-arm]").forEach(button => button.addEventListener("click", () => void setArm(button.dataset.arm as Arm)));
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach(button => button.addEventListener("click", () => void setView(button.dataset.view as View)));
  radiusInput.addEventListener("change", () => void setRadius(Number(radiusInput.value)));
  frameSelect.addEventListener("change", () => void setSourceFrame(Number(frameSelect.value)));
  new ResizeObserver(resize).observe(stage);
  resize();
  await controllerFor(arm);
  ready = true;
  controls.addEventListener("change", () => { if (!suppressControlsRender) void render(); });
  await setView(view);
  await new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
  Object.assign(window, {
    __done: true,
    __setArm: setArm,
    __setView: setView,
    __setSourceFrame: setSourceFrame,
    __setRadius: setRadius,
    __setState: setState,
    __cameraState: () => ({
      scene: sceneName,
      arm,
      view,
      radiusSize,
      sourceFrameIndex,
      sourceImageId: bundle.surfels.metadata.frames[sourceFrameIndex].image_id,
      diagnosticEnabled,
      visibilityMode,
      outputMode: diagnosticKind === "source-frame" ? outputMode : arm === "candidate" ? "opaque" : null,
      sourceSelectionEnabled,
      sourceDepthTolerance,
      pointCount: bundle.pointCount,
      visibleCount: (visibility.array as Uint8Array).reduce((total, value) => total + value, 0),
      canvas: { width: canvas.width, height: canvas.height, clientWidth: canvas.clientWidth, clientHeight: canvas.clientHeight },
      fov: camera.fov,
      position: camera.position.toArray(),
      target: controls.target.toArray(),
      up: camera.up.toArray(),
      projection: camera.projectionMatrix.toArray(),
    }),
    __render: render,
    __setDiagnosticVisibility: setDiagnosticVisibility,
    __setDiagnosticOutput: setDiagnosticOutput,
    __setDiagnosticSourceSelection: setDiagnosticSourceSelection,
    __setDiagnosticSourceDepthTolerance: setDiagnosticSourceDepthTolerance,
    __readDiagnosticCoverage: readDiagnosticCoverage,
    __showDiagnosticCoverageMask: showDiagnosticCoverageMask,
    __dispose: () => {
      if (disposed) return;
      disposed = true;
      for (const controller of controllers.values()) controller.dispose();
      controllers.clear();
      bundle.disposeAssets?.();
      controls.dispose();
      renderer.dispose();
    },
  });
}

run().catch(error => {
  const message = String(error?.stack ?? error);
  (window as any).__error = message;
  status.textContent = "failed";
  console.error(error);
});
