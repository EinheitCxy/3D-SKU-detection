import {
  Box3, Color, Matrix4, NoToneMapping, PerspectiveCamera, SRGBColorSpace,
  Uint8BufferAttribute, Vector2, Vector3, WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { loadDockerViewerBundle } from "@viewer/docker-zip-loader";
import { loadSurfels } from "@viewer/surfel-loader";

type Arm = "baseline" | "uniform" | "candidate" | "major";
type View = "overview" | "close" | "side";
type SurfelController = {
  render(): void; setSize(width: number, height: number): void; setRadius(size: number): void; dispose(): void;
};

const params = new URLSearchParams(location.search);
const sceneName = params.get("scene") ?? "video3";
if (sceneName !== "video3" && sceneName !== "fd6") throw new Error("scene must be video3 or fd6");
const zip = params.get("zip") ?? `/bundles/${sceneName}.zip`;
const canvas = document.querySelector<HTMLCanvasElement>("#view")!;
const stage = document.querySelector<HTMLElement>("#stage")!;
const status = document.querySelector<HTMLElement>("#status")!;
const label = document.querySelector<HTMLElement>("#label")!;
const radiusInput = document.querySelector<HTMLInputElement>("#radius")!;
const sceneSelect = document.querySelector<HTMLSelectElement>("#scene")!;
sceneSelect.value = sceneName === "fd6" ? "fd6" : "video3";
sceneSelect.addEventListener("change", () => {
  const url = new URL(location.href);
  url.searchParams.set("scene", sceneSelect.value);
  url.searchParams.delete("zip");
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
let view: View = "overview";
let radiusSize = Number(params.get("radiusSize") ?? "0.004");
radiusInput.value = String(radiusSize);
let controllers: Record<Arm, SurfelController>;
let center = new Vector3();
let span = 1;
let ready = false;

function updateButtons() {
  document.querySelectorAll<HTMLButtonElement>("[data-arm]").forEach(button => button.ariaPressed = String(button.dataset.arm === arm));
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach(button => button.ariaPressed = String(button.dataset.view === view));
  const radius = Math.min(2, Math.max(0.6, 1.05 * radiusSize / 0.004));
  const armLabel = arm === "baseline" ? "Before" : arm === "uniform" ? "Uniform" : arm === "candidate" ? "After (Axis cap)" : "Axis snapshot";
  label.textContent = `${sceneName} crop · ${armLabel} · ${view} · pointSize=${radiusSize} · radius=${radius.toFixed(2)}`;
}

function render() {
  if (!ready) return;
  controllers?.[arm].render();
  updateButtons();
}

function setArm(next: Arm) { arm = next; render(); }

function setView(next: View) {
  view = next;
  const direction = next === "side" ? new Vector3(1, 0.12, 0)
    : new Vector3(1, next === "close" ? 0.45 : 0.7, 1);
  direction.normalize();
  const fov = camera.fov * Math.PI / 180;
  const fitDistance = span / (2 * Math.tan(fov / 2));
  const distance = fitDistance * (next === "close" ? 0.48 : next === "side" ? 0.88 : 1.15);
  controls.target.copy(center);
  camera.position.copy(center).addScaledVector(direction, distance);
  camera.near = Math.max(distance / 1000, 0.001);
  camera.far = Math.max(distance * 20, 100);
  camera.updateProjectionMatrix();
  camera.lookAt(center);
  controls.update();
  render();
}

function setRadius(size: number) {
  radiusSize = size;
  radiusInput.value = String(size);
  controllers.baseline.setRadius(size);
  controllers.uniform.setRadius(size);
  controllers.candidate.setRadius(size);
  controllers.major.setRadius(size);
  render();
}

function setState(nextArm: Arm, nextView: View, nextRadius = radiusSize) {
  const wasReady = ready;
  ready = false;
  arm = nextArm;
  radiusSize = nextRadius;
  radiusInput.value = String(nextRadius);
  controllers.baseline.setRadius(nextRadius);
  controllers.uniform.setRadius(nextRadius);
  controllers.candidate.setRadius(nextRadius);
  controllers.major.setRadius(nextRadius);
  setView(nextView);
  ready = wasReady;
  render();
}

function resize() {
  const width = Math.max(stage.clientWidth, 1), height = Math.max(stage.clientHeight, 1);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  const physical = renderer.getDrawingBufferSize(new Vector2());
  controllers?.baseline.setSize(physical.x, physical.y);
  controllers?.uniform.setSize(physical.x, physical.y);
  controllers?.candidate.setSize(physical.x, physical.y);
  controllers?.major.setSize(physical.x, physical.y);
  if (controllers) render();
}

async function run() {
  const [
    { createSurfelRenderer: createBaseline }, { createSurfelRenderer: createUniform }, { createSurfelRenderer: createCandidate },
    { createSurfelRenderer: createMajor },
  ] = await Promise.all([
    import("@baseline-renderer"), import("@uniform-renderer"), import("@candidate-renderer"), import("@major-renderer"),
  ]);
  const blob = await (await fetch(zip)).blob();
  const base = await loadDockerViewerBundle(blob, "surfel");
  const surfels = await loadSurfels(base, fetch);
  const bundle = { ...base, surfels };
  const visibility = new Uint8BufferAttribute(new Uint8Array(bundle.pointCount).fill(1), 1);
  const world = new Matrix4().fromArray(bundle.manifest.world_to_view).transpose();
  controllers = {
    baseline: createBaseline(renderer, camera, bundle, world, visibility),
    uniform: createUniform(renderer, camera, bundle, world, visibility),
    candidate: createCandidate(renderer, camera, bundle, world, visibility),
    major: createMajor(renderer, camera, bundle, world, visibility),
  };
  controllers.baseline.setRadius(radiusSize);
  controllers.uniform.setRadius(radiusSize);
  controllers.candidate.setRadius(radiusSize);
  controllers.major.setRadius(radiusSize);
  const bounds = bundle.manifest.display_bounds;
  const worldBox = new Box3(new Vector3(bounds[0], bounds[1], bounds[2]), new Vector3(bounds[3], bounds[4], bounds[5])).applyMatrix4(world);
  center = worldBox.getCenter(new Vector3());
  span = Math.max(worldBox.getSize(new Vector3()).length(), 0.1);
  document.querySelectorAll<HTMLButtonElement>("[data-arm]").forEach(button => button.addEventListener("click", () => setArm(button.dataset.arm as Arm)));
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach(button => button.addEventListener("click", () => setView(button.dataset.view as View)));
  radiusInput.addEventListener("change", () => setRadius(Number(radiusInput.value)));
  new ResizeObserver(resize).observe(stage);
  resize();
  setView("overview");
  ready = true;
  controls.addEventListener("change", render);
  render();
  status.textContent = `${bundle.pointCount.toLocaleString()} surfels · on-demand`;
  Object.assign(window, {
    __done: true, __setArm: setArm, __setView: setView, __setRadius: setRadius, __setState: setState,
    __cameraState: () => ({
      canvas: { width: canvas.width, height: canvas.height, clientWidth: canvas.clientWidth, clientHeight: canvas.clientHeight },
      position: camera.position.toArray(), target: controls.target.toArray(),
      projection: camera.projectionMatrix.toArray(),
    }),
    __render: render,
    __dispose: () => {
      controllers.baseline.dispose(); controllers.uniform.dispose(); controllers.candidate.dispose(); controllers.major.dispose();
      bundle.disposeAssets?.(); renderer.dispose();
    },
  });
}

run().catch(error => {
  const message = String(error?.stack ?? error);
  (window as any).__error = message;
  status.textContent = "failed";
  console.error(error);
});
