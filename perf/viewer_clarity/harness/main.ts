import {
  Color, Matrix4, PerspectiveCamera, Scene, SRGBColorSpace, Uint8BufferAttribute,
  WebGLRenderer, NoToneMapping,
} from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { loadDockerViewerBundle } from "@viewer/docker-zip-loader";
import { loadSurfels } from "@viewer/surfel-loader";
import { createSurfelRenderer } from "@viewer/surfel-renderer";

const params = new URLSearchParams(location.search);
const mode = params.get("mode") ?? "surfel";
const frameIndex = Number(params.get("frame") ?? 0);
const scale = Number(params.get("scale") ?? 2);
const radiusSize = Number(params.get("radiusSize") ?? 0.004);
const zip = params.get("zip");
const glb = params.get("glb");
const canvas = document.querySelector<HTMLCanvasElement>("#view")!;
const renderer = new WebGLRenderer({ canvas, antialias: false, alpha: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.outputColorSpace = SRGBColorSpace;
renderer.toneMapping = NoToneMapping;
renderer.setClearColor(mode === "glb" ? 0xff00ff : 0xffffff, 1);
const scene = new Scene();
scene.background = new Color(mode === "glb" ? 0xff00ff : 0xffffff);
const camera = new PerspectiveCamera();
camera.matrixAutoUpdate = false;
let controller: ReturnType<typeof createSurfelRenderer> | undefined;

function setSourceCamera(metadata: any, frame: any, w: number, h: number, worldToView?: number[]) {
  const [fx, fy, cx0, cy0] = [frame.intrinsic[0][0], frame.intrinsic[1][1], frame.intrinsic[0][2], frame.intrinsic[1][2]];
  const cx = scale * (cx0 + 0.5), cy = scale * (cy0 + 0.5);
  const fxS = scale * fx, fyS = scale * fy;
  const n = 0.01, f = 100;
  camera.projectionMatrix.set(2 * fxS / w, 0, 1 - 2 * cx / w, 0,
    0, 2 * fyS / h, 2 * cy / h - 1, 0,
    0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
    0, 0, -1, 0);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  const e = frame.extrinsic.flat();
  const w2c = new Matrix4().set(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11], 0, 0, 0, 1);
  const axis = new Matrix4().makeScale(1, -1, -1);
  let view = axis.multiply(w2c);
  if (worldToView) view = view.multiply(new Matrix4().fromArray(worldToView).transpose().invert());
  camera.matrixWorldInverse.copy(view);
  camera.matrixWorld.copy(view).invert();
  camera.matrixWorld.decompose(camera.position, camera.quaternion, camera.scale);
  camera.matrixWorldInverse.copy(view);
  camera.matrixWorldNeedsUpdate = false;
  camera.updateMatrixWorld = () => undefined;
}

async function run() {
  if (mode === "glb") {
    if (!glb) throw new Error("glb query parameter is required");
    const loaded = await new GLTFLoader().loadAsync(glb);
    scene.add(loaded.scene);
    const metadataResponse = await fetch(params.get("manifest")!);
    const manifest = await metadataResponse.json();
    const metadata = await (await fetch(params.get("surfel")!)).json();
    const frame = metadata.frames[frameIndex];
    const [gw, gh] = metadata.grid_size;
    canvas.width = scale * gw; canvas.height = scale * gh;
    renderer.setSize(canvas.width, canvas.height, false);
    setSourceCamera(metadata, frame, canvas.width, canvas.height, manifest.world_to_view);
    renderer.render(scene, camera);
  } else {
    if (!zip) throw new Error("zip query parameter is required");
    const blob = await (await fetch(zip)).blob();
    const base = await loadDockerViewerBundle(blob, "surfel");
    const surfels = await loadSurfels(base, fetch);
    const bundle = { ...base, surfels };
    const visibility = new Uint8BufferAttribute(new Uint8Array(bundle.pointCount).fill(1), 1);
    const [gw, gh] = surfels.metadata.grid_size;
    canvas.width = scale * gw; canvas.height = scale * gh;
    renderer.setSize(canvas.width, canvas.height, false);
    setSourceCamera(surfels.metadata, surfels.metadata.frames[frameIndex], canvas.width, canvas.height);
    controller = createSurfelRenderer(renderer, camera, bundle, new Matrix4(), visibility);
    controller.setRadius(radiusSize);
    controller.setSize(canvas.width, canvas.height);
    controller.render();
    (window as any).__dispose = () => { controller?.dispose(); bundle.disposeAssets?.(); };
  }
  (window as any).__done = true;
}

run().catch((error) => { (window as any).__error = String(error?.stack ?? error); console.error(error); });
