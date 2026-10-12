import {
  DataArrayTexture, DataUtils, FloatType, HalfFloatType, Matrix4, NearestFilter,
  OrthographicCamera, RedFormat, RGBAFormat, SRGBColorSpace, Uint8BufferAttribute,
  UnsignedByteType, WebGLRenderer,
} from "three";

type Factory = typeof import("@candidate-renderer").createSurfelRenderer;
const canvas = document.querySelector<HTMLCanvasElement>("#probe")!;
const renderer = new WebGLRenderer({ canvas, antialias: false, alpha: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(1); renderer.setSize(512, 512, false); renderer.outputColorSpace = SRGBColorSpace;
const camera = new OrthographicCamera(-0.1, 0.1, 0.1, -0.1, 0.01, 20);
camera.position.set(0, 0, 0); camera.lookAt(0, 0, -1); camera.updateMatrixWorld();
const halfs = (values: number[]) => Uint16Array.from(values, DataUtils.toHalfFloat);

function bundle(u: number[], v: number[], scale: [number, number]) {
  const width = 64, height = 64;
  const texture = new DataArrayTexture(new Uint8Array(width * height * 4), width, height, 1);
  for (let i = 3; i < texture.image.data.length; i += 4) texture.image.data[i] = 255;
  texture.format = RGBAFormat; texture.type = UnsignedByteType; texture.minFilter = texture.magFilter = NearestFilter; texture.needsUpdate = true;
  const depth = new DataArrayTexture(new Float32Array(width * height).fill(4), width, height, 1);
  depth.format = RedFormat;
  depth.type = FloatType; depth.minFilter = depth.magFilter = NearestFilter; depth.needsUpdate = true;
  const frame = {
    image_id: 0, texture: "synthetic", texture_size: [width, height] as [number, number], source_size: [width, height] as [number, number],
    intrinsic: [[400, 0, 32], [0, 400, 32], [0, 0, 1]],
    extrinsic: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
    processed_to_source: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  };
  return {
    pointCount: 1, positions: new Float32Array([0, 0, 4]), manifest: {}, objects: {},
    surfels: {
      u: halfs(u), v: halfs(v), scale: halfs(scale), frames: new Uint8Array([0]),
      metadata: { version: 3 as const, point_count: 1, grid_size: [width, height] as [number, number], frames: [frame] },
      textures: texture, depths: depth, textureSize: [width, height] as [number, number],
    },
  } as any;
}

function silhouette() {
  const context = document.createElement("canvas").getContext("2d", { willReadFrequently: true })!;
  context.canvas.width = canvas.width; context.canvas.height = canvas.height;
  context.drawImage(canvas, 0, 0);
  const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
  let left = canvas.width, right = -1, top = canvas.height, bottom = -1, pixels = 0;
  for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) {
    const offset = (y * canvas.width + x) * 4;
    if (data[offset] < 250 || data[offset + 1] < 250 || data[offset + 2] < 250) {
      left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y); pixels++;
    }
  }
  return pixels ? { width: right-left+1, height: bottom-top+1, pixels, bounds: [left, top, right, bottom] } : { width: 0, height: 0, pixels: 0, bounds: null };
}

async function measure(factory: Factory, spec: { u:number[]; v:number[]; scale:[number,number]; radiusSize:number }) {
  const data = bundle(spec.u, spec.v, spec.scale);
  const visibility = new Uint8BufferAttribute(new Uint8Array([1]), 1);
  const controller = factory(renderer, camera, data, new Matrix4().makeScale(1, -1, -1), visibility);
  controller.setRadius(spec.radiusSize); controller.setSize(512, 512); controller.render();
  const result = {
    ...silhouette(),
    picks: { center: controller.pick(0.5, 0.5), edge: controller.pick(0.9, 0.5) },
  };
  controller.dispose();
  return result;
}

async function run() {
  const [
    { createSurfelRenderer: baseline }, { createSurfelRenderer: candidate },
    { createSurfelRenderer: rootBaseline }, { createSurfelRenderer: rootCandidate },
    { createSurfelRenderer: major },
  ] = await Promise.all([
    import("@baseline-renderer"), import("@candidate-renderer"),
    import("@root-baseline-renderer"), import("@root-candidate-renderer"), import("@major-renderer"),
  ]);
  const radius2Size = 0.004 * 2 / 1.05;
  const cases = {
    scale8_radius2: { u:[0.01,0,0], v:[0,0.01,0], scale:[8,0.5] as [number,number], radiusSize:radius2Size },
    skewed_radius2: { u:[0.018,0,0], v:[0.016,0.008,0], scale:[1,1] as [number,number], radiusSize:radius2Size },
    ordinary: { u:[0.01,0,0], v:[0,0.01,0], scale:[1,1] as [number,number], radiusSize:0.004 },
  };
  const results: Record<string, any> = {};
  for (const [name, spec] of Object.entries(cases)) {
    results[name] = {
      baseline: await measure(baseline, spec),
      candidate: await measure(candidate, spec),
      rootBaseline: await measure(rootBaseline, spec),
      rootCandidate: await measure(rootCandidate, spec),
      major: await measure(major, spec),
    };
  }
  const pxPerUnit = canvas.width / 0.2;
  const capPixels = 6 * 0.01 * pxPerUnit;
  const tolerancePixels = 4;
  const maxDimension = (value: any) => Math.max(value.width, value.height);
  const checks = {
    scale8_bounded: maxDimension(results.scale8_radius2.candidate) <= capPixels + tolerancePixels,
    skewed_bounded: maxDimension(results.skewed_radius2.candidate) <= capPixels + tolerancePixels,
    ordinary_unchanged: Math.abs(results.ordinary.baseline.width-results.ordinary.candidate.width) <= 1
      && Math.abs(results.ordinary.baseline.height-results.ordinary.candidate.height) <= 1
      && results.ordinary.baseline.pixels === results.ordinary.candidate.pixels,
    stress_reduced: maxDimension(results.scale8_radius2.candidate) < maxDimension(results.scale8_radius2.baseline),
    short_axis_preserved: results.scale8_radius2.candidate.height === results.scale8_radius2.baseline.height,
    skewed_reduced: maxDimension(results.skewed_radius2.candidate) < maxDimension(results.skewed_radius2.baseline),
    root_scale8_bounded: maxDimension(results.scale8_radius2.rootCandidate) <= capPixels + tolerancePixels,
    root_skewed_bounded: maxDimension(results.skewed_radius2.rootCandidate) <= capPixels + tolerancePixels,
    root_stress_reduced: maxDimension(results.scale8_radius2.rootCandidate) < maxDimension(results.scale8_radius2.rootBaseline),
    root_short_axis_preserved: results.scale8_radius2.rootCandidate.height === results.scale8_radius2.rootBaseline.height,
    root_skewed_reduced: maxDimension(results.skewed_radius2.rootCandidate) < maxDimension(results.skewed_radius2.rootBaseline),
    root_ordinary_unchanged: Math.abs(results.ordinary.rootBaseline.width-results.ordinary.rootCandidate.width) <= 1
      && Math.abs(results.ordinary.rootBaseline.height-results.ordinary.rootCandidate.height) <= 1
      && results.ordinary.rootBaseline.pixels === results.ordinary.rootCandidate.pixels,
    all_silhouettes_nonempty: Object.values(results).every(arms =>
      Object.values(arms as Record<string, any>).every(value => value.pixels > 0)),
    scale8_pick_center_hits: ["baseline", "candidate", "rootBaseline", "rootCandidate", "major"].every(
      arm => results.scale8_radius2[arm].picks.center === 0),
    scale8_pick_edge_boundary: results.scale8_radius2.baseline.picks.edge === 0
      && results.scale8_radius2.rootBaseline.picks.edge === 0
      && results.scale8_radius2.candidate.picks.edge === null
      && results.scale8_radius2.rootCandidate.picks.edge === null
      && results.scale8_radius2.major.picks.edge === null,
    major_scale8_bounded: maxDimension(results.scale8_radius2.major) <= capPixels + tolerancePixels,
    major_skewed_bounded: maxDimension(results.skewed_radius2.major) <= capPixels + tolerancePixels,
    major_ordinary_unchanged: results.ordinary.baseline.width === results.ordinary.major.width
      && results.ordinary.baseline.height === results.ordinary.major.height
      && results.ordinary.baseline.pixels === results.ordinary.major.pixels,
  };
  const glErrors: number[] = [];
  const gl = renderer.getContext();
  for (let error = gl.getError(); error !== gl.NO_ERROR; error = gl.getError()) glErrors.push(error);
  Object.assign(window, { __done:true, __probe: { capPixels, tolerancePixels, cases:results, checks, glErrors } });
  renderer.dispose();
}
run().catch(error => { (window as any).__error=String(error?.stack??error); console.error(error); });
