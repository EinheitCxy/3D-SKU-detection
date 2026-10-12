import {
  DataArrayTexture, DataUtils, FloatType, Matrix4, NearestFilter, OrthographicCamera,
  RedFormat, RGBAFormat, SRGBColorSpace, Uint8BufferAttribute, UnsignedByteType, WebGLRenderer,
} from "three";
import { createSurfelRenderer } from "@candidate-renderer";

// Exercise the real renderer: a black opaque surface must stay black even where
// only one low-weight Gaussian contributes. Samples avoid the SMAA boundary.
const canvas = document.querySelector<HTMLCanvasElement>("#probe")!;
const renderer = new WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(512, 512, false);
renderer.outputColorSpace = SRGBColorSpace;
const camera = new OrthographicCamera(-0.1, 0.1, 0.1, -0.1, 0.01, 20);
camera.lookAt(0, 0, -1);
camera.updateMatrixWorld();
const side = 64;
const pixels = new Uint8Array(side * side * 4);
for (let i = 3; i < pixels.length; i += 4) pixels[i] = 255;
const texture = new DataArrayTexture(pixels, side, side, 1);
texture.format = RGBAFormat;
texture.type = UnsignedByteType;
texture.minFilter = texture.magFilter = NearestFilter;
texture.colorSpace = SRGBColorSpace;
texture.needsUpdate = true;
const depth = new DataArrayTexture(new Float32Array(side * side).fill(4), side, side, 1);
depth.format = RedFormat;
depth.type = FloatType;
depth.minFilter = depth.magFilter = NearestFilter;
depth.needsUpdate = true;
const half = (values: number[]) => Uint16Array.from(values, DataUtils.toHalfFloat);
const bundle = {
  pointCount: 1, positions: new Float32Array([0, 0, 4]), manifest: {}, objects: {},
  surfels: {
    u: half([0.01, 0, 0]), v: half([0, 0.01, 0]), scale: half([1, 1]), frames: new Uint8Array([0]),
    metadata: { version: 3, point_count: 1, grid_size: [side, side], frames: [{
      image_id: 0, texture: "synthetic", texture_size: [side, side], source_size: [side, side],
      intrinsic: [[400, 0, 32], [0, 400, 32], [0, 0, 1]],
      extrinsic: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
      processed_to_source: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }] },
    textures: texture, depths: depth, textureSize: [side, side],
  },
} as any;
const visibility = new Uint8BufferAttribute(new Uint8Array([1]), 1);
const controller = createSurfelRenderer(renderer, camera, bundle, new Matrix4().makeScale(1, -1, -1), visibility);
const readback = document.createElement("canvas");
readback.width = readback.height = 512;
const context = readback.getContext("2d", { willReadFrequently: true })!;
function sample(x: number, y: number): number[] {
  return Array.from(context.getImageData(x, y, 1, 1).data).slice(0, 3);
}
try {
  controller.setSize(512, 512);
  controller.setRadius(0.004);
  controller.render();
  context.drawImage(canvas, 0, 0);
  const center = sample(256, 256);
  const lowWeightInterior = [sample(277, 256), sample(234, 256), sample(256, 277)];
  const background = sample(0, 0);
  const centerPick = controller.pick(0.5, 0.5);
  const backgroundPick = controller.pick(0.05, 0.05);
  visibility.setX(0, 0);
  visibility.needsUpdate = true;
  controller.render();
  context.drawImage(canvas, 0, 0);
  const hiddenCenter = sample(256, 256);
  const checks = {
    center_stays_black: center.every(value => value <= 4),
    low_weight_interior_stays_black: lowWeightInterior.every(rgb => rgb.every(value => value <= 4)),
    background_stays_white: background.every(value => value >= 251),
    hidden_surfel_stays_white: hiddenCenter.every(value => value >= 251),
    center_pick_hits: centerPick === 0,
    background_pick_misses: backgroundPick === null,
  };
  const glErrors: number[] = [];
  const gl = renderer.getContext();
  for (let error = gl.getError(); error !== gl.NO_ERROR; error = gl.getError()) glErrors.push(error);
  Object.assign(window, { __done: true, __probe: { center, lowWeightInterior, background, hiddenCenter, checks, glErrors } });
} catch (error) {
  Object.assign(window, { __error: String(error) });
  console.error(error);
} finally {
  controller.dispose();
  renderer.dispose();
}
