import {
  DataArrayTexture, DataUtils, FloatType, Matrix4, NearestFilter,
  OrthographicCamera, PerspectiveCamera, RedFormat, RGBAFormat, SRGBColorSpace,
  Uint8BufferAttribute, UnsignedByteType, WebGLRenderer,
} from "three";
import type { Camera } from "three";
import { createSurfelRenderer } from "@source-selection-renderer";

// Exercise source selection with two coincident surfels. Their geometry,
// source-camera pose, and resolution are identical, so the deterministic
// frame-index tie break must choose red frame/slot 0 over blue frame/slot 1.
const canvas = document.querySelector<HTMLCanvasElement>("#probe")!;
const renderer = new WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(256, 256, false);
renderer.outputColorSpace = SRGBColorSpace;
const side = 64;
const half = (values: number[]) => Uint16Array.from(values, DataUtils.toHalfFloat);
const worldMatrix = new Matrix4().makeScale(1, -1, -1);
const readback = document.createElement("canvas");
readback.width = readback.height = 256;
const context = readback.getContext("2d", { willReadFrequently: true })!;

function makeBundle() {
  const pixels = new Uint8Array(side * side * 4 * 2);
  for (let layer = 0; layer < 2; layer += 1) {
    for (let pixel = 0; pixel < side * side; pixel += 1) {
      const offset = (layer * side * side + pixel) * 4;
      pixels[offset + (layer === 0 ? 0 : 2)] = 255;
      pixels[offset + 3] = 255;
    }
  }
  const texture = new DataArrayTexture(pixels, side, side, 2);
  texture.format = RGBAFormat;
  texture.type = UnsignedByteType;
  texture.minFilter = texture.magFilter = NearestFilter;
  texture.colorSpace = SRGBColorSpace;
  texture.needsUpdate = true;
  const depth = new DataArrayTexture(new Float32Array(side * side * 2).fill(4), side, side, 2);
  depth.format = RedFormat;
  depth.type = FloatType;
  depth.minFilter = depth.magFilter = NearestFilter;
  depth.needsUpdate = true;
  const frame = (imageId: number) => ({
    image_id: imageId, texture: `synthetic-${imageId}`, texture_size: [side, side], source_size: [side, side],
    intrinsic: [[400, 0, 32], [0, 400, 32], [0, 0, 1]],
    extrinsic: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
    processed_to_source: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  });
  return {
    pointCount: 2,
    positions: new Float32Array([0, 0, 4, 0, 0, 4]),
    manifest: {}, objects: {},
    surfels: {
      u: half([0.01, 0, 0, 0.01, 0, 0]),
      v: half([0, 0.01, 0, 0, 0.01, 0]),
      scale: half([1, 1, 1, 1]),
      frames: new Uint8Array([0, 1]),
      metadata: {
        version: 3, point_count: 2, grid_size: [side, side],
        frames: [frame(0), frame(1)],
      },
      textures: texture, depths: depth, textureSize: [side, side],
    },
  } as any;
}

function centerPixel(): number[] {
  context.clearRect(0, 0, 256, 256);
  context.drawImage(canvas, 0, 0);
  return Array.from(context.getImageData(128, 128, 1, 1).data).slice(0, 3);
}

async function exercise(camera: Camera, includeOff: boolean) {
  camera.updateMatrixWorld();
  const visibility = new Uint8BufferAttribute(new Uint8Array([1, 1]), 1);
  const controller = createSurfelRenderer(renderer, camera, makeBundle(), worldMatrix, visibility);
  controller.setSize(256, 256);
  controller.setRadius(0.008);
  try {
    let off: number[] | null = null;
    if (includeOff) {
      controller.setSourceSelection(false);
      controller.render();
      off = centerPixel();
    }
    controller.setSourceSelection(true);
    controller.render();
    const selected = centerPixel();
    const selectedPick = controller.pick(0.5, 0.5);
    visibility.setX(0, 0);
    visibility.needsUpdate = true;
    controller.render();
    const fallback = centerPixel();
    const fallbackPick = controller.pick(0.5, 0.5);
    return { off, selected, selectedPick, fallback, fallbackPick };
  } finally {
    controller.dispose();
  }
}

const isMixedRedBlue = (rgb: number[] | null) => rgb !== null
  && rgb[0] >= 170 && rgb[2] >= 170 && rgb[1] <= 8 && Math.abs(rgb[0] - rgb[2]) <= 8;
const isRed = (rgb: number[]) => rgb[0] >= 247 && rgb[1] <= 8 && rgb[2] <= 8;
const isBlue = (rgb: number[]) => rgb[2] >= 247 && rgb[0] <= 8 && rgb[1] <= 8;

async function run() {
  const orthographic = new OrthographicCamera(-0.1, 0.1, 0.1, -0.1, 0.01, 20);
  orthographic.position.set(0, 0, 0);
  orthographic.lookAt(0, 0, -1);
  const perspective = new PerspectiveCamera(30, 1, 0.01, 20);
  perspective.position.set(0, 0, 0);
  perspective.lookAt(0, 0, -1);
  const ortho = await exercise(orthographic, true);
  const perspectiveResult = await exercise(perspective, false);
  const checks = {
    off_blends_both_sources: isMixedRedBlue(ortho.off),
    orthographic_tie_selects_lower_frame: isRed(ortho.selected),
    orthographic_pick_matches_lower_frame: ortho.selectedPick === 0,
    orthographic_hidden_winner_falls_back: isBlue(ortho.fallback),
    orthographic_fallback_pick_matches: ortho.fallbackPick === 1,
    perspective_tie_selects_lower_frame: isRed(perspectiveResult.selected),
    perspective_pick_matches_lower_frame: perspectiveResult.selectedPick === 0,
    perspective_hidden_winner_falls_back: isBlue(perspectiveResult.fallback),
    perspective_fallback_pick_matches: perspectiveResult.fallbackPick === 1,
  };
  const glErrors: number[] = [];
  const gl = renderer.getContext();
  for (let error = gl.getError(); error !== gl.NO_ERROR; error = gl.getError()) glErrors.push(error);
  Object.assign(window, {
    __done: true,
    __probe: { orthographic: ortho, perspective: perspectiveResult, checks, glErrors },
  });
  renderer.dispose();
}

run().catch(error => {
  Object.assign(window, { __error: String(error instanceof Error ? error.stack ?? error.message : error) });
  console.error(error);
  renderer.dispose();
});
