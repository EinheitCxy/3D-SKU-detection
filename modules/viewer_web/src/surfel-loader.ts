import { DataArrayTexture, HalfFloatType, LinearFilter, RedFormat, RGBAFormat, UnsignedByteType } from "three";
import type { ViewerBundle } from "./bundle-loader";

export interface SurfelFrame {
  image_id: number; texture: string; texture_size: [number, number]; source_size: [number, number];
  intrinsic: number[][]; extrinsic: number[][]; processed_to_source: number[][];
}
export interface SurfelData {
  u: Uint16Array; v: Uint16Array; scale: Uint16Array; frames: Uint8Array;
  metadata: { version: 2 | 3; point_count: number; grid_size: [number, number]; frames: SurfelFrame[] };
  textures: DataArrayTexture; depths: DataArrayTexture;
  textureSize: [number, number];
}

function matrix(value: unknown, rows: number, columns: number): boolean {
  return Array.isArray(value) && value.length === rows && value.every(row =>
    Array.isArray(row) && row.length === columns && row.every(x => typeof x === "number" && Number.isFinite(x)));
}
function dimensions(value: unknown): value is [number, number] {
  return Array.isArray(value) && value.length === 2 && value.every(x => Number.isInteger(x) && x > 0 && x <= 16384);
}

export function validateSurfelVersion(metadata: unknown): 2 | 3 {
  const version = typeof metadata === "object" && metadata !== null && "version" in metadata ? metadata.version : undefined;
  if (version !== 2 && version !== 3) throw new Error("Unsupported Surfel version; expected v2 or v3");
  return version;
}

export async function loadSurfels(bundle: ViewerBundle, fetcher: typeof fetch = globalThis.fetch): Promise<SurfelData> {
  const fetchAsset = async (name: string) => {
    const response = await fetcher(bundle.resolveAssetUrl(name));
    if (!response.ok) throw new Error(`Surfel asset ${name}: HTTP ${response.status}`);
    return response;
  };
  const metadata = await (await fetchAsset("surfel.json")).json();
  const version = validateSurfelVersion(metadata);
  if (metadata.point_count !== bundle.pointCount || !dimensions(metadata.grid_size)
    || !Array.isArray(metadata.frames) || metadata.frames.length < 1 || metadata.frames.length > 256) {
    throw new Error("Invalid Surfel metadata or slot count; regenerate the Surfel bundle");
  }
  for (const frame of metadata.frames) {
    if (!dimensions(frame.texture_size) || Math.max(...frame.texture_size) > 4096 || !dimensions(frame.source_size)
      || !matrix(frame.intrinsic, 3, 3) || !matrix(frame.extrinsic, 3, 4) || !matrix(frame.processed_to_source, 3, 3)
      || !/^surfel-texture-\d+\.jpg$/.test(frame.texture)) throw new Error("Invalid Surfel source frame");
  }
  const halfs = async (name: string, size: number) => {
    const buffer = await (await fetchAsset(name)).arrayBuffer();
    if (buffer.byteLength !== size * 2) throw new Error(`Surfel ${name} shape mismatch`);
    const values = new Uint16Array(buffer);
    if (!values.every(x => (x & 0x7c00) !== 0x7c00)) throw new Error(`Surfel ${name} contains nonfinite values`);
    return values;
  };
  const frameIndices = async () => {
    const buffer = await (await fetchAsset("surfel-frame.u8.bin")).arrayBuffer();
    if (buffer.byteLength !== bundle.pointCount) throw new Error("Surfel frame shape mismatch");
    return new Uint8Array(buffer);
  };
  const [u, v, frames, depth, scale] = await Promise.all([
    halfs("surfel-u.f16.bin", bundle.pointCount * 3), halfs("surfel-v.f16.bin", bundle.pointCount * 3),
    frameIndices(),
    halfs("surfel-depth.f16.bin", metadata.grid_size[0] * metadata.grid_size[1] * metadata.frames.length),
    version === 3 ? halfs("surfel-scale.f16.bin", bundle.pointCount * 2)
      : new Uint16Array(bundle.pointCount * 2).fill(0x3c00), // v2 uses unit scale (Float16 1.0).
  ]);
  // Positive Float16 bit patterns are ordered; these are exactly 0.5 and 8.0.
  if (!scale.every(x => x >= 0x3800 && x <= 0x4800)) throw new Error("Surfel scale must be finite and within [0.5, 8]");
  if (!frames.every(x => Number.isInteger(x) && x >= 0 && x < metadata.frames.length)
    || !depth.every(x => (x & 0x8000) === 0)) throw new Error("Invalid Surfel frame index or source depth");
  const textureWidth = Math.max(...metadata.frames.map((f: SurfelFrame) => f.texture_size[0]));
  const textureHeight = Math.max(...metadata.frames.map((f: SurfelFrame) => f.texture_size[1]));
  const pixels = new Uint8Array(textureWidth * textureHeight * 4 * metadata.frames.length);
  // Decode sequentially to bound transient image/canvas memory.
  for (let i = 0; i < metadata.frames.length; i++) {
    const frame = metadata.frames[i];
    const bitmap = await createImageBitmap(await (await fetchAsset(frame.texture)).blob());
    try {
      if (bitmap.width !== frame.texture_size[0] || bitmap.height !== frame.texture_size[1]) throw new Error("Surfel texture dimensions mismatch");
      const canvas = document.createElement("canvas");
      canvas.width = textureWidth; canvas.height = textureHeight;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) throw new Error("Cannot decode Surfel texture");
      context.drawImage(bitmap, 0, 0);
      pixels.set(context.getImageData(0, 0, textureWidth, textureHeight).data, i * textureWidth * textureHeight * 4);
      canvas.width = canvas.height = 0;
    } finally { bitmap.close(); }
  }
  const textures = new DataArrayTexture(pixels, textureWidth, textureHeight, metadata.frames.length);
  textures.format = RGBAFormat; textures.type = UnsignedByteType;
  textures.minFilter = textures.magFilter = LinearFilter; textures.needsUpdate = true;
  const depths = new DataArrayTexture(depth, ...metadata.grid_size, metadata.frames.length);
  depths.format = RedFormat; depths.type = HalfFloatType; depths.needsUpdate = true;
  return { u, v, scale, frames, metadata, textures, depths, textureSize: [textureWidth, textureHeight] };
}
