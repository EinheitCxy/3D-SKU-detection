import {
  type ArrayDescriptor,
  type CurrentPointer,
  type FootprintBundle,
  type Manifest,
  type ObjectIndex,
  validateCurrent,
  validateFootprints,
  validateManifest,
  validateObjectIndex,
} from "./contracts";

export interface ViewerBundle {
  readonly current: CurrentPointer;
  readonly manifest: Manifest;
  readonly objects: ObjectIndex;
  readonly footprints: FootprintBundle;
  readonly positions: Float32Array;
  readonly colors: Uint8Array;
  readonly confidences: Float32Array;
  readonly frameIds: Int32Array;
}

export function isLittleEndian(): boolean {
  const probe = new Uint16Array(new Uint8Array([1, 0]).buffer);
  return probe[0] === 1;
}

export function assertLittleEndian(): void {
  if (!isLittleEndian()) throw new Error("Viewer bundle requires a little-endian runtime");
}

export async function loadViewerBundle(
  baseUrl: string,
  fetcher: typeof fetch = globalThis.fetch,
): Promise<ViewerBundle> {
  if (!baseUrl.endsWith("/")) throw new Error("Viewer bundle baseUrl must end with a slash");
  assertLittleEndian();
  const current = validateCurrent(await fetchJson(`${baseUrl}CURRENT`, fetcher));
  const generationUrl = `${baseUrl}runs/${current.run_id}/`;
  const [manifestValue, objectsValue, footprintsValue, positionsBuffer, colorsBuffer, confidencesBuffer, frameIdsBuffer] = await Promise.all([
    fetchJson(`${generationUrl}manifest.json`, fetcher),
    fetchJson(`${generationUrl}objects.json`, fetcher),
    fetchJson(`${generationUrl}footprints.json`, fetcher),
    fetchBinary(`${generationUrl}positions.f32.bin`, fetcher),
    fetchBinary(`${generationUrl}colors.u8.bin`, fetcher),
    fetchBinary(`${generationUrl}confidences.f32.bin`, fetcher),
    fetchBinary(`${generationUrl}frame_ids.i32.bin`, fetcher),
  ]);
  const manifest = validateManifest(manifestValue);
  const objects = validateObjectIndex(objectsValue);
  const footprints = validateFootprints(footprintsValue);
  const positions = decodeFloat32(positionsBuffer, manifest.arrays.positions, manifest.point_count);
  const colors = decodeUint8(colorsBuffer, manifest.arrays.colors, manifest.point_count);
  const confidences = decodeFloat32(confidencesBuffer, manifest.arrays.confidences, manifest.point_count);
  const frameIds = decodeInt32(frameIdsBuffer, manifest.arrays.frame_ids, manifest.point_count);
  return { current, manifest, objects, footprints, positions, colors, confidences, frameIds };
}

async function fetchJson(url: string, fetcher: typeof fetch): Promise<unknown> {
  const response = await fetcher(url);
  if (!response.ok) throw new Error(`Viewer bundle HTTP error ${response.status} for ${url}`);
  try {
    return await response.json() as unknown;
  } catch (error) {
    throw new Error(`Viewer bundle JSON error for ${url}`, { cause: error });
  }
}

async function fetchBinary(url: string, fetcher: typeof fetch): Promise<ArrayBuffer> {
  const response = await fetcher(url);
  if (!response.ok) throw new Error(`Viewer bundle HTTP error ${response.status} for ${url}`);
  return response.arrayBuffer();
}

function decodeFloat32(buffer: ArrayBuffer, descriptor: ArrayDescriptor, pointCount: number): Float32Array {
  assertByteLength(buffer, descriptor, pointCount);
  return new Float32Array(buffer);
}

function decodeUint8(buffer: ArrayBuffer, descriptor: ArrayDescriptor, pointCount: number): Uint8Array {
  assertByteLength(buffer, descriptor, pointCount);
  return new Uint8Array(buffer);
}

function decodeInt32(buffer: ArrayBuffer, descriptor: ArrayDescriptor, pointCount: number): Int32Array {
  assertByteLength(buffer, descriptor, pointCount);
  return new Int32Array(buffer);
}

function assertByteLength(buffer: ArrayBuffer, descriptor: ArrayDescriptor, pointCount: number): void {
  if (buffer.byteLength !== descriptor.byte_length) throw new Error(`Viewer bundle byte length mismatch for ${descriptor.path}`);
  const bytesPerElement = descriptor.dtype === "uint8" ? 1 : 4;
  const expected = pointCount * descriptor.components * bytesPerElement;
  if (buffer.byteLength !== expected) throw new Error(`Viewer bundle byte length mismatch for ${descriptor.path}`);
}
