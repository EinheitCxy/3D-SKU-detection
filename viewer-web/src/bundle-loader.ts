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
  readonly normals: Int8Array;
  /** Immutable generation base URL; instance thumbnails resolve against it. */
  readonly generationUrl: string;
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
  const current = validateCurrent(await fetchJson(`${baseUrl}CURRENT`, fetcher, { cache: "no-store" }));
  const generationUrl = `${baseUrl}runs/${current.run_id}/`;
  const manifest = validateManifest(await fetchJson(`${generationUrl}manifest.json`, fetcher));
  const [objectsValue, footprintsValue, positionsBuffer, colorsBuffer, normalsBuffer] = await Promise.all([
    fetchJson(`${generationUrl}objects.json`, fetcher),
    fetchJson(`${generationUrl}footprints.json`, fetcher),
    fetchBinary(`${generationUrl}positions.f32.bin`, fetcher),
    fetchBinary(`${generationUrl}colors.u8.bin`, fetcher),
    fetchBinary(`${generationUrl}normals.i8.bin`, fetcher),
  ]);
  const objects = validateObjectIndex(objectsValue, manifest.point_count);
  assertSam3InstanceImageIds(
    objects,
    manifest.source.export.sam3_mask_entries.map((entry) => entry.image_id),
  );
  const footprints = validateFootprints(footprintsValue);
  if (
    manifest.source.footprint.run_id !== footprints.run_id
    || manifest.source.footprint.status !== footprints.status
  ) {
    throw new Error("Invalid viewer bundle: manifest source footprint does not match footprints.json");
  }
  const positions = decodeFloat32(positionsBuffer, manifest.arrays.positions, manifest.point_count);
  const colors = decodeUint8(colorsBuffer, manifest.arrays.colors, manifest.point_count);
  const normals = decodeInt8(normalsBuffer, manifest.arrays.normals, manifest.point_count);
  return { current, manifest, objects, footprints, positions, colors, normals, generationUrl };
}

function assertSam3InstanceImageIds(objects: ObjectIndex, sam3ImageIds: readonly number[]): void {
  const instanceImageIds = [...new Set(
    Object.values(objects).flatMap((entry) => entry.instances.map((instance) => instance.image_id)),
  )].sort((left, right) => left - right);
  const entryImageIds = [...new Set(sam3ImageIds)].sort((left, right) => left - right);
  if (
    instanceImageIds.length === 0
    || entryImageIds.length === 0
    || instanceImageIds.length !== entryImageIds.length
    || instanceImageIds.some((imageId, index) => imageId !== entryImageIds[index])
  ) {
    throw new Error(
      "Invalid viewer bundle: object instance image IDs must exactly match manifest source SAM3 entry image IDs",
    );
  }
}

async function fetchJson(url: string, fetcher: typeof fetch, init?: RequestInit): Promise<unknown> {
  const response = await fetcher(url, init);
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

function decodeInt8(buffer: ArrayBuffer, descriptor: ArrayDescriptor, pointCount: number): Int8Array {
  assertByteLength(buffer, descriptor, pointCount);
  return new Int8Array(buffer);
}

function assertByteLength(buffer: ArrayBuffer, descriptor: ArrayDescriptor, pointCount: number): void {
  if (buffer.byteLength !== descriptor.byte_length) throw new Error(`Viewer bundle byte length mismatch for ${descriptor.path}`);
  const bytesPerElement = descriptor.dtype === "float32" ? 4 : 1;
  const expected = pointCount * descriptor.components * bytesPerElement;
  if (buffer.byteLength !== expected) throw new Error(`Viewer bundle byte length mismatch for ${descriptor.path}`);
}
