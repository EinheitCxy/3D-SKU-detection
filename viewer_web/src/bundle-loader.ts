import {
  type Manifest,
  type ObjectIndex,
  validateCurrent,
  validateManifest,
  validateObjectIndex,
} from "./contracts";

export interface ViewerBundle {
  readonly manifest: Manifest;
  readonly objects: ObjectIndex;
  readonly positions: Float32Array;
  readonly colors: Uint8Array;
  readonly normals: Int8Array;
  readonly pointCount: number;
  readonly generationUrl: string;
  readonly resolveAssetUrl: (relativePath: string) => string;
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
  const [objectsValue, positionsBuffer, colorsBuffer, normalsBuffer] = await Promise.all([
    fetchJson(`${generationUrl}objects.json`, fetcher),
    fetchBinary(`${generationUrl}positions.f32.bin`, fetcher),
    fetchBinary(`${generationUrl}colors.u8.bin`, fetcher),
    fetchBinary(`${generationUrl}normals.i8.bin`, fetcher),
  ]);
  const positions = decodePositions(positionsBuffer);
  const pointCount = positions.length / 3;
  const colors = decodeColors(colorsBuffer, pointCount);
  const normals = decodeNormals(normalsBuffer, pointCount);
  const objects = validateObjectIndex(objectsValue, pointCount);
  return {
    manifest,
    objects,
    positions,
    colors,
    normals,
    pointCount,
    generationUrl,
    resolveAssetUrl: (relativePath) => new URL(relativePath, generationUrl).toString(),
  };
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

function decodePositions(buffer: ArrayBuffer): Float32Array {
  if (buffer.byteLength % (Float32Array.BYTES_PER_ELEMENT * 3) !== 0) {
    throw new Error("Viewer bundle positions shape is invalid");
  }
  return new Float32Array(buffer);
}

function decodeColors(buffer: ArrayBuffer, pointCount: number): Uint8Array {
  assertShape(buffer, pointCount, Uint8Array.BYTES_PER_ELEMENT, "colors");
  return new Uint8Array(buffer);
}

function decodeNormals(buffer: ArrayBuffer, pointCount: number): Int8Array {
  assertShape(buffer, pointCount, Int8Array.BYTES_PER_ELEMENT, "normals");
  return new Int8Array(buffer);
}

function assertShape(buffer: ArrayBuffer, pointCount: number, bytesPerComponent: number, name: string): void {
  const expected = pointCount * 3 * bytesPerComponent;
  if (buffer.byteLength !== expected) {
    throw new Error(`Viewer bundle ${name} shape is invalid: expected ${expected} bytes, got ${buffer.byteLength}`);
  }
}
