import { assertLittleEndian, type ViewerBundle } from "./bundle-loader";
import { validateManifest, validateObjectIndex } from "./contracts";

const FIXED_FILES = new Set([
  "manifest.json",
  "positions.f32.bin",
  "colors.u8.bin",
  "normals.i8.bin",
  "objects.json",
]);
const textDecoder = new TextDecoder("utf-8", { fatal: true });

export async function loadDockerViewerBundle(archive: Blob): Promise<ViewerBundle> {
  assertLittleEndian();
  const entries = parseFlatStoredZip(await archive.arrayBuffer());
  const manifest = validateManifest(parseJson(entries, "manifest.json"));
  const positions = decodePositions(requiredEntry(entries, "positions.f32.bin"));
  const pointCount = positions.length / 3;
  const colors = decodeComponents(requiredEntry(entries, "colors.u8.bin"), pointCount, "colors", Uint8Array);
  const normals = decodeComponents(requiredEntry(entries, "normals.i8.bin"), pointCount, "normals", Int8Array);
  const objects = validateObjectIndex(parseJson(entries, "objects.json"), pointCount);
  const assetUrls = thumbnailUrls(entries, objects);
  return {
    manifest,
    objects,
    positions,
    colors,
    normals,
    pointCount,
    generationUrl: "docker-zip://viewer/",
    resolveAssetUrl: (relativePath) => {
      const url = assetUrls.get(relativePath);
      if (url === undefined) throw new Error(`Viewer bundle asset is missing: ${relativePath}`);
      return url;
    },
  };
}

function parseFlatStoredZip(buffer: ArrayBuffer): ReadonlyMap<string, Uint8Array> {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const eocdOffset = findEndOfCentralDirectory(view);
  const entryCount = view.getUint16(eocdOffset + 10, true);
  const centralSize = view.getUint32(eocdOffset + 12, true);
  const centralOffset = view.getUint32(eocdOffset + 16, true);
  if (view.getUint16(eocdOffset + 8, true) !== entryCount) throw new Error("Viewer ZIP uses multiple disks");
  if (entryCount === 0xffff || centralSize === 0xffffffff || centralOffset === 0xffffffff) throw new Error("Viewer ZIP64 archives are not supported");
  if (centralOffset + centralSize > eocdOffset) throw new Error("Viewer ZIP central directory is invalid");
  const entries = new Map<string, Uint8Array>();
  let cursor = centralOffset;
  for (let index = 0; index < entryCount; index += 1) {
    if (cursor + 46 > centralOffset + centralSize || view.getUint32(cursor, true) !== 0x02014b50) throw new Error("Viewer ZIP central directory entry is invalid");
    const flags = view.getUint16(cursor + 8, true);
    const compression = view.getUint16(cursor + 10, true);
    const compressedSize = view.getUint32(cursor + 20, true);
    const uncompressedSize = view.getUint32(cursor + 24, true);
    const nameLength = view.getUint16(cursor + 28, true);
    const extraLength = view.getUint16(cursor + 30, true);
    const commentLength = view.getUint16(cursor + 32, true);
    const localOffset = view.getUint32(cursor + 42, true);
    const entryEnd = cursor + 46 + nameLength + extraLength + commentLength;
    if (entryEnd > centralOffset + centralSize) throw new Error("Viewer ZIP entry length is invalid");
    if ((flags & 1) !== 0 || compression !== 0 || compressedSize !== uncompressedSize) throw new Error("Viewer ZIP entries must use ZIP_STORED without encryption");
    const name = decodeName(bytes.subarray(cursor + 46, cursor + 46 + nameLength));
    assertMemberName(name);
    if (entries.has(name)) throw new Error(`Viewer ZIP contains duplicate member: ${name}`);
    entries.set(name, localContent(bytes, view, localOffset, name, compressedSize));
    cursor = entryEnd;
  }
  if (cursor !== centralOffset + centralSize) throw new Error("Viewer ZIP central directory size is invalid");
  for (const name of FIXED_FILES) requiredEntry(entries, name);
  return entries;
}

function findEndOfCentralDirectory(view: DataView): number {
  const firstOffset = Math.max(0, view.byteLength - 22 - 0xffff);
  for (let offset = view.byteLength - 22; offset >= firstOffset; offset -= 1) {
    if (view.getUint32(offset, true) === 0x06054b50 && offset + 22 + view.getUint16(offset + 20, true) === view.byteLength) return offset;
  }
  throw new Error("Viewer ZIP end record is missing");
}

function localContent(bytes: Uint8Array, view: DataView, offset: number, name: string, size: number): Uint8Array {
  if (offset + 30 > bytes.length || view.getUint32(offset, true) !== 0x04034b50) throw new Error(`Viewer ZIP local entry is invalid: ${name}`);
  const nameLength = view.getUint16(offset + 26, true);
  const extraLength = view.getUint16(offset + 28, true);
  const localName = decodeName(bytes.subarray(offset + 30, offset + 30 + nameLength));
  const contentStart = offset + 30 + nameLength + extraLength;
  if (localName !== name || contentStart + size > bytes.length) throw new Error(`Viewer ZIP local entry content is invalid: ${name}`);
  return bytes.slice(contentStart, contentStart + size);
}

function decodeName(value: Uint8Array): string {
  try { return textDecoder.decode(value); }
  catch (error) { throw new Error("Viewer ZIP member name is not UTF-8", { cause: error }); }
}

function assertMemberName(name: string): void {
  if (FIXED_FILES.has(name)) return;
  if (/^thumbs\/[^/]+\.jpg$/.test(name) && name !== "thumbs/.jpg") return;
  throw new Error(`Viewer ZIP contains an unexpected member: ${name}`);
}

function requiredEntry(entries: ReadonlyMap<string, Uint8Array>, name: string): Uint8Array {
  const entry = entries.get(name);
  if (entry === undefined) throw new Error(`Viewer ZIP is missing fixed member: ${name}`);
  return entry;
}

function parseJson(entries: ReadonlyMap<string, Uint8Array>, name: string): unknown {
  try { return JSON.parse(textDecoder.decode(requiredEntry(entries, name))) as unknown; }
  catch (error) { throw new Error(`Viewer ZIP JSON is invalid: ${name}`, { cause: error }); }
}

function decodePositions(content: Uint8Array): Float32Array {
  if (content.byteLength % (Float32Array.BYTES_PER_ELEMENT * 3) !== 0) throw new Error("Viewer ZIP positions shape is invalid");
  return new Float32Array(copyArrayBuffer(content));
}

function decodeComponents<T extends Int8Array | Uint8Array>(content: Uint8Array, pointCount: number, name: string, Constructor: { new(buffer: ArrayBuffer): T; readonly BYTES_PER_ELEMENT: number }): T {
  if (content.byteLength !== pointCount * 3 * Constructor.BYTES_PER_ELEMENT) throw new Error(`Viewer ZIP ${name} shape is invalid`);
  return new Constructor(copyArrayBuffer(content));
}

function copyArrayBuffer(content: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(content.byteLength);
  copy.set(content);
  return copy.buffer;
}

function thumbnailUrls(entries: ReadonlyMap<string, Uint8Array>, objects: ViewerBundle["objects"]): ReadonlyMap<string, string> {
  const urls = new Map<string, string>();
  for (const object of Object.values(objects)) for (const observation of object.observations) {
    if (urls.has(observation.thumbnail)) continue;
    const content = entries.get(observation.thumbnail);
    if (content === undefined) throw new Error(`Viewer ZIP thumbnail is missing: ${observation.thumbnail}`);
    urls.set(observation.thumbnail, URL.createObjectURL(new Blob([copyArrayBuffer(content)], { type: "image/jpeg" })));
  }
  return urls;
}
