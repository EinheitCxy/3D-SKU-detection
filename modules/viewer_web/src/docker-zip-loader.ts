import { assertLittleEndian, type ViewerBundle } from "./bundle-loader";
import { validateManifest, validateObjectIndex, validateSkuMasterData } from "./contracts";
import { validateSurfelVersion } from "./surfel-loader";
import masterDataUrl from "../masterdata.json?url";

export type DockerRenderMode = "points" | "surfel";
type CompactMasterData = Record<string, [string | null, string | null, string | null, boolean]>;
let sharedMasterData: Promise<CompactMasterData> | undefined;

function loadMasterData(): Promise<CompactMasterData> {
  sharedMasterData ??= fetch(masterDataUrl).then(async response => {
    if (!response.ok) throw new Error(`SKU 主数据下载失败：HTTP ${response.status}`);
    return await response.json() as CompactMasterData;
  });
  return sharedMasterData;
}

const COMMON_FILES = new Set(["manifest.json", "positions.f32.bin", "normals.i8.bin", "objects.json"]);
const POINT_FILES = new Set([...COMMON_FILES, "colors.u8.bin"]);
const SURFEL_FILES = new Set([
  ...POINT_FILES,
  "surfel.json",
  "surfel-u.f16.bin",
  "surfel-v.f16.bin",
  "surfel-frame.u8.bin",
  "surfel-depth.f16.bin",
]);
const SURFEL_SCALE_FILE = "surfel-scale.f16.bin";
const textDecoder = new TextDecoder("utf-8", { fatal: true });

export async function loadDockerViewerBundle(
  archive: Blob,
  masterData?: CompactMasterData,
  mode: DockerRenderMode = "surfel",
): Promise<ViewerBundle> {
  assertLittleEndian();
  const entries = parseFlatStoredZip(await archive.arrayBuffer(), mode);
  const manifest = validateManifest(parseJson(entries, "manifest.json"));
  const positions = decodePositions(requiredEntry(entries, "positions.f32.bin"));
  const pointCount = positions.length / 3;
  const colors = decodeComponents(requiredEntry(entries, "colors.u8.bin"), pointCount, "colors", Uint8Array);
  const normals = decodeComponents(requiredEntry(entries, "normals.i8.bin"), pointCount, "normals", Int8Array);
  const objects = validateObjectIndex(parseJson(entries, "objects.json"), pointCount);
  const knownSkuIds = new Set(Object.values(objects).flatMap(object => object.ordered_skus.map(sku => sku.sku_id)));
  const sharedData = masterData ?? await loadMasterData();
  const selectedData = Object.fromEntries([...knownSkuIds].map(skuId => {
    const row = sharedData[skuId];
    if (row === undefined) throw new Error(`SKU 主数据缺少 SKU ID：${skuId}`);
    const [manufacturer, brand, category, is_posm] = row;
    return [skuId, { manufacturer, brand, category, is_posm }];
  }));
  const skuMasterData = validateSkuMasterData(selectedData, knownSkuIds);
  const assets = createAssetUrls(entries, objects, mode);
  return {
    manifest, objects, skuMasterData, positions, colors, normals, pointCount,
    generationUrl: "docker-zip://viewer/",
    resolveAssetUrl(relativePath) {
      const url = assets.urls.get(relativePath);
      if (url === undefined) throw new Error(`数据包缺少资源：${relativePath}`);
      return url;
    },
    disposeAssets: assets.dispose,
  };
}

function parseFlatStoredZip(buffer: ArrayBuffer, mode: DockerRenderMode): ReadonlyMap<string, Uint8Array> {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const eocdOffset = findEndOfCentralDirectory(view);
  const entryCount = view.getUint16(eocdOffset + 10, true);
  const centralSize = view.getUint32(eocdOffset + 12, true);
  const centralOffset = view.getUint32(eocdOffset + 16, true);
  if (view.getUint16(eocdOffset + 8, true) !== entryCount) throw new Error("数据包 ZIP 跨越多个磁盘");
  if (entryCount === 0xffff || centralSize === 0xffffffff || centralOffset === 0xffffffff) throw new Error("不支持 ZIP64 数据包");
  if (centralOffset + centralSize > eocdOffset) throw new Error("数据包 ZIP 中央目录无效");
  const entries = new Map<string, Uint8Array>();
  let cursor = centralOffset;
  for (let index = 0; index < entryCount; index += 1) {
    if (cursor + 46 > centralOffset + centralSize || view.getUint32(cursor, true) !== 0x02014b50) throw new Error("数据包 ZIP 中央目录条目无效");
    const flags = view.getUint16(cursor + 8, true);
    const compression = view.getUint16(cursor + 10, true);
    const compressedSize = view.getUint32(cursor + 20, true);
    const uncompressedSize = view.getUint32(cursor + 24, true);
    const nameLength = view.getUint16(cursor + 28, true);
    const extraLength = view.getUint16(cursor + 30, true);
    const commentLength = view.getUint16(cursor + 32, true);
    const localOffset = view.getUint32(cursor + 42, true);
    const entryEnd = cursor + 46 + nameLength + extraLength + commentLength;
    if (entryEnd > centralOffset + centralSize) throw new Error("数据包 ZIP 条目长度无效");
    if ((flags & 1) !== 0 || compression !== 0 || compressedSize !== uncompressedSize) throw new Error("数据包 ZIP 条目必须使用未加密的 ZIP_STORED");
    const name = decodeName(bytes.subarray(cursor + 46, cursor + 46 + nameLength));
    assertMemberName(name, mode);
    if (entries.has(name)) throw new Error(`数据包 ZIP 包含重复成员：${name}`);
    entries.set(name, localContent(bytes, view, localOffset, name, compressedSize));
    cursor = entryEnd;
  }
  if (cursor !== centralOffset + centralSize) throw new Error("数据包 ZIP 中央目录大小无效");
  for (const name of mode === "points" ? POINT_FILES : SURFEL_FILES) {
    if (!entries.has(name)) {
      const prefix = mode === "surfel" && !COMMON_FILES.has(name) ? "必需的 Surfel" : "必需的";
      throw new Error(`数据包 ZIP 缺少${prefix}成员：${name}`);
    }
  }
  if (mode === "surfel") {
    const version = validateSurfelVersion(parseJson(entries, "surfel.json"));
    if (version === 3 && !entries.has(SURFEL_SCALE_FILE)) throw new Error(`数据包 ZIP 缺少必需的 Surfel 成员：${SURFEL_SCALE_FILE}`);
  }
  return entries;
}

function findEndOfCentralDirectory(view: DataView): number {
  const firstOffset = Math.max(0, view.byteLength - 22 - 0xffff);
  for (let offset = view.byteLength - 22; offset >= firstOffset; offset -= 1) {
    if (view.getUint32(offset, true) === 0x06054b50 && offset + 22 + view.getUint16(offset + 20, true) === view.byteLength) return offset;
  }
  throw new Error("数据包 ZIP 缺少结束记录");
}

function localContent(bytes: Uint8Array, view: DataView, offset: number, name: string, size: number): Uint8Array {
  if (offset + 30 > bytes.length || view.getUint32(offset, true) !== 0x04034b50) throw new Error(`数据包 ZIP 本地条目无效：${name}`);
  const nameLength = view.getUint16(offset + 26, true);
  const extraLength = view.getUint16(offset + 28, true);
  const localName = decodeName(bytes.subarray(offset + 30, offset + 30 + nameLength));
  const contentStart = offset + 30 + nameLength + extraLength;
  if (localName !== name || contentStart + size > bytes.length) throw new Error(`数据包 ZIP 本地条目内容无效：${name}`);
  return bytes.slice(contentStart, contentStart + size);
}

function decodeName(value: Uint8Array): string {
  try { return textDecoder.decode(value); }
  catch (error) { throw new Error("数据包 ZIP 成员名不是 UTF-8", { cause: error }); }
}

function assertMemberName(name: string, mode: DockerRenderMode): void {
  const fixed = mode === "points" ? POINT_FILES : SURFEL_FILES;
  if (fixed.has(name)) return;
  if (mode === "surfel" && name === SURFEL_SCALE_FILE) return;
  if (/^thumbs\/[^/]+\.jpg$/.test(name) && name !== "thumbs/.jpg") return;
  if (mode === "surfel" && /^surfel-texture-\d+\.jpg$/.test(name)) return;
  throw new Error(`数据包 ZIP 包含不支持的成员：${name}`);
}

function requiredEntry(entries: ReadonlyMap<string, Uint8Array>, name: string): Uint8Array {
  const entry = entries.get(name);
  if (entry === undefined) throw new Error(`数据包 ZIP 缺少必需成员：${name}`);
  return entry;
}

function parseJson(entries: ReadonlyMap<string, Uint8Array>, name: string): unknown {
  try { return JSON.parse(textDecoder.decode(requiredEntry(entries, name))) as unknown; }
  catch (error) { throw new Error(`数据包 ZIP JSON 无效：${name}`, { cause: error }); }
}

function decodePositions(content: Uint8Array): Float32Array {
  if (content.byteLength % (Float32Array.BYTES_PER_ELEMENT * 3) !== 0) throw new Error("数据包 ZIP positions 数据形状无效");
  return new Float32Array(copyArrayBuffer(content));
}

function decodeComponents<T extends Int8Array | Uint8Array>(content: Uint8Array, pointCount: number, name: string, Constructor: { new(buffer: ArrayBuffer): T; readonly BYTES_PER_ELEMENT: number }): T {
  if (content.byteLength !== pointCount * 3 * Constructor.BYTES_PER_ELEMENT) throw new Error(`数据包 ZIP ${name} 数据形状无效`);
  return new Constructor(copyArrayBuffer(content));
}

function copyArrayBuffer(content: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(content.byteLength);
  copy.set(content);
  return copy.buffer;
}

function createAssetUrls(entries: ReadonlyMap<string, Uint8Array>, objects: ViewerBundle["objects"], mode: DockerRenderMode): { urls: ReadonlyMap<string, string>; dispose(): void } {
  const names = new Set<string>();
  for (const object of Object.values(objects)) for (const observation of object.observations) names.add(observation.thumbnail);
  if (mode === "surfel") for (const name of SURFEL_FILES) if (!COMMON_FILES.has(name)) names.add(name);
  if (mode === "surfel" && entries.has(SURFEL_SCALE_FILE)) names.add(SURFEL_SCALE_FILE);
  if (mode === "surfel") for (const name of entries.keys()) if (/^surfel-texture-\d+\.jpg$/.test(name)) names.add(name);
  const urls = new Map<string, string>();
  try {
    for (const name of names) {
      const content = entries.get(name);
      if (content === undefined) throw new Error(`数据包 ZIP 缺少资源：${name}`);
      const type = name.endsWith(".jpg") ? "image/jpeg" : "application/octet-stream";
      urls.set(name, URL.createObjectURL(new Blob([copyArrayBuffer(content)], { type })));
    }
    return {
      urls,
      dispose() {
        for (const url of urls.values()) URL.revokeObjectURL(url);
        urls.clear();
      },
    };
  } catch (error) {
    for (const url of urls.values()) URL.revokeObjectURL(url);
    throw error;
  }
}
