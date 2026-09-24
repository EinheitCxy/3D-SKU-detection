import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DataUtils } from "three";
import { loadDockerViewerBundle } from "../../modules/viewer_web/src/docker-zip-loader";
import { loadSurfels } from "../../modules/viewer_web/src/surfel-loader";
import type { ViewerBundle } from "../../modules/viewer_web/src/bundle-loader";

const encoder = new TextEncoder();
const scaleName = "surfel-scale.f16.bin";
const frame = {
  image_id: 0, texture: "surfel-texture-0.jpg", texture_size: [2, 2], source_size: [2, 2],
  intrinsic: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  extrinsic: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
  processed_to_source: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
};
const encode = (value: unknown) => encoder.encode(JSON.stringify(value));
const halfs = (values: number[]) => new Uint8Array(Uint16Array.from(values, DataUtils.toHalfFloat).buffer);

function assets(version: number): Map<string, Uint8Array> {
  const result = new Map<string, Uint8Array>([
    ["manifest.json", encode({
      schema_version: "3.0.0", dataset_name: "surfel-scale", backend: "DA3", frame_count: 1,
      display_bounds: [0, 0, 0, 1, 1, 1],
      world_to_view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    })],
    ["positions.f32.bin", new Uint8Array(new Float32Array([0, 0, 1, 1, 0, 1]).buffer)],
    ["colors.u8.bin", new Uint8Array(6)],
    ["normals.i8.bin", new Uint8Array([0, 0, 127, 0, 0, 127])],
    ["objects.json", encode({})],
    ["surfel.json", encode({ version, point_count: 2, grid_size: [2, 2], frames: [frame] })],
    ["surfel-u.f16.bin", halfs([1, 0, 0, 1, 0, 0])],
    ["surfel-v.f16.bin", halfs([0, 1, 0, 0, 1, 0])],
    ["surfel-frame.u8.bin", new Uint8Array(2)],
    ["surfel-depth.f16.bin", halfs([1, 1, 1, 1])],
    ["surfel-texture-0.jpg", new Uint8Array([255, 216, 255, 217])],
  ]);
  if (version === 3) result.set(scaleName, halfs([0.5, 8, 2, 1]));
  return result;
}

function staticLoader(entries: ReadonlyMap<string, Uint8Array>) {
  const fetcher = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
    const content = entries.get(String(input));
    return content === undefined ? new Response("missing", { status: 404 }) : new Response(content.slice().buffer);
  });
  const bundle = { pointCount: 2, resolveAssetUrl: (name: string) => name } as ViewerBundle;
  return { fetcher, load: () => loadSurfels(bundle, fetcher) };
}

function storedZip(entries: ReadonlyMap<string, Uint8Array>): Blob {
  const parts: ArrayBuffer[] = [];
  const directory: ArrayBuffer[] = [];
  let offset = 0;
  let directorySize = 0;
  for (const [name, content] of entries) {
    const nameBytes = encoder.encode(name);
    const local = new ArrayBuffer(30);
    const lv = new DataView(local);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(4, 20, true);
    lv.setUint32(18, content.byteLength, true);
    lv.setUint32(22, content.byteLength, true);
    lv.setUint16(26, nameBytes.byteLength, true);
    parts.push(local, nameBytes.buffer, content.slice().buffer);
    const central = new ArrayBuffer(46);
    const cv = new DataView(central);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint32(20, content.byteLength, true);
    cv.setUint32(24, content.byteLength, true);
    cv.setUint16(28, nameBytes.byteLength, true);
    cv.setUint32(42, offset, true);
    directory.push(central, nameBytes.buffer);
    offset += 30 + nameBytes.byteLength + content.byteLength;
    directorySize += 46 + nameBytes.byteLength;
  }
  const end = new ArrayBuffer(22);
  const ev = new DataView(end);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, entries.size, true);
  ev.setUint16(10, entries.size, true);
  ev.setUint32(12, directorySize, true);
  ev.setUint32(16, offset, true);
  return new Blob([...parts, ...directory, end], { type: "application/zip" });
}

beforeEach(() => {
  vi.stubGlobal("createImageBitmap", vi.fn(async () => ({ width: 2, height: 2, close() {} })));
  vi.stubGlobal("document", {
    createElement: () => ({
      width: 0, height: 0,
      getContext: () => ({ drawImage() {}, getImageData: () => ({ data: new Uint8ClampedArray(16) }) }),
    }),
  });
});
afterEach(() => vi.unstubAllGlobals());

describe("Surfel sampling scales", () => {
  it.each([2, 3])("loads v%i from a ZIP with scale and native tangents kept separate", async version => {
    const bundle = await loadDockerViewerBundle(storedZip(assets(version)), {});
    try {
      const data = await loadSurfels(bundle);
      try {
        expect(Array.from(data.scale, DataUtils.fromHalfFloat)).toEqual(version === 3 ? [0.5, 8, 2, 1] : [1, 1, 1, 1]);
        expect(Array.from(data.u, DataUtils.fromHalfFloat)).toEqual([1, 0, 0, 1, 0, 0]);
        expect(Array.from(data.v, DataUtils.fromHalfFloat)).toEqual([0, 1, 0, 0, 1, 0]);
        if (version === 3) expect(bundle.resolveAssetUrl(scaleName)).toMatch(/^blob:/);
        else expect(() => bundle.resolveAssetUrl(scaleName)).toThrow(/缺少资源/);
      } finally { data.textures.dispose(); data.depths.dispose(); }
    } finally { bundle.disposeAssets?.(); }
  });

  it("loads static v2 data without requesting a scale sidecar", async () => {
    const loader = staticLoader(assets(2));
    const data = await loader.load();
    expect(Array.from(data.scale, DataUtils.fromHalfFloat)).toEqual([1, 1, 1, 1]);
    expect(loader.fetcher.mock.calls.some(([name]) => name === scaleName)).toBe(false);
    data.textures.dispose(); data.depths.dispose();
  });

  it("requires the v3 scale sidecar in both static and ZIP bundles", async () => {
    const entries = assets(3);
    entries.delete(scaleName);
    await expect(staticLoader(entries).load()).rejects.toThrow(/surfel-scale.*404/);
    await expect(loadDockerViewerBundle(storedZip(entries), {})).rejects.toThrow(/缺少.*surfel-scale/);
  });

  it.each([
    ["wrong shape", new Uint8Array(6), /shape mismatch/],
    ["zero", halfs([0, 1, 1, 1]), /within/],
    ["negative", halfs([-1, 1, 1, 1]), /within/],
    ["too small", halfs([0.25, 1, 1, 1]), /within/],
    ["too large", halfs([16, 1, 1, 1]), /within/],
    ["infinite", new Uint8Array(new Uint16Array([0x7c00, 0x3c00, 0x3c00, 0x3c00]).buffer), /nonfinite/],
    ["NaN", new Uint8Array(new Uint16Array([0x7e00, 0x3c00, 0x3c00, 0x3c00]).buffer), /nonfinite/],
  ])("rejects %s v3 scales before texture decode", async (_label, bytes, error) => {
    const entries = assets(3);
    entries.set(scaleName, bytes);
    await expect(staticLoader(entries).load()).rejects.toThrow(error);
    expect(createImageBitmap).not.toHaveBeenCalled();
  });

  it("rejects an unknown version in static and ZIP bundles", async () => {
    const entries = assets(4);
    await expect(staticLoader(entries).load()).rejects.toThrow(/Unsupported Surfel version/);
    await expect(loadDockerViewerBundle(storedZip(entries), {})).rejects.toThrow(/Unsupported Surfel version/);
  });

  it("keeps point-mode and nested ZIP member restrictions", async () => {
    await expect(loadDockerViewerBundle(storedZip(assets(3)), {}, "points")).rejects.toThrow(/不支持的成员/);
    const entries = assets(3);
    entries.set(`nested/${scaleName}`, halfs([1, 1, 1, 1]));
    await expect(loadDockerViewerBundle(storedZip(entries), {})).rejects.toThrow(/不支持的成员/);
  });
});
