import { describe, expect, it } from "vitest";
import { loadViewerBundle } from "./bundle-loader";

const baseUrl = "https://example.test/viewer/";
const current = { run_id: "run-20260826", complete: true };
const manifest = {
  schema_version: "3.0.0",
  dataset_name: "floor_display6",
  backend: "DA3",
  frame_count: 2,
  display_bounds: [0, 0, 0, 1, 1, 1],
  world_to_view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
};
const objects = {
  "1": {
    ordered_skus: [{ sku_id: "123", sku_name: "产品" }],
    point_ranges: [[0, 1]],
    observations: [{ image_id: 0, object_id: 3, removed: false, thumbnail: "thumbs/1_0.jpg" }],
  },
};

function bufferOf<T extends ArrayBufferView>(array: T): ArrayBuffer {
  return array.buffer.slice(array.byteOffset, array.byteOffset + array.byteLength) as ArrayBuffer;
}

function makeFetcher(
  overrides: Record<string, unknown> = {},
  binaryOverrides: Record<string, ArrayBuffer> = {},
) {
  const runUrl = `${baseUrl}runs/${current.run_id}/`;
  const json: Record<string, unknown> = {
    [`${baseUrl}CURRENT`]: current,
    [`${runUrl}manifest.json`]: manifest,
    [`${runUrl}objects.json`]: objects,
    ...overrides,
  };
  const binary: Record<string, ArrayBuffer> = {
    [`${runUrl}positions.f32.bin`]: bufferOf(new Float32Array([1, 2, 3, 4, 5, 6])),
    [`${runUrl}colors.u8.bin`]: bufferOf(new Uint8Array([10, 20, 30, 40, 50, 60])),
    [`${runUrl}normals.i8.bin`]: bufferOf(new Int8Array([0, 0, 127, 0, 127, 0])),
    ...binaryOverrides,
  };
  return async (input: RequestInfo | URL): Promise<Response> => {
    const url = String(input);
    if (url in json) return new Response(JSON.stringify(json[url]), { status: 200 });
    if (url in binary) return new Response(binary[url], { status: 200 });
    return new Response("missing", { status: 404 });
  };
}

describe("loadViewerBundle", () => {
  it("uses no-store only for CURRENT and loads the fixed minimal bundle", async () => {
    const calls: Array<readonly [RequestInfo | URL, RequestInit | undefined]> = [];
    const fetcher: typeof fetch = async (input, init) => {
      calls.push([input, init]);
      return makeFetcher()(input);
    };
    const bundle = await loadViewerBundle(baseUrl, fetcher);
    expect(calls.find(([url]) => String(url).endsWith("/CURRENT"))?.[1]).toEqual({ cache: "no-store" });
    expect(calls.filter(([url]) => !String(url).endsWith("/CURRENT")).every(([, init]) => init === undefined)).toBe(true);
    expect(bundle.manifest.dataset_name).toBe("floor_display6");
    expect(bundle.pointCount).toBe(2);
    expect(bundle.positions).toBeInstanceOf(Float32Array);
    expect(Array.from(bundle.positions)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(bundle.colors).toBeInstanceOf(Uint8Array);
    expect(bundle.normals).toBeInstanceOf(Int8Array);
    expect(bundle.objects).toEqual(objects);
    expect(bundle.generationUrl).toBe(`${baseUrl}runs/${current.run_id}/`);
    expect(Object.keys(bundle)).toEqual(["manifest", "objects", "positions", "colors", "normals", "pointCount", "generationUrl"]);
  });

  it("validates the manifest before fetching objects or binary arrays", async () => {
    const calls: string[] = [];
    const fetcher: typeof fetch = async (input) => {
      const url = String(input);
      calls.push(url);
      if (url.endsWith("CURRENT")) return new Response(JSON.stringify(current));
      if (url.endsWith("manifest.json")) return new Response(JSON.stringify({ ...manifest, world_to_view: [1, 2] }));
      return new Response("must not fetch", { status: 500 });
    };
    await expect(loadViewerBundle(baseUrl, fetcher)).rejects.toThrow(/world_to_view/);
    expect(calls).toEqual([`${baseUrl}CURRENT`, `${baseUrl}runs/${current.run_id}/manifest.json`]);
  });

  it("derives point count from positions and checks color and normal shapes", async () => {
    const runUrl = `${baseUrl}runs/${current.run_id}/`;
    await expect(loadViewerBundle(baseUrl, makeFetcher({}, {
      [`${runUrl}colors.u8.bin`]: bufferOf(new Uint8Array([1, 2, 3])),
    }))).rejects.toThrow(/shape|length|color/i);
    await expect(loadViewerBundle(baseUrl, makeFetcher({}, {
      [`${runUrl}normals.i8.bin`]: bufferOf(new Int8Array([1, 2, 3])),
    }))).rejects.toThrow(/shape|length|normal/i);
    await expect(loadViewerBundle(baseUrl, makeFetcher({}, {
      [`${runUrl}positions.f32.bin`]: bufferOf(new Uint8Array([1, 2, 3])),
    }))).rejects.toThrow(/positions|length|shape/i);
  });

  it("validates object point ranges against the derived point count", async () => {
    await expect(loadViewerBundle(baseUrl, makeFetcher({
      [`${baseUrl}runs/${current.run_id}/objects.json`]: {
        "1": { ...objects["1"], point_ranges: [[0, 3]] },
      },
    }))).rejects.toThrow(/point range|out of bounds/i);
  });

  it("fails closed for an invalid base URL or HTTP response", async () => {
    await expect(loadViewerBundle("https://example.test/viewer", makeFetcher())).rejects.toThrow(/slash/i);
    await expect(loadViewerBundle(baseUrl, async () => new Response("no", { status: 503 }))).rejects.toThrow(/503/);
  });
});
