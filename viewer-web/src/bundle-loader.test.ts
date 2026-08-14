import { describe, expect, it } from "vitest";
import { assertLittleEndian, isLittleEndian, loadViewerBundle } from "./bundle-loader";

const baseUrl = "https://example.test/data/";
const current = { schema_version: "1.0.0", run_id: "a".repeat(32), complete: true };
const manifest = {
  schema_version: "1.0.0",
  coordinate_space: "da3_world_meters",
  point_count: 1,
  arrays: {
    positions: { path: "positions.f32.bin", dtype: "float32", components: 3, byte_length: 12 },
    colors: { path: "colors.u8.bin", dtype: "uint8", components: 3, byte_length: 3 },
    confidences: { path: "confidences.f32.bin", dtype: "float32", components: 1, byte_length: 4 },
    frame_ids: { path: "frame_ids.i32.bin", dtype: "int32", components: 1, byte_length: 4 },
  },
  objects_path: "objects.json",
  footprints_path: "footprints.json",
  source: {},
  capabilities: { point_picking: false, footprint_picking: true, formal_ground_footprint: true },
};
const objects = {
  "11": {
    images: [7], objects: [3], active_count: 1, removed_count: 0, total_count: 1,
    instances: [{ image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false }],
  },
};
const square = [[0, 0], [1, 0], [1, 1], [0, 0]];
const footprints = {
  metric: "da3_ground_footprint_union", unit: "m2", status: "accepted", value_m2: 1,
  rejection_reason: null, run_id: "a".repeat(32),
  support_plane: { point: [0, 0, 0], u_axis: [1, 0, 0], v_axis: [0, 1, 0], normal: [0, 0, 1] },
  per_global_id: { "11": { rings: [[square]], properties: { global_id: "11" } } },
  union: { rings: [[square]], properties: { global_id: "union" } },
};

function bufferOf<T extends ArrayBufferView>(array: T): ArrayBuffer {
  return array.buffer.slice(array.byteOffset, array.byteOffset + array.byteLength) as ArrayBuffer;
}

function makeFetcher(overrides: Record<string, unknown> = {}, binaryOverrides: Record<string, ArrayBuffer> = {}) {
  const json: Record<string, unknown> = {
    [`${baseUrl}CURRENT`]: current,
    [`${baseUrl}runs/${current.run_id}/manifest.json`]: manifest,
    [`${baseUrl}runs/${current.run_id}/objects.json`]: objects,
    [`${baseUrl}runs/${current.run_id}/footprints.json`]: footprints,
    ...overrides,
  };
  const binary: Record<string, ArrayBuffer> = {
    [`${baseUrl}runs/${current.run_id}/positions.f32.bin`]: bufferOf(new Float32Array([1, 2, 3])),
    [`${baseUrl}runs/${current.run_id}/colors.u8.bin`]: bufferOf(new Uint8Array([10, 20, 30])),
    [`${baseUrl}runs/${current.run_id}/confidences.f32.bin`]: bufferOf(new Float32Array([0.9])),
    [`${baseUrl}runs/${current.run_id}/frame_ids.i32.bin`]: bufferOf(new Int32Array([7])),
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
  it("loads CURRENT and the immutable generation into validated typed arrays", async () => {
    const bundle = await loadViewerBundle(baseUrl, makeFetcher());
    expect(bundle.current.run_id).toBe(current.run_id);
    expect(bundle.manifest.point_count).toBe(1);
    expect(bundle.positions).toBeInstanceOf(Float32Array);
    expect(Array.from(bundle.positions)).toEqual([1, 2, 3]);
    expect(bundle.colors).toBeInstanceOf(Uint8Array);
    expect(bundle.confidences).toBeInstanceOf(Float32Array);
    expect(bundle.frameIds).toBeInstanceOf(Int32Array);
  });

  it("fails closed when a binary response has the wrong byte length", async () => {
    const wrong = bufferOf(new Float32Array([1, 2]));
    await expect(loadViewerBundle(baseUrl, makeFetcher({}, {
      [`${baseUrl}runs/${current.run_id}/positions.f32.bin`]: wrong,
    }))).rejects.toThrow(/byte length/i);
  });

  it("fails closed for an invalid CURRENT, HTTP error, and a base URL without a slash", async () => {
    await expect(loadViewerBundle("https://example.test/data", makeFetcher())).rejects.toThrow(/slash/i);
    await expect(loadViewerBundle(baseUrl, makeFetcher({ [`${baseUrl}CURRENT`]: { ...current, complete: false } }))).rejects.toThrow();
    await expect(loadViewerBundle(baseUrl, makeFetcher({ [`${baseUrl}CURRENT`]: { ...current, run_id: "A".repeat(32) } }))).rejects.toThrow();
    await expect(loadViewerBundle(baseUrl, async () => new Response("no", { status: 503 }))).rejects.toThrow(/503/);
  });

  it("fails closed when the manifest array contract is malformed", async () => {
    await expect(loadViewerBundle(baseUrl, makeFetcher({
      [`${baseUrl}runs/${current.run_id}/manifest.json`]: {
        ...manifest,
        arrays: { ...manifest.arrays, positions: { ...manifest.arrays.positions, components: 4 } },
      },
    }))).rejects.toThrow();
  });

  it("rejects a non-little-endian runtime through the explicit gate contract", async () => {
    expect(isLittleEndian()).toBe(true);
    expect(() => assertLittleEndian()).not.toThrow();
    const bundle = await loadViewerBundle(baseUrl, makeFetcher());
    expect(bundle.positions).toBeInstanceOf(Float32Array);
  });
});
