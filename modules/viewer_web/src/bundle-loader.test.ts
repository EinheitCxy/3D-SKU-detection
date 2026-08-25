import { describe, expect, it } from "vitest";
import { assertLittleEndian, isLittleEndian, loadViewerBundle } from "./bundle-loader";

const baseUrl = "https://example.test/data/";
const current = { schema_version: "2.0.0", run_id: "a".repeat(32), complete: true };
const source = {
  da3_cache: {
    schema_version: 2,
    source_model: "depth-anything/DA3NESTED-GIANT-LARGE",
    affine_convention: "pixel_center_v1",
    preprocess_resolution: 2,
    preprocess_method: "upper_bound_resize",
    frame_count: 1,
    processed_size: [2, 2],
    image_ids: [7],
    source_image_sha256: ["0".repeat(64)],
  },
  footprint: { run_id: "a".repeat(32), status: "accepted" },
  sam3_mask: {
    schema: "sam3_self_exemplar_processed_mask_cache_v1",
    coordinate_space: "da3_processed_pixels",
    producer: "sku_matching",
  },
  export: {
    voxel_size_m: 0.01, max_points: 10,
    filter_config: { enabled: true, sor_nb_neighbors: 20, sor_std_ratio: 2, keep_main_clusters: true, cluster_eps_scale: 5, cluster_min_points: 10, min_cluster_ratio: 0.01, remove_ground: true, ground_dist_scale: 3, ground_min_inlier_ratio: 0.08, min_remaining_ratio: 0.2, min_points: 1000 },
    exporter_source_sha256: "1".repeat(64), global_mapping_sha256: "2".repeat(64),
  },
};
const manifest = {
  schema_version: "2.0.0",
  coordinate_space: "da3_world_meters",
  point_count: 1,
  display_bounds: [1, 2, 3, 1, 2, 3],
  arrays: {
    positions: { path: "positions.f32.bin", dtype: "float32", components: 3, byte_length: 12 },
    colors: { path: "colors.u8.bin", dtype: "uint8", components: 3, byte_length: 3 },
    normals: { path: "normals.i8.bin", dtype: "int8", components: 3, byte_length: 3 },
  },
  world_to_view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  coordinate_convention: "DA3 native CV coordinates; world_to_view maps to viewer Y-up (row-major)",
  objects_path: "objects.json",
  footprints_path: "footprints.json",
  source,
  capabilities: { point_picking: false, footprint_picking: true, formal_ground_footprint: true },
};
const objects = {
  "11": {
    images: [7], objects: [3], active_count: 1, removed_count: 0, total_count: 1,
    instances: [{ image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false, point_index_range: [0, 1], thumbnail: "thumbs/11_0.jpg" }],
  },
};
const square = [[0, 0], [1, 0], [1, 1], [0, 0]];
const footprints = {
  metric: "da3_self_exemplar_ground_footprint_union", unit: "m2", status: "accepted", value_m2: 1,
  rejection_reason: null, run_id: "a".repeat(32),
  support_plane: { point: [0, 0, 0], u_axis: [1, 0, 0], v_axis: [0, 1, 0], normal: [0, 0, 1] },
  per_global_id: {
    "11": {
      rings: [[square]],
      properties: { coordinate_space: "local_support_plane_meters", global_id: "11", area_m2: 0.5, observations_used: 1 },
    },
  },
  union: {
    rings: [[square]],
    properties: { coordinate_space: "local_support_plane_meters", global_id: "union", area_m2: 1 },
  },
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
    [`${baseUrl}runs/${current.run_id}/normals.i8.bin`]: bufferOf(new Int8Array([0, 0, 127])),
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
  it("uses no-store only for the mutable CURRENT pointer", async () => {
    const calls: Array<readonly [RequestInfo | URL, RequestInit | undefined]> = [];
    const fetcher: typeof fetch = async (input, init) => {
      calls.push([input, init]);
      return makeFetcher()(input);
    };
    await loadViewerBundle(baseUrl, fetcher);
    expect(calls.find(([url]) => String(url).endsWith("/CURRENT"))?.[1]).toEqual({ cache: "no-store" });
    expect(calls.filter(([url]) => !String(url).endsWith("/CURRENT")).every(([, init]) => init === undefined)).toBe(true);
  });

  it("loads CURRENT and the immutable generation into validated typed arrays", async () => {
    const bundle = await loadViewerBundle(baseUrl, makeFetcher());
    expect(bundle.current.run_id).toBe(current.run_id);
    expect(bundle.manifest.point_count).toBe(1);
    expect(bundle.positions).toBeInstanceOf(Float32Array);
    expect(Array.from(bundle.positions)).toEqual([1, 2, 3]);
    expect(bundle.colors).toBeInstanceOf(Uint8Array);
    expect(bundle.normals).toBeInstanceOf(Int8Array);
    expect(Array.from(bundle.normals)).toEqual([0, 0, 127]);
    expect(bundle.generationUrl).toBe(`${baseUrl}runs/${current.run_id}/`);
    expect(bundle.objects["11"].instances[0].thumbnail).toBe("thumbs/11_0.jpg");
    expect(new URL(bundle.objects["11"].instances[0].thumbnail, bundle.generationUrl).toString())
      .toBe(`${baseUrl}runs/${current.run_id}/thumbs/11_0.jpg`);
  });

  it("fails closed with a re-export hint when an instance thumbnail is missing", async () => {
    await expect(loadViewerBundle(baseUrl, makeFetcher({
      [`${baseUrl}runs/${current.run_id}/objects.json`]: {
        "11": {
          images: [7], objects: [3], active_count: 1, removed_count: 0, total_count: 1,
          instances: [{ image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false, point_index_range: [0, 1] }],
        },
      },
    }))).rejects.toThrow(/缺 instance thumbnail.*重新导出/);
  });

  it("validates the manifest before fetching objects, footprints, or binaries", async () => {
    const calls: string[] = [];
    const fetcher: typeof fetch = async (input) => {
      const url = String(input);
      calls.push(url);
      if (url.endsWith("CURRENT")) return new Response(JSON.stringify(current));
      if (url.endsWith("manifest.json")) return new Response(JSON.stringify({ ...manifest, world_to_view: Array(16).fill(0) }));
      return new Response("must not fetch", { status: 500 });
    };
    await expect(loadViewerBundle(baseUrl, fetcher)).rejects.toThrow(/world_to_view/);
    expect(calls).toEqual([`${baseUrl}CURRENT`, `${baseUrl}runs/${current.run_id}/manifest.json`]);
  });

  it("fails closed when an instance thumbnail carries an unsafe path", async () => {
    await expect(loadViewerBundle(baseUrl, makeFetcher({
      [`${baseUrl}runs/${current.run_id}/objects.json`]: {
        "11": {
          images: [7], objects: [3], active_count: 1, removed_count: 0, total_count: 1,
          instances: [{ image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false, point_index_range: [0, 1], thumbnail: "thumbs/../../secrets.jpg" }],
        },
      },
    }))).rejects.toThrow(/thumbnail/);
  });

  it("fails closed when the bundle has no normals binary or a wrong-length normals array", async () => {
    const runUrl = `${baseUrl}runs/${current.run_id}/`;
    const withoutNormals = makeFetcher({
      [`${runUrl}manifest.json`]: { ...manifest, arrays: { positions: manifest.arrays.positions, colors: manifest.arrays.colors } },
    });
    await expect(loadViewerBundle(baseUrl, withoutNormals)).rejects.toThrow(/缺 normals.*重新导出/);
    await expect(loadViewerBundle(baseUrl, makeFetcher({}, {
      [`${runUrl}normals.i8.bin`]: bufferOf(new Int8Array([0, 0])),
    }))).rejects.toThrow(/byte length/i);
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

  it("fails closed when real JSON responses tamper with source binding", async () => {
    await expect(loadViewerBundle(baseUrl, makeFetcher({
      [`${baseUrl}runs/${current.run_id}/manifest.json`]: {
        ...manifest,
        source: { ...source, footprint: { run_id: "b".repeat(32), status: "accepted" } },
      },
    }))).rejects.toThrow();
    await expect(loadViewerBundle(baseUrl, makeFetcher({
      [`${baseUrl}runs/${current.run_id}/manifest.json`]: {
        ...manifest,
        source: { ...source, footprint: { run_id: source.footprint.run_id, status: "rejected" } },
      },
    }))).rejects.toThrow();
  });

  it("fails closed when real JSON responses tamper with instance-derived arrays", async () => {
    await expect(loadViewerBundle(baseUrl, makeFetcher({
      [`${baseUrl}runs/${current.run_id}/objects.json`]: {
        "11": {
          images: [999], objects: [3], active_count: 1, removed_count: 0, total_count: 1,
          instances: [{ image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false, point_index_range: [0, 1], thumbnail: "thumbs/11_0.jpg" }],
        },
      },
    }))).rejects.toThrow();
  });

  it("rejects a manifest with a non-canonical SAM3 source", async () => {
    await expect(loadViewerBundle(baseUrl, makeFetcher({
      [`${baseUrl}runs/${current.run_id}/manifest.json`]: {
        ...manifest,
        source: { ...source, sam3_mask: { ...source.sam3_mask, coordinate_space: "source_pixels" } },
      },
    }))).rejects.toThrow();
  });

  it("rejects v1 with explicit rerun guidance", async () => {
    await expect(loadViewerBundle(baseUrl, makeFetcher({
      [`${baseUrl}CURRENT`]: { ...current, schema_version: "1.0.0" },
    }))).rejects.toThrow(/rerun matching, footprint, and viewer export/i);
  });

  it("fails closed when the manifest lacks world_to_view or instance ranges exceed point_count", async () => {
    await expect(loadViewerBundle(baseUrl, makeFetcher({
      [`${baseUrl}runs/${current.run_id}/manifest.json`]: { ...manifest, world_to_view: undefined },
    }))).rejects.toThrow(/world_to_view/);
    await expect(loadViewerBundle(baseUrl, makeFetcher({
      [`${baseUrl}runs/${current.run_id}/manifest.json`]: { ...manifest, world_to_view: [1, 0, 0] },
    }))).rejects.toThrow();
    await expect(loadViewerBundle(baseUrl, makeFetcher({
      [`${baseUrl}runs/${current.run_id}/objects.json`]: {
        "11": {
          images: [7], objects: [3], active_count: 1, removed_count: 0, total_count: 1,
          instances: [{ image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false, point_index_range: [0, 2], thumbnail: "thumbs/11_0.jpg" }],
        },
      },
    }))).rejects.toThrow(/out of bounds/i);
  });

  it("fails closed when real JSON responses tamper with exact footprint properties", async () => {
    await expect(loadViewerBundle(baseUrl, makeFetcher({
      [`${baseUrl}runs/${current.run_id}/footprints.json`]: {
        ...footprints,
        per_global_id: {
          "11": {
            ...footprints.per_global_id["11"],
            properties: { coordinate_space: "local_support_plane_meters", global_id: "11", area_m2: 0.5 },
          },
        },
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
