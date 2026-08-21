import { describe, expect, it } from "vitest";
import { validateFootprints, validateManifest, validateObjectIndex } from "./contracts";

const validSource = {
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
  export: {
    voxel_size_m: 0.01,
    max_points: 10,
    filter_config: {
      enabled: true,
      sor_nb_neighbors: 20,
      sor_std_ratio: 2,
      keep_main_clusters: true,
      cluster_eps_scale: 5,
      cluster_min_points: 10,
      min_cluster_ratio: 0.01,
      remove_ground: true,
      ground_dist_scale: 3,
      ground_min_inlier_ratio: 0.08,
      min_remaining_ratio: 0.2,
      min_points: 1000,
    },
    exporter_source_sha256: "1".repeat(64),
    global_mapping_sha256: "2".repeat(64),
    sam3_mask_entries: [{ image_id: 7, key: "3".repeat(64), payload_sha256: "4".repeat(64) }],
  },
} as const;

const validManifest = {
  schema_version: "1.0.0",
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
  source: validSource,
  capabilities: { point_picking: false, footprint_picking: true, formal_ground_footprint: true },
} as const;

const validObjects = {
  "11": {
    images: [7],
    objects: [3],
    active_count: 1,
    removed_count: 0,
    total_count: 1,
    instances: [{ image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false, point_index_range: [0, 1], thumbnail: "thumbs/11_0.jpg" }],
  },
} as const;

const square = [[0, 0], [1, 0], [1, 1], [0, 0]];
const validFootprints = {
  metric: "da3_ground_footprint_union",
  unit: "m2",
  status: "accepted",
  value_m2: 1,
  rejection_reason: null,
  run_id: "a".repeat(32),
  support_plane: {
    point: [0, 0, 0],
    u_axis: [1, 0, 0],
    v_axis: [0, 1, 0],
    normal: [0, 0, 1],
  },
  per_global_id: {
    "11": {
      rings: [[square]],
      properties: {
        coordinate_space: "local_support_plane_meters",
        global_id: "11",
        area_m2: 0.5,
        observations_used: 1,
      },
    },
  },
  union: {
    rings: [[square]],
    properties: { coordinate_space: "local_support_plane_meters", global_id: "union", area_m2: 1 },
  },
} as const;

describe("strict bundle contracts", () => {
  it("accepts the exact manifest schema and array layout", () => {
    expect(validateManifest(validManifest)).toMatchObject({ schema_version: "1.0.0", point_count: 1 });
  });

  it("fails closed with a re-export hint when world_to_view is missing", () => {
    const { world_to_view: _missing, ...withoutWorldToView } = validManifest;
    expect(() => validateManifest(withoutWorldToView)).toThrow(/缺 world_to_view.*重新导出/);
  });

  it("fails closed with a re-export hint when the normals array is missing", () => {
    const { normals: _missing, ...withoutNormals } = validManifest.arrays;
    expect(() => validateManifest({ ...validManifest, arrays: withoutNormals })).toThrow(/缺 normals.*重新导出/);
  });

  it("rejects a normals descriptor with an unexpected path or dtype", () => {
    expect(() => validateManifest({
      ...validManifest,
      arrays: { ...validManifest.arrays, normals: { ...validManifest.arrays.normals, path: "normals.f32.bin" } },
    })).toThrow();
    expect(() => validateManifest({
      ...validManifest,
      arrays: { ...validManifest.arrays, normals: { ...validManifest.arrays.normals, dtype: "float32", path: "normals.f32.bin", byte_length: 12 } },
    })).toThrow();
    expect(() => validateManifest({
      ...validManifest,
      arrays: { ...validManifest.arrays, normals: { ...validManifest.arrays.normals, byte_length: 4 } },
    })).toThrow();
  });

  it("rejects a world_to_view that is not sixteen finite numbers", () => {
    expect(() => validateManifest({ ...validManifest, world_to_view: [1, 0, 0] })).toThrow();
    expect(() => validateManifest({ ...validManifest, world_to_view: Array.from({ length: 16 }, () => "x") })).toThrow();
  });

  it("requires world_to_view to be a proper row-major rigid affine transform", () => {
    for (const matrix of [
      Array(16).fill(0),
      [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      [1, 0.25, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, 0, 0, 1],
    ]) expect(() => validateManifest({ ...validManifest, world_to_view: matrix })).toThrow(/world_to_view/);
  });

  it("requires six finite ordered display bounds", () => {
    expect(() => validateManifest({ ...validManifest, display_bounds: [1, 2, 3] })).toThrow(/display_bounds/);
    expect(() => validateManifest({ ...validManifest, display_bounds: [2, 0, 0, 1, 1, 1] })).toThrow(/display_bounds/);
  });

  it("rejects a manifest with an unsupported array dtype or component count", () => {
    expect(() => validateManifest({
      ...validManifest,
      arrays: { ...validManifest.arrays, positions: { ...validManifest.arrays.positions, dtype: "float64" } },
    })).toThrow();
    expect(() => validateManifest({
      ...validManifest,
      arrays: { ...validManifest.arrays, colors: { ...validManifest.arrays.colors, components: 4 } },
    })).toThrow();
  });

  it("validates every manifest source provenance field", () => {
    expect(validateManifest(validManifest).source.da3_cache).toMatchObject({ schema_version: 2, frame_count: 1 });
    expect(() => validateManifest({ ...validManifest, source: { ...validSource, export: { ...validSource.export, max_points: 0 } } })).toThrow();
    expect(() => validateManifest({ ...validManifest, source: { ...validSource, da3_cache: { ...validSource.da3_cache, image_ids: [7, 7] } } })).toThrow();
    expect(() => validateManifest({ ...validManifest, source: { ...validSource, footprint: { run_id: "A".repeat(32), status: "accepted" } } })).toThrow();
  });

  it("requires exact filter, exporter, mapping, and used-SAM3 provenance", () => {
    expect(validateManifest(validManifest).source.export.sam3_mask_entries).toHaveLength(1);
    const { global_mapping_sha256: _missing, ...withoutMappingDigest } = validSource.export;
    expect(() => validateManifest({ ...validManifest, source: { ...validSource, export: withoutMappingDigest } })).toThrow();
    expect(() => validateManifest({ ...validManifest, source: { ...validSource, export: { ...validSource.export, exporter_source_sha256: "not-a-digest" } } })).toThrow();
    expect(() => validateManifest({ ...validManifest, source: { ...validSource, export: { ...validSource.export, sam3_mask_entries: [{ ...validSource.export.sam3_mask_entries[0], image_id: 7.5 }] } } })).toThrow();
  });

  it("rejects a global object index with missing fields or inconsistent counts", () => {
    expect(validateObjectIndex(validObjects, 1)).toMatchObject({ "11": { total_count: 1 } });
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], total_count: 2 } }, 1)).toThrow();
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], instances: [{ image_id: 7 }] } }, 1)).toThrow();
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], instances: [{ ...validObjects["11"].instances[0], point_index_range: [0, 2] }] } }, 1)).toThrow(/out of bounds/i);
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], instances: [{ ...validObjects["11"].instances[0], point_index_range: [1, 0] }] } }, 1)).toThrow();
  });

  it("binds thumbnails to their global-ID instance identity and forbids overlapping ranges", () => {
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], instances: [{ ...validObjects["11"].instances[0], thumbnail: "thumbs/12_0.jpg" }] } }, 1)).toThrow(/thumbnail/);
    expect(() => validateObjectIndex({
      ...validObjects,
      "12": { images: [8], objects: [4], active_count: 1, removed_count: 0, total_count: 1, instances: [{ image_id: 8, object_id: 4, bbox: [1, 2, 3, 4], removed: false, point_index_range: [0, 1], thumbnail: "thumbs/12_0.jpg" }] },
    }, 1)).toThrow(/overlap/);
  });

  it("fails closed with a re-export hint when an instance thumbnail is missing", () => {
    const { thumbnail: _missing, ...withoutThumbnail } = validObjects["11"].instances[0];
    expect(() => validateObjectIndex({
      "11": { ...validObjects["11"], instances: [withoutThumbnail] },
    }, 1)).toThrow(/缺 instance thumbnail.*重新导出/);
  });

  it("rejects unsafe or malformed instance thumbnail paths", () => {
    for (const thumbnail of [
      "../escape.jpg",
      "thumbs/../escape.jpg",
      "/absolute/thumbs/11_0.jpg",
      "thumbs/11_0.png",
      "thumbs/11_x.jpg",
      "https://evil.test/thumbs/11_0.jpg",
    ]) {
      expect(() => validateObjectIndex({
        "11": { ...validObjects["11"], instances: [{ ...validObjects["11"].instances[0], thumbnail }] },
      }, 1)).toThrow(/thumbnail/);
    }
  });

  it("derives images and all sorted object IDs from instances", () => {
    const derived = {
      "11": {
        images: [7, 8],
        objects: [3, 3],
        active_count: 2,
        removed_count: 0,
        total_count: 2,
        instances: [
          { image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false, point_index_range: [0, 1], thumbnail: "thumbs/11_0.jpg" },
          { image_id: 8, object_id: 3, bbox: [5, 6, 7, 8], removed: false, point_index_range: [1, 2], thumbnail: "thumbs/11_1.jpg" },
        ],
      },
    };
    expect(validateObjectIndex(derived, 2)["11"]).toMatchObject({ images: [7, 8], objects: [3, 3] });
    expect(() => validateObjectIndex({ "11": { ...derived["11"], images: [7] } }, 2)).toThrow();
    expect(() => validateObjectIndex({ "11": { ...derived["11"], objects: [3] } }, 2)).toThrow();
  });

  it("enforces accepted and rejected footprint value/geometry relations", () => {
    expect(validateFootprints(validFootprints).value_m2).toBe(1);
    expect(() => validateFootprints({ ...validFootprints, rejection_reason: "unexpected" })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, per_global_id: {}, union: validFootprints.union })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, union: { ...validFootprints.union, properties: { ...validFootprints.union.properties, area_m2: 0.5 } } })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, per_global_id: { "11": { ...validFootprints.per_global_id["11"], properties: { coordinate_space: "local_support_plane_meters", global_id: "11", area_m2: 0.5 } } } })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, status: "rejected", value_m2: 0 })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, status: "accepted", value_m2: null })).toThrow();
    expect(validateFootprints({ ...validFootprints, status: "rejected", value_m2: null, rejection_reason: "input rejected", support_plane: null, union: null, per_global_id: {} }).status).toBe("rejected");
    expect(() => validateFootprints({ ...validFootprints, status: "rejected", value_m2: null, rejection_reason: null, support_plane: null, union: null, per_global_id: {} })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, status: "rejected", value_m2: null, support_plane: null, union: validFootprints.union, per_global_id: {} })).toThrow();
  });
});
