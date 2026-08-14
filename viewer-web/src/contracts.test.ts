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
  export: { voxel_size_m: 0.01, max_points: 10 },
} as const;

const validManifest = {
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
    instances: [{ image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false }],
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

  it("rejects a global object index with missing fields or inconsistent counts", () => {
    expect(validateObjectIndex(validObjects)).toMatchObject({ "11": { total_count: 1 } });
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], total_count: 2 } })).toThrow();
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], instances: [{ image_id: 7 }] } })).toThrow();
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
          { image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false },
          { image_id: 8, object_id: 3, bbox: [5, 6, 7, 8], removed: false },
        ],
      },
    };
    expect(validateObjectIndex(derived)["11"]).toMatchObject({ images: [7, 8], objects: [3, 3] });
    expect(() => validateObjectIndex({ "11": { ...derived["11"], images: [7] } })).toThrow();
    expect(() => validateObjectIndex({ "11": { ...derived["11"], objects: [3] } })).toThrow();
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
