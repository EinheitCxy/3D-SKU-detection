import { describe, expect, it } from "vitest";
import { validateFootprints, validateManifest, validateObjectIndex } from "./contracts";

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
  source: {},
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
  per_global_id: { "11": { rings: [[square]], properties: { global_id: "11" } } },
  union: { rings: [[square]], properties: { global_id: "union" } },
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

  it("rejects a global object index with missing fields or inconsistent counts", () => {
    expect(validateObjectIndex(validObjects)).toMatchObject({ "11": { total_count: 1 } });
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], total_count: 2 } })).toThrow();
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], instances: [{ image_id: 7 }] } })).toThrow();
  });

  it("enforces accepted and rejected footprint value/geometry relations", () => {
    expect(validateFootprints(validFootprints).value_m2).toBe(1);
    expect(() => validateFootprints({ ...validFootprints, status: "rejected", value_m2: 0 })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, status: "accepted", value_m2: null })).toThrow();
    expect(validateFootprints({ ...validFootprints, status: "rejected", value_m2: null, support_plane: null, union: null, per_global_id: {} }).status).toBe("rejected");
    expect(() => validateFootprints({ ...validFootprints, status: "rejected", value_m2: null, support_plane: null, union: validFootprints.union, per_global_id: {} })).toThrow();
  });
});
