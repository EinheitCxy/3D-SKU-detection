import { describe, expect, it } from "vitest";
import { validateCurrent, validateManifest, validateObjectIndex } from "./contracts";

const validManifest = {
  schema_version: "3.0.0",
  dataset_name: "floor_display6",
  backend: "DA3",
  frame_count: 11,
  display_bounds: [0, 0, 0, 1, 1, 1],
  world_to_view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
};

const validObjects = {
  "1": {
    ordered_skus: [
      { sku_id: "123", sku_name: "产品" },
      { sku_id: "56642", sku_name: "其他品类" },
    ],
    point_ranges: [[0, 2]],
    observations: [
      { image_id: 0, object_id: 3, removed: false, thumbnail: "thumbs/1_0.jpg" },
    ],
  },
  "2": {
    ordered_skus: [{ sku_id: "456", sku_name: "另一产品" }],
    point_ranges: [[2, 3]],
    observations: [],
  },
};

describe("minimal viewer contracts", () => {
  it("accepts CURRENT with only a non-empty run ID and ignores unknown fields", () => {
    expect(validateCurrent({ run_id: "run-20260826", complete: true, extra: "ignored" })).toEqual({
      run_id: "run-20260826",
    });
  });

  it("rejects a missing or empty CURRENT run ID", () => {
    expect(() => validateCurrent({})).toThrow(/run_id/);
    expect(() => validateCurrent({ run_id: "  " })).toThrow(/run_id/);
  });

  it("accepts the minimal manifest and ignores unknown fields", () => {
    expect(validateManifest({ ...validManifest, source: { ignored: true } })).toEqual(validManifest);
  });

  it("checks schema, dataset, frame count, bounds, and matrix shape", () => {
    expect(() => validateManifest({ ...validManifest, schema_version: "2.0.0" })).toThrow(/schema_version/);
    expect(() => validateManifest({ ...validManifest, dataset_name: "  " })).toThrow(/dataset_name/);
    expect(() => validateManifest({ ...validManifest, frame_count: -1 })).toThrow(/frame_count/);
    expect(() => validateManifest({ ...validManifest, display_bounds: [0, 0, 0] })).toThrow(/display_bounds/);
    expect(() => validateManifest({ ...validManifest, world_to_view: [1, 0, 0] })).toThrow(/world_to_view/);
    expect(() => validateManifest({ ...validManifest, world_to_view: Array(16).fill(Number.NaN) })).toThrow(/world_to_view/);
  });

  it("requires a non-empty manifest backend string", () => {
    expect(() => validateManifest({ ...validManifest, backend: undefined })).toThrow(/backend/);
    expect(() => validateManifest({ ...validManifest, backend: "  " })).toThrow(/backend/);
    expect(validateManifest({ ...validManifest, backend: "DA3", extra: "ignored" }).backend).toBe("DA3");
  });

  it("accepts minimal objects, preserves SKU and range order, and ignores unknown fields", () => {
    expect(validateObjectIndex({
      "1": { ...validObjects["1"], evidence: { ignored: true } },
      "2": validObjects["2"],
    }, 3)).toEqual(validObjects);
  });

  it("requires numeric global IDs and non-empty SKU strings", () => {
    expect(() => validateObjectIndex({ "-1": validObjects["1"] }, 3)).toThrow(/global ID/);
    expect(() => validateObjectIndex({ "1": { ...validObjects["1"], ordered_skus: [{ sku_id: "", sku_name: "产品" }] } }, 3)).toThrow(/sku_id/);
    expect(() => validateObjectIndex({ "1": { ...validObjects["1"], ordered_skus: [{ sku_id: "123", sku_name: "  " }] } }, 3)).toThrow(/sku_name/);
  });

  it("rejects malformed or out-of-bounds point ranges", () => {
    expect(() => validateObjectIndex({ "1": { ...validObjects["1"], point_ranges: [[0, 4]] } }, 3)).toThrow(/point_ranges|point range/);
    expect(() => validateObjectIndex({ "1": { ...validObjects["1"], point_ranges: [[2, 1]] } }, 3)).toThrow(/point_ranges|point range/);
    expect(() => validateObjectIndex({ "1": { ...validObjects["1"], point_ranges: [[0, 1.5]] } }, 3)).toThrow(/point_ranges|point range/);
  });

  it("requires observations with the minimal thumbnail fields", () => {
    expect(() => validateObjectIndex({
      "1": { ordered_skus: [], point_ranges: [] },
    }, 3)).toThrow(/observations/);
    expect(() => validateObjectIndex({
      "1": { ...validObjects["1"], observations: [{ image_id: 0, object_id: 3, removed: false }] },
    }, 3)).toThrow(/thumbnail/);
    expect(() => validateObjectIndex({
      "1": { ...validObjects["1"], observations: [{ image_id: 0, object_id: 3, removed: "false", thumbnail: "thumbs/1_0.jpg" }] },
    }, 3)).toThrow(/removed/);
    expect(() => validateObjectIndex({
      "1": { ...validObjects["1"], observations: [{ image_id: -1, object_id: 3, removed: false, thumbnail: "thumbs/1_0.jpg" }] },
    }, 3)).toThrow(/image_id/);
  });

  it("rejects globally overlapping non-empty point ranges", () => {
    expect(() => validateObjectIndex({
      "1": validObjects["1"],
      "2": { ...validObjects["2"], point_ranges: [[1, 3]] },
    }, 3)).toThrow(/overlap/);
  });
});
