import { describe, expect, it } from "vitest";
import type { ObjectIndex } from "./contracts";
import { buildSelectedObjectView, canFocusGlobalId, entryHasGeometry, formatDatasetSummary, listGlobalIds, summarizeObjectCounts, summarizeObservationCounts } from "./presentation";

const objects: ObjectIndex = {
  "2": { ordered_skus: [{ sku_id: "B", sku_name: "产品B" }], point_ranges: [], observations: [] },
  "11": {
    ordered_skus: [{ sku_id: "A", sku_name: "产品A" }, { sku_id: "C", sku_name: "产品C" }],
    point_ranges: [[0, 2]],
    observations: [
      { image_id: 7, object_id: 3, removed: false, thumbnail: "thumbs/11_0.jpg" },
      { image_id: 8, object_id: 4, removed: true, thumbnail: "thumbs/11_1.jpg" },
    ],
  },
};

describe("minimal presentation", () => {
  it("formats the real dataset frame count", () => {
    expect(formatDatasetSummary("floor_display6", 11)).toBe("floor_display6 · 11 frames");
  });

  it("uses only point ranges to determine focus availability", () => {
    expect(entryHasGeometry(objects["11"])).toBe(true);
    expect(entryHasGeometry(objects["2"])).toBe(false);
    expect(canFocusGlobalId(objects["11"])).toBe(true);
    expect(canFocusGlobalId(objects["2"])).toBe(false);
  });

  it("builds Selected Object content with resolved thumbnail URLs and removed observations", () => {
    expect(buildSelectedObjectView(objects, "11", "https://example.test/runs/run-1/")).toEqual({
      globalId: "11",
      orderedSkus: [{ sku_id: "A", sku_name: "产品A" }, { sku_id: "C", sku_name: "产品C" }],
      observations: [
        { imageId: 7, objectId: 3, removed: false, thumbnailUrl: "https://example.test/runs/run-1/thumbs/11_0.jpg" },
        { imageId: 8, objectId: 4, removed: true, thumbnailUrl: "https://example.test/runs/run-1/thumbs/11_1.jpg" },
      ],
    });
    expect(buildSelectedObjectView(objects, "404", "https://example.test/runs/run-1/")).toBeNull();
    expect(listGlobalIds(objects)).toEqual(["2", "11"]);
  });

  it("derives total and visible counts from the object index and current visible IDs", () => {
    expect(summarizeObjectCounts(objects, new Set(["11"]))).toEqual({ total: 2, visible: 1 });
    expect(summarizeObjectCounts(objects, new Set(["11", "missing"]))).toEqual({ total: 2, visible: 1 });
  });

  it("derives observation total, active, and removed counts", () => {
    expect(summarizeObservationCounts(Object.values(objects).flatMap((object) => object.observations ?? []))).toEqual({
      total: 2,
      active: 1,
      removed: 1,
    });
    expect(summarizeObservationCounts(objects["11"].observations ?? [])).toEqual({
      total: 2,
      active: 1,
      removed: 1,
    });
  });
});
