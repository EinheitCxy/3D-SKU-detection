import { describe, expect, it } from "vitest";
import type { ObjectIndexEntry } from "./contracts";
import { buildSkuFacets, filterGlobalIdsBySku } from "./sku-filters";

const pendingMetadata = () => ({ status: "master_data_pending" as const, manufacturer: null, brand: null, category: null, object_kind: null });
const candidate = (sku_id: string, sku_name: string, confidence_sum: number, support_count: number, max_confidence: number) => ({ sku_id, sku_name, confidence_sum, support_count, max_confidence });
const unavailableObservation = () => ({ schema_version: "1.0.0" as const, source: "personalcare" as const, project_id: 51, status: "unavailable" as const, reason: "invalid_bbox" as const });
const filterEntry = (candidates: readonly ReturnType<typeof candidate>[]): ObjectIndexEntry => ({
  images: [0], objects: [0], active_count: 1, removed_count: 0, total_count: 1,
  instances: [{ image_id: 0, object_id: 0, bbox: [0, 0, 1, 1], removed: false, point_index_range: [0, 1], thumbnail: "thumbs/1_0.jpg", classification: unavailableObservation() }],
  classification: { status: candidates.length > 1 ? "conflict" : "resolved", primary_sku_id: candidates[0]?.sku_id ?? null, candidates, metadata: pendingMetadata() },
});

describe("SKU facets", () => {
  it("counts and filters only by the primary candidate", () => {
    const objects = {
      "1": filterEntry([candidate("A", "产品A", 1.1, 2, 0.6), candidate("B", "产品B", 0.9, 1, 0.9)]),
      "2": filterEntry([candidate("B", "产品B", 0.8, 1, 0.8)]),
    };
    expect(buildSkuFacets(objects)).toEqual([
      { skuId: "A", skuName: "产品A", count: 1 },
      { skuId: "B", skuName: "产品B", count: 1 },
    ]);
    expect(filterGlobalIdsBySku(objects, "A")).toEqual(["1"]);
    expect(filterGlobalIdsBySku(objects, "B")).toEqual(["2"]);
  });

  it("sorts facets by descending count then SKU ID", () => {
    const objects = {
      "1": filterEntry([candidate("B", "产品B", 1, 1, 1)]),
      "2": filterEntry([candidate("A", "产品A", 1, 1, 1)]),
      "3": filterEntry([candidate("B", "产品B", 1, 1, 1)]),
    };
    expect(buildSkuFacets(objects)).toEqual([
      { skuId: "B", skuName: "产品B", count: 2 },
      { skuId: "A", skuName: "产品A", count: 1 },
    ]);
  });
});
