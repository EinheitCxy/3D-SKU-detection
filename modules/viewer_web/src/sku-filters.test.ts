import { describe, expect, it } from "vitest";
import { validateObjectIndex, type ObjectIndexEntry } from "./contracts";
import { buildSkuFacets, filterGlobalIdsBySku } from "./sku-filters";

const pendingMetadata = () => ({ status: "master_data_pending" as const, manufacturer: null, brand: null, category: null, object_kind: null });
const candidate = (sku_id: string, sku_name: string, confidence_sum: number, support_count: number, max_confidence: number) => ({ sku_id, sku_name, confidence_sum, support_count, max_confidence });
const resolvedObservation = (sku_id: string, sku_name: string, confidence: number) => ({
  schema_version: "1.0.0" as const,
  source: "personalcare" as const,
  project_id: 51,
  status: "resolved" as const,
  sku_id,
  sku_name,
  confidence,
  metadata: pendingMetadata(),
});
const unavailableObservation = () => ({ schema_version: "1.0.0" as const, source: "personalcare" as const, project_id: 51, status: "unavailable" as const, reason: "invalid_bbox" as const });

const validatedEntry = (
  globalId: string,
  instances: readonly { image_id: number; object_id: number; classification: ReturnType<typeof resolvedObservation> | ReturnType<typeof unavailableObservation>; removed?: boolean }[],
  classification: { status: "resolved" | "conflict" | "unavailable"; primary_sku_id: string | null; candidates: readonly ReturnType<typeof candidate>[]; metadata: ReturnType<typeof pendingMetadata> },
): ObjectIndexEntry => {
  const raw = {
    images: [...new Set(instances.map((instance) => instance.image_id))].sort((left, right) => left - right),
    objects: instances.map((instance) => instance.object_id).sort((left, right) => left - right),
    active_count: instances.filter((instance) => !instance.removed).length,
    removed_count: instances.filter((instance) => instance.removed).length,
    total_count: instances.length,
    instances: instances.map((instance, index) => ({
      image_id: instance.image_id,
      object_id: instance.object_id,
      bbox: [0, 0, 10, 10],
      removed: instance.removed ?? false,
      point_index_range: [index, index + 1],
      thumbnail: `thumbs/${globalId}_${index}.jpg`,
      classification: instance.classification,
    })),
    classification,
  };
  return validateObjectIndex({ [globalId]: raw }, instances.length + 1)[globalId];
};

const filterEntry = (globalId: string, instances: Parameters<typeof validatedEntry>[1], classification: Parameters<typeof validatedEntry>[2]): ObjectIndexEntry =>
  validatedEntry(globalId, instances, classification);

describe("SKU facets", () => {
  it("counts and filters only by the primary candidate", () => {
    const objects = {
      "1": filterEntry("1", [
        { image_id: 0, object_id: 0, classification: resolvedObservation("A", "产品A", 0.95) },
        { image_id: 1, object_id: 1, classification: resolvedObservation("B", "产品B", 0.9) },
      ], {
        status: "conflict", primary_sku_id: "A",
        candidates: [candidate("A", "产品A", 0.95, 1, 0.95), candidate("B", "产品B", 0.9, 1, 0.9)], metadata: pendingMetadata(),
      }),
      "2": filterEntry("2", [{ image_id: 2, object_id: 2, classification: resolvedObservation("B", "产品B", 0.8) }], {
        status: "resolved", primary_sku_id: "B", candidates: [candidate("B", "产品B", 0.8, 1, 0.8)], metadata: pendingMetadata(),
      }),
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
      "1": filterEntry("1", [{ image_id: 0, object_id: 0, classification: resolvedObservation("B", "产品B", 1) }], {
        status: "resolved", primary_sku_id: "B", candidates: [candidate("B", "产品B", 1, 1, 1)], metadata: pendingMetadata(),
      }),
      "2": filterEntry("2", [{ image_id: 1, object_id: 1, classification: resolvedObservation("A", "产品A", 1) }], {
        status: "resolved", primary_sku_id: "A", candidates: [candidate("A", "产品A", 1, 1, 1)], metadata: pendingMetadata(),
      }),
      "3": filterEntry("3", [{ image_id: 2, object_id: 2, classification: resolvedObservation("B", "产品B", 1) }], {
        status: "resolved", primary_sku_id: "B", candidates: [candidate("B", "产品B", 1, 1, 1)], metadata: pendingMetadata(),
      }),
    };
    expect(buildSkuFacets(objects)).toEqual([
      { skuId: "B", skuName: "产品B", count: 2 },
      { skuId: "A", skuName: "产品A", count: 1 },
    ]);
  });

  it("does not count a global object with only unavailable observations", () => {
    const objects = {
      "1": filterEntry("1", [{ image_id: 0, object_id: 0, classification: unavailableObservation() }], {
        status: "unavailable", primary_sku_id: null, candidates: [], metadata: pendingMetadata(),
      }),
    };
    expect(buildSkuFacets(objects)).toEqual([]);
    expect(filterGlobalIdsBySku(objects, "A")).toEqual([]);
  });
});
