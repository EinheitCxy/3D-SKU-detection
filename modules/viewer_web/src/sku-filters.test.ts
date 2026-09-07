import { describe, expect, it } from "vitest";
import { validateObjectIndex, type ObjectIndex } from "./contracts";
import { buildSkuFacets, buildSkuFacetsForGlobalIds, filterGlobalIdsBySku } from "./sku-filters";

function objects(raw: unknown): ObjectIndex {
  return validateObjectIndex(raw, 4);
}

describe("SKU facets", () => {
  it("uses only ordered_skus[0] for counts and filtering", () => {
    const value = objects({
      "1": {
        ordered_skus: [
          { sku_id: "A", sku_name: "产品A" },
          { sku_id: "B", sku_name: "产品B" },
        ],
        point_ranges: [[0, 1]],
        observations: [],
      },
      "2": { ordered_skus: [{ sku_id: "B", sku_name: "产品B" }], point_ranges: [[1, 2]], observations: [] },
    });
    expect(buildSkuFacets(value)).toEqual([
      { skuId: "A", skuName: "产品A", count: 1 },
      { skuId: "B", skuName: "产品B", count: 1 },
    ]);
    expect(filterGlobalIdsBySku(value, "A")).toEqual(["1"]);
    expect(filterGlobalIdsBySku(value, "B")).toEqual(["2"]);
  });

  it("preserves the first-candidate identity while sorting facet counts", () => {
    const value = objects({
      "1": { ordered_skus: [{ sku_id: "B", sku_name: "产品B" }], point_ranges: [[0, 1]], observations: [] },
      "2": { ordered_skus: [{ sku_id: "A", sku_name: "产品A" }], point_ranges: [[1, 2]], observations: [] },
      "3": { ordered_skus: [{ sku_id: "B", sku_name: "产品B" }], point_ranges: [[2, 3]], observations: [] },
    });
    expect(buildSkuFacets(value)).toEqual([
      { skuId: "B", skuName: "产品B", count: 2 },
      { skuId: "A", skuName: "产品A", count: 1 },
    ]);
  });

  it("summarizes only the global IDs matched by a selected facet", () => {
    const value = objects({
      "1": { ordered_skus: [{ sku_id: "A", sku_name: "产品A" }], point_ranges: [[0, 1]], observations: [] },
      "2": { ordered_skus: [{ sku_id: "B", sku_name: "产品B" }], point_ranges: [[1, 2]], observations: [] },
      "3": { ordered_skus: [{ sku_id: "A", sku_name: "产品A" }], point_ranges: [[2, 3]], observations: [] },
    });

    expect(buildSkuFacetsForGlobalIds(value, ["1", "3"])).toEqual([
      { skuId: "A", skuName: "产品A", count: 2 },
    ]);
  });

  it("does not count objects with no ordered SKU", () => {
    const value = objects({ "1": { ordered_skus: [], point_ranges: [[0, 1]], observations: [] } });
    expect(buildSkuFacets(value)).toEqual([]);
    expect(filterGlobalIdsBySku(value, "A")).toEqual([]);
  });
});

it("orders SKU drilldown IDs numerically instead of lexicographically", () => {
  const entry = { ordered_skus: [{sku_id: "A", sku_name: "Product"}], point_ranges: [], observations: [] };
  expect(filterGlobalIdsBySku({"100": entry, "10": entry, "2": entry}, "A")).toEqual(["2", "10", "100"]);
});
