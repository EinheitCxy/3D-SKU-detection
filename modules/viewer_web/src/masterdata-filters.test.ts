import { describe, expect, it } from "vitest";
import { validateObjectIndex, type ObjectIndex, type SkuMasterDataIndex } from "./contracts";
import { buildMasterDataFacets, filterGlobalIdsByMasterData } from "./masterdata-filters";

function objects(raw: unknown): ObjectIndex {
  return validateObjectIndex(raw, 4);
}

const masterData: SkuMasterDataIndex = {
  A: { manufacturer: "百事", brand: "冲劲", category: "饮料酒水", is_posm: false },
  B: { manufacturer: "可口可乐", brand: "Coca Cola", category: "饮料酒水", is_posm: true },
};

describe("master-data facets", () => {
  it("counts each primary SKU under manufacturer, brand, and category", () => {
    const value = objects({
      "1": { ordered_skus: [{ sku_id: "A", sku_name: "产品A" }], point_ranges: [[0, 1]], observations: [] },
      "2": { ordered_skus: [{ sku_id: "B", sku_name: "产品B" }], point_ranges: [[1, 2]], observations: [] },
      "3": { ordered_skus: [{ sku_id: "B", sku_name: "产品B" }], point_ranges: [[2, 3]], observations: [] },
    });

    expect(buildMasterDataFacets(value, masterData, "manufacturer")).toEqual([
      { id: "可口可乐", label: "可口可乐", count: 2 },
      { id: "百事", label: "百事", count: 1 },
    ]);
    expect(buildMasterDataFacets(value, masterData, "brand")).toEqual([
      { id: "Coca Cola", label: "Coca Cola", count: 2 },
      { id: "冲劲", label: "冲劲", count: 1 },
    ]);
    expect(buildMasterDataFacets(value, masterData, "category")).toEqual([
      { id: "饮料酒水", label: "饮料酒水", count: 3 },
    ]);
  });

  it("filters primary SKUs by a metadata facet", () => {
    const value = objects({
      "1": { ordered_skus: [{ sku_id: "A", sku_name: "产品A" }], point_ranges: [[0, 1]], observations: [] },
      "2": { ordered_skus: [{ sku_id: "B", sku_name: "产品B" }], point_ranges: [[1, 2]], observations: [] },
    });

    expect(filterGlobalIdsByMasterData(value, masterData, "manufacturer", "百事")).toEqual(["1"]);
  });
});
