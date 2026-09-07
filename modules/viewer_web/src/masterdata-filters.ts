import type { ObjectIndex, SkuMasterDataIndex } from "./contracts";

export type MasterDataFacetKey = "manufacturer" | "brand" | "category";

export interface MasterDataFacet {
  readonly id: string;
  readonly label: string;
  readonly count: number;
}

export function buildMasterDataFacets(
  objects: ObjectIndex,
  masterData: SkuMasterDataIndex,
  key: MasterDataFacetKey,
): readonly MasterDataFacet[] {
  const counts = new Map<string, { label: string; count: number }>();
  for (const entry of Object.values(objects)) {
    const skuId = entry.ordered_skus[0]?.sku_id;
    if (skuId === undefined) continue;
    const metadata = masterData[skuId];
    const value = facetValue(metadata, key);
    if (value === null) continue;
    const current = counts.get(value.id);
    counts.set(value.id, { label: value.label, count: (current?.count ?? 0) + 1 });
  }
  return [...counts.entries()]
    .map(([id, value]) => ({ id, label: value.label, count: value.count }))
    .sort((left, right) => right.count - left.count || compareStrings(left.label, right.label));
}

export function filterGlobalIdsByMasterData(
  objects: ObjectIndex,
  masterData: SkuMasterDataIndex,
  key: MasterDataFacetKey,
  facetId: string,
): readonly string[] {
  return Object.entries(objects)
    .filter(([, entry]) => {
      const skuId = entry.ordered_skus[0]?.sku_id;
      const value = skuId === undefined ? null : facetValue(masterData[skuId], key);
      return value?.id === facetId;
    })
    .map(([globalId]) => globalId)
    .sort(compareGlobalIds);
}

function facetValue(
  metadata: SkuMasterDataIndex[string] | undefined,
  key: MasterDataFacetKey,
): { readonly id: string; readonly label: string } | null {
  if (metadata === undefined) return null;
  const value = metadata[key];
  return value === null ? null : { id: value, label: value };
}

function compareGlobalIds(left: string, right: string): number {
  const leftValue = BigInt(left);
  const rightValue = BigInt(right);
  return leftValue === rightValue ? 0 : leftValue < rightValue ? -1 : 1;
}

function compareStrings(left: string, right: string): number {
  return left === right ? 0 : left < right ? -1 : 1;
}
