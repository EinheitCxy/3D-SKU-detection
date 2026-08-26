import type { ObjectIndex, OrderedSku } from "./contracts";

export interface SkuFacet {
  readonly skuId: string;
  readonly skuName: string;
  readonly count: number;
}

export function buildSkuFacets(objects: ObjectIndex): readonly SkuFacet[] {
  const counts = new Map<string, { skuName: string; count: number }>();
  for (const entry of Object.values(objects)) {
    const primary: OrderedSku | undefined = entry.ordered_skus[0];
    if (primary === undefined) continue;
    const current = counts.get(primary.sku_id);
    counts.set(primary.sku_id, { skuName: primary.sku_name, count: (current?.count ?? 0) + 1 });
  }
  return [...counts.entries()]
    .map(([skuId, value]) => ({ skuId, skuName: value.skuName, count: value.count }))
    .sort((left, right) => right.count - left.count || compareStrings(left.skuId, right.skuId));
}

export function filterGlobalIdsBySku(objects: ObjectIndex, skuId: string): readonly string[] {
  return Object.entries(objects)
    .filter(([, entry]) => entry.ordered_skus[0]?.sku_id === skuId)
    .map(([globalId]) => globalId)
    .sort(compareGlobalIds);
}

function compareGlobalIds(left: string, right: string): number {
  const leftValue = BigInt(left);
  const rightValue = BigInt(right);
  return leftValue === rightValue ? 0 : leftValue < rightValue ? -1 : 1;
}

function compareStrings(left: string, right: string): number {
  return left === right ? 0 : left < right ? -1 : 1;
}
