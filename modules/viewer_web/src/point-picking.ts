import type { ObjectIndex } from "./contracts";

export interface PointOwnerRange {
  readonly start: number;
  readonly end: number;
  readonly globalId: string;
}

export function buildPointRangeLookup(objects: ObjectIndex): readonly PointOwnerRange[] {
  const ranges: PointOwnerRange[] = [];
  for (const [globalId, entry] of Object.entries(objects)) {
    for (const instance of entry.instances) {
      const [start, end] = instance.point_index_range;
      if (end > start) ranges.push({ start, end, globalId });
    }
  }
  return ranges.sort((left, right) => left.start - right.start || left.end - right.end || compareIds(left.globalId, right.globalId));
}

export function globalIdForPointIndex(lookup: readonly PointOwnerRange[], pointIndex: number): string | null {
  if (!Number.isSafeInteger(pointIndex) || pointIndex < 0) return null;
  let low = 0;
  let high = lookup.length - 1;
  while (low <= high) {
    const middle = low + Math.floor((high - low) / 2);
    const range = lookup[middle];
    if (pointIndex < range.start) high = middle - 1;
    else if (pointIndex >= range.end) low = middle + 1;
    else return range.globalId;
  }
  return null;
}

export function visiblePointRanges(objects: ObjectIndex, ids: ReadonlySet<string>): readonly PointOwnerRange[] {
  const ranges: PointOwnerRange[] = [];
  for (const [globalId, entry] of Object.entries(objects)) {
    if (!ids.has(globalId)) continue;
    for (const instance of entry.instances) {
      const [start, end] = instance.point_index_range;
      if (end > start) ranges.push({ start, end, globalId });
    }
  }
  return ranges.sort((left, right) => left.start - right.start || left.end - right.end || compareIds(left.globalId, right.globalId));
}

export function resolvePickGlobalId(
  footprintGlobalId: string | null,
  pointIndex: number | null,
  lookup: readonly PointOwnerRange[],
  visibleIds: ReadonlySet<string>,
): string | null {
  const candidate = footprintGlobalId ?? (pointIndex === null ? null : globalIdForPointIndex(lookup, pointIndex));
  return candidate !== null && visibleIds.has(candidate) ? candidate : null;
}

function compareIds(left: string, right: string): number {
  const leftValue = BigInt(left);
  const rightValue = BigInt(right);
  return leftValue === rightValue ? 0 : leftValue < rightValue ? -1 : 1;
}
