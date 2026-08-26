import type { ObjectIndex } from "./contracts";

export interface PointOwnerRange { readonly start: number; readonly end: number; readonly globalId: string; }
export interface VisibilityRangeUpdate { readonly start: number; readonly end: number; readonly value: 0 | 1; }
export interface VisibilityDelta { readonly changedIds: readonly string[]; readonly ranges: readonly VisibilityRangeUpdate[]; }

export function buildPointRangeLookup(objects: ObjectIndex): readonly PointOwnerRange[] {
  return rangesForIds(objects, new Set(Object.keys(objects)));
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
  return rangesForIds(objects, ids);
}

export function firstVisiblePointGlobalId(hitPointIndices: readonly number[], lookup: readonly PointOwnerRange[], visibleIds: ReadonlySet<string>): string | null {
  for (const pointIndex of hitPointIndices) {
    const globalId = globalIdForPointIndex(lookup, pointIndex);
    if (globalId !== null && visibleIds.has(globalId)) return globalId;
  }
  return null;
}

export function buildVisibilityDelta(objects: ObjectIndex, previousIds: ReadonlySet<string>, nextIds: ReadonlySet<string>): VisibilityDelta {
  const changedIds = [...new Set([...previousIds, ...nextIds])]
    .filter((globalId) => previousIds.has(globalId) !== nextIds.has(globalId))
    .sort(compareIds);
  const ranges = changedIds.flatMap((globalId) => (objects[globalId]?.point_ranges ?? [])
    .filter(([start, end]) => end > start)
    .map(([start, end]) => ({ start, end, value: nextIds.has(globalId) ? 1 as const : 0 as const })))
    .sort((left, right) => left.start - right.start || left.end - right.end || left.value - right.value);
  return { changedIds, ranges: mergeVisibilityRanges(ranges) };
}

export function applyVisibilityUpdates(visibility: Uint8Array, updates: readonly VisibilityRangeUpdate[]): readonly PointRange[] {
  for (const { start, end, value } of updates) visibility.fill(value, start, end);
  return mergePointRanges(updates.map(({ start, end }) => [start, end] as PointRange));
}

function rangesForIds(objects: ObjectIndex, ids: ReadonlySet<string>): PointOwnerRange[] {
  return Object.entries(objects)
    .flatMap(([globalId, entry]) => ids.has(globalId)
      ? entry.point_ranges.filter(([start, end]) => end > start).map(([start, end]) => ({ start, end, globalId }))
      : [])
    .sort((left, right) => left.start - right.start || left.end - right.end || compareIds(left.globalId, right.globalId));
}

function mergeVisibilityRanges(ranges: readonly VisibilityRangeUpdate[]): VisibilityRangeUpdate[] {
  const merged: VisibilityRangeUpdate[] = [];
  for (const range of ranges) {
    const previous = merged[merged.length - 1];
    if (previous !== undefined && previous.value === range.value && range.start <= previous.end) {
      merged[merged.length - 1] = { start: previous.start, end: Math.max(previous.end, range.end), value: previous.value };
    } else merged.push({ ...range });
  }
  return merged;
}

function mergePointRanges(ranges: readonly PointRange[]): PointRange[] {
  const merged: PointRange[] = [];
  for (const range of ranges) {
    const previous = merged[merged.length - 1];
    if (previous !== undefined && range[0] <= previous[1]) merged[merged.length - 1] = [previous[0], Math.max(previous[1], range[1])];
    else merged.push([range[0], range[1]]);
  }
  return merged;
}

type PointRange = readonly [number, number];

function compareIds(left: string, right: string): number {
  const leftValue = BigInt(left);
  const rightValue = BigInt(right);
  return leftValue === rightValue ? 0 : leftValue < rightValue ? -1 : 1;
}
