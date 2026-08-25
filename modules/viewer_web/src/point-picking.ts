import type { ObjectIndex } from "./contracts";

export interface PointOwnerRange {
  readonly start: number;
  readonly end: number;
  readonly globalId: string;
}

export interface VisibilityRangeUpdate {
  readonly start: number;
  readonly end: number;
  readonly value: 0 | 1;
}

export interface VisibilityDelta {
  readonly changedIds: readonly string[];
  readonly ranges: readonly VisibilityRangeUpdate[];
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
  const candidate = footprintGlobalId !== null && visibleIds.has(footprintGlobalId)
    ? footprintGlobalId
    : pointIndex === null ? null : globalIdForPointIndex(lookup, pointIndex);
  return candidate !== null && visibleIds.has(candidate) ? candidate : null;
}

export function buildVisibilityDelta(
  objects: ObjectIndex,
  previousIds: ReadonlySet<string>,
  nextIds: ReadonlySet<string>,
): VisibilityDelta {
  const allIds = new Set([...previousIds, ...nextIds]);
  const changedIds = [...allIds]
    .filter((globalId) => previousIds.has(globalId) !== nextIds.has(globalId))
    .sort(compareIds);
  const ranges: VisibilityRangeUpdate[] = [];
  for (const globalId of changedIds) {
    const entry = objects[globalId];
    if (entry === undefined) continue;
    const value: 0 | 1 = nextIds.has(globalId) ? 1 : 0;
    for (const instance of entry.instances) {
      const [start, end] = instance.point_index_range;
      if (end > start) ranges.push({ start, end, value });
    }
  }
  ranges.sort((left, right) => left.start - right.start || left.end - right.end || left.value - right.value);
  return { changedIds, ranges: mergeVisibilityRanges(ranges) };
}

export function applyVisibilityUpdates(
  visibility: Uint8Array,
  updates: readonly VisibilityRangeUpdate[],
): readonly PointRange[] {
  for (const { start, end, value } of updates) visibility.fill(value, start, end);
  return mergePointRanges(updates.map(({ start, end }) => [start, end] as PointRange));
}

export function syncVisibleTargets<T extends { visible: boolean }>(
  targets: ReadonlyMap<string, T>,
  changedIds: readonly string[],
  visibleIds: ReadonlySet<string>,
): void {
  for (const globalId of changedIds) {
    const target = targets.get(globalId);
    if (target !== undefined) target.visible = visibleIds.has(globalId);
  }
}

function mergeVisibilityRanges(ranges: readonly VisibilityRangeUpdate[]): VisibilityRangeUpdate[] {
  const merged: VisibilityRangeUpdate[] = [];
  for (const range of ranges) {
    const previous = merged[merged.length - 1];
    if (previous !== undefined && previous.value === range.value && range.start <= previous.end) {
      merged[merged.length - 1] = { start: previous.start, end: Math.max(previous.end, range.end), value: previous.value };
    } else {
      merged.push({ ...range });
    }
  }
  return merged;
}

function mergePointRanges(ranges: readonly PointRange[]): PointRange[] {
  const merged: PointRange[] = [];
  for (const range of ranges) {
    const previous = merged[merged.length - 1];
    if (previous !== undefined && range[0] <= previous[1]) {
      merged[merged.length - 1] = [previous[0], Math.max(previous[1], range[1])];
    } else {
      merged.push([range[0], range[1]]);
    }
  }
  return merged;
}

type PointRange = readonly [number, number];

function compareIds(left: string, right: string): number {
  const leftValue = BigInt(left);
  const rightValue = BigInt(right);
  return leftValue === rightValue ? 0 : leftValue < rightValue ? -1 : 1;
}
