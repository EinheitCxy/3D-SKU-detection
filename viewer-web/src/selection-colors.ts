export type PointRange = readonly [number, number];
const TINT: readonly [number, number, number] = [255, 0, 255];

export function mergePointRanges(ranges: readonly PointRange[]): PointRange[] {
  const sorted = ranges.filter(([start, end]) => end > start).slice().sort((a, b) => a[0] - b[0]);
  const merged: PointRange[] = [];
  for (const range of sorted) {
    const previous = merged[merged.length - 1];
    if (previous !== undefined && range[0] <= previous[1]) merged[merged.length - 1] = [previous[0], Math.max(previous[1], range[1])];
    else merged.push([range[0], range[1]]);
  }
  return merged;
}

export function applySelectionColors(colors: Uint8Array, original: Uint8Array, previous: readonly PointRange[], current: readonly PointRange[]): PointRange[] {
  const prior = mergePointRanges(previous);
  const next = mergePointRanges(current);
  if (sameRanges(prior, next)) return [];
  for (const [start, end] of prior) colors.set(original.subarray(start * 3, end * 3), start * 3);
  for (const [start, end] of next) for (let index = start * 3; index < end * 3; index += 3) colors.set(TINT, index);
  return mergePointRanges([...prior, ...next]);
}

export function queueSelectionAttributeUpdates(attribute: BufferAttribute, changed: readonly PointRange[]): boolean {
  const additions = mergePointRanges(changed);
  if (additions.length === 0) return false;
  const pending = attribute.updateRanges.map(({ start, count }) => [start / 3, (start + count) / 3] as PointRange);
  attribute.clearUpdateRanges();
  for (const [start, end] of mergePointRanges([...pending, ...additions])) attribute.addUpdateRange(start * 3, (end - start) * 3);
  attribute.needsUpdate = true;
  return true;
}

function sameRanges(left: readonly PointRange[], right: readonly PointRange[]): boolean {
  return left.length === right.length && left.every(([start, end], index) => start === right[index][0] && end === right[index][1]);
}
import type { BufferAttribute } from "three";
