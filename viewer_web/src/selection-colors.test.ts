import { describe, expect, it } from "vitest";
import { applySelectionColors, mergePointRanges } from "./selection-colors";
import { Uint8BufferAttribute } from "three";
import { queueSelectionAttributeUpdates } from "./selection-colors";

describe("selection colors", () => {
  it("merges adjacent point ranges for one GPU update", () => {
    expect(mergePointRanges([[1, 3], [3, 5], [8, 9]])).toEqual([[1, 5], [8, 9]]);
  });

  it("accumulates merged BufferAttribute updates across A to B to C before render", () => {
    const attribute = new Uint8BufferAttribute(new Uint8Array(18), 3);
    expect(queueSelectionAttributeUpdates(attribute, [[1, 2]])).toBe(true);
    expect(queueSelectionAttributeUpdates(attribute, [[2, 3]])).toBe(true);
    expect(queueSelectionAttributeUpdates(attribute, [[3, 4]])).toBe(true);
    expect(attribute.updateRanges).toEqual([{ start: 3, count: 9 }]);
    expect(queueSelectionAttributeUpdates(attribute, [])).toBe(false);
  });

  it("does not request an upload for repeated or empty selections", () => {
    const original = new Uint8Array(9);
    const colors = original.slice();
    expect(applySelectionColors(colors, original, [[1, 2]], [[1, 2]])).toEqual([]);
    expect(applySelectionColors(colors, original, [], [])).toEqual([]);
  });

  it("leaves only C tinted while retaining merged A/B/C pending uploads", () => {
    const original = new Uint8Array(12).fill(7);
    const colors = original.slice();
    const attribute = new Uint8BufferAttribute(colors, 3);
    let previous: ReadonlyArray<readonly [number, number]> = [];
    for (const current of [[[0, 1]], [[1, 2]], [[2, 3]]] as const) {
      const changed = applySelectionColors(colors, original, previous, current);
      queueSelectionAttributeUpdates(attribute, changed);
      previous = current;
    }
    expect(Array.from(colors)).toEqual([7, 7, 7, 7, 7, 7, 255, 0, 255, 7, 7, 7]);
    expect(attribute.updateRanges).toEqual([{ start: 0, count: 9 }]);
    const version = attribute.version;
    const pending = attribute.updateRanges.slice();
    expect(queueSelectionAttributeUpdates(attribute, [])).toBe(false);
    expect(attribute.version).toBe(version);
    expect(attribute.updateRanges).toEqual(pending);
  });

  it("restores prior tint when moving to empty selection and uploads only that range", () => {
    const original = new Uint8Array([1, 1, 1, 2, 2, 2, 3, 3, 3]);
    const colors = original.slice();
    colors.set([255, 0, 255], 3);
    const attribute = new Uint8BufferAttribute(colors, 3);
    const changed = applySelectionColors(colors, original, [[1, 2]], []);
    expect(changed).toEqual([[1, 2]]);
    expect(Array.from(colors)).toEqual(Array.from(original));
    expect(queueSelectionAttributeUpdates(attribute, changed)).toBe(true);
    expect(attribute.updateRanges).toEqual([{ start: 3, count: 3 }]);
  });

  it("restores only prior ranges and tints only current ranges", () => {
    const original = new Uint8Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]);
    const colors = original.slice();
    colors.set([255, 0, 255], 3);
    const changed = applySelectionColors(colors, original, [[1, 2]], [[2, 4]]);
    expect(Array.from(colors)).toEqual([1, 1, 1, 2, 2, 2, 255, 0, 255, 255, 0, 255]);
    expect(changed).toEqual([[1, 4]]);
  });
});
