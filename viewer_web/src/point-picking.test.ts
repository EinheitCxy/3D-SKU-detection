import { describe, expect, it } from "vitest";
import { applyVisibilityUpdates, buildPointRangeLookup, buildVisibilityDelta, firstVisiblePointGlobalId, globalIdForPointIndex, visiblePointRanges } from "./point-picking";
import type { ObjectIndex } from "./contracts";

function objectIndexWithRanges(rangesById: Record<string, readonly (readonly [number, number])[]>): ObjectIndex {
  return Object.fromEntries(Object.entries(rangesById).map(([globalId, point_ranges]) => [globalId, {
    ordered_skus: [], point_ranges, observations: [],
  }])) as ObjectIndex;
}

describe("point-only picking", () => {
  it("maps point ranges to global IDs and ignores empty ranges", () => {
    const lookup = buildPointRangeLookup(objectIndexWithRanges({ "1": [[0, 3], [6, 8]], "2": [[3, 6], [8, 8]] }));
    expect(globalIdForPointIndex(lookup, 0)).toBe("1");
    expect(globalIdForPointIndex(lookup, 4)).toBe("2");
    expect(globalIdForPointIndex(lookup, 8)).toBeNull();
  });

  it("picks the first visible point owner without a footprint path", () => {
    const lookup = buildPointRangeLookup(objectIndexWithRanges({ "1": [[0, 3]], "2": [[3, 6]] }));
    expect(firstVisiblePointGlobalId([0, 3], lookup, new Set(["2"]))).toBe("2");
    expect(firstVisiblePointGlobalId([0], lookup, new Set(["2"]))).toBeNull();
  });

  it("uses point ranges for visibility deltas", () => {
    const objects = objectIndexWithRanges({ "1": [[0, 3]], "2": [[3, 6]], "3": [[6, 8]] });
    expect(visiblePointRanges(objects, new Set(["2"]))).toEqual([{ start: 3, end: 6, globalId: "2" }]);
    expect(buildVisibilityDelta(objects, new Set(["1", "2"]), new Set(["2", "3"]))).toEqual({
      changedIds: ["1", "3"], ranges: [{ start: 0, end: 3, value: 0 }, { start: 6, end: 8, value: 1 }],
    });
    const bytes = Uint8Array.from([1, 1, 1, 1, 1, 1]);
    expect(applyVisibilityUpdates(bytes, [{ start: 1, end: 5, value: 0 }])).toEqual([[1, 5]]);
    expect([...bytes]).toEqual([1, 0, 0, 0, 0, 1]);
  });
});
