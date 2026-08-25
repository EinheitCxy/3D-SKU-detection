import { describe, expect, it } from "vitest";
import { applyVisibilityUpdates, buildPointRangeLookup, buildVisibilityDelta, firstVisibleFootprintGlobalId, firstVisiblePointGlobalId, globalIdForPointIndex, resolvePickGlobalId, syncVisibleTargets, visiblePointRanges } from "./point-picking";
import type { ObjectIndex } from "./contracts";

function objectIndexWithRanges(rangesById: Record<string, readonly (readonly [number, number])[]>): ObjectIndex {
  return Object.fromEntries(Object.entries(rangesById).map(([globalId, ranges]) => [globalId, {
    images: ranges.map((_range, index) => index),
    objects: ranges.map((_range, index) => index),
    active_count: ranges.length,
    removed_count: 0,
    total_count: ranges.length,
    instances: ranges.map((range, index) => ({
      image_id: index,
      object_id: index,
      bbox: [0, 0, 1, 1] as const,
      removed: false,
      point_index_range: range,
      thumbnail: `thumbs/${globalId}_${index}.jpg`,
      classification: { schema_version: "1.0.0", source: "personalcare", project_id: 51, status: "unavailable", reason: "invalid_bbox" },
    })),
    classification: { status: "unavailable", primary_sku_id: null, candidates: [], metadata: { status: "master_data_pending", manufacturer: null, brand: null, category: null, object_kind: null } },
  }])) as ObjectIndex;
}

describe("point picking", () => {
  it("maps point indices to global IDs and ignores empty ranges", () => {
    const lookup = buildPointRangeLookup(objectIndexWithRanges({ "1": [[0, 3], [6, 8]], "2": [[3, 6], [8, 8]] }));
    expect(globalIdForPointIndex(lookup, 0)).toBe("1");
    expect(globalIdForPointIndex(lookup, 4)).toBe("2");
    expect(globalIdForPointIndex(lookup, 7)).toBe("1");
    expect(globalIdForPointIndex(lookup, 8)).toBeNull();
  });

  it("returns only ranges owned by visible IDs", () => {
    const objects = objectIndexWithRanges({ "1": [[0, 3], [6, 8]], "2": [[3, 6]] });
    expect(visiblePointRanges(objects, new Set(["2"]))).toEqual([{ start: 3, end: 6, globalId: "2" }]);
  });

  it("uses footprint first, then points, and rejects hidden IDs", () => {
    const lookup = buildPointRangeLookup(objectIndexWithRanges({ "1": [[0, 3]], "2": [[3, 6]] }));
    expect(resolvePickGlobalId("2", 1, lookup, new Set(["1", "2"]))).toBe("2");
    expect(resolvePickGlobalId(null, 1, lookup, new Set(["1", "2"]))).toBe("1");
    expect(resolvePickGlobalId(null, 1, lookup, new Set(["2"]))).toBeNull();
  });

  it("falls through a hidden footprint hit to a visible point hit", () => {
    const lookup = buildPointRangeLookup(objectIndexWithRanges({ "1": [[0, 3]], "2": [[3, 6]] }));
    expect(resolvePickGlobalId("2", 1, lookup, new Set(["1"]))).toBe("1");
  });

  it("chooses the first visible footprint after hidden nearer hits", () => {
    expect(firstVisibleFootprintGlobalId(["2", "1"], new Set(["1"]))).toBe("1");
    expect(firstVisibleFootprintGlobalId(["2", null, undefined], new Set(["1"]))).toBeNull();
  });

  it("falls through hidden nearer point hits to the first visible owner", () => {
    const lookup = buildPointRangeLookup(objectIndexWithRanges({ "1": [[0, 3]], "2": [[3, 6]] }));
    expect(firstVisiblePointGlobalId([0, 3], lookup, new Set(["2"]))).toBe("2");
    expect(firstVisiblePointGlobalId([0], lookup, new Set(["2"]))).toBeNull();
  });
});

describe("visibility deltas", () => {
  it("changes only symmetric-difference IDs and merges adjacent ranges", () => {
    const objects = objectIndexWithRanges({ "1": [[0, 3]], "2": [[3, 6]], "3": [[6, 8]] });
    const delta = buildVisibilityDelta(objects, new Set(["1", "2"]), new Set(["2", "3"]));
    expect(delta.changedIds).toEqual(["1", "3"]);
    expect(delta.ranges).toEqual([
      { start: 0, end: 3, value: 0 },
      { start: 6, end: 8, value: 1 },
    ]);
  });

  it("mutates only requested bytes and returns GPU dirty ranges", () => {
    const bytes = Uint8Array.from([1, 1, 1, 1, 1, 1]);
    const dirty = applyVisibilityUpdates(bytes, [{ start: 1, end: 5, value: 0 }]);
    expect([...bytes]).toEqual([1, 0, 0, 0, 0, 1]);
    expect(dirty).toEqual([[1, 5]]);
  });

  it("syncs footprint targets only for changed IDs", () => {
    const targets = new Map([
      ["1", { visible: true }],
      ["2", { visible: false }],
    ]);
    syncVisibleTargets(targets, ["1"], new Set());
    expect(targets.get("1")?.visible).toBe(false);
    expect(targets.get("2")?.visible).toBe(false);
  });
});
