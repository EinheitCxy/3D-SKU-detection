import { describe, expect, it } from "vitest";
import { buildPointRangeLookup, globalIdForPointIndex, resolvePickGlobalId, visiblePointRanges } from "./point-picking";
import type { ObjectIndex, ObjectIndexEntry } from "./contracts";

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
});
