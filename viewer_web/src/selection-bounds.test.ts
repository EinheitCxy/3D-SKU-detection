import { describe, expect, it } from "vitest";
import { Box3, Vector3 } from "three";
import { cachedSelectionBox } from "./selection-bounds";

describe("cachedSelectionBox", () => {
  it("computes once per ID and returns clones safe for transform mutation", () => {
    const cache = new Map<string, Box3 | null>();
    let calls = 0;
    const compute = () => { calls += 1; return new Box3(new Vector3(1, 2, 3), new Vector3(4, 5, 6)); };
    const first = cachedSelectionBox(cache, "11", compute)!;
    first.translate(new Vector3(10, 0, 0));
    const second = cachedSelectionBox(cache, "11", compute)!;
    expect(calls).toBe(1);
    expect(second.min.toArray()).toEqual([1, 2, 3]);
  });
});
