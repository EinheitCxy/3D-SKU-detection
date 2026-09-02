import { describe, expect, it } from "vitest";
import { navigationState, stepVisibleId } from "./navigation";

describe("filtered global-ID navigation", () => {
  const visible = ["11", "31", "41"];

  it("steps only within IDs visible after search", () => {
    expect(stepVisibleId(visible, "31", -1)).toBe("11");
    expect(stepVisibleId(visible, "31", 1)).toBe("41");
    expect(stepVisibleId(visible, "11", -1)).toBeNull();
    expect(stepVisibleId(visible, "41", 1)).toBeNull();
  });

  it("disables navigation for an ID excluded by the active filter", () => {
    expect(navigationState(visible, "2")).toEqual({ previousDisabled: true, nextDisabled: true });
  });

  it("does not jump into a filtered list from an excluded selection", () => {
    expect(stepVisibleId(visible, "2", 1)).toBeNull();
  });
});
