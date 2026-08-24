import { describe, expect, it } from "vitest";
import { focusedCameraPosition } from "./focus";

describe("focusedCameraPosition", () => {
  it("keeps the camera on the current observation side of the target", () => {
    const position = focusedCameraPosition([10, 0, 0], [2, 0, 0], [3, 0, 0], 5);
    expect(position).toEqual([8, 0, 0]);
    expect((position[0] - 3) * (10 - 2)).toBeGreaterThan(0);
  });

  it("uses a stable fallback when camera equals current target and preserves a 3D side", () => {
    expect(focusedCameraPosition([1, 2, 3], [1, 2, 3], [4, 5, 6], 2)).toEqual([6, 5, 6]);
    const position = focusedCameraPosition([3, 5, 7], [0, 1, 2], [10, 20, 30], 4);
    expect((position[0] - 10) * 3 + (position[1] - 20) * 4 + (position[2] - 30) * 5).toBeGreaterThan(0);
  });
});
