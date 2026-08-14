import { describe, expect, it } from "vitest";
import { isFootprintClickRelease } from "./scene";

describe("isFootprintClickRelease", () => {
  it("accepts only a short primary-button release from the same pointer", () => {
    const press = { pointerId: 7, clientX: 100, clientY: 200 };
    expect(isFootprintClickRelease(press, { pointerId: 7, clientX: 104, clientY: 203, button: 0, isPrimary: true })).toBe(true);
    expect(isFootprintClickRelease(press, { pointerId: 7, clientX: 107, clientY: 200, button: 0, isPrimary: true })).toBe(false);
    expect(isFootprintClickRelease(press, { pointerId: 7, clientX: 100, clientY: 200, button: 2, isPrimary: true })).toBe(false);
    expect(isFootprintClickRelease(press, { pointerId: 8, clientX: 100, clientY: 200, button: 0, isPrimary: true })).toBe(false);
    expect(isFootprintClickRelease(null, { pointerId: 7, clientX: 100, clientY: 200, button: 0, isPrimary: true })).toBe(false);
  });
});
