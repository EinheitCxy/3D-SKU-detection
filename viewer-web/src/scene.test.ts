import { describe, expect, it } from "vitest";
import { DynamicDrawUsage, Matrix4, Uint8BufferAttribute } from "three";
import { createPoints, isFootprintClickRelease } from "./scene";

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

it("marks the mutable selection color attribute for dynamic GPU draws", () => {
  const points = createPoints({ positions: new Float32Array([0, 0, 0]), colors: new Uint8Array([1, 2, 3]), normals: new Int8Array([0, 0, 127]) } as never, new Matrix4());
  expect((points.geometry.getAttribute("aColor") as Uint8BufferAttribute).usage).toBe(DynamicDrawUsage);
});
