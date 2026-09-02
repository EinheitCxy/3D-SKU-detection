import { describe, expect, it } from "vitest";
import { DynamicDrawUsage, Matrix4, ShaderMaterial, Uint8BufferAttribute } from "three";
import { createPoints, isPointClickRelease, selectionRangesForGlobalIds } from "./scene";

describe("point-only scene", () => {
  it("accepts only a short primary point click", () => {
    const press = { pointerId: 7, clientX: 100, clientY: 200 };
    expect(isPointClickRelease(press, { pointerId: 7, clientX: 104, clientY: 203, button: 0, isPrimary: true })).toBe(true);
    expect(isPointClickRelease(press, { pointerId: 7, clientX: 107, clientY: 200, button: 0, isPrimary: true })).toBe(false);
  });

  it("keeps one dynamic point geometry and tints minimal point ranges", () => {
    const points = createPoints({ positions: new Float32Array([0, 0, 0, 1, 1, 1]), colors: new Uint8Array([1, 2, 3, 4, 5, 6]), normals: new Int8Array([0, 0, 127, 0, 0, 127]) } as never, new Matrix4());
    expect((points.geometry.getAttribute("aColor") as Uint8BufferAttribute).usage).toBe(DynamicDrawUsage);
    expect((points.geometry.getAttribute("aVisible") as Uint8BufferAttribute).count).toBe(2);
    expect(points.children).toHaveLength(0);
    expect(selectionRangesForGlobalIds({ "1": { point_ranges: [[0, 2]] }, "2": { point_ranges: [[2, 5]] } } as never, new Set(["1", "2"]))).toEqual([[0, 2], [2, 5]]);
  });

  it("keeps custom fog includes on standalone preprocessor lines", () => {
    const points = createPoints({ positions: new Float32Array([0, 0, 0]), colors: new Uint8Array([1, 2, 3]), normals: new Int8Array([0, 0, 127]) } as never, new Matrix4());
    const material = points.material as ShaderMaterial;
    for (const include of ["#include <fog_vertex>", "#include <fog_fragment>"]) {
      const lines = (include === "#include <fog_vertex>" ? material.vertexShader : material.fragmentShader).split("\n");
      expect(lines.filter((line) => line.trim() === include)).toHaveLength(1);
      expect(lines.some((line) => line.includes(include) && line.trim() !== include)).toBe(false);
    }
  });
});
