import { describe, expect, it } from "vitest";
import { DynamicDrawUsage, Matrix4, OrthographicCamera, ShaderMaterial, Uint8BufferAttribute } from "three";
import { areaMeasurementAvailability, horizontalPointAtNdc, viewPresetDirection, createPoints, isPointClickRelease, selectionRangesForGlobalIds } from "../../modules/viewer_web/src/scene";
import { polygonArea } from "../../modules/viewer_web/src/area-measurement";
import type { Manifest } from "../../modules/viewer_web/src/contracts";

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

it("separates front, top, and diagonal preset angles", () => {
  const front = viewPresetDirection("fit");
  const top = viewPresetDirection("top");
  const iso = viewPresetDirection("isometric");
  expect(front.toArray()).toEqual([0, 0, 1]);
  expect(top.toArray()).toEqual([0, 1, 0]);
  expect(iso.y).toBeCloseTo(Math.SQRT1_2);
  expect(iso.x).toBeCloseTo(iso.z);
});

describe("horizontal measurement coordinates", () => {
  const manifest: Manifest = {
    schema_version: "3.0.0", dataset_name: "metric scene", backend: "DA3", frame_count: 1,
    display_bounds: [0, 0, 0, 2, 1, 3], world_to_view: new Matrix4().elements,
    measurement: { coordinate_unit: "m", horizontal_plane: "viewer_xz", orientation_status: "fitted" },
  };

  it("requires metric units and a fitted horizontal orientation without scaling", () => {
    expect(areaMeasurementAvailability(manifest)).toBeNull();
    expect(areaMeasurementAvailability({ ...manifest, measurement: undefined })).not.toBeNull();
    expect(areaMeasurementAvailability({ ...manifest, measurement: { ...manifest.measurement!, coordinate_unit: "unknown" } })).not.toBeNull();
    expect(areaMeasurementAvailability({ ...manifest, measurement: { ...manifest.measurement!, orientation_status: "not_found" } })).not.toBeNull();
    expect(areaMeasurementAvailability({ ...manifest, world_to_view: new Matrix4().makeScale(2, 2, 2).elements })).not.toBeNull();
    const transform = new Matrix4().makeRotationX(0.6).setPosition(10, 20, -30);
    expect(areaMeasurementAvailability({ ...manifest, world_to_view: transform.clone().transpose().elements })).toBeNull();
  });

  it("measures a 2 by 3 rectangle independently of the horizontal plane height", () => {
    const camera = new OrthographicCamera(-1, 1, 1.5, -1.5, 0.01, 100);
    camera.up.set(0, 0, -1);
    camera.position.set(10, 20, 30);
    camera.lookAt(10, 0, 30);
    camera.updateMatrixWorld();
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    for (const height of [-4, 0, 3]) {
      const points = corners.map(([x, y]) => horizontalPointAtNdc(camera, x, y, height)!);
      expect(polygonArea(points)).toBeCloseTo(6);
      expect(points[0][0]).toBeCloseTo(9);
      expect(points[0][1]).toBeCloseTo(31.5);
    }
    camera.zoom = 2;
    camera.updateProjectionMatrix();
    const zoomed = corners.map(([x, y]) => horizontalPointAtNdc(camera, x, y, 0)!);
    expect(polygonArea(zoomed)).toBeCloseTo(1.5);
  });
});
