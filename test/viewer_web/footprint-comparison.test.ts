import { describe, it, expect } from "vitest";
import { Matrix4 } from "three";
import type { Manifest } from "../../modules/viewer_web/src/contracts";
import { AreaMeasurement } from "../../modules/viewer_web/src/area-measurement";
import { FootprintComparison, regionViewPoints, worldRegion, validateOutlineDocument, regionsOverlap } from "../../modules/viewer_web/src/footprint-comparison";
const manifest: Manifest = {
  schema_version: "3.0.0", dataset_name: "fixture", backend: "DA3", frame_count: 2,
  display_bounds: [0, 0, 0, 10, 10, 10],
  world_to_view: new Matrix4().makeRotationX(0.6).setPosition(10, 20, -30).transpose().elements,
  measurement: { coordinate_unit: "m", horizontal_plane: "viewer_xz", orientation_status: "fitted" },
};
const rectangle = (x = 0, z = 0): [number, number][] => [[x,z],[x+2,z],[x+2,z+3],[x,z+3]];
const draft = () => { const d = new AreaMeasurement(); d.start(); rectangle().forEach(p => d.addPoint(p)); d.close(); return d; };
const candidate = () => {
  const store = new FootprintComparison(manifest); store.saveDraft(draft().state, 7);
  return { ...store.state.manual, kind: "candidate" as const, status: "candidate", metric: "da3_manual_equivalent_stack_outline_candidate", value_m2: null, parameters: {}, diagnostics: {} };
};
describe("outline exchange", () => {
  it("roundtrips tilted translated row-major geometry exactly once with rectangle area parity", () => {
    const region = worldRegion("one", rectangle(), manifest, 7);
    expect(region.area_m2).toBe(6);
    regionViewPoints(region, manifest).forEach((p,i) => p.forEach((v,j) => expect(v).toBeCloseTo(rectangle()[i][j], 10)));
    expect(validateOutlineDocument(candidate(), manifest, "candidate").total_area_m2).toBeCloseTo(6);
  });
  it("separates draft, saved references, candidate and survives draft start/stop", () => {
    const store = new FootprintComparison(manifest), d = draft();
    expect(store.state.manual.total_area_m2).toBe(0);
    store.saveDraft(d.state, 5); d.start(); d.stop();
    expect(store.state.manual.total_area_m2).toBeCloseTo(6);
    d.start(); rectangle(3).forEach(p => d.addPoint(p)); d.close(); store.saveDraft(d.state, 5);
    expect(store.state.manual.regions).toHaveLength(2);
    expect(store.state.manual.total_area_m2).toBeCloseTo(12);
    store.importJson(JSON.stringify(candidate()), "candidate");
    expect(store.state.candidate?.total_area_m2).toBeCloseTo(6);
    store.clearManual(); expect(store.state.candidate?.regions).toHaveLength(1);
    store.importJson(store.exportManual(), "manual"); expect(store.state.manual.regions).toHaveLength(0);
  });
  it("rejects overlap including nesting, identical and coincident-edge rectangles but allows touch", () => {
    expect(regionsOverlap(rectangle(), rectangle(1))).toBe(true);
    expect(regionsOverlap(rectangle(), rectangle())).toBe(true);
    expect(regionsOverlap(rectangle(), [[.5,.5],[1,.5],[1,1],[.5,1]])).toBe(true);
    expect(regionsOverlap(rectangle(), rectangle(2))).toBe(false);
    expect(regionsOverlap(rectangle(), rectangle(2,3))).toBe(false);
    const store = new FootprintComparison(manifest); store.saveDraft(draft().state, 0);
    expect(() => store.saveDraft(draft().state, 0)).toThrow(/重叠/);
    expect(store.state.manual.regions).toHaveLength(1);
  });
  it.each([
    (d: any) => { d.scene.dataset_name = "wrong"; },
    (d: any) => { d.scene.world_to_view[0] = "1"; },
    (d: any) => { d.scene.world_to_view[0] *= 2; },
    (d: any) => { d.regions[0].area_m2 = 7; },
    (d: any) => { d.total_area_m2 = false; },
    (d: any) => { d.regions[0].vertices_world[0][0] = null; },
    (d: any) => { d.regions[0].vertices_world[0][0] = Infinity; },
    (d: any) => { d.regions[0].vertices_world[0][1] += 1; },
    (d: any) => { d.regions[0].vertices_world.push(d.regions[0].vertices_world[0]); },
    (d: any) => { d.status = "accepted"; },
    (d: any) => { d.value_m2 = 6; },
  ])("preserves state after invalid file import", mutate => {
    const store = new FootprintComparison(manifest); store.saveDraft(draft().state, 0); store.importJson(JSON.stringify(candidate()), "candidate");
    const before = store.state, invalid = candidate(); mutate(invalid);
    expect(() => store.importJson(JSON.stringify(invalid), "candidate")).toThrow(); expect(store.state).toEqual(before);
  });
  it("requires simple geometry, permits incomplete empty candidate and rejects complete empty", () => {
    const c = candidate(); c.regions = []; c.total_area_m2 = 0;
    expect(() => validateOutlineDocument(c, manifest, "candidate")).toThrow();
    expect(validateOutlineDocument({ ...c, complete: false, status: "needs_review" }, manifest, "candidate").regions).toHaveLength(0);
    const bad = candidate(); bad.regions[0] = worldRegion("cross", rectangle(), manifest, 0);
    const v = bad.regions[0].vertices_world; [v[1], v[2]] = [v[2], v[1]];
    expect(() => validateOutlineDocument(bad, manifest, "candidate")).toThrow();
    expect(() => validateOutlineDocument(candidate(), { ...manifest, measurement: undefined }, "candidate")).toThrow();
  });
});
