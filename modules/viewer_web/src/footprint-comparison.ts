import { Matrix4, ShapeUtils, Vector2, Vector3 } from "three";
import { polygonArea, type AreaPoint, type AreaMeasurementState } from "./area-measurement";
import type { Manifest } from "./contracts";

export type OutlineScene = Pick<Manifest, "dataset_name" | "backend" | "frame_count" | "world_to_view" | "display_bounds">;
export interface OutlineRegion { id: string; vertices_world: number[][]; area_m2: number; }
export interface OutlineDocument {
  format: "viewer-horizontal-outlines"; version: 1; coordinate_unit: "m";
  kind: "manual" | "candidate"; scene: OutlineScene; regions: OutlineRegion[];
  total_area_m2: number; status: string; complete: boolean; metric: string;
  value_m2?: null; parameters?: unknown; diagnostics?: unknown;
}
export interface FootprintState { manual: OutlineDocument; candidate: OutlineDocument | null; }
function fail(message: string): never { throw new Error(message); }
function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : fail("JSON 对象无效");
}
function numbers(value: unknown, length: number): number[] {
  if (!Array.isArray(value) || value.length !== length || !value.every(v => typeof v === "number" && Number.isFinite(v))) fail("坐标必须是有限 JSON numbers");
  return value;
}
function agrees(a: unknown, b: number): boolean { return typeof a === "number" && Number.isFinite(a) && Math.abs(a - b) <= 1e-6 + 1e-6 * Math.abs(b); }
export function validateOutlineScene(value: unknown): OutlineScene {
  const s = record(value);
  if (typeof s.dataset_name !== "string" || !s.dataset_name || typeof s.backend !== "string" || !s.backend || typeof s.frame_count !== "number" || !Number.isInteger(s.frame_count) || s.frame_count <= 0) fail("scene 描述无效");
  const m = numbers(s.world_to_view, 16), b = numbers(s.display_bounds, 6);
  const rows = [m.slice(0, 3), m.slice(4, 7), m.slice(8, 11)];
  if (!rows.every((r, i) => rows.every((o, j) => Math.abs(r.reduce((sum, v, k) => sum + v * o[k], 0) - Number(i === j)) <= 1e-6))
    || Math.abs(new Matrix4().fromArray(m).transpose().determinant() - 1) > 1e-6
    || m.slice(12).some((v, i) => Math.abs(v - Number(i === 3)) > 1e-6)) fail("scene world_to_view 必须为刚体变换");
  if (b.slice(0, 3).some((v, i) => v > b[i + 3])) fail("scene display_bounds 无效");
  return { dataset_name: s.dataset_name, backend: s.backend, frame_count: s.frame_count, world_to_view: [...m], display_bounds: [b[0], b[1], b[2], b[3], b[4], b[5]] };
}
export function footprintAvailability(manifest: Manifest): string | null {
  if (manifest.measurement?.coordinate_unit !== "m") return "此数据包缺少米制尺度信息，请重新导出支持面积测量的数据包。";
  if (manifest.measurement.orientation_status !== "fitted") return "此数据包尚未确定水平方向，无法测量水平占地面积。";
  try { validateOutlineScene(manifest); return null; } catch { return "场景变换含缩放或畸变，无法直接按平方米测量。"; }
}
export function regionViewPoints(region: OutlineRegion, scene: OutlineScene): AreaPoint[] {
  const matrix = new Matrix4().fromArray(scene.world_to_view).transpose();
  const view = region.vertices_world.map(p => new Vector3(...p as [number, number, number]).applyMatrix4(matrix));
  if (view.some(p => ![p.x, p.y, p.z].every(Number.isFinite)) || Math.max(...view.map(p => p.y)) - Math.min(...view.map(p => p.y)) > 1e-6) fail("区域必须平行 Viewer XZ 平面");
  return view.map(p => [p.x, p.z]);
}
export function worldRegion(id: string, points: readonly AreaPoint[], scene: OutlineScene, height: number): OutlineRegion {
  const area = polygonArea(points), inverse = new Matrix4().fromArray(scene.world_to_view).transpose().invert();
  return { id, vertices_world: points.map(([x, z]) => new Vector3(x, height, z).applyMatrix4(inverse).toArray()), area_m2: area };
}
const cross = (a: AreaPoint, b: AreaPoint, p: AreaPoint) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
function bounds(p: readonly AreaPoint[]) { return [Math.min(...p.map(v => v[0])), Math.min(...p.map(v => v[1])), Math.max(...p.map(v => v[0])), Math.max(...p.map(v => v[1]))]; }
function boxesOverlap(a: number[], b: number[]) { return a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]; }
/** Triangle clipping catches coincident edges and containment as well as crossings. */
export function regionsOverlap(a: readonly AreaPoint[], b: readonly AreaPoint[]): boolean {
  if (!boxesOverlap(bounds(a), bounds(b))) return false;
  const triangles = (p: readonly AreaPoint[]) => ShapeUtils.triangulateShape(p.map(v => new Vector2(...v)), []).map(t => t.map(i => p[i]));
  for (const ta of triangles(a)) for (const tb of triangles(b)) {
    if (!boxesOverlap(bounds(ta), bounds(tb))) continue;
    let clipped = [...ta];
    const sign = Math.sign(cross(tb[0], tb[1], tb[2]));
    for (let i = 0; i < 3 && clipped.length; i++) {
      const edgeA = tb[i], edgeB = tb[(i + 1) % 3], input = clipped;
      clipped = [];
      for (let j = 0; j < input.length; j++) {
        const p = input[j], q = input[(j + 1) % input.length];
        const cp = sign * cross(edgeA, edgeB, p), cq = sign * cross(edgeA, edgeB, q);
        if (cp >= 0) clipped.push(p);
        if ((cp < 0 && cq > 0) || (cp > 0 && cq < 0)) {
          const t = cp / (cp - cq); clipped.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
        }
      }
    }
    if (clipped.length >= 3) {
      const origin = clipped[0];
      const twice = clipped.reduce((sum, p, i) => sum + cross(origin, p, clipped[(i + 1) % clipped.length]), 0);
      if (Math.abs(twice) / 2 > 1e-10) return true;
    }
  }
  return false;
}
export function validateOutlineDocument(value: unknown, manifest: Manifest, kind: "manual" | "candidate"): OutlineDocument {
  const unavailable = footprintAvailability(manifest); if (unavailable) fail(unavailable);
  const d = record(value);
  if (d.format !== "viewer-horizontal-outlines" || d.version !== 1 || d.coordinate_unit !== "m" || d.kind !== kind) fail("outline format/version/unit/kind 无效");
  const scene = validateOutlineScene(d.scene), target = validateOutlineScene(manifest);
  for (const key of ["dataset_name", "backend", "frame_count"] as const) if (scene[key] !== target[key]) fail(`scene mismatch: ${key}`);
  for (const key of ["world_to_view", "display_bounds"] as const) if (scene[key].some((v, i) => Math.abs(v - target[key][i]) > 1e-6)) fail(`scene mismatch: ${key}`);
  if (!Array.isArray(d.regions)) fail("regions 必须为列表");
  const ids = new Set<string>(), polygons: AreaPoint[][] = [];
  const regions = d.regions.map(value => {
    const r = record(value);
    if (typeof r.id !== "string" || !r.id || ids.has(r.id)) fail("区域 ID 必须唯一且非空"); ids.add(r.id);
    if (!Array.isArray(r.vertices_world) || r.vertices_world.length < 3) fail("区域至少需要三个顶点");
    const region = { id: r.id, vertices_world: r.vertices_world.map(p => numbers(p, 3)), area_m2: 0 };
    const polygon = regionViewPoints(region, scene), area = polygonArea(polygon);
    if (typeof r.area_m2 !== "number" || !Number.isFinite(r.area_m2) || r.area_m2 <= 0 || !agrees(r.area_m2, area)) fail("区域 area_m2 与几何不一致");
    if (polygons.some(other => regionsOverlap(polygon, other))) fail("区域存在正面积重叠");
    polygons.push(polygon); return { ...region, area_m2: area };
  });
  const total = regions.reduce((sum, r) => sum + r.area_m2, 0);
  if (!agrees(d.total_area_m2, total)) fail("total_area_m2 与几何不一致");
  if (kind === "manual") {
    if (d.status !== "reference" || d.complete !== true || d.metric !== "manual_horizontal_outline") fail("人工 reference 状态无效");
  } else if (typeof d.complete !== "boolean" || d.status !== (d.complete ? "candidate" : "needs_review") || d.value_m2 !== null || d.metric !== "da3_manual_equivalent_stack_outline_candidate" || (d.complete && !regions.length)
    || !d.parameters || !d.diagnostics || Array.isArray(d.parameters) || Array.isArray(d.diagnostics) || typeof d.parameters !== "object" || typeof d.diagnostics !== "object") fail("candidate 状态/参数/诊断无效");
  return { ...d, scene, regions, total_area_m2: total } as OutlineDocument;
}
export class FootprintComparison {
  private manual: OutlineDocument;
  private candidate: OutlineDocument | null = null;
  constructor(private manifest: Manifest) {
    this.manual = { format: "viewer-horizontal-outlines", version: 1, coordinate_unit: "m", kind: "manual", scene: { dataset_name: manifest.dataset_name, backend: manifest.backend, frame_count: manifest.frame_count, world_to_view: [...manifest.world_to_view], display_bounds: [...manifest.display_bounds] }, regions: [], total_area_m2: 0, status: "reference", complete: true, metric: "manual_horizontal_outline" };
  }
  get state(): FootprintState { return structuredClone({ manual: this.manual, candidate: this.candidate }); }
  importJson(text: string, kind: "manual" | "candidate"): void {
    const next = validateOutlineDocument(JSON.parse(text), this.manifest, kind);
    if (kind === "manual") this.manual = next; else this.candidate = next;
  }
  saveDraft(draft: AreaMeasurementState, height: number): void {
    if (!draft.closed) fail("请先闭合当前货堆区域");
    let index = this.manual.regions.length + 1; while (this.manual.regions.some(r => r.id === `manual-${index}`)) index++;
    const region = worldRegion(`manual-${index}`, draft.vertices, this.manual.scene, height);
    const next = { ...this.manual, regions: [...this.manual.regions, region], total_area_m2: this.manual.total_area_m2 + region.area_m2 };
    this.manual = validateOutlineDocument(next, this.manifest, "manual");
  }
  clearManual(): void { this.manual = { ...this.manual, regions: [], total_area_m2: 0 }; }
  exportManual(): string { const unavailable = footprintAvailability(this.manifest); if (unavailable) fail(unavailable); return JSON.stringify(this.manual, null, 2); }
}
