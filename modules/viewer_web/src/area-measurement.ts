/** Horizontal coordinates in viewer XZ, expressed in meters. */
export type AreaPoint = readonly [number, number];

export type AreaMeasurementState = {
  readonly active: boolean;
  readonly vertices: readonly AreaPoint[];
  readonly closed: boolean;
  readonly areaM2: number | null;
  readonly error: string | null;
};

const DISTANCE_EPSILON_M = 1e-8;

function distance(a: AreaPoint, b: AreaPoint): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function cross(a: AreaPoint, b: AreaPoint, c: AreaPoint): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function orientation(a: AreaPoint, b: AreaPoint, c: AreaPoint): number {
  const value = cross(a, b, c);
  const tolerance = DISTANCE_EPSILON_M * distance(a, b);
  return Math.abs(value) <= tolerance ? 0 : Math.sign(value);
}

function onSegment(a: AreaPoint, b: AreaPoint, p: AreaPoint): boolean {
  return orientation(a, b, p) === 0
    && p[0] >= Math.min(a[0], b[0]) - DISTANCE_EPSILON_M
    && p[0] <= Math.max(a[0], b[0]) + DISTANCE_EPSILON_M
    && p[1] >= Math.min(a[1], b[1]) - DISTANCE_EPSILON_M
    && p[1] <= Math.max(a[1], b[1]) + DISTANCE_EPSILON_M;
}

function segmentsTouch(a: AreaPoint, b: AreaPoint, c: AreaPoint, d: AreaPoint): boolean {
  const abc = orientation(a, b, c);
  const abd = orientation(a, b, d);
  const cda = orientation(c, d, a);
  const cdb = orientation(c, d, b);
  return (abc * abd < 0 && cda * cdb < 0)
    || onSegment(a, b, c) || onSegment(a, b, d)
    || onSegment(c, d, a) || onSegment(c, d, b);
}

function finitePoint(point: AreaPoint): boolean {
  return point.length === 2 && point.every(Number.isFinite);
}

/** Validate a simple polygon and return its unsigned horizontal area in m². */
export function polygonArea(points: readonly AreaPoint[]): number {
  if (points.length < 3) throw new Error("请至少标记三个不同的顶点。");
  if (!points.every(finitePoint)) throw new Error("顶点坐标无效，请重新标记。");
  // Translate before products to avoid cancellation far from the scene origin.
  const origin = points[0];
  const local: AreaPoint[] = points.map(([x, z]) => [x - origin[0], z - origin[1]]);
  if (!local.every(finitePoint)) throw new Error("坐标范围过大，无法计算面积。");
  for (let i = 0; i < local.length; i++) {
    for (let j = i + 1; j < local.length; j++) {
      if (distance(local[i], local[j]) <= DISTANCE_EPSILON_M) {
        throw new Error("顶点重复或距离过近，请撤销后重新标记。");
      }
    }
  }
  let twiceArea = 0;
  let perimeter = 0;
  for (let i = 0; i < local.length; i++) {
    const a = local[i];
    const b = local[(i + 1) % local.length];
    const c = local[(i + 2) % local.length];
    twiceArea += a[0] * b[1] - b[0] * a[1];
    perimeter += distance(a, b);
    if (orientation(a, b, c) === 0 && (onSegment(a, b, c) || onSegment(b, c, a))) {
      throw new Error("相邻边发生重叠，请撤销后重新标记。");
    }
    for (let j = i + 1; j < local.length; j++) {
      if (j === i + 1 || (i === 0 && j === local.length - 1)) continue;
      if (segmentsTouch(a, b, local[j], local[(j + 1) % local.length])) {
        throw new Error("边界自交或相互接触，请撤销后重新标记。");
      }
    }
  }
  if (!Number.isFinite(twiceArea) || !Number.isFinite(perimeter)) {
    throw new Error("坐标范围过大，无法计算面积。");
  }
  if (Math.abs(twiceArea) <= DISTANCE_EPSILON_M * perimeter) {
    throw new Error("顶点共线或区域过窄，无法形成有效面积。");
  }
  return Math.abs(twiceArea) / 2;
}

export class AreaMeasurement {
  private active = false;
  private vertices: AreaPoint[] = [];
  private closed = false;
  private areaM2: number | null = null;
  private error: string | null = null;

  get state(): AreaMeasurementState {
    return {
      active: this.active,
      vertices: this.vertices.map(([x, z]) => [x, z] as AreaPoint),
      closed: this.closed,
      areaM2: this.areaM2,
      error: this.error,
    };
  }

  start(): void {
    this.active = true;
    this.clear();
  }

  stop(): void { this.active = false; }

  addPoint(point: AreaPoint): void {
    if (!this.active) return;
    if (this.closed) {
      this.error = "区域已闭合，请先撤销闭合或清空。";
      return;
    }
    if (!finitePoint(point)) {
      this.error = "顶点坐标无效，请重新标记。";
      return;
    }
    const last = this.vertices.at(-1);
    if (last && distance(last, point) <= DISTANCE_EPSILON_M) {
      this.error = "相邻顶点距离过近，请选择其他位置。";
      return;
    }
    this.vertices.push([point[0], point[1]]);
    this.error = null;
  }

  close(): void {
    if (!this.active || this.closed) return;
    try {
      this.areaM2 = polygonArea(this.vertices);
      this.closed = true;
      this.error = null;
    } catch (error) {
      this.areaM2 = null;
      this.error = error instanceof Error ? error.message : "无法计算面积，请检查边界。";
    }
  }

  undo(): void {
    if (this.closed) this.closed = false;
    else this.vertices.pop();
    this.areaM2 = null;
    this.error = null;
  }

  clear(): void {
    this.vertices = [];
    this.closed = false;
    this.areaM2 = null;
    this.error = null;
  }
}
