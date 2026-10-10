import { describe, expect, it } from "vitest";
import { AreaMeasurement, polygonArea, type AreaPoint } from "../../modules/viewer_web/src/area-measurement";
const square: AreaPoint[] = [[0, 0], [2, 0], [2, 2], [0, 2]];

describe("polygonArea", () => {
  it("measures either winding, concavity and redundant collinear vertices", () => {
    expect(polygonArea(square)).toBe(4);
    expect(polygonArea([...square].reverse())).toBe(4);
    expect(polygonArea([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]])).toBe(5);
    expect(polygonArea([[0, 0], [1, 0], [2, 0], [2, 2], [0, 2]])).toBe(4);
  });
  it("is stable under large translation", () => {
    expect(polygonArea(square.map(([x, z]) => [x + 1e12, z - 1e12]))).toBe(4);
  });
  it.each([
    [[0, 0], [1, 0]],
    [[0, 0], [1, 0], [2, 0]],
    [[0, 0], [2, 2], [0, 2], [2, 0]],
    [[0, 0], [2, 0], [2, 2], [1, 0], [0, 2]],
    [[0, 0], [2, 0], [2, 2], [0, 0]],
    [[0, 0], [1e-9, 0], [2, 2], [0, 2]],
    [[0, 0], [2, 0], [Infinity, 2]],
    [[0, 0], [NaN, 0], [0, 2]],
    [[0, 0], [2, 0], [1, 0], [1, 2], [0, 2]],
  ].map((points) => ({ points })))("rejects invalid boundary $points", ({ points }) => {
    expect(() => polygonArea(points as AreaPoint[])).toThrow();
  });
});

describe("AreaMeasurement", () => {
  it("closes, stops retaining results, reopens then removes points", () => {
    const measurement = new AreaMeasurement();
    measurement.start();
    square.forEach((point) => measurement.addPoint(point));
    measurement.close();
    expect(measurement.state).toMatchObject({ active: true, closed: true, areaM2: 4, error: null });
    measurement.stop();
    expect(measurement.state).toMatchObject({ active: false, closed: true, areaM2: 4 });
    measurement.undo();
    expect(measurement.state).toMatchObject({ closed: false, areaM2: null, vertices: square });
    measurement.undo();
    expect(measurement.state.vertices).toHaveLength(3);
    measurement.start();
    expect(measurement.state.vertices).toHaveLength(0);
    measurement.addPoint([0, 0]);
    measurement.clear();
    expect(measurement.state).toMatchObject({ active: true, vertices: [], closed: false, areaM2: null });
  });
  it("keeps failed closure editable and rejects invalid additions", () => {
    const measurement = new AreaMeasurement();
    measurement.start();
    measurement.addPoint([0, 0]);
    measurement.addPoint([1e-9, 0]);
    expect(measurement.state.vertices).toHaveLength(1);
    expect(measurement.state.error).not.toBeNull();
    measurement.addPoint([NaN, 1]);
    expect(measurement.state.vertices).toHaveLength(1);
    measurement.addPoint([2, 2]);
    measurement.addPoint([0, 2]);
    measurement.addPoint([2, 0]);
    measurement.close();
    expect(measurement.state).toMatchObject({ closed: false, areaM2: null });
    expect(measurement.state.error).not.toBeNull();
    expect(measurement.state.vertices).toHaveLength(4);
    measurement.undo();
    measurement.close();
    expect(measurement.state).toMatchObject({ closed: true, areaM2: 2, error: null });
  });
  it("copies input points and state snapshots", () => {
    const measurement = new AreaMeasurement();
    measurement.start();
    const point: [number, number] = [1, 2];
    measurement.addPoint(point);
    point[0] = 99;
    const snapshot = measurement.state.vertices as [number, number][];
    snapshot[0][0] = 100;
    snapshot.push([3, 4]);
    expect(measurement.state.vertices).toEqual([[1, 2]]);
  });
});
