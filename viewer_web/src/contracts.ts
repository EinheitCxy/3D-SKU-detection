export type ReadonlyRecord = Readonly<Record<string, unknown>>;

export interface CurrentPointer {
  readonly run_id: string;
}

export interface Manifest {
  readonly schema_version: "3.0.0";
  readonly dataset_name: string;
  readonly backend: string;
  readonly frame_count: number;
  readonly display_bounds: readonly [number, number, number, number, number, number];
  readonly world_to_view: readonly number[];
}

export interface OrderedSku {
  readonly sku_id: string;
  readonly sku_name: string;
}

export type PointRange = readonly [number, number];

export interface ObjectObservation {
  readonly image_id: number;
  readonly object_id: number;
  readonly removed: boolean;
  readonly thumbnail: string;
}

export interface ObjectIndexEntry {
  readonly ordered_skus: readonly OrderedSku[];
  readonly point_ranges: readonly PointRange[];
  readonly observations: readonly ObjectObservation[];
}

export type ObjectIndex = Readonly<Record<string, ObjectIndexEntry>>;

const GLOBAL_ID = /^(0|[1-9][0-9]*)$/;

export function validateCurrent(value: unknown): CurrentPointer {
  const record = asRecord(value, "CURRENT");
  const runId = asNonEmptyString(record.run_id, "CURRENT run_id");
  return { run_id: runId };
}

export function validateManifest(value: unknown): Manifest {
  const record = asRecord(value, "manifest");
  if (record.schema_version !== "3.0.0") {
    throw contractError("manifest schema_version must be 3.0.0");
  }
  const datasetName = asNonEmptyString(record.dataset_name, "manifest dataset_name");
  const backend = asNonEmptyString(record.backend, "manifest backend");
  const frameCount = asNonNegativeInteger(record.frame_count, "manifest frame_count");
  const displayBounds = validateDisplayBounds(record.display_bounds);
  const worldToView = validateWorldToView(record.world_to_view);
  return {
    schema_version: "3.0.0",
    dataset_name: datasetName,
    backend,
    frame_count: frameCount,
    display_bounds: displayBounds,
    world_to_view: worldToView,
  };
}

export function validateObjectIndex(value: unknown, pointCount: number): ObjectIndex {
  if (!Number.isSafeInteger(pointCount) || pointCount < 0) {
    throw contractError("point_count must be a non-negative safe integer");
  }
  const record = asRecord(value, "objects");
  const result: Record<string, ObjectIndexEntry> = {};
  const nonEmptyRanges: PointRange[] = [];
  for (const [globalId, rawEntry] of Object.entries(record)) {
    if (!GLOBAL_ID.test(globalId)) {
      throw contractError(`objects global ID key is invalid: ${globalId}`);
    }
    const entry = asRecord(rawEntry, `objects[${globalId}]`);
    const orderedSkus = validateOrderedSkus(entry.ordered_skus, `objects[${globalId}].ordered_skus`);
    const pointRanges = validatePointRanges(entry.point_ranges, pointCount, `objects[${globalId}].point_ranges`);
    const observations = validateObservations(entry.observations, `objects[${globalId}].observations`);
    for (const pointRange of pointRanges) {
      if (pointRange[1] > pointRange[0]) nonEmptyRanges.push(pointRange);
    }
    result[globalId] = { ordered_skus: orderedSkus, point_ranges: pointRanges, observations };
  }
  assertNonOverlappingRanges(nonEmptyRanges);
  return result;
}

function validateDisplayBounds(value: unknown): readonly [number, number, number, number, number, number] {
  const bounds = asArray(value, "manifest display_bounds");
  if (bounds.length !== 6 || bounds.some((item) => !isFiniteNumber(item))) {
    throw contractError("manifest display_bounds must contain six finite numbers");
  }
  const numericBounds = bounds as number[];
  return [
    numericBounds[0], numericBounds[1], numericBounds[2],
    numericBounds[3], numericBounds[4], numericBounds[5],
  ];
}

function validateWorldToView(value: unknown): readonly number[] {
  const matrix = asArray(value, "manifest world_to_view");
  if (matrix.length !== 16 || matrix.some((item) => !isFiniteNumber(item))) {
    throw contractError("manifest world_to_view must contain sixteen finite numbers");
  }
  return matrix as number[];
}

function validateOrderedSkus(value: unknown, label: string): readonly OrderedSku[] {
  return asArray(value, label).map((rawSku, index) => {
    const sku = asRecord(rawSku, `${label}[${index}]`);
    return {
      sku_id: asNonEmptyString(sku.sku_id, `${label}[${index}].sku_id`),
      sku_name: asNonEmptyString(sku.sku_name, `${label}[${index}].sku_name`),
    };
  });
}

function validatePointRanges(value: unknown, pointCount: number, label: string): readonly PointRange[] {
  return asArray(value, label).map((rawRange, index) => {
    const range = asArray(rawRange, `${label}[${index}]`);
    if (range.length !== 2 || range.some((item) => !Number.isSafeInteger(item))) {
      throw contractError(`${label}[${index}] must contain two safe integers`);
    }
    const start = range[0] as number;
    const end = range[1] as number;
    if (start < 0 || end < start || end > pointCount) {
      throw contractError(`${label}[${index}] point range is out of bounds`);
    }
    return [start, end] as const;
  });
}

function validateObservations(value: unknown, label: string): readonly ObjectObservation[] {
  return asArray(value, label).map((rawObservation, index) => {
    const observation = asRecord(rawObservation, `${label}[${index}]`);
    return {
      image_id: asNonNegativeInteger(observation.image_id, `${label}[${index}].image_id`),
      object_id: asNonNegativeInteger(observation.object_id, `${label}[${index}].object_id`),
      removed: asBoolean(observation.removed, `${label}[${index}].removed`),
      thumbnail: asNonEmptyString(observation.thumbnail, `${label}[${index}].thumbnail`),
    };
  });
}

function assertNonOverlappingRanges(ranges: readonly PointRange[]): void {
  const sorted = [...ranges].sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  for (let index = 1; index < sorted.length; index += 1) {
    if (sorted[index][0] < sorted[index - 1][1]) {
      throw contractError("objects point ranges overlap");
    }
  }
}

function asRecord(value: unknown, label: string): ReadonlyRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw contractError(`${label} must be an object`);
  }
  return value as ReadonlyRecord;
}

function asArray(value: unknown, label: string): readonly unknown[] {
  if (!Array.isArray(value)) throw contractError(`${label} must be an array`);
  return value;
}

function asNonEmptyString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw contractError(`${label} must be a non-empty string`);
  }
  return value;
}

function asNonNegativeInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw contractError(`${label} must be a non-negative safe integer`);
  }
  return value as number;
}

function asBoolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") throw contractError(`${label} must be a boolean`);
  return value;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function contractError(message: string): Error {
  return new Error(`Invalid viewer bundle: ${message}`);
}
