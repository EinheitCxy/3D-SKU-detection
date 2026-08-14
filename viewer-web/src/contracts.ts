export type ReadonlyRecord = Readonly<Record<string, unknown>>;

export interface CurrentPointer {
  readonly schema_version: "1.0.0";
  readonly run_id: string;
  readonly complete: true;
}

export interface ArrayDescriptor {
  readonly path: string;
  readonly dtype: "float32" | "uint8" | "int32";
  readonly components: 1 | 3;
  readonly byte_length: number;
}

export interface Manifest {
  readonly schema_version: "1.0.0";
  readonly coordinate_space: "da3_world_meters";
  readonly point_count: number;
  readonly arrays: Readonly<{
    readonly positions: ArrayDescriptor;
    readonly colors: ArrayDescriptor;
    readonly confidences: ArrayDescriptor;
    readonly frame_ids: ArrayDescriptor;
  }>;
  readonly objects_path: "objects.json";
  readonly footprints_path: "footprints.json";
  readonly source: ManifestSource;
  readonly capabilities: Readonly<{
    readonly point_picking: false;
    readonly footprint_picking: true;
    readonly formal_ground_footprint: true;
  }>;
}

export interface Da3CacheSource {
  readonly schema_version: 2;
  readonly source_model: string;
  readonly affine_convention: "pixel_center_v1";
  readonly preprocess_resolution: number;
  readonly preprocess_method: "upper_bound_resize";
  readonly frame_count: number;
  readonly processed_size: readonly [number, number];
  readonly image_ids: readonly number[];
  readonly source_image_sha256: readonly string[];
}

export interface FootprintSource {
  readonly run_id: string;
  readonly status: "accepted" | "rejected";
}

export interface ExportSource {
  readonly voxel_size_m: number;
  readonly max_points: number;
}

export interface ManifestSource {
  readonly da3_cache: Da3CacheSource;
  readonly footprint: FootprintSource;
  readonly export: ExportSource;
}

export interface ObjectInstance {
  readonly image_id: number;
  readonly object_id: number;
  readonly bbox: readonly [number, number, number, number];
  readonly removed: boolean;
}

export interface ObjectIndexEntry {
  readonly images: readonly number[];
  readonly objects: readonly number[];
  readonly active_count: number;
  readonly removed_count: number;
  readonly total_count: number;
  readonly instances: readonly ObjectInstance[];
}

export type ObjectIndex = Readonly<Record<string, ObjectIndexEntry>>;

export interface SupportPlane {
  readonly point: readonly [number, number, number];
  readonly u_axis: readonly [number, number, number];
  readonly v_axis: readonly [number, number, number];
  readonly normal: readonly [number, number, number];
}

export type FootprintRing = readonly (readonly [number, number])[];
export type FootprintPolygon = readonly FootprintRing[];

export interface FootprintGeometry {
  readonly rings: readonly FootprintPolygon[];
  readonly properties: PerGlobalIDProperties | UnionProperties;
}

export interface PerGlobalIDProperties {
  readonly coordinate_space: "local_support_plane_meters";
  readonly global_id: string;
  readonly area_m2: number;
  readonly observations_used: number;
}

export interface UnionProperties {
  readonly coordinate_space: "local_support_plane_meters";
  readonly global_id: "union";
  readonly area_m2: number;
}

export interface FootprintBundle {
  readonly metric: "da3_ground_footprint_union";
  readonly unit: "m2";
  readonly status: "accepted" | "rejected";
  readonly value_m2: number | null;
  readonly rejection_reason: string | null;
  readonly run_id: string;
  readonly support_plane: SupportPlane | null;
  readonly per_global_id: Readonly<Record<string, FootprintGeometry>>;
  readonly union: FootprintGeometry | null;
}

const RUN_ID = /^[0-9a-f]{32}$/;
const GLOBAL_ID = /^(0|[1-9][0-9]*)$/;

export function validateCurrent(value: unknown): CurrentPointer {
  const record = asRecord(value, "CURRENT");
  requireExactKeys(record, ["schema_version", "run_id", "complete"], "CURRENT");
  if (record.schema_version !== "1.0.0") throw contractError("CURRENT schema_version must be 1.0.0");
  if (record.complete !== true) throw contractError("CURRENT complete must be true");
  const runId = asString(record.run_id, "CURRENT run_id");
  if (!RUN_ID.test(runId)) throw contractError("CURRENT run_id must be 32 lowercase hex characters");
  return { schema_version: "1.0.0", run_id: runId, complete: true };
}

export function validateManifest(value: unknown): Manifest {
  const record = asRecord(value, "manifest");
  requireExactKeys(record, [
    "schema_version", "coordinate_space", "point_count", "arrays", "objects_path",
    "footprints_path", "source", "capabilities",
  ], "manifest");
  if (record.schema_version !== "1.0.0") throw contractError("manifest schema_version must be 1.0.0");
  if (record.coordinate_space !== "da3_world_meters") throw contractError("manifest coordinate_space is invalid");
  const pointCount = asSafeInteger(record.point_count, "manifest point_count");
  if (pointCount < 0) throw contractError("manifest point_count must be non-negative");
  if (record.objects_path !== "objects.json" || record.footprints_path !== "footprints.json") {
    throw contractError("manifest JSON paths are invalid");
  }
  const source = validateManifestSource(record.source);
  const capabilities = asRecord(record.capabilities, "manifest capabilities");
  requireExactKeys(capabilities, ["point_picking", "footprint_picking", "formal_ground_footprint"], "manifest capabilities");
  if (capabilities.point_picking !== false || capabilities.footprint_picking !== true || capabilities.formal_ground_footprint !== true) {
    throw contractError("manifest capabilities are invalid");
  }
  const arraysRecord = asRecord(record.arrays, "manifest arrays");
  requireExactKeys(arraysRecord, ["positions", "colors", "confidences", "frame_ids"], "manifest arrays");
  const arrays = {
    positions: validateArrayDescriptor(arraysRecord.positions, "positions", "positions.f32.bin", "float32", 3, pointCount),
    colors: validateArrayDescriptor(arraysRecord.colors, "colors", "colors.u8.bin", "uint8", 3, pointCount),
    confidences: validateArrayDescriptor(arraysRecord.confidences, "confidences", "confidences.f32.bin", "float32", 1, pointCount),
    frame_ids: validateArrayDescriptor(arraysRecord.frame_ids, "frame_ids", "frame_ids.i32.bin", "int32", 1, pointCount),
  };
  return {
    schema_version: "1.0.0",
    coordinate_space: "da3_world_meters",
    point_count: pointCount,
    arrays,
    objects_path: "objects.json",
    footprints_path: "footprints.json",
    source,
    capabilities: { point_picking: false, footprint_picking: true, formal_ground_footprint: true },
  };
}

export function validateObjectIndex(value: unknown): ObjectIndex {
  const record = asRecord(value, "objects");
  const result: Record<string, ObjectIndexEntry> = {};
  for (const [globalId, rawEntry] of Object.entries(record)) {
    if (!GLOBAL_ID.test(globalId)) throw contractError(`objects global ID key is invalid: ${globalId}`);
    const entry = asRecord(rawEntry, `objects[${globalId}]`);
    requireExactKeys(entry, ["images", "objects", "active_count", "removed_count", "total_count", "instances"], `objects[${globalId}]`);
    const images = validateIntegerArray(entry.images, `objects[${globalId}].images`, true);
    const objects = validateIntegerArray(entry.objects, `objects[${globalId}].objects`, false);
    const activeCount = asNonNegativeInteger(entry.active_count, `objects[${globalId}].active_count`);
    const removedCount = asNonNegativeInteger(entry.removed_count, `objects[${globalId}].removed_count`);
    const totalCount = asNonNegativeInteger(entry.total_count, `objects[${globalId}].total_count`);
    if (activeCount + removedCount !== totalCount) throw contractError(`objects[${globalId}] counts do not add up`);
    const rawInstances = asArray(entry.instances, `objects[${globalId}].instances`);
    const instances = rawInstances.map((instance, index) => validateInstance(instance, `objects[${globalId}].instances[${index}]`));
    if (instances.length !== totalCount) throw contractError(`objects[${globalId}] total_count does not match instances`);
    if (instances.filter((instance) => instance.removed).length !== removedCount) throw contractError(`objects[${globalId}] removed_count does not match instances`);
    if (instances.filter((instance) => !instance.removed).length !== activeCount) throw contractError(`objects[${globalId}] active_count does not match instances`);
    const derivedImages = [...new Set(instances.map((instance) => instance.image_id))].sort((left, right) => left - right);
    const derivedObjects = instances.map((instance) => instance.object_id).sort((left, right) => left - right);
    if (!sameNumberArray(images, derivedImages)) throw contractError(`objects[${globalId}].images does not match instances`);
    if (!sameNumberArray(objects, derivedObjects)) throw contractError(`objects[${globalId}].objects does not match instances`);
    result[globalId] = { images, objects, active_count: activeCount, removed_count: removedCount, total_count: totalCount, instances };
  }
  return result;
}

export function validateFootprints(value: unknown): FootprintBundle {
  const record = asRecord(value, "footprints");
  requireExactKeys(record, ["metric", "unit", "status", "value_m2", "rejection_reason", "run_id", "support_plane", "per_global_id", "union"], "footprints");
  if (record.metric !== "da3_ground_footprint_union" || record.unit !== "m2") throw contractError("footprint metric or unit is invalid");
  if (record.status !== "accepted" && record.status !== "rejected") throw contractError("footprint status is invalid");
  const status = record.status;
  const runId = asString(record.run_id, "footprints run_id");
  if (!RUN_ID.test(runId)) throw contractError("footprints run_id must be 32 lowercase hex characters");
  const rejectionReason = record.rejection_reason === null ? null : asString(record.rejection_reason, "footprints rejection_reason");
  const measurementValue = record.value_m2;
  const perGlobalIdRecord = asRecord(record.per_global_id, "footprints per_global_id");
  if (status === "accepted") {
    if (!isFiniteNumber(measurementValue) || measurementValue < 0) throw contractError("accepted footprint value_m2 must be non-negative and finite");
    if (rejectionReason !== null) throw contractError("accepted footprint rejection_reason must be null");
    if (Object.keys(perGlobalIdRecord).length === 0) throw contractError("accepted footprint must contain per-ID geometry");
    const supportPlane = validateSupportPlane(record.support_plane);
    const perGlobalId: Record<string, FootprintGeometry> = {};
    for (const [globalId, geometry] of Object.entries(perGlobalIdRecord)) {
      if (!GLOBAL_ID.test(globalId)) throw contractError(`footprint global ID key is invalid: ${globalId}`);
      perGlobalId[globalId] = validateGeometry(geometry, globalId, "per-global-id");
    }
    if (record.union === null) throw contractError("accepted footprint must contain union geometry");
    const union = validateGeometry(record.union, "union", "union");
    if (union.properties.area_m2 !== measurementValue) throw contractError("footprint union area must equal value_m2");
    return { metric: "da3_ground_footprint_union", unit: "m2", status, value_m2: measurementValue, rejection_reason: rejectionReason, run_id: runId, support_plane: supportPlane, per_global_id: perGlobalId, union };
  }
  if (measurementValue !== null) throw contractError("rejected footprint value_m2 must be null");
  if (rejectionReason === null || rejectionReason.trim().length === 0) throw contractError("rejected footprint rejection_reason must be non-empty");
  if (record.support_plane !== null || record.union !== null || Object.keys(perGlobalIdRecord).length !== 0) throw contractError("rejected footprint must not contain geometry");
  return { metric: "da3_ground_footprint_union", unit: "m2", status, value_m2: null, rejection_reason: rejectionReason, run_id: runId, support_plane: null, per_global_id: {}, union: null };
}

function validateArrayDescriptor(value: unknown, name: string, path: string, dtype: ArrayDescriptor["dtype"], components: 1 | 3, pointCount: number): ArrayDescriptor {
  const record = asRecord(value, `manifest arrays.${name}`);
  requireExactKeys(record, ["path", "dtype", "components", "byte_length"], `manifest arrays.${name}`);
  if (record.path !== path || record.dtype !== dtype || record.components !== components) throw contractError(`manifest arrays.${name} descriptor is invalid`);
  const byteLength = asNonNegativeInteger(record.byte_length, `manifest arrays.${name}.byte_length`);
  const bytesPerElement = dtype === "uint8" ? 1 : 4;
  if (byteLength !== pointCount * components * bytesPerElement) throw contractError(`manifest arrays.${name}.byte_length is inconsistent with point_count`);
  return { path, dtype, components, byte_length: byteLength };
}

function validateInstance(value: unknown, label: string): ObjectInstance {
  const record = asRecord(value, label);
  requireExactKeys(record, ["image_id", "object_id", "bbox", "removed"], label);
  const imageId = asSafeInteger(record.image_id, `${label}.image_id`);
  const objectId = asSafeInteger(record.object_id, `${label}.object_id`);
  const rawBbox = asArray(record.bbox, `${label}.bbox`);
  if (rawBbox.length !== 4 || rawBbox.some((item) => !isFiniteNumber(item))) throw contractError(`${label}.bbox must contain four finite numbers`);
  if (typeof record.removed !== "boolean") throw contractError(`${label}.removed must be boolean`);
  return { image_id: imageId, object_id: objectId, bbox: [rawBbox[0] as number, rawBbox[1] as number, rawBbox[2] as number, rawBbox[3] as number], removed: record.removed };
}

function validateSupportPlane(value: unknown): SupportPlane {
  const record = asRecord(value, "footprints support_plane");
  requireExactKeys(record, ["point", "u_axis", "v_axis", "normal"], "footprints support_plane");
  return {
    point: validateVector(record.point, "support_plane.point"),
    u_axis: validateVector(record.u_axis, "support_plane.u_axis"),
    v_axis: validateVector(record.v_axis, "support_plane.v_axis"),
    normal: validateVector(record.normal, "support_plane.normal"),
  };
}

function validateVector(value: unknown, label: string): readonly [number, number, number] {
  const vector = asArray(value, label);
  if (vector.length !== 3 || vector.some((item) => !isFiniteNumber(item))) throw contractError(`${label} must contain three finite numbers`);
  return [vector[0] as number, vector[1] as number, vector[2] as number];
}

function validateGeometry(value: unknown, expectedGlobalId: string, kind: "per-global-id" | "union"): FootprintGeometry {
  const record = asRecord(value, "footprint geometry");
  requireExactKeys(record, ["rings", "properties"], "footprint geometry");
  const properties = asRecord(record.properties, "footprint geometry properties");
  let validatedProperties: PerGlobalIDProperties | UnionProperties;
  if (kind === "per-global-id") {
    requireExactKeys(properties, ["coordinate_space", "global_id", "area_m2", "observations_used"], "per-ID footprint properties");
    if (properties.coordinate_space !== "local_support_plane_meters" || properties.global_id !== expectedGlobalId) throw contractError("per-ID footprint properties identity is invalid");
    const area = properties.area_m2;
    const observations = properties.observations_used;
    if (!isFiniteNumber(area) || area < 0) throw contractError("per-ID footprint area_m2 is invalid");
    if (typeof observations !== "number" || !Number.isSafeInteger(observations) || observations < 0) throw contractError("per-ID footprint observations_used is invalid");
    validatedProperties = { coordinate_space: "local_support_plane_meters", global_id: expectedGlobalId, area_m2: area, observations_used: observations };
  } else {
    requireExactKeys(properties, ["coordinate_space", "global_id", "area_m2"], "union footprint properties");
    if (properties.coordinate_space !== "local_support_plane_meters" || properties.global_id !== "union") throw contractError("union footprint properties identity is invalid");
    const area = properties.area_m2;
    if (!isFiniteNumber(area) || area < 0) throw contractError("union footprint area_m2 is invalid");
    validatedProperties = { coordinate_space: "local_support_plane_meters", global_id: "union", area_m2: area };
  }
  const rawPolygons = asArray(record.rings, "footprint geometry rings");
  if (rawPolygons.length === 0) throw contractError("footprint geometry rings must be non-empty");
  const rings = rawPolygons.map((rawPolygon, polygonIndex) => {
    const rawRings = asArray(rawPolygon, `footprint polygon ${polygonIndex}`);
    if (rawRings.length === 0) throw contractError("footprint polygon must contain a ring");
    return rawRings.map((rawRing, ringIndex) => {
      const rawCoordinates = asArray(rawRing, `footprint ring ${ringIndex}`);
      if (rawCoordinates.length < 4) throw contractError("footprint ring must contain at least four coordinates");
      const coordinates = rawCoordinates.map((coordinate) => validateCoordinate(coordinate));
      const first = coordinates[0];
      const last = coordinates[coordinates.length - 1];
      if (first[0] !== last[0] || first[1] !== last[1]) throw contractError("footprint ring must be closed");
      return coordinates;
    });
  });
  return { rings, properties: validatedProperties };
}

function validateCoordinate(value: unknown): readonly [number, number] {
  const coordinate = asArray(value, "footprint coordinate");
  if (coordinate.length !== 2 || coordinate.some((item) => !isFiniteNumber(item))) throw contractError("footprint coordinate must contain two finite numbers");
  return [coordinate[0] as number, coordinate[1] as number];
}

function validateIntegerArray(value: unknown, label: string, requireUnique: boolean): readonly number[] {
  const values = asArray(value, label).map((item) => asSafeInteger(item, label));
  if (requireUnique && new Set(values).size !== values.length) throw contractError(`${label} must contain unique values`);
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] < values[index - 1]) throw contractError(`${label} must be sorted`);
  }
  return values;
}

function sameNumberArray(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function validateManifestSource(value: unknown): ManifestSource {
  const source = asRecord(value, "manifest source");
  requireExactKeys(source, ["da3_cache", "footprint", "export"], "manifest source");
  const da3Cache = asRecord(source.da3_cache, "manifest source da3_cache");
  requireExactKeys(da3Cache, [
    "schema_version", "source_model", "affine_convention", "preprocess_resolution", "preprocess_method",
    "frame_count", "processed_size", "image_ids", "source_image_sha256",
  ], "manifest source da3_cache");
  if (da3Cache.schema_version !== 2) throw contractError("manifest source da3_cache schema_version must be 2");
  const sourceModel = asString(da3Cache.source_model, "manifest source source_model");
  if (!/^[A-Za-z0-9._/-]+$/.test(sourceModel)) throw contractError("manifest source source_model is unsafe");
  if (da3Cache.affine_convention !== "pixel_center_v1" || da3Cache.preprocess_method !== "upper_bound_resize") throw contractError("manifest source preprocessing contract is invalid");
  const preprocessResolution = asPositiveInteger(da3Cache.preprocess_resolution, "manifest source preprocess_resolution");
  const frameCount = asPositiveInteger(da3Cache.frame_count, "manifest source frame_count");
  const processedSize = validatePositiveIntegerTuple(da3Cache.processed_size, "manifest source processed_size");
  const imageIds = validateInt32Array(da3Cache.image_ids, "manifest source image_ids");
  if (imageIds.length !== frameCount || new Set(imageIds).size !== imageIds.length) throw contractError("manifest source image_ids must be unique and match frame_count");
  const hashes = asArray(da3Cache.source_image_sha256, "manifest source source_image_sha256").map((hash) => asString(hash, "manifest source source_image_sha256 item"));
  if (hashes.length !== frameCount || hashes.some((hash) => !/^[0-9a-f]{64}$/.test(hash))) throw contractError("manifest source source_image_sha256 is invalid");

  const footprint = asRecord(source.footprint, "manifest source footprint");
  requireExactKeys(footprint, ["run_id", "status"], "manifest source footprint");
  const footprintRunId = asString(footprint.run_id, "manifest source footprint run_id");
  if (!RUN_ID.test(footprintRunId) || (footprint.status !== "accepted" && footprint.status !== "rejected")) throw contractError("manifest source footprint is invalid");

  const exportSource = asRecord(source.export, "manifest source export");
  requireExactKeys(exportSource, ["voxel_size_m", "max_points"], "manifest source export");
  if (!isFiniteNumber(exportSource.voxel_size_m) || exportSource.voxel_size_m <= 0) throw contractError("manifest source voxel_size_m is invalid");
  const maxPoints = asPositiveInteger(exportSource.max_points, "manifest source max_points");
  return {
    da3_cache: {
      schema_version: 2,
      source_model: sourceModel,
      affine_convention: "pixel_center_v1",
      preprocess_resolution: preprocessResolution,
      preprocess_method: "upper_bound_resize",
      frame_count: frameCount,
      processed_size: processedSize,
      image_ids: imageIds,
      source_image_sha256: hashes,
    },
    footprint: { run_id: footprintRunId, status: footprint.status },
    export: { voxel_size_m: exportSource.voxel_size_m, max_points: maxPoints },
  };
}

function validatePositiveIntegerTuple(value: unknown, label: string): readonly [number, number] {
  const tuple = asArray(value, label);
  if (tuple.length !== 2) throw contractError(`${label} must contain two values`);
  return [asPositiveInteger(tuple[0], `${label}[0]`), asPositiveInteger(tuple[1], `${label}[1]`)];
}

function validateInt32Array(value: unknown, label: string): readonly number[] {
  const values = asArray(value, label).map((item) => asSafeInteger(item, label));
  if (values.some((item) => item < -2147483648 || item > 2147483647)) throw contractError(`${label} must contain int32 values`);
  return values;
}

function asRecord(value: unknown, label: string): ReadonlyRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw contractError(`${label} must be an object`);
  return value as ReadonlyRecord;
}

function asArray(value: unknown, label: string): readonly unknown[] {
  if (!Array.isArray(value)) throw contractError(`${label} must be an array`);
  return value;
}

function asString(value: unknown, label: string): string {
  if (typeof value !== "string") throw contractError(`${label} must be a string`);
  return value;
}

function asSafeInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value)) throw contractError(`${label} must be a safe integer`);
  return value as number;
}

function asNonNegativeInteger(value: unknown, label: string): number {
  const integer = asSafeInteger(value, label);
  if (integer < 0) throw contractError(`${label} must be non-negative`);
  return integer;
}

function asPositiveInteger(value: unknown, label: string): number {
  const integer = asSafeInteger(value, label);
  if (integer <= 0) throw contractError(`${label} must be positive`);
  return integer;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function requireExactKeys(record: ReadonlyRecord, expected: readonly string[], label: string): void {
  const actual = Object.keys(record).sort();
  const required = [...expected].sort();
  if (actual.length !== required.length || actual.some((key, index) => key !== required[index])) {
    throw contractError(`${label} fields are invalid`);
  }
}

function contractError(message: string): Error {
  return new Error(`Invalid viewer bundle: ${message}`);
}
