export type ReadonlyRecord = Readonly<Record<string, unknown>>;

export interface CurrentPointer {
  readonly schema_version: "2.0.0";
  readonly run_id: string;
  readonly complete: true;
}

export interface ArrayDescriptor {
  readonly path: string;
  readonly dtype: "float32" | "uint8" | "int8";
  readonly components: 3;
  readonly byte_length: number;
}

export interface Manifest {
  readonly schema_version: "2.0.0";
  readonly coordinate_space: "da3_world_meters";
  readonly point_count: number;
  readonly display_bounds: readonly [number, number, number, number, number, number];
  readonly arrays: Readonly<{
    readonly positions: ArrayDescriptor;
    readonly colors: ArrayDescriptor;
    readonly normals: ArrayDescriptor;
  }>;
  readonly world_to_view: readonly number[];
  readonly coordinate_convention: string;
  readonly objects_path: "objects.json";
  readonly footprints_path: "footprints.json";
  readonly source: ManifestSource;
  readonly capabilities: Readonly<{
    readonly point_picking: true;
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
  readonly filter_config: PointCloudFilterSource;
  readonly exporter_source_sha256: string;
  readonly global_mapping_sha256: string;
}

export interface PointCloudFilterSource {
  readonly enabled: boolean;
  readonly sor_nb_neighbors: number;
  readonly sor_std_ratio: number;
  readonly keep_main_clusters: boolean;
  readonly cluster_eps_scale: number;
  readonly cluster_min_points: number;
  readonly min_cluster_ratio: number;
  readonly remove_ground: boolean;
  readonly ground_dist_scale: number;
  readonly ground_min_inlier_ratio: number;
  readonly min_remaining_ratio: number;
  readonly min_points: number;
}

export interface Sam3MaskSource {
  readonly schema: "sam3_self_exemplar_processed_mask_cache_v1";
  readonly coordinate_space: "da3_processed_pixels";
  readonly producer: "sku_matching";
}

export interface ManifestSource {
  readonly sam3_mask: Sam3MaskSource;
  readonly da3_cache: Da3CacheSource;
  readonly footprint: FootprintSource;
  readonly export: ExportSource;
}

export interface ObjectInstance {
  readonly image_id: number;
  readonly object_id: number;
  readonly bbox: readonly [number, number, number, number];
  readonly removed: boolean;
  readonly point_index_range: readonly [number, number];
  readonly thumbnail: string;
  readonly classification: ObjectClassificationObservation;
}

export interface ProductMetadata {
  readonly status: "master_data_pending";
  readonly manufacturer: null;
  readonly brand: null;
  readonly category: null;
  readonly object_kind: null;
}

export type ObjectClassificationObservation =
  | Readonly<{
      readonly schema_version: "1.0.0";
      readonly source: "personalcare";
      readonly project_id: number;
      readonly status: "resolved";
      readonly sku_id: string;
      readonly sku_name: string;
      readonly confidence: number;
      readonly metadata: ProductMetadata;
    }>
  | Readonly<{
      readonly schema_version: "1.0.0";
      readonly source: "personalcare";
      readonly project_id: number;
      readonly status: "unavailable";
      readonly reason: "invalid_bbox";
    }>;

export interface ClassificationCandidate {
  readonly sku_id: string;
  readonly sku_name: string;
  readonly confidence_sum: number;
  readonly support_count: number;
  readonly max_confidence: number;
}

export interface ClassificationAggregate {
  readonly status: "resolved" | "conflict" | "unavailable";
  readonly primary_sku_id: string | null;
  readonly candidates: readonly ClassificationCandidate[];
  readonly metadata: ProductMetadata;
}

export interface ObjectIndexEntry {
  readonly images: readonly number[];
  readonly objects: readonly number[];
  readonly active_count: number;
  readonly removed_count: number;
  readonly total_count: number;
  readonly instances: readonly ObjectInstance[];
  readonly classification: ClassificationAggregate;
}

const AGGREGATE_SUM_TOLERANCE = 1e-12;

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
  readonly metric: "da3_self_exemplar_ground_footprint_union";
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
const THUMBNAIL_PATH = /^thumbs\/(0|[1-9][0-9]*)_[0-9]+\.jpg$/;

export function validateCurrent(value: unknown): CurrentPointer {
  const record = asRecord(value, "CURRENT");
  requireExactKeys(record, ["schema_version", "run_id", "complete"], "CURRENT");
  if (record.schema_version === "1.0.0") {
    throw contractError("bundle schema 1.0.0 is obsolete; rerun matching, footprint, and viewer export");
  }
  if (record.schema_version !== "2.0.0") throw contractError("CURRENT schema_version must be 2.0.0");
  if (record.complete !== true) throw contractError("CURRENT complete must be true");
  const runId = asString(record.run_id, "CURRENT run_id");
  if (!RUN_ID.test(runId)) throw contractError("CURRENT run_id must be 32 lowercase hex characters");
  return { schema_version: "2.0.0", run_id: runId, complete: true };
}

export function validateManifest(value: unknown): Manifest {
  const record = asRecord(value, "manifest");
  if (!("world_to_view" in record)) {
    throw contractError("bundle 缺 world_to_view，请用最新导出器重新导出");
  }
  requireExactKeys(record, [
    "schema_version", "coordinate_space", "point_count", "display_bounds", "arrays", "world_to_view",
    "coordinate_convention", "objects_path", "footprints_path", "source", "capabilities",
  ], "manifest");
  if (record.schema_version === "1.0.0") {
    throw contractError("bundle schema 1.0.0 is obsolete; rerun matching, footprint, and viewer export");
  }
  if (record.schema_version !== "2.0.0") throw contractError("manifest schema_version must be 2.0.0");
  if (record.coordinate_space !== "da3_world_meters") throw contractError("manifest coordinate_space is invalid");
  const pointCount = asSafeInteger(record.point_count, "manifest point_count");
  if (pointCount < 0) throw contractError("manifest point_count must be non-negative");
  const displayBounds = validateDisplayBounds(record.display_bounds);
  if (record.objects_path !== "objects.json" || record.footprints_path !== "footprints.json") {
    throw contractError("manifest JSON paths are invalid");
  }
  const worldToView = validateWorldToView(record.world_to_view);
  const coordinateConvention = asString(record.coordinate_convention, "manifest coordinate_convention");
  if (coordinateConvention.trim().length === 0) throw contractError("manifest coordinate_convention must be non-empty");
  const source = validateManifestSource(record.source);
  const capabilities = asRecord(record.capabilities, "manifest capabilities");
  requireExactKeys(capabilities, ["point_picking", "footprint_picking", "formal_ground_footprint"], "manifest capabilities");
  if (capabilities.point_picking !== true || capabilities.footprint_picking !== true || capabilities.formal_ground_footprint !== true) {
    throw contractError("manifest capabilities are invalid");
  }
  const arraysRecord = asRecord(record.arrays, "manifest arrays");
  if (!("normals" in arraysRecord)) {
    throw contractError("bundle 缺 normals，请用最新导出器重新导出");
  }
  requireExactKeys(arraysRecord, ["positions", "colors", "normals"], "manifest arrays");
  const arrays = {
    positions: validateArrayDescriptor(arraysRecord.positions, "positions", "positions.f32.bin", "float32", pointCount),
    colors: validateArrayDescriptor(arraysRecord.colors, "colors", "colors.u8.bin", "uint8", pointCount),
    normals: validateArrayDescriptor(arraysRecord.normals, "normals", "normals.i8.bin", "int8", pointCount),
  };
  return {
    schema_version: "2.0.0",
    coordinate_space: "da3_world_meters",
    point_count: pointCount,
    display_bounds: displayBounds,
    arrays,
    world_to_view: worldToView,
    coordinate_convention: coordinateConvention,
    objects_path: "objects.json",
    footprints_path: "footprints.json",
    source,
    capabilities: { point_picking: true, footprint_picking: true, formal_ground_footprint: true },
  };
}

function validateDisplayBounds(value: unknown): readonly [number, number, number, number, number, number] {
  const bounds = asArray(value, "manifest display_bounds");
  if (bounds.length !== 6 || bounds.some((item) => !isFiniteNumber(item))) {
    throw contractError("manifest display_bounds must contain six finite numbers");
  }
  const numericBounds = bounds as number[];
  if (numericBounds[0] > numericBounds[3] || numericBounds[1] > numericBounds[4] || numericBounds[2] > numericBounds[5]) {
    throw contractError("manifest display_bounds minimum must not exceed maximum");
  }
  return bounds as readonly [number, number, number, number, number, number];
}

function validateWorldToView(value: unknown): readonly number[] {
  const matrix = asArray(value, "manifest world_to_view");
  if (matrix.length !== 16 || matrix.some((item) => !isFiniteNumber(item))) {
    throw contractError("manifest world_to_view must contain sixteen finite numbers (row-major 4x4)");
  }
  const m = matrix as number[];
  const epsilon = 1e-5;
  if (![0, 0, 0, 1].every((value, index) => Math.abs(m[12 + index] - value) <= epsilon)) {
    throw contractError("manifest world_to_view must be a row-major rigid affine transform");
  }
  const rotation = [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]];
  for (let row = 0; row < 3; row += 1) for (let column = 0; column < 3; column += 1) {
    const dot = rotation[row][0] * rotation[column][0] + rotation[row][1] * rotation[column][1] + rotation[row][2] * rotation[column][2];
    if (Math.abs(dot - (row === column ? 1 : 0)) > epsilon) throw contractError("manifest world_to_view rotation must be orthonormal");
  }
  const determinant = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) + m[2] * (m[4] * m[9] - m[5] * m[8]);
  if (Math.abs(determinant - 1) > epsilon) throw contractError("manifest world_to_view rotation must have determinant +1");
  return m;
}

export function validateObjectIndex(value: unknown, pointCount: number): ObjectIndex {
  const record = asRecord(value, "objects");
  const result: Record<string, ObjectIndexEntry> = {};
  const nonEmptyRanges: Array<readonly [number, number]> = [];
  for (const [globalId, rawEntry] of Object.entries(record)) {
    if (!GLOBAL_ID.test(globalId)) throw contractError(`objects global ID key is invalid: ${globalId}`);
    const entry = asRecord(rawEntry, `objects[${globalId}]`);
    requireExactKeys(entry, ["images", "objects", "active_count", "removed_count", "total_count", "instances", "classification"], `objects[${globalId}]`);
    const images = validateIntegerArray(entry.images, `objects[${globalId}].images`, true);
    const objects = validateIntegerArray(entry.objects, `objects[${globalId}].objects`, false);
    const activeCount = asNonNegativeInteger(entry.active_count, `objects[${globalId}].active_count`);
    const removedCount = asNonNegativeInteger(entry.removed_count, `objects[${globalId}].removed_count`);
    const totalCount = asNonNegativeInteger(entry.total_count, `objects[${globalId}].total_count`);
    if (activeCount + removedCount !== totalCount) throw contractError(`objects[${globalId}] counts do not add up`);
    const rawInstances = asArray(entry.instances, `objects[${globalId}].instances`);
    const instances = rawInstances.map((instance, index) => validateInstance(instance, `objects[${globalId}].instances[${index}]`, pointCount, `thumbs/${globalId}_${index}.jpg`));
    const classification = validateClassificationAggregate(entry.classification, `objects[${globalId}].classification`);
    if (instances.length !== totalCount) throw contractError(`objects[${globalId}] total_count does not match instances`);
    if (instances.filter((instance) => instance.removed).length !== removedCount) throw contractError(`objects[${globalId}] removed_count does not match instances`);
    if (instances.filter((instance) => !instance.removed).length !== activeCount) throw contractError(`objects[${globalId}] active_count does not match instances`);
    const derivedImages = [...new Set(instances.map((instance) => instance.image_id))].sort((left, right) => left - right);
    const derivedObjects = instances.map((instance) => instance.object_id).sort((left, right) => left - right);
    if (!sameNumberArray(images, derivedImages)) throw contractError(`objects[${globalId}].images does not match instances`);
    if (!sameNumberArray(objects, derivedObjects)) throw contractError(`objects[${globalId}].objects does not match instances`);
    assertClassificationAggregateMatchesInstances(
      classification,
      instances,
      `objects[${globalId}].classification`,
    );
    for (const instance of instances) if (instance.point_index_range[1] > instance.point_index_range[0]) nonEmptyRanges.push(instance.point_index_range);
    result[globalId] = { images, objects, active_count: activeCount, removed_count: removedCount, total_count: totalCount, instances, classification };
  }
  nonEmptyRanges.sort((left, right) => left[0] - right[0]);
  for (let index = 1; index < nonEmptyRanges.length; index += 1) if (nonEmptyRanges[index][0] < nonEmptyRanges[index - 1][1]) throw contractError("objects point_index_range values overlap");
  return result;
}

export function validateFootprints(value: unknown): FootprintBundle {
  const record = asRecord(value, "footprints");
  requireExactKeys(record, ["metric", "unit", "status", "value_m2", "rejection_reason", "run_id", "support_plane", "per_global_id", "union"], "footprints");
  if (record.metric !== "da3_self_exemplar_ground_footprint_union" || record.unit !== "m2") throw contractError("footprint metric or unit is invalid");
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
    return { metric: "da3_self_exemplar_ground_footprint_union", unit: "m2", status, value_m2: measurementValue, rejection_reason: rejectionReason, run_id: runId, support_plane: supportPlane, per_global_id: perGlobalId, union };
  }
  if (measurementValue !== null) throw contractError("rejected footprint value_m2 must be null");
  if (rejectionReason === null || rejectionReason.trim().length === 0) throw contractError("rejected footprint rejection_reason must be non-empty");
  if (record.support_plane !== null || record.union !== null || Object.keys(perGlobalIdRecord).length !== 0) throw contractError("rejected footprint must not contain geometry");
  return { metric: "da3_self_exemplar_ground_footprint_union", unit: "m2", status, value_m2: null, rejection_reason: rejectionReason, run_id: runId, support_plane: null, per_global_id: {}, union: null };
}

function validateArrayDescriptor(value: unknown, name: string, path: string, dtype: ArrayDescriptor["dtype"], pointCount: number): ArrayDescriptor {
  const record = asRecord(value, `manifest arrays.${name}`);
  requireExactKeys(record, ["path", "dtype", "components", "byte_length"], `manifest arrays.${name}`);
  if (record.path !== path || record.dtype !== dtype || record.components !== 3) throw contractError(`manifest arrays.${name} descriptor is invalid`);
  const byteLength = asNonNegativeInteger(record.byte_length, `manifest arrays.${name}.byte_length`);
  const bytesPerElement = dtype === "float32" ? 4 : 1;
  if (byteLength !== pointCount * 3 * bytesPerElement) throw contractError(`manifest arrays.${name}.byte_length is inconsistent with point_count`);
  return { path, dtype, components: 3, byte_length: byteLength };
}

function validateInstance(value: unknown, label: string, pointCount: number, expectedThumbnail: string): ObjectInstance {
  const record = asRecord(value, label);
  if (!("thumbnail" in record)) {
    throw contractError(`bundle 缺 instance thumbnail，请用最新导出器重新导出`);
  }
  requireExactKeys(record, ["image_id", "object_id", "bbox", "removed", "point_index_range", "thumbnail", "classification"], label);
  const imageId = asSafeInteger(record.image_id, `${label}.image_id`);
  const objectId = asSafeInteger(record.object_id, `${label}.object_id`);
  const rawBbox = asArray(record.bbox, `${label}.bbox`);
  if (rawBbox.length !== 4 || rawBbox.some((item) => !isFiniteNumber(item))) throw contractError(`${label}.bbox must contain four finite numbers`);
  if ((rawBbox[0] as number) > (rawBbox[2] as number) || (rawBbox[1] as number) > (rawBbox[3] as number)) throw contractError(`${label}.bbox must be ordered`);
  if (typeof record.removed !== "boolean") throw contractError(`${label}.removed must be boolean`);
  const range = asArray(record.point_index_range, `${label}.point_index_range`);
  if (range.length !== 2 || range.some((item) => !Number.isSafeInteger(item))) throw contractError(`${label}.point_index_range must contain two safe integers`);
  const start = range[0] as number;
  const end = range[1] as number;
  if (start < 0 || end < start || end > pointCount) throw contractError(`${label}.point_index_range is out of bounds`);
  const thumbnail = asString(record.thumbnail, `${label}.thumbnail`);
  if (!THUMBNAIL_PATH.test(thumbnail) || thumbnail !== expectedThumbnail) throw contractError(`${label}.thumbnail must match its global-ID instance identity`);
  const classification = validateClassificationObservation(record.classification, `${label}.classification`);
  return { image_id: imageId, object_id: objectId, bbox: [rawBbox[0] as number, rawBbox[1] as number, rawBbox[2] as number, rawBbox[3] as number], removed: record.removed, point_index_range: [start, end], thumbnail, classification };
}

function validateProductMetadata(value: unknown, label: string): ProductMetadata {
  const record = asRecord(value, label);
  requireExactKeys(record, ["status", "manufacturer", "brand", "category", "object_kind"], label);
  if (record.status !== "master_data_pending" || record.manufacturer !== null || record.brand !== null || record.category !== null || record.object_kind !== null) {
    throw contractError(`${label} is invalid`);
  }
  return { status: "master_data_pending", manufacturer: null, brand: null, category: null, object_kind: null };
}

function validateClassificationObservation(value: unknown, label: string): ObjectClassificationObservation {
  const record = asRecord(value, label);
  if (record.status === "resolved") {
    requireExactKeys(record, ["schema_version", "source", "project_id", "status", "sku_id", "sku_name", "confidence", "metadata"], label);
    if (record.schema_version !== "1.0.0" || record.source !== "personalcare") throw contractError(`${label} identity is invalid`);
    const projectId = asSafeInteger(record.project_id, `${label}.project_id`);
    if (projectId !== 51) throw contractError(`${label}.project_id must be 51`);
    const skuId = asNonEmptyString(record.sku_id, `${label}.sku_id`);
    const skuName = asNonEmptyString(record.sku_name, `${label}.sku_name`);
    const confidence = record.confidence;
    if (!isFiniteNumber(confidence) || confidence < 0 || confidence > 1) throw contractError(`${label}.confidence must be within [0, 1]`);
    const metadata = validateProductMetadata(record.metadata, `${label}.metadata`);
    return { schema_version: "1.0.0", source: "personalcare", project_id: projectId, status: "resolved", sku_id: skuId, sku_name: skuName, confidence, metadata };
  }
  if (record.status === "unavailable") {
    requireExactKeys(record, ["schema_version", "source", "project_id", "status", "reason"], label);
    if (record.schema_version !== "1.0.0" || record.source !== "personalcare" || record.reason !== "invalid_bbox") throw contractError(`${label} is invalid`);
    const projectId = asSafeInteger(record.project_id, `${label}.project_id`);
    if (projectId !== 51) throw contractError(`${label}.project_id must be 51`);
    return { schema_version: "1.0.0", source: "personalcare", project_id: projectId, status: "unavailable", reason: "invalid_bbox" };
  }
  throw contractError(`${label}.status is invalid`);
}

function validateClassificationAggregate(value: unknown, label: string): ClassificationAggregate {
  const record = asRecord(value, label);
  requireExactKeys(record, ["status", "primary_sku_id", "candidates", "metadata"], label);
  if (record.status !== "resolved" && record.status !== "conflict" && record.status !== "unavailable") throw contractError(`${label}.status is invalid`);
  const rawCandidates = asArray(record.candidates, `${label}.candidates`);
  const candidates = rawCandidates.map((candidate, index) => validateClassificationCandidate(candidate, `${label}.candidates[${index}]`));
  const seen = new Set<string>();
  for (const candidate of candidates) {
    const key = `${candidate.sku_id}\u0000${candidate.sku_name}`;
    if (seen.has(key)) throw contractError(`${label}.candidates must be unique`);
    seen.add(key);
  }
  for (let index = 1; index < candidates.length; index += 1) {
    if (compareCandidates(candidates[index - 1], candidates[index]) > 0) throw contractError(`${label}.candidates order is invalid`);
  }
  const primary = record.primary_sku_id === null ? null : asNonEmptyString(record.primary_sku_id, `${label}.primary_sku_id`);
  if (record.status === "unavailable") {
    if (primary !== null || candidates.length !== 0) throw contractError(`${label} unavailable status requires null primary and no candidates`);
  } else {
    const validCardinality = record.status === "resolved" ? candidates.length === 1 : candidates.length >= 2;
    if (!validCardinality || primary === null || candidates[0].sku_id !== primary) throw contractError(`${label} status, primary, and candidates are inconsistent`);
  }
  const metadata = validateProductMetadata(record.metadata, `${label}.metadata`);
  return { status: record.status, primary_sku_id: primary, candidates, metadata };
}

function validateClassificationCandidate(value: unknown, label: string): ClassificationCandidate {
  const record = asRecord(value, label);
  requireExactKeys(record, ["sku_id", "sku_name", "confidence_sum", "support_count", "max_confidence"], label);
  const skuId = asNonEmptyString(record.sku_id, `${label}.sku_id`);
  const skuName = asNonEmptyString(record.sku_name, `${label}.sku_name`);
  if (!isFiniteNumber(record.confidence_sum) || record.confidence_sum < 0) throw contractError(`${label}.confidence_sum must be finite and non-negative`);
  const supportCount = asPositiveInteger(record.support_count, `${label}.support_count`);
  if (record.confidence_sum > supportCount) throw contractError(`${label}.confidence_sum cannot exceed support_count`);
  if (!isFiniteNumber(record.max_confidence) || record.max_confidence < 0 || record.max_confidence > 1) throw contractError(`${label}.max_confidence must be within [0, 1]`);
  return { sku_id: skuId, sku_name: skuName, confidence_sum: record.confidence_sum, support_count: supportCount, max_confidence: record.max_confidence };
}

function assertClassificationAggregateMatchesInstances(
  aggregate: ClassificationAggregate,
  instances: readonly ObjectInstance[],
  label: string,
): void {
  const groups = new Map<string, {
    skuId: string;
    skuName: string;
    confidences: number[];
    metadata: ProductMetadata;
  }>();
  for (const instance of instances) {
    const observation = instance.classification;
    if (observation.status === "unavailable") continue;
    const key = `${observation.sku_id}\u0000${observation.sku_name}`;
    const group = groups.get(key);
    if (group === undefined) {
      groups.set(key, {
        skuId: observation.sku_id,
        skuName: observation.sku_name,
        confidences: [observation.confidence],
        metadata: observation.metadata,
      });
    } else {
      group.confidences.push(observation.confidence);
    }
  }
  const candidates = [...groups.values()].map((group) => {
    const confidences = [...group.confidences].sort((left, right) => left - right);
    return {
      sku_id: group.skuId,
      sku_name: group.skuName,
      confidence_sum: compensatedSum(confidences),
      support_count: confidences.length,
      max_confidence: confidences[confidences.length - 1],
      metadata: group.metadata,
    };
  });
  candidates.sort(compareCandidates);
  const expected = candidates.length === 0
    ? { status: "unavailable" as const, primary_sku_id: null, candidates: [], metadata: pendingProductMetadata() }
    : {
      status: candidates.length === 1 ? "resolved" as const : "conflict" as const,
      primary_sku_id: candidates[0].sku_id,
      candidates,
      metadata: candidates[0].metadata,
    };
  if (aggregate.status !== expected.status || aggregate.primary_sku_id !== expected.primary_sku_id || !sameProductMetadata(aggregate.metadata, expected.metadata) || aggregate.candidates.length !== expected.candidates.length) {
    throw contractError(`${label} does not match resolved instance observations`);
  }
  for (let index = 0; index < expected.candidates.length; index += 1) {
    const actual = aggregate.candidates[index];
    const expectedCandidate = expected.candidates[index];
    if (actual.sku_id !== expectedCandidate.sku_id
      || actual.sku_name !== expectedCandidate.sku_name
      || actual.support_count !== expectedCandidate.support_count
      || actual.max_confidence !== expectedCandidate.max_confidence
      || !sameAggregateSum(actual.confidence_sum, expectedCandidate.confidence_sum)) {
      throw contractError(`${label} does not match resolved instance observations`);
    }
  }
}

function compensatedSum(values: readonly number[]): number {
  let sum = 0;
  let correction = 0;
  for (const value of values) {
    const adjusted = value - correction;
    const next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
  }
  return sum;
}

function sameAggregateSum(left: number, right: number): boolean {
  return Math.abs(left - right) <= AGGREGATE_SUM_TOLERANCE * Math.max(1, Math.abs(left), Math.abs(right));
}

function pendingProductMetadata(): ProductMetadata {
  return { status: "master_data_pending", manufacturer: null, brand: null, category: null, object_kind: null };
}

function sameProductMetadata(left: ProductMetadata, right: ProductMetadata): boolean {
  return left.status === right.status
    && left.manufacturer === right.manufacturer
    && left.brand === right.brand
    && left.category === right.category
    && left.object_kind === right.object_kind;
}

function compareCandidates(left: ClassificationCandidate, right: ClassificationCandidate): number {
  return right.confidence_sum - left.confidence_sum
    || right.support_count - left.support_count
    || right.max_confidence - left.max_confidence
    || compareStrings(left.sku_id, right.sku_id)
    || compareStrings(left.sku_name, right.sku_name);
}

function compareStrings(left: string, right: string): number {
  return left === right ? 0 : left < right ? -1 : 1;
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
  requireExactKeys(source, ["sam3_mask", "da3_cache", "footprint", "export"], "manifest source");
  const sam3Mask = asRecord(source.sam3_mask, "manifest source sam3_mask");
  requireExactKeys(sam3Mask, ["schema", "coordinate_space", "producer"], "manifest source sam3_mask");
  if (
    sam3Mask.schema !== "sam3_self_exemplar_processed_mask_cache_v1"
    || sam3Mask.coordinate_space !== "da3_processed_pixels"
    || sam3Mask.producer !== "sku_matching"
  ) {
    throw contractError("manifest source sam3_mask contract is invalid");
  }
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
  requireExactKeys(exportSource, ["voxel_size_m", "max_points", "filter_config", "exporter_source_sha256", "global_mapping_sha256"], "manifest source export");
  if (!isFiniteNumber(exportSource.voxel_size_m) || exportSource.voxel_size_m <= 0) throw contractError("manifest source voxel_size_m is invalid");
  const maxPoints = asPositiveInteger(exportSource.max_points, "manifest source max_points");
  const filterConfig = validatePointCloudFilterSource(exportSource.filter_config);
  const exporterSourceSha256 = asSha256(exportSource.exporter_source_sha256, "manifest source exporter_source_sha256");
  const globalMappingSha256 = asSha256(exportSource.global_mapping_sha256, "manifest source global_mapping_sha256");
  return {
    sam3_mask: {
      schema: "sam3_self_exemplar_processed_mask_cache_v1",
      coordinate_space: "da3_processed_pixels",
      producer: "sku_matching",
    },
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
    export: {
      voxel_size_m: exportSource.voxel_size_m,
      max_points: maxPoints,
      filter_config: filterConfig,
      exporter_source_sha256: exporterSourceSha256,
      global_mapping_sha256: globalMappingSha256,
    },
  };
}

function validatePointCloudFilterSource(value: unknown): PointCloudFilterSource {
  const record = asRecord(value, "manifest source filter_config");
  requireExactKeys(record, ["enabled", "sor_nb_neighbors", "sor_std_ratio", "keep_main_clusters", "cluster_eps_scale", "cluster_min_points", "min_cluster_ratio", "remove_ground", "ground_dist_scale", "ground_min_inlier_ratio", "min_remaining_ratio", "min_points"], "manifest source filter_config");
  if (typeof record.enabled !== "boolean" || typeof record.keep_main_clusters !== "boolean" || typeof record.remove_ground !== "boolean") throw contractError("manifest source filter_config booleans are invalid");
  const sorNbNeighbors = asPositiveInteger(record.sor_nb_neighbors, "manifest source filter_config.sor_nb_neighbors");
  const clusterMinPoints = asPositiveInteger(record.cluster_min_points, "manifest source filter_config.cluster_min_points");
  const minPoints = asPositiveInteger(record.min_points, "manifest source filter_config.min_points");
  const sorStdRatio = asPositiveFiniteNumber(record.sor_std_ratio, "manifest source filter_config.sor_std_ratio");
  const clusterEpsScale = asPositiveFiniteNumber(record.cluster_eps_scale, "manifest source filter_config.cluster_eps_scale");
  const minClusterRatio = asPositiveFiniteNumber(record.min_cluster_ratio, "manifest source filter_config.min_cluster_ratio");
  const groundDistScale = asPositiveFiniteNumber(record.ground_dist_scale, "manifest source filter_config.ground_dist_scale");
  const groundMinInlierRatio = asPositiveFiniteNumber(record.ground_min_inlier_ratio, "manifest source filter_config.ground_min_inlier_ratio");
  const minRemainingRatio = asPositiveFiniteNumber(record.min_remaining_ratio, "manifest source filter_config.min_remaining_ratio");
  return {
    enabled: record.enabled,
    sor_nb_neighbors: sorNbNeighbors,
    sor_std_ratio: sorStdRatio,
    keep_main_clusters: record.keep_main_clusters,
    cluster_eps_scale: clusterEpsScale,
    cluster_min_points: clusterMinPoints,
    min_cluster_ratio: minClusterRatio,
    remove_ground: record.remove_ground,
    ground_dist_scale: groundDistScale,
    ground_min_inlier_ratio: groundMinInlierRatio,
    min_remaining_ratio: minRemainingRatio,
    min_points: minPoints,
  };
}

function asSha256(value: unknown, label: string): string {
  const digest = asString(value, label);
  if (!/^[0-9a-f]{64}$/.test(digest)) throw contractError(`${label} must be a lowercase SHA-256 digest`);
  return digest;
}

function asPositiveFiniteNumber(value: unknown, label: string): number {
  if (!isFiniteNumber(value) || value <= 0) throw contractError(`${label} must be positive and finite`);
  return value;
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

function asNonEmptyString(value: unknown, label: string): string {
  const string = asString(value, label);
  if (string.trim().length === 0) throw contractError(`${label} must be non-empty`);
  return string;
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
