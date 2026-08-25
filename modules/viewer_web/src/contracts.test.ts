import { describe, expect, it } from "vitest";
import { validateFootprints, validateManifest, validateObjectIndex } from "./contracts";

const validSource = {
  da3_cache: {
    schema_version: 2,
    source_model: "depth-anything/DA3NESTED-GIANT-LARGE",
    affine_convention: "pixel_center_v1",
    preprocess_resolution: 2,
    preprocess_method: "upper_bound_resize",
    frame_count: 1,
    processed_size: [2, 2],
    image_ids: [7],
    source_image_sha256: ["0".repeat(64)],
  },
  footprint: { run_id: "a".repeat(32), status: "accepted" },
  sam3_mask: {
    schema: "sam3_self_exemplar_processed_mask_cache_v1",
    coordinate_space: "da3_processed_pixels",
    producer: "sku_matching",
  },
  export: {
    voxel_size_m: 0.01,
    max_points: 10,
    filter_config: {
      enabled: true,
      sor_nb_neighbors: 20,
      sor_std_ratio: 2,
      keep_main_clusters: true,
      cluster_eps_scale: 5,
      cluster_min_points: 10,
      min_cluster_ratio: 0.01,
      remove_ground: true,
      ground_dist_scale: 3,
      ground_min_inlier_ratio: 0.08,
      min_remaining_ratio: 0.2,
      min_points: 1000,
    },
    exporter_source_sha256: "1".repeat(64),
    global_mapping_sha256: "2".repeat(64),
  },
} as const;

const validManifest = {
  schema_version: "2.0.0",
  coordinate_space: "da3_world_meters",
  point_count: 1,
  display_bounds: [1, 2, 3, 1, 2, 3],
  arrays: {
    positions: { path: "positions.f32.bin", dtype: "float32", components: 3, byte_length: 12 },
    colors: { path: "colors.u8.bin", dtype: "uint8", components: 3, byte_length: 3 },
    normals: { path: "normals.i8.bin", dtype: "int8", components: 3, byte_length: 3 },
  },
  world_to_view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  coordinate_convention: "DA3 native CV coordinates; world_to_view maps to viewer Y-up (row-major)",
  objects_path: "objects.json",
  footprints_path: "footprints.json",
  source: validSource,
  capabilities: { point_picking: true, footprint_picking: true, formal_ground_footprint: true },
} as const;

const pendingMetadata = () => ({
  status: "master_data_pending" as const,
  manufacturer: null,
  brand: null,
  category: null,
  object_kind: null,
});

const candidate = (sku_id: string, sku_name: string, confidence_sum: number, support_count: number, max_confidence: number) => ({
  sku_id, sku_name, confidence_sum, support_count, max_confidence,
});

const resolvedObservation = () => ({
  schema_version: "1.0.0" as const,
  source: "personalcare" as const,
  project_id: 51,
  status: "resolved" as const,
  sku_id: "A",
  sku_name: "产品A",
  confidence: 0.9,
  metadata: pendingMetadata(),
});

const resolvedAggregate = () => ({
  status: "resolved" as const,
  primary_sku_id: "A",
  candidates: [candidate("A", "产品A", 0.9, 1, 0.9)],
  metadata: pendingMetadata(),
});

const validObjects = {
  "11": {
    images: [7],
    objects: [3],
    active_count: 1,
    removed_count: 0,
    total_count: 1,
    instances: [{ image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false, point_index_range: [0, 1], thumbnail: "thumbs/11_0.jpg", classification: resolvedObservation() }],
    classification: resolvedAggregate(),
  },
} as const;

const square = [[0, 0], [1, 0], [1, 1], [0, 0]];
const validFootprints = {
  metric: "da3_self_exemplar_ground_footprint_union",
  unit: "m2",
  status: "accepted",
  value_m2: 1,
  rejection_reason: null,
  run_id: "a".repeat(32),
  support_plane: {
    point: [0, 0, 0],
    u_axis: [1, 0, 0],
    v_axis: [0, 1, 0],
    normal: [0, 0, 1],
  },
  per_global_id: {
    "11": {
      rings: [[square]],
      properties: {
        coordinate_space: "local_support_plane_meters",
        global_id: "11",
        area_m2: 0.5,
        observations_used: 1,
      },
    },
  },
  union: {
    rings: [[square]],
    properties: { coordinate_space: "local_support_plane_meters", global_id: "union", area_m2: 1 },
  },
} as const;

describe("strict bundle contracts", () => {
  it("accepts the exact manifest schema and array layout", () => {
    expect(validateManifest(validManifest)).toMatchObject({ schema_version: "2.0.0", point_count: 1 });
  });

  it("rejects a manifest that does not advertise point picking", () => {
    expect(() => validateManifest({
      ...validManifest,
      capabilities: { ...validManifest.capabilities, point_picking: false },
    })).toThrow(/capabilities/);
  });

  it("accepts only the canonical processed mask bundle v2 source", () => {
    expect(validateManifest(validManifest).source.sam3_mask).toEqual({
      schema: "sam3_self_exemplar_processed_mask_cache_v1",
      coordinate_space: "da3_processed_pixels",
      producer: "sku_matching",
    });
  });

  it("rejects v1 with rerun guidance", () => {
    expect(() => validateManifest({ ...validManifest, schema_version: "1.0.0" }))
      .toThrow(/rerun matching, footprint, and viewer export/i);
  });

  it("rejects non-canonical SAM3 source fields", () => {
    expect(() => validateManifest({
      ...validManifest,
      source: { ...validSource, sam3_mask: { ...validSource.sam3_mask, producer: "other" } },
    })).toThrow();
    expect(() => validateManifest({
      ...validManifest,
      source: { ...validSource, sam3_mask: { ...validSource.sam3_mask, cache_schema: "legacy" } },
    })).toThrow();
  });

  it("fails closed with a re-export hint when world_to_view is missing", () => {
    const { world_to_view: _missing, ...withoutWorldToView } = validManifest;
    expect(() => validateManifest(withoutWorldToView)).toThrow(/缺 world_to_view.*重新导出/);
  });

  it("fails closed with a re-export hint when the normals array is missing", () => {
    const { normals: _missing, ...withoutNormals } = validManifest.arrays;
    expect(() => validateManifest({ ...validManifest, arrays: withoutNormals })).toThrow(/缺 normals.*重新导出/);
  });

  it("rejects a normals descriptor with an unexpected path or dtype", () => {
    expect(() => validateManifest({
      ...validManifest,
      arrays: { ...validManifest.arrays, normals: { ...validManifest.arrays.normals, path: "normals.f32.bin" } },
    })).toThrow();
    expect(() => validateManifest({
      ...validManifest,
      arrays: { ...validManifest.arrays, normals: { ...validManifest.arrays.normals, dtype: "float32", path: "normals.f32.bin", byte_length: 12 } },
    })).toThrow();
    expect(() => validateManifest({
      ...validManifest,
      arrays: { ...validManifest.arrays, normals: { ...validManifest.arrays.normals, byte_length: 4 } },
    })).toThrow();
  });

  it("rejects a world_to_view that is not sixteen finite numbers", () => {
    expect(() => validateManifest({ ...validManifest, world_to_view: [1, 0, 0] })).toThrow();
    expect(() => validateManifest({ ...validManifest, world_to_view: Array.from({ length: 16 }, () => "x") })).toThrow();
  });

  it("requires world_to_view to be a proper row-major rigid affine transform", () => {
    for (const matrix of [
      Array(16).fill(0),
      [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      [1, 0.25, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, 0, 0, 1],
    ]) expect(() => validateManifest({ ...validManifest, world_to_view: matrix })).toThrow(/world_to_view/);
  });

  it("requires six finite ordered display bounds", () => {
    expect(() => validateManifest({ ...validManifest, display_bounds: [1, 2, 3] })).toThrow(/display_bounds/);
    expect(() => validateManifest({ ...validManifest, display_bounds: [2, 0, 0, 1, 1, 1] })).toThrow(/display_bounds/);
  });

  it("rejects a manifest with an unsupported array dtype or component count", () => {
    expect(() => validateManifest({
      ...validManifest,
      arrays: { ...validManifest.arrays, positions: { ...validManifest.arrays.positions, dtype: "float64" } },
    })).toThrow();
    expect(() => validateManifest({
      ...validManifest,
      arrays: { ...validManifest.arrays, colors: { ...validManifest.arrays.colors, components: 4 } },
    })).toThrow();
  });

  it("validates every manifest source provenance field", () => {
    expect(validateManifest(validManifest).source.da3_cache).toMatchObject({ schema_version: 2, frame_count: 1 });
    expect(() => validateManifest({ ...validManifest, source: { ...validSource, export: { ...validSource.export, max_points: 0 } } })).toThrow();
    expect(() => validateManifest({ ...validManifest, source: { ...validSource, da3_cache: { ...validSource.da3_cache, image_ids: [7, 7] } } })).toThrow();
    expect(() => validateManifest({ ...validManifest, source: { ...validSource, footprint: { run_id: "A".repeat(32), status: "accepted" } } })).toThrow();
  });

  it("requires exact filter, exporter, mapping, and SAM3 provenance", () => {
    expect(validateManifest(validManifest).source.sam3_mask.schema).toBe("sam3_self_exemplar_processed_mask_cache_v1");
    const { global_mapping_sha256: _missing, ...withoutMappingDigest } = validSource.export;
    expect(() => validateManifest({ ...validManifest, source: { ...validSource, export: withoutMappingDigest } })).toThrow();
    expect(() => validateManifest({ ...validManifest, source: { ...validSource, export: { ...validSource.export, exporter_source_sha256: "not-a-digest" } } })).toThrow();
    expect(() => validateManifest({ ...validManifest, source: { ...validSource, sam3_mask: { ...validSource.sam3_mask, coordinate_space: "source_pixels" } } })).toThrow();
  });

  it("rejects a global object index with missing fields or inconsistent counts", () => {
    expect(validateObjectIndex(validObjects, 1)).toMatchObject({ "11": { total_count: 1 } });
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], total_count: 2 } }, 1)).toThrow();
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], instances: [{ image_id: 7 }] } }, 1)).toThrow();
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], instances: [{ ...validObjects["11"].instances[0], point_index_range: [0, 2] }] } }, 1)).toThrow(/out of bounds/i);
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], instances: [{ ...validObjects["11"].instances[0], point_index_range: [1, 0] }] } }, 1)).toThrow();
  });

  it("accepts conflict candidates in pre-ranked order", () => {
    const first = validObjects["11"].instances[0];
    const entry = {
      ...validObjects["11"],
      images: [7, 8, 9],
      objects: [3, 4, 5],
      active_count: 3,
      total_count: 3,
      instances: [
        { ...first, thumbnail: "thumbs/1_0.jpg", classification: { ...resolvedObservation(), confidence: 0.6 } },
        { ...first, image_id: 8, object_id: 4, point_index_range: [1, 2] as const, thumbnail: "thumbs/1_1.jpg", classification: { ...resolvedObservation(), sku_id: "B", sku_name: "产品B", confidence: 0.9 } },
        { ...first, image_id: 9, object_id: 5, point_index_range: [2, 3] as const, thumbnail: "thumbs/1_2.jpg", classification: { ...resolvedObservation(), confidence: 0.5 } },
      ],
      classification: {
        status: "conflict" as const,
        primary_sku_id: "A",
        candidates: [candidate("A", "产品A", 1.1, 2, 0.6), candidate("B", "产品B", 0.9, 1, 0.9)],
        metadata: pendingMetadata(),
      },
    };
    expect(validateObjectIndex({ "1": entry }, 3)["1"].classification.candidates.map((item) => item.sku_id)).toEqual(["A", "B"]);
  });

  it.each([
    ["candidate sum", candidate("A", "产品A", 0.8, 1, 0.9)],
    ["candidate support", candidate("A", "产品A", 0.9, 2, 0.9)],
    ["candidate max", candidate("A", "产品A", 0.9, 1, 0.8)],
    ["impossible candidate sum", candidate("A", "产品A", 1.1, 1, 0.9)],
  ])("rejects aggregate tampering of %s against resolved observations", (_label, candidateMutation) => {
    expect(() => validateObjectIndex({ "11": {
      ...validObjects["11"],
      classification: { ...resolvedAggregate(), candidates: [candidateMutation] },
    } }, 1)).toThrow(/classification/);
  });

  it("rejects a validly shaped aggregate whose candidate order contradicts observations", () => {
    const entry = {
      ...validObjects["11"],
      classification: {
        status: "conflict" as const,
        primary_sku_id: "B",
        candidates: [candidate("B", "产品B", 0.8, 1, 0.8), candidate("A", "产品A", 0.9, 1, 0.9)],
        metadata: pendingMetadata(),
      },
    };
    expect(() => validateObjectIndex({ "1": { ...entry, instances: [{ ...entry.instances[0], thumbnail: "thumbs/1_0.jpg" }] } }, 1)).toThrow(/classification/);
  });

  it.each([
    ["non-finite score", { candidates: [candidate("A", "产品A", Number.NaN, 1, 0.8)] }],
    ["negative score", { candidates: [candidate("A", "产品A", -0.1, 1, 0.8)] }],
    ["primary mismatch", { primary_sku_id: "B" }],
    ["duplicate candidate", { candidates: [candidate("A", "产品A", 1, 1, 1), candidate("A", "产品A", 0.5, 1, 0.5)] }],
    ["invalid metadata", { metadata: { ...pendingMetadata(), status: "resolved" } }],
  ])("rejects %s", (_label, mutation) => {
    const valid = { status: "resolved", primary_sku_id: "A", candidates: [candidate("A", "产品A", 1, 1, 1)], metadata: pendingMetadata() };
    expect(() => validateObjectIndex({ "1": { ...validObjects["11"], instances: [{ ...validObjects["11"].instances[0], thumbnail: "thumbs/1_0.jpg" }], classification: { ...valid, ...mutation } } }, 12)).toThrow();
  });

  it("accepts an unavailable observation with an unavailable aggregate", () => {
    const unavailable = { schema_version: "1.0.0", source: "personalcare", project_id: 51, status: "unavailable", reason: "invalid_bbox" };
    const result = validateObjectIndex({ "11": {
      ...validObjects["11"],
      instances: [{ ...validObjects["11"].instances[0], classification: unavailable }],
      classification: { status: "unavailable", primary_sku_id: null, candidates: [], metadata: pendingMetadata() },
    } }, 1);
    expect(result["11"].instances[0].classification).toMatchObject({ status: "unavailable", reason: "invalid_bbox" });
  });

  it.each([
    ["source other", { source: "other" }],
    ["project_id float", { project_id: 51.5 }],
    ["project_id boolean", { project_id: true }],
    ["project_id other", { project_id: 52 }],
    ["reason other", { reason: "other" }],
    ["reason empty", { reason: "" }],
    ["extra metadata key", { metadata: pendingMetadata() }],
    ["extra sku_id key", { sku_id: "A" }],
  ])("rejects unavailable observation %s", (_label, mutation) => {
    const unavailable = { schema_version: "1.0.0", source: "personalcare", project_id: 51, status: "unavailable", reason: "invalid_bbox", ...mutation };
    expect(() => validateObjectIndex({ "11": {
      ...validObjects["11"],
      instances: [{ ...validObjects["11"].instances[0], classification: unavailable }],
      classification: { status: "unavailable", primary_sku_id: null, candidates: [], metadata: pendingMetadata() },
    } }, 1)).toThrow();
  });

  it.each([
    ["missing reason", (({ reason: _reason, ...value }) => value)( { schema_version: "1.0.0", source: "personalcare", project_id: 51, status: "unavailable", reason: "invalid_bbox" })],
    ["missing schema", (({ schema_version: _schema, ...value }) => value)( { schema_version: "1.0.0", source: "personalcare", project_id: 51, status: "unavailable", reason: "invalid_bbox" })],
    ["missing source", (({ source: _source, ...value }) => value)( { schema_version: "1.0.0", source: "personalcare", project_id: 51, status: "unavailable", reason: "invalid_bbox" })],
    ["missing project_id", (({ project_id: _projectId, ...value }) => value)( { schema_version: "1.0.0", source: "personalcare", project_id: 51, status: "unavailable", reason: "invalid_bbox" })],
    ["missing status", (({ status: _status, ...value }) => value)( { schema_version: "1.0.0", source: "personalcare", project_id: 51, status: "unavailable", reason: "invalid_bbox" })],
  ])("rejects unavailable observation %s", (_label, unavailable) => {
    expect(() => validateObjectIndex({ "11": {
      ...validObjects["11"],
      instances: [{ ...validObjects["11"].instances[0], classification: unavailable }],
      classification: { status: "unavailable", primary_sku_id: null, candidates: [], metadata: pendingMetadata() },
    } }, 1)).toThrow();
  });

  it.each([
    ["below zero", -0.01],
    ["above one", 1.01],
    ["NaN", Number.NaN],
    ["Infinity", Number.POSITIVE_INFINITY],
    ["boolean", true],
  ])("rejects resolved observation confidence %s", (_label, confidence) => {
    expect(() => validateObjectIndex({ "11": {
      ...validObjects["11"],
      instances: [{ ...validObjects["11"].instances[0], classification: { ...resolvedObservation(), confidence } }],
    } }, 1)).toThrow(/confidence/);
  });

  it("rejects observation identity, exact keys, and metadata violations", () => {
    expect(() => validateObjectIndex({ "11": {
      ...validObjects["11"],
      instances: [{ ...validObjects["11"].instances[0], classification: { ...resolvedObservation(), source: "other" } }],
    } }, 1)).toThrow(/identity/);
    expect(() => validateObjectIndex({ "11": {
      ...validObjects["11"],
      instances: [{ ...validObjects["11"].instances[0], classification: { ...resolvedObservation(), project_id: true } }],
    } }, 1)).toThrow(/project_id/);
    expect(() => validateObjectIndex({ "11": {
      ...validObjects["11"],
      instances: [{ ...validObjects["11"].instances[0], classification: { ...resolvedObservation(), project_id: 51.5 } }],
    } }, 1)).toThrow(/project_id/);
    const { confidence: _missingConfidence, ...withoutConfidence } = resolvedObservation();
    expect(() => validateObjectIndex({ "11": {
      ...validObjects["11"],
      instances: [{ ...validObjects["11"].instances[0], classification: withoutConfidence }],
    } }, 1)).toThrow(/fields are invalid/);
    expect(() => validateObjectIndex({ "11": {
      ...validObjects["11"],
      instances: [{ ...validObjects["11"].instances[0], classification: { ...resolvedObservation(), extra: true } }],
    } }, 1)).toThrow(/fields are invalid/);
    expect(() => validateObjectIndex({ "11": {
      ...validObjects["11"],
      instances: [{ ...validObjects["11"].instances[0], classification: { ...resolvedObservation(), metadata: { ...pendingMetadata(), extra: null } } }],
    } }, 1)).toThrow(/metadata/);
    expect(() => validateObjectIndex({ "11": {
      ...validObjects["11"],
      instances: [{ ...validObjects["11"].instances[0], classification: { ...resolvedObservation(), metadata: { ...pendingMetadata(), status: "resolved" } } }],
    } }, 1)).toThrow(/metadata/);
  });

  it("rejects candidates outside deterministic order", () => {
    const classification = {
      status: "conflict",
      primary_sku_id: "B",
      candidates: [candidate("B", "产品B", 0.5, 1, 0.5), candidate("A", "产品A", 1.0, 1, 1.0)],
      metadata: pendingMetadata(),
    };
    expect(() => validateObjectIndex({ "1": { ...validObjects["11"], instances: [{ ...validObjects["11"].instances[0], thumbnail: "thumbs/1_0.jpg" }], classification } }, 12)).toThrow(/order/);
  });

  it("enforces resolved and unavailable candidate cardinality", () => {
    const instance = { ...validObjects["11"].instances[0], thumbnail: "thumbs/1_0.jpg" };
    expect(() => validateObjectIndex({ "1": {
      ...validObjects["11"],
      instances: [instance],
      classification: { ...resolvedAggregate(), candidates: [candidate("A", "产品A", 1, 1, 1), candidate("B", "产品B", 0.5, 1, 0.5)] },
    } }, 12)).toThrow();
    expect(() => validateObjectIndex({ "1": {
      ...validObjects["11"],
      instances: [instance],
      classification: { status: "unavailable", primary_sku_id: null, candidates: [candidate("A", "产品A", 1, 1, 1)], metadata: pendingMetadata() },
    } }, 12)).toThrow();
  });

  it("requires object and instance classification keys", () => {
    const { classification: _classification, ...withoutEntryClassification } = validObjects["11"];
    expect(() => validateObjectIndex({ "11": withoutEntryClassification }, 1)).toThrow();
    const { classification: _instanceClassification, ...withoutInstanceClassification } = validObjects["11"].instances[0];
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], instances: [withoutInstanceClassification] } }, 1)).toThrow();
  });

  it("binds thumbnails to their global-ID instance identity and forbids overlapping ranges", () => {
    expect(() => validateObjectIndex({ "11": { ...validObjects["11"], instances: [{ ...validObjects["11"].instances[0], thumbnail: "thumbs/12_0.jpg" }] } }, 1)).toThrow(/thumbnail/);
    expect(() => validateObjectIndex({
      ...validObjects,
      "12": { images: [8], objects: [4], active_count: 1, removed_count: 0, total_count: 1, instances: [{ image_id: 8, object_id: 4, bbox: [1, 2, 3, 4], removed: false, point_index_range: [0, 1], thumbnail: "thumbs/12_0.jpg", classification: resolvedObservation() }], classification: resolvedAggregate() },
    }, 1)).toThrow(/overlap/);
  });

  it("fails closed with a re-export hint when an instance thumbnail is missing", () => {
    const { thumbnail: _missing, ...withoutThumbnail } = validObjects["11"].instances[0];
    expect(() => validateObjectIndex({
      "11": { ...validObjects["11"], instances: [withoutThumbnail] },
    }, 1)).toThrow(/缺 instance thumbnail.*重新导出/);
  });

  it("rejects unsafe or malformed instance thumbnail paths", () => {
    for (const thumbnail of [
      "../escape.jpg",
      "thumbs/../escape.jpg",
      "/absolute/thumbs/11_0.jpg",
      "thumbs/11_0.png",
      "thumbs/11_x.jpg",
      "https://evil.test/thumbs/11_0.jpg",
    ]) {
      expect(() => validateObjectIndex({
        "11": { ...validObjects["11"], instances: [{ ...validObjects["11"].instances[0], thumbnail }] },
      }, 1)).toThrow(/thumbnail/);
    }
  });

  it("derives images and all sorted object IDs from instances", () => {
    const derived = {
      "11": {
        images: [7, 8],
        objects: [3, 3],
        active_count: 2,
        removed_count: 0,
        total_count: 2,
        instances: [
          { image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false, point_index_range: [0, 1], thumbnail: "thumbs/11_0.jpg", classification: resolvedObservation() },
          { image_id: 8, object_id: 3, bbox: [5, 6, 7, 8], removed: false, point_index_range: [1, 2], thumbnail: "thumbs/11_1.jpg", classification: resolvedObservation() },
        ],
        classification: { ...resolvedAggregate(), candidates: [candidate("A", "产品A", 1.8, 2, 0.9)] },
      },
    };
    expect(validateObjectIndex(derived, 2)["11"]).toMatchObject({ images: [7, 8], objects: [3, 3] });
    expect(() => validateObjectIndex({ "11": { ...derived["11"], images: [7] } }, 2)).toThrow();
    expect(() => validateObjectIndex({ "11": { ...derived["11"], objects: [3] } }, 2)).toThrow();
  });

  it("enforces accepted and rejected footprint value/geometry relations", () => {
    expect(validateFootprints(validFootprints).value_m2).toBe(1);
    expect(() => validateFootprints({ ...validFootprints, metric: "da3_ground_footprint_union" })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, rejection_reason: "unexpected" })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, per_global_id: {}, union: validFootprints.union })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, union: { ...validFootprints.union, properties: { ...validFootprints.union.properties, area_m2: 0.5 } } })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, per_global_id: { "11": { ...validFootprints.per_global_id["11"], properties: { coordinate_space: "local_support_plane_meters", global_id: "11", area_m2: 0.5 } } } })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, status: "rejected", value_m2: 0 })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, status: "accepted", value_m2: null })).toThrow();
    expect(validateFootprints({ ...validFootprints, status: "rejected", value_m2: null, rejection_reason: "input rejected", support_plane: null, union: null, per_global_id: {} }).status).toBe("rejected");
    expect(() => validateFootprints({ ...validFootprints, status: "rejected", value_m2: null, rejection_reason: null, support_plane: null, union: null, per_global_id: {} })).toThrow();
    expect(() => validateFootprints({ ...validFootprints, status: "rejected", value_m2: null, support_plane: null, union: validFootprints.union, per_global_id: {} })).toThrow();
  });
});
