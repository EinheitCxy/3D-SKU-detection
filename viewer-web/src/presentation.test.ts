import { describe, expect, it } from "vitest";
import type { ViewerBundle } from "./bundle-loader";
import { buildEvidenceView, formatFormalMetric, listGlobalIds } from "./presentation";

const runId = "a".repeat(32);
const square = [[0, 0], [1, 0], [1, 1], [0, 0]] as const;

function makeBundle(): ViewerBundle {
  return {
    current: { schema_version: "1.0.0", run_id: runId, complete: true },
    manifest: {
      schema_version: "1.0.0", coordinate_space: "da3_world_meters", point_count: 0,
      arrays: {
        positions: { path: "positions.f32.bin", dtype: "float32", components: 3, byte_length: 0 },
        colors: { path: "colors.u8.bin", dtype: "uint8", components: 3, byte_length: 0 },
        confidences: { path: "confidences.f32.bin", dtype: "float32", components: 1, byte_length: 0 },
        frame_ids: { path: "frame_ids.i32.bin", dtype: "int32", components: 1, byte_length: 0 },
      },
      objects_path: "objects.json", footprints_path: "footprints.json",
      source: {
        da3_cache: {
          schema_version: 2, source_model: "da3", affine_convention: "pixel_center_v1",
          preprocess_resolution: 2, preprocess_method: "upper_bound_resize", frame_count: 1,
          processed_size: [2, 2], image_ids: [7], source_image_sha256: ["0".repeat(64)],
        },
        footprint: { run_id: runId, status: "accepted" }, export: { voxel_size_m: 0.01, max_points: 10 },
      },
      capabilities: { point_picking: false, footprint_picking: true, formal_ground_footprint: true },
    },
    objects: {
      "2": {
        images: [7], objects: [3], active_count: 1, removed_count: 0, total_count: 1,
        instances: [{ image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false }],
      },
      "11": {
        images: [8, 9], objects: [4, 5], active_count: 1, removed_count: 1, total_count: 2,
        instances: [
          { image_id: 8, object_id: 4, bbox: [10, 20, 30, 40], removed: false },
          { image_id: 9, object_id: 5, bbox: [11, 21, 31, 41], removed: true },
        ],
      },
    },
    footprints: {
      metric: "da3_ground_footprint_union", unit: "m2", status: "accepted", value_m2: 3.5,
      rejection_reason: null, run_id: runId,
      support_plane: { point: [0, 0, 0], u_axis: [1, 0, 0], v_axis: [0, 1, 0], normal: [0, 0, 1] },
      per_global_id: {
        "11": {
          rings: [[square]],
          properties: { coordinate_space: "local_support_plane_meters", global_id: "11", area_m2: 1.25, observations_used: 2 },
        },
      },
      union: {
        rings: [[square]],
        properties: { coordinate_space: "local_support_plane_meters", global_id: "union", area_m2: 3.5 },
      },
    },
    positions: new Float32Array(), colors: new Uint8Array(), confidences: new Float32Array(), frameIds: new Int32Array(),
  };
}

describe("presentation", () => {
  it("formats accepted formal values in fixed square metres and hides unavailable values", () => {
    const accepted = makeBundle().footprints;
    expect(formatFormalMetric(accepted)).toBe("3.50 m²");
    expect(formatFormalMetric({ ...accepted, status: "rejected", value_m2: null, rejection_reason: "support plane invalid" })).toBe("—");
  });

  it("lists numeric global IDs without admitting the union outline as an SKU", () => {
    const bundle = makeBundle();
    expect(listGlobalIds(bundle.objects)).toEqual(["2", "11"]);
    expect(buildEvidenceView(bundle, "union")).toBeNull();
  });

  it("returns object evidence while marking a missing per-ID footprint unavailable", () => {
    const bundle = makeBundle();
    expect(buildEvidenceView(bundle, "11")).toMatchObject({
      globalId: "11", footprint: { available: true, areaM2: 1.25, observationsUsed: 2 },
      object: { active_count: 1, removed_count: 1 },
    });
    expect(buildEvidenceView(bundle, "2")).toMatchObject({
      globalId: "2", footprint: { available: false, areaM2: null, observationsUsed: null },
    });
  });
});
