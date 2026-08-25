import { describe, expect, it } from "vitest";
import type { ViewerBundle } from "./bundle-loader";
import { buildEvidenceView, canFocusGlobalId, entryHasGeometry, formatFormalMetric, listGlobalIds } from "./presentation";
import { dataCandidates } from "./data-candidates";

const runId = "a".repeat(32);
const square = [[0, 0], [1, 0], [1, 1], [0, 0]] as const;

function makeBundle(): ViewerBundle {
  return {
    current: { schema_version: "2.0.0", run_id: runId, complete: true },
    generationUrl: `https://example.test/data/runs/${runId}/`,
    manifest: {
      schema_version: "2.0.0", coordinate_space: "da3_world_meters", point_count: 0,
      display_bounds: [0, 0, 0, 0, 0, 0],
      arrays: {
        positions: { path: "positions.f32.bin", dtype: "float32", components: 3, byte_length: 0 },
        colors: { path: "colors.u8.bin", dtype: "uint8", components: 3, byte_length: 0 },
        normals: { path: "normals.i8.bin", dtype: "int8", components: 3, byte_length: 0 },
      },
      objects_path: "objects.json", footprints_path: "footprints.json",
      world_to_view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      coordinate_convention: "DA3 native CV coordinates; world_to_view maps to viewer Y-up (row-major)",
      source: {
        da3_cache: {
          schema_version: 2, source_model: "da3", affine_convention: "pixel_center_v1",
          preprocess_resolution: 2, preprocess_method: "upper_bound_resize", frame_count: 1,
          processed_size: [2, 2], image_ids: [7], source_image_sha256: ["0".repeat(64)],
        },
        footprint: { run_id: runId, status: "accepted" },
        sam3_mask: {
          schema: "sam3_self_exemplar_processed_mask_cache_v1",
          coordinate_space: "da3_processed_pixels",
          producer: "sku_matching",
        },
        export: {
          voxel_size_m: 0.01, max_points: 10,
          filter_config: { enabled: true, sor_nb_neighbors: 20, sor_std_ratio: 2, keep_main_clusters: true, cluster_eps_scale: 5, cluster_min_points: 10, min_cluster_ratio: 0.01, remove_ground: true, ground_dist_scale: 3, ground_min_inlier_ratio: 0.08, min_remaining_ratio: 0.2, min_points: 1000 },
          exporter_source_sha256: "1".repeat(64), global_mapping_sha256: "2".repeat(64),
        },
      },
      capabilities: { point_picking: false, footprint_picking: true, formal_ground_footprint: true },
    },
    objects: {
      "2": {
        images: [7], objects: [3], active_count: 1, removed_count: 0, total_count: 1,
        instances: [{ image_id: 7, object_id: 3, bbox: [1, 2, 3, 4], removed: false, point_index_range: [0, 0], thumbnail: "thumbs/2_0.jpg" }],
      },
      "11": {
        images: [8, 9], objects: [4, 5], active_count: 1, removed_count: 1, total_count: 2,
        instances: [
          { image_id: 8, object_id: 4, bbox: [10, 20, 30, 40], removed: false, point_index_range: [0, 2], thumbnail: "thumbs/11_0.jpg" },
          { image_id: 9, object_id: 5, bbox: [11, 21, 31, 41], removed: true, point_index_range: [2, 2], thumbnail: "thumbs/11_1.jpg" },
        ],
      },
    },
    footprints: {
      metric: "da3_self_exemplar_ground_footprint_union", unit: "m2", status: "accepted", value_m2: 3.5,
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
    positions: new Float32Array(), colors: new Uint8Array(), normals: new Int8Array(),
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

  it("maps every instance to an absolute thumbnail URL next to the generation", () => {
    const view = buildEvidenceView(makeBundle(), "11");
    expect(view?.instances).toEqual([
      { imageId: 8, objectId: 4, removed: false, thumbnailUrl: `https://example.test/data/runs/${runId}/thumbs/11_0.jpg` },
      { imageId: 9, objectId: 5, removed: true, thumbnailUrl: `https://example.test/data/runs/${runId}/thumbs/11_1.jpg` },
    ]);
    expect(buildEvidenceView(makeBundle(), "2")?.instances).toEqual([
      { imageId: 7, objectId: 3, removed: false, thumbnailUrl: `https://example.test/data/runs/${runId}/thumbs/2_0.jpg` },
    ]);
  });

  it("keeps evidence thumbnail URLs absolute when using the default data root", () => {
    const dataRoot = dataCandidates("https://example.test/review/")[0];
    const bundle = { ...makeBundle(), generationUrl: `${dataRoot}runs/${runId}/` };
    expect(buildEvidenceView(bundle, "2")?.instances[0].thumbnailUrl).toBe(`https://example.test/review/data/runs/${runId}/thumbs/2_0.jpg`);
  });

  it("derives hasGeometry from the instance point ranges", () => {
    const bundle = makeBundle();
    expect(entryHasGeometry(bundle.objects["11"])).toBe(true);
    expect(entryHasGeometry(bundle.objects["2"])).toBe(false);
    expect(buildEvidenceView(bundle, "11")?.hasGeometry).toBe(true);
    expect(buildEvidenceView(bundle, "2")?.hasGeometry).toBe(false);
    const flipped = {
      ...bundle,
      objects: {
        ...bundle.objects,
        "11": {
          ...bundle.objects["11"],
          instances: bundle.objects["11"].instances.map((instance, index) => (
            index === 0
              ? { ...instance, point_index_range: [0, 0] as const }
              : { ...instance, point_index_range: [0, 4] as const }
          )),
        },
      },
    };
    expect(entryHasGeometry(flipped.objects["11"])).toBe(true);
  });

  it("disables focus only when an ID has neither footprint nor points", () => {
    const bundle = makeBundle();
    expect(canFocusGlobalId(bundle.objects["11"], bundle.footprints, "11")).toBe(true);
    expect(canFocusGlobalId(bundle.objects["2"], bundle.footprints, "2")).toBe(false);
  });
});
