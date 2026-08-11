# Ground-stack calibrated bbox area sum — V1 design

## Status and decision

Approved for implementation on `feat/ground-stack-bbox-area` after review of this document.

The user-defined metric is **the sum of the detected object bounding-box areas in a ground-stack**, expressed in `cm²` and `m²`. This is a bbox-equivalent display-area metric. It is not a ground footprint, a union of overlapping boxes, a segmentation-mask area, or a package surface area.

## Goal

Given a prepared video dataset (`images/`, `detections_results/`) and the existing cross-frame `global_mapping.json`, calculate the calibrated bbox area of each physical object exactly once and report their sum.

The user supplies one detected anchor object and its known physical front-face width and height. The anchor establishes a planar image-to-physical-area calibration. The measurement is valid only when the target boxes are approximately on the anchor's front-facing plane.

## Definitions

For a global ID `g`, select one accepted observation `o_g` from its list of `(image_id, object_id, bbox)` entries. Let `Q` be the anchor's projective transform from pixels to centimetres. The per-instance and total metrics are:

```text
A_g_cm2 = polygon_area(Q(corners(bbox(o_g))))
A_total_cm2 = sum(A_g_cm2 for each accepted global ID g)
A_total_m2 = A_total_cm2 / 10_000
```

This is an arithmetic sum: two boxes are two physical products and each contributes its own bbox-equivalent area, even if their projected boxes overlap.

`global_mapping.json` is the source of physical identity. A global ID is counted once; repeated observations across frames are evidence, not additional area.

## Inputs

The new module consumes existing artifacts and does not trigger detection, reconstruction, matching, or deduplication itself:

```text
dataset/
  images/<0-based frame id>.<jpg|jpeg|png>
  detections_results/<0-based frame id>.json
output/<dataset>/dedup_detections/global_mapping.json
```

The CLI request supplies:

```text
--mode ground-stack-area
--dataset <dataset directory>
--save_root <output directory>
--area-anchor-frame <0-based frame id>
--area-anchor-object <0-based object index>
--area-anchor-width-cm <positive float>
--area-anchor-height-cm <positive float>
```

The anchor box is retrieved from `detections_results/<frame>.json`; it is not duplicated in CLI arguments. The request fails if its index is absent, malformed, or not represented in the mapping.

## Calibration and instance selection

1. Read the anchor bbox `(x1, y1, x2, y2)` and derive independent axis-aligned centimetre-per-pixel scales from its known width and height.
2. For every global ID, only consider observations from `anchor_frame`; this prevents camera motion or perspective changes in another frame from corrupting the anchor calibration.
3. Convert the selected bbox pixel area with those two scales. A global ID without a valid anchor-frame observation is rejected.
4. Reject, rather than silently repair, invalid boxes, non-finite calibration results, non-positive mapped area, or a global ID for which no valid observation remains.

The report records whether an instance was accepted or rejected and why. It makes no claim about boxes never detected in the video.

## Quality gates and limitations

The measurement is accepted only when all of these conditions hold:

- Anchor width and height are finite and positive.
- Anchor and candidate bboxes are finite, have `x2 > x1` and `y2 > y1`, and lie within the source image bounds after clipping.
- The anchor is a user-confirmed front view of a known-size package.
- The operator attests that measured boxes are approximately coplanar with that anchor. V1 does not apply depth correction.
- At least one unique global ID has an accepted observation.

The report must set `status: "rejected"` and `value_cm2/value_m2: null` if calibration fails or no accepted global ID exists. Per-instance rejections may still produce an `accepted_with_warnings` report if at least one valid instance remains. The response must expose `accepted_global_ids`, `rejected_global_ids`, and their reasons.

V1 explicitly does **not** estimate hidden objects, merge overlapping bboxes into a union silhouette, infer ground footprint, or treat bboxes as object masks. A future V2 may use DA3 depth per box when the pile is not planar; it is out of scope because GPU inference is currently unavailable and needs separate accuracy validation.

## Outputs

All outputs are new artifacts under `Output/<dataset>/ground_stack_area/`. Existing detections and `global_mapping.json` remain byte-for-byte unchanged.

```text
ground_stack_area/
  measurement_report.json
  selected_instances.json
  annotated_frames/
    <frame-id>.jpg
```

`measurement_report.json` has this stable top-level schema:

```json
{
  "schema_version": "1.0",
  "status": "accepted",
  "metric": "calibrated_bbox_area_sum",
  "unit": {"instance": "cm2", "total": "m2"},
  "value_cm2": 23040.0,
  "value_m2": 2.304,
  "accepted_global_ids": 30,
  "rejected_global_ids": 2,
  "calibration": {
    "anchor_frame": 0,
    "anchor_object": 3,
    "anchor_width_cm": 32.0,
    "anchor_height_cm": 24.0,
    "method": "axis_aligned_bbox_scale_same_frame"
  },
  "warnings": [],
  "artifacts": {
    "instances": "selected_instances.json",
    "annotated_frames_dir": "annotated_frames"
  }
}
```

`selected_instances.json` contains one record per global ID, its selected source frame/object/bbox, `area_cm2`, source pixel area, and any rejection reason.

## Architecture

The implementation remains independent from the DA3 front-facing-area prototype:

```text
main.py (--mode ground-stack-area)
  -> modules/ground_stack_area_stage.py       # artifact paths and report writes
  -> utils/ground_stack_area.py               # pure validation, calibration, selection, area math
  -> existing deduplicate_detections loader   # read-only mapping/detection contracts
```

`utils/ground_stack_area.py` is pure NumPy/Python and has no model, GPU, file-write, or CLI dependency. `modules/ground_stack_area_stage.py` owns JSON/image artifacts. `main.py` only parses arguments and dispatches the new mode. `config.yaml` gains an isolated `ground_stack_area` section; it must not reuse `facing_area` defaults or matching-backend defaults.

## Tests and acceptance criteria

Focused tests run with `uv run pytest` from `code/` and cover:

1. Identity calibration: an anchor with known `20 cm × 10 cm` maps itself to `200 cm²`.
2. Multiple global IDs sum once each despite multiple observations per ID.
3. Largest valid observation is selected deterministically.
4. Invalid/inverted/non-finite bboxes and invalid anchor dimensions are rejected with an explicit reason.
5. The stage does not modify its input `global_mapping.json` or detection JSON.
6. Report schema, units, and `cm² -> m²` conversion are correct.

The focused test suite must pass before handoff. GPU end-to-end validation is deferred until the host exposes a usable CUDA driver and adequate free disk; that limitation will be documented rather than hidden by a CPU fallback.

## Documentation

The root README and `code/README.md` will document the metric definition, planar assumption, CLI inputs, outputs, and the distinction from actual package/mask/footprint area.
