# DA3 ground-stack footprint — design

## Status and decision

Rick approved this design on 2026-08-12. It supersedes
`2026-08-11-ground-stack-bbox-area-v1-design.md`: the required metric is the
**footprint union of all detected cartons on their supporting plane**, not a
sum of visible/front-facing bbox areas.

For every detected carton, project its recovered 3D volume vertically onto the
supporting tabletop/floor plane. The total is the area of the 2D polygon union.
An upper carton contributes its full projection even when it does not touch the
plane. Detection or geometry that cannot form a reliable footprint rejects the
whole total; the stage must not silently publish a partial total as complete.

## Goal

Given an already prepared dataset and its DA3 metric cache, calculate one
auditable metric footprint in square metres:

```text
footprint_m2 = area(union(footprint_polygon(global_id) for every global_id))
```

The stage is read-only with respect to images, detection JSON, DA3 cache, and
`global_mapping.json`. It does not infer hidden cartons that are not detected.
No physical reference object is a runtime input because the selected DA3 model
supplies metric `world_points`; a reference remains optional deployment QA.

## Inputs and coordinate contract

```text
dataset/
  images/<0-based image id>.<jpg|jpeg|png>
  detections_results/<0-based image id>.json
<save_root>/<dataset>/
  da3_cache/predictions.npz
  dedup_detections/global_mapping.json
```

The DA3 cache must contain aligned `(N, H, W)` / `(N, H, W, 3)` arrays:
`world_points_conf`, `world_points`, `image_ids`, `source_image_sizes`, and
`source_to_processed_affine`, plus cache schema, model ID, ordered source-image
SHA-256 values, and the pixel-centre affine convention. `source_image_sizes`
and source-image hashes must equal the current inputs. `world_points` are a
shared DA3 world coordinate system, so points belonging to one `global_id` can
be fused across frames. A cache predating this contract is rejected and must be
rebuilt; no legacy affine interpretation is guessed.

`global_mapping.json` is the physical-identity source. Every observation in a
global-ID list is a view of one carton, not an extra carton.

## Required pipeline

### 1. Object masks and cache alignment

Flatten every detection JSON using the same stable enumeration as deduplication.
The `(image_id, object_id)` set in `global_mapping.json` must equal this full
detection set exactly. For every detection, load the original image and
generate a SAM3 box-prompt mask clipped to that bbox. SAM3 is mandatory: a
raw-bbox fallback is prohibited because it includes tabletop and neighbouring
cartons. Mapping observations use their corresponding masks as object points;
all detections' masks, dilated by two DA3-grid pixels, are excluded from
background plane fitting.

Warp each original-image boolean mask to DA3's `(H, W)` grid with the cached
original-pixel-to-processed affine and nearest-neighbour sampling. The affine
uses pixel centres: `x' = sx * x + (sx - 1) / 2 - crop_left` (and the same for
`y`). Reject an observation when its warped mask is empty, has a wrong shape,
or its source image/cache metadata is inconsistent. Keep only finite, nonzero
`world_points` with `world_points_conf >= 1.0`.

### 2. Supporting-plane RANSAC

Build the candidate background set from all valid DA3 points outside the
dilated masks, sampling each valid frame equally before the deterministic
50,000-point cap. Extract up to five planes by deterministic RANSAC (seed 13,
12-mm threshold, non-collinear triples); compute the RANSAC trial count from
the observed inlier rate for 0.999 success probability and cap it at 10,000.
Refine each candidate with SVD over full inliers. This avoids treating the
largest background plane as the table: a wall or floor can be larger.

Choose a support candidate only if it has at least 10,000 background inliers,
at least 10% background inlier fraction, P95 residual at most 10 mm, support
in at least three frames and 30% of valid frames, two in-plane spans of at
least 0.30 m, and a 2D inlier hull of at least 0.25 m². Orient its normal so
that fused carton points have positive median signed height; require at least
95% of object points to be no lower than -12 mm, a lowest-object P01 height no
greater than 80 mm, and at least 80% of object centres inside the table hull
buffered by 150 mm. If a differently oriented candidate is within 5% of the
best score, reject the run as support-plane ambiguous. These gates prevent a
wall, a small patch, or noisy depth from being misreported as the tabletop.

Construct deterministic orthonormal in-plane axes `(u, v)`. A world point `p`
maps to plane coordinates `(dot(p - p0, u), dot(p - p0, v))`; discarding the
normal component is precisely the required vertical projection.

### 3. One footprint polygon per physical carton

For one global ID, merge all valid masked points from all of its observations.
Remove points within 15 mm of the support plane so table leakage inside a mask
cannot enlarge a carton. Require at least 32 valid points for each observation
and 64 after fusion. Project remaining points to `(u, v)`, voxelise to a 5-mm
grid so a high-density view cannot dominate, and require one density-connected
component (`eps=20 mm`, `min_samples=4`). A second component with at least 20%
of non-noise points **or** 32 points is an identity/pose inconsistency and
rejects the global ID; it is never silently discarded.

Cartons are the defined object class, so reconstruct their base as a robust
oriented bounding box (OBB): find the minimum-area rectangle of the cleaned
2D hull and use that rectangle as the carton footprint polygon. Its two side
lengths must both be at least 50 mm, its area must be finite and positive, and
the source must contain at least 64 valid 3D points. A line-like front face
without enough side/top evidence fails this gate instead of inventing depth.

This differs from the discarded previous pipeline: it deliberately fuses all
observations of one `global_id`, then emits one polygon. It does not select a
single best frame or add visible surface areas.

### 4. Footprint union and outcome

Snap validated OBBs to a 0.1-mm precision grid and use Shapely's planar
`unary_union` on every accepted carton OBB. The union's area is the only
`value_m2`; individual OBB areas are diagnostics and must never be
arithmetically summed as the total. Repeat at a 1-mm grid and reject if the
areas differ by more than `max(0.5%, 1e-4 m²)`. Overlap is counted once;
vertical stacking is irrelevant after projection; overhang is included because
every carton is projected independently.

Any rejected global ID makes the report `rejected`, `value_m2: null`, and
preserves its observation diagnostics. There is no partial-success total.

## CLI, modules, and artifacts

Replace the incorrect `da3_metric` and `calibrated_bbox` area modes with one
concise command:

```bash
uv run python main.py --mode ground-stack-area \
  --dataset <dataset> --save_root <save_root>
```

Remove anchor CLI arguments and their front-facing bbox utilities/stages. The
new implementation owns these boundaries:

```text
main.py
  -> modules/da3_footprint_stage.py        # files, SAM3 calls, report/artifacts
  -> utils/ground_stack_footprint.py       # pure RANSAC, projection, OBB, union
```

Write only new artifacts under
`<save_root>/<dataset>/ground_stack_footprint/`:

```text
measurement_report.json     # result, plane, gates, per-global-ID diagnostics
footprints.geojson          # plane-coordinate OBBs and their union
top_down_footprint.png      # review image of individual OBBs and union
```

The report metric is `da3_ground_footprint_union`, unit is `m2`, and its
status is only `accepted` or `rejected`.

## Tests and acceptance criteria

Run tests through the existing `/home/xingyu/3D_Recognization/code/.venv` with
`uv run --active --no-project pytest`; no worktree-specific environment may be
created.

Focused tests must prove:

1. deterministic RANSAC recovers a known metric support plane and rejects an
   insufficient/nonplanar background set;
2. synthetic masked points for two overlapping cartons produce the known OBB
   union area rather than the sum of their areas;
3. repeated views of one global ID yield exactly one footprint polygon;
4. an upper carton contributes to the union despite its points being far from
   the support plane;
5. a line-like point set, empty mask, stale source size, or missing global-ID
   geometry rejects the total with observation-level diagnostics;
6. the stage leaves all four input artifact classes byte-for-byte unchanged;
7. the real `fd_area_test.mp4` preparation produces an accepted/rejected,
   reviewable report and top-down artifact without using a reference object.

README files must state that the metric is a projected occupied-table area,
not package surface, front-face area, or actual contact-only area.
