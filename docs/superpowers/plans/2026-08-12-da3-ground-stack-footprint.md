# DA3 Ground-Stack Footprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incorrect visible-bbox area metric with an auditable DA3 metric union of every detected carton’s projection onto its supporting plane.

**Architecture:** `utils/ground_stack_footprint.py` holds deterministic, model-free geometry: support-plane RANSAC, stable plane coordinates, robust carton OBB recovery, and polygon union. `modules/da3_footprint_stage.py` validates DA3 cache/image alignment, obtains mandatory SAM3 masks, forms background/object point clouds, and writes reports and visual artifacts. `main.py` exposes only this footprint calculation through `ground-stack-area`.

**Tech Stack:** Python 3.11; NumPy; SciPy `ConvexHull`; scikit-learn `DBSCAN`; direct `shapely>=2,<3` dependency with `set_precision`/`unary_union`; OpenCV mask warping; Matplotlib review image; existing DA3 cache and local SAM3 checkpoint.

## Progress checkpoint — 2026-08-12

- Task 1 is accepted: deterministic support-plane/OBB/union geometry, with
  ambiguous-component and RANSAC boundary coverage.
- Task 2 is accepted: schema-v2 DA3 provenance, strict SAM3/cache/mapping
  contracts, table-candidate rejection gates, 2-D footprint fusion, and
  rejected-state artifacts. Its final focused validation was 38 passed.
- Task 3 is intentionally paused at its RED checkpoint. Two new CLI contract
  tests state that `ground-stack-area` must call only `run_da3_footprint` and
  reject the removed `--area-mode` flag. Production CLI replacement, deletion
  of old bbox-area files, README/config updates, and real-video validation are
  still pending. The first attempted RED command used a path relative to the
  wrong working directory and executed zero tests; it is not evidence of a
  passing or failing result.

## Global Constraints

- Metric definition is the m² area of the polygon union after every detected carton is projected along the fitted support-plane normal; never sum carton areas.
- A carton is identified by one `global_id`; fuse every valid masked observation for that ID and emit exactly one OBB polygon.
- SAM3 masks are mandatory; raw bbox point extraction and any fallback to the old visible/front-facing/calibrated-bbox area are prohibited.
- Reuse only `/home/xingyu/3D_Recognization/code/.venv`, `/home/xingyu/3D_Recognization/Depth-Anything-3/.venv`, and `/home/xingyu/3D_Recognization/.venv`; do not create a worktree-specific environment.
- DA3 cache must validate `world_points`, `world_points_conf`, `image_ids`, `source_image_sizes`, `source_to_processed_affine`, schema version, model ID, pixel-centre affine convention, and ordered source-image SHA-256 values; current images must match the cached sizes and hashes. Do not interpret legacy cache variants.
- Mapping observations must equal the complete flattened detection-object set. SAM3 masks every detection; mapping masks define object points and all masks dilated two DA3 pixels define background exclusion.
- RANSAC uses seed `13`, at most `50_000` frame-balanced candidate points, `0.012 m` threshold, adaptive 0.999-success trials capped at 10,000, and extracts up to five candidate planes. A selected support must meet inlier/residual/frame-span/table-hull/object-height/ambiguity gates specified in the design.
- Valid 3D object points are finite, nonzero, and `world_points_conf >= 1.0`; points within `0.015 m` of the fitted support plane are table leakage and excluded from carton fitting.
- A carton needs `>=32` valid points in every accepted observation, `>=64` after fusion, 5-mm voxel balancing, and no second DBSCAN component containing `>=20%` or `>=32` non-noise points. Its OBB needs two finite side lengths each `>=0.05 m` and positive finite area. Any missing/rejected global ID makes the total `rejected` and `value_m2: null`.
- Input images, detections, DA3 NPZ cache, and `global_mapping.json` remain byte-for-byte unchanged. Runtime artifacts must not be staged.
- Output only `<save_root>/<dataset>/ground_stack_footprint/{measurement_report.json,footprints.geojson,top_down_footprint.png}`. Report metric is `da3_ground_footprint_union`, unit `m2`, and status is only `accepted` or `rejected`.
- Remove the obsolete `da3_metric`/`calibrated_bbox` CLI modes, anchor CLI options, old bbox-area stage/utilities/tests, and their README/config claims.
- All Python test commands use `uv run --active --no-project` with the existing core `VIRTUAL_ENV` and offline cache.

---

## File Structure

| Path | Responsibility |
|---|---|
| `code/utils/ground_stack_footprint.py` | Pure metric geometry and explicit `FootprintError` failures. |
| `code/tests/test_ground_stack_footprint.py` | Deterministic synthetic geometry acceptance/rejection tests. |
| `code/modules/da3_footprint_stage.py` | DA3/SAM3 IO boundary, cache/image contracts, support-plane selection, output artifacts. |
| `code/tests/test_da3_footprint_stage.py` | Stage test fixtures with a monkeypatched SAM3 boundary. |
| `code/modules/da3_runner.py` | Writes cache schema/provenance and pixel-centre affine. |
| `code/main.py` | Single `ground-stack-area` dispatch, no area-mode/anchor API. |
| `README.md`, `code/README.md`, `code/config.yaml` | Correct footprint definition and command. |

### Task 1: Deterministic support-plane and footprint geometry

**Files:**
- Create: `code/utils/ground_stack_footprint.py`
- Create: `code/tests/test_ground_stack_footprint.py`

**Interfaces:**
- Produces `SupportPlane(point: np.ndarray, normal: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray, inlier_count: int, inlier_fraction: float, p95_residual_m: float)`.
- Produces `fit_support_plane(background_points: np.ndarray) -> SupportPlane`.
- Produces `carton_footprint_polygon(points: np.ndarray, plane: SupportPlane) -> tuple[Polygon, dict[str, float | int]]`.
- Produces `union_footprints(polygons: list[Polygon]) -> Polygon | MultiPolygon`.
- Raises `FootprintError` for every quality-gate failure; functions contain no file/model/CLI code.

- [ ] **Step 1: Write failing tests for support-plane recovery and rejection**

```python
def test_fit_support_plane_recovers_metric_table_with_outliers():
    table = make_plane_grid(point=np.array([0.0, 0.0, 1.0]), normal=np.array([0.0, 0.0, 1.0]))
    plane = fit_support_plane(np.vstack([table, random_outliers]))
    assert abs(np.dot(plane.normal, [0.0, 0.0, 1.0])) == pytest.approx(1.0, abs=1e-3)
    assert plane.inlier_count >= 10_000

def test_fit_support_plane_rejects_insufficient_background():
    with pytest.raises(FootprintError, match="support plane"):
        fit_support_plane(np.zeros((9_999, 3)))
```

- [ ] **Step 2: Run the new tests and verify they fail because the geometry module does not exist**

Run: `VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/fd-area-code-uv-cache UV_OFFLINE=1 uv run --active --no-project pytest -q tests/test_ground_stack_footprint.py -k support_plane`

Expected: import failure for `utils.ground_stack_footprint`.

- [ ] **Step 3: Implement deterministic plane fitting**

```python
def fit_support_plane(background_points: np.ndarray) -> SupportPlane:
    points = _validate_points(background_points, minimum=10_000, label="background")
    sampled = _deterministic_subsample(points, maximum=50_000, seed=13)
    candidate = _best_ransac_plane(sampled, trials=2_048, threshold_m=0.012, seed=13)
    full_distances = np.abs((points - candidate.point) @ candidate.normal)
    inliers = points[full_distances <= 0.012]
    return _refine_support_plane(inliers, total_points=len(points))
```

`_refine_support_plane` must run SVD, reject fewer than 10,000 inliers or fraction below 0.10, reject p95 residual above 0.012m, and generate deterministic in-plane axes by choosing the least-parallel world axis before cross products.

- [ ] **Step 4: Run support-plane tests and verify they pass**

Run the Step 2 command. Expected: both tests pass.

- [ ] **Step 5: Write failing tests for OBB recovery, multi-view fusion, upper cartons, union, and degenerate points**

```python
def test_overlapping_carton_footprints_use_union_not_sum():
    plane = horizontal_support_plane()
    first = carton_points(x_range=(0, 1), y_range=(0, 1), height=0.2)
    upper = carton_points(x_range=(0.5, 1.5), y_range=(0, 1), height=0.8)
    first_polygon, _ = carton_footprint_polygon(first, plane)
    upper_polygon, _ = carton_footprint_polygon(upper, plane)
    assert union_footprints([first_polygon, upper_polygon]).area == pytest.approx(1.5)

def test_line_like_carton_points_are_rejected():
    with pytest.raises(FootprintError, match="OBB"):
        carton_footprint_polygon(line_points, horizontal_support_plane())
```

The fixture must include at least 64 points per non-degenerate carton and use one carton’s points from two sampled faces to demonstrate that concatenating views still returns one polygon.

- [ ] **Step 6: Run OBB/union tests and verify they fail for missing functions**

Run: `VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/fd-area-code-uv-cache UV_OFFLINE=1 uv run --active --no-project pytest -q tests/test_ground_stack_footprint.py -k "footprint or union or line_like"`

Expected: function import/attribute failure.

- [ ] **Step 7: Implement object projection, robust OBB, and union**

```python
def carton_footprint_polygon(points: np.ndarray, plane: SupportPlane) -> tuple[Polygon, dict[str, float | int]]:
    object_points = _validate_points(points, minimum=64, label="carton")
    heights = (object_points - plane.point) @ plane.normal
    object_points = object_points[heights > 0.015]
    projected = project_to_plane(object_points, plane)
    component = _largest_density_component(projected, eps_m=0.03, min_samples=8)
    cleaned = _trim_projected_outliers(component, lower=0.01, upper=0.99)
    polygon = MultiPoint(cleaned).convex_hull.minimum_rotated_rectangle
    return _validate_obb_polygon(polygon, len(object_points))

def union_footprints(polygons: list[Polygon]) -> Polygon | MultiPolygon:
    union = unary_union(polygons)
    if union.is_empty or not np.isfinite(union.area) or union.area <= 0:
        raise FootprintError("footprint union is not positive and finite")
    return union
```

`_largest_density_component` uses `DBSCAN`, discards label `-1`, and deterministically chooses the greatest population (then smallest label); `_trim_projected_outliers` uses independent 1st--99th percentile bounds and must reject fewer than three points.

- [ ] **Step 8: Run all geometry tests and commit**

Run: `VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/fd-area-code-uv-cache UV_OFFLINE=1 uv run --active --no-project pytest -q tests/test_ground_stack_footprint.py`

Expected: PASS. Commit only the two task files with `feat: add ground footprint geometry`.

### Task 2: DA3/SAM3 footprint stage and review artifacts

**Files:**
- Create: `code/modules/da3_footprint_stage.py`
- Create: `code/tests/test_da3_footprint_stage.py`
- Modify: `code/utils/ground_stack_footprint.py`
- Modify: `code/tests/test_ground_stack_footprint.py`
- Modify: `code/modules/da3_runner.py`
- Modify: `code/pyproject.toml`
- Modify: `code/uv.lock`

**Interfaces:**
- Consumes Task 1 `SupportPlane`, `fit_support_plane`, `carton_footprint_polygon`, `union_footprints`, and `FootprintError`.
- Produces `run_da3_footprint(dataset_path: str, save_root: Path) -> dict[str, object]` with keys `success`, `status`, `report_path`.
- Stage output files are exactly `measurement_report.json`, `footprints.geojson`, and `top_down_footprint.png` under `ground_stack_footprint/`.

- [ ] **Step 1: Write failing stage tests around an injected SAM3 boundary**

```python
def test_stage_fuses_global_id_views_and_uses_polygon_union(monkeypatch, tmp_path):
    dataset, save_root, input_paths = make_metric_fixture(tmp_path)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)
    before = {path: path.read_bytes() for path in input_paths}
    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["metric"] == "da3_ground_footprint_union"
    assert report["value_m2"] == pytest.approx(1.5)
    assert report["per_global_id"]["1"]["observations_used"] == 2
    assert all(path.read_bytes() == content for path, content in before.items())

def test_stage_rejects_complete_total_when_one_global_id_has_no_mask(monkeypatch, tmp_path):
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", masks_with_one_empty)
    report = run_and_load_report(tmp_path)
    assert report["status"] == "rejected"
    assert report["value_m2"] is None
```

The metric fixture must write a schema-v2 `world_points` cache with at least
10,000 unmasked table points, two overlapping carton point sets (one at greater
height), exact pixel-centre affine/source-size/hash metadata, and a two-frame
repeated `global_id`. Add failure tests for stale source hash, legacy cache
schema, incomplete mapping-versus-detection coverage, and an empty SAM3 mask.

- [ ] **Step 2: Run tests and verify the stage import fails**

Run: `VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/fd-area-code-uv-cache UV_OFFLINE=1 uv run --active --no-project pytest -q tests/test_da3_footprint_stage.py`

Expected: import failure for `modules.da3_footprint_stage`.

- [ ] **Step 3: Implement cache validation and exact SAM3-to-DA3 mask warp**

```python
def _warp_mask_to_da3_grid(mask: np.ndarray, affine: np.ndarray, height: int, width: int) -> np.ndarray:
    warped = cv2.warpAffine(mask.astype(np.uint8), affine.astype(np.float32), (width, height), flags=cv2.INTER_NEAREST)
    return warped.astype(bool)

def _valid_points(points: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) > 0) & np.isfinite(confidence) & (confidence >= 1.0)
```

Add `shapely>=2,<3` as a direct project dependency and regenerate `uv.lock`
offline. Update the DA3 runner to write schema-v2 cache provenance: a safe
string model ID, per-frame source SHA-256, `pixel_center_v1` affine convention,
and the exact `x' = sx*x + (sx-1)/2-crop_left` affine. Test its affine without
running DA3 inference. Load every stage cache field with `allow_pickle=False`.
Validate image IDs are unique, affine shape is `(N,2,3)`, source sizes are
`(N,2)`, all provenance fields agree, and current image dimensions and hashes
exactly match the cache. Group all detection bboxes by frame; make one SAM3
call per frame, map each result using the corresponding affine, and append
clear observation diagnostics for empty/invalid masks.

- [ ] **Step 4: Implement background plane, per-ID fusion, and strict all-ID outcome**

```python
background_points = world_points[valid_points & ~dilated_all_detection_masks]
plane = select_support_plane(background_points, per_frame_points, all_object_points)
for global_id, observations in mapping.items():
    fused = np.concatenate(masked_points_for_observation(observation) for observation in observations)
    polygon, metrics = carton_footprint_polygon(fused, plane)
    polygons.append(polygon)
```

Implement and test deterministic multi-candidate plane extraction/selection:
frame-balanced background sampling, up to five RANSAC planes, residual/span/frame
coverage gates, object height/table hull compatibility, and ambiguity rejection
for a wall-versus-table synthetic fixture. Use every mapped observation that
passes the point gate, fuse by global ID with 5-mm voxel balancing, and reject
an ID with any substantial second DBSCAN component. If any mapped global ID has
no accepted polygon, write all diagnostics but set `status: "rejected"`,
`value_m2: null`, and `success: False`. Otherwise snap polygons to 0.1-mm and
1-mm precision grids, require union precision sensitivity within the design
tolerance, call `union_footprints` exactly once for the accepted precision, and
set its area as the only total. Never add individual areas.

- [ ] **Step 5: Implement auditable GeoJSON and top-down image**

`footprints.geojson` must contain one feature per global-ID OBB and one `union`
feature, in plane `(u,v)` metres, with `coordinate_space` explicitly set to
`local_support_plane_meters`, plus `global_id`, `area_m2`, and
`observations_used` properties. `top_down_footprint.png` must draw each OBB
outline plus a filled union boundary with axes labelled metres. The report
contains cache provenance, all plane candidates/final gates, per-ID
observation/voxel/component diagnostics, union algebra and precision
sensitivity, library versions, and artifact paths.

- [ ] **Step 6: Run stage tests, compile, and commit**

Run:

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/fd-area-code-uv-cache UV_OFFLINE=1 uv run --active --no-project python -m pytest -q tests/test_da3_footprint_stage.py tests/test_ground_stack_footprint.py
VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/fd-area-code-uv-cache UV_OFFLINE=1 uv run --active --no-project python -m py_compile modules/da3_footprint_stage.py utils/ground_stack_footprint.py
```

Expected: PASS/no output. Commit only Task 2 files with `feat: add DA3 footprint measurement stage`.

### Task 3: Replace obsolete CLI/API and document the real metric

**Files:**
- Modify: `code/main.py`
- Modify: `code/config.yaml`
- Modify: `README.md`
- Modify: `code/README.md`
- Delete: `code/modules/da3_metric_area_stage.py`
- Delete: `code/modules/ground_stack_area_stage.py`
- Delete: `code/utils/ground_stack_area.py`
- Delete: `code/tests/test_ground_stack_area.py`
- Modify: `code/tests/test_da3_footprint_stage.py`

**Interfaces:**
- Consumes `run_da3_footprint(dataset_path, save_root)` from Task 2.
- Produces `python main.py --mode ground-stack-area --dataset <dataset> --save_root <save_root>` as the sole supported area command.

- [ ] **Step 1: Write a failing CLI test with no anchors or mode flag**

```python
def test_main_ground_stack_area_dispatches_footprint_without_anchor(monkeypatch, tmp_path):
    monkeypatch.setattr("modules.da3_footprint_stage.run_da3_footprint", accepted_stub)
    completed = subprocess.run([... "--mode", "ground-stack-area", "--dataset", str(dataset), "--save_root", str(save_root)])
    assert completed.returncode == 0

def test_main_rejects_removed_area_mode_argument(tmp_path):
    completed = subprocess.run([... "--area-mode", "da3_metric"], capture_output=True, text=True)
    assert completed.returncode == 2
```

Use a subprocess fixture that places a minimal required DA3 cache/mapping, not a network/model stub; the first test must observe the new stage report metric.

- [ ] **Step 2: Run CLI tests and confirm the old parser makes the removed-option test fail**

Run: `VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/fd-area-code-uv-cache UV_OFFLINE=1 uv run --active --no-project pytest -q tests/test_da3_footprint_stage.py -k main`

Expected: failure because `--area-mode` is still accepted or dispatches the old stage.

- [ ] **Step 3: Replace the CLI and delete obsolete bbox-area code**

Remove `--area-mode`, all four `--area-anchor-*` parser arguments, every import/dispatch branch for calibrated or visible-bbox measurement, and old `ground_stack_area` config. `ground-stack-area` calls only `run_da3_footprint(args.dataset, app.save_root)` and keeps existing `success/status/report_path` exit semantics. Delete exactly the obsolete modules and test file listed above; do not delete DA3 reconstruction, dedup, SAM3, or facing-area code.

- [ ] **Step 4: Correct README and config language**

Document that the result is the union of carton OBB projections onto the inferred supporting plane, includes overhang, does not mean package surface/front-face/contact-only area, requires existing DA3 cache + global mapping + local SAM3 checkpoint, and can reject if one physical carton lacks geometry. Provide the exact anchor-free command and output artifact paths. `code/config.yaml` must state metric `da3_ground_footprint_union` and no calibration fields.

- [ ] **Step 5: Run full focused suite, then real-video command**

Run:

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/fd-area-code-uv-cache UV_OFFLINE=1 uv run --active --no-project pytest -q tests/test_ground_stack_footprint.py tests/test_da3_footprint_stage.py tests/test_da3_3d_reconstructor.py
VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/fd-area-code-uv-cache UV_OFFLINE=1 uv run --active --no-project python main.py --mode ground-stack-area --dataset /tmp/ground-stack-bbox-area/code/video_dedup_runs/fd_area_test --save_root /tmp/ground-stack-bbox-area/code/Output
```

Expected: focused suite PASS. The real command writes an accepted or rejected reviewable footprint report without a reference; inspect the report and `top_down_footprint.png` before claiming accepted.

- [ ] **Step 6: Commit, excluding all runtime artifacts**

Run `git diff --check`, inspect `git status --short -uall`, and use an explicit allowlist. Do not stage `code/video_dedup_runs/fd_area_test/`, `code/Output/`, any venv, or SDD scratch files. Commit with `refactor: measure ground-stack footprint union`.

## Plan Self-Review

## Execution Progress

- 2026-08-13: Task 3 public CLI migration completed in `43d1400 refactor: measure ground-stack footprint union`; `ground-stack-area` now dispatches only to `run_da3_footprint`, and obsolete bbox/anchor stages, utility, configuration, and tests were removed.
- 2026-08-13: Post-review correction completed: parser help and top-level README now use the footprint-union definition; regression tests reject all five retired bbox/anchor flags and assert schema-v2 provenance survives `save_predictions_cache()`.
- 2026-08-13 validation: existing `code/.venv` ran `46 passed` across geometry, footprint-stage, and DA3-reconstructor tests; `py_compile` and `git diff --check` passed.
- 2026-08-13 real-video evidence: `fd_area_test` cache passed the schema-v2/provenance check. The public command wrote a reviewable rejected report with `value_m2: null` because this process could not see a CUDA GPU for local SAM3 (`No CUDA GPUs are available`); no partial total was released.
- 2026-08-13 support-plane correction: 12 mm RANSAC now remains candidate generation only. Up to three SVD passes retain only points within 10 mm; the final retained set must still meet the 10,000-point, 10%-background, and P95<=10 mm gates. Table hull, spans, and frame coverage use that same final set, while raw 12 mm statistics remain diagnostic-only. Focused geometry/stage/reconstructor validation: 51 passed.
- 2026-08-13 real-video rerun after `a0784cd`: the final 10 mm geometry selected candidate 1 (163,531 retained points, 10.94% background, P95 8.84 mm) and rejected candidate 0 by the object-hull gate. The overall measurement remained rejected, correctly, because global ID 7 has one low-confidence (0.403) `bottle` detection on image 11's right boundary only; its observed projected strip is 0.1656 m by 0.0205 m, below the carton minimum-side gate. A read-only diagnostic union of the other six reconstructable IDs is 0.276002 m2; it is not a published all-detections result. No partial total was written to the formal report.

- Spec coverage: Task 1 implements deterministic plane/OBB/union gates; Task 2 implements mandatory masks, cache alignment, all-ID rejection, and audit artifacts; Task 3 removes the wrong public API, documents the new definition, and runs real-video evidence.
- Placeholder scan: no TBD/TODO or undefined interfaces remain. Each test invokes an API defined in Task 1 or Task 2.
- Type consistency: the stage consumes Task 1’s `SupportPlane`/geometry APIs; `main.py` consumes Task 2’s `run_da3_footprint` report contract.

## Execution Handoff

Rick explicitly selected subagent completion. Execute using `superpowers:subagent-driven-development`: one fresh implementer per task, a scoped review after each task, then one whole-branch review before branch handoff.
