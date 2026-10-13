# DA3 Ground-Stack Footprint Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DA3 ground-stack footprint reproducible and remove the systematic geometric biases found in the 2026-10-13 review; only if Rick approves, let it publish when every excluded global ID is provably inside the included union.

**Architecture:** All per-ID geometry moves into one shared function in `utils/ground_stack_footprint.py` that the formal stage and both shadow-evidence paths call. The selected support plane gains a deterministic normal refinement and per-frame support offsets, and the Viewer-shared scene orientation becomes deterministic. Publication-policy changes are isolated in decision-gated tasks (Tasks 8–9).

**Tech Stack:** Python 3.11, numpy, shapely 2, scikit-learn (`DBSCAN`, `NearestNeighbors`), Open3D (nearest-neighbour distances only), pytest.

## Review evidence this plan acts on

| # | Finding | Evidence | Task |
|---|---------|----------|------|
| R0 | Formal result is not reproducible: the shared orientation uses Open3D `segment_plane`, whose parallel RANSAC differs between runs even with `open3d.utility.random.seed`. Three evidence-isolation tests in `test/test_da3_footprint_stage.py` fail intermittently on unmodified `main` (3 of 5 runs). | Repeated runs on 2026-10-13; diff is in `report["plane"]` | 1 |
| R1 | One observation with < 32 points rejects its whole global ID even when other views are good. | video3: 10 of 44 such IDs had another good view | 3 |
| R2 | 1 %/99 % trimming on arbitrary world-derived u/v axes: 0.40×0.30 m face gives −4.0 % when aligned with u/v and ≈0 % at 15°–45°; an 8-point sparse chain on an edge gives +7.5 % at 30°/45°. No replacement was uniformly better on synthetic data (see Task 4), so the estimator is chosen with real data. | Synthetic, `voxel_balance_projected` + `carton_footprint_polygon_from_projected` | 4 (D4) |
| R3 | "Lowest eligible plane" selects a per-frame depth-offset layer: video3 selected −0.598 m (11/29 frames) instead of −0.561 m (26/29 frames); per-frame floor medians span −0.533 to −0.614 m, so the 15 mm elevation filter is wrong by several centimetres in most frames. The 36.8 mm two-layer behaviour is deliberately tested, so selection stays; elevation becomes per-frame. | `test_refined_tail_mode_is_deduplicated_without_merging_separate_layers`; video3 report | 5 |
| R4 | The projection normal is never checked: the `p95_residual` gate is tautological (points are first filtered to ≤ 10 mm, then their p95 is compared with 10 mm). A 2° tilt inflates a 1.2 m four-carton column union by +5.5 %, 5° by +16.4 %. Independent 1 cm RANSAC on video3 floor: 0.55°–0.90° from the shared up. | `ground_stack_footprint.py:297-336`; synthetic column; video3 cache | 6 |
| R5 | `KeyError` from programming errors is converted into a measurement rejection. | `da3_footprint_stage.py:326-332` | 7 |
| R6 | All-or-nothing publication: 169 of 408 video3 IDs failed (74 OBB side < 5 cm, 44 weak observation, 38 multi-component, 13 too few voxels); every real run so far is rejected. | 6 runs since 2026-10-10 | 8 (D1) |
| R7 | DBSCAN "substantial second component" uses `min(32, 20 %)`: deliberate (see `test_absolute_32_point_second_component_is_rejected`), but any 32-voxel blob rejects regardless of the main component; 8 of 38 video3 multi-component IDs had second/largest < 10 %. | `ground_stack_footprint.py:716-719` | 9 (D2) |

Correction to the review message: R7 was described as a probable `min`/`max` typo. The existing test shows it is intentional, so it is a policy decision, not a bug.

## Decisions required from Rick

- **D1 – publication policy.** A: keep all-or-nothing (skip Task 8). **B (recommended): exclude failed IDs and publish only if every excluded ID's visible points lie inside the included union (Task 8).** C: per-stack footprint (out of scope for this plan).
- **D2 – component threshold.** Keep `min(32, 20 %)` (skip Task 9) or **use 20 % of the largest component (recommended, Task 9).**
- **D3 – shared orientation (Task 1).** Task 1 changes the Viewer-shared fit from Open3D `segment_plane` to a seeded numpy RANSAC + SVD refit and bumps `ALGORITHM_VERSION`; Viewer leveling may shift by RANSAC-noise-sized amounts once sidecars are recomputed. Included by default because the formal area is otherwise not reproducible.

- **D4 – OBB extent estimator (Task 4).** Decided after the Task 4 experiment. Candidates: keep world-axis 1 % trimming; **OBB-frame 0.5 % trimming (recommended default: density-agnostic, no catastrophic synthetic failure)**; density-adaptive core (p90 radius). Synthetic results from 2026-10-13 are in Task 4.

Tasks 0–3, 5–7 and 10 do not depend on D1/D2/D4.

## Global Constraints

- Work on branch `fix/footprint-area-logic` created from `main` in worktree `runtime/worktrees/footprint-area-fix` (use superpowers:using-git-worktrees). Do not modify `feat/viewer-clarity-20261012` or the Docker branch.
- Run Python only through `uv run` with the shared environment: in the worktree shell, `export UV_PROJECT_ENVIRONMENT=/home/xingyu/3D_Recognization/.venv`.
- Run tests as `uv run python -m pytest -q -p no:cacheprovider <paths>`. Bare `pytest test/<file>.py` cannot import `main`.
- `FOOTPRINT_SUITE` = `test/test_ground_stack_footprint.py test/test_directional_support_plane.py test/test_da3_footprint_stage.py test/test_footprint_evidence.py test/test_scene_orientation.py`.
- No fallback that hides an error. Every estimation rule (for example a frame without enough support points using the selected plane) must be recorded in the report.
- Every new numeric threshold is an initial engineering parameter and must appear in the report diagnostics.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not push and do not rebuild Docker images.

## File map

| File | Responsibility | Tasks |
|------|----------------|-------|
| `utils/scene_orientation.py` | Deterministic shared plane fit | 1 |
| `utils/ground_stack_footprint.py` | Shared per-ID fusion, dense-core OBB, frame offsets, normal refinement, coverage, component rule | 2–6, 8, 9 |
| `src/da3_footprint_stage.py` | Stage wiring, report fields, publication rule | 2, 3, 5, 6, 7, 8, 10 |
| `utils/footprint_evidence.py` | Shadow evidence calls the shared function | 2, 3, 8 |
| `test/test_scene_orientation.py` | Orientation reproducibility | 1 |
| `test/test_ground_stack_footprint.py` | Geometry unit tests | 2–6, 8, 9 |
| `test/test_directional_support_plane.py` | Gate removal | 6 |
| `test/test_da3_footprint_stage.py` | Stage behaviour | 3, 5, 6, 7, 8, 10 |
| `test/test_footprint_evidence.py` | Snapshot helper | 2 |
| `docs/ground_stack_footprint_algorithm.md` | Algorithm contract | 10 |

---

### Task 0: Worktree and baseline

**Files:**
- Create: `docs/superpowers/plans/2026-10-13-footprint-area-fixes.md` (this file, copied into the worktree)
- Create (untracked): `runtime/footprint-fix-validation-20261013/run.sh`, `runtime/footprint-fix-validation-20261013/summarize.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `run.sh <label>` and `summarize.py <label>`, reused in Task 10.

- [x] **Step 1: Create the worktree and commit the plan**

```bash
cd /home/xingyu/3D_Recognization
git worktree add runtime/worktrees/footprint-area-fix -b fix/footprint-area-logic main
mkdir -p runtime/worktrees/footprint-area-fix/docs/superpowers/plans
cp docs/superpowers/plans/2026-10-13-footprint-area-fixes.md \
  runtime/worktrees/footprint-area-fix/docs/superpowers/plans/
cd runtime/worktrees/footprint-area-fix
export UV_PROJECT_ENVIRONMENT=/home/xingyu/3D_Recognization/.venv
export FOOTPRINT_SUITE="test/test_ground_stack_footprint.py test/test_directional_support_plane.py test/test_da3_footprint_stage.py test/test_footprint_evidence.py test/test_scene_orientation.py"
git add docs/superpowers/plans/2026-10-13-footprint-area-fixes.md
git commit -m "docs(plan): add ground-stack footprint fix plan

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [x] **Step 2: Record the flaky baseline**

```bash
for i in 1 2 3; do
  uv run python -m pytest -q -p no:cacheprovider test/test_da3_footprint_stage.py \
    -k "evidence_failures_do_not_change or failed_camera_contract_without or valid_shadow_evidence_without" | tail -1
done
```

Expected: at least one run shows `failed` (R0). Record the three result lines in the Task 1 commit body.

- [x] **Step 3: Write the validation scripts**

`runtime/footprint-fix-validation-20261013/run.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/xingyu/3D_Recognization
VAL="$ROOT/runtime/footprint-fix-validation-20261013/$1"
for n in 4 5 6 7 8; do
  name="floor_display$n"
  mkdir -p "$VAL/$name"
  ln -sfn "$ROOT/Output/$name/da3_cache" "$VAL/$name/da3_cache"
  ln -sfn "$ROOT/Output/$name/dedup_detections" "$VAL/$name/dedup_detections"
  uv run python main.py --mode ground-stack-area --dataset "$ROOT/imdata/$name" \
    --save_root "$VAL" \
    --sam3_mask_cache_root "$ROOT/Output/$name/sam3_mask_cache_sam31/v2" \
    --device cpu > "$VAL/$name.log" 2>&1 || test $? -eq 2
done
```

`runtime/footprint-fix-validation-20261013/summarize.py`:

```python
"""Print one JSON line per dataset for a validation label."""

import json
import sys
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent / sys.argv[1]
for dataset in sorted(path for path in root.iterdir() if path.is_dir()):
    output = dataset / "ground_stack_footprint"
    run_id = json.loads((output / "CURRENT").read_text())["run_id"]
    report = json.loads((output / "runs" / run_id / "measurement_report.json").read_text())
    per_id = report["per_global_id"]
    plane = report.get("plane") or {}
    offsets = [
        item["offset_m"]
        for item in (plane.get("frame_support_offsets") or {}).get("frames", {}).values()
    ]
    print(json.dumps({
        "dataset": dataset.name,
        "schema": report["schema_version"],
        "status": report["status"],
        "value_m2": report["value_m2"],
        "rejection_reason": (report.get("rejection_reason") or "")[:160],
        "ids": len(per_id),
        "id_outcomes": dict(Counter(item.get("rejection", "ok") for item in per_id.values()).most_common()),
        "selected_index": plane.get("selected_index"),
        "normal_angle_deg": (plane.get("normal_refinement") or {}).get("angle_to_shared_up_deg"),
        "frame_offset_range_m": [min(offsets), max(offsets)] if offsets else None,
        "coverage": report.get("coverage"),
    }, ensure_ascii=False))
```

The stage writes `scene_orientation.json` beside each resolved `Output/floor_display*/da3_cache/predictions.npz`, which is the existing shared-sidecar behaviour. video3 inputs (`runtime/fd-videos-20260911/`) were deleted on 2026-10-13 14:31 by another process, so fd4–fd8 are the only complete real inputs.

- [x] **Step 4: Run the baseline**

```bash
bash /home/xingyu/3D_Recognization/runtime/footprint-fix-validation-20261013/run.sh baseline
uv run python /home/xingyu/3D_Recognization/runtime/footprint-fix-validation-20261013/summarize.py baseline
```

Expected: five JSON lines, `schema` `2.0.0`. Keep the output for Task 10.

---

### Task 1: Deterministic shared scene orientation (R0, D3)

**Files:**
- Modify: `utils/scene_orientation.py:17` (`ALGORITHM_VERSION`), `:30-34` (constants), `:58-104` (fit loop); add `_seeded_plane_ransac`
- Test: `test/test_scene_orientation.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `fit_scene_orientation(valid_points, extrinsic) -> SceneOrientation` is bitwise reproducible; `ALGORITHM_VERSION == "viewer-plane-v2"`.

- [x] **Step 1: Write the failing test** (append to `test/test_scene_orientation.py`)

```python
def test_fit_is_bitwise_reproducible_and_versioned():
    rng = np.random.default_rng(3)
    count = 30_000
    floor = np.column_stack(
        [
            rng.uniform(-2.0, 2.0, count),
            1.0 + rng.normal(0.0, 0.005, count),
            rng.uniform(2.0, 6.0, count),
        ]
    )
    extrinsic = np.array([np.c_[np.eye(3), np.zeros(3)]])
    first = orientation.fit_scene_orientation(floor, extrinsic)
    second = orientation.fit_scene_orientation(floor.copy(), extrinsic)
    assert first.status == "fitted"
    assert np.array_equal(first.normal_world, second.normal_world)
    assert first.plane_offset_m == second.plane_offset_m
    assert orientation.ALGORITHM_VERSION == "viewer-plane-v2"
```

- [x] **Step 2: Run it to verify it fails**

Run: `uv run python -m pytest -q -p no:cacheprovider test/test_scene_orientation.py -k bitwise`
Expected: FAIL (`ALGORITHM_VERSION` is `viewer-plane-v1`; normals may also differ).

- [x] **Step 3: Implement**

In `utils/scene_orientation.py` set `ALGORITHM_VERSION = "viewer-plane-v2"` and add after `_PLANE_MAX_BELOW_RATIO = 0.15`:

```python
_PLANE_RANSAC_TRIALS = 1_000
_PLANE_RANSAC_SEED = 13
_PLANE_RANSAC_BATCH = 32
```

Add above `fit_scene_orientation`:

```python
def _seeded_plane_ransac(
    points: np.ndarray, threshold: float, seed: int
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Deterministic fixed-trial RANSAC followed by an SVD refit on its inliers.

    Replaces Open3D ``segment_plane``, whose parallel sampling is not
    reproducible even with ``open3d.utility.random.seed``.
    """
    generator = np.random.default_rng(seed)
    best_count = -1
    best: tuple[np.ndarray, float] | None = None
    for start in range(0, _PLANE_RANSAC_TRIALS, _PLANE_RANSAC_BATCH):
        size = min(_PLANE_RANSAC_BATCH, _PLANE_RANSAC_TRIALS - start)
        triplets = np.stack(
            [generator.choice(len(points), size=3, replace=False) for _ in range(size)]
        )
        first, second, third = (points[triplets[:, column]] for column in range(3))
        normals = np.cross(second - first, third - first)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > np.finfo(np.float64).eps
        if not valid.any():
            continue
        normals = normals[valid] / norms[valid, None]
        offsets = -np.einsum("ij,ij->i", normals, first[valid])
        counts = np.count_nonzero(
            np.abs(points @ normals.T + offsets) <= threshold, axis=0
        )
        index = int(np.argmax(counts))
        if counts[index] > best_count:
            best_count = int(counts[index])
            best = (normals[index], float(offsets[index]))
    if best is None:
        return None
    normal, offset = best
    inliers = np.flatnonzero(np.abs(points @ normal + offset) <= threshold)
    centre = points[inliers].mean(axis=0)
    _, _, vectors = np.linalg.svd(points[inliers] - centre, full_matrices=False)
    refined = vectors[-1] / np.linalg.norm(vectors[-1])
    if refined @ normal < 0:
        refined = -refined
    return refined, float(-refined @ centre), inliers
```

Replace the part of `fit_scene_orientation` from `remaining = pcd` through the loop's final line (the second `remaining = remaining.select_by_index(inliers, invert=True)`, at the end of the `for` body) with:

```python
    remaining = np.arange(len(subsample))
    for candidate_index in range(_PLANE_MAX_CANDIDATES):
        if len(remaining) < max(min_inliers, 3):
            break
        seed = _PLANE_RANSAC_SEED + candidate_index
        fitted = _seeded_plane_ransac(subsample[remaining], distance_threshold, seed)
        if fitted is None:
            logger.warning("RANSAC found no non-degenerate plane; stopping search")
            break
        normal, plane_d, inliers = fitted
        if normal @ (camera_mean - subsample.mean(axis=0)) < 0:
            normal = -normal
            plane_d = -plane_d
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -normal[1]))))
        below_ratio = float(np.mean(subsample @ normal + plane_d < -0.1))
        if (
            len(inliers) >= min_inliers
            and tilt_deg <= _PLANE_MAX_TILT_DEG
            and below_ratio <= _PLANE_MAX_BELOW_RATIO
        ):
            r_level = _shortest_arc_to_y_up(m_flip[:3, :3] @ normal)
            return SceneOrientation(
                "fitted",
                (r_level @ m_flip)[:3, :3],
                normal,
                float(plane_d),
                {
                    "distance_threshold_m": distance_threshold,
                    "inlier_count": int(len(inliers)),
                    "sample_count": int(len(subsample)),
                    "tilt_deg": tilt_deg,
                    "below_ratio": below_ratio,
                    "ransac_trials": _PLANE_RANSAC_TRIALS,
                    "ransac_seed": seed,
                },
            )
        remaining = np.delete(remaining, inliers)
```

Keep the Open3D `PointCloud` and `compute_nearest_neighbor_distance` lines above the loop; they are deterministic.

- [x] **Step 4: Run the tests**

Run: `uv run python -m pytest -q -p no:cacheprovider test/test_scene_orientation.py test/test_web_viewer_export.py`
Expected: PASS.

Then run the flaky trio five times (same command as Task 0 Step 2, `for i in 1 2 3 4 5`).
Expected: `4 passed, ... deselected` in all five runs.

Then run: `uv run python -m pytest -q -p no:cacheprovider $FOOTPRINT_SUITE`
Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add utils/scene_orientation.py test/test_scene_orientation.py
git commit -m "fix(orientation): make the shared floor fit reproducible

Open3D segment_plane is not reproducible even when seeded, so the
footprint report changed between identical runs. Use a seeded 1,000-trial
numpy RANSAC with an SVD refit and bump the algorithm version so v1
sidecars are recomputed.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: One shared per-ID fusion function (behaviour-preserving)

**Files:**
- Modify: `utils/ground_stack_footprint.py` (imports; new constants, `ObservationPoints`, `elevated_projected_points`, `fused_carton_footprint`)
- Modify: `src/da3_footprint_stage.py:45-53` (imports), `:158-166` (locals), `:199-204`, `:260-293` (per-ID loop), `:341-348` (`FormalSnapshot`), `:621-766` (`_masked_observations`)
- Modify: `utils/footprint_evidence.py` (`FormalSnapshot`, `_fixed_plane_variant_area`, `_build_valid_camera_evidence`, `_leave_one_observation_out`, `_fixed_plane_polygon`)
- Test: `test/test_ground_stack_footprint.py`, `test/test_footprint_evidence.py`

**Interfaces:**
- Consumes: `SupportPlane`, `project_to_plane`, `voxel_balance_projected`, `carton_footprint_polygon_from_projected`.
- Produces:
  - `ObservationPoints(image_id: int, points: np.ndarray)` (frozen dataclass)
  - `elevated_projected_points(observations: Sequence[ObservationPoints], plane: SupportPlane, frame_offsets_m: Mapping[int, float]) -> np.ndarray` of shape `(M, 2)`
  - `fused_carton_footprint(observations, plane, *, frame_offsets_m) -> tuple[Polygon, dict[str, object]]`; a raised `FootprintError.diagnostics` holds the metrics computed before the failure
  - `FormalSnapshot.frame_offsets_m: Mapping[int, float]` (required field)

- [x] **Step 1: Write the failing tests** (append to `test/test_ground_stack_footprint.py`; add `ObservationPoints` and `fused_carton_footprint` to its import list)

```python
def test_fused_footprint_fuses_views_with_explicit_frame_offsets():
    plane = horizontal_support_plane()
    points = carton_points(x_range=(0.0, 1.0), y_range=(0.0, 1.0), height=0.2)
    polygon, metrics = fused_carton_footprint(
        [ObservationPoints(0, points[:6400]), ObservationPoints(1, points[6400:])],
        plane,
        frame_offsets_m={0: 0.0, 1: 0.0},
    )
    assert polygon.area == pytest.approx(1.0)
    assert metrics["observations_used"] == 2
    assert metrics["projected_voxel_point_count"] >= 64


def test_fused_footprint_rejection_carries_partial_metrics():
    flat = carton_points(x_range=(0.0, 1.0), y_range=(0.0, 1.0), height=0.01)
    with pytest.raises(FootprintError, match="fewer than 64 valid points") as raised:
        fused_carton_footprint(
            [ObservationPoints(0, flat)],
            horizontal_support_plane(),
            frame_offsets_m={0: 0.0},
        )
    assert raised.value.diagnostics["observations_used"] == 1
    assert raised.value.diagnostics["elevated_point_count"] == 0


def test_missing_frame_offset_is_a_contract_error():
    points = carton_points(x_range=(0.0, 1.0), y_range=(0.0, 1.0), height=0.2)
    with pytest.raises(FootprintError, match="missing support offset for image 7"):
        fused_carton_footprint(
            [ObservationPoints(7, points)],
            horizontal_support_plane(),
            frame_offsets_m={},
        )
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run python -m pytest -q -p no:cacheprovider test/test_ground_stack_footprint.py -k "fused_footprint or missing_frame_offset"`
Expected: collection error, `ImportError: cannot import name 'ObservationPoints'`.

- [x] **Step 3: Implement the shared function**

In `utils/ground_stack_footprint.py` add `from typing import Mapping, Sequence` and, after the `SupportPlane` dataclass:

```python
OBSERVATION_MIN_POINTS = 32
FUSED_MIN_POINTS = 64
ELEVATION_THRESHOLD_M = 0.015
FOOTPRINT_VOXEL_M = 0.005


@dataclass(frozen=True)
class ObservationPoints:
    """Valid DA3 world points inside one detection's processed mask."""

    image_id: int
    points: np.ndarray
```

After `carton_footprint_polygon_from_projected`:

```python
def elevated_projected_points(
    observations: Sequence[ObservationPoints],
    plane: SupportPlane,
    frame_offsets_m: Mapping[int, float],
) -> np.ndarray:
    """Project points more than 15 mm above their own frame's support surface."""
    parts: list[np.ndarray] = []
    for item in observations:
        image_id = int(item.image_id)
        if image_id not in frame_offsets_m:
            raise FootprintError(f"missing support offset for image {image_id}")
        points = np.asarray(item.points, dtype=float).reshape(-1, 3)
        heights = (points - plane.point) @ plane.normal - float(frame_offsets_m[image_id])
        parts.append(points[heights > ELEVATION_THRESHOLD_M])
    elevated = np.concatenate(parts) if parts else np.empty((0, 3))
    return project_to_plane(elevated, plane)


def fused_carton_footprint(
    observations: Sequence[ObservationPoints],
    plane: SupportPlane,
    *,
    frame_offsets_m: Mapping[int, float],
) -> tuple[Polygon, dict[str, object]]:
    """Fuse every observation of one global ID into its support-plane OBB.

    Shared by the formal stage and both shadow-evidence paths. A raised
    FootprintError carries the metrics computed before the failure.
    """
    if not observations:
        raise FootprintError("global id has no observations")
    if any(len(item.points) < OBSERVATION_MIN_POINTS for item in observations):
        raise FootprintError("observation has fewer than 32 valid masked points")
    fused = np.concatenate([np.asarray(item.points, dtype=float) for item in observations])
    metrics: dict[str, object] = {
        "observations_used": len(observations),
        "height_median_m": float(np.median((fused - plane.point) @ plane.normal)),
    }
    projected = elevated_projected_points(observations, plane, frame_offsets_m)
    metrics["elevated_point_count"] = int(len(projected))
    if len(projected) < FUSED_MIN_POINTS:
        raise FootprintError("fused global id has fewer than 64 valid points", metrics)
    balanced = voxel_balance_projected(projected, voxel_size_m=FOOTPRINT_VOXEL_M)
    metrics["projected_voxel_point_count"] = int(len(balanced))
    if len(balanced) < FUSED_MIN_POINTS:
        raise FootprintError(
            "fused global id has fewer than 64 projected voxel points", metrics
        )
    try:
        polygon, obb_metrics = carton_footprint_polygon_from_projected(balanced)
    except FootprintError as error:
        raise FootprintError(
            str(error), {**metrics, "component_diagnostics": error.diagnostics}
        ) from error
    return polygon, {**metrics, **obb_metrics}
```

- [x] **Step 4: Wire the stage**

In `src/da3_footprint_stage.py`:

1. In the `utils.ground_stack_footprint` import, remove `carton_footprint_polygon_from_projected`, `project_to_plane`, `voxel_balance_projected` and add `ObservationPoints`, `fused_carton_footprint`.
2. After `polygons: dict[str, Any] = {}` add `frame_offsets: dict[int, float] = {}`.
3. Replace the `all_object_points` comprehension with:

```python
        all_object_points = [
            observation.points
            for observations in masked_observations.values()
            for observation in observations
            if len(observation.points)
        ]
```

4. Immediately before `active_stage = "per_id_obb_union"`, add:

```python
        frame_offsets = {int(image_id): 0.0 for image_id in cache_frame_ids}
```

5. Replace the whole `for global_id, observation_points in masked_observations.items():` loop with:

```python
        for global_id, observations in masked_observations.items():
            per_id = report["per_global_id"][global_id]
            try:
                polygon, metrics = fused_carton_footprint(
                    observations, plane, frame_offsets_m=frame_offsets
                )
            except FootprintError as error:
                per_id["rejection"] = str(error)
                per_id.update(error.diagnostics)
                continue
            per_id.update(metrics)
            polygons[global_id] = polygon
```

6. Add `frame_offsets_m=dict(frame_offsets),` to the `FormalSnapshot(...)` call.
7. In `_masked_observations`: change `observations = {global_id: [] for global_id in mapping_by_id}` to `observations: dict[str, list[ObservationPoints]] = {global_id: [] for global_id in mapping_by_id}`; change the empty-mask append to `observations[global_id].append(ObservationPoints(image_id, np.empty((0, 3), dtype=float)))` and the other append to `observations[global_id].append(ObservationPoints(image_id, points))`; change the return annotation's first element to `dict[str, list[ObservationPoints]]`.

- [x] **Step 5: Wire the evidence**

In `utils/footprint_evidence.py`:

1. Import `ObservationPoints` and `fused_carton_footprint` from `utils.ground_stack_footprint` (drop `carton_footprint_polygon_from_projected` and `voxel_balance_projected` if no longer used; `project_to_plane` is still used by the guard).
2. Add the last field to `FormalSnapshot`: `frame_offsets_m: Mapping[int, float]`.
3. In `_fixed_plane_variant_area`: declare `grouped: dict[str, list[ObservationPoints]] = {}`, append `ObservationPoints(image_id, points)`, and replace everything inside `for global_id in sorted(expected_ids, key=_global_id_key):` with:

```python
        group = grouped[global_id]
        fused = np.concatenate([item.points for item in group])
        try:
            with np.errstate(over="raise", invalid="raise"):
                heights = (fused - plane.point) @ plane.normal
        except FloatingPointError as error:
            raise EvidenceComputationError(
                f"{variant} support-plane height arithmetic failed"
            ) from error
        _require_finite(
            heights,
            error_type=EvidenceComputationError,
            label=f"{variant} support-plane heights",
        )
        try:
            polygon, _ = fused_carton_footprint(
                group, plane, frame_offsets_m=formal_snapshot.frame_offsets_m
            )
        except FootprintError as error:
            raise FootprintError(f"{variant} global id {global_id}: {error}") from error
        polygons.append(set_precision(polygon, 0.0001))
```

4. Add a parameter `frame_offsets_m: Mapping[int, float]` to `_leave_one_observation_out` and `_fixed_plane_polygon`. In `_build_valid_camera_evidence`, call `_leave_one_observation_out(camera, group, plane, formal_snapshot.polygons.get(global_id), formal_snapshot.frame_offsets_m)`; inside `_leave_one_observation_out` call `_fixed_plane_polygon(camera, remaining, plane, frame_offsets_m)`.
5. In `_fixed_plane_polygon`: build `parts: list[ObservationPoints]` with `ObservationPoints(int(camera.frame_ids[item.frame_index]), points)` instead of `point_parts`; compute `fused = np.concatenate([part.points for part in parts])`; keep every existing guard unchanged (height errstate, finite heights, `elevated`, projection errstate, finite projection, int64 range); replace the lines from `balanced = voxel_balance_projected(` through `polygon, _ = carton_footprint_polygon_from_projected(balanced)` with:

```python
    polygon, _ = fused_carton_footprint(parts, plane, frame_offsets_m=frame_offsets_m)
```

- [x] **Step 6: Update the evidence test helper**

In `test/test_footprint_evidence.py`, add below the imports:

```python
_ZERO_OFFSETS = {image_id: 0.0 for image_id in range(100)}
```

and add `frame_offsets_m=_ZERO_OFFSETS,` to `_snapshot()` and to the two direct `FormalSnapshot(...)` constructions (in `test_absent_formal_plane_keeps_all_shadow_geometry_unavailable` and in the extreme-plane test near the end of the file).

- [x] **Step 7: Run the tests**

Run: `uv run python -m pytest -q -p no:cacheprovider $FOOTPRINT_SUITE`
Expected: all pass, including the three new tests. The formal value of `test_stage_fuses_global_id_views_and_uses_polygon_union` stays 1.5 ± 0.03.

- [x] **Step 8: Commit**

```bash
git add utils/ground_stack_footprint.py src/da3_footprint_stage.py utils/footprint_evidence.py \
  test/test_ground_stack_footprint.py test/test_footprint_evidence.py
git commit -m "refactor(area): share per-ID footprint fusion with shadow evidence

The formal stage, mask-robustness variants and leave-one-out evidence each
reimplemented elevation, projection, voxel balance and OBB fitting. Route
all three through fused_carton_footprint with explicit per-frame support
offsets (all zero for now) so later rule changes cannot diverge.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Exclude weak observations instead of rejecting the ID (R1)

**Files:**
- Modify: `utils/ground_stack_footprint.py` (`fused_carton_footprint`)
- Modify: `utils/footprint_evidence.py` (remove both evidence-side `< 32` raises)
- Test: `test/test_ground_stack_footprint.py`, `test/test_da3_footprint_stage.py`

**Interfaces:**
- Consumes: `fused_carton_footprint` from Task 2.
- Produces: metric `excluded_observations: list[{"image_id": int, "valid_point_count": int}]`; error message `global id has no observation with at least 32 valid masked points`.

- [x] **Step 1: Write the failing tests**

Append to `test/test_ground_stack_footprint.py`:

```python
def test_weak_observation_is_excluded_instead_of_rejecting_the_global_id():
    points = carton_points(x_range=(0.0, 1.0), y_range=(0.0, 1.0), height=0.2)
    polygon, metrics = fused_carton_footprint(
        [ObservationPoints(0, points), ObservationPoints(1, points[:10])],
        horizontal_support_plane(),
        frame_offsets_m={0: 0.0, 1: 0.0},
    )
    assert polygon.area == pytest.approx(1.0)
    assert metrics["observations_used"] == 1
    assert metrics["excluded_observations"] == [{"image_id": 1, "valid_point_count": 10}]


def test_global_id_with_only_weak_observations_is_rejected():
    points = carton_points(x_range=(0.0, 1.0), y_range=(0.0, 1.0), height=0.2)
    with pytest.raises(FootprintError, match="no observation with at least 32") as raised:
        fused_carton_footprint(
            [ObservationPoints(0, points[:10])],
            horizontal_support_plane(),
            frame_offsets_m={0: 0.0},
        )
    assert raised.value.diagnostics["excluded_observations"] == [
        {"image_id": 0, "valid_point_count": 10}
    ]
```

Append to `test/test_da3_footprint_stage.py`:

```python
def masks_with_tiny_frame_one(
    image_path: str, bboxes: list[list[float]], *args: object
) -> list[np.ndarray]:
    masks = exact_bbox_masks(image_path, bboxes, *args)
    if Path(image_path).stem == "1":
        x1, y1 = int(bboxes[0][0]), int(bboxes[0][1])
        masks[0][:] = False
        masks[0][y1 + 10 : y1 + 13, x1 + 10 : x1 + 13] = True
    return masks


def test_weak_second_view_no_longer_rejects_the_measurement(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(
        tmp_path, mask_factory=masks_with_tiny_frame_one
    )
    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())
    assert result["success"] is True
    assert report["value_m2"] == pytest.approx(1.5, abs=0.03)
    assert report["per_global_id"]["1"]["observations_used"] == 1
    assert report["per_global_id"]["1"]["excluded_observations"] == [
        {"image_id": 1, "valid_point_count": 9}
    ]
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run python -m pytest -q -p no:cacheprovider test/test_ground_stack_footprint.py test/test_da3_footprint_stage.py -k "weak_observation or only_weak or weak_second_view"`
Expected: 3 FAIL (`observation has fewer than 32 valid masked points`).

- [x] **Step 3: Implement**

In `fused_carton_footprint` replace from `if any(len(item.points) < OBSERVATION_MIN_POINTS ...` through the `metrics` dict with:

```python
    usable = [item for item in observations if len(item.points) >= OBSERVATION_MIN_POINTS]
    metrics: dict[str, object] = {
        "observations_used": len(usable),
        "excluded_observations": [
            {"image_id": int(item.image_id), "valid_point_count": int(len(item.points))}
            for item in observations
            if len(item.points) < OBSERVATION_MIN_POINTS
        ],
    }
    if not usable:
        raise FootprintError(
            "global id has no observation with at least 32 valid masked points", metrics
        )
    fused = np.concatenate([np.asarray(item.points, dtype=float) for item in usable])
    metrics["height_median_m"] = float(np.median((fused - plane.point) @ plane.normal))
```

and change the next call to `elevated_projected_points(usable, plane, frame_offsets_m)`.

In `utils/footprint_evidence.py` delete the `if len(points) < 32: raise FootprintError(f"{variant} observation ...")` block in `_fixed_plane_variant_area` and the `if len(points) < 32: raise FootprintError("leave-one-out observation ...")` block in `_fixed_plane_polygon`; the shared function now applies the rule.

- [x] **Step 4: Run the tests**

Run: `uv run python -m pytest -q -p no:cacheprovider $FOOTPRINT_SUITE`
Expected: all pass. `test_nonempty_mask_without_valid_da3_points_rejects_total` still rejects: global ID 2 has no usable observation.

- [x] **Step 5: Commit**

```bash
git add utils/ground_stack_footprint.py utils/footprint_evidence.py \
  test/test_ground_stack_footprint.py test/test_da3_footprint_stage.py
git commit -m "fix(area): drop weak observations instead of rejecting the global ID

An extra view with fewer than 32 masked points made a well-observed ID fail,
so more views could only lower acceptance. Exclude such observations,
record them, and fail only when no usable observation remains.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Choose and implement the OBB extent estimator (R2, D4)

No candidate was uniformly better on 2026-10-13 synthetic data (0.40×0.30 m face, 5 mm voxel balance, rotations 0°/15°/30°/45°, with and without an 8-point 15 mm chain on an edge; values are area error vs the sample MRR):

| Case | Current world-axis 1 % | B: OBB-frame 0.5 % | C: density core, p90 radius |
|------|------------------------|--------------------|-----------------------------|
| Uniform random | −4.00 … −0.01 % | −1.93 … −0.69 % | −0.05 … −0.01 % |
| Uniform + chain | −4.04 … +7.49 % | −1.95 … −0.67 % | −0.05 … −0.01 % |
| 6×12 mm grid | −2.24 … 0 % | −1.67 % | 0 % |
| 12.7 mm grid + chain | 0 % | 0 % | +5.41 % |
| 5 mm / 13 mm density halves | −3.25 … 0 % | 0 % | 0 % |
| Sparse quarter (< 10 % of points) | −3.27 … 0 % | 0 % | −22.86 % |
| 5→10 mm gradient + chain | −0.99 … +7.67 % | −0.47 … −0.44 % | +0.49 … +0.61 % |

C loses whole regions that hold under ~10 % of the points, which happens in fused multi-view data when only a far camera sees one side. B is density-agnostic, with a bounded −0.4 % to −1.9 % bias. Real data decides.

**Files:**
- Create (untracked): `runtime/footprint-fix-validation-20261013/obb_candidates.py`, `runtime/footprint-fix-validation-20261013/compare_obb.py`
- Modify (after D4): `utils/ground_stack_footprint.py` (`carton_footprint_polygon_from_projected`; replace `_trim_projected_outliers`)
- Test: `test/test_ground_stack_footprint.py`

**Interfaces:**
- Consumes: `fused_carton_footprint` (Task 2) calls `carton_footprint_polygon_from_projected` through the module global, which the capture step relies on.
- Produces: `carton_footprint_polygon_from_projected(projected) -> (Polygon, metrics)` with `component_point_count`, `cleaned_point_count`, `footprint_area_m2`, `component_diagnostics`, plus `extent_estimator`.

- [x] **Step 1: Write the candidates**

`runtime/footprint-fix-validation-20261013/obb_candidates.py`:

```python
import numpy as np
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import utils.ground_stack_footprint as geometry


def world_axis_trim(points):
    return geometry._trim_projected_outliers(points, lower=0.01, upper=0.99)


def obb_frame_trim(points, fraction=0.005):
    rectangle = geometry._validate_obb_polygon(
        MultiPoint(points).convex_hull.minimum_rotated_rectangle
    )
    corners = np.asarray(rectangle.exterior.coords)
    edges = np.diff(corners[:3], axis=0)
    axis = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    axis = axis / np.linalg.norm(axis)
    local = points @ np.column_stack([axis, [-axis[1], axis[0]]])
    lower = np.quantile(local, fraction, axis=0)
    upper = np.quantile(local, 1.0 - fraction, axis=0)
    return points[np.all((local >= lower) & (local <= upper), axis=1)]


def density_core(points, percentile=90.0):
    unique = np.unique(points, axis=0)
    distances, _ = NearestNeighbors(n_neighbors=min(5, len(unique))).fit(unique).kneighbors(unique)
    radius = 1.5 * float(np.percentile(distances[:, -1], percentile))
    clustering = DBSCAN(eps=radius, min_samples=5).fit(unique)
    core = np.zeros(len(unique), dtype=bool)
    core[clustering.core_sample_indices_] = True
    return unique[core]


CANDIDATES = {"current": world_axis_trim, "B_obb_frame": obb_frame_trim, "C_density_core": density_core}
```

- [x] **Step 2: Capture real per-ID inputs and compare**

`runtime/footprint-fix-validation-20261013/compare_obb.py` runs the stage once per dataset with `carton_footprint_polygon_from_projected` wrapped to record each balanced input, then scores every candidate against `current`:

```python
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils.ground_stack_footprint as geometry
from obb_candidates import CANDIDATES
from src.da3_footprint_stage import run_da3_footprint

ROOT = Path("/home/xingyu/3D_Recognization")
SAVE_ROOT = Path(__file__).resolve().parent / "obb-capture"
captured: list[np.ndarray] = []
original = geometry.carton_footprint_polygon_from_projected


def capture(points):
    captured.append(np.asarray(points).copy())
    return original(points)


geometry.carton_footprint_polygon_from_projected = capture
for number in (4, 5, 6, 7, 8):
    name = f"floor_display{number}"
    target = SAVE_ROOT / name
    target.mkdir(parents=True, exist_ok=True)
    for link in ("da3_cache", "dedup_detections"):
        path = target / link
        if not path.exists():
            path.symlink_to(ROOT / "Output" / name / link)
    start = len(captured)
    run_da3_footprint(
        str(ROOT / "imdata" / name),
        SAVE_ROOT,
        sam3_mask_cache_root=ROOT / "Output" / name / "sam3_mask_cache_sam31" / "v2",
    )
    areas: dict[str, list[float]] = {key: [] for key in CANDIDATES}
    for points in captured[start:]:
        try:
            component, _ = geometry.select_footprint_component(points)
        except geometry.FootprintError:
            continue
        for key, estimator in CANDIDATES.items():
            cleaned = estimator(component)
            hull = MultiPoint(cleaned).convex_hull if len(cleaned) >= 3 else None
            rectangle = hull.minimum_rotated_rectangle if hull is not None else None
            areas[key].append(float(rectangle.area) if rectangle is not None else 0.0)
    current = np.asarray(areas["current"])
    valid = current > 0
    summary = {"dataset": name, "ids_scored": int(valid.sum())}
    for key in ("B_obb_frame", "C_density_core"):
        ratio = np.asarray(areas[key])[valid] / current[valid]
        summary[key] = {
            "median_ratio": float(np.median(ratio)),
            "p05_ratio": float(np.percentile(ratio, 5)),
            "p95_ratio": float(np.percentile(ratio, 95)),
            "below_0_8": int(np.count_nonzero(ratio < 0.8)),
            "above_1_05": int(np.count_nonzero(ratio > 1.05)),
        }
    print(json.dumps(summary))
```

Run (after Tasks 1–3 are committed):

```bash
cd /home/xingyu/3D_Recognization/runtime/worktrees/footprint-area-fix
uv run python /home/xingyu/3D_Recognization/runtime/footprint-fix-validation-20261013/compare_obb.py
```

Expected: five JSON lines. `below_0_8` counts IDs where a candidate removed more than 20 % of the current area (possible lost regions); `above_1_05` counts IDs where it kept material the current trimming removed.

- [x] **Step 3: Decision D4**

Send Rick the synthetic table above and the five JSON lines. Recommended rule: choose B unless it has more `below_0_8` IDs than C on real data; never choose C if `C_density_core.below_0_8` > 0 on any dataset. Record the decision in the Task 4 commit body.

- [x] **Step 4: Write the failing test for the chosen estimator** (add `carton_footprint_polygon_from_projected` to the import list)

```python
def _voxel_rectangle(angle_deg: float, *, chain: bool) -> np.ndarray:
    rng = np.random.default_rng(0)
    points = rng.uniform([-0.20, -0.15], [0.20, 0.15], size=(60_000, 2))
    if chain:
        sparse_chain = np.column_stack([0.20 + 0.015 * np.arange(1, 9), np.zeros(8)])
        points = np.vstack([points, sparse_chain])
    theta = np.deg2rad(angle_deg)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return voxel_balance_projected(points @ rotation.T, voxel_size_m=0.005)


def test_obb_extent_is_stable_across_orientation_and_sparse_chains():
    truth = 0.40 * 0.30
    clean = [
        carton_footprint_polygon_from_projected(_voxel_rectangle(angle, chain=False))[0].area
        for angle in (0.0, 15.0, 30.0, 45.0)
    ]
    chained = [
        carton_footprint_polygon_from_projected(_voxel_rectangle(angle, chain=True))[0].area
        for angle in (0.0, 15.0, 30.0, 45.0)
    ]
    assert (max(clean) - min(clean)) / truth < 0.015
    assert all(abs(area / truth - 1.0) < 0.025 for area in clean)
    assert all(with_chain <= without * 1.005 for with_chain, without in zip(chained, clean))
```

Run: `uv run python -m pytest -q -p no:cacheprovider test/test_ground_stack_footprint.py -k stable_across_orientation`
Expected: FAIL with the current estimator (spread 4 %, chain +7.5 % at 30°).

- [x] **Step 5: Implement the chosen estimator**

If D4 = B, add to `utils/ground_stack_footprint.py`:

```python
OBB_TRIM_FRACTION = 0.005


def _trim_in_obb_frame(points: np.ndarray) -> np.ndarray:
    """Trim extreme points along the component's own OBB axes, not world axes."""
    rectangle = _validate_obb_polygon(MultiPoint(points).convex_hull.minimum_rotated_rectangle)
    corners = np.asarray(rectangle.exterior.coords)
    edges = np.diff(corners[:3], axis=0)
    axis = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    axis = axis / np.linalg.norm(axis)
    local = points @ np.column_stack([axis, [-axis[1], axis[0]]])
    lower = np.quantile(local, OBB_TRIM_FRACTION, axis=0)
    upper = np.quantile(local, 1.0 - OBB_TRIM_FRACTION, axis=0)
    cleaned = points[np.all((local >= lower) & (local <= upper), axis=1)]
    if len(cleaned) < 3:
        raise FootprintError("carton footprint has fewer than three cleaned points")
    return cleaned
```

and in `carton_footprint_polygon_from_projected` replace `cleaned = _trim_projected_outliers(component, lower=0.01, upper=0.99)` with `cleaned = _trim_in_obb_frame(component)` and add `"extent_estimator": "obb_frame_trim_0.005"` to the returned metrics.

If D4 = C, add instead:

```python
DENSE_CORE_RADIUS_FACTOR = 1.5
DENSE_CORE_SPACING_PERCENTILE = 90.0
DENSE_CORE_MIN_SAMPLES = 5


def _dense_core_points(points: np.ndarray) -> np.ndarray:
    """Keep points whose neighbourhood matches the component's sampling density."""
    unique = np.unique(points, axis=0)
    distances, _ = (
        NearestNeighbors(n_neighbors=min(5, len(unique))).fit(unique).kneighbors(unique)
    )
    radius = DENSE_CORE_RADIUS_FACTOR * float(
        np.percentile(distances[:, -1], DENSE_CORE_SPACING_PERCENTILE)
    )
    if not np.isfinite(radius) or radius <= 0:
        raise FootprintError("carton footprint has no positive sampling radius")
    clustering = DBSCAN(eps=radius, min_samples=DENSE_CORE_MIN_SAMPLES).fit(unique)
    core = np.zeros(len(unique), dtype=bool)
    core[clustering.core_sample_indices_] = True
    if np.count_nonzero(core) < 3:
        raise FootprintError("carton footprint has fewer than three dense core points")
    return unique[core]
```

with `from sklearn.neighbors import NearestNeighbors`, `cleaned = _dense_core_points(component)` and `"extent_estimator": "density_core_p90"`.

In both cases delete `_trim_projected_outliers`.

- [x] **Step 6: Run the tests**

Run: `uv run python -m pytest -q -p no:cacheprovider $FOOTPRINT_SUITE`
Expected: all pass. A 2026-10-13 in-process probe of B and of C against the pre-Task-1 suite failed only tests from the R0-flaky family, so with Task 1 in place every test should pass.

- [x] **Step 7: Commit**

```bash
git add utils/ground_stack_footprint.py test/test_ground_stack_footprint.py
git commit -m "fix(area): make OBB extent independent of world-axis orientation

World-axis 1%/99% trimming biased areas by 0 to -4% depending on carton
heading and let a sparse edge chain add up to 7.5% when rotated.
<D4 decision and the compare_obb.py summary go here>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Replace the `<D4 ...>` line with the actual decision and the five summary lines before committing.

---

### Task 5: Per-frame support offsets for the elevation filter (R3)

**Files:**
- Modify: `utils/ground_stack_footprint.py` (add constants, `_horizontal_support_mask`, `estimate_frame_support_offsets`)
- Modify: `src/da3_footprint_stage.py` (replace the zero offset mapping; report field)
- Test: `test/test_ground_stack_footprint.py`, `test/test_da3_footprint_stage.py`

**Interfaces:**
- Consumes: `fused_carton_footprint(..., frame_offsets_m=...)`.
- Produces:
  - `_horizontal_support_mask(normals: np.ndarray, up: np.ndarray) -> np.ndarray`
  - `estimate_frame_support_offsets(background_points, background_normals, frame_ids, plane, *, image_ids) -> tuple[dict[int, float], dict[str, object]]`
  - report `plane.frame_support_offsets = {"window_m", "min_points", "frames": {"<id>": {"offset_m", "source", "point_count"}}}`, where `source` is `"frame"` or `"selected_plane"`.

- [x] **Step 1: Write the failing tests** (add `estimate_frame_support_offsets` to the import list)

```python
def _horizontal_floor(height: float, image_id: int, count: int = 2_000, seed: int = 0):
    rng = np.random.default_rng(seed + image_id)
    points = np.column_stack(
        [rng.uniform(0.0, 2.0, count), rng.uniform(0.0, 2.0, count), np.full(count, height)]
    )
    return points, np.tile([0.0, 0.0, 1.0], (count, 1)), np.full(count, image_id)


def test_frame_support_offsets_follow_each_frames_own_floor():
    parts = [
        _horizontal_floor(0.000, 0),
        _horizontal_floor(0.200, 0, seed=7),
        _horizontal_floor(0.037, 1),
        _horizontal_floor(-0.020, 2),
        _horizontal_floor(0.000, 3, count=500),
    ]
    offsets, diagnostics = estimate_frame_support_offsets(
        np.vstack([part[0] for part in parts]),
        np.vstack([part[1] for part in parts]),
        np.concatenate([part[2] for part in parts]),
        horizontal_support_plane(),
        image_ids=[0, 1, 2, 3, 4],
    )
    assert offsets[0] == pytest.approx(0.0, abs=1e-12)
    assert offsets[1] == pytest.approx(0.037, abs=1e-12)
    assert offsets[2] == pytest.approx(-0.020, abs=1e-12)
    assert offsets[3] == 0.0 and offsets[4] == 0.0
    assert diagnostics["frames"]["1"]["source"] == "frame"
    assert diagnostics["frames"]["3"]["source"] == "selected_plane"
    assert diagnostics["frames"]["4"]["point_count"] == 0


def test_frame_offsets_remove_support_bleed_seen_at_a_shifted_floor():
    plane = horizontal_support_plane()
    top = carton_points(x_range=(0.0, 1.0), y_range=(0.0, 1.0), height=0.33)
    bleed = carton_points(x_range=(1.0, 1.3), y_range=(0.0, 1.0), height=0.035)[:6400]
    observations = [ObservationPoints(1, np.vstack([top, bleed]))]
    shifted, _ = fused_carton_footprint(observations, plane, frame_offsets_m={1: 0.03})
    global_only, _ = fused_carton_footprint(observations, plane, frame_offsets_m={1: 0.0})
    assert shifted.area == pytest.approx(1.0, rel=0.01)
    assert global_only.area == pytest.approx(1.3, rel=0.01)
```

Append to `test/test_da3_footprint_stage.py`:

```python
def test_stage_records_per_frame_support_offsets(monkeypatch, tmp_path):
    report = _run_and_load(tmp_path, monkeypatch)
    frames = report["plane"]["frame_support_offsets"]["frames"]
    assert set(frames) == {"0", "1", "2"}
    assert all(item["source"] == "frame" for item in frames.values())
    assert all(abs(item["offset_m"]) < 1e-6 for item in frames.values())
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run python -m pytest -q -p no:cacheprovider test/test_ground_stack_footprint.py test/test_da3_footprint_stage.py -k "frame_support_offsets or shifted_floor"`
Expected: collection error in `test_ground_stack_footprint.py` (`cannot import name 'estimate_frame_support_offsets'`) and `KeyError: 'frame_support_offsets'` in the stage test. `test_frame_offsets_remove_support_bleed_seen_at_a_shifted_floor` passes once the import exists, because Task 2 already added the parameter; it pins the behaviour and holds for every D4 candidate.

- [x] **Step 3: Implement**

Add to `utils/ground_stack_footprint.py`:

```python
HORIZONTAL_NORMAL_TOLERANCE_DEG = 15.0
FRAME_OFFSET_WINDOW_M = 0.080
FRAME_OFFSET_MIN_POINTS = 1_000
FRAME_OFFSET_MODE_WIDTH_M = 0.024
FRAME_OFFSET_REFINE_M = 0.012


def _horizontal_support_mask(normals: np.ndarray, up: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(normals).all(axis=1) & np.isfinite(lengths) & (lengths > 0)
    agreement = np.zeros(len(normals))
    agreement[valid] = np.abs(normals[valid] @ up) / lengths[valid]
    return valid & (agreement >= np.cos(np.deg2rad(HORIZONTAL_NORMAL_TOLERANCE_DEG)))


def estimate_frame_support_offsets(
    background_points: np.ndarray,
    background_normals: np.ndarray,
    frame_ids: np.ndarray,
    plane: SupportPlane,
    *,
    image_ids: Sequence[int],
) -> tuple[dict[int, float], dict[str, object]]:
    """Height of each frame's own support surface relative to the selected plane.

    DA3 frames disagree on absolute support height by several centimetres, so
    the 15 mm elevation filter is applied against each observation's own frame.
    A frame with fewer than 1,000 nearby horizontal points uses the selected
    plane, and the report records that source.
    """
    points = np.asarray(background_points, dtype=float)
    frames = np.asarray(frame_ids)
    heights = (points - plane.point) @ plane.normal
    near = _horizontal_support_mask(np.asarray(background_normals, dtype=float), plane.normal)
    near &= np.abs(heights) <= FRAME_OFFSET_WINDOW_M
    offsets: dict[int, float] = {}
    per_frame: dict[str, dict[str, object]] = {}
    for image_id in (int(value) for value in image_ids):
        values = np.sort(heights[near & (frames == image_id)])
        if len(values) < FRAME_OFFSET_MIN_POINTS:
            offset, source = 0.0, "selected_plane"
        else:
            ends = np.searchsorted(values, values + FRAME_OFFSET_MODE_WIDTH_M, side="right")
            start = int(np.argmax(ends - np.arange(len(values))))
            offset = float(np.median(values[start : ends[start]]))
            offset = float(np.median(values[np.abs(values - offset) <= FRAME_OFFSET_REFINE_M]))
            source = "frame"
        offsets[image_id] = offset
        per_frame[str(image_id)] = {
            "offset_m": offset,
            "source": source,
            "point_count": int(len(values)),
        }
    return offsets, {
        "window_m": FRAME_OFFSET_WINDOW_M,
        "min_points": FRAME_OFFSET_MIN_POINTS,
        "mode_width_m": FRAME_OFFSET_MODE_WIDTH_M,
        "refine_m": FRAME_OFFSET_REFINE_M,
        "frames": per_frame,
    }
```

In `src/da3_footprint_stage.py`, import `estimate_frame_support_offsets` and replace `frame_offsets = {int(image_id): 0.0 for image_id in cache_frame_ids}` with:

```python
        frame_offsets, offset_diagnostics = estimate_frame_support_offsets(
            background_points,
            background_normals,
            background_frames,
            plane,
            image_ids=[int(value) for value in cache_frame_ids],
        )
        report["plane"]["frame_support_offsets"] = offset_diagnostics
```

- [x] **Step 4: Run the tests**

Run: `uv run python -m pytest -q -p no:cacheprovider $FOOTPRINT_SUITE`
Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add utils/ground_stack_footprint.py src/da3_footprint_stage.py \
  test/test_ground_stack_footprint.py test/test_da3_footprint_stage.py
git commit -m "fix(area): measure elevation against each frame's own support height

DA3 frames place the same floor up to ~8 cm apart (video3), and the lowest
eligible candidate can be a frame-subset layer, so a single 15 mm threshold
let support bleed into footprints. Estimate a per-frame support offset near
the selected plane and record its source.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Refine the support normal; remove the tautological residual gate (R4)

**Files:**
- Modify: `utils/ground_stack_footprint.py` (constants, `_plane_axes`, `refine_support_normal`; delete the `"p95_residual"` gate entry in `select_support_plane_with_direction`)
- Modify: `src/da3_footprint_stage.py` (call order select → refine → offsets; report)
- Test: `test/test_ground_stack_footprint.py`, `test/test_directional_support_plane.py`, `test/test_da3_footprint_stage.py`

**Interfaces:**
- Consumes: `_horizontal_support_mask`, `_frame_balanced_subsample`, `_adaptive_ransac_plane`, `_fit_plane_svd`.
- Produces: `refine_support_normal(plane, background_points, background_normals, frame_ids) -> tuple[SupportPlane, dict[str, object]]`; raises `SupportPlaneSelectionError` with diagnostics; report `plane.normal_refinement` with key `angle_to_shared_up_deg`.

- [x] **Step 1: Write the failing tests** (add `refine_support_normal` to the import list)

```python
def _tilted_floor(tilt_deg: float):
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 200), np.linspace(-1.0, 1.0, 200))
    tilt = np.deg2rad(tilt_deg)
    normal = np.array([0.0, -np.sin(tilt), np.cos(tilt)])
    points = np.column_stack([x.ravel(), y.ravel(), y.ravel() * np.tan(tilt)])
    frames = np.arange(len(points)) % 3
    return points, np.tile(normal, (len(points), 1)), frames, normal


def test_support_normal_refinement_recovers_tilted_floor():
    points, normals, frames, true_normal = _tilted_floor(2.0)
    refined, diagnostics = refine_support_normal(
        horizontal_support_plane(), points, normals, frames
    )
    assert np.degrees(np.arccos(min(1.0, refined.normal @ true_normal))) < 0.01
    assert diagnostics["angle_to_shared_up_deg"] == pytest.approx(2.0, abs=0.01)
    assert abs(refined.u_axis @ refined.normal) < 1e-12
    assert abs(refined.v_axis @ refined.normal) < 1e-12


def test_support_normal_refinement_rejects_large_disagreement():
    points, normals, frames, _ = _tilted_floor(5.0)
    with pytest.raises(SupportPlaneSelectionError, match="disagrees"):
        refine_support_normal(horizontal_support_plane(), points, normals, frames)
```

Append to `test/test_directional_support_plane.py`:

```python
def test_direction_candidates_have_no_tautological_residual_gate():
    _, diagnostics = select(grid())
    candidate = diagnostics["candidates"][0]
    assert "p95_residual" not in candidate["gates"]
    assert "p95_residual_m" in candidate
```

Append to `test/test_da3_footprint_stage.py`:

```python
def test_stage_reports_support_normal_refinement(monkeypatch, tmp_path):
    report = _run_and_load(tmp_path, monkeypatch)
    refinement = report["plane"]["normal_refinement"]
    assert refinement["angle_to_shared_up_deg"] < 0.01
    assert refinement["max_angle_deg"] == 3.0
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run python -m pytest -q -p no:cacheprovider test/test_ground_stack_footprint.py test/test_directional_support_plane.py test/test_da3_footprint_stage.py -k "refinement or tautological"`
Expected: FAIL (import error, then `p95_residual` present, then `KeyError: 'normal_refinement'`).

- [x] **Step 3: Implement**

In `select_support_plane_with_direction` delete the line `"p95_residual": plane.p95_residual_m <= 0.010,` from `gates` (keep `p95_residual_m` in `info`).

Add:

```python
SUPPORT_NORMAL_BAND_M = 0.100
SUPPORT_NORMAL_RANSAC_THRESHOLD_M = 0.012
SUPPORT_NORMAL_MIN_INLIERS = 10_000
SUPPORT_NORMAL_MAX_ANGLE_DEG = 3.0
SUPPORT_NORMAL_SEED = 29


def _plane_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u_axis = np.cross(normal, reference_axis)
    u_axis /= np.linalg.norm(u_axis)
    return u_axis, np.cross(normal, u_axis)


def refine_support_normal(
    plane: SupportPlane,
    background_points: np.ndarray,
    background_normals: np.ndarray,
    frame_ids: np.ndarray,
) -> tuple[SupportPlane, dict[str, object]]:
    """Fit the selected support's own normal; the shared up only seeds the search.

    The +/-100 mm band is much thicker than DA3 depth noise, so the fit is not
    forced toward the seed direction by the band itself. Disagreement above
    3 degrees means the shared orientation does not describe this support.
    """
    points = np.asarray(background_points, dtype=float)
    frames = np.asarray(frame_ids)
    up = plane.normal
    band = _horizontal_support_mask(np.asarray(background_normals, dtype=float), up)
    band &= np.abs((points - plane.point) @ up) <= SUPPORT_NORMAL_BAND_M
    diagnostics: dict[str, object] = {
        "method": "seeded_ransac_svd",
        "band_m": SUPPORT_NORMAL_BAND_M,
        "ransac_threshold_m": SUPPORT_NORMAL_RANSAC_THRESHOLD_M,
        "min_inliers": SUPPORT_NORMAL_MIN_INLIERS,
        "max_angle_deg": SUPPORT_NORMAL_MAX_ANGLE_DEG,
        "band_point_count": int(np.count_nonzero(band)),
    }
    if np.count_nonzero(band) < SUPPORT_NORMAL_MIN_INLIERS:
        raise SupportPlaneSelectionError(
            "support normal refinement has too few band points", diagnostics
        )
    sampled, _ = _frame_balanced_subsample(points[band], frames[band], maximum=50_000)
    ransac = _adaptive_ransac_plane(
        sampled, threshold_m=SUPPORT_NORMAL_RANSAC_THRESHOLD_M, seed=SUPPORT_NORMAL_SEED
    )
    inliers = sampled[
        np.abs((sampled - ransac.point) @ ransac.normal) <= SUPPORT_NORMAL_RANSAC_THRESHOLD_M
    ]
    diagnostics["sample_count"] = int(len(sampled))
    diagnostics["inlier_count"] = int(len(inliers))
    if len(inliers) < SUPPORT_NORMAL_MIN_INLIERS:
        raise SupportPlaneSelectionError(
            "support normal refinement has too few inliers", diagnostics
        )
    _, normal = _fit_plane_svd(inliers)
    if normal @ up < 0:
        normal = -normal
    angle = float(np.degrees(np.arccos(np.clip(normal @ up, -1.0, 1.0))))
    diagnostics["angle_to_shared_up_deg"] = angle
    diagnostics["normal"] = normal.tolist()
    if angle > SUPPORT_NORMAL_MAX_ANGLE_DEG:
        raise SupportPlaneSelectionError(
            "support normal disagrees with the shared up direction", diagnostics
        )
    u_axis, v_axis = _plane_axes(normal)
    return (
        SupportPlane(
            point=plane.point,
            normal=normal,
            u_axis=u_axis,
            v_axis=v_axis,
            inlier_count=plane.inlier_count,
            inlier_fraction=plane.inlier_fraction,
            p95_residual_m=plane.p95_residual_m,
        ),
        diagnostics,
    )
```

In `src/da3_footprint_stage.py`, import `refine_support_normal`. After the existing `try/except` around `select_support_plane_with_direction` and before `report["plane"] = {...}`, insert:

```python
        try:
            plane, normal_refinement = refine_support_normal(
                plane, background_points, background_normals, background_frames
            )
        except SupportPlaneSelectionError as error:
            report["plane"] = {
                "selected": None,
                **plane_diagnostics,
                "normal_refinement": error.diagnostics,
            }
            raise
```

and add `"normal_refinement": normal_refinement,` to the `report["plane"]` dict after `**plane_diagnostics`. Its `selected` entry is built from the refined `plane`. The Task 5 offset block stays after this, so offsets are measured along the refined normal.

- [x] **Step 4: Run the tests**

Run: `uv run python -m pytest -q -p no:cacheprovider $FOOTPRINT_SUITE`
Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add utils/ground_stack_footprint.py src/da3_footprint_stage.py test/test_ground_stack_footprint.py \
  test/test_directional_support_plane.py test/test_da3_footprint_stage.py
git commit -m "fix(area): refine and gate the support normal

The projection direction came unchecked from the Viewer heuristic, and the
p95 residual gate could never fail because it was evaluated on points
already filtered to the same 10 mm bound. A 2 degree tilt inflates a 1.2 m
column union by 5.5%. Refit the selected support's normal with seeded
RANSAC and SVD, reject disagreement above 3 degrees, and drop the dead gate.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Stop reporting programming errors as rejections (R5)

**Files:**
- Modify: `src/da3_footprint_stage.py:326-332`
- Test: `test/test_da3_footprint_stage.py`

**Interfaces:**
- Consumes: the stage's `fused_carton_footprint` import.
- Produces: `KeyError` propagates out of `run_da3_footprint`.

- [x] **Step 1: Write the failing test**

```python
def test_programming_key_error_is_not_reported_as_a_measurement_rejection(
    monkeypatch, tmp_path
):
    dataset, save_root, _ = make_metric_fixture(tmp_path)

    def broken(*args, **kwargs):
        raise KeyError("programming error")

    monkeypatch.setattr(stage, "fused_carton_footprint", broken)
    with pytest.raises(KeyError, match="programming error"):
        stage.run_da3_footprint(str(dataset), save_root)
```

- [x] **Step 2: Run it to verify it fails**

Run: `uv run python -m pytest -q -p no:cacheprovider test/test_da3_footprint_stage.py -k programming_key_error`
Expected: FAIL, `DID NOT RAISE <class 'KeyError'>`.

- [x] **Step 3: Implement**

Change the `except` tuple to `(FootprintStageError, FootprintError, OSError, ValueError)`. `ValueError` stays because `utils/scene_orientation.py` and `utils/detection_objects.py` report input-contract violations as plain `ValueError`.

- [x] **Step 4: Run the tests**

Run: `uv run python -m pytest -q -p no:cacheprovider $FOOTPRINT_SUITE`
Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add src/da3_footprint_stage.py test/test_da3_footprint_stage.py
git commit -m "fix(area): let programming KeyErrors propagate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8 (only if D1 = B): Exclude failed IDs and require geometric coverage (R6)

**Files:**
- Modify: `utils/ground_stack_footprint.py` (constants, `classify_excluded_coverage`)
- Modify: `src/da3_footprint_stage.py` (per-ID loop, union, coverage gate, report)
- Modify: `utils/footprint_evidence.py` (`_fixed_plane_variant_area` evaluates only formally included IDs)
- Test: `test/test_ground_stack_footprint.py`, `test/test_da3_footprint_stage.py`

**Interfaces:**
- Consumes: `elevated_projected_points`, `ObservationPoints`, `union_footprints`.
- Produces:
  - `classify_excluded_coverage(observations, plane, frame_offsets_m, union) -> {"status": "covered"|"uncovered"|"unverifiable", "covered_fraction": float | None, "projected_point_count": int}`
  - per-ID `status: "included"|"excluded"`, per-ID `coverage` for excluded IDs
  - report `coverage = {"included_count", "excluded_count", "covered_excluded_count", "uncovered_excluded_ids", "unverifiable_excluded_ids", "buffer_m", "min_covered_fraction"}`

- [x] **Step 1: Write the failing tests**

Add `from shapely.geometry import box` and `classify_excluded_coverage` to `test/test_ground_stack_footprint.py`:

```python
def _front_face(x0: float, x1: float) -> np.ndarray:
    return np.column_stack(
        [np.linspace(x0, x1, 200), np.full(200, 0.5), np.linspace(0.05, 0.3, 200)]
    )


def test_excluded_coverage_classifies_inside_outside_and_empty():
    plane = horizontal_support_plane()
    union = box(0.0, 0.0, 1.0, 1.0)
    inside = classify_excluded_coverage(
        [ObservationPoints(0, _front_face(0.2, 0.8))], plane, {0: 0.0}, union
    )
    outside = classify_excluded_coverage(
        [ObservationPoints(0, _front_face(1.6, 1.9))], plane, {0: 0.0}, union
    )
    empty = classify_excluded_coverage(
        [ObservationPoints(0, np.empty((0, 3)))], plane, {0: 0.0}, union
    )
    assert inside["status"] == "covered" and inside["covered_fraction"] == 1.0
    assert outside["status"] == "uncovered" and outside["covered_fraction"] == 0.0
    assert empty == {
        "status": "unverifiable",
        "covered_fraction": None,
        "projected_point_count": 0,
    }
```

In `test/test_da3_footprint_stage.py`, change the fixture signature to `def make_metric_fixture(tmp_path: Path, mask_factory=None, extra_face=None)`. Directly after the `boxes = {...}` literal add:

```python
    if extra_face is not None:
        boxes[0].append(extra_face[0])
```

After the three `_put_carton(...)` calls add:

```python
    if extra_face is not None:
        _put_vertical_face(world_points, 0, *extra_face)
```

After the `mapping = {...}` literal add:

```python
    if extra_face is not None:
        mapping["3"] = [{"image_id": 0, "object_id": 1, "bbox": extra_face[0]}]
```

Add the helper next to `_put_carton`:

```python
def _put_vertical_face(
    world_points: np.ndarray,
    frame: int,
    bbox: list[int],
    x_range: tuple[float, float],
    depth: float,
) -> None:
    """A front face with no observable depth: it projects to a line."""
    x1, y1, x2, y2 = bbox
    heights = np.linspace(0.30, 0.05, y2 - y1)
    x_values = np.linspace(*x_range, x2 - x1)
    x_grid, height_grid = np.meshgrid(x_values, heights, indexing="xy")
    world_points[frame, y1:y2, x1:x2] = np.stack(
        [x_grid, 1.0 - height_grid, np.full_like(x_grid, 3.0 + depth)], axis=-1
    )
```

Add the stage tests:

```python
def test_excluded_front_face_inside_union_does_not_block_publication(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(
        tmp_path, extra_face=([88, 30, 120, 70], (0.2, 0.8), 0.5)
    )
    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())
    assert result["success"] is True
    assert report["value_m2"] == pytest.approx(1.5, abs=0.03)
    face = report["per_global_id"]["3"]
    assert face["status"] == "excluded"
    assert face["coverage"]["status"] == "covered"
    assert report["coverage"]["included_count"] == 2


def test_excluded_front_face_outside_union_rejects(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(
        tmp_path, extra_face=([88, 30, 120, 70], (1.6, 1.9), 0.5)
    )
    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())
    assert result["success"] is False
    assert report["value_m2"] is None
    assert report["coverage"]["uncovered_excluded_ids"] == ["3"]
    assert "not covered" in report["rejection_reason"]
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run python -m pytest -q -p no:cacheprovider test/test_ground_stack_footprint.py test/test_da3_footprint_stage.py -k "excluded_coverage or excluded_front_face"`
Expected: FAIL (import error; then the inside case is rejected with `one or more global ids were rejected: 3`).

- [x] **Step 3: Implement the classifier**

Add `import shapely` to `utils/ground_stack_footprint.py`, then:

```python
COVERAGE_BUFFER_M = 0.020
COVERAGE_MIN_FRACTION = 0.95


def classify_excluded_coverage(
    observations: Sequence[ObservationPoints],
    plane: SupportPlane,
    frame_offsets_m: Mapping[int, float],
    union: Polygon | MultiPolygon,
) -> dict[str, object]:
    """Decide whether an excluded ID's visible footprint lies inside the included union."""
    projected = elevated_projected_points(observations, plane, frame_offsets_m)
    if not len(projected):
        return {"status": "unverifiable", "covered_fraction": None, "projected_point_count": 0}
    covered = shapely.covers(union.buffer(COVERAGE_BUFFER_M), shapely.points(projected))
    fraction = float(np.mean(covered))
    return {
        "status": "covered" if fraction >= COVERAGE_MIN_FRACTION else "uncovered",
        "covered_fraction": fraction,
        "projected_point_count": int(len(projected)),
    }
```

- [x] **Step 4: Implement the stage rule**

Import `classify_excluded_coverage`, `COVERAGE_BUFFER_M`, `COVERAGE_MIN_FRACTION`. Replace the Task 2 per-ID loop and the following `missing` check through `union = union_footprints(...)` with:

```python
        excluded_ids: list[str] = []
        for global_id, observations in masked_observations.items():
            per_id = report["per_global_id"][global_id]
            try:
                polygon, metrics = fused_carton_footprint(
                    observations, plane, frame_offsets_m=frame_offsets
                )
            except FootprintError as error:
                per_id["status"] = "excluded"
                per_id["rejection"] = str(error)
                per_id.update(error.diagnostics)
                excluded_ids.append(global_id)
                continue
            per_id["status"] = "included"
            per_id.update(metrics)
            polygons[global_id] = polygon
        if not polygons:
            raise FootprintStageError("no global id produced a trusted footprint")
        precise_polygons = {
            key: set_precision(polygon, 0.0001) for key, polygon in polygons.items()
        }
        union = union_footprints(list(precise_polygons.values()))
        coverage = {
            global_id: classify_excluded_coverage(
                masked_observations[global_id], plane, frame_offsets, union
            )
            for global_id in excluded_ids
        }
        for global_id, item in coverage.items():
            report["per_global_id"][global_id]["coverage"] = item
        uncovered = sorted(
            (key for key, item in coverage.items() if item["status"] == "uncovered"),
            key=_global_id_key,
        )
        unverifiable = sorted(
            (key for key, item in coverage.items() if item["status"] == "unverifiable"),
            key=_global_id_key,
        )
        report["coverage"] = {
            "included_count": len(polygons),
            "excluded_count": len(excluded_ids),
            "covered_excluded_count": len(excluded_ids) - len(uncovered) - len(unverifiable),
            "uncovered_excluded_ids": uncovered,
            "unverifiable_excluded_ids": unverifiable,
            "buffer_m": COVERAGE_BUFFER_M,
            "min_covered_fraction": COVERAGE_MIN_FRACTION,
        }
        if uncovered or unverifiable:
            raise FootprintStageError(
                "excluded global ids are not covered by included footprints: "
                + ", ".join(uncovered + unverifiable)
            )
```

The existing `coarse_union` / sensitivity block follows unchanged and still uses `polygons` and `precise_polygons`.

- [x] **Step 5: Align mask-robustness evidence**

In `_fixed_plane_variant_area` replace the `expected_ids` construction and its two checks with:

```python
    expected_ids = {str(global_id) for global_id in formal_snapshot.polygons}
    if not expected_ids:
        raise FootprintError(f"{variant} robustness has no formally included global ids")
    if not expected_ids <= set(grouped):
        raise FootprintError(f"{variant} robustness is missing one or more global ids")
```

Mask robustness now measures how the published (included) geometry responds to eroded/dilated masks.

- [x] **Step 6: Run the tests**

Run: `uv run python -m pytest -q -p no:cacheprovider $FOOTPRINT_SUITE`
Expected: all pass. If an existing stage test asserted that rejected-formal mask variants are `accepted`, update it to expect `rejected` with reason `robustness has no formally included global ids` and say so in the commit body.

- [x] **Step 7: Commit**

```bash
git add utils/ground_stack_footprint.py src/da3_footprint_stage.py utils/footprint_evidence.py \
  test/test_ground_stack_footprint.py test/test_da3_footprint_stage.py
git commit -m "feat(area): publish when excluded IDs are covered by the included union

A single unobservable or fragmented ID rejected the whole measurement, so
no real scene was ever accepted. Exclude IDs without a trusted OBB and
publish only if each excluded ID's elevated points lie at least 95% inside
the included union buffered by 20 mm; record coverage per ID.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9 (only if D2 = relative): Second-component threshold relative to the largest (R7)

**Files:**
- Modify: `utils/ground_stack_footprint.py:716-719` (`select_footprint_component`)
- Test: `test/test_ground_stack_footprint.py` (replace `test_absolute_32_point_second_component_is_rejected`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `component_diagnostics.substantial_component_threshold = ceil(0.20 * largest)` and `rule: "fraction_of_largest"`.

- [x] **Step 1: Write the failing tests** (replace `test_absolute_32_point_second_component_is_rejected`)

```python
def test_small_second_component_relative_to_largest_is_ignored():
    first_component = connected_component_points(168, x_offset=0.0)
    second_component = connected_component_points(32, x_offset=1.0)
    polygon, metrics = carton_footprint_polygon(
        np.vstack([first_component, second_component]), horizontal_support_plane()
    )
    assert polygon.area == pytest.approx(0.06 * 0.23, rel=0.05)
    assert metrics["component_diagnostics"]["substantial_component_threshold"] == 34


def test_second_component_at_twenty_percent_of_largest_is_rejected():
    first_component = connected_component_points(100, x_offset=0.0)
    second_component = connected_component_points(20, x_offset=1.0)
    with pytest.raises(FootprintError, match="multiple substantial components"):
        carton_footprint_polygon(
            np.vstack([first_component, second_component]), horizontal_support_plane()
        )
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run python -m pytest -q -p no:cacheprovider test/test_ground_stack_footprint.py -k "relative_to_largest or twenty_percent"`
Expected: the first FAILS (`multiple substantial components`).

- [x] **Step 3: Implement**

Replace the two threshold lines in `select_footprint_component` with:

```python
    threshold = int(np.ceil(0.20 * int(populations.max())))
    diagnostics["substantial_component_threshold"] = threshold
    diagnostics["rule"] = "fraction_of_largest"
```

- [x] **Step 4: Run the tests**

Run: `uv run python -m pytest -q -p no:cacheprovider $FOOTPRINT_SUITE`
Expected: all pass; `test_low_count_substantial_second_component_is_rejected` (35 vs 29) still rejects.

- [x] **Step 5: Commit**

```bash
git add utils/ground_stack_footprint.py test/test_ground_stack_footprint.py
git commit -m "fix(area): judge a second DBSCAN component against the largest one

min(32, 20%) made any 32-voxel blob reject a carton regardless of the main
component (8 of 38 video3 multi-component rejections had second/largest
below 10%). A second component now counts only at 20% of the largest.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Report schema, algorithm document, real-data validation

**Files:**
- Modify: `src/da3_footprint_stage.py:121` (`"schema_version": "3.0.0"`)
- Modify: `test/test_da3_footprint_stage.py:600` (expect `"3.0.0"`)
- Modify: `docs/ground_stack_footprint_algorithm.md`

**Interfaces:**
- Consumes: everything above; `run.sh` and `summarize.py` from Task 0.
- Produces: schema `3.0.0`; updated contract; validation record.

- [x] **Step 1: Bump the schema** in the stage and in the assertion at `test/test_da3_footprint_stage.py:600`, then run `uv run python -m pytest -q -p no:cacheprovider $FOOTPRINT_SUITE`. Expected: all pass.

- [x] **Step 2: Update the algorithm document**

In `docs/ground_stack_footprint_algorithm.md`:

1. Section 3, after the orientation reading rules, add:

```markdown
方向拟合（`viewer-plane-v2`）使用固定种子、固定 1,000 次试验的 numpy RANSAC，并对内点做 SVD 复拟合，同一输入逐位可复现。v1 使用 Open3D `segment_plane`，即使设置种子，其并行采样也不可复现；v1 sidecar 会被判为 stale 并重算。
```

2. In section 3, item 4, delete the residual-gate wording; the gates are inlier count, frame span, in-plane span, hull area and object-above fraction.
3. After item 6 of section 3, add:

```markdown
7. **法向精炼。** 在选中平面 ±100 mm 内取水平支持点（按帧均衡，最多 50,000），用固定种子 RANSAC（12 mm）求内点，对内点做 SVD 得到支撑面自身法向，并以此作为投影方向。内点少于 10,000，或与共享 up 夹角超过 3° 时拒绝。报告字段为 `plane.normal_refinement`。带宽与 3° 均为初始工程参数。原 p95 残差门先把点筛到 ≤10 mm 再比较 p95 与 10 mm，永远通过，已删除。
8. **逐帧支撑高度。** DA3 各帧对同一地面的绝对高度可相差数厘米：2026-10 video3 实测逐帧中位高度跨度 81 mm，"最低合格平面"选中的是 11/29 帧支持的错层。对每帧取精炼平面 ±80 mm 内的水平支持点，用 24 mm 滑窗找最密模式，再以 ±12 mm 中位数精炼，得到该帧 offset。少于 1,000 点的帧使用选中平面（`source: selected_plane`）。报告字段为 `plane.frame_support_offsets`。"最低可见水平支撑面"的选择语义不变，但 15 mm 高度过滤改为相对各观测所属帧的支撑面。
```

4. Replace section 4 items 1, 2 and 7 with:

```markdown
1. 有效点少于 32 的观测不参与融合，记录在 `excluded_observations`；只有在没有任何合格观测时，该 `global_id` 才失败。
2. 只保留高于所属帧支撑面 15 mm 的点。
7. （D4 = B）先对组件拟合 minimum rotated rectangle，在该矩形自身的两条轴上各裁掉 0.5% 的极端点，再对剩余点重新拟合。原先在任意 u/v 轴上做 1%/99% 分位裁剪，会随纸箱朝向产生 0 至 −4% 的面积偏差，旋转后贴边稀疏点链可放大到 +7.5%；新方法在合成测试中偏差为 −0.4% 至 −1.9%，且与密度无关。
```

If D4 = C, use this item 7 instead:

```markdown
7. （D4 = C）在组件内取密度核心点：半径为第 4 近邻距离 90 分位数的 1.5 倍，取 DBSCAN `min_samples=5` 的核心点，再对其拟合 minimum rotated rectangle。点数占比低于约 10% 的稀疏区域可能被整体丢弃（合成测试中最坏 −22.9%）。
```

5. If Task 9 ran, replace section 4 item 6 with: `若次大分量点数不低于最大分量的 20%，视为混入其他物体或严重断裂，该 global_id 失败。`
6. If Task 8 ran, replace the Fail-closed opening sentence and list with:

```markdown
未能生成可信 OBB 的 `global_id` 被排除。只有当每个被排除 ID 的高架投影点至少 95% 落在已纳入 OBB 并集外扩 20 mm 的范围内（`covered`）时才发布面积；任何 `uncovered` 或没有投影点的 `unverifiable` ID 都会使整次测量被拒绝。报告字段为 `coverage` 以及逐 ID 的 `status` 和 `coverage`。
```

7. In "已知适用边界", replace the bullet beginning `5 mm voxel 当前保留首个观测点` with:

```markdown
- 5 mm voxel 仍保留首个观测点，结果可能随点顺序有亚体素变化。OBB 范围估计的已知偏差见第 4 节第 7 条。
- 法向精炼只纠正 ≤3° 的方向偏差。逐帧 offset 只补偿帧间高度不一致，不修复帧间水平错位；多视角融合的 OBB 仍会被帧间水平错位放大。
```

8. In "输出与复核", state that the report schema is `3.0.0`.

- [x] **Step 3: Run the real-data validation**

```bash
bash /home/xingyu/3D_Recognization/runtime/footprint-fix-validation-20261013/run.sh fixed
uv run python /home/xingyu/3D_Recognization/runtime/footprint-fix-validation-20261013/summarize.py baseline
uv run python /home/xingyu/3D_Recognization/runtime/footprint-fix-validation-20261013/summarize.py fixed
```

Expected: five `fixed` lines with `schema` `3.0.0`, a `normal_angle_deg` value and a `frame_offset_range_m` value for every dataset whose plane was selected. Acceptance status is not predetermined; there is no ground-truth area for these scenes.

- [x] **Step 4: Record the validation**

Append to `docs/ground_stack_footprint_algorithm.md`:

```markdown
## 验证记录（2026-10-13 修复）

fd4–fd8，CPU，SAM3.1 mask cache；baseline 为 `main` 原代码，fixed 为本次修复。没有独立实测面积，下表只说明发布与诊断行为，不代表准确率。
```

followed by a Markdown table built from the two `summarize.py` outputs. Columns: dataset, baseline status / value_m2, fixed status / value_m2, fixed id_outcomes, normal_angle_deg, frame_offset_range_m. Copy the numbers verbatim.

- [x] **Step 5a: Final combined six-file suite** (196 passed in 86.05s; reused after documentation-only edits).

- [x] **Step 5b: Final Task10 commit** (coordinator-owned; schema, documentation and this completed plan are committed together).

- [x] **Whole-branch independent review and finalization** (spec/code/docs/evidence PASS; `final-code-review.md` and `final-review.md` in the retained SDD records).

Run: `uv run python -m pytest -q -p no:cacheprovider $FOOTPRINT_SUITE test/test_web_viewer_export.py`
Expected: all pass.

```bash
git add src/da3_footprint_stage.py test/test_da3_footprint_stage.py docs/ground_stack_footprint_algorithm.md
git commit -m "docs(area): document footprint schema 3.0.0 and record fd4-fd8 validation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Out of scope

- Ground-truth accuracy evaluation (needs independently measured areas).
- Option C (per-stack footprint) and any multi-view horizontal-misalignment correction.
- Deleting the unused legacy plane selector (`fit_support_plane`, `select_support_plane`, `_refine_support_plane`, `_orient_for_objects`, `_best_ransac_plane`) and its tests.
- Docker image rebuild and any push.

## Actual decisions and validation (2026-10-13; supersedes illustrative expected results)

Tasks0–10 completed. Tasks0–9 have coordinator commits through `4e0f1df`; Task10 schema/documentation and this completion record are committed together. Per-task reports and review fixes are in `.superpowers/sdd/2026-10-13-footprint-area-fixes/`. The preceding implementation snippets and expected numerical examples are planning material, not evidence of the final behavior.

- D1=B: every excluded ID must have >=95% of its visible elevated projected points in the included union buffered20mm; hidden depth and undetected objects remain unmeasured. D2 uses `ceil(.20*largest)` with inclusive rejection. D3 uses numpy seeded RANSAC+SVD, cache algorithm viewer-plane-v2; cross-platform bitwise equality is not claimed.
- D4=B selected after actual formal per-ID comparisons: B/C paired ratios <0.8 were0/7; newly rejected old-accepted IDs18/22, newly accepted old-rejected20/26. Gates remain unchanged. See `runtime/footprint-fix-validation-20261013/obb-capture/summaries.json`, `obb-outliers.json`, and task-4-report.md. Synthetic fixtures establish bounded tested bias, not density independence or exact rotation invariance.
- Review corrections: Task3 evidence arithmetic checks now exclude weak observations before numeric guards; Task8 restored local polygons initialization and tested included-ID mask robustness; Task10 outcome summary marks observation-only entries not_evaluated instead of ok. Failure receipts remain retained.
- Task10 report schema3.0.0 differs from DA3 cache schema-v3/SAM v2. Combined final six-file suite once:196 passed in86.05s, exit0 (`task10-logs/suite.log`). One CPU fixed replay of fd4–fd8: each CLI exit2; overall harness exit0. Each is rejected for uncovered/unverifiable excluded IDs; value_m2 null. No independently measured area or accuracy claim.
- Baseline ok/total334/702,132/229,222/299,269/518,73/248; fixed333/702,153/229,232/299,290/518,110/248. Final angles and offset ranges, full ID outcomes and coverage counts are recorded verbatim in `docs/ground_stack_footprint_algorithm.md` and baseline-summary.jsonl/fixed-summary.jsonl under the validation root.
- Final independent review passed specification, code, documentation and evidence checks, including the ten baseline/fixed raw reports. Coordinator owns the final local commit; branch/worktree and evidence are retained. No push, merge, model inference, installs or Docker rebuild.
- Coordination rulings and process deviations are retained in `.superpowers/sdd/2026-10-13-footprint-area-fixes/progress.md`: existing uv environment/cache reuse; worker write boundaries with coordinator commits; formal-ID-only D4 capture; evidence guard/offset alignment; explicit selection-statistics labeling; exact-floor fixture correction; corrected unevaluated-ID summaries; consolidated final testing. Task9 unnecessarily repeated its successful suite after two diagnostic assertions; both receipts are retained and no further unchanged suite was run.
