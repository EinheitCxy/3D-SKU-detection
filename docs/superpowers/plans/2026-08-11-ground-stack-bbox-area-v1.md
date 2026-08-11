# Ground-stack bbox area V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only CLI mode that sums one calibrated bbox observation per deduplicated physical object and writes an auditable `cm²/m²` ground-stack measurement report.

**Architecture:** Keep calibration and selection logic in a pure NumPy utility so it is independently testable without models or files. A small stage reads `global_mapping.json` and detections, writes fresh report/overlay artifacts, and never mutates upstream files. `main.py` adds only CLI parsing and dispatch; it does not chain detection, matching, or reconstruction.

**Tech Stack:** Python 3.11, NumPy, OpenCV, Pillow, pytest, project `uv` environment.

## Global Constraints

- Metric name is exactly `calibrated_bbox_area_sum`; it is an arithmetic sum of one bbox-equivalent physical area per global ID, not a bbox union, mask, footprint, or package surface area.
- Calibration consumes one detected anchor bbox plus positive physical `width_cm` and `height_cm`; all measured fronts must be approximately coplanar with that anchor.
- Read `detections_results` and `global_mapping.json` only; never write or rewrite either input.
- New artifacts are confined to `<save_root>/<dataset>/ground_stack_area/`.
- Reject invalid dimensions, boxes, calibration, and empty valid-instance results; never silently clamp or substitute a value.
- Use `uv` for all Python commands; on this host use `VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv`, `UV_CACHE_DIR=/tmp/ground-stack-area-uv-cache`, and `uv run --active` for focused tests, because the isolated worktree has no checked-in virtual environment.
- Do not invoke GPU models and do not add CPU fallback behavior.
- Update both root `README.md` and `code/README.md` with the final CLI, definition, assumptions, outputs, and validation scope.

---

### Task 1: Pure calibrated-bbox area model

**Files:**
- Create: `code/tests/test_ground_stack_area.py`
- Create: `code/utils/ground_stack_area.py`

**Interfaces:**
- Consumes: a four-number bbox `(x1, y1, x2, y2)`, anchor physical width/height, and mapping records with `global_id`, `image_id`, `object_id`, and `bbox`.
- Produces: `BBoxAreaError`, `PlanarCalibration`, `calibrate_from_anchor()`, `calibrated_bbox_area_cm2()`, and `select_best_instances()`.
- Used by: `modules/ground_stack_area_stage.py` in Task 2.

- [ ] **Step 1: Write the failing pure-logic tests**

```python
import pytest

from utils.ground_stack_area import (
    BBoxAreaError,
    calibrate_from_anchor,
    calibrated_bbox_area_cm2,
    select_best_instances,
)


def test_anchor_bbox_maps_to_its_known_physical_area():
    calibration = calibrate_from_anchor([10, 20, 110, 70], 20.0, 10.0)
    assert calibrated_bbox_area_cm2([10, 20, 110, 70], calibration) == pytest.approx(200.0)


def test_select_best_instances_counts_each_global_id_once():
    selected, rejected = select_best_instances(
        {
            "1": [
                {"image_id": 0, "object_id": 0, "bbox": [0, 0, 10, 10]},
                {"image_id": 1, "object_id": 2, "bbox": [0, 0, 20, 20]},
            ],
            "2": [{"image_id": 1, "object_id": 3, "bbox": [0, 0, 5, 10]}],
        }
    )
    assert [(item.global_id, item.bbox) for item in selected] == [
        ("1", (0.0, 0.0, 20.0, 20.0)),
        ("2", (0.0, 0.0, 5.0, 10.0)),
    ]
    assert rejected == []


@pytest.mark.parametrize("bbox", ([0, 0, 0, 1], [0, 0, float("nan"), 1]))
def test_invalid_bbox_is_rejected(bbox):
    with pytest.raises(BBoxAreaError, match="bbox"):
        calibrate_from_anchor(bbox, 20.0, 10.0)
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/ground-stack-area-uv-cache uv run --active pytest tests/test_ground_stack_area.py -q
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'utils.ground_stack_area'`.

- [ ] **Step 3: Write the minimum pure implementation**

```python
class BBoxAreaError(ValueError):
    pass


@dataclass(frozen=True)
class PlanarCalibration:
    source_bbox: tuple[float, float, float, float]
    width_cm: float
    height_cm: float


def calibrate_from_anchor(bbox, width_cm, height_cm) -> PlanarCalibration:
    validated_bbox = validate_bbox(bbox)
    validate_positive_dimension(width_cm, "width_cm")
    validate_positive_dimension(height_cm, "height_cm")
    return PlanarCalibration(validated_bbox, float(width_cm), float(height_cm))


def calibrated_bbox_area_cm2(bbox, calibration: PlanarCalibration) -> float:
    x1, y1, x2, y2 = validate_bbox(bbox)
    ax1, ay1, ax2, ay2 = calibration.source_bbox
    scale_x = calibration.width_cm / (ax2 - ax1)
    scale_y = calibration.height_cm / (ay2 - ay1)
    return (x2 - x1) * (y2 - y1) * scale_x * scale_y
```

Implement `select_best_instances()` by validating every mapping entry, selecting the largest valid source pixel area for each global ID, and returning deterministic global-ID order plus structured rejection records. Do not import OpenCV or perform filesystem I/O in this module.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run the Step 2 command again.

Expected: PASS with all four tests passing.

- [ ] **Step 5: Commit the pure model**

```bash
git add code/utils/ground_stack_area.py code/tests/test_ground_stack_area.py
git commit -m "feat: add calibrated bbox area model"
```

### Task 2: Read-only measurement stage and artifacts

**Files:**
- Modify: `code/tests/test_ground_stack_area.py`
- Create: `code/modules/ground_stack_area_stage.py`

**Interfaces:**
- Consumes: `run_ground_stack_area(dataset_path: str, save_root: Path, anchor_frame: int, anchor_object: int, anchor_width_cm: float, anchor_height_cm: float) -> dict`.
- Uses: Task 1 calibration/selection functions and existing `modules.deduplicate_detections.load_detection_objects()`.
- Produces: `<save_root>/<dataset>/ground_stack_area/{measurement_report.json,selected_instances.json,annotated_frames/}` and a result dictionary with `success`, `status`, and `report_path`.
- Used by: CLI dispatch in Task 3.

- [ ] **Step 1: Extend the test file with a failing stage test**

```python
import json

import cv2
import numpy as np

from modules.ground_stack_area_stage import run_ground_stack_area


def test_stage_writes_report_without_mutating_inputs(tmp_path):
    dataset = tmp_path / "stack"
    images = dataset / "images"
    detections = dataset / "detections_results"
    images.mkdir(parents=True)
    detections.mkdir()
    cv2.imwrite(str(images / "0.jpg"), np.full((100, 100, 3), 255, dtype=np.uint8))
    (detections / "0.json").write_text(json.dumps({"skus": [{"objects": [
        {"position": [10, 10, 30, 20]}, {"position": [40, 10, 60, 20]}
    ]}]}), encoding="utf-8")
    save_root = tmp_path / "Output"
    mapping_path = save_root / "stack" / "dedup_detections" / "global_mapping.json"
    mapping_path.parent.mkdir(parents=True)
    mapping = {"1": [{"image_id": 0, "object_id": 0, "bbox": [10, 10, 30, 20], "removed": False}], "2": [{"image_id": 0, "object_id": 1, "bbox": [40, 10, 60, 20], "removed": False}]}
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    before = mapping_path.read_bytes()

    result = run_ground_stack_area(str(dataset), save_root, 0, 0, 20.0, 10.0)

    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert result["success"] is True
    assert report["value_cm2"] == pytest.approx(400.0)
    assert mapping_path.read_bytes() == before
    assert (Path(result["report_path"]).parent / "annotated_frames" / "0.jpg").is_file()
```

- [ ] **Step 2: Run the stage test to verify RED**

Run:

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/ground-stack-area-uv-cache uv run --active pytest tests/test_ground_stack_area.py::test_stage_writes_report_without_mutating_inputs -q
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'modules.ground_stack_area_stage'`.

- [ ] **Step 3: Write the minimum stage**

```python
def run_ground_stack_area(
    dataset_path: str,
    save_root: Path,
    anchor_frame: int,
    anchor_object: int,
    anchor_width_cm: float,
    anchor_height_cm: float,
) -> dict:
    """Measure one calibrated bbox per global ID without mutating inputs."""
```

Load the anchor detection using `load_detection_objects()`. Read the mapping as JSON, calibrate/select/count through Task 1, then write the two JSON artifacts and a CV2 annotation only under `ground_stack_area/`. The report must use the exact schema fields from the approved design, include source paths as strings, convert `value_cm2 / 10_000`, and make rejected reports explicit instead of throwing after output setup. Preserve all input bytes by never opening input files for writing.

- [ ] **Step 4: Run stage and full focused tests to verify GREEN**

Run:

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/ground-stack-area-uv-cache uv run --active pytest tests/test_ground_stack_area.py -q
```

Expected: PASS; the report contains two unique instances totaling `400.0 cm²`, and the input mapping is byte-identical.

- [ ] **Step 5: Commit the stage**

```bash
git add code/modules/ground_stack_area_stage.py code/tests/test_ground_stack_area.py
git commit -m "feat: add ground-stack bbox measurement stage"
```

### Task 3: CLI, configuration, and user documentation

**Files:**
- Modify: `code/main.py`
- Modify: `code/config.yaml`
- Modify: `README.md`
- Modify: `code/README.md`
- Modify: `code/tests/test_ground_stack_area.py`

**Interfaces:**
- Consumes: `run_ground_stack_area()` from Task 2 and six CLI anchor fields.
- Produces: `uv run python main.py --mode ground-stack-area ...` and a nonzero exit only for rejected/failed requests.
- Preserves: all existing mode choices, matching backends, detection input formats, and upstream artifact immutability.

- [ ] **Step 1: Add a failing CLI integration test**

```python
import subprocess
import sys


def test_main_ground_stack_area_mode_runs_stage(tmp_path):
    dataset, save_root = make_measurement_fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable, "main.py", "--mode", "ground-stack-area",
            "--dataset", str(dataset), "--save_root", str(save_root),
            "--area-anchor-frame", "0", "--area-anchor-object", "0",
            "--area-anchor-width-cm", "20", "--area-anchor-height-cm", "10",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (save_root / "stack" / "ground_stack_area" / "measurement_report.json").is_file()
```

Factor the Task 2 fixture setup into `make_measurement_fixture()` in the test module before adding this test so both stage and CLI tests exercise the same artifact contract.

- [ ] **Step 2: Run the CLI test to verify RED**

Run:

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/ground-stack-area-uv-cache uv run --active pytest tests/test_ground_stack_area.py::test_main_ground_stack_area_mode_runs_stage -q
```

Expected: FAIL because argparse rejects `ground-stack-area` or its `--area-*` arguments.

- [ ] **Step 3: Add the smallest CLI/config/doc integration**

Add `ground-stack-area` to `--mode` choices, add the four required `--area-anchor-*` arguments with `None` defaults, validate that all four are supplied only in this mode, import and call the Task 2 stage, and exit nonzero only when the returned status is `rejected` or `success` is false. Add a separate `ground_stack_area` YAML section for descriptive defaults only; do not use it to silently provide an anchor.

Document this exact command in both READMEs:

```bash
cd code
uv run python main.py --mode ground-stack-area \
  --dataset ../imdata/my_stack --save_root ../Output \
  --area-anchor-frame 0 --area-anchor-object 3 \
  --area-anchor-width-cm 32.0 --area-anchor-height-cm 24.0
```

State that the result is a sum of one calibrated bbox per `global_id`, assumes coplanar front faces, does not estimate undetected/hidden objects, and is distinct from mask or footprint area.

- [ ] **Step 4: Run focused verification and static checks**

Run:

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/ground-stack-area-uv-cache uv run --active pytest tests/test_ground_stack_area.py -q
VIRTUAL_ENV=/home/xingyu/3D_Recognization/code/.venv UV_CACHE_DIR=/tmp/ground-stack-area-uv-cache uv run --active python main.py --help
```

Expected: all tests pass and help text lists `ground-stack-area` plus all four anchor arguments.

- [ ] **Step 5: Commit the integration**

```bash
git add code/main.py code/config.yaml README.md code/README.md code/tests/test_ground_stack_area.py
git commit -m "feat: expose calibrated ground-stack bbox area CLI"
```

## Plan self-review

- Spec coverage: Task 1 implements deterministic calibration, bbox validation, one-count-per-global-ID selection, and explicit rejections. Task 2 implements independent artifacts, report units/statuses, overlays, and input immutability. Task 3 exposes only the measurement stage through CLI/config and documents the metric/assumptions.
- Placeholder scan: the plan contains no unfinished markers, deferred V1 implementation, or unspecified error-handling steps. GPU/DA3 depth correction is explicitly excluded from V1 by the approved specification.
- Type consistency: Task 1 defines `PlanarCalibration` and selection records; Task 2 imports only the named calibration/selection API; Task 3 invokes the sole stage function with the same six positional values.
