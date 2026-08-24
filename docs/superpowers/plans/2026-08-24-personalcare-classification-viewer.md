# Personalcare Classification and Viewer Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run personalcare bbox classification concurrently with DA3 reconstruction/matching, propagate ordered SKU candidates into global viewer data, and add efficient SKU totals, filtering, point picking, and selected-object details.

**Architecture:** A local classifier subprocess reads immutable source images/detections and atomically publishes enriched detection JSON. The core pipeline joins that artifact before dedup, carries per-observation classification through `global_mapping.json`, and aggregates candidates in `objects.json`; the Three.js viewer validates that schema and reuses the current global-ID selection highlight for list, footprint, and point-cloud selection.

**Tech Stack:** Python 3.11, `uv`, PyTorch 2.7.1, torchvision 0.22.1, OpenCV, pytest, TypeScript, Vitest, Three.js r185, Vite.

**Spec:** `docs/superpowers/specs/2026-08-24-personalcare-classification-viewer-design.md`

## Global Constraints

- Run every Python command through `uv`; classifier dependencies stay in `modules/personalcare_classifier` and core dependencies stay at repository root.
- Do not start FastAPI/uvicorn or add HTTP/BSON compatibility paths.
- Do not add classifier hashes, signatures, encryption, feature-vector serialization, content fingerprints, or CPU/model fallbacks.
- Do not remove or redesign existing viewer provenance checks.
- Preserve `detections_results/` as immutable pipeline input and preserve bbox/object order.
- Invalid bboxes become explicit `classification.status = "unavailable"`; missing frames, index mismatches, model/CUDA failures, and incomplete publication fail the classification stage.
- All conflicting SKU candidates remain ordered by confidence sum; Total and SKU facet counts use only the primary candidate.
- Viewer never displays confidence values and never adds SKU-specific selection colors.
- Publish viewer `CURRENT` and manifest schema `2.0.0`; reject older bundles with a re-export error and do not add compatibility branches.
- Preserve the active root-layout migration and unrelated dirty files; stage only explicit task-owned paths.
- Workers do not commit concurrently. The coordinator performs exact-allowlist commits after each task passes review.

## Locked file ownership

| Owner | Files |
| --- | --- |
| Terra classifier | `modules/personalcare_classifier/**`, `tests/test_personalcare_classifier.py` |
| Terra backend | `main.py`, `src/deduplicate_detections.py`, `src/web_viewer_export.py`, `utils/classification_aggregation.py`, `utils/global_id_mapper.py`, `utils/global_object_index.py`, `tests/test_classification_aggregation.py`, `tests/test_main_pipeline.py`, `tests/test_web_viewer_export.py`, `modules/video_to_dedup/run.sh` |
| Luna viewer | `modules/viewer_web/src/**`, `modules/viewer_web/README.md` |
| Coordinator | `README.md`, `docs/3d_core.md`, integration review, validation evidence |

---

### Task 1: Pure classifier contracts and normalized bbox schema

**Files:**
- Create: `modules/personalcare_classifier/source/__init__.py`
- Create: `modules/personalcare_classifier/source/contracts.py`
- Create: `tests/test_personalcare_classifier.py`

**Interfaces:**
- Consumes: Raw labels formatted exactly as `sku_id^sku_name` and bbox prediction confidence in `[0, 1]`.
- Produces: `split_sku_label(label: str) -> tuple[str, str]`, `lookup_sku_metadata(sku_id: str, sku_name: str) -> dict[str, object]`, and `resolved_classification(project_id: int, label: str, confidence: float) -> dict[str, object]`.

- [ ] **Step 1: Write failing contract tests**

```python
from modules.personalcare_classifier.source.contracts import (
    lookup_sku_metadata,
    resolved_classification,
    split_sku_label,
)


def test_split_sku_label_preserves_name_suffix() -> None:
    assert split_sku_label("430085^产品^限定版") == ("430085", "产品^限定版")


def test_mapping_placeholder_is_explicit_and_empty() -> None:
    assert lookup_sku_metadata("430085", "产品A") == {
        "status": "master_data_pending",
        "manufacturer": None,
        "brand": None,
        "category": None,
        "object_kind": None,
    }


def test_resolved_classification_has_stable_schema() -> None:
    assert resolved_classification(51, "430085^产品A", 0.75) == {
        "schema_version": "1.0.0",
        "source": "personalcare",
        "project_id": 51,
        "status": "resolved",
        "sku_id": "430085",
        "sku_name": "产品A",
        "confidence": 0.75,
        "metadata": lookup_sku_metadata("430085", "产品A"),
    }
```

- [ ] **Step 2: Verify tests fail because contracts do not exist**

Run:

```bash
uv run --offline pytest -q tests/test_personalcare_classifier.py
```

Expected: collection fails with `ModuleNotFoundError` for `source.contracts`.

- [ ] **Step 3: Implement strict pure contracts**

```python
from __future__ import annotations

import math


def split_sku_label(label: str) -> tuple[str, str]:
    sku_id, separator, sku_name = label.partition("^")
    if separator == "" or sku_id.strip() == "" or sku_name.strip() == "":
        raise ValueError("personalcare label must be 'sku_id^sku_name'")
    return sku_id.strip(), sku_name.strip()


def lookup_sku_metadata(sku_id: str, sku_name: str) -> dict[str, object]:
    if not sku_id or not sku_name:
        raise ValueError("sku_id and sku_name must be non-empty")
    return {
        "status": "master_data_pending",
        "manufacturer": None,
        "brand": None,
        "category": None,
        "object_kind": None,
    }


def resolved_classification(project_id: int, label: str, confidence: float) -> dict[str, object]:
    if isinstance(project_id, bool) or not isinstance(project_id, int):
        raise ValueError("project_id must be an integer")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and within [0, 1]")
    sku_id, sku_name = split_sku_label(label)
    return {
        "schema_version": "1.0.0",
        "source": "personalcare",
        "project_id": project_id,
        "status": "resolved",
        "sku_id": sku_id,
        "sku_name": sku_name,
        "confidence": float(confidence),
        "metadata": lookup_sku_metadata(sku_id, sku_name),
    }
```

- [ ] **Step 4: Add malformed-label and confidence boundary tests, then pass them**

```python
@pytest.mark.parametrize("label", ["^产品A", "430085", "430085^"])
def test_split_sku_label_rejects_malformed_values(label: str) -> None:
    with pytest.raises(ValueError, match="sku_id\\^sku_name"):
        split_sku_label(label)


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.01, 1.01])
def test_resolved_classification_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        resolved_classification(51, "430085^产品A", confidence)
```

Run the same focused command and expect all tests to pass.

- [ ] **Step 5: Coordinator review and exact allowlist commit**

```bash
git add -- modules/personalcare_classifier/source/__init__.py modules/personalcare_classifier/source/contracts.py tests/test_personalcare_classifier.py
git commit -m "feat: define personalcare classification contract"
```

---

### Task 2: Efficient local classifier runner and atomic enriched detections

**Files:**
- Create: `modules/personalcare_classifier/pyproject.toml`
- Create: `modules/personalcare_classifier/source/classify_dataset.py`
- Modify: `modules/personalcare_classifier/source/processor.py`
- Modify: `modules/personalcare_classifier/README.md`
- Modify: `tests/test_personalcare_classifier.py`
- Generate: `modules/personalcare_classifier/uv.lock`

**Interfaces:**
- Consumes: `classify_dataset(dataset: Path, output_root: Path, device: str, predictor: BatchPredictor) -> ClassificationRunResult` and original `images/{frame}` plus `detections_results/{frame}.json`.
- Produces: `ClassificationRunResult(run_id: str, detection_dir: Path, result_path: Path, frame_count: int, object_count: int, unavailable_count: int)` and CLI JSON containing `success: true` plus those six fields.
- CLI: `uv run --project modules/personalcare_classifier python modules/personalcare_classifier/source/classify_dataset.py --dataset imdata/floor_display6 --output-root Output --device cuda:0`.

- [ ] **Step 1: Extend failing tests with an injected predictor**

```python
def make_dataset(root: Path, positions: list[list[int]]) -> Path:
    dataset = root / "dataset"
    images = dataset / "images"
    detections = dataset / "detections_results"
    images.mkdir(parents=True)
    detections.mkdir()
    image = np.full((24, 24, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(images / "0.jpg"), image)
    objects = [
        {"position": position, "classes": {"det": 0}, "confidences": {"det": 0.9}}
        for position in positions
    ]
    (detections / "0.json").write_text(json.dumps({
        "skus": [{"classes": {"det": ["sku"]}, "objects": objects}],
    }), encoding="utf-8")
    return dataset


def test_classify_dataset_preserves_order_and_publishes_enriched_json(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, positions=[[0, 0, 10, 10], [10, 0, 20, 10]])

    class FakePredictor:
        project_id = 51

        def predict(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
            assert len(crops) == 2
            return [("430085^产品A", 0.9), ("428987^产品B", 0.8)]

    result = classify_dataset(dataset, tmp_path / "Output", "cuda:0", FakePredictor())
    enriched = json.loads((result.detection_dir / "0.json").read_text())
    objects = enriched["skus"][0]["objects"]
    assert [item["position"] for item in objects] == [[0, 0, 10, 10], [10, 0, 20, 10]]
    assert [item["classification"]["sku_id"] for item in objects] == ["430085", "428987"]
    assert result.run_id.split("-")[0].isdigit()


def test_invalid_bbox_is_unavailable_and_features_are_not_published(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, positions=[[5, 5, 5, 12]])

    class PredictorThatMustNotRun:
        project_id = 51

        def predict(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
            raise AssertionError("invalid bbox must not reach predictor")

    result = classify_dataset(dataset, tmp_path / "Output", "cuda:0", PredictorThatMustNotRun())
    enriched = json.loads((result.detection_dir / "0.json").read_text())
    classification = enriched["skus"][0]["objects"][0]["classification"]
    assert classification == {"schema_version": "1.0.0", "source": "personalcare", "project_id": 51, "status": "unavailable", "reason": "invalid_bbox"}
    assert "features" not in json.dumps(enriched)


def test_missing_frame_does_not_replace_current(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, positions=[[0, 0, 10, 10]])
    (dataset / "images" / "0.jpg").rename(dataset / "images" / "1.jpg")
    current = tmp_path / "Output" / dataset.name / "personalcare_classification" / "CURRENT"
    current.parent.mkdir(parents=True)
    current.write_text('{"run_id":"old","complete":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="image/detection frame IDs differ"):
        classify_dataset(dataset, tmp_path / "Output", "cuda:0", PredictorThatMustNotRun())
    assert json.loads(current.read_text(encoding="utf-8"))["run_id"] == "old"
```

- [ ] **Step 2: Run the tests and verify failure at the missing runner**

```bash
uv run --offline pytest -q tests/test_personalcare_classifier.py
```

Expected: import fails for `source.classify_dataset`.

- [ ] **Step 3: Add the module `uv` project**

Use this dependency contract:

```toml
[project]
name = "personalcare-classifier"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "numpy==1.26.1",
  "opencv-python-headless>=4.8,<5",
  "torch==2.7.1",
  "torchvision==0.22.1",
]

[dependency-groups]
dev = ["pytest>=8.4,<9"]
```

Run `uv lock --project modules/personalcare_classifier`. Do not add FastAPI, BSON, uvicorn, Nuitka, or cryptography packages.

- [ ] **Step 4: Refactor processor into one lazy model instance**

Replace import-time CUDA work with a `PersonalcarePredictor` whose constructor accepts `device`, resolves `model/model.bin` relative to `__file__`, decodes the existing model bytes once, loads MobileNetV3 once, and exposes:

```python
class PersonalcarePredictor:
    project_id: int

    def predict(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Return one combined class label and confidence per OpenCV BGR uint8 crop."""
```

Use `torch.inference_mode()`, batch size 32, float32 ImageNet normalization, and no deep-feature collection. Reject an unavailable requested CUDA device; do not switch to CPU.

- [ ] **Step 5: Implement bounded-memory dataset classification**

Implement these behaviors in `classify_dataset.py`:

```python
@dataclass(frozen=True)
class ClassificationRunResult:
    run_id: str
    detection_dir: Path
    result_path: Path
    frame_count: int
    object_count: int
    unavailable_count: int

    def to_cli_payload(self) -> dict[str, object]:
        return {
            "success": True,
            "run_id": self.run_id,
            "detection_dir": str(self.detection_dir),
            "result_path": str(self.result_path),
            "frame_count": self.frame_count,
            "object_count": self.object_count,
            "unavailable_count": self.unavailable_count,
        }
```

- Enumerate numeric image and JSON stems and require exact equality.
- Read each image and JSON once.
- Collect valid crops in stable object order and send chunks of at most 32 to `predictor.predict`.
- Preserve the complete input payload; add raw `classes.cls`/`confidences.cls` plus normalized `classification` per object.
- For invalid/reversed/empty/out-of-image boxes, write `status = "unavailable"` with `reason = "invalid_bbox"` and do not call the predictor for that object.
- Write a unique `.run_id.tmp` directory under `runs/`, validate counts, rename it to `runs/run_id`, then atomically replace `CURRENT` with `{"run_id": run_id, "complete": true}`.
- Use `run_id = f"{time.time_ns()}-{os.getpid()}"`; do not hash or encrypt content.
- Print exactly one result JSON object to stdout; send diagnostics to stderr.

- [ ] **Step 6: Run focused tests and module help**

```bash
uv run --offline pytest -q tests/test_personalcare_classifier.py
uv run --project modules/personalcare_classifier python modules/personalcare_classifier/source/classify_dataset.py --help
```

Expected: tests pass; help lists `--dataset`, `--output-root`, and `--device`.

- [ ] **Step 7: Coordinator review and exact allowlist commit**

```bash
git add -- modules/personalcare_classifier/pyproject.toml modules/personalcare_classifier/uv.lock modules/personalcare_classifier/source/processor.py modules/personalcare_classifier/source/classify_dataset.py modules/personalcare_classifier/README.md tests/test_personalcare_classifier.py
git commit -m "feat: add local personalcare classification runner"
```

---

### Task 3: Global candidate aggregation and strict viewer object schema

**Files:**
- Create: `utils/classification_aggregation.py`
- Modify: `src/deduplicate_detections.py:305-753`
- Modify: `utils/global_id_mapper.py:14-101`
- Modify: `utils/global_object_index.py:15-27`
- Modify: `src/web_viewer_export.py:225-242`
- Create: `tests/test_classification_aggregation.py`
- Modify: `tests/test_web_viewer_export.py`

**Interfaces:**
- Consumes: Every global-mapping instance has a `classification` object with `resolved` or `unavailable` status.
- Produces: `aggregate_classifications(classifications: Sequence[Mapping[str, Any]]) -> dict[str, Any]`; `InstanceInfo.classification`; and `ObjectIndexEntry.classification` matching the approved spec.

- [ ] **Step 1: Write aggregation tests before production code**

```python
def resolved(sku_id: str, sku_name: str, confidence: float) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "source": "personalcare",
        "project_id": 51,
        "status": "resolved",
        "sku_id": sku_id,
        "sku_name": sku_name,
        "confidence": confidence,
        "metadata": {
            "status": "master_data_pending",
            "manufacturer": None,
            "brand": None,
            "category": None,
            "object_kind": None,
        },
    }


def test_conflicts_keep_all_candidates_but_primary_is_highest_sum() -> None:
    result = aggregate_classifications([
        resolved("A", "产品A", 0.60),
        resolved("B", "产品B", 0.95),
        resolved("A", "产品A", 0.50),
    ])
    assert result["status"] == "conflict"
    assert result["primary_sku_id"] == "A"
    assert [item["sku_id"] for item in result["candidates"]] == ["A", "B"]
    assert result["candidates"][0]["confidence_sum"] == pytest.approx(1.10)


def test_aggregation_is_permutation_stable() -> None:
    inputs = [resolved("2", "乙", 0.8), resolved("1", "甲", 0.8)]
    assert aggregate_classifications(inputs) == aggregate_classifications(list(reversed(inputs)))
    assert aggregate_classifications(inputs)["primary_sku_id"] == "1"


def test_unavailable_observations_produce_no_primary() -> None:
    unavailable = {"schema_version": "1.0.0", "source": "personalcare", "project_id": 51, "status": "unavailable", "reason": "invalid_bbox"}
    assert aggregate_classifications([unavailable]) == {
        "status": "unavailable",
        "primary_sku_id": None,
        "candidates": [],
        "metadata": {
            "status": "master_data_pending",
            "manufacturer": None,
            "brand": None,
            "category": None,
            "object_kind": None,
        },
    }


def test_invalid_confidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="classification confidence"):
        aggregate_classifications([resolved("A", "产品A", float("nan"))])


def test_same_id_with_different_names_remains_distinct_and_deterministic() -> None:
    result = aggregate_classifications([resolved("A", "产品乙", 0.8), resolved("A", "产品甲", 0.8)])
    assert [(item["sku_id"], item["sku_name"]) for item in result["candidates"]] == [("A", "产品乙"), ("A", "产品甲")]
```

- [ ] **Step 2: Verify tests fail because aggregation is missing**

```bash
uv run --offline pytest -q tests/test_classification_aggregation.py
```

Expected: `ModuleNotFoundError` for `utils.classification_aggregation`.

- [ ] **Step 3: Implement deterministic aggregation**

```python
def aggregate_classifications(classifications: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[float]] = {}
    for item in classifications:
        if item.get("status") == "unavailable":
            continue
        validated = validate_resolved_classification(item)
        groups.setdefault((validated.sku_id, validated.sku_name), []).append(validated.confidence)
    candidates = [candidate_from_group(key, values) for key, values in groups.items()]
    candidates.sort(key=lambda item: (
        -item["confidence_sum"],
        -item["support_count"],
        -item["max_confidence"],
        item["sku_id"],
        item["sku_name"],
    ))
    return build_aggregate(candidates)
```

Use `math.fsum` for confidence sums. Return `primary_sku_id = None` and an empty candidate list when unavailable. Copy the primary candidate's metadata; V1 metadata must be `master_data_pending` with four null fields.

- [ ] **Step 4: Carry classification through dedup and mapping**

- Add an explicit `detections_dir: Path | None = None` parameter to `resolve_dataset_paths`, `deduplicate_sequence`, and `add_global_id_to_jsons`.
- Read enriched objects from that directory while retaining the original dataset path for images and matching output.
- In `build_global_mapping`, deep-copy each object's `classification` into the mapping instance and reject missing/malformed classification.
- Update `InstanceInfo` to accept/store/return `classification`.
- Add aggregated `classification` to every `build_global_object_index` entry.

Add this regression assertion to `tests/test_classification_aggregation.py` using a two-observation mapper fixture:

```python
index = build_global_object_index(mapper_with_classifications([
    resolved("A", "产品A", 0.6),
    resolved("B", "产品B", 0.9),
]))
assert [inst["classification"]["sku_id"] for inst in index["1"]["instances"]] == ["A", "B"]
assert [item["sku_id"] for item in index["1"]["classification"]["candidates"]] == ["B", "A"]
```

- [ ] **Step 5: Extend exporter validation and capabilities**

Update `_validate_object_index_for_export` to require exact classification keys and validate finite values/ranges. Set viewer `CURRENT` and manifest `schema_version` to `2.0.0`, set `capabilities.point_picking` to `true`, and add no parser for schema `1.0.0`. Use this mutation table in `tests/test_web_viewer_export.py`; every case must raise `WebViewerExportError`:

```python
@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        ("missing_primary", lambda value: value.pop("primary_sku_id")),
        ("unsorted_candidates", lambda value: value["candidates"].reverse()),
        ("duplicate_candidate", lambda value: value["candidates"].append(copy.deepcopy(value["candidates"][0]))),
        ("invalid_metadata", lambda value: value["metadata"].update(status="resolved")),
    ],
)
def test_export_rejects_invalid_classification(case: str, mutate) -> None:
    objects = valid_classified_object_index()
    mutate(objects["1"]["classification"])
    with pytest.raises(WebViewerExportError, match="classification"):
        _validate_object_index_for_export(objects)
```

- [ ] **Step 6: Run focused backend tests**

```bash
uv run --offline pytest -q tests/test_classification_aggregation.py tests/test_web_viewer_export.py
```

Expected: all tests pass without loading GPU models.

- [ ] **Step 7: Coordinator review and exact allowlist commit**

```bash
git add -- utils/classification_aggregation.py utils/global_id_mapper.py utils/global_object_index.py src/deduplicate_detections.py src/web_viewer_export.py tests/test_classification_aggregation.py tests/test_web_viewer_export.py
git commit -m "feat: propagate SKU classification into viewer objects"
```

---

### Task 4: Pipeline concurrency and classification join

**Files:**
- Modify: `main.py:892-1071`
- Modify: `tests/test_main_pipeline.py`
- Modify: `modules/video_to_dedup/run.sh`

**Interfaces:**
- Consumes: Task 2 CLI JSON with `run_id`, `detection_dir`, `result_path`, `frame_count`, `object_count`, and `unavailable_count`; Task 3 `deduplicate_sequence(..., detections_dir=Path)`.
- Produces: `SKUDetectionMain.run_personalcare_classification(dataset_path: str) -> dict[str, object]`; pipeline summary key `classification`; CLI option `--classifier-device` defaulting to `cuda:0`.

- [ ] **Step 1: Write ordering/concurrency tests**

```python
def test_pipeline_joins_classification_only_before_dedup(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "detections_results").mkdir()
    app = main.SKUDetectionMain()
    app.save_root = tmp_path / "Output"
    app.match_backend = "da3"
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def classify(_dataset: str) -> dict[str, object]:
        entered.set()
        assert release.wait(timeout=2)
        calls.append("classification_done")
        return {"success": True, "detection_dir": str(tmp_path / "classified")}

    monkeypatch.setattr(app, "run_personalcare_classification", classify)
    monkeypatch.setattr(app, "run_reconstruction", lambda *a, **k: entered.wait(2) and {"success": True})
    monkeypatch.setattr(app, "run_sku_matching", lambda *a, **k: release.set() or calls.append("matching_done") or {"success": True})
    monkeypatch.setattr(app, "run_dedup_sequence", lambda *a, **k: calls.append(f"dedup:{k['detection_dir']}") or {"success": True})

    summary = app.run_complete_pipeline(str(dataset), algorithm="3d")
    assert summary["classification"] is True
    assert calls.index("classification_done") < calls.index(f"dedup:{tmp_path / 'classified'}")
    assert calls.index("matching_done") < calls.index(f"dedup:{tmp_path / 'classified'}")


def test_classification_failure_stops_before_dedup(monkeypatch, tmp_path: Path) -> None:
    app, dataset = pipeline_fixture(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(app, "run_personalcare_classification", lambda _path: {"success": False, "error": "model failed"})
    monkeypatch.setattr(app, "run_sku_matching", lambda *a, **k: calls.append("matching") or {"success": True})
    monkeypatch.setattr(app, "run_dedup_sequence", lambda *a, **k: calls.append("dedup") or {"success": True})
    summary = app.run_complete_pipeline(str(dataset), algorithm="3d")
    assert summary["reconstruction"] is True
    assert summary["matching"] is True
    assert summary["classification"] is False
    assert calls == ["matching"]
```

- [ ] **Step 2: Run focused pipeline tests and verify failure**

```bash
uv run --offline pytest -q tests/test_main_pipeline.py
```

Expected: failure because `run_personalcare_classification` and the dedup detection-directory argument do not exist.

- [ ] **Step 3: Implement the subprocess boundary**

Build this non-shell command list and call `subprocess.run(..., capture_output=True, text=True, check=False)`:

```python
[
    "uv", "run", "--project", str(CLASSIFIER_ROOT),
    "python", str(CLASSIFIER_SCRIPT),
    "--dataset", str(dataset),
    "--output-root", str(self.save_root or DEFAULT_SAVE_ROOT),
    "--device", self.classifier_device,
]
```

Parse stdout as one JSON object. Require return code zero, `success is True`, and an existing `detection_dir`; otherwise return a failed stage result with stderr.

- [ ] **Step 4: Launch classification concurrently and join before dedup**

Use one `ThreadPoolExecutor(max_workers=1)` around the classifier subprocess. Start the future immediately after dataset validation. Run reconstruction, visualization, matching, and analysis on the current thread. Resolve the future immediately before dedup. On classification failure, set `summary["classification"] = False`, set dedup and later publication summaries false, and return without calling dedup.

Pass the exact classified detection path to `run_dedup_sequence` and onward to `deduplicate_sequence`.

- [ ] **Step 5: Wire CLI and maintained video output**

- Add `--classifier-device` with default `cuda:0` and store it on `SKUDetectionMain`.
- Keep `CUDA_VISIBLE_DEVICES` ownership in `modules/video_to_dedup/run.sh`; report the classification output directory in `extract_results`.
- Do not add a disable switch, CPU fallback, hash command, or second shell implementation.

- [ ] **Step 6: Run focused tests and shell syntax**

```bash
uv run --offline pytest -q tests/test_main_pipeline.py tests/test_classification_aggregation.py
bash -n modules/video_to_dedup/run.sh modules/video_to_dedup/quickstart.sh
```

Expected: tests pass and both scripts parse successfully.

- [ ] **Step 7: Coordinator review and exact allowlist commit**

```bash
git add -- main.py tests/test_main_pipeline.py modules/video_to_dedup/run.sh
git commit -m "feat: overlap classification with DA3 matching"
```

---

### Task 5: Viewer classification contracts, facets, and presentation

**Files:**
- Modify: `modules/viewer_web/src/contracts.ts:14-341`
- Modify: `modules/viewer_web/src/contracts.test.ts`
- Modify: `modules/viewer_web/src/bundle-loader.test.ts`
- Create: `modules/viewer_web/src/sku-filters.ts`
- Create: `modules/viewer_web/src/sku-filters.test.ts`
- Modify: `modules/viewer_web/src/presentation.ts`
- Modify: `modules/viewer_web/src/presentation.test.ts`

**Interfaces:**
- Consumes: Python `ObjectIndexEntry.classification` from Task 3.
- Produces: `ProductMetadata`, `ObjectClassificationObservation`, `ClassificationAggregate`, `ClassificationCandidate`, `SkuFacet`, `buildSkuFacets(objects)`, and `filterGlobalIdsBySku(objects, skuId)`.

- [ ] **Step 1: Add strict contract tests**

```typescript
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

const objectEntry = (classification: unknown) => ({
  images: [0], objects: [0], active_count: 1, removed_count: 0, total_count: 1,
  instances: [{ image_id: 0, object_id: 0, bbox: [0, 0, 10, 10], removed: false, point_index_range: [0, 12], thumbnail: "thumbs/1_0.jpg" }],
  classification,
});


it("accepts conflict candidates in pre-ranked order", () => {
  const objects = validateObjectIndex({
    "1": objectEntry({
      status: "conflict",
      primary_sku_id: "A",
      candidates: [candidate("A", "产品A", 1.1, 2, 0.6), candidate("B", "产品B", 0.9, 1, 0.9)],
      metadata: pendingMetadata(),
    }),
  }, 12);
  expect(objects["1"].classification.candidates.map((item) => item.sku_id)).toEqual(["A", "B"]);
});

it.each([
  ["non-finite score", { candidates: [candidate("A", "产品A", Number.NaN, 1, 0.8)] }],
  ["primary mismatch", { primary_sku_id: "B" }],
  ["duplicate candidate", { candidates: [candidate("A", "产品A", 1, 1, 1), candidate("A", "产品A", 0.5, 1, 0.5)] }],
  ["invalid metadata", { metadata: { ...pendingMetadata(), status: "resolved" } }],
])("rejects %s", (_label, mutation) => {
  const valid = { status: "resolved", primary_sku_id: "A", candidates: [candidate("A", "产品A", 1, 1, 1)], metadata: pendingMetadata() };
  expect(() => validateObjectIndex({ "1": objectEntry({ ...valid, ...mutation }) }, 12)).toThrow();
});
```

```typescript
it("requires schema 2 and point picking", () => {
  const manifest = validManifest();
  expect(() => validateManifest({ ...manifest, schema_version: "1.0.0" })).toThrow(/schema_version/);
  expect(() => validateManifest({ ...manifest, capabilities: { ...manifest.capabilities, point_picking: false } })).toThrow(/capabilities/);
});

it("rejects candidates outside deterministic order", () => {
  const classification = {
    status: "conflict",
    primary_sku_id: "B",
    candidates: [candidate("B", "产品B", 0.5, 1, 0.5), candidate("A", "产品A", 1.0, 1, 1.0)],
    metadata: pendingMetadata(),
  };
  expect(() => validateObjectIndex({ "1": objectEntry(classification) }, 12)).toThrow(/order/);
});
```

- [ ] **Step 2: Verify the contract test fails**

```bash
npm --prefix modules/viewer_web test -- --run src/contracts.test.ts
```

Expected: exact-key validation rejects the new `classification` key.

- [ ] **Step 3: Add TypeScript types and validators**

```typescript
export interface ProductMetadata {
  readonly status: "master_data_pending";
  readonly manufacturer: null;
  readonly brand: null;
  readonly category: null;
  readonly object_kind: null;
}

export type ObjectClassificationObservation =
  | Readonly<{
      schema_version: "1.0.0";
      source: "personalcare";
      project_id: number;
      status: "resolved";
      sku_id: string;
      sku_name: string;
      confidence: number;
      metadata: ProductMetadata;
    }>
  | Readonly<{
      schema_version: "1.0.0";
      source: "personalcare";
      project_id: number;
      status: "unavailable";
      reason: "invalid_bbox";
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
```

Add `classification: ObjectClassificationObservation` to `ObjectInstance` and `classification: ClassificationAggregate` to `ObjectIndexEntry`. Validate exact keys, finite confidence values, `[0, 1]` max confidence, positive support counts, unique `(sku_id, sku_name)`, deterministic order, and status/primary/candidate cardinality consistency. Update `validateCurrent` and `validateManifest` to require `schema_version: "2.0.0"`; do not accept `1.0.0`. Update every fixture in `contracts.test.ts` and `bundle-loader.test.ts` to schema `2.0.0`.

- [ ] **Step 4: Implement pure facet/filter functions test-first**

```typescript
export function buildSkuFacets(objects: ObjectIndex): readonly SkuFacet[] {
  const counts = new Map<string, { skuName: string; count: number }>();
  for (const entry of Object.values(objects)) {
    const primary = entry.classification.candidates[0];
    if (primary === undefined) continue;
    const current = counts.get(primary.sku_id);
    counts.set(primary.sku_id, { skuName: primary.sku_name, count: (current?.count ?? 0) + 1 });
  }
  return [...counts.entries()]
    .map(([skuId, value]) => ({ skuId, skuName: value.skuName, count: value.count }))
    .sort((left, right) => right.count - left.count || left.skuId.localeCompare(right.skuId));
}
```

```typescript
const filterEntry = (candidates: readonly ClassificationCandidate[]): ObjectIndexEntry => ({
  images: [0],
  objects: [0],
  active_count: 1,
  removed_count: 0,
  total_count: 1,
  instances: [{ image_id: 0, object_id: 0, bbox: [0, 0, 1, 1], removed: false, point_index_range: [0, 1], thumbnail: "thumbs/1_0.jpg", classification: unavailableObservation() }],
  classification: {
    status: candidates.length > 1 ? "conflict" : "resolved",
    primary_sku_id: candidates[0]?.sku_id ?? null,
    candidates,
    metadata: pendingMetadata(),
  },
});

it("counts and filters only by the primary candidate", () => {
  const objects = {
    "1": filterEntry([candidate("A", "产品A", 1.1, 2, 0.6), candidate("B", "产品B", 0.9, 1, 0.9)]),
    "2": filterEntry([candidate("B", "产品B", 0.8, 1, 0.8)]),
  };
  expect(buildSkuFacets(objects)).toEqual([
    { skuId: "A", skuName: "产品A", count: 1 },
    { skuId: "B", skuName: "产品B", count: 1 },
  ]);
  expect(Object.keys(objects)).toHaveLength(2);
  expect(filterGlobalIdsBySku(objects, "A")).toEqual(["1"]);
  expect(filterGlobalIdsBySku(objects, "B")).toEqual(["2"]);
});
```

- [ ] **Step 5: Extend evidence presentation without confidence**

Add ordered `{skuId, skuName}` candidates to `EvidenceView` and add this assertion:

```typescript
const view = buildEvidenceView(conflictBundle(), "1");
expect(view?.skuCandidates).toEqual([
  { skuId: "A", skuName: "产品A" },
  { skuId: "B", skuName: "产品B" },
]);
expect(JSON.stringify(view?.skuCandidates)).not.toMatch(/confidence|0\\.[0-9]+|%/);
```

- [ ] **Step 6: Run focused viewer contract tests**

```bash
npm --prefix modules/viewer_web test -- --run src/contracts.test.ts src/sku-filters.test.ts src/presentation.test.ts
```

Expected: all tests pass.

- [ ] **Step 7: Coordinator review and exact allowlist commit**

```bash
git add -- modules/viewer_web/src/contracts.ts modules/viewer_web/src/contracts.test.ts modules/viewer_web/src/sku-filters.ts modules/viewer_web/src/sku-filters.test.ts modules/viewer_web/src/presentation.ts modules/viewer_web/src/presentation.test.ts
git commit -m "feat: add viewer SKU contracts and facets"
```

---

### Task 6: Efficient point picking, visibility filtering, and viewer UI

**Files:**
- Create: `modules/viewer_web/src/point-picking.ts`
- Create: `modules/viewer_web/src/point-picking.test.ts`
- Modify: `modules/viewer_web/src/scene.ts`
- Modify: `modules/viewer_web/src/scene.test.ts`
- Modify: `modules/viewer_web/src/main.ts`
- Modify: `modules/viewer_web/src/main.test.ts`
- Modify: `modules/viewer_web/src/style.css`
- Modify: `modules/viewer_web/README.md`

**Interfaces:**
- Consumes: Task 5 contracts/facets and existing instance `point_index_range` values.
- Produces: `buildPointRangeLookup(objects: ObjectIndex) -> readonly PointOwnerRange[]`; `globalIdForPointIndex(lookup: readonly PointOwnerRange[], pointIndex: number) -> string | null`; `visiblePointRanges(objects: ObjectIndex, ids: ReadonlySet<string>) -> readonly PointOwnerRange[]`; `ViewerSceneController.setVisibleGlobalIds(ids: ReadonlySet<string>)`; one global-ID pick handler shared by points and footprints.

- [ ] **Step 1: Write pure point-index lookup tests**

```typescript
function objectIndexWithRanges(rangesById: Record<string, readonly (readonly [number, number])[]>): ObjectIndex {
  return Object.fromEntries(Object.entries(rangesById).map(([globalId, ranges]) => [globalId, {
    images: ranges.map((_range, index) => index),
    objects: ranges.map((_range, index) => index),
    active_count: ranges.length,
    removed_count: 0,
    total_count: ranges.length,
    instances: ranges.map((range, index) => ({
      image_id: index,
      object_id: index,
      bbox: [0, 0, 1, 1] as const,
      removed: false,
      point_index_range: range,
      thumbnail: `thumbs/${globalId}_${index}.jpg`,
      classification: unavailableObservation(),
    })),
    classification: unavailableAggregate(),
  }])) as ObjectIndex;
}

it("maps point indices to global IDs and ignores empty ranges", () => {
  const lookup = buildPointRangeLookup({
    "1": objectEntryWithRanges([[0, 3], [6, 8]]),
    "2": objectEntryWithRanges([[3, 6], [8, 8]]),
  });
  expect(globalIdForPointIndex(lookup, 0)).toBe("1");
  expect(globalIdForPointIndex(lookup, 4)).toBe("2");
  expect(globalIdForPointIndex(lookup, 7)).toBe("1");
  expect(globalIdForPointIndex(lookup, 8)).toBeNull();
});

it("returns only ranges owned by visible IDs", () => {
  const objects = objectIndexWithRanges({ "1": [[0, 3], [6, 8]], "2": [[3, 6]] });
  expect(visiblePointRanges(objects, new Set(["2"]))).toEqual([
    { start: 3, end: 6, globalId: "2" },
  ]);
});
```

Implement a sorted range array and binary search; reject overlaps in the existing contract validator.

- [ ] **Step 2: Add failing selection/visibility tests**

```typescript
it("hidden IDs are excluded from pick resolution", () => {
  const lookup = buildPointRangeLookup(objectIndexWithRanges({ "1": [[0, 3]], "2": [[3, 6]] }));
  const visible = new Set(["2"]);
  const hit = globalIdForPointIndex(lookup, 1);
  expect(hit !== null && visible.has(hit) ? hit : null).toBeNull();
});

it("keeps the existing selected magenta bytes", () => {
  const colors = Uint8Array.from([10, 20, 30, 40, 50, 60]);
  applySelectionColors(colors, Uint8Array.from(colors), [], [[0, 1]]);
  expect([...colors.slice(0, 3)]).toEqual([255, 0, 255]);
});
```

Add a pure resolver used by the scene's pointer-release handler:

```typescript
export function resolvePickGlobalId(
  footprintGlobalId: string | null,
  pointIndex: number | null,
  lookup: readonly PointOwnerRange[],
  visibleIds: ReadonlySet<string>,
): string | null {
  const candidate = footprintGlobalId ?? (pointIndex === null ? null : globalIdForPointIndex(lookup, pointIndex));
  return candidate !== null && visibleIds.has(candidate) ? candidate : null;
}

it("uses footprint first, then points, and rejects hidden IDs", () => {
  const lookup = buildPointRangeLookup(objectIndexWithRanges({ "1": [[0, 3]], "2": [[3, 6]] }));
  expect(resolvePickGlobalId("2", 1, lookup, new Set(["1", "2"]))).toBe("2");
  expect(resolvePickGlobalId(null, 1, lookup, new Set(["1", "2"]))).toBe("1");
  expect(resolvePickGlobalId(null, 1, lookup, new Set(["2"]))).toBeNull();
});
```

- [ ] **Step 3: Add one runtime byte visibility attribute**

- Initialize `aVisible` as a `Uint8BufferAttribute` of length `point_count`, filled with one and marked `DynamicDrawUsage`.
- Add `attribute float aVisible` and a varying to the current point shader; discard hidden fragments before color output.
- `setVisibleGlobalIds` updates byte ranges from `bundle.objects[gid].instances[*].point_index_range`, queues update ranges, and toggles footprint mesh visibility.
- Do not create a second Points object or copy positions/colors/normals.

- [ ] **Step 4: Add point-cloud click selection**

On the existing click-release path, raycast visible footprint meshes first, then the existing `Points`. Convert `intersection.index` through the binary-search lookup, reject hidden IDs, and call the shared pick handler. Set `raycaster.params.Points.threshold` from the current point size and scene span; update it when point size changes.

- [ ] **Step 5: Build the approved UI**

- Add Total and Visible counts at the top of the left panel.
- Render SKU facets as buttons labelled `sku_name (count)`; provide `显示所有`.
- Add disabled rows `厂商/品牌/品类：主数据待接入` and `POSM/价签/空缺位：检测能力待接入`.
- Filter the global-ID list and call `controller.setVisibleGlobalIds` with the same IDs.
- If filtering hides the selected ID, clear selection through the existing `select(null, false)` path.
- In the right drawer render every candidate as `sku_id · sku_name`, in supplied order, with no confidence value.
- Keep existing selection classes/colors and navigation semantics.

- [ ] **Step 6: Run frontend tests and production build**

```bash
npm --prefix modules/viewer_web test -- --run
npm --prefix modules/viewer_web run build
```

Expected: all Vitest tests and the Vite production build pass.

- [ ] **Step 7: Coordinator review and exact allowlist commit**

```bash
git add -- modules/viewer_web/src/point-picking.ts modules/viewer_web/src/point-picking.test.ts modules/viewer_web/src/scene.ts modules/viewer_web/src/scene.test.ts modules/viewer_web/src/main.ts modules/viewer_web/src/main.test.ts modules/viewer_web/src/style.css modules/viewer_web/README.md
git commit -m "feat: add SKU filtering and point picking"
```

---

### Task 7: Documentation, integration verification, and real-model evidence

**Files:**
- Modify: `README.md`
- Modify: `docs/3d_core.md`
- Verify: all files owned by Tasks 1-6

**Interfaces:**
- Consumes: completed classifier, backend, pipeline, and viewer tasks.
- Produces: current usage documentation and evidence-bounded completion report.

- [ ] **Step 1: Update operational documentation**

Document the automatic classifier stage, `--classifier-device`, classification output directory, conflict-count semantics, disabled facets, no-confidence UI, no-new-hash policy, and exact commands for classifier-only, pipeline, viewer export, and frontend launch.

- [ ] **Step 2: Run the exact focused Python suite**

```bash
uv run --offline pytest -q \
  tests/test_personalcare_classifier.py \
  tests/test_classification_aggregation.py \
  tests/test_main_pipeline.py \
  tests/test_web_viewer_export.py
```

- [ ] **Step 3: Run the full non-GPU regression suite**

```bash
uv run --offline pytest -q
npm --prefix modules/viewer_web test -- --run
npm --prefix modules/viewer_web run build
bash -n modules/video_to_dedup/*.sh scripts/3d/evaluation/*.sh scripts/3d/ops/*.sh scripts/3d/pipeline/*.sh scripts/3d/tuning/*.sh
```

- [ ] **Step 4: Run explicit classifier model smoke**

First inspect available resources. Then run one approved small frame without substituting the model or device:

```bash
uv run --project modules/personalcare_classifier python \
  modules/personalcare_classifier/source/classify_dataset.py \
  --dataset imdata/floor_display6 \
  --output-root /tmp/personalcare-classifier-smoke \
  --device cuda:0
```

Record frame count, bbox count, unavailable count, wall time, and peak classifier VRAM. Treat model or CUDA failure as a failed smoke, not a CPU result.

- [ ] **Step 5: Run bounded overlap smoke only when VRAM permits**

Run the normal pipeline on the same small approved dataset with the chosen GPU and capture stage timings/peak VRAM. Verify classification starts before matching finishes and dedup starts after both complete. If GPU capacity is insufficient, report the overlap hard failure; do not serialize silently or claim concurrency success.

- [ ] **Step 6: Review worktree and temporary artifacts**

Use `git diff --check`, inspect `git status --short`, and ensure no `/tmp` outputs, runtime bundles, model copies, caches, or build outputs are staged. Temporary smoke output under `/tmp/personalcare-classifier-smoke` may be removed after evidence is recorded.

- [ ] **Step 7: Coordinator exact allowlist commit**

```bash
git add -- README.md docs/3d_core.md
git commit -m "docs: document classified SKU viewer workflow"
```

## Final verification checklist

- [ ] Classifier model loads once and classifies all valid bbox crops in batches.
- [ ] Original detections remain untouched; enriched detections publish atomically without new hashes/encryption.
- [ ] Classification overlaps reconstruction/matching and joins before dedup.
- [ ] `global_mapping.json` includes per-observation classification.
- [ ] `objects.json` includes deterministic all-candidate aggregation.
- [ ] Conflict candidates all display; confidence values do not.
- [ ] Total/SKU counts use one primary candidate per global ID.
- [ ] Point and footprint selection reuse the existing highlight.
- [ ] SKU filters update both list and scene visibility without geometry duplication.
- [ ] Unsupported facets are disabled with explicit pending messages.
- [ ] Focused/full Python tests, frontend tests, build, and shell syntax pass.
- [ ] Real-model and overlap claims are limited to actually observed GPU evidence.
