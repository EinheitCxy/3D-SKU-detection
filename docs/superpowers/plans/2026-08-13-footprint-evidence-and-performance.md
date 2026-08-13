# Ground-stack Footprint Evidence and Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the support-plane OBB polygon-union area formula unchanged while making its plane fitting faster, reusing verified SAM3 masks, and adding non-authoritative multi-view evidence.

**Architecture:** Track A changes only the deterministic NumPy RANSAC workspace and adds auditable trial diagnostics. Track B introduces a per-frame immutable SAM3 mask bundle and immutable artifact generations, then makes the stage consume those bundles without weakening formal validation. Track C loads camera tensors separately after the formal result is frozen and produces shadow-only reprojection and stability evidence.

**Tech Stack:** Python 3, NumPy, OpenCV, Shapely, Matplotlib, PIL, pytest, `fcntl.flock`, SHA-256, existing DA3/SAM3 CUDA runtime.

**Spec:** `docs/superpowers/specs/2026-08-13-footprint-evidence-and-performance-design.md`

## Global Constraints

- The metric remains `area(union(OBB(project_to_support_plane(points(global_id)))))` in `m2`.
- Keep the 10 mm refined-plane residual, 10 percent background coverage, all-global-ID, and support-plane ambiguity gates unchanged.
- Never use a raw bbox when SAM3 mask computation or cache validation fails; publish `rejected` with `value_m2: null` and no polygon features.
- `_adaptive_ransac_plane` preserves its seed, triplet draw order, `norm == 0` handling, threshold comparison, strict `count > best_count`, 0.999 probability, and 10,000 trial cap.
- Only one candidate's RANSAC trial loop may early-return at `count == M`; `select_support_plane` still evaluates up to five candidates and ambiguity.
- A SAM3 cache key is canonical UTF-8 JSON SHA-256 and includes image, every ordered detection, checkpoint digest, code/runtime fingerprints, inference contract, and boolean source-mask output contract.
- Formal result fields and formal geometries freeze before evidence starts. Evidence errors are additive `unavailable_*` or `failed_*` data, never formal rejections.
- Use `/home/xingyu/3D_Recognization/code/.venv` through `uv run --active --no-project`; do not create an environment or fall back to CPU.
- Stage only explicit owned files. Do not stage `video_dedup_runs/`, `code/y/`, `sam3/checkpoints/`, DA3 cache payloads, detection JSON, or generated images.

## File Structure

- Modify `code/utils/ground_stack_footprint.py`: deterministic RANSAC workspace reuse and per-candidate trial diagnostics only.
- Modify `code/tests/test_ground_stack_footprint.py`: reference-loop parity, early-exit, threshold-edge, tie, and ambiguity tests.
- Create `code/utils/sam3_mask_cache.py`: canonical cache request/result types, immutable bundle validation, locking, quarantine, and atomic publication.
- Create `code/tests/test_sam3_mask_cache.py`: cache hash, corruption, concurrency, and payload-contract tests using a fake mask producer.
- Modify `code/utils/sam3_utils.py`: checkpoint-digest-stable model loading and process-cache keying.
- Modify `code/modules/da3_footprint_stage.py`: cache integration, immutable formal artifact generations, formal snapshot, and evidence attachment boundary.
- Create `code/utils/footprint_evidence.py`: optional camera loader, camera contract validation, reprojection calculations, and leave-one-observation-out metrics.
- Modify `code/tests/test_da3_footprint_stage.py`: cache integration, generation publication, and shadow-isolation fixtures.
- Create `code/tests/test_footprint_evidence.py`: pure NumPy camera/reprojection/LOO tests.
- Modify `README.md` and `code/README.md`: cache location/invalidation, `CURRENT` resolution, evidence interpretation, and no-calibration claim.

---

### Task 1: Record strict RANSAC trial diagnostics and prove reference parity

**Files:**
- Modify: `code/utils/ground_stack_footprint.py:48-183,403-433`
- Modify: `code/tests/test_ground_stack_footprint.py`
- Modify: `README.md:112-120`
- Modify: `code/README.md:153-155`

**Interfaces:**
- Consumes: `_adaptive_ransac_plane(points, threshold_m, seed)` and its current candidate loop.
- Produces: `RansacOutcome(point: np.ndarray, normal: np.ndarray, trial_count: int, early_exit: bool)` and candidate diagnostics under `diagnostics["ransac"]`.

- [ ] **Step 1: Write the failing reference-parity and early-exit tests**

```python
def _reference_adaptive_ransac(points, threshold_m, seed):
    generator = np.random.default_rng(seed)
    best_count, best_candidate, trial, target_trials = 0, None, 0, 10_000
    while trial < target_trials:
        indices = _sample_ransac_triplet(generator, population_size=len(points))
        first, second, third = points[indices]
        normal = np.cross(second - first, third - first)
        norm = np.linalg.norm(normal)
        if norm == 0:
            trial += 1
            continue
        normal /= norm
        count = int(np.count_nonzero(np.abs((points - first) @ normal) <= threshold_m))
        if count > best_count:
            best_count, best_candidate = count, (first, normal)
            ratio = min(max(count / len(points), 1e-9), 1.0 - 1e-12)
            target_trials = min(10_000, max(128, int(np.ceil(np.log(0.001) / np.log(1.0 - ratio**3)))))
        trial += 1
    return best_candidate, trial

def test_adaptive_ransac_workspace_matches_reference_candidate_exactly():
    table = make_plane_grid(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    points = np.vstack([table, np.random.default_rng(4).uniform(-2.0, 2.0, (900, 3))])
    expected, expected_trials = _reference_adaptive_ransac(points, 0.012, 13)
    actual = _adaptive_ransac_plane(points, 0.012, 13)
    np.testing.assert_array_equal(actual.point, expected[0])
    np.testing.assert_array_equal(actual.normal, expected[1])
    assert actual.trial_count == expected_trials
    assert actual.early_exit is False

def test_perfect_candidate_returns_on_first_non_degenerate_trial():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    outcome = _adaptive_ransac_plane(points, 0.012, 13)
    assert outcome.early_exit is True
    assert outcome.trial_count == 1
```

- [ ] **Step 2: Run the focused tests and confirm they fail because `RansacOutcome` and `ransac` diagnostics do not exist**

Run: `cd code && uv run --active --no-project python -m pytest -q tests/test_ground_stack_footprint.py -k 'workspace or early_exit'`

Expected: FAIL through missing diagnostics or the old two-value return contract.

- [ ] **Step 3: Implement the result type and the allocation-free trial loop**

```python
@dataclass(frozen=True)
class RansacOutcome:
    point: np.ndarray
    normal: np.ndarray
    trial_count: int
    early_exit: bool

offsets = np.empty_like(points)
distances = np.empty(len(points), dtype=points.dtype)
np.subtract(points, first, out=offsets)
np.matmul(offsets, normal, out=distances)
np.abs(distances, out=distances)
count = int(np.count_nonzero(distances <= threshold_m))
```

Preserve the original trial increment and adaptive target update order. Return only from this function when `count == len(points)`, and make `select_support_plane` continue its outer candidate loop while adding exactly `trial_count` and `early_exit` beneath each candidate's `ransac` key. Update the existing line-130 test monkeypatch in `test_ground_stack_footprint.py` to return `RansacOutcome(np.zeros(3), np.array([0.0, 0.0, 1.0]), 0, False)`.

- [ ] **Step 4: Add boundary and ambiguity assertions**

```python
def test_adaptive_ransac_keeps_threshold_plus_minus_one_ulp_behavior():
    threshold = np.float64(0.012)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.2, threshold]])
    for candidate_threshold in (np.nextafter(threshold, 0.0), np.nextafter(threshold, np.inf)):
        expected, expected_trials = _reference_adaptive_ransac(points, candidate_threshold, 13)
        actual = _adaptive_ransac_plane(points, candidate_threshold, 13)
        np.testing.assert_array_equal(actual.point, expected[0])
        np.testing.assert_array_equal(actual.normal, expected[1])
        assert actual.trial_count <= expected_trials

def test_adaptive_ransac_tie_preserves_first_strictly_better_triplet():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
    expected, expected_trials = _reference_adaptive_ransac(points, 0.012, 13)
    outcome = _adaptive_ransac_plane(points, 0.012, 13)
    np.testing.assert_array_equal(outcome.point, expected[0])
    np.testing.assert_array_equal(outcome.normal, expected[1])
    assert outcome.trial_count == expected_trials
```

- [ ] **Step 5: Document the additive RANSAC diagnostics**

State in both READMEs that `measurement_report.json` records per support-plane candidate `ransac.trial_count` and `ransac.early_exit` for performance audit; they do not relax any support-plane gate or change the m² definition.

- [ ] **Step 6: Run geometry and stage contracts**

Run: `cd code && uv run --active --no-project python -m pytest -q tests/test_ground_stack_footprint.py tests/test_da3_footprint_stage.py`

Expected: PASS; no existing candidate diagnostic changes except the new `ransac` object.

- [ ] **Step 7: Commit the isolated optimization**

```bash
git add code/utils/ground_stack_footprint.py code/tests/test_ground_stack_footprint.py README.md code/README.md
git commit -m "perf: reuse footprint RANSAC workspaces"
```

### Task 2: Add the immutable SAM3 frame-mask cache primitive

**Files:**
- Create: `code/utils/sam3_mask_cache.py`
- Create: `code/tests/test_sam3_mask_cache.py`
- Modify: `README.md:112-120`
- Modify: `code/README.md:153-155`

**Interfaces:**
- Consumes: one image path, ordered `(object_id, bbox_xyxy)` prompts, checkpoint path/digest, code/runtime fingerprint, output `(N,H,W)` source-mask contract, and a `compute_masks()` callback.
- Produces: `FrameMaskCacheResult(masks, key, events, payload_sha256, checkpoint_sha256, code_fingerprint, invalid_reason)`, where `events` is an ordered nonempty tuple of `hit`, `miss`, `invalid`, `written`, or `cache_write_failed`.

- [ ] **Step 1: Write failing cache-key, hit, and invalid-payload tests**

```python
def test_frame_cache_hit_does_not_call_mask_producer(tmp_path):
    request = _request(tmp_path, detections=((7, (1.0, 2.0, 8.0, 9.0)),))
    first = load_or_compute_frame_masks(request, _producer_counting_calls())
    second = load_or_compute_frame_masks(request, _producer_that_fails_if_called())
    assert np.array_equal(first.masks[0], second.masks[0])
    assert second.events == ("hit",)

@pytest.mark.parametrize("mutation", ["image", "bbox", "order", "checkpoint", "contract"])
def test_key_input_mutation_is_a_miss(tmp_path, mutation):
    first_request = _request(tmp_path, detections=((7, (1.0, 2.0, 8.0, 9.0)),))
    load_or_compute_frame_masks(first_request, lambda: [np.ones((8, 8), dtype=bool)])
    changed_request = _mutate_request(first_request, mutation)
    result = load_or_compute_frame_masks(changed_request, lambda: [np.ones((8, 8), dtype=bool)])
    assert result.events == ("miss", "written")

def test_corrupt_payload_is_quarantined_then_recomputed(tmp_path):
    request = _request(tmp_path, detections=((7, (1.0, 2.0, 8.0, 9.0)),))
    first = load_or_compute_frame_masks(request, lambda: [np.ones((8, 8), dtype=bool)])
    (request.cache_root / "entries" / first.key / "masks.npz").write_bytes(b"truncated")
    second = load_or_compute_frame_masks(request, lambda: [np.ones((8, 8), dtype=bool)])
    assert second.events == ("invalid", "written")
    assert any((request.cache_root / "corrupt").iterdir())
```

- [ ] **Step 2: Run the new tests and confirm import failure**

Run: `cd code && uv run --active --no-project python -m pytest -q tests/test_sam3_mask_cache.py`

Expected: FAIL because `utils.sam3_mask_cache` does not exist.

- [ ] **Step 3: Implement canonical request/result types and key construction**

```python
@dataclass(frozen=True)
class DetectionPrompt:
    object_id: int
    bbox_xyxy_f64be_hex: tuple[str, str, str, str]

@dataclass(frozen=True)
class FrameMaskCacheRequest:
    cache_root: Path
    image_id: int
    image_path: Path
    detections: Sequence[DetectionPrompt]
    checkpoint_path: Path
    checkpoint_sha256: str
    code_fingerprint: dict[str, object]
    runtime_fingerprint: dict[str, object]
    inference_contract: dict[str, object]
    output_shape_hw: tuple[int, int]

def canonical_frame_mask_key(request: FrameMaskCacheRequest) -> str:
    return _sha256_bytes(_canonical_json_bytes(_request_key_payload(request)))
```

Reject non-finite bbox coordinates before converting them to normalized binary64 big-endian hex. Serialize canonical JSON with sorted keys and compact separators. Do not include global mapping or DA3 cache fields in this key.

- [ ] **Step 4: Implement bundle validation, locking, and publication**

```python
def load_or_compute_frame_masks(
    request: FrameMaskCacheRequest,
    compute_masks: Callable[[], Sequence[np.ndarray]],
) -> FrameMaskCacheResult:
    key = canonical_frame_mask_key(request)
    with _key_lock(request.cache_root, key, exclusive=False):
        cached = _load_valid_bundle(request)
        if cached is not None:
            return cached
    with _key_lock(request.cache_root, key, exclusive=True):
        cached = _load_valid_bundle(request)
        if cached is not None:
            return cached
        masks = _validate_produced_masks(request, tuple(compute_masks()))
        return _publish_new_bundle(request, masks)
```

At the top of this test module define `_request`, `_mutate_request`, `_producer_counting_calls`, `_producer_that_fails_if_called`, and `_concurrently_load_same_request`; each creates the `FrameMaskCacheRequest` shown above with an 8 by 8 PNG/checkpoint fixture and uses `multiprocessing.Barrier(2)` for the concurrent producer. Use `entries/<key>/masks.npz` and `manifest.json`; require `complete: true`, `allow_pickle=False`, exactly one bool array of shape `(N,H,W)`, payload SHA-256, each mask digest/count, and no true pixels outside its clipped bbox. Hold a shared `fcntl.flock` while reading a final bundle. On invalid final content, take exclusive lock, atomically rename the directory to `corrupt/<key>.<uuid>`, fsync both parents, recompute, fsync a sibling temporary bundle, then rename it only to a nonexistent `entries/<key>` and fsync `entries`.

- [ ] **Step 5: Add concurrent writer and empty-mask tests**

```python
def test_two_processes_publish_one_complete_bundle(tmp_path):
    first, second = _concurrently_load_same_request(tmp_path)
    assert np.array_equal(first.masks[0], second.masks[0])
    assert len(list((tmp_path / "sam3_mask_cache" / "v1" / "entries").iterdir())) == 1

def test_complete_empty_mask_is_cached_without_bbox_fallback(tmp_path):
    result = load_or_compute_frame_masks(_request(tmp_path), lambda: [np.zeros((8, 8), bool)])
    assert result.masks[0].sum() == 0
```

- [ ] **Step 6: Run cache tests**

Run: `cd code && uv run --active --no-project python -m pytest -q tests/test_sam3_mask_cache.py`

Expected: PASS; the producer is not called on valid hit and malformed bundles never return masks.

- [ ] **Step 7: Document the internal cache primitive accurately**

State in both READMEs that the per-frame SAM3 cache utility has immutable verified bundles, but the public `ground-stack-area` command starts using it only when Task 4 lands. Explain that it cannot provide a bbox fallback or a partial total and that users must not depend on the cache directory before stage integration.

- [ ] **Step 8: Commit the cache primitive**

```bash
git add code/utils/sam3_mask_cache.py code/tests/test_sam3_mask_cache.py README.md code/README.md
git commit -m "feat: add audited SAM3 frame mask cache"
```

### Task 3: Bind SAM3 model caching to immutable checkpoint evidence

**Files:**
- Modify: `code/utils/sam3_utils.py:129-152,550-745`
- Modify: `code/tests/test_sam3_mask_cache.py`

**Interfaces:**
- Consumes: checkpoint bytes and normalized CUDA device plus the fixed `predict_inst` inference contract.
- Produces: `checkpoint_sha256(path)` and SAM3 process-cache identity `(checkpoint_sha256, normalized_device, inference_contract_fingerprint)`.

- [ ] **Step 1: Write failing checkpoint-replacement tests**

```python
def test_model_cache_key_changes_when_checkpoint_bytes_change(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path, b"first")
    _load_model_for_digest(checkpoint)
    checkpoint.write_bytes(b"second")
    _load_model_for_digest(checkpoint)
    assert _model_loader_call_count() == 2

def test_digest_change_during_model_load_rejects_without_cache_publish(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path, b"first")
    expected = checkpoint_sha256(checkpoint)
    monkeypatch.setattr(sam3_utils, "_build_sam3_model_and_processor", lambda *_: checkpoint.write_bytes(b"second"))
    with pytest.raises(RuntimeError, match="changed while loading"):
        sam3_utils._get_sam3_model_and_processor(str(checkpoint), "cuda", expected_checkpoint_sha256=expected)
```

- [ ] **Step 2: Run the focused tests and confirm old path-only cache behavior fails**

Run: `cd code && uv run --active --no-project python -m pytest -q tests/test_sam3_mask_cache.py -k checkpoint`

Expected: FAIL because the current model cache key is `(checkpoint_path, device)`.

- [ ] **Step 3: Implement before/after digest verification**

```python
def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _get_sam3_model_and_processor(
    checkpoint_path: str, device: str, *, expected_checkpoint_sha256: str
) -> tuple[object, object]:
    before = checkpoint_sha256(Path(checkpoint_path))
    if before != expected_checkpoint_sha256:
        raise RuntimeError("SAM3 checkpoint digest changed before loading")
    cache_key = (before, _normalize_device(device), _PREDICT_INST_CONTRACT_FINGERPRINT)
    cached = _SAM3_PREDICT_INST_CACHE.get(cache_key)
    if cached is None:
        cached = _build_sam3_model_and_processor(checkpoint_path, device)
        _SAM3_PREDICT_INST_CACHE[cache_key] = cached
    model, processor = cached
    if checkpoint_sha256(Path(checkpoint_path)) != before:
        raise RuntimeError("SAM3 checkpoint changed while loading")
    return model, processor
```

Define `_build_sam3_model_and_processor(checkpoint_path, device)` by moving the current `_ensure_sam3_in_path`, `build_sam3_image_model`, and `Sam3Processor` construction into that helper. Hash before load, require it equals `expected_checkpoint_sha256`, hash again after load, and raise `RuntimeError` on change. Make `sam3_masks_from_bboxes_predict_inst` compute/pass the expected digest and include it in the in-process cache key. Keep `predict_inst`, box batching, candidate selection, clipping, and center-point retry unchanged.

- [ ] **Step 4: Run cache tests and existing footprint stage tests**

Run: `cd code && uv run --active --no-project python -m pytest -q tests/test_sam3_mask_cache.py tests/test_da3_footprint_stage.py`

Expected: PASS; no cache entry can claim a different checkpoint from the loaded model.

- [ ] **Step 5: Commit checkpoint provenance hardening**

```bash
git add code/utils/sam3_utils.py code/tests/test_sam3_mask_cache.py
git commit -m "fix: bind SAM3 cache to checkpoint digest"
```

### Task 4: Integrate cached masks and publish immutable artifact generations

**Files:**
- Modify: `code/modules/da3_footprint_stage.py:60-179,357-450`
- Modify: `code/tests/test_da3_footprint_stage.py`
- Modify: `README.md:112-120`
- Modify: `code/README.md:149-155`

**Interfaces:**
- Consumes: `load_or_compute_frame_masks()` after existing complete DA3/image/detection/mapping validation.
- Produces: report `sam3_mask_cache.frames[]` and immutable `ground_stack_footprint/runs/<run_id>/{measurement_report.json,footprints.geojson,top_down_footprint.png,manifest.json}` selected by `CURRENT`.

- [ ] **Step 1: Write failing stage hit/miss, empty-cache-mask, and generation tests**

```python
def test_stage_second_run_uses_cached_masks_and_keeps_area(monkeypatch, tmp_path):
    first = _run_fixture(tmp_path, monkeypatch)
    second = _run_fixture_with_sam3_that_raises_if_called(tmp_path, monkeypatch)
    assert _formal_projection(first) == _formal_projection(second)
    assert second["sam3_mask_cache"]["frames"][0]["events"] == ["hit"]

def test_cached_empty_mask_rejects_full_total(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    monkeypatch.setattr(stage, "load_or_compute_frame_masks", _cache_result_with_empty_frame_two_mask)
    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["status"] == "rejected"
    assert report["value_m2"] is None

def test_current_points_to_complete_single_artifact_generation(tmp_path):
    result = _run_fixture(tmp_path, monkeypatch)
    current = json.loads((Path(result["report_path"]).parent.parent.parent / "CURRENT").read_text())
    generation = Path(result["report_path"]).parent
    assert {path.name for path in generation.iterdir()} == {"measurement_report.json", "footprints.geojson", "top_down_footprint.png", "manifest.json"}
```

Replace the existing accepted-stage `output_names` assertion with this four-file generation assertion and add an assertion that the output root has a `CURRENT` file and `runs` directory. Keep rejected GeoJSON verification, but resolve it from `Path(result["report_path"]).parent`.

- [ ] **Step 2: Run the focused tests and confirm current direct-SAM3/direct-file output fails**

Run: `cd code && uv run --active --no-project python -m pytest -q tests/test_da3_footprint_stage.py -k 'cached or current or generation'`

Expected: FAIL because the stage has no cache report section or generation publisher.

- [ ] **Step 3: Integrate the cache without weakening validation**

```python
request = FrameMaskCacheRequest(
    cache_root=cache_root, image_id=image_id, image_path=image_paths[image_id],
    detections=tuple(_detection_prompt(item) for item in frame_detections),
    checkpoint_path=Path(_SAM3_CHECKPOINT), checkpoint_sha256=checkpoint_sha256,
    code_fingerprint=_sam3_code_fingerprint(), runtime_fingerprint=_sam3_runtime_fingerprint(_SAM3_DEVICE),
    inference_contract=_PREDICT_INST_CONTRACT, output_shape_hw=source_image_hw,
)
cache_result = load_or_compute_frame_masks(request, compute_masks=lambda: sam3_masks_from_bboxes_predict_inst(
    str(image_paths[image_id]), [item["bbox"] for item in frame_detections], _SAM3_CHECKPOINT, _SAM3_DEVICE,
))
report["sam3_mask_cache"]["frames"].append(cache_result.report_entry())
masks = cache_result.masks
```

Run `_validate_complete_source_ids`, `_validate_cache`, `_load_detections`, and `_load_mapping` before the first request. Preserve source-mask shape, warp, valid-point, background dilation, 32-point, 64-point, component, OBB, union, and all-ID rejection checks. Treat cache write failure as fresh masks with reported `cache_write_failed`; treat producer failure or invalid/empty resulting masks through existing formal rejection semantics.

- [ ] **Step 4: Implement generation publication and crash tests**

```python
def _publish_generation(output_root: Path, report: dict[str, Any], polygons: dict[str, Any], union: Any) -> dict[str, str]:
    run_id = uuid.uuid4().hex
    report["artifacts"] = {
        "measurement_report": "measurement_report.json",
        "footprints_geojson": "footprints.geojson",
        "top_down_footprint_png": "top_down_footprint.png",
    }
    generation = _write_fsynced_generation(output_root / "runs", run_id, report, polygons, union)
    _atomic_replace_current(output_root / "CURRENT", {"run_id": run_id, "complete": True})
    return _artifact_paths_from_current(output_root)
```

Write report, GeoJSON, PNG, and manifest into an fsynced immutable `runs/<run_id>` directory. The report stores only the three generation-relative artifact names before its first and only write; the manifest stores SHA-256 for those three files plus the report. Atomically replace `CURRENT` only after the generation exists. All return paths must resolve through CURRENT. Tests monkeypatch replace/write failure at every publication boundary and assert CURRENT still names a complete old generation or no generation.

- [ ] **Step 5: Update user documentation**

Document the cache root, key invalidation conditions, audited hit/miss statuses, corruption recomputation, `CURRENT` resolution, immutable prior generations, and the fact that cache reuse never permits a bbox fallback or partial total.

- [ ] **Step 6: Run focused cache/stage tests and compile**

Run: `cd code && uv run --active --no-project python -m pytest -q tests/test_sam3_mask_cache.py tests/test_da3_footprint_stage.py && uv run --active --no-project python -m py_compile modules/da3_footprint_stage.py utils/sam3_mask_cache.py utils/sam3_utils.py`

Expected: PASS; no direct output-file consumer assumption remains in README tests.

- [ ] **Step 7: Commit stage integration and docs**

```bash
git add code/modules/da3_footprint_stage.py code/tests/test_da3_footprint_stage.py README.md code/README.md
git commit -m "feat: cache SAM3 masks for footprint measurement"
```

### Task 5: Add a shadow-only camera evidence primitive

**Files:**
- Create: `code/utils/footprint_evidence.py`
- Create: `code/tests/test_footprint_evidence.py`

**Interfaces:**
- Consumes: NPZ path, cache frame IDs, `EvidenceObservation(global_id, image_id, object_id, processed_mask, valid_mask)`, and frozen `SupportPlane` plus per-ID full-data polygons.
- Produces: a JSON-safe `dict` with `status`, `camera_contract`, `per_global_id`, pairwise reprojection counters/residuals, `distinct_image_id_count`, and fixed-plane leave-one-observation-out metrics.

- [ ] **Step 1: Write failing camera-contract and reprojection tests**

```python
def test_missing_camera_fields_returns_unavailable_without_raise(tmp_path):
    evidence = build_shadow_evidence(_formal_only_npz(tmp_path), observations=(), formal_snapshot=_snapshot())
    assert evidence["status"] == "unavailable_missing_camera_fields"

def test_bidirectional_reprojection_classifies_occluded_and_foreground_conflict(tmp_path):
    evidence = build_shadow_evidence(_two_view_camera_npz(tmp_path), observations=_observations(), formal_snapshot=_snapshot())
    pair = evidence["per_global_id"]["1"]["pairs"][0]
    assert pair["occluded_count"] == 1
    assert pair["foreground_conflict_count"] == 1

def test_loo_metrics_change_when_the_only_side_view_is_removed(tmp_path):
    evidence = build_shadow_evidence(_three_view_camera_npz(tmp_path), observations=_three_view_observations(), formal_snapshot=_snapshot())
    loo = evidence["per_global_id"]["1"]["leave_one_observation_out"]
    assert max(item["polygon_iou"] for item in loo) < 1.0
```

- [ ] **Step 2: Run the pure evidence tests and confirm import failure**

Run: `cd code && uv run --active --no-project python -m pytest -q tests/test_footprint_evidence.py`

Expected: FAIL because `utils.footprint_evidence` does not exist.

- [ ] **Step 3: Implement the optional camera loader and validation**

```python
def build_shadow_evidence(
    cache_path: Path,
    *,
    cache_frame_ids: np.ndarray,
    observations: Sequence[EvidenceObservation],
    formal_snapshot: FormalSnapshot,
) -> dict[str, object]:
    try:
        camera = _load_optional_camera_cache(cache_path, cache_frame_ids)
        return _build_valid_camera_evidence(camera, observations, formal_snapshot)
    except EvidenceUnavailable as error:
        return {"mode": "shadow", "status": error.status, "reason": str(error)}
    except Exception as error:
        return {"mode": "shadow", "status": "failed_evidence", "reason": str(error)}
```

Load `world_points`, `world_points_conf`, `depth`, `intrinsic`, and `extrinsic` in a second `np.load(cache_path, allow_pickle=False)` only. Require `world_points=(N,H,W,3)`, `world_points_conf=(N,H,W)`, `depth=(N,H,W,1)`, `intrinsic=(N,3,3)`, `extrinsic=(N,3,4)`, finite arrays, positive focal lengths, `R.T @ R` within `1e-4` of identity, `det(R) > 0`, and invertible homogeneous world-to-camera transform. Reconstruct source-frame world points from depth/intrinsic/extrinsic and report their residual to stored `world_points`. Return `unavailable_missing_camera_fields` for absent camera fields and `failed_camera_contract` for invalid data; do not raise to the stage.

- [ ] **Step 4: Implement deterministic pairwise reprojection and fixed-plane LOO**

```python
DEPTH_TOLERANCE_M = 0.020  # shadow diagnostic only, not a formal threshold

def _classify_target_depth(z_projected: np.ndarray, z_target: np.ndarray) -> np.ndarray:
    return np.where(z_projected > z_target + DEPTH_TOLERANCE_M, "occluded", np.where(
        z_projected < z_target - DEPTH_TOLERANCE_M, "foreground_conflict", "visible_consistent"))

def _leave_one_observation_out(observations: Sequence[EvidenceObservation], plane: SupportPlane) -> list[dict[str, object]]:
    return [_loo_metrics_for_removed_observation(observations, plane, index) for index in range(len(observations))]
```

Sample at most 512 confidence-qualified source-mask grid points in increasing flattened-index order. Project `X_world` through target world-to-camera `[R|t]` and intrinsic. Count behind-camera, outside-grid, occluded (`z_projected > z_target + 0.020`), visible-consistent (`abs(delta) <= 0.020`), foreground-conflict (`z_projected < z_target - 0.020`), visible-mask-supported, and visible-mask-unsupported. Run every pair in both directions. For fixed plane LOO, remove one observation, repeat current projection/5-mm voxel/component/OBB logic, and report polygon IoU, Hausdorff distance, centre, angle, side, and area changes relative to the frozen full-data OBB. A one-observation ID reports `single_observation_insufficient_cross_view_evidence`.

- [ ] **Step 5: Run pure evidence tests**

Run: `cd code && uv run --active --no-project python -m pytest -q tests/test_footprint_evidence.py`

Expected: PASS; all outputs are JSON-safe and malformed optional camera data raises no exception.

- [ ] **Step 6: Commit the pure evidence utility**

```bash
git add code/utils/footprint_evidence.py code/tests/test_footprint_evidence.py
git commit -m "feat: add shadow multiview footprint evidence"
```

### Task 6: Attach shadow evidence after formal result freeze

**Files:**
- Modify: `code/modules/da3_footprint_stage.py:60-179,357-407`
- Modify: `code/tests/test_da3_footprint_stage.py`
- Modify: `README.md:112-120`
- Modify: `code/README.md:149-155`

**Interfaces:**
- Consumes: frozen formal snapshot with `plane: SupportPlane | None`, `EvidenceObservation` emitted while masks are warped, and `build_shadow_evidence()`.
- Produces: additive `report["evidence"]`, with no mutation of formal status/value/polygons/union/rejection reason.

- [ ] **Step 1: Write failing formal-isolation tests**

```python
@pytest.mark.parametrize("formal_status", ["accepted", "rejected"])
def test_evidence_failures_do_not_change_frozen_formal_result(monkeypatch, tmp_path, formal_status):
    baseline = _run_fixture_for_status(tmp_path, monkeypatch, formal_status, evidence_enabled=False)
    monkeypatch.setattr(stage, "build_shadow_evidence", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
    observed = _run_fixture_for_status(tmp_path, monkeypatch, formal_status, evidence_enabled=True)
    assert _formal_projection(observed) == _formal_projection(baseline)
    expected_status = "failed_evidence" if formal_status == "accepted" else "unavailable_no_formal_geometry"
    assert observed["evidence"]["status"] == expected_status
    assert _geojson_geometry(observed) == _geojson_geometry(baseline)
```

- [ ] **Step 2: Run the focused test and confirm the stage cannot yet attach evidence**

Run: `cd code && uv run --active --no-project python -m pytest -q tests/test_da3_footprint_stage.py -k evidence`

Expected: FAIL because the report has no evidence section or exception isolation.

- [ ] **Step 3: Capture observation masks and freeze formal state**

```python
formal_snapshot = FormalSnapshot(
    status=report["status"], value_m2=report["value_m2"], plane=plane,
    polygons=dict(polygons), union=union, rejection_reason=report.get("rejection_reason"),
)
report["evidence"] = _attach_shadow_evidence(cache_path, cache, observations, formal_snapshot)
```

Initialize `plane = None` before the formal block and pass it unchanged into a rejected `FormalSnapshot`. Make `_masked_observations()` preserve processed-grid source masks and valid masks in `EvidenceObservation` alongside its existing point lists. Call `_attach_shadow_evidence()` after the formal `try/except` has set final formal values and before generation publication. For a rejected snapshot with no plane, return `{"mode": "shadow", "status": "unavailable_no_formal_geometry"}` without LOO. Catch `Exception` only inside `_attach_shadow_evidence()` and return `{ "mode": "shadow", "status": "failed_evidence", "reason": str(error) }`; it must not mutate its inputs.

- [ ] **Step 4: Add valid-camera, malformed-camera, duplicate-view, and wrong-mask fixture coverage**

```python
def test_stage_reports_valid_shadow_evidence_without_changing_formal_area(monkeypatch, tmp_path):
    assert _formal_projection(_run_with_valid_camera(tmp_path, monkeypatch)) == _formal_projection(_run_without_camera(tmp_path, monkeypatch))

def test_stage_reports_failed_camera_contract_without_changing_formal_area(monkeypatch, tmp_path):
    baseline = _run_without_camera(tmp_path, monkeypatch)
    observed = _run_with_bad_camera(tmp_path, monkeypatch)
    assert observed["evidence"]["status"] == "failed_camera_contract"
    assert _formal_projection(observed) == _formal_projection(baseline)

def test_duplicate_observation_does_not_increase_distinct_image_count(monkeypatch, tmp_path):
    assert _run_with_duplicate_view(tmp_path, monkeypatch)["evidence"]["per_global_id"]["1"]["distinct_image_id_count"] == 2

def test_wrong_mask_is_the_highest_leave_one_out_influence(monkeypatch, tmp_path):
    assert _highest_loo_influence(_run_with_wrong_mask(tmp_path, monkeypatch), global_id="1")["image_id"] == 1
```

- [ ] **Step 5: Update documentation**

State that `evidence.mode=shadow` is internal consistency and sensitivity reporting, not calibrated uncertainty or accuracy. Document the 20-mm diagnostic depth tolerance, bidirectional occlusion semantics, single-observation label, and the prerequisite independent calibration before a hard gate or error bound.

- [ ] **Step 6: Run all changed tests, compile, and inspect the staged diff**

Run: `cd code && uv run --active --no-project python -m pytest -q tests/test_ground_stack_footprint.py tests/test_sam3_mask_cache.py tests/test_footprint_evidence.py tests/test_da3_footprint_stage.py && uv run --active --no-project python -m py_compile modules/da3_footprint_stage.py utils/footprint_evidence.py && git diff --check`

Expected: PASS; evidence failures preserve both accepted and rejected formal snapshots exactly.

- [ ] **Step 7: Commit evidence attachment and documentation**

```bash
git add code/modules/da3_footprint_stage.py code/tests/test_da3_footprint_stage.py README.md code/README.md
git commit -m "feat: report shadow multiview footprint evidence"
```

### Task 7: Verify measured performance and preserve a reproducible audit trail

**Files:**
- Modify: `README.md:112-120`
- Modify: `code/README.md:149-155`

**Interfaces:**
- Consumes: fixed `fd_area_test` inputs, the existing CUDA host, and formal reports before/after Track A/B.
- Produces: documented three-run timing table plus cache miss/hit formal-equivalence evidence; no generated artifacts are committed.

- [ ] **Step 1: Run CPU parity before GPU measurement**

Run: `cd code && uv run --active --no-project python -m pytest -q tests/test_ground_stack_footprint.py tests/test_sam3_mask_cache.py tests/test_footprint_evidence.py tests/test_da3_footprint_stage.py`

Expected: PASS before accessing the real video workload.

- [ ] **Step 2: Run fixed-input cache miss and hit measurements three times each**

Run: `cd code && uv run --active --no-project python main.py --mode ground-stack-area --dataset /home/xingyu/3D_Recognization/code/video_dedup_runs/fd_area_test --save_root /home/xingyu/3D_Recognization/code/Output`

Expected: the same formal result and formal geometry for miss/hit pairs; record RANSAC `trial_count`, `early_exit`, segment seconds, cache statuses, and report digests outside Git.

- [ ] **Step 3: Add only stable user-facing performance facts to documentation**

Describe observed median latency separately for cold and warm SAM3 cache, host/GPU envelope, and the fact that timing is not a geometry-accuracy claim. Do not copy cache payloads, reports, GeoJSON, PNGs, checkpoints, or video inputs into the commit.

- [ ] **Step 4: Validate and commit documentation only**

Run: `git diff --check`

```bash
git add README.md code/README.md
git commit -m "docs: record footprint cache performance evidence"
```

## Plan Self-Review

- Spec coverage: Task 1 implements strict-equivalent RANSAC and diagnostics; Tasks 2-4 implement cache provenance, checkpoint TOCTOU protection, corruption handling, and immutable artifacts; Tasks 5-6 implement optional shadow camera evidence and isolation; Task 7 implements the required measured validation and documentation.
- Scope: Each task has a separately testable output and a separate commit. Track C depends only on frozen formal state, not on cache acceptance semantics.
- Naming consistency: `FrameMaskCacheRequest`, `FrameMaskCacheResult`, `RansacOutcome`, `EvidenceObservation`, `FormalSnapshot`, and `build_shadow_evidence` are defined where first used and consumed with the same names later.
- Placeholder scan: The plan contains no incomplete implementation marker, unbound interface, or unqualified error-handling instruction.
