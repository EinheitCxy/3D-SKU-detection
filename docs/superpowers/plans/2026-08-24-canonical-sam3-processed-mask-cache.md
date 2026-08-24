# Canonical Processed-Space SAM3 Mask Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SKU matching the sole producer of losslessly packed processed-space self-exemplar SAM3 masks and make footprint, evidence, and viewer export strict read-only consumers.

**Architecture:** `utils/sam3_mask_cache.py` becomes a per-image processed-grid cache under `sam3_mask_cache/v2`, keyed by exact manifest fields rather than content hashes. Matching publishes complete frames; footprint and exporter load one frame at a time, while the formal metric and browser bundle move to version 2 contracts with no v1 fallback.

**Tech Stack:** Python 3.11, `uv`, NumPy packbits, OpenCV, PyTorch/SAM3 only in matching, pytest, TypeScript, Vitest, Three.js/Vite.

**Spec:** `docs/superpowers/specs/2026-08-24-canonical-sam3-processed-mask-cache-design.md`

## Global Constraints

- Keep `enable_sam3_mask_sampling` as a default-true master gate; remove `sam3_use_self_exemplar` everywhere except the exact removed-key error test/message.
- The removed-key error is exactly: `sam3_use_self_exemplar was removed; self-exemplar is now the only SAM3 mode`.
- Matching is the only SAM3 producer. Footprint and viewer export must not import/build/load/run SAM3 or compute masks on cache miss.
- v2 root is exactly `{save_root}/{dataset}/sam3_mask_cache/v2`; never read, migrate, fallback to, or delete v1.
- Cache schema is exactly `sam3_self_exemplar_processed_mask_cache_v1`; masks are processed-space two-dimensional bool arrays keyed by object ID.
- Use `np.packbits(..., axis=1, bitorder="little")` and uncompressed `np.savez`; load/unpack one frame at a time.
- Add no v2 content hash, signature, encryption, checkpoint/payload/source digest, code fingerprint, or runtime fingerprint.
- Preserve current self-exemplar preprocessing, detection order, matching batches, threshold 0.5, image size 1008, max batch 32, one detection per query, bbox clipping, 50-point sampling, and assignments.
- Missing/partial/mismatched cache is fail-closed; there is no bbox-mask fallback in footprint or exporter.
- Viewer filtering remains ordinary filtering; SAM3 labels never protect points from noise, ground, cluster, voxel, or sky filtering.
- Formal metric is `da3_self_exemplar_ground_footprint_union`; report, viewer CURRENT, and viewer manifest schema are `2.0.0`; old bundles fail with re-run guidance and no compatibility branch.
- Run all Python commands through `uv`. Use the existing root environment through `VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python ...` for no-GPU tests.
- Preserve unrelated dirty work. Every Terra implementer owns only its task files, does not spawn subagents, uses TDD, and stages an explicit allowlist.

## Locked task ownership

| Task | Files |
| --- | --- |
| 1 cache | `utils/sam3_mask_cache.py`, `tests/test_sam3_mask_cache.py` |
| 2 matching producer | `utils/sam3_utils.py`, `utils/matching_algorithms.py`, `utils/config.py`, `utils/sku_matching_system.py`, `src/inference.py`, `main.py`, `config.yaml`, `tests/test_matching_sam3_cache.py`, affected config tests |
| 3 footprint consumer | `src/da3_footprint_stage.py`, `tests/test_da3_footprint_stage.py`, `tests/test_ground_stack_footprint.py` only if metric fixtures require it |
| 4 evidence | `utils/footprint_evidence.py`, `tests/test_footprint_evidence.py`, evidence-only footprint fixtures |
| 5 exporter | `src/web_viewer_export.py`, `tests/test_web_viewer_export.py` |
| 6 frontend contract | `modules/viewer_web/src/contracts.ts`, `contracts.test.ts`, `bundle-loader.ts`, `bundle-loader.test.ts`, `presentation.ts`, `presentation.test.ts` |
| 7 docs/verification | `README.md`, `docs/3d_core.md`, `modules/viewer_web/README.md`, validation receipts |

---

### Task 1: Replace the canonical cache with processed-space packbits v2

**Files:**
- Modify: `utils/sam3_mask_cache.py`
- Modify: `tests/test_sam3_mask_cache.py`

**Interfaces:**
- Produces: `ProcessedDetectionPrompt`, `FrameMaskCacheRequest`, `FrameMaskCacheResult`, `FrameMaskCacheError`, `load_complete_frame_masks(request)`, and `load_or_compute_frame_masks(request, compute_masks)` exactly as the spec.
- Later consumers construct the same request and receive `Mapping[int, np.ndarray]` of independent processed-space bool masks.

- [ ] **Step 1: Replace old cache tests with v2 contract RED tests**

```python
def request(tmp_path: Path, detections: tuple[ProcessedDetectionPrompt, ...] = (
    ProcessedDetectionPrompt(7, (0.0, 0.0, 8.0, 8.0), (0.0, 0.0, 4.0, 4.0)),
)) -> FrameMaskCacheRequest:
    image = tmp_path / "7.jpg"
    image.write_bytes(b"immutable-fixture")
    return FrameMaskCacheRequest(
        cache_root=tmp_path / "sam3_mask_cache" / "v2",
        image_id=7,
        image_path=image,
        source_size_wh=(8, 8),
        processed_shape_hw=(4, 4),
        source_to_processed_affine=np.asarray([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float64),
        detections=detections,
        inference_contract={
            "api": "self_exemplar",
            "threshold": 0.5,
            "image_size": 1008,
            "max_batch_size": 32,
            "max_dets_per_query": 1,
            "clip_to_bbox": True,
        },
    )


def test_packbits_round_trip_is_byte_exact_and_object_keyed(tmp_path: Path) -> None:
    expected = np.asarray([
        [True, False, True, False],
        [False, True, False, True],
        [True, True, False, False],
        [False, False, True, True],
    ], dtype=bool)
    result = load_or_compute_frame_masks(request(tmp_path), lambda: {7: expected})
    loaded = load_complete_frame_masks(request(tmp_path))
    assert result.cache_event == "miss"
    assert loaded.cache_event == "hit"
    np.testing.assert_array_equal(loaded.masks_by_object_id[7], expected)
    assert loaded.masks_by_object_id[7].dtype == np.bool_
```

```python
def test_non_byte_aligned_tail_bits_are_lossless_and_validated(tmp_path: Path) -> None:
    prompt = ProcessedDetectionPrompt(7, (0.0, 0.0, 6.0, 6.0), (0.0, 0.0, 3.0, 3.0))
    req = dataclasses.replace(request(tmp_path), processed_shape_hw=(3, 3), detections=(prompt,))
    expected = np.eye(3, dtype=bool)
    load_or_compute_frame_masks(req, lambda: {7: expected})
    np.testing.assert_array_equal(load_complete_frame_masks(req).masks_by_object_id[7], expected)
    payload_path = req.cache_root / "entries" / "7" / "masks.npz"
    with np.load(payload_path, allow_pickle=False) as loaded:
        packed = loaded["packed_masks"].copy()
    packed[0, -1] |= np.uint8(0b10000000)
    np.savez(payload_path, packed_masks=packed)
    with pytest.raises(FrameMaskCacheError, match="tail bits"):
        load_complete_frame_masks(req)
```

- [ ] **Step 2: Run focused RED**

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python -m pytest -q tests/test_sam3_mask_cache.py
```

Expected: old source-space request/schema assertions fail.

- [ ] **Step 3: Implement exact request and manifest validation**

Use exact manifest keys from the spec. Validate path root name `v2`, safe IDs, finite bbox/affine values, source/processed bounds, unique object IDs, inference exact keys/values, payload exact keys, payload shape/dtype/tail bits, and that every true pixel lies within its declared processed bbox under the producer's canonical clip rule. Entry, lock, and corrupt paths are derived only from decimal `image_id`; no digest is computed.

```python
SCHEMA = "sam3_self_exemplar_processed_mask_cache_v1"


def _entry_path(request: FrameMaskCacheRequest) -> Path:
    return request.cache_root / "entries" / str(request.image_id)


def _lock_path(request: FrameMaskCacheRequest) -> Path:
    return request.cache_root / "locks" / f"{request.image_id}.lock"
```

- [ ] **Step 4: Implement lossless per-frame payload I/O**

```python
def _pack_masks(masks: np.ndarray) -> np.ndarray:
    mask_count = masks.shape[0]
    return np.packbits(masks.reshape(mask_count, -1), axis=1, bitorder="little")


def _unpack_masks(packed: np.ndarray, mask_count: int, shape_hw: tuple[int, int]) -> np.ndarray:
    flat_size = shape_hw[0] * shape_hw[1]
    unpacked = np.unpackbits(packed, axis=1, count=flat_size, bitorder="little")
    return unpacked.reshape(mask_count, *shape_hw).astype(bool, copy=False)
```

Write `packed_masks` with `np.savez`, not `np.savez_compressed`.

- [ ] **Step 5: Implement atomic load/compute with one writer**

Shared-lock hits never invoke compute. Exclusive-lock misses double-check, quarantine an invalid existing entry under `corrupt/{image_id}.{time_ns}`, require compute output keys to exactly equal requested object IDs, validate masks, write a same-filesystem temporary sibling, rename to the final entry, and return `cache_event="miss"`. A compute exception removes the temporary directory and leaves the previous readable entry/pointer state unchanged.

- [ ] **Step 6: Add completeness, mismatch, no-v1, interruption, and concurrency tests**

```python
def test_hit_never_calls_compute(tmp_path: Path) -> None:
    req = request(tmp_path)
    load_or_compute_frame_masks(req, lambda: {7: np.ones((4, 4), dtype=bool)})
    result = load_or_compute_frame_masks(req, lambda: (_ for _ in ()).throw(AssertionError("must not compute")))
    assert result.cache_event == "hit"


@pytest.mark.parametrize("field", ["processed_shape_hw", "source_to_processed_affine", "detections", "inference_contract"])
def test_request_mismatch_is_not_a_hit(tmp_path: Path, field: str) -> None:
    original = request(tmp_path)
    load_or_compute_frame_masks(original, lambda: {7: np.ones((4, 4), dtype=bool)})
    changed = changed_request(original, field)
    with pytest.raises(FrameMaskCacheError):
        load_complete_frame_masks(changed)
```

Test a missing object, duplicate object ID, reordered detections, non-bool/source-shaped payload, a compute exception, a pre-created partial temp directory, absence of all reads under a populated sibling v1 root, and two processes racing on one image with a shared counter that proves one compute.

- [ ] **Step 7: Run Task 1 tests and commit**

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python -m pytest -q tests/test_sam3_mask_cache.py
git add -- utils/sam3_mask_cache.py tests/test_sam3_mask_cache.py
git commit -m "feat: replace SAM3 cache with processed self-exemplar masks"
```

---

### Task 2: Make matching the sole cache producer

**Files:**
- Modify: `utils/sam3_utils.py`
- Modify: `utils/matching_algorithms.py`
- Modify: `utils/config.py`
- Modify: `utils/sku_matching_system.py`
- Modify: `src/inference.py`
- Modify: `main.py`
- Modify: `config.yaml`
- Create: `tests/test_matching_sam3_cache.py`
- Test config behavior in the new `tests/test_matching_sam3_cache.py`; no existing test currently names either SAM3 config key.

**Interfaces:**
- Consumes: Task 1 cache API.
- Produces: `get_self_exemplar_masks_for_reference(config, *, image_path, image_id, frame_detections, matching_object_ids, transform) -> dict[int, np.ndarray]`; `SKUMatchingConfig.sam3_mask_cache_root: str`.

- [ ] **Step 1: Capture the unchanged direct self-exemplar baseline before editing matching**

Create a temporary output root with the current DA3 cache linked read-only, then run the unchanged direct self-exemplar path:

```bash
mkdir -p /tmp/sam3-v2-direct-baseline
ln -s /home/xingyu/3D_Recognization/Output/floor_display2/da3_cache /tmp/sam3-v2-direct-baseline/da3_cache
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python -m src.inference \
  --config config.yaml \
  --image_folder imdata/floor_display2/images \
  --detection_dir imdata/floor_display2/detections_results \
  --output_dir /tmp/sam3-v2-direct-baseline \
  --reference_idx 0 \
  --max_images 3 \
  --algorithm 3d \
  --backend da3 \
  --device cuda \
  --seed 42 \
  --save_json
```

Preserve the generated `matching_summary.txt` and `correspondences.json` plus the exact git HEAD/device receipt. After the new producer exists, its GPU equivalence test calls both the unchanged low-level `sam3_masks_self_exemplar` function and the v2 cold/warm wrapper on the same processed image/bboxes to compare per-object masks, sampled points, and RNG states. If GPU/model/cache inputs are unavailable, record the hard gate and continue only with no-GPU implementation; do not claim matching equivalence.

- [ ] **Step 2: Write config and cache-boundary RED tests**

```python
def test_removed_self_exemplar_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="sam3_use_self_exemplar was removed; self-exemplar is now the only SAM3 mode"):
        build_matching_config({"sam3_use_self_exemplar": True})


def test_master_gate_defaults_true_and_cache_root_is_explicit() -> None:
    config = SKUMatchingConfig.for_3d_mapping(sam3_mask_cache_root="/tmp/output/dataset/sam3_mask_cache/v2")
    assert config.enable_sam3_mask_sampling is True
    assert not hasattr(config, "sam3_use_self_exemplar")
    assert config.sam3_mask_cache_root == "/tmp/output/dataset/sam3_mask_cache/v2"
```

Add injected producer tests proving miss computes once, second access hits, all frame object IDs publish even when matching consumes a subset, direct/cold/warm masks are byte-equal, and the old v1 root is never inspected.

- [ ] **Step 3: Run focused RED**

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python -m pytest -q tests/test_matching_sam3_cache.py
```

- [ ] **Step 4: Remove the mode branch and add explicit root plumbing**

Delete the dataclass/YAML field and conditional branch. Add `sam3_mask_cache_root` to the config allowlist. Set it in `main.py` from `save_root/dataset/sam3_mask_cache/v2`; when `src.inference` is invoked directly, set it from `Path(args.output_dir) / "sam3_mask_cache" / "v2"`. Reject the removed key before unknown-field filtering so the exact error is stable.

- [ ] **Step 5: Implement the complete-frame producer**

Construct prompts from the complete flattened frame detection order and map bboxes through the DA3 transform. The cache miss closure calls the existing final-space self-exemplar implementation with currently matched object IDs first in their unchanged order/batches, then remaining IDs. Return masks by object ID; matching slices only its original IDs and leaves sampling/assignment functions unchanged.

Use a bounded RNG preservation context around cache lookup/compute. Snapshot and restore Python `random`, NumPy, torch CPU, and selected CUDA RNG states so cold and hit paths enter point sampling identically. Tests use seeded fake producers that deliberately consume every RNG stream.

- [ ] **Step 6: Add batch-all-refs and concurrency tests**

Create a three-frame fixture and assert one complete v2 entry per detection frame after batch-all-refs. Use two concurrent references for the same image and assert one producer call plus two identical returned mappings. Compare fake direct, cold, and hit masks and the exact sampled point arrays.

- [ ] **Step 7: Run focused matching/config tests and commit**

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python -m pytest -q tests/test_matching_sam3_cache.py tests/test_main_pipeline.py
git add -- utils/sam3_utils.py utils/matching_algorithms.py utils/config.py utils/sku_matching_system.py src/inference.py main.py config.yaml tests/test_matching_sam3_cache.py tests/test_main_pipeline.py
git commit -m "feat: make matching the canonical self-exemplar mask producer"
```

---

### Task 3: Make formal footprint a cache-only consumer and upgrade the metric

**Files:**
- Modify: `src/da3_footprint_stage.py`
- Modify: `tests/test_da3_footprint_stage.py`
- Modify: `tests/test_ground_stack_footprint.py` only for the metric constant if needed

**Interfaces:**
- Consumes: Task 1 `load_complete_frame_masks` and complete detections/DA3 transforms.
- Produces: report schema `2.0.0`, metric `da3_self_exemplar_ground_footprint_union`, `mask_contract`, performance stage `load_self_exemplar_masks`.

- [ ] **Step 1: Write cache-only footprint RED tests**

```python
def test_footprint_uses_processed_masks_without_sam3_producer(monkeypatch, footprint_inputs) -> None:
    monkeypatch.setattr("utils.sam3_utils.sam3_masks_from_bboxes_predict_inst", lambda *a, **k: (_ for _ in ()).throw(AssertionError("SAM3 producer forbidden")))
    result = run_da3_footprint(**footprint_inputs.with_valid_v2_cache())
    assert result["report"]["metric"] == "da3_self_exemplar_ground_footprint_union"
    assert result["report"]["schema_version"] == "2.0.0"
    assert result["report"]["mask_contract"] == {
        "source": "sam3_self_exemplar",
        "coordinate_space": "da3_processed_pixels",
        "cache_schema": "sam3_self_exemplar_processed_mask_cache_v1",
    }


def test_missing_cache_rejects_with_canonical_message(footprint_inputs) -> None:
    result = run_da3_footprint(**footprint_inputs.without_v2_cache())
    assert result["report"]["status"] == "rejected"
    assert result["report"]["rejection_reason"] == "canonical self-exemplar mask cache is incomplete; run SKU matching first"
```

- [ ] **Step 2: Run focused RED**

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python -m pytest -q tests/test_da3_footprint_stage.py
```

- [ ] **Step 3: Delete producer/runtime code and load one processed frame**

Remove torch/checkpoint/SAM3 producer imports, constants, fingerprinting, miss callback, and source-mask warp. Build exact requests, use the read-only loader, compute `valid_grid` once per frame, and index `world_points` directly with `valid_grid & mask`. Dilate the union of processed masks with a 5x5 kernel for background support-plane exclusion.

- [ ] **Step 4: Upgrade formal report and timing**

Change metric/schema/mask contract everywhere in the stage and fixtures. Rename `sam3_source_masks` to `load_self_exemplar_masks`; do not leave both keys. Preserve all-ID rejection, `value_m2=null`, immutable generation, and existing artifact provenance.

- [ ] **Step 5: Add shape/affine/bbox/object mismatch tests**

Use a mutation table with cases `missing_frame`, `missing_object`, `duplicate_object`, `bbox_mismatch`, `processed_shape_mismatch`, `affine_mismatch`, and `wrong_schema`. For each fixture mutation, assert the same canonical cache-incomplete rejection and a producer spy call count of zero.

- [ ] **Step 6: Run focused footprint tests and commit**

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python -m pytest -q tests/test_da3_footprint_stage.py tests/test_ground_stack_footprint.py
git add -- src/da3_footprint_stage.py tests/test_da3_footprint_stage.py tests/test_ground_stack_footprint.py
git commit -m "feat: consume matching masks in ground footprint"
```

---

### Task 4: Move shadow evidence fully into processed-mask space

**Files:**
- Modify: `utils/footprint_evidence.py`
- Modify: `tests/test_footprint_evidence.py`
- Modify: evidence-specific fixtures in `tests/test_da3_footprint_stage.py`

**Interfaces:**
- Consumes: `EvidenceObservation(global_id, image_id, object_id, processed_mask, valid_mask)`.
- Produces: processed-space robustness report with coordinate space, variants, 3x3 kernel, one iteration; unchanged camera/leave-one-out results.

- [ ] **Step 1: Write processed-only observation and robustness tests**

```python
def test_evidence_observation_requires_no_source_mask_or_affine() -> None:
    fields = {field.name for field in dataclasses.fields(EvidenceObservation)}
    assert fields == {"global_id", "image_id", "object_id", "processed_mask", "valid_mask"}


def test_mask_robustness_operates_on_processed_grid(observation) -> None:
    result = compute_mask_robustness([observation])
    assert result["coordinate_space"] == "da3_processed_pixels"
    assert result["variants"] == ["original", "eroded", "dilated"]
    assert result["kernel"] == [3, 3]
    assert result["iterations"] == 1
```

- [ ] **Step 2: Run focused RED**

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python -m pytest -q tests/test_footprint_evidence.py
```

- [ ] **Step 3: Remove source-mask warping from evidence**

Delete `source_mask`, `source_to_processed_affine`, `warp_source_mask_nearest` usage, and the original-warp equality assertion. Apply original/erosion/dilation directly to `processed_mask`; keep `valid_mask` intersection and processed camera projection unchanged.

- [ ] **Step 4: Prove formal invariance and camera equivalence**

For accepted and rejected fixtures, snapshot formal `status`, `value_m2`, `rejection_reason`, and geometry before an injected evidence exception; assert byte/canonical equality afterward. Reuse fixed camera tensors to assert reprojection/depth residual values match the pre-change expected numbers.

- [ ] **Step 5: Run evidence/footprint tests and commit**

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python -m pytest -q tests/test_footprint_evidence.py tests/test_da3_footprint_stage.py
git add -- utils/footprint_evidence.py tests/test_footprint_evidence.py tests/test_da3_footprint_stage.py
git commit -m "feat: move footprint evidence to processed mask space"
```

---

### Task 5: Export viewer labels directly from processed masks

**Files:**
- Modify: `src/web_viewer_export.py`
- Modify: `tests/test_web_viewer_export.py`

**Interfaces:**
- Consumes: Task 1 read-only v2 loader, DA3 grid, global mapping identity, new footprint metric.
- Produces: processed-mask point labels/ranges and manifest source `sam3_mask` with exact schema/coordinate-space/producer fields.

- [ ] **Step 1: Write direct-label and no-producer RED tests**

```python
def test_processed_masks_label_grid_without_inverse_affine(exporter_inputs) -> None:
    report = export_viewer_bundle(**exporter_inputs.with_processed_v2_masks())
    objects = json.loads(Path(report["generation_dir"], "objects.json").read_text())
    assert objects["1"]["instances"][0]["point_index_range"][1] > 0
    assert report["manifest"]["source"]["sam3_mask"] == {
        "schema": "sam3_self_exemplar_processed_mask_cache_v1",
        "coordinate_space": "da3_processed_pixels",
        "producer": "sku_matching",
    }


def test_exporter_does_not_call_sam3(monkeypatch, exporter_inputs) -> None:
    monkeypatch.setattr("utils.sam3_utils.sam3_masks_self_exemplar", lambda *a, **k: (_ for _ in ()).throw(AssertionError("SAM3 forbidden")))
    export_viewer_bundle(**exporter_inputs.with_processed_v2_masks())
```

- [ ] **Step 2: Run focused RED**

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python -m pytest -q tests/test_web_viewer_export.py
```

- [ ] **Step 3: Delete source-mask and inverse-affine code**

Remove old schema constants, source image digest/index lookup, source-mask payload/hash/count validation, inverse affine, source coordinate rounding/clamping, and source bbox sampling. Load one processed frame and fill `frame_grid` in stable instance order using `covered & (frame_grid < 0)`.

- [ ] **Step 4: Preserve filter order and no-protection semantics**

Keep `filter_scene_points` before label propagation, cut label arrays with every keep/sky mask, and choose voxel representatives by confidence only. Add regression tests with labeled outliers/ground/sky points proving they are removed identically to unlabeled points; assert no `protect_mask` argument is passed.

- [ ] **Step 5: Add strict cache mismatch and overlap tests**

Use exporter fixture mutations named `missing_root`, `missing_frame`, `missing_object`, `duplicate_object`, `bbox_mismatch`, `shape_mismatch`, `affine_mismatch`, `wrong_schema`, and `partial_payload`; each must raise `WebViewerExportError`. Add two overlapping masks whose shared pixel is true and assert the first stable global-instance label owns it. Keep the existing zero-range no-geometry assertion unchanged.

- [ ] **Step 6: Upgrade exporter manifest and footprint validation**

Set manifest/CURRENT schema `2.0.0`, require metric `da3_self_exemplar_ground_footprint_union`, replace old `sam3_mask_entries` with the exact `sam3_mask` source object, and keep existing unrelated export/global-mapping provenance fields.

- [ ] **Step 7: Run exporter tests and commit**

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python -m pytest -q tests/test_web_viewer_export.py
git add -- src/web_viewer_export.py tests/test_web_viewer_export.py
git commit -m "feat: export viewer labels from matching mask cache"
```

---

### Task 6: Upgrade the TypeScript viewer bundle contract to v2

**Files:**
- Modify: `modules/viewer_web/src/contracts.ts`
- Modify: `modules/viewer_web/src/contracts.test.ts`
- Modify: `modules/viewer_web/src/bundle-loader.ts`
- Modify: `modules/viewer_web/src/bundle-loader.test.ts`
- Modify: `modules/viewer_web/src/presentation.ts`
- Modify: `modules/viewer_web/src/presentation.test.ts`

**Interfaces:**
- Consumes: Task 5 manifest and footprint JSON.
- Produces: `Sam3MaskSource` exact type, schema `2.0.0` validators, new metric validator, explicit v1 re-export error.

- [ ] **Step 1: Write schema/metric/source RED tests**

```typescript
it("accepts only the canonical processed mask bundle v2", () => {
  const manifest = validateManifest(validManifestV2());
  expect(manifest.schema_version).toBe("2.0.0");
  expect(manifest.source.sam3_mask).toEqual({
    schema: "sam3_self_exemplar_processed_mask_cache_v1",
    coordinate_space: "da3_processed_pixels",
    producer: "sku_matching",
  });
});


it("rejects v1 with rerun guidance", () => {
  expect(() => validateManifest({ ...validManifestV2(), schema_version: "1.0.0" }))
    .toThrow(/rerun matching, footprint, and viewer export/i);
});
```

Add footprint tests accepting only `da3_self_exemplar_ground_footprint_union` and exact-key rejection for wrong SAM3 source fields.

- [ ] **Step 2: Run focused RED**

```bash
npm --prefix modules/viewer_web test -- --run src/contracts.test.ts src/bundle-loader.test.ts src/presentation.test.ts
```

- [ ] **Step 3: Implement strict v2 types and validators**

```typescript
export interface Sam3MaskSource {
  readonly schema: "sam3_self_exemplar_processed_mask_cache_v1";
  readonly coordinate_space: "da3_processed_pixels";
  readonly producer: "sku_matching";
}
```

Replace the old mask-entry source, require exact keys, update CURRENT/manifest literals to `2.0.0`, update the footprint metric literal, and issue explicit rerun guidance for v1. Do not add a compatibility parser.

- [ ] **Step 4: Preserve unchanged viewer behavior**

Update fixtures only where contract values changed. Existing selection, magenta colors, Focus, ID sorting, no-geometry handling, and evidence presentation tests must remain behaviorally identical.

- [ ] **Step 5: Run all frontend tests and build, then commit**

```bash
npm --prefix modules/viewer_web test -- --run
npm --prefix modules/viewer_web run build
git add -- modules/viewer_web/src/contracts.ts modules/viewer_web/src/contracts.test.ts modules/viewer_web/src/bundle-loader.ts modules/viewer_web/src/bundle-loader.test.ts modules/viewer_web/src/presentation.ts modules/viewer_web/src/presentation.test.ts
git commit -m "feat: validate self-exemplar footprint bundle v2"
```

---

### Task 7: Document, verify, and benchmark the canonical workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/3d_core.md`
- Modify: `modules/viewer_web/README.md`
- Verify: `config.yaml` from Task 2 and all Task 1-6 files
- Create receipts only under the existing ignored `perf/` convention or `/tmp`; never stage runtime masks/bundles.

**Interfaces:**
- Consumes: completed v2 producer/consumers.
- Produces: current operational contract and evidence-bounded validation report.

- [ ] **Step 1: Update mandatory run-order documentation**

State that matching is the only SAM3 producer, footprint/export never run SAM3, full batch-all-refs matching must precede footprint, v1 is incompatible and untouched, the master gate defaults true, processed masks are losslessly packed, and the new metric/bundle schema are not comparable with old area outputs.

- [ ] **Step 2: Run the focused no-GPU suite**

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python -m pytest -q \
  tests/test_sam3_mask_cache.py \
  tests/test_matching_sam3_cache.py \
  tests/test_da3_footprint_stage.py \
  tests/test_footprint_evidence.py \
  tests/test_web_viewer_export.py \
  tests/test_main_pipeline.py
```

- [ ] **Step 3: Run full no-GPU regression and frontend verification**

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/3d-recognition-uv-cache uv run --active --no-project python -m pytest -q
npm --prefix modules/viewer_web test -- --run
npm --prefix modules/viewer_web run build
```

- [ ] **Step 4: Run cold/warm GPU matching equivalence**

Use the exact dataset/reference/seed/device receipt captured before Task 2. Compare direct baseline, v2 cold, and v2 warm masks, sampled points, `matching_summary.txt`, `correspondences.json`, assignments, and Recall/Precision/F1. Use byte comparison where required. Any mismatch is a failed gate, not an acceptable tolerance.

- [ ] **Step 5: Record the new footprint baseline and consumer GPU memory**

Run full matching first, then footprint, then viewer export. Record accepted/rejected, per-ID point counts, plane, OBBs, union, robustness, leave-one-out, and rejection reasons under the new metric. Record process/NVML memory before/after footprint/export and require no additional consumer allocation; do not call `empty_cache()` to manufacture the result.

- [ ] **Step 6: Verify performance semantics**

Show cold matching computes each frame once, warm matching performs zero SAM3 calls, footprint logs `load_self_exemplar_masks`, viewer export directly loads processed masks, payload size is consistent with packbits, and support-plane time is reported separately rather than attributed to masks.

- [ ] **Step 7: Review worktree and commit documentation only**

```bash
git diff --check
git status --short
git add -- README.md docs/3d_core.md modules/viewer_web/README.md
git commit -m "docs: document canonical matching mask workflow"
```

Do not stage `Output/`, `runtime/`, `perf` receipts unless already tracked and explicitly intended, v1/v2 cache payloads, checkpoints, frontend `dist`, or temporary files.

## Final acceptance checklist

- [ ] `sam3_use_self_exemplar` is absent except the removed-key error test/message.
- [ ] Self-exemplar is the sole mode; master gate defaults true.
- [ ] Matching is the sole cache producer and publishes complete processed frames once.
- [ ] v2 pack/unpack is byte-exact and never reads/deletes v1.
- [ ] Footprint and exporter contain no SAM3 inference fallback.
- [ ] Footprint metric/report and viewer bundle use the locked v2 values.
- [ ] Evidence robustness operates only on processed masks and cannot alter formal status.
- [ ] Viewer source-mask inverse sampling is gone; no protection mask is introduced.
- [ ] Direct/cold/warm matching masks, sampled points, assignments, and artifacts are equivalent.
- [ ] New footprint results are reported as a new metric baseline.
- [ ] Focused/full Python tests, frontend tests, and build pass.
- [ ] README and operational docs describe the mandatory run order and incompatibility.
