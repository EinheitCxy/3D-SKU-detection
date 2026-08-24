# Canonical processed-space self-exemplar SAM3 cache design

## Status and locked decision

Rick supplied and approved this architecture for implementation on 2026-08-24. The remaining gate is resolved as follows:

- Keep `enable_sam3_mask_sampling` as the diagnostic master gate and default it to `true`.
- Remove `sam3_use_self_exemplar` from dataclasses, YAML, loaders, docs, and tests.
- If a loaded YAML still contains `sam3_use_self_exemplar`, fail with: `sam3_use_self_exemplar was removed; self-exemplar is now the only SAM3 mode`.
- When the master gate is true, self-exemplar is the only SAM3 matching mode and matching publishes processed-space cache v2.
- When the master gate is false, matching uses its existing bbox sampling path and publishes no cache; footprint and viewer export then fail closed because canonical masks are missing.

The v2 cache adds no content hash, signature, encryption, checkpoint digest, payload digest, source-image digest, code fingerprint, or runtime fingerprint. Existing unrelated DA3/footprint/viewer artifact provenance checks remain unchanged. Cache identity comes from exact manifest fields and the stable `image_id` entry path.

This deliberately relies on the pipeline invariant that source images, detections, and the selected checkpoint path are immutable during one dataset run. Replacing file content in place without changing the declared image size, bbox/affine contract, or cache root is outside the v2 invalidation contract; the operator must start from a fresh v2 root in that case. This is the explicit efficiency trade-off for omitting content hashing.

The final classifier integration remains in scope. Completed classifier contracts and enriched bbox detections are preserved. Its pending global aggregation and UI work resumes only after this cache/metric/bundle v2 contract lands.

## Objective

Make SKU matching the only SAM3 producer. It computes one complete processed-space self-exemplar mask bundle per detection frame. Matching, formal footprint, shadow evidence, and viewer export then consume the same losslessly packed masks in DA3 processed-pixel coordinates.

```text
DA3 reconstruction
        |
DA3 world grid + source_to_processed affine
        |
SKU matching first access to one frame
        |
        +-- SAM3 self-exemplar inference
        +-- processed-space bool masks
        +-- atomic sam3_mask_cache/v2 frame entry
                      |
                      +-- matching samples 50 points
                      +-- footprint extracts metric points
                      +-- viewer exporter assigns point labels
```

Footprint and viewer export contain no SAM3 model import, builder, checkpoint load, CUDA inference, or cache-miss producer.

## Non-negotiable boundaries

- `Output/<dataset>/sam3_mask_cache/v1` is never read, migrated, copied, used as fallback, or automatically deleted.
- v2 masks are `(object_count, processed_height, processed_width)` boolean masks before packing; source-resolution masks are invalid payloads.
- No bbox-mask fallback or partial-frame hit is permitted.
- Cache serialization is lossless and matching object identity is keyed by `object_id`, not payload position alone.
- Matching retains detection order, batch grouping for currently matched objects, threshold `0.5`, image size `1008`, `max_batch_size=32`, `max_dets_per_query=1`, bbox clipping, 50-point sampling, and assignment semantics.
- Ordinary viewer point filtering runs before SAM3 labels are propagated. Masks never become a protection mask and never exempt points from noise, ground, cluster, voxel, or sky filtering.
- Old viewer bundle schema `1.0.0` and old footprint metric are rejected without compatibility branches.

## Cache layout and schema

```text
Output/<dataset>/sam3_mask_cache/v2/
├── entries/<image_id>/
│   ├── manifest.json
│   └── masks.npz
├── locks/<image_id>.lock
└── corrupt/<image_id>.<unix_time_ns>/
```

The internal cache schema is exactly `sam3_self_exemplar_processed_mask_cache_v1`. A frame manifest has exact keys:

```json
{
  "schema": "sam3_self_exemplar_processed_mask_cache_v1",
  "image_id": 1,
  "source_size_wh": [3024, 4032],
  "processed_shape_hw": [504, 378],
  "source_to_processed_affine": [[0.125, 0.0, 0.0], [0.0, 0.125, 0.0]],
  "detections": [
    {
      "object_id": 0,
      "source_bbox_xyxy": [100.0, 200.0, 300.0, 500.0],
      "processed_bbox_xyxy": [12.5, 25.0, 37.5, 62.5],
      "mask_index": 0
    }
  ],
  "inference": {
    "api": "self_exemplar",
    "threshold": 0.5,
    "image_size": 1008,
    "max_batch_size": 32,
    "max_dets_per_query": 1,
    "clip_to_bbox": true
  },
  "payload": {
    "path": "masks.npz",
    "array": "packed_masks",
    "dtype": "uint8",
    "bitorder": "little",
    "mask_count": 1,
    "flat_mask_size": 190512,
    "packed_width": 23814
  },
  "complete": true
}
```

No additional or missing manifest keys are accepted. All numeric values must be finite; IDs are safe integers; bboxes are ordered and clipped to the declared coordinate-space bounds; affine shape is `(2, 3)`; detections have unique object IDs and unique mask indices covering `0..mask_count-1`.

The payload uses uncompressed NumPy storage after bit packing:

```python
packed_masks = np.packbits(
    masks.reshape(mask_count, -1),
    axis=1,
    bitorder="little",
)
np.savez(payload_path, packed_masks=packed_masks)
```

Load one frame at a time. Unpack with the declared `flat_mask_size`, reshape to the declared processed shape, and return independent two-dimensional bool arrays indexed by object ID. Tail bits beyond `flat_mask_size` must be zero; non-`uint8`, wrong packed width, wrong mask count, or wrong unpacked shape is invalid.

## Canonical cache API

`utils/sam3_mask_cache.py` keeps its module path but replaces its old source-space contract:

```python
@dataclass(frozen=True)
class ProcessedDetectionPrompt:
    object_id: int
    source_bbox_xyxy: tuple[float, float, float, float]
    processed_bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameMaskCacheRequest:
    cache_root: Path
    image_id: int
    image_path: Path
    source_size_wh: tuple[int, int]
    processed_shape_hw: tuple[int, int]
    source_to_processed_affine: np.ndarray
    detections: tuple[ProcessedDetectionPrompt, ...]
    inference_contract: Mapping[str, object]


@dataclass(frozen=True)
class FrameMaskCacheResult:
    masks_by_object_id: Mapping[int, np.ndarray]
    cache_event: Literal["hit", "miss"]
    schema: str


def load_complete_frame_masks(request: FrameMaskCacheRequest) -> FrameMaskCacheResult:
    ...


def load_or_compute_frame_masks(
    request: FrameMaskCacheRequest,
    compute_masks: Callable[[], Mapping[int, np.ndarray]],
) -> FrameMaskCacheResult:
    ...
```

`load_complete_frame_masks` is read-only and raises a typed cache error for missing, malformed, mismatched, partial, or corrupt entries. `load_or_compute_frame_masks` first tries a shared-lock load, then takes the per-image exclusive lock, double-checks, quarantines any invalid existing entry, computes one complete frame, validates it, and atomically renames a temporary sibling into `entries/<image_id>`.

There is no lookup under v1. A cache hit never invokes `compute_masks`. A miss must produce every requested object exactly once. Interrupted writes leave no readable entry. Concurrent references to the same image produce one published entry and one effective compute.

## Matching as the sole producer

Replace the optional-mode branch with:

```python
def get_self_exemplar_masks_for_reference(
    config: SKUMatchingConfig,
    *,
    image_path: Path,
    image_id: int,
    frame_detections: Sequence[dict[str, object]],
    matching_object_ids: Sequence[int],
    transform: ImageTransformBase,
) -> dict[int, np.ndarray]:
    ...
```

The function:

1. Validates the numeric image ID, source size, DA3 processed size, and explicit `source_to_processed_affine` from the transform contract.
2. Builds the complete frame request in original detection order.
3. On hit, returns the processed bool masks.
4. On miss, follows the exact current final-space self-exemplar preprocessing: RGB conversion, LANCZOS source-to-processed resize, bbox corner mapping, SAM3 square image size 1008, threshold 0.5, one detection per query, max batch 32, postprocess back to processed size, and bbox clip.
5. Computes currently matched objects first using their unchanged order and batch partition, then computes remaining detection objects, publishes the complete frame, and returns the same validated masks.
6. Matching consumes only `matching_object_ids` and passes their masks into the existing `sample_3d_points_from_mask(..., mask_space="final")` path.

Add `sam3_mask_cache_root: str` to `SKUMatchingConfig`; `main.py` supplies `<save_root>/<dataset>/sam3_mask_cache/v2` explicitly. No consumer infers the root from `output_3dmapping_da3/<ref>`.

Cold compute and warm hit must not change downstream RNG state. Tests record Python, NumPy, torch CPU, and selected CUDA RNG states before and after the cache boundary. If live SAM3 inference consumes RNG, the implementation must isolate producer RNG so cold/hit sampling begins from the same state, then prove old direct-mask and new cold/warm assignment equivalence on the locked GPU dataset. No unverified equivalence claim is allowed.

## Footprint as a read-only consumer

Remove from `src/da3_footprint_stage.py`:

- `_PREDICT_INST_CONTRACT`, `_SAM3_CHECKPOINT`, and `_SAM3_DEVICE`.
- `_compute_verified_sam3_masks()`.
- `sam3_masks_from_bboxes_predict_inst`, SAM3 builder, checkpoint loading, checkpoint/runtime/code fingerprints, and cache-miss inference.
- Source-mask warp and source-mask cache-event logic.

For every frame, construct the expected v2 request from DA3 affine/shape and complete detections, then call `load_complete_frame_masks`. Any missing/mismatched frame/object/bbox/shape/affine/schema rejects the complete formal run with:

`canonical self-exemplar mask cache is incomplete; run SKU matching first`

Each processed mask directly indexes the DA3 grid:

```python
valid_grid = frame_valid_world_points & frame_confidence_gate
object_points = world_points[frame_index][valid_grid & processed_mask]
```

Compute `valid_grid` once per frame. Background support-plane masking also stays in processed space and ORs 5x5 dilated processed masks. The performance stage becomes `load_self_exemplar_masks`.

The formal metric becomes `da3_self_exemplar_ground_footprint_union`. The report uses schema `2.0.0` and adds:

```json
{
  "mask_contract": {
    "source": "sam3_self_exemplar",
    "coordinate_space": "da3_processed_pixels",
    "cache_schema": "sam3_self_exemplar_processed_mask_cache_v1"
  }
}
```

The formal all-ID fail-closed and `value_m2: null` rejection semantics remain unchanged.

## Processed-space shadow evidence

`EvidenceObservation` becomes:

```python
@dataclass(frozen=True)
class EvidenceObservation:
    global_id: str
    image_id: int
    object_id: int
    processed_mask: np.ndarray
    valid_mask: np.ndarray
```

Remove source masks, affine warping, and original-warp equality checks. Mask robustness operates on processed masks with a 3x3 kernel and one iteration for `original`, `eroded`, and `dilated` variants. Report coordinate space `da3_processed_pixels`.

This is a new evidence schema and physical scale; it is not numerically comparable to the old source-pixel morphology. Camera reprojection, depth residual, and leave-one-out continue on the same processed grid. Evidence failures remain shadow-only and cannot change the frozen formal status, value, or geometry.

## Viewer exporter as a read-only consumer

Delete source-mask lookup, source image digest/index logic, inverse affine sampling, source coordinates, source bbox coverage sampling, and full source-mask validation from `src/web_viewer_export.py`.

For each image ID, load one complete v2 processed frame and assign labels directly:

```python
for label, object_id in instance_order:
    covered = masks_by_object_id[object_id]
    frame_grid[covered & (frame_grid < 0)] = label
```

Detection/global-mapping bbox identity, object uniqueness, processed shape, and affine remain fail-closed. Missing v2 cache fails export. Zero-range IDs remain valid no-geometry entries.

Point-cloud filters operate before labels are propagated through keep masks, voxel selection, and sky cuts. Voxel representative selection continues to use confidence only; labels never protect or prioritize points.

The manifest SAM3 source becomes exactly:

```json
{
  "schema": "sam3_self_exemplar_processed_mask_cache_v1",
  "coordinate_space": "da3_processed_pixels",
  "producer": "sku_matching"
}
```

## Viewer bundle v2

`CURRENT.schema_version` and `manifest.schema_version` become `2.0.0`. `footprints.metric` becomes `da3_self_exemplar_ground_footprint_union`. TypeScript adds:

```typescript
interface Sam3MaskSource {
  readonly schema: "sam3_self_exemplar_processed_mask_cache_v1";
  readonly coordinate_space: "da3_processed_pixels";
  readonly producer: "sku_matching";
}
```

Schema `1.0.0` fails with an instruction to rerun matching, footprint, and viewer export. Current selection, magenta highlight, Focus, Global-ID list, point filtering, and no-geometry presentation remain unchanged.

The classifier extension later adds classification to `objects.json` under the same bundle schema `2.0.0`. Classification never changes SAM3 masks, point labels, filtering, or point ranges.

## Performance and GPU contract

- Matching cold computes each frame once; matching warm never invokes SAM3.
- Footprint and viewer export import no SAM3 producer and allocate no additional CUDA memory.
- Consumers unpack one frame at a time and release it before loading the next.
- Packbits serialization is lossless; no compression pass follows bit packing.
- Formal footprint logging contains `load_self_exemplar_masks`, not `sam3_source_masks`.
- GPU receipts record process memory before/after matching, footprint, and export. Consumer tests monkeypatch all SAM3 builders/producers to raise and must still pass with valid v2 entries.
- No `empty_cache()` call may be used to disguise consumer allocations.

## Validation gates

### No-GPU

- Pack/unpack byte-exact, including non-byte-aligned tail bits.
- Detection order changes never misbind object IDs.
- Exact manifest validation for bbox, shape, affine, inference, object completeness, and schema.
- Cache hits never compute; v1 is never read; partial entries never hit; interrupted writes are unreadable; parallel writers compute once.
- Config removes `sam3_use_self_exemplar`, errors on the legacy YAML key, preserves the default-true master gate, and passes an explicit v2 root.
- Matching direct/cold/warm masks and sampled points are identical in synthetic deterministic tests.
- Footprint and viewer succeed from v2 while every SAM3 producer/import stub raises.
- Missing/mismatched cache rejects footprint/export with the canonical message.
- Processed masks align one-to-one with world grid and object IDs.
- Processed-mask robustness cannot alter formal snapshot; camera evidence remains numerically unchanged.
- Viewer labeling handles overlaps by first instance order, preserves zero-range IDs, and never changes filtering behavior.
- Frontend rejects schema 1.0.0 and the old metric.

### GPU matching equivalence

For one locked dataset and seed, capture the current direct self-exemplar baseline before replacing the path, then compare v2 cold and warm:

- Mask count and per-object mask pixels byte-exact.
- Sampled points byte-exact.
- Correspondence JSON and matching summary byte-equivalent.
- Matching assignments identical.
- Recall/Precision/F1 do not decline.

If RNG state or assignments differ, implementation is not accepted until the cause is resolved. Cache serialization losslessness alone is insufficient.

### New footprint baseline

Because the source changes from predict-inst to self-exemplar, old area values are not equivalence targets. Record the new accepted/rejected outcome, observation point counts, support plane, OBB dimensions/areas, union area, erosion/dilation interval, leave-one-out evidence, and per-object rejection reasons. Report it only as the baseline for `da3_self_exemplar_ground_footprint_union`.

### Final acceptance

- No `sam3_use_self_exemplar` remains in code, YAML, docs, or tests except the explicit removed-key error test/message.
- Self-exemplar is the only SAM3 mode and the master gate defaults true.
- Matching is the sole v2 producer.
- Footprint and viewer export have no SAM3 inference fallback.
- v2 uses processed-space lossless bit packing and never reads v1.
- Footprint metric/report and viewer bundle use the locked v2 contracts.
- Matching equivalence and the new footprint baseline are evidence-backed rather than inferred.
- README, `docs/3d_core.md`, viewer README, and `config.yaml` state the mandatory run order.
