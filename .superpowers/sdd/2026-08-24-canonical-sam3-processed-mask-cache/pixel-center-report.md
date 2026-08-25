# DA3 pixel-center bbox unification report

Date: 2026-08-25. Repository: `/home/xingyu/3D_Recognization`.

## Implemented contract

- DA3 now uses `DA3ImageTransform`, whose source-to-processed mapping is the
  cache convention: `x'=sx*x+(sx-1)/2`, `y'=sy*y+(sy-1)/2`.
- SAM3 matching builds its complete v2 request and every processed prompt from
  that affine. It no longer creates a DA3 scale-only manifest.
- `map_source_bbox_to_processed` is the single bbox-affine helper used by the
  producer, footprint, and exporter.
- The exporter constructs its exact complete-frame request in original
  `object_id` order, not global-ID iteration order. It remains fail-closed for
  missing, reordered, or otherwise mismatched entries.
- No consumer fallback, v1 migration, compatibility branch, or manifest
  relaxation was added. Existing scale-only v2 entries mismatch exactly and
  are recomputed by matching.
- `config.yaml` now declares
  `da3_self_exemplar_ground_footprint_union`; README/core documentation state
  that old scale-only assignments, global IDs, footprints, and bundles are not
  comparable to this baseline.

## TDD and focused validation

RED tests were added first and observed failing:

1. `test_da3_processed_request_uses_pixel_center_affine_exactly` initially
   failed because `DA3ImageTransform` did not exist. It locks source
   `4032x3024`, processed `504x378`, affine translation `-0.4375`, and object
   0 bbox `[1143,2198,1322,2612] ->
   [142.4375,274.3125,164.8125,326.0625]`.
2. `test_processed_mask_request_uses_original_object_order_not_global_id_order`
   initially failed with the expected exact-manifest error. It locks the
   exporter request ordering needed to consume the producer's complete frame.

After the minimal implementations and formatting:

```text
202 passed in 54.23s
```

Command:

```bash
VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv \
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --active --no-project python -m pytest -q \
  tests/test_sam3_mask_cache.py tests/test_matching_sam3_cache.py \
  tests/test_da3_footprint_stage.py tests/test_footprint_evidence.py \
  tests/test_web_viewer_export.py tests/test_main_pipeline.py
```

`git diff --check` also passed.

## GPU2 fd2 direct/cold/warm

All runs used `CUDA_VISIBLE_DEVICES=2`, existing
`Output/floor_display2/da3_cache/predictions.npz` through a temporary symlink,
and seed 42. No DA3 reconstruction was run.

- Direct: 75 masks in `[32,32,11]`, 53 matches.
- Cold: 75 masks in `[32,32,11]`, 53 matches.
- Warm: 53 matches in 5.03s and no SAM3 build/memory/batch/segmentation log
  lines, so it made zero SAM3 calls.

The direct, separately retained cold, and warm artifacts are byte-identical:

| artifact | SHA-256 |
| --- | --- |
| `matching_summary.txt` | `01a5570d69ce3d294cd740fb1ab2adba82561b330500b5c9c2df8844035dbaef` |
| `correspondences.json` | `77e78772a6fa33670bff1d783ae00550288803564b87b7389c37bf3fc10881b4` |
| v2 `manifest.json` | `9f6e6ac96c9ad376a57e842d49cd41b09d9983ed8e3b9c76c84ff157956bb98f` |
| packed `masks.npz` | `1ed334852d585f8852894c7273d22846ee3d6816461de390c58a222b47ec22fe` |

The old locked scale-only fd2 direct result had 52 matches. The approved
pixel-center result has 53; this is an intentional non-comparable assignment
change, not an equivalence regression.

## Full fd2 pipeline and consumers

Full batch-all-refs matching ran under
`/tmp/da3-pixel-center-20260825/full`, reusing the DA3 cache and explicitly
skipping reconstruction. Match counts by reference are `53/43/42/55/1`, total
`194`; dedup produced `243` global IDs.

Footprint loaded all five canonical v2 frames as `hit` with the exact
pixel-center manifest; no affine mismatch occurred. Its formal result is
`rejected`, `value_m2: null`, solely because the existing support-plane table
compatibility gates found no acceptable plane. Therefore there is no accepted
new footprint area baseline to report.

Viewer export then succeeded from the same strict cache contract:

```text
bundle: /tmp/da3-pixel-center-20260825/viewer/runs/7d35a058b36e4590b4b59fb4ce1b2cae/
point_count: 459962
thumbnails: 436
footprint_status: rejected
```

The rejected formal footprint is represented truthfully in the bundle; it was
not replaced with a partial area.

## Existing-cache fd6/fd12 A/B

Both datasets already had DA3 caches and were run with batch-all-refs matching
plus the existing accuracy evaluator. No reconstruction was attempted.

| dataset | baseline matches | pixel-center matches | baseline recall | pixel-center recall | baseline precision | pixel-center precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fd6 | 188 | 188 | 174/198 (87.88%) | 174/198 (87.88%) | 172/179 (96.09%) | 172/178 (96.63%) |
| fd12 | 436 | 437 | 267/359 (74.37%) | 270/359 (75.21%) | 267/321 (83.18%) | 270/323 (83.59%) |

These are observational A/B results under the approved changed matching
semantics, not a causal claim about accuracy.

## Retained evidence

- Direct/cold/warm: `/tmp/da3-pixel-center-20260825/{direct,cold,equivalence}`.
- fd2 full/footprint/export: `/tmp/da3-pixel-center-20260825/full` and
  `/tmp/da3-pixel-center-20260825/viewer`.
- fd6/fd12 A/B: `/tmp/da3-pixel-center-20260825/ab/{fd6,fd12}`.

The only remaining concern is the independent formal support-plane rejection
on fd2. It is not a pixel-center, SAM3 cache, or viewer-contract failure.

## Round 2 reviewer corrections

Round 2 removes the producer's final scale-only escape hatch. A DA3 transform
is provisional until matching binds it by image ID to the cache's exact
`source_to_processed_affine`, `source_image_sizes`, and `(height,width)` grid;
missing or mismatched geometry now fails closed. The map helper also clips
every processed bbox to that grid and guarantees a one-pixel extent, while the
cache validator rejects raw out-of-grid requests. Footprint and exporter use
the same map-plus-clip helper.

New RED tests cover the true fd2 coordinate direction (`source_size_wh`
`(3024,4032)`, `processed_shape_hw` `(504,378)`), unbound transforms, an
explicit non-14-aligned cache geometry, and raw out-of-grid prompt rejection.
Focused no-GPU results after these changes are recorded in the final command
receipt. GPU2 reran fd2 direct/cold/warm: all produced 53 matches; direct and
warm masks, manifests, summary, and JSON are byte-equal. The changed clipped
manifest SHA-256 is `39b41ee1b3fcff14c148f5457f921ec7beb8bc2323df9ebba6e4d81056cd6edb`.

The round-2 fd2 batch-all-refs pipeline again produced 194 matches and 243
global IDs without reconstruction. Footprint had five exact v2 cache hits and
no affine/bbox mismatch, then rejected only on the existing support-plane
compatibility gate. Viewer export succeeded with 456455 points and 436
thumbnails at `/tmp/da3-pixel-center-r2/viewer/runs/30d9db8edeff47a19ea152ed8966e465/`.
Because the core fd2 matching count did not change, fd6/fd12 were not rerun.
