# SAM3 v2 Final Fix Wave

## Scope

- `main.py`: batch matching now aggregates each reference `StepResult` and every
  future exception. A failed batch returns `success: false` and a stable,
  reference-index-sorted `failed_references` list containing `reference_idx`,
  `image_index`, and `error`.
- `utils/sam3_mask_cache.py`: before NumPy loads the NPZ archive, the loader
  requires the only ZIP member to be `packed_masks.npy` and requires that
  member to use `ZIP_STORED`.
- Regression tests cover serial failure, parallel false-result plus exception
  handling while an unaffected reference completes, complete-pipeline summary
  preservation, compressed payload rejection, and quarantine/recompute.

## TDD Record

### RED: matching scheduler

```text
UV_CACHE_DIR=/tmp/codex-uv-cache uv run pytest tests/test_main_pipeline.py -k \
  'batch_matching_serial_failure or parallel_batch_matching_collects or complete_pipeline_preserves'
```

Result: 2 failed, 1 passed. Both scheduler tests observed `result["success"] is
True` despite one returned failed `StepResult` and one failed future.

### GREEN: matching scheduler

The same command passed: 3 passed, 5 deselected.

### RED: compressed payload

```text
UV_CACHE_DIR=/tmp/codex-uv-cache uv run pytest tests/test_sam3_mask_cache.py \
  -k 'compressed_payload'
```

Result: 2 failed. A real `np.savez_compressed` replacement was accepted as a
cache hit and did not recompute.

### GREEN: compressed payload

The same command passed: 2 passed, 36 deselected.

## Final Verification

```text
UV_CACHE_DIR=/tmp/codex-uv-cache uv run pytest \
  tests/test_main_pipeline.py tests/test_sam3_mask_cache.py
```

Result: 46 passed in 24.67s.

The final patch also passed `git diff --check` for the precise allowlist.
