# Texture candidate 1: frozen primary sources and common-view residual patches

Status: implementation and focused synthetic checks complete. **Full-scene coverage, runtime and visual benefit remain unproven.** No full-scene run, model/GPU work, export, dependency installation, production edit, old-asset edit or Git action was performed by this worker.

## Implementation

Entry point: [`texture.py:74`](texture.py#L74), `choose_sources(vertices, faces, data, preferred_sources=...) -> (int32 sources, stats)`. Supply `np.repeat(fused['frame'], 8)` for the fixed fused-disc mesh. `-1` explicitly preserves uncovered faces.

1. Evaluate the existing [`face_view_candidates`](../roi_fusion/texture.py#L60) in batches of 32,768 faces. The original three-vertex-plus-centroid mask/depth gate, `max(2*spacing, 0.005*Z)` tolerance, and front cosine `>= 0.15` are unchanged.
2. Freeze every valid original/primary frame. No source-quality score can replace it.
3. Group strictly observable residual faces with the original [`_patches`](../roi_fusion/texture.py#L26): 24 nearest neighbors, `2.5*spacing` neighbor radius, `12*spacing` seed radius, and 25-degree normal agreement.
4. Accept only patches of at least four faces with one source that passes visibility for **every** face. Assign one source to the whole patch. Conflicting per-face candidates and smaller islands remain gray.
5. Prefer candidates matching the area-weighted nearest compatible frozen boundary source (at most four neighbors inside `2.5*spacing`); break ties with the original summed quality, then frame index. Newly filled patches never anchor subsequent patches, so there is no iterative propagation.

Stats include chosen-source validity, valid-primary area retained, original-source switches, recovered area, remaining area without any strict view, observable-but-deferred area, patch rejection reasons, source transition counts, boundary-source agreement, batch size, candidate-array bytes, and wall times. `restored_original_source_area_m2` means the retained valid-primary area; it is **not** a measured changed-source-area comparison against the old spatial-patch labels. That comparison needs coordinator-provided baseline labels.

## Runtime and memory limits

Projection work is `O(views * faces)` with fixed-size temporaries; no global four-sample projection array is kept. Candidate arrays require `(5*views + 56)*faces` bytes: **331,995,720 bytes (about 316.6 MiB)** for 1,651,720 faces and 29 views. Visibility is boolean. Quality storage is float32 with float64 patch sums; a near-tie can change from the old float64 ranking, but candidate visibility cannot change.

Additional work is the old spatial patch construction on `R` eligible residual faces and a nearest-boundary query with at most four candidates per residual face. `_patches` uses up to 24 neighbors and may temporarily allocate approximately `384*R` bytes for query distances and indices, plus tree/point/patch storage. At the all-residual extreme this query storage alone is about 604.9 MiB. These figures exclude caller-owned mesh/depth arrays and do not claim a measured peak RSS or scene runtime. The coordinator's 15-minute command limit still applies.

## Focused validation

Command, repository root:

```bash
UV_CACHE_DIR=/tmp/codex-surfel-coverage-uv OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run --no-sync python -m pytest -q perf/surfel_coverage/test_texture.py
```

Receipt: **8 passed in 0.61s**, exit code 0. Tests cover:

- Valid primary retained despite a higher-quality alternative.
- Whole residual patch recovery, frozen boundary anchoring, and area/source-switch accounting.
- Individually visible but mutually conflicting sources left unassigned rather than creating a per-face mosaic.
- Alternative rejection for depth occlusion, an unsupported vertex despite a supported centroid, and backfaces.
- Multiple projection batches and deliberate rejection of an isolated one-face island.
- Invalid primary frame index rejected before selection.

Tests: [`test_texture.py`](test_texture.py). No unrelated regression suite was run.

## Tradeoffs and acceptance boundary

The prior [negative texture result](../../runtime/roi-fusion-video3-gid3-v2/REPORT.md#L64) reduced gray area while introducing repeated or shifted lettering. Freezing primary labels prevents changes on those already-valid faces; whole-patch common visibility limits additional source fragmentation. Neither property proves photometric alignment at the new boundary, nor that the primary labels themselves are visually superior to the old spatial-patch baseline.

Four-face minimum and complete common-view agreement deliberately sacrifice observable coverage. Spatial/normal proximity is the old patch connectivity proxy and is not exact surface adjacency; nearby overlapping surfels can still influence anchor preference. No exposure correction, photo registration or seam optimization is implemented. Strict depth rejection remains an observation-consistency check, not geometric ground truth.

The coordinator must evaluate this candidate on exactly the old fused geometry, compare with its old spatial-patch texture, raycast the fixed input views, and inspect synchronized front/side/close-up/source-color views. A lower untextured-area fraction alone does not establish success.
