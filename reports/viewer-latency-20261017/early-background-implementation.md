# Early background candidate implementation — 2026-10-17

Status: source implementation and focused fixture tests complete, included in the **66 passing geometry/export tests**. Full-scene latency, RSS and rendered quality comparisons remain **PENDING**. No formal experiment, GPU execution or publication performed.

## Opt-in API and execution

`ViewerExportTuning(early_background_candidates=500_000)` enables the candidate approximation; the default `0` preserves the existing path. Values must be nonnegative Python integers (booleans are rejected). The subsequent CLI integration exposes `--early-background-candidates` in `scripts/export_surfel_viewer.py`. Existing Surfel-only tuning controls still require Surfel export; this candidate option alone also supports plain point export.

After labels, the candidate domain contains **every valid product observation** plus at most the requested number of background observations. Background selection first uses the existing voxel rule (first highest-confidence winner, first-seen voxel order), then the existing deterministic `EdgeAdaptiveSampler` over native frame/tile coordinates with semantic/depth boundary weights. Plain-point early export uses this identical candidate selection. It retains its existing final point-budget sampler.

Geometric filtering runs once on the sorted candidate domain, with candidate-aligned product protection. Its configuration and stage order (including protection before each stage's 20% guard) remain unchanged. The returned candidate mask maps back into the original valid-domain mask before quality, spill, ownership and publication selection. All valid product observations reach filtering; the existing 2M product publication budget remains unchanged.

`prepare_web_viewer_geometry` automatically defers NN/SOR when this tuning is enabled, even if `prepare_filter=True` was requested. Full native tangents, depth, textures, valid indices and semantic regions remain available. The early path never passes full-domain `PreparedSceneFilter` into candidate filtering. It does not perform full-domain NN/SOR first.

Plain-point early export retains its original finite/nonzero/confidence valid domain. It explicitly rejects any valid observation with nonpositive/nonfinite camera depth, because `EdgeAdaptiveSampler` requires positive depth. It never silently removes those product observations or switches to another sampler. The ordinary plain-point default has no new camera-depth restriction. Surfel export retains its existing positive-depth validity rule.

## Support, counts and underfill

Candidate sampling uses the **full valid background** as its dense support. Final early Surfel background scales also use that same full valid background, including observations removed by geometry filtering: this experimental density convention intentionally differs from baseline. Baseline continues using **full geometrically filtered background** as its support. Native derivatives, boundary caps, scale limits, texture longest-edge limit and Surfel maximum diameter are unchanged.

The private sampled dictionary includes `early_background_counts` only when enabled. Logs record full-valid background, voxel winners, selected candidates, background immediately after geometry filtering, background after other selection filters, final background and publication budget. Geometry-domain counts are logged before subsequent selection, including scenes that fail because filtering leaves no points. The public bundle schema is unchanged.

No refill or fallback exists. A 500K-candidate selection can yield fewer than 500K final background points after filtering; zero/small background pools remain zero/small. Coverage loss must be judged by the separately approved rendered comparison and may reject this option as a default.

## Validation

AST parsing passed for the three edited Python files:

```sh
UV_CACHE_DIR=/tmp/uv-viewer-early-background uv run --no-project --python /usr/bin/python3.11 python -c 'import ast; from pathlib import Path; paths = ["src/web_viewer_export.py", "src/viewer_export_tuning.py", "test/test_web_viewer_export.py"]; [ast.parse(Path(p).read_text(), filename=p) for p in paths]; print("AST parse passed: 3 files")'
```

Focused tests were added to the existing `test/test_web_viewer_export.py`: integer validation; implicit/default-zero equivalence; automatic preparation deferral; all products entering the reduced filter domain; background candidate cap; post-filter underfill without refill; empty/small pools; original source/position/color/label alignment; full native U/V/depth alignment; full-valid background support for both sampling stages; and explicit plain-point camera-depth rejection. `test/test_viewer_export_tuning.py` does not exist, so no redundant test file was created.

The independent Python 3.11 engineering environment was subsequently restored offline from existing locked dependencies, and these tests passed in the focused 66-test geometry/export run: [receipt](geometry-tests.json), [raw log](geometry-tests.log). This supports fixture-level behavior, not full-scene speedup or quality acceptance. Next: compare the isolated candidate against its frozen baseline under the separately approved CPU budget and review real rendered coverage.
