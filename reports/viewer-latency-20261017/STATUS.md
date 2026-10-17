# Viewer latency implementation status — 2026-10-17

The coordinator used medium leaf agents with separate source ownership and independent reviews. Source changes and focused engineering tests are complete for the items below. Rick subsequently authorized the bounded atlas 1/2/4-thread test: both complete comparisons passed, with about 37% less complete-export wall time against their own serial controls and identical output. See [execution](atlas-execution.md), [independent review](atlas-independent-review.md) and [scope/limits](atlas-test-authorization.md). This is a fixed-scene CPU exporter test with common page-capture instrumentation. Fixed-camera geometry render comparison, GPU execution, publication and deployment remain unperformed. The 267.565 s historical full-pipeline profile remains unchanged.

Git records contain the reviewable reports, runner and compact JSON receipts/results. Raw `.log` files, generated GLB files and the task-local Python environment remain local and are not versioned; links to those raw logs refer to the original execution workspace.

## Implemented and validated

| Direction | Implementation / default | Verified so far |
| --- | --- | --- |
| Auxiliary pictures | `quiet_outputs=True`; `--no-quiet_outputs` restores optional output. CLI/YAML precedence and interactive/pipeline/concise propagation are covered. Required summaries, source frames, masks, Viewer textures and thumbnails remain. | 89 output/main/identity tests passed. |
| Atlas encoding | `export_scene_glb(..., encoder_workers=1/2/4)`; default 1. Parallel FIFO holds at most 2/4 pages, single ordered writer, immutable page inputs, cancel/drain on failure. | 34 GLB tests passed. Two full video3 sequences also passed: 8 complete exports, each 31 pages and 121,513,344-byte identical GLB within its comparison, with all captured arrays and decoded assets equal. Both 2 and 4 workers saved about 37% against their own serial controls. |
| NN/SOR preparation | Full-domain raw median NN and SOR mask prepare alongside existing geometry/texture work. Labels still control protection and the 20% guard at finalization; DBSCAN/ground follow. | Included in 66 geometry/export tests; no-recompute, raw/direct equality, protection and domain/config/order checks covered. Real matching overlap gain unmeasured. |
| DBSCAN-off | Existing `keep_main_clusters=False`, now exposed as `--no-dbscan`. Default remains enabled. | Configuration path exists; actual video3 quality/time comparison unexecuted. |
| Proxy DBSCAN | Explicit `cluster_proxy_voxel_scale`, CLI `--cluster-proxy-voxel-scale`. Default 0. Only clustering uses representatives; labels map to original members before size/protection/guard. | Member mapping and original-member cluster-size tests passed. Proxy neighbor counts differ; visual quality unverified. |
| Early background candidates | `ViewerExportTuning(early_background_candidates=...)`, CLI `--early-background-candidates`. Default 0. Keep all valid product candidates; sample background after labels; retain dense grid; no silent refill. | Candidate limits, product retention, source/geometry alignment, preparation deferral and underfilled-budget tests passed. New support/coverage semantics require real-scene review. |

The three test commands ran separate focused files once, with no failures or repeat full runs: **66 + 89 + 34 = 189 passed**. These are engineering correctness checks, not latency measurements or a full regression suite.

- [Geometry/export command and resources](geometry-tests.json), [raw log](geometry-tests.log): 66 passed in 11.08 s; outer wall 15.696 s.
- [Output/main command and resources](outputs-tests.json), [raw log](outputs-tests.log): 89 passed in 4.96 s; outer wall 7.407 s.
- [Atlas command and resources](atlas-tests.json), [raw log](atlas-tests.log): 34 passed in 4.43 s; outer wall 6.997 s.
- [Environment restoration](environment.md): task-local Python 3.11, existing frozen dependencies restored offline via hardlinks; root `.venv` and lockfile unchanged. Tests hide GPU devices.

Static review caught and fixed interactive quiet-option propagation, a nullable inference YAML section, and invalid `protect_mask` validation running after expensive preparation. Source syntax and changed shell syntax passed. Existing dirty work was retained; the complete working-tree diff includes earlier work and is not all attributable to this task.

Final lightweight checks passed: the Surfel exporter `--help` imports successfully in the restored environment and exposes all three experimental flags; 74 local links and Markdown fences in ten relevant documents resolve; scoped `git diff --check` and the changed shell's `bash -n` pass. These checks did not run an export or benchmark.

## Complete atlas comparison

Both sequences used the frozen 29-frame / 2,402,589-point video3 generation, the same four logical CPUs, unchanged image/geometry/LOD parameters, and B1/C1/C2/B2 order. Times include the same full-page capture instrumentation; comparisons run outside the primary timer.

| Candidate | Serial B1 / B2 (s) | Candidate C1 / C2 (s) | Paired savings (s) | Verdict |
| --- | ---: | ---: | ---: | --- |
| 2 workers | 112.818 / 111.459 | 70.663 / 70.555 | 42.155 / 40.903 | Both positive; equality passed |
| 4 workers | 112.707 / 109.559 | 69.699 / 69.908 | 43.009 / 39.650 | Both positive; equality passed |

The mean descriptive reduction is 37.03% for the 2-worker comparison and 37.19% for the 4-worker comparison. Their candidate means differ by 0.806 s, but these are separate sequences with separate controls; this is not a controlled 2-versus-4 ranking. Two workers capture the demonstrated practical gain. Four remain a tested option, with no evidence here for increasing the thread count further. The API default remains 1 and is not changed by this test.

Whole-sequence RSS peaks were 2.321 GiB and 2.296 GiB, including baseline page/GLB references, current captures and allocator history; they cannot establish standalone per-worker memory differences. GPU usage was zero. Each runner finished within its CPU, wall, memory and file ceilings. The first smoke failed in NumPy's lazily imported testing helper, which spawned a CPU-feature probe; the failure is retained, a single harness-only repair replaced that helper with strict direct array comparisons, and the authorized retry plus the 4-worker smoke passed. No production exporter change or repeated unit suite was needed.

The independent reviewer subsequently checked both receipts and the completed execution report and found no material mismatch. Final scoped checks passed: runner syntax, whitespace in the changed tracked documentation, and Markdown fences/local links in nine task documents (113 links). Execution and independent review are complete; no further benchmark is running.

## Available use

Run the normal pipeline for default quiet output. Add `--no-quiet_outputs` only to restore optional debug pictures/reports. The independent Surfel exporter accepts a single experimental change per comparison:

```bash
uv run --no-sync python scripts/export_surfel_viewer.py \
  --dataset path/to/dataset --cache path/to/predictions.npz \
  --mapping path/to/global_mapping.json --mask-cache-root path/to/masks \
  --output path/to/new-viewer --no-dbscan
```

For separate trials, replace `--no-dbscan` with either `--cluster-proxy-voxel-scale 2` or `--early-background-candidates 500000`. These values are experiment inputs, not accepted production settings. The exporter retains its existing masterdata/texture requirements. Early sampling reports background shortfall rather than refilling; plain-point mode additionally requires positive camera depth when this experiment is enabled.

Atlas parallelism uses the compact Python API, leaving the ordinary non-compact CLI unchanged:

```python
from src.surfel_glb import export_scene_glb

export_scene_glb("path/to/generation", "path/to/new-scene.glb", encoder_workers=4)
```

## Remaining decisions and experiments

1. Rick confirmed the 0.4 m limit means **product thickness**. The point cloud has no guaranteed product front axis; camera-Z span includes oblique-view width, while a fitted normal adds assumptions. The axis/measurement question is pending. No guessed threshold has been added, and the existing centroid-distance and cache-consistency gates were not repurposed.
2. Atlas execution completed within [the bounded atlas allocation](atlas-test-authorization.md); previous RGB-only approval was not reused. The other full-scene CPU experiments in [PLAN.md](PLAN.md) remain pending.
3. [Atlas runner](run-atlas-compare.py), [commands/limits](atlas-experiment.md), [execution receipts](atlas-execution.md) and [independent review](atlas-independent-review.md) now contain the complete results. These support fixed-scene CPU export savings, not a new end-to-end service/browser result or an optimal thread count.
4. DBSCAN-off/proxy/early sampling still need full-input timing, point/source counts, fixed orientation/centering/camera comparison and real Viewer inspection. Product retention alone does not establish background quality. The exact NN/SOR change also needs a real matching-overlap timeline and RSS measurement before claiming Viewer-ready improvement.

Four threads have now passed the full comparison but are not a measured optimum. A full raw page owns about 20 MiB of arrays, so four pending pages are about 80 MiB before producer/Pillow/input buffers. Pillow 12.3.0 releases the GIL in major [MEDIANCUT](https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Quant.c#L1699) and [PNG](https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/ZipEncode.c#L138) loops; page supply, CPU contention and memory still limit scaling.

Detailed geometry loops and thresholds: [algorithm explanation](../../docs/viewer_export_algorithm.md#6-几何过滤全量点估计逐阶段保护商品). Current implementation notes: [geometry](geometry-implementation.md), [early background](early-background-implementation.md), [outputs](outputs-implementation.md), [atlas](atlas-implementation.md).
