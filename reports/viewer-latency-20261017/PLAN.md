# Viewer latency follow-up — 2026-10-17

Status: code/read-only design authorized by Rick. Rick subsequently requested the atlas test; its bounded 1/2/4-thread execution is authorized and recorded in [atlas-test-authorization.md](atlas-test-authorization.md). Hardware execution for the other geometry directions remains pending. The previous RGB-only budget is closed and is not reused. This plan does not authorize publication, deployment, GPU execution, new dependencies, or edits outside the paths below.

## Scope and ownership

Three concurrent medium leaf workers at most. Workers preserve existing dirty changes and own disjoint paths. The coordinator owns scheduling, decisions, integration review, and this record. Existing datasets, prediction caches, orientation sidecars, generations and reports are read-only.

1. **Atlas worker:** `src/surfel_glb.py`, `test/test_surfel_glb.py`, task-local encoding runner/report. Two encoder threads/two pending pages initially; Rick's subsequent question about more aggressive threading added a four-thread/four-page candidate. Ordered single BIN writer; preserve palette, alpha, texture sampling, geometry, material order, LOD and Surfel maximum diameter. Serial remains the default and comparison baseline. Do not subtract overlapping encode time from bake time as if stages were sequential.
2. **Geometry worker:** `utils/pointcloud_filter.py`, `src/web_viewer_export.py`, `src/viewer_export_tuning.py`, `scripts/export_surfel_viewer.py`, `test/test_pointcloud_filter.py`, `test/test_web_viewer_export.py`, task-local filtering runner/report. Prepare only label-independent NN/SOR raw results on the identical valid domain and point order. Apply product protection, per-stage 20% guard, DBSCAN and ground removal after labels. Provide independent DBSCAN-off, explicitly scaled proxy DBSCAN, and early-background-candidate options. Approximate options remain off by default pending quality review. Keep the original dense grid/depth for Surfel derivatives, edge caps, sampling support and rendering.
3. **Matching/output worker:** `utils/config.py`, `utils/geometry_3d.py`, `utils/matching_algorithms.py`, `utils/sam3_utils.py`, `src/inference.py`, `main.py`, relevant existing matching/main tests, `modules/video_to_dedup/run.sh`, `scripts/3d/pipeline/video_to_viewer.sh`, `config.yaml`. Turn off auxiliary matching/dedup/debug pictures by default with an explicit opt-in; retain source frames, Viewer textures/thumbnails, masks/caches and machine-readable summaries required downstream. Add the confirmed 0.4 m product-thickness rejection after its axis/measurement definition is resolved. Do not substitute the existing centroid-distance or cache-consistency thresholds.

Workers may hand off read-only reviews or execution after completing their owned implementation. Documentation updates: `docs/3d_core.md`, `docs/surfel_glb.md`, `docs/viewer_export_algorithm.md`, `docs/profiling_breakdown.md`, and this directory. No unrelated changes, broad staging, reset, cleanup, commit or push.

## Experiment budget proposed for approval

- CPU process time, including tests and comparisons: at most 7,200 seconds.
- Experiment elapsed wall time: at most 7,200 seconds; every command at most 900 seconds.
- GPU time: zero. No inference/model launch, service call, COS upload or network install.
- At most two concurrent heavy CPU commands, at most eight executing CPU threads in total; one heavy comparison at a time when timing is being assessed.
- Aggregate RSS at most 24 GiB; new temporary plus retained files at most 2 GiB. Stop on a limit; preserve partial measurements rather than silently reducing data or changing backend.
- At most six official single-change candidates within the same proposed hardware caps: atlas two threads, atlas four threads, NN/SOR preparation, DBSCAN-off, proxy DBSCAN, early background sampling; at most one debug candidate and one bounded retry, retaining the original failure. No cumulative worker-call/wall quota, as previously requested; each worker task still has a finite allocation and stop condition. Four threads do not authorize an extra sequence outside the CPU/wall budget.
- Keep existing locked dependency versions. The repository `.venv` currently points to a missing directory. As engineering preparation, attempt an offline, hardlink-only restoration from the existing uv cache into a new task-local environment, without replacing the root link or changing `uv.lock`. A missing cached package stops restoration; no downloads, new dependency versions or replacement runtime are authorized. Report readiness before tests or comparisons. Formal candidate measurements still require the hardware approval above.

## Acceptance fixed before observing candidate results

- Atlas: smoke first, then full fixed video3 generation in B1/C1/C2/B2 order, same process/input/CPU set/encoding parameters. Compare complete page RGBA including transparent RGB, final GLB geometry/UV/index/material associations, decoded textures and byte length outside the raw export timer. Both paired complete-export savings must be positive for a performance acceptance; incomplete/mixed results remain inconclusive.
- NN/SOR preparation: same valid input/order/config, raw result and final masks/selected source indices/labels/Surfel arrays equal to serial execution. Ground/orientation random state must be fixed for a comparison. Unit equality alone establishes correctness, not actual latency hidden behind GPU matching. Record separate preparation/join/finalization timings; do not claim a new end-to-end Viewer gain without that measurement.
- DBSCAN-off/proxy/early sampling: isolate one change against the same frozen input, orientation and camera. Preserve valid product points before the existing product budget, report all retained source counts and underfilled background budgets, and compare rendered subject completeness, new holes, background floaters and ground coverage. Weighted proxy cluster size does not imply an identical neighborhood graph.
- Early sampling: a 500,000-candidate run may finish below the final background budget after filtering; report that result explicitly. Do not silently add a refill or fallback policy. Keep full dense geometry/depth available. If this run loses coverage, reject it as a default and use the evidence to propose a separate refinement.
- Approximate candidates may be implemented as opt-in controls but cannot replace the default from point counts or timing alone. Render comparisons and limitations must remain reviewable.
- Matching: 0.4 m is a product-thickness threshold, not camera distance, inter-frame centroid distance or projected depth mismatch. The exact observable axis and span rule are still pending. Test the inclusive 0.4 m boundary and rejection on either observation without admitting rejected candidates through uniqueness reassignment.
- Run only focused tests covering the modified behaviors. An independent reviewer checks the reports and diffs before acceptance. Keep the existing 267.565 s historical profile separate; this CPU-only cycle does not produce a new full GPU/service/browser baseline.

## Open decisions and next action

1. Complete the authorized atlas subset under [its recorded limits](atlas-test-authorization.md); the remaining geometry experiments do not inherit that execution authorization.
2. Resolve product-thickness measurement: the current point cloud has no guaranteed product front axis, and the target matcher currently samples a bbox rather than a canonical mask. A camera-Z span is measurable but includes oblique-view width; a fitted plane normal requires additional geometric assumptions. Resolve this before changing matching behavior.

Read-only design and source preparation completed by medium agents. Current code, 189 passing engineering tests and atlas execution progress are tracked in [STATUS.md](STATUS.md). The remaining geometry experiment approval and the thickness-axis choice remain pending. No full-scene atlas performance result existed at the start of this test; subsequent results will retain their own timing boundary.

## Atlas test completed

The authorized focused engineering test has completed. Both 2-worker and 4-worker full-generation sequences passed the unchanged two-positive-pairs rule and all output comparisons; [execution](atlas-execution.md) and [independent evaluation](atlas-independent-review.md) retain the actual commands, resources and limitations. Both reductions were approximately 37% against their own serial controls. This test did not change production defaults, initialize a new managed ResearchState, or execute the other geometry experiments. No extra benchmark is scheduled.
