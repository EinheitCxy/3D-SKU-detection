# PromptDA / DA3 detailed profiling — d4d5b1bcd2

## Scope

Profile the current Scanner reconstruction paths on the full 1 Hz selection of `fd-videos/d4d5b1bcd2`. Preserve source data, production behavior, existing working-tree changes, and other GPU processes. No publication or dependency installation. Use the current defaults: DA3 joint 504/BF16 with Scanner K/E; PromptDA 1008/FP16/batch 2 with native LiDAR. Record unequal input/output sizes explicitly.

## Phases

- [x] Inspect input, entry points, checkpoints, environment and available resources.
- [x] Prepare shared synchronized inputs; measure selection, video decoding, metadata and serialization.
- [x] Run serial, low-overhead cold + 3 warm model-to-POINTS-GLB measurements, including process startup and memory. Also completed native prepared-input cold delivery and continuous raw-video→GLB measurements; kept their distinct timing windows.
- [x] Run separate diagnostic requests for model loading, CPU preprocessing, GPU submodules, transfers and export internals; report instrumentation effects separately.
- [x] Inspect downstream inputs: no detections/product labels supplied. Complete the independent Scanner reconstruction scope; explicitly exclude SKU/Viewer/Surfel/area rather than fabricate labels or performance claims. Optional scope clarification received no answer during this execution.
- [x] Write detailed report, reproducible commands, measured limits and optimization priorities. 392 timing groups, 20 per-frame rows and 10 per-batch rows retained; focused syntax, accounting and retained native GLB boundary checks passed. Final host query shows no owned containers or GPU jobs.

## Follow-up: download links and acceleration-path audit

Rick requested GLB links and subagent investigation of PromptDA acceleration. This is a bounded source/evidence review, not implementation or a new GPU experiment.

- [x] Confirm both native raw-pipeline GLBs exist and provide links (about16 MB each, 1M colored POINTS).
- [x] Three medium read-only subagents: model loading/checkpoint initialization; request preprocessing/GPU overlap; point output and exact sampling semantics.
- [x] Coordinator: distinguish one-shot CLI and resident-request latency, reconcile findings, rank cheapest useful experiments, retain limitations in ACCELERATION_PATHS.md. Added existing DA3 CPU-standby reference and preserved PromptDA cache/producer boundary. No implementation or new GPU results claimed.

No production/default/Git changes or additional GPU launches are authorized by this review step. Preserve the completed profiling measurements above.

## Resource / measurement decisions

GPU2 has active unrelated jobs; physical GPU1 currently has about 40 GiB free. Use one selected GPU serially, recheck before launch, and retain all other processes. Root filesystem has about 6.25 GiB free; avoid copying weights, keep generated outputs bounded and check before large writes. Cold means a fresh process/model, not an emptied filesystem cache. GPU substage timings belong to a separate diagnostic run.

## Issues

- Sandbox cannot access NVIDIA devices/driver. Host read-only `nvidia-smi` and Docker environment inspection succeeded through scoped escalation.
- Existing `profile_candidates.py` asserts exactly 14 frames and a fixed DA3 output grid; this recording has a different duration. Use a report-local profiling harness with dynamic actual shapes rather than changing production inference.
- Native raw→GLB DA3 run passed. The next PromptDA launch stopped in preflight before model launch: GPU free memory was restored, but the driver utilization window still reported 100%. Preserve this attempt; inspect processes and use a bounded readiness wait and an explicit fresh recovery output path. This is not a model failure.
- Recovery completed: host inspection found only the original GPU1 service with utilization 0%; `promptda-full-ready` finished with exit 0. Both raw-pipeline measurements are now complete. Original partial orchestration and empty preflight output directory remain retained.
