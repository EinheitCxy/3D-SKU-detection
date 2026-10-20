# Progress

## 2026-10-20

- Started input/source/environment inspection. Applied autoresearch, resource inspection and planning-with-files workflows. Read current project instructions and retained existing dirty work.
- Verified host GPU resources and existing Docker image. No models, source data or Git state changed; no GPU run started yet.
- Identified the old profiler's 14-frame-specific constraints before execution; preparing a report-local dynamic harness.
- Prepared all 20 selected frames with the native Scanner functions; recorded per-operation preparation events.
- Completed serial low-overhead baseline runs on GPU1 (cold + 3 warm per model), both exit 0. Completed separate diagnostic runs for both models, both exit 0; includes CUDA events without per-layer synchronize and a separate operator-profile request.
- Asked an optional scope clarification about missing SKU detections; continue the independently executable reconstruction profiling while awaiting a reply. Do not fabricate product labels or call Scanner POINTS GLBs textured Surfel output.
- Added native one-shot delivery mode to measure fresh-process → NPZ/GLB completion directly, rather than misusing the four-request baseline process lifetime.
- Both prepared-input native delivery runs passed. Added continuous raw-video→GLB measurements. DA3 completed; PromptDA was not launched because the GPU preflight saw 100% utilization immediately after the DA3 exit, despite restored free memory. Retained receipt and empty output directory; investigating the rolling driver-utilization window and continuing with an explicit new attempt.
- Host inspection confirmed GPU1 idle with only the original service. Added a bounded readiness wait; fresh tagged PromptDA raw run passed. Both real continuous raw-input measurements complete, with no quality or precision changes.
- Generated README, summary.json, SOURCE_SPANS.md, CUDA_MODULES.md, OPERATORS.md, CPU/operator/module/stage/request CSVs and per-frame/per-batch tables from retained artifacts. Updated docs/profiling_breakdown.md with a scoped link; historical full SKU timings remain intact.
- Focused report validation PASS: all five script syntax parses; both candidates' baseline/diagnostic/full receipts complete; 20 frames each; native GLB readback PASS and one million points each; additive native-stage closure and <0.001s outer residual; 392 timing groups/20 frame rows/10 batch rows. No repeated GLB reads or unrelated regression/hash checks.
- Final host check: no owned containers or GPU jobs, GPU1 restored to original service state. No production algorithm changes, Git commits/push, deployment, installs or source-data edits.
- Follow-up16:04: Rick requested GLB downloads and subagent acceleration review. Both files confirmed present (~16MB each), links provided immediately. Started three medium read-only explorers with nonoverlapping source responsibilities and no GPU/write permissions; coordinator will synthesize a ranked report.
- All three review agents completed; source path correction confirmed by loading reviewer. Coordinator inspected existing DA3 CPU standby/main cache dispatch and wrote ACCELERATION_PATHS.md with ranked candidates, measured time windows, cheapest nextA/Bs and explicit unverified equivalence limits. Official PyTorch CUDA/checkpoint documentation consulted for asynchronouscopy and loader mechanisms; no additional GPU/model execution or production edits.
