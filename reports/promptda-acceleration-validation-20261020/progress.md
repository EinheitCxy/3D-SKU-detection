# Progress

## 2026-10-20 16:21 onward

- Re-read retained profiling, acceleration audit and project constraints. Applied autoresearch/resource inspection and continued file-based planning. No production or Git changes.
- Assigned three medium workers separate prototype ownership: selective initialization; preprojection sampling; bounded prefetch/copy reduction. Workers may perform cheap syntax/toy checks, while coordinator serializes actual timing and GPU execution.
- Sampling worker confirmed the original all-dense finite check needs explicit treatment; retained counterexample requested, plus optional conservative finite-domain certification for the real input.
- All three prototypes completed. Their cheap toy checks passed; no worker launched GPU/heavy runs. Coordinator reviewed implementation before serial execution.
- First load CPU A/B passed:433 state entries and2buffers equal, skipped338210816parameter elements covered. Native first timing still contained late vision_transformer import; corrected explicit dependency warmup and retained first run as preliminary evidence. A fresh tagged run is executing.
- Real20frame CPU input check passed: pureprefetch/reducedserial/combined have exactly equal per-batch networkRGB/prompt and every output metadata/RGB field; CUDA remained uninitialized.
- Cached sampling A/B complete: reference median1.691466s→candidate0.482424s; all1M sourceIDs,RGB,world/display coordinates bitwise equal in3pairs. Float32pixel allocation variant gives no material further change in this run. Finite-domain certificate is true for all15240960realpoints; adversarial unselected overflow still exposes weaker generic failure semantics, so no production-default claim.
- Corrected import-warmed CPUload rerun passed withnative5.369615/5.084495s versuscandidate0.915479/0.706410s; bothstate/buffer comparisons exact. Preliminaryconfoundedtimings retained aspreliminary.
- GPUrequestA/Bcompletedexit0:6variants,1coldreference+5warmups+18timedrequests+1skip-loadercheck=25requests, alloutputfields exact. Three-roundinterleaving retained. Pureprefetch median2.976782→1.822156s; reducedserial didnot accelerate. SampledtaskGPUpeak4996MiB. Noold/newpipeline timing subtraction.
- Began directraw→GLB matchednative/optimizedn1 comparison; optimizedcandidate usesselectiveskip-init+pureprefetch+sameindexprojection only afterreal-input finitecertificate passes. Productiondefaults remain unchanged.
- Both continuous raw-delivery arms completedexit0. Native23.778659s, optimized17.679596s; task sampledGPUpeak4996MiB each. Certificate cost included in export. Recorded new unrelatedGPU1process during native; n1/sharedload limitation explicit.
- Direct actual-output comparison completedPASS:9NPZfields plusmillionGLBpositions/colors/geometry associations/node transforms exactly equal. Comparison runs separately outside delivery timing.
- Independent medium evidence reviewer updated combined equality toPASS, observed speedupCONDITIONAL, generic failure-semantic/default-readinessFAIL. Preserved the counterexample and small-sample/shared-load limits.
- Generated README.md, summary.json and focused CSVs from retained receipts only; no model reruns and no production/upstream edits, Git publication or service changes.
- Final focused checks passed: all report Python files parse, summary counts match25requests/24comparisons, delivery partition sums to each actual outer window, all local report links resolve. Independent reviewer rechecked final README/summary/review and found no material misleading claim.
- At16:51CST, read-only Docker filter returned no running codex-promptda-ab containers. GPU1 still hosts unrelated work; no lifecycle action was taken. All owned experiment processes had exited successfully.
