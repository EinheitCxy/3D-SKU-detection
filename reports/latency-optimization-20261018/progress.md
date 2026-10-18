# Progress

## O01 — 2026-10-18, start

Execution: in progress. Evidence: not yet evaluated.

Rick approved implementation and video3 timing of DA3 initialization, point ranges, sampled ground proposals and GLB projection/encoding work. Reusing three medium workers with disjoint ownership; coordinator retains experiment scheduling and synthesis. Skills: autoresearch, planning-with-files, work-journal, get-available-resources.

Read current Git status, target scripts, input records and existing profiling status. Preserving unrelated tracked/untracked work and historical reports. No GPU experiment or production mutation yet.

Next: capture an isolated baseline before workers edit, inspect capacity, and begin scoped implementation.

## O02 — 2026-10-18 21:03 CST, baseline running / implementation

Saved original src/utils/main/defaults/config under `runtime/latency-optimization-20261018/baseline-code`; model/dependency/data trees are symlinked, not copied. Capacity: GPU2 free19038MiB, existing6221/8550 residents preserved; RAM available427.81GiB, disk12.68GiB.

Started paired baseline via `run_case.py --case runtime/latency-optimization-20261018/baseline-video3`, source root set to the isolated snapshot, encoder_workers=1, fixed video3 input and seed42. Session79826. Scalar function spans and identical diagnostic domain captures in both arms; full historical six-video AST timing mode is not the new denominator.

Implemented typed vector searchsorted in the coordinator-owned function, preserving JSON integer endpoints and sorted point order. Added int32/int64, empty/missing/background/removed-observation tests; validation pending until a CPU window is available. Worker delivery reports89 focused mesh/GLB/LOD tests passed, real-data projection comparison pending. DA3 coverage and ground implementation remain in progress.

## O03 — 2026-10-18 21:09 CST, baseline complete / real coverage rejection

Baseline completed: total253.139s, Viewer129.854s, ZIP138.097s, GLB114.041s, Viewer2402589 points. [Exact scalar summary](baseline-summary.json). It uses fixed seed42 and function instrumentation, so its denominator is this experiment rather than the previous unpaired profile.

First real CPU DA3 pair rejected baseline at coverage audit: six missing state tensors, `model.da3.head.scratch.output_conv2_aux.{1,2,3}.2.{weight,bias}`; no unexpected keys or shape/dtype mismatches. Receipt retained at `runtime/latency-optimization-20261018/da3-cpu-pair/`. No candidate inference ran. DA3 worker is tracing these heads before revising the initial overly broad full-state coverage requirement to coverage of skipped tensors while preserving native initialization of uncovered tensors. This is a finding about the existing checkpoint, not evidence of a speedup.

Corrected the new diagnostic's classmethod wrapper to preserve subclass binding before candidate execution. Baseline calls the base class and is unaffected. Starting independent real-label and captured-ground probes while DA3 worker resolves coverage.

## O04 — 2026-10-18 21:16 CST, real component results / conservative defaults

The initial DA3 key-only rejection was a diagnostic error: the6keys are shared LayerNorm aliases, all physical parameters are covered. Alias-aware strict load succeeded on the real checkpoint. CPU pair constructor11.826516→0.651774s, construct+audit+load12.993895→2.045339s;1383state/1377physical keys,406skipped weights,0uncovered. Native nonpersistent buffer and sampled loaded weights equal. GPU forward still pending.

Point ranges real comparison PASS,3.489543→0.000964s/all711endpoints equal;10targeted tests PASS. Ground probe first failed only at readonly-mmap→Open3D conversion; fixed diagnostic creates a writable cloud copy before either arm timer, failed receipt retained. Successful same-domain probe10.879015→0.980081s,0product changes/51286background deletions. Near-vertical candidate relative to known floor prevents automatic promotion; cap0 remains production default,50K stays explicit experiment.

GLB real sample has6alpha differences; geometry unchanged, all-RGB max41 includes transparent dilation. Worker is exposing explicit source_world/camera_basis modes, retaining original source_world arithmetic by default and sharing frame constants. Full same-generation encoder1/2/4 exports will compare decoded alpha/RGB and geometry, not hashes. The projection experiment will not be advertised as lossless.

Fixed profiler closure binding so candidate ground/helper spans reference instrumented functions. Baseline lacks these helper subspans; no missing span will be reported as0. Outer filter/pipeline spans and the independent ground pair remain valid.

## O05 — 2026-10-18 21:23 CST, GLB modes and encoder comparison

Mode implementation completed. New real29frame probe:source_world snapshot geometry/RGBA/alpha exact;camera_basis same geometry but6alpha/9318RGB texels differ, both-opaque maximum RGB delta1. Default remains source_world; frame constants reuse enabled. No threshold relaxation or JPEG/f16 bypass.

Same-generation full source_world exports underway, session87835. Candidate1 completed108.396069s/RSS sampled peak957362176bytes;candidate2 completed67.627083s/1041502208bytes. Candidate4 and final all-page comparisons still pending. One arm each, same process and fixed order; no final default chosen yet. The script's initial aggregate status is conservative until finalization; root progress is running, not a terminal failure.

Independent reviews passed physical alias loading and ground default control; GLB reviewer found new invalid-mode mkdir side effect, worker is moving validation before directory creation and adding the small boundary test. Root documentation patch initially contained an empty hunk and was rejected without source changes; reapplied valid hunks. No benchmark repeated for that editing error.

## O06 — 2026-10-18 21:25 CST, full encoder comparison accepted

Session87835 exited0. Samegeneration source_world exports1/2/4 workers108.396069/67.627083/66.638055s; sampled selfRSS957362176/1041502208/1084465152bytes. Three complete comparisons (old reference→new1,new1→2,new1→4) passed document/accessor andall128295600decodedRGBA texels,alpha/RGB0changed. Comparisons excluded from armtimers; fixedorder/sameprocess/oneeach, not a repeated distribution.

Decision: default2, explicit4 retained. Two workers capture most observed benefit;4's additional0.989s(1.46%) is not demonstrated stable advantage and consumes~41MiB extraRSS. Full video3 safe candidate uses2,source_world,ground0. Worker updating onlydefault/docstring/tinytests; coordinator runs existing DA3reconstructor/worker gate before snapshot/launch. No camera_basis/50K speed gain will be counted in final production path.

## O07 — 2026-10-18 21:28 CST, final safe candidate running

Default2 and invalid-mode-before-mkdir changes completed;6direct default/FIFO tests passed. ExistingDA3reconstructor/worker23tests passed. Final source copied to`runtime/latency-optimization-20261018/candidate-code` afterworkers froze production files. Started finalGPU2 video3 via`run_case.py`,session88625, encoder2/source_world/ground0; model/result processes fresh, same29-frame input/seed42/threads4.

Same-run DA3 constructor baseline12.624388s→candidate0.914212s;CPUload13.562327→2.424568s. Matching/dedup completed and386globalIDs/711instances present in both; final Viewer/GLB still running. These are observations, GPU numeric output equivalence/fulltotal pending. IndependentCPU timings remain separately labeled.

## O08 — 2026-10-18 21:39 CST, complete / final evidence and review

Final safe candidate exited0 for both core and delivery: complete253.138706→186.894435s (26.169% less), Viewer129.854374→109.162660s, ZIP138.096691→117.362010s, GLB114.041383→68.770372s. Same-run point ranges3.939481→.001608s. Encode cumulative wall43.640→45.379s overlaps bake; main-thread join43.649→.900s is waiting rather than total encoder work. No attribution to unchanged matching/filter fluctuations.

Direct output comparison PASS:15DA3 numeric arrays, global mapping,8Viewer binary arrays, objects/manifest and finalGLB entire payload are exactly equal. Both386globalIDs/711observations/2402589Viewer points/1282759baked surfels/2565518triangles/31pages/121197580GLB bytes. Task driver200ms sampled peak15132MiB (14.777GiB) in both arms. All owned GPU/runtime processes exited; last GPU2 snapshot21:32CST retained only original6221/8550residents. No GPU0/1 workloads launched.

Three medium final reviews PASS: DA3 audit/constructor/output evidence, geometry/ranges/ground nonpromotion, and GLB overlap/default/output accounting. Focused tests include final DA3loader11, existing DA3reconstructor/worker23, point-range/schema10, filter39 plus later focused6/1, mesh/GLB/LOD89 plus later focused7/6; overlapping selections are not summed. No repeated full suite or hash/CRC.

Canonical docs and task status updated. Heavy-operation survey recommends a bounded existing-Torch RGB/mask/dilation experiment with transfers/context included; no new GPU algorithm in this implementation. Ground50K and camera_basis remain explicit experiments after failed output/quality promotion gates. Current defaults preserve the pairedvideo3 output exactly. No commit/push/deploy/upload; no change to unrelated files or historical root planning records. Each performance arm was run once; optimized multi-video mean/P95 and external accuracy/visual acceptance remain unmeasured.
