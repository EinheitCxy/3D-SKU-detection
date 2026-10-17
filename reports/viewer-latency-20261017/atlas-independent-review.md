# Atlas independent review — 2026-10-17

## Method review, before results

Reviewed only the authorized atlas documents, comparison runner and `src/surfel_glb.py` / `src/surfel_mesh.py`. No benchmark, production edit, input-data access or metric change was performed. No timing or performance acceptance conclusion is made at this stage.

No blocking defect was found in the prepared comparison. Encoder worker count and its matching FIFO bound are the only intended arm changes. Both candidates use the same source, export parameters, seed, four-CPU affinity and B1/C1/C2/B2 order. The timer encloses the complete exporter call; comparison and GLB readback occur afterward. Full GLB byte equality plus complete prequantization page arrays, decoded PNGs, compact accessors and JSON associations provide strong output-equivalence evidence if every arm completes. See `run-atlas-compare.py`, `compare()` and the arm loop.

The following limits must accompany the eventual interpretation:

- Every page is copied and retained by the capture wrapper inside the export timer. The measurement is a complete export **with this common instrumentation**, not an uninstrumented production latency measurement. Copies also extend the serial producer path and may affect overlap. See `captured_baker()` and the immediately following exporter call.
- After B1 the process retains B1 page captures and GLB bytes for subsequent comparisons. Subsequent allocations, allocator retention and OS file caching share the same process history. Per-export absolute sampled RSS is useful for resource diagnostics; cumulative `ru_maxrss` cannot isolate an arm, and even per-arm samples cannot establish standalone exporter peak-memory differences. See baseline assignments and the resource sampler. Report this confound explicitly rather than subtracting retained reference bytes to manufacture isolated peaks.
- Each 2-worker or 4-worker sequence has its own serial controls. Their separate candidate absolute times do not support a controlled direct 2-versus-4 ranking. Report both paired savings for each sequence; the prespecified both-positive rule is a local engineering acceptance criterion, without statistical certainty or host exclusivity.
- The baker is a single serial producer in frame order (`src/surfel_mesh.py:204` onward), while encoding is submitted to a bounded FIFO (`src/surfel_glb.py:175` onward). All 31 pages must be emitted, but their availability rate, FIFO head waits and CPU contention can limit encoder concurrency. A small or negative 4-worker gain does not imply a broken parallel implementation. Component join/bake wall times do not measure total encoding CPU or thread occupancy.

Next review: inspect smoke and official receipts and execution notes once complete; verify all four arms, 31 pages, comparisons, paired times and resource ceilings. Do not infer success from a partial sequence. Production default remains 1 worker.

## Staged results review: 2 workers complete; 4-worker official pending

Independently read the initial failed 2-worker smoke, its single retry, successful 4-worker smoke, 2-worker official receipts/stdout, readiness receipt and current runner. No experiment was rerun. Recalculation used only the recorded numbers and Python standard library.

The initial smoke correctly remains failed (exit 1): its traceback shows `numpy.testing` lazily importing a CPU-feature probe that executes `subprocess.run`, followed by the runner's SIGUSR1 child guard. The repair uses explicit dtype/shape checks and `np.array_equal` (`run-atlas-compare.py:195`); complete GLB byte comparison, page scope, timer and both-positive-pairs acceptance remain intact. This does not relax the metric or equality contract. In particular, `array_equal` without `equal_nan` is stricter for NaNs. The 2-worker retry and first 4-worker smoke each completed all four arms, exited 0 and returned `smoke_pass`. Each captured only one page, so these are path/comparison smoke results, not concurrency saturation or performance evidence.

| 2-worker official arm | Workers | Raw complete export seconds | Pages | Complete / compared |
| --- | ---: | ---: | ---: | --- |
| B1 | 1 | 112.817729498 | 31 | yes / yes |
| C1 | 2 | 70.663098771 | 31 | yes / yes |
| C2 | 2 | 70.555384575 | 31 | yes / yes |
| B2 | 1 | 111.458616294 | 31 | yes / yes |

Recomputed B1−C1 = **42.154630727 s (37.3653%)** and B2−C2 = **40.903231719 s (36.6981%)**. Both are positive, matching the stored primary minimum 40.903231719 s and the prespecified `pass`. Serial mean is 112.138172896 s; 2-worker mean is 70.609241673 s, a descriptive mean reduction of 37.0337% (ratio of means). Stdout agrees with the receipt. The 2-worker result satisfies this fixed-input engineering acceptance criterion, subject to the method limits above; 4-worker official assessment is pending.

All four official arms report 2,402,589 input points; 1,297,863 LOD-selected points; 1,282,497 baked surfels; 2,564,994 triangles; and 31 captured/baked pages. Each capture contains 646,398,888 array bytes. Every full GLB is 121,513,344 bytes and every comparison flag is true, including full GLB byte equality, complete page RGBA/geometry/UV/index arrays, decoded PNG RGBA, accessors and document associations. This supports output identity to the serial baseline, not a separate absolute-quality claim.

The official outer command exited 0 in 383.882902168 s, with 1,033.342718 child CPU seconds; runner resources record 383.218687972 s and 1,032.730240 total CPU seconds. These remain below 900 s / 2,000 CPU-s; observed maximum RSS is 2,491,920,384 bytes (2.32078 GiB), below 12 GiB. Each GLB is below the 256 MiB file bound. The externally enforced timeout was 889 s, conservatively shorter than the prepared 899 s wrapper and runner's 900 s ceiling, without affecting this successful sequence. All reported import/post-export affinities stay within CPUs 3/22/27/29 and CUDA is hidden. `children_observed` is false, but reaped-child CPU is nonzero (0.000674 s): do not describe the sampling guard as proof that no transient child ever existed. This does not change measured self/export timing or exceed the total budget.

Per-arm sampled export RSS is 1,567,088,640 / 2,400,165,888 / 2,370,256,896 / 2,405,748,736 bytes in B1/C1/C2/B2 order. These are same-process absolute diagnostics, with the retained-reference/allocator confound already documented; B2 itself is substantially above B1, so interpreting their differences as encoder-worker memory overhead would be invalid. The readiness receipt is capacity context only, not exclusive-host or isolated-memory evidence. Command UTC start/end fields explicitly derive retrospectively from write mtime and duration, not direct clock captures.

Source receipts: `atlas-official-2w-01/{partial.json,command.json,stdout.log}`, `atlas-smoke-2w-01/{partial.json,command.json,stdout.log}`, `atlas-smoke-2w-01/retry/{partial.json,command.json,stdout.log}`, `atlas-smoke-4w-01/{partial.json,command.json,stdout.log}`, and `atlas-resource-readiness.json`.

## Final receipt evaluation: both official candidates pass

The 4-worker official receipt and outer command/stdout are now complete. The execution narrative was not yet present at this review checkpoint; narrative consistency remains a separate small follow-up, without rerunning experiments.

| 4-worker official arm | Workers | Raw complete export seconds | Pages | Complete / compared |
| --- | ---: | ---: | ---: | --- |
| B1 | 1 | 112.707407506 | 31 | yes / yes |
| C1 | 4 | 69.698650377 | 31 | yes / yes |
| C2 | 4 | 69.908322206 | 31 | yes / yes |
| B2 | 1 | 109.558556817 | 31 | yes / yes |

Independently recomputed 4-worker B1−C1 = **43.008757129 s (38.1597%)** and B2−C2 = **39.650234611 s (36.1909%)**. Both are positive; the stored primary minimum is correct. Four-worker serial mean is 111.132982161 s and candidate mean is 69.803486292 s, giving a descriptive mean reduction of 37.1892%. The two-worker corresponding means remain 112.138172896 / 70.609241673 s, or 37.0337%. Each candidate passes its own locked criterion against its own controls.

Across both official sequences, all **eight arms** completed and compared all 31 pages, each with 646,398,888 captured array bytes and a 121,513,344-byte GLB. All five complete-output comparison flags are true in every arm. Both phases are `official`, both outer return codes are 0, and each receipt's `performance_accepted` agrees with independent arithmetic. All reported thread affinities remain inside the fixed four CPUs. The equality repair remains outside the raw export timer and preserves the complete array dtype/shape/value and GLB byte requirements; neither acceptance nor timing has been substituted.

Four-worker runner wall/total CPU is 379.585673285 / 1,023.788256 s, versus its 900 / 2,000 s limits. Outer wall/CPU is 380.310366315 / 1,024.460784 s; maximum observed RSS is 2,465,116,160 bytes, below 12 GiB. Its command has direct UTC captures, unlike the retrospective 2-worker timestamps. Reaped-child CPU is again tiny but nonzero (0.000841 s) despite no sampled child; the same monitoring qualification applies. Together with both successful smokes and the preserved failed smoke, all five outer command receipts sum to approximately 785.402 s wall and 2,101.483 CPU-s, below the execution worker's 2,400 / 5,000 s allocation. Final GLB size is below the per-file ceiling; temporary-file count/size limits are structurally enforced by the runner/exporter, rather than a separately measured aggregate disk high-water mark.

Accepted conclusion: on this fixed video3 generation, within the four-CPU allocation, both 2 and 4 encoding workers reduce the instrumented complete-export wall time by roughly 37% against their respective serial controls while producing identical outputs. This is a useful engineering result for an opt-in configuration. It is not an end-to-end reconstruction/service latency result, a multi-scene generalization, a statistical significance claim, a standalone memory-overhead comparison, or evidence of an optimal thread count.

The independent sequences do **not** establish that four workers outperform two: candidate means differ by only 0.805755382 s while their serial controls also differ. No additional-thread experiment is justified by these receipts alone. Four-worker encoding FIFO waits are already about 0.757 / 0.759 s, whereas the reported bake remainder is about 66.642 / 66.860 s. These disjoint caller-thread diagnostics are consistent with the serial producer limiting additional gains; they do not directly measure encoder occupancy. If choosing among the tested opt-in settings solely from this evidence, 2 workers is the smaller concurrency setting with essentially the same demonstrated relative gain. Production default remains 1; this review does not change it or authorize a broader experiment.

Additional final sources: `atlas-official-4w-01/{partial.json,command.json,stdout.log}` and the current `run-atlas-compare.py`.

## Execution narrative consistency confirmation

Read the completed `atlas-execution.md` after all execution stopped. Its per-arm times/CPU/RSS, paired savings/percentages, return codes, command durations, UTC timestamp qualifications, single preserved smoke failure and repair/retry history agree with the reviewed receipts. Its recommendation of 2 workers as the smaller opt-in adoption candidate, retention of 4 as an option, unchanged default of 1, and refusal to claim 2/4 equivalence, optimality or end-to-end gains agree with this evaluation. The narrative correctly qualifies one-page smoke coverage, capture timing and shared-process memory confounds. No material reporting discrepancy was found. The current linked-file inventory and outer collector deadline are execution-worker statements, not independently remeasured here; they are not used to establish the performance verdict. No experiment or validation export was rerun. Independent review is complete.
