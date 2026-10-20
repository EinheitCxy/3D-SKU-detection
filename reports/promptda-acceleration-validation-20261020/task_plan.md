# PromptDA acceleration feasibility experiments — 2026-10-20

Rick requested a deeper subagent investigation of whether the proposed paths actually accelerate PromptDA. Use isolated prototypes and measured A/Bs, preserve production algorithms and prior profiling evidence.

## Fixed scope

- Input: all20 selected frames of fd-videos/d4d5b1bcd2, existing prepared data and native predictions from Output/promptda-da3-d4d5b1bcd2-20261020.
- Model/config: existing local PromptDA Large checkpoint, RGB1008, FP16 autocast, FP32 weights, batch2, original AREA and RGB quantization, same cameras/frame IDs/point choice seed.
- Editable: files in this report directory and this task's new Output/promptda-acceleration-validation-20261020 artifacts. No production/upstream/Git edits, publication, deployment, installation or service lifecycle changes.
- Three medium workers own load_probe.py, sampling_probe.py and request_probe.py respectively; coordinator owns orchestration, actual heavy/timed runs, evaluation and synthesis. No worker GPU launches or child agents.

## Phases

- [x] Recover latest evidence and choose three independent candidate families.
- [x] Check live resources and lock focused experiment commands/configuration. GPU1 has an additional intermittent other-user process; retain it and record shared-load conditions, use within-run interleaving rather than compare directly against old idle-device timings.
- [x] Implement isolated load, sampling and request prototypes; cheap syntax/toy checks only before coordinator runs.
- [x] CPU A/B: selective skip-init coverage/final-state equality/loading times; cached-point selection/projection times and output semantics.
- [x] GPU A/B on one available device: baseline vs bounded prefetch vs simple copy reduction; include CPU output and memory peaks. Validate loader candidate depth only after CPU state gate passes.25requests,24comparisons against coldreference all exact; sampledtaskpeak4996MiB. Componentcopy reduction did not accelerate; combinations were close to pureprefetch.
- [x] If component improvements pass, measure one combined native-scope delivery A/B with retained outputs; keep startup/request/export evidence distinct. Fresh raw preparation through process exit:23.778659→17.679596s, n=1 per arm; full NPZ/GLB comparison passed, sampled task peak4996MiB both arms.
- [x] Record successful, failed and inconclusive candidates, quality/equivalence limits and cheapest next steps. README.md, summary.json and focused CSVs preserve the results; independent review distinguishes current-input equality from unproven generic failure semantics.

## Resource and evidence bounds

Use existing local Docker model environment, network disabled. Timed experiments run serially, never alongside another owned heavy CPU/GPU experiment. Initial disk~9.9GiB, available RAM~421GiB; keep new arrays in RAM and save only scalar receipts plus at most a few necessary GLBs/NPZ outputs. Physical GPU1 was used previously but must be rechecked. Additional task driver peak must stay within the measured original20-frame DA3 reference16642MiB; no lower artificial DA3 cap. No wall-clock speedup claim before output equivalence and actual timing evidence. Do not add component speedups from different windows.

## Known risks

- DINO hub7.377s includes first import3.302s; fair loader timing must distinguish imports and model construction.
- Forward sampling can weaken full-domain nonfinite validation and can change BLAS roundoff. Record both rather than hide them.
- Producer prefetch must preserve input arithmetic, output order, bounded memory, exception propagation and worker shutdown.
- First loader A/B prewarmed hubconf but missed factory-lazy vision_transformer import. Final-state gate passed, timing marked preliminary; explicit dependency import added and rerun with newtag.
- A third unrelated GPU1 process started during the native delivery arm and was resident before the optimized arm. Preserve it; report shared-load single-pair timing as an observation, not a controlled service or cross-scene estimate.
