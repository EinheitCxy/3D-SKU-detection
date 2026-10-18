# Video3 latency optimization

Approved scope: Rick's 2026-10-18 request to check DA3 checkpoint coverage and skip overwritten random initialization, vectorize point ranges, reduce ground-plane proposal cost, reuse bake projection work, and test bake/atlas pipelining. Preserve f16/JPEG boundaries, product protection and Surfel diameter cap. GPU experiments on GPU2 only; no publishing or service changes.

## Phases

- [x] Record current sources/resources; prepare an isolated baseline and fixed video3 input.
- [x] Audit actual DA3 checkpoint keys/shapes, implement selective initialization skipping and focused tests.
- [x] Implement typed vector point ranges and edge-case tests.
- [x] Implement sampled ground proposals with full-domain validation and focused tests (explicit experiment, default0 retained).
- [x] Implement camera-basis bake projection and validate alpha/RGB/corner boundaries; test existing 2/4-worker encoding (camera_basis experimental; source_world+frame constants default).
- [x] Run one controlled baseline and candidate video3 comparison, plus the smallest component comparisons needed to attribute changes.
- [x] Review heavy operations for existing PyTorch alternatives, update canonical documentation and record all measured/remaining limitations.

## Ownership

- Coordinator: this plan/findings/progress, `src/web_viewer_export.py`, its point-range tests, experiment orchestration, final docs/review.
- DA3 medium worker: `src/da3_model_loading.py`, `src/da3_runner.py`, dedicated loading tests and DA3 coverage/component probe under this report directory.
- Geometry medium worker: `utils/pointcloud_filter.py`, `test/test_pointcloud_filter.py`, ground component probe/report.
- Delivery medium worker: `src/surfel_mesh.py`, directly necessary `src/surfel_lod.py`/`src/surfel_glb.py` edits and related tests, bake/encoding component probe/report.

## Acceptance

Checkpoint coverage is measured, not assumed. Skipped parameters must all be supplied by the checkpoint. Point ranges keep exact endpoints and schema. Ground evaluates its final proposal on the full post-SOR/DBSCAN domain and retains existing guards; this is a separately reported algorithm change. GLB keeps the f16/JPEG input boundary, alpha/depth checks, ordering and long-diameter cap. Performance conclusions use fixed video3 inputs and explicitly identify component vs full-pipeline timing. No unrelated edits or extra GPU workloads.

## Errors / decisions

- Root planning files belong to historical personalcare work and are left untouched. These three task-local planning files are the current task's record.
- Existing six-video timings remain historical evidence; a new paired experiment will not be silently mixed into their mean.
- Comparison uses the identical existing 29-frame/711-box video3 dataset and explicit seed42, independent model/result processes, OMP/OpenBLAS/MKL=4, GPU2 only. The new profiler uses function spans rather than the previous per-statement AST/CUDA-event mode; compare this pair internally, not against the old 236.866s as a speedup denominator.
- Both arms capture the post-DBSCAN ground domain and final labels for component attribution. Their copy/save costs are recorded separately and remain visible in total experiment wall time.
- First actual DA3 audit found six missing auxiliary-head state tensors. The full-state-strict candidate was refused. Before proceeding, establish their usage and retain normal initialization for uncovered parameters; all skipped parameters still require complete checkpoint coverage. The failed audit is retained.
- Actual resolution: six keys alias a shared LayerNorm; physical parameters are fully covered. Alias-aware auditing plus native strict loading passed. Retain the failed key-only audit as corrected diagnostic history.
- Ground50K changes which plane is selected and its candidate is nearly perpendicular to orientation floor in this video. Keep0/full-domain default; do not count the experiment's speed gain as a promoted production gain.
- Camera-basis reassociation changed6alpha texels in real sample. Keep original source_world arithmetic default, per-frame constants reuse enabled; camera_basis remains explicit experimental mode. Final encoding workers selected after same-generation1/2/4 timing/equivalence.
- Full same-generation source_world1/2/4 exports completed108.396069/67.627083/66.638055s. Allaccessors/document and128295600decodedRGBA texels match old source and among worker counts. Select default2:37.61% faster than1 in this single-order component comparison;4 only1.46% faster than2, without repeated evidence and with~41MiB additional sampled RSS. Explicit4 remains supported. Final full candidate will use2.
- Final safe candidate completed with source_world/ground0/encoder2: full253.138706→186.894435s, Viewer129.854374→109.162660s, GLB114.041383→68.770372s. All15DA3 numeric arrays, mapping,8Viewer arrays/metadata and entire final GLB payload match exactly. Each arm once; no optimized multi-video average claimed. Three medium final read-only reviews PASS. See README/review for final validation and limits.
