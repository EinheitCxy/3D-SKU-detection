# Ground-stack footprint evidence and performance design

## Status and invariant

Rick approved this design on 2026-08-13. The required metric remains the support-plane projection footprint union.

A_stack = area(union(OBB(project_to_support_plane(points(global_id)))))

Each physical carton contributes exactly one support-plane OBB. Polygon union is computed once, so overlap is counted once and upper-box overhang is included. This is not surface, front-face, contact-only area, or an arithmetic sum of carton areas.

This work must not lower the refined-plane 10 mm residual, 10 percent background coverage, all-ID, or support-plane ambiguity gates. It must not silently skip an incomplete global ID, replace a SAM3 mask with a bbox, publish a partial total, or replace OBB/union with TSDF, mesh, learned occupancy, or a new trained model.

The work has three isolated tracks: strict-result performance, SAM3 mask reuse, and multi-view evidence. New evidence begins in shadow mode: it is reported but cannot alter accepted/rejected status until separately calibrated.

## Measured baseline

On 2026-08-13, fd_area_test ran 18 images and 81 detection boxes without changing the final formal result:

| Segment | Seconds | Share |
| --- | ---: | ---: |
| select_support_plane | 105.01 | 82.4 percent |
| SAM3 source masks | 19.25 | 15.1 percent |
| per-ID OBB | 0.14 | 0.1 percent |
| artifact creation | 0.20 | 0.2 percent |
| validation and I/O | 2.77 | 2.2 percent |
| total | 127.37 | 100 percent |

The DA3 cache already stores depth, intrinsic, and extrinsic, but the footprint stage reads only world_points and confidence. It does not yet test that observations of one global ID agree under the DA3 camera model. The existing precision-grid comparison proves polygon arithmetic stability, not physical measurement uncertainty.

## Architecture

The stage retains strict DA3, image, detection, and global-mapping validation. A verified per-frame SAM3 source-mask bundle supplies masks to the existing support-plane, OBB, and union calculation. The formal accepted/rejected output is unchanged. A parallel shadow evidence report contains reprojection, leave-one-out, and mask-robustness evidence.

The formal calculation completes and freezes its status, value, polygons, union, and rejection reason before optional evidence loading starts. Camera fields never become formal required cache fields in this work. Missing, malformed, or numerically invalid camera data, and every evidence-only exception, produces evidence.status of unavailable_* or failed_* without entering the formal failure handler, clearing formal artifacts, or changing the formal result.

code/modules/da3_footprint_stage.py remains stage owner. code/utils/ground_stack_footprint.py owns pure geometry and strict-equivalence RANSAC optimization. A small mask-cache utility owns only cache keys, validation, locks, and atomic publication; it cannot decide which objects belong to the pile.

## Track A: strict-result support-plane performance

The RANSAC function keeps exact random seed, triplet order, threshold comparison, strict count-greater-than-best replacement, adaptive trial formula, 0.999 success probability, and 10000-trial cap.

It gains preallocated offsets of shape M by 3 and distances of shape M. NumPy out operations replace per-trial temporary allocation. Each support candidate retains one full_inlier_mask and reuses it for support-point and frame indexing.

The selector must not stop after its first eligible plane: later candidates can expose a similarly scored differently oriented plane and correctly reject an ambiguity. Only the _adaptive_ransac_plane trial loop for one candidate may return when count equals M for that candidate; select_support_plane must still remove and evaluate up to five candidates. Later trials of that one candidate cannot replace the count under strict comparison and later candidates use separate deterministic seeds.

For a fixed cache, dependency versions, NumPy thread configuration, dtype, point order, C layout, and current norm == 0 degeneracy rule, baseline and optimized code must match candidate by candidate: raw/refined point and normal, inlier counts, retained indices, every gate, score, selected index, accepted/rejected state, value_m2, GeoJSON, and top-down geometry. The sole additional diagnostics are trial_count and early_exit; their definitions are fixed, and every pre-existing diagnostic remains identical.

Tests cover perfect-plane early return, threshold plus/minus one ULP, tied candidates, all-five-candidate diagnostics, and wall-table ambiguity. Candidate-batched GEMM, altered RNG batching, lower trial budgets, and candidate pruning are forbidden because they can change threshold-edge inliers or ambiguity.

## Track B: persistent SAM3 mask cache

One immutable bundle represents all detections of one source frame:

save_root/dataset/sam3_mask_cache/v1/locks/key.lock
save_root/dataset/sam3_mask_cache/v1/entries/key/masks.npz
save_root/dataset/sam3_mask_cache/v1/entries/key/manifest.json
save_root/dataset/sam3_mask_cache/v1/corrupt/key.uuid

The SHA-256 key is SHA-256 of UTF-8 canonical JSON with lexicographically sorted keys and no whitespace. It covers cache schema, image ID, source image SHA-256 and size, the complete ordered detection list with object ID and exact canonical XYXY float representation, SAM3 checkpoint content SHA-256, SAM3 code and runtime fingerprints, the complete predict_inst prompt contract, and source-pixel boolean output shape/dtype. Each bbox coordinate must be finite binary64, with negative zero normalized to positive zero, and is encoded as its eight-byte IEEE-754 big-endian hex string. Non-finite coordinates reject the formal input before any cache operation.

The code fingerprint is canonical JSON of sorted relative paths and SHA-256 content digests for utils/sam3_utils.py, the mask-cache utility, and the stage call site. The runtime fingerprint records Python, NumPy, PyTorch, SAM3 package, CUDA, cuDNN, normalized device, precision/TF32/autocast, and deterministic-algorithm settings. The checkpoint digest is read immediately before and after model loading; a mismatch rejects the run and prevents cache publication. The in-process model key is checkpoint digest, normalized device, and inference-contract fingerprint.

The in-process SAM3 model cache must use checkpoint content digest in its key, not only a mutable checkpoint path. Global mapping and DA3 cache are deliberately absent from the mask key because masks depend on image, complete frame detections, and SAM3 contract. Existing complete DA3 and mapping validation still runs before every cache read.

A hit is valid only when manifest, payload digest, source image, ordered detections, checkpoint digest, code/runtime fingerprints, count, bool dtype, shape, per-mask digest, pixel count, and bbox clipping exactly match. Any mismatch is an audited miss, never a bbox fallback.

The writer holds a per-key fcntl.flock exclusive lock and rechecks after acquisition. It writes a same-filesystem temporary sibling under entries, fsyncs its files and directory, then publishes only to a nonexistent final path. If an existing final bundle is invalid, the writer first atomically renames the complete old directory to corrupt/key.uuid and fsyncs both entries and corrupt. It then renames the temporary sibling to the now-absent final key and fsyncs entries. The reader holds a shared lock, validates and loads completely before release. Invalid entries move to corrupt rather than being deleted.

A fresh complete SAM3 result may be measured after cache write failure but the report records cache_write_failed. SAM3 failure or cached empty/invalid object masks preserve rejected/null semantics. Each per-frame report entry has a nonempty ordered cache_events list; every event is exactly hit, miss, invalid, written, or cache_write_failed. A normal cold run records miss then written, a valid reuse records hit, a corrupt replacement records invalid then written, and a nonfatal write failure records miss then cache_write_failed.

Formal artifacts use immutable generations at save_root/dataset/ground_stack_footprint/runs/run_id containing the report, GeoJSON, PNG, and a complete manifest. After all files and the generation directory are fsynced, a single CURRENT manifest/pointer is atomically replaced. Public artifact readers resolve CURRENT before opening any artifact; a reader therefore gets one complete old generation or one complete new generation, never a three-file mixture. A failure before the CURRENT replace leaves the prior complete generation or no generation visible. A successful CURRENT replace is the logical publication point: a subsequent parent-directory fsync failure is reported as a durability warning but does not falsely report publication failure or attempt to roll a complete visible generation back. Tests inject each pre-replace boundary and the post-replace fsync boundary, then verify this contract.

The formal report records per-frame key, payload digest, checkpoint digest, code fingerprint, and exact cache_events. Tests cover hit/miss equivalence, every key invalidator, partial payloads, cache quarantine, concurrent readers/writers, empty cached masks rejecting the total, and concurrent formal artifacts not mixing.

## Track C: multi-view evidence in shadow mode

### Camera contract

When depth, intrinsic, and extrinsic exist, validate shape, finite values, positive focal lengths, invertible world-to-camera transforms, orthonormal rotation, and positive determinant. Rebuild source-frame world points from depth, intrinsics, and extrinsics and report source-frame reprojection error. This checks coordinate convention; it is not independent depth accuracy evidence.

If a schema-v2 cache lacks these fields, formal measurement preserves its current decision and evidence reports unavailable_missing_camera_fields. If these fields are malformed, non-finite, or an evidence algorithm throws, evidence reports failed_camera_contract or failed_evidence. None of these cases may change formal status, value, polygons, union, rejection reason, or formal artifacts. A separate approval is required before missing camera fields become a hard rejection.

### Bidirectional occlusion-aware reprojection

For each pair of observations of one global ID, deterministically sample confidence-qualified source-mask points and project with target extrinsic and intrinsic. Classify each point as behind camera, outside target grid, occluded because it is farther than target depth, visible-consistent, foreground-conflict because it is closer than target depth, visible mask-supported, or visible mask-unsupported.

Report both directions, eligible/occluded/conflict counts, visible mask support, and P50/P95 residuals. Initial depth tolerance is a diagnostic setting, not a rejection threshold. Occluded points are neutral; treating them as conflicts would reject correct stacked cartons.

### Per-object evidence and leave-one-out

Report source mask pixels, valid point count, confidence summary, elevated-point fraction, camera centre, viewing direction, and pairwise matrix per observation. A one-observation global ID is labelled single_observation_insufficient_cross_view_evidence, not falsely described as multi-view.

With full support plane fixed, omit each observation of one global ID once and recompute that ID OBB. Report polygon IoU, Hausdorff distance, centre/angle/side changes, and area delta. Full-view OBB and union remain the only formal result.

Optional shadow_full omits one source frame from the whole pipeline and reports support-plane normal, offset, and union change. It is not default because it remains expensive after mask caching. Both profiles produce empirical sensitivity, never a confidence interval.

A shadow mask robustness diagnostic runs documented one-pixel source-space erosion and dilation through existing warp, point, OBB, and union operations. It reports area interval and rejection transitions but never substitutes a perturbed mask for the formal mask.

### Promotion protocol

No shadow metric becomes a hard gate until threshold is locked on an independent calibration set and all tests below pass:

1. Pose injections of 0.25, 0.5, 1, and 2 degrees and 5, 10, and 20 mm worsen reprojection anomalies monotonically and identify the injected frame at the pre-registered operating point.
2. Assigning one observation mask or ID to an adjacent carton makes pairwise support and leave-one-out influence identify it as anomalous.
3. Duplicating one frame five times does not increase diversity/confidence and changes geometry only at numerical tolerance.
4. At least 20 independently ground-truthed stacks, each captured three times, use capture set A to lock thresholds and scale model and capture set B to measure accepted coverage, area error, and interval coverage. A scale reference cannot also verify that run.

DA3 depth/pose and upstream global mapping are not independent truth sources. Internal agreement cannot justify a calibrated m2 accuracy claim.

## Additive report schema

Primary report fields remain stable. New additive sections are performance with stage seconds and RANSAC diagnostics, sam3_mask_cache with per-frame provenance, and evidence with mode, camera contract, per-global-ID observations, numerical sensitivity, and empirical sensitivity.

No field authorizes accepted_with_warnings, partial totals, or a new unit. Formal status remains accepted or rejected, and metric remains da3_ground_footprint_union in m2.

## Implementation order

1. Add RANSAC instrumentation and strict-equivalence workspace optimization. Verify geometry tests, pre/post candidate diagnostics, and three fixed-input timing runs.
2. Add SAM3 cache utility, source-mask metadata, atomic output publication, README usage, corruption and concurrency tests. Verify GPU cache miss/hit equivalence on fd_area_test.
3. Add optional camera evidence loading and basic shadow reprojection, leave-one-out, and report sections with synthetic pose, mismatch, duplicated-view, and evidence-failure injections. For both accepted and rejected fixtures, formal JSON fields and GeoJSON geometry must be byte-identical or canonically equivalent with and without every evidence failure. Do not alter status.
4. Add optional full-frame leave-one-out and mask perturbation only after basic shadow results remain reproducible.
5. Perform independent calibration before proposing hard-gate promotion or an accuracy claim.

Use existing code/.venv through uv run --active --no-project. Stage only owned source, test, README, and specification files through an explicit allowlist. Never stage video inputs, detection JSON, cache payloads, runtime artifacts, or checkpoints.

## Source basis

- DA3 camera API: https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md
- Open3D registration evidence: https://open3d.org/docs/latest/tutorial/t_pipelines/t_icp_registration.html
- ODAM multi-view object mapping: https://openaccess.thecvf.com/content/ICCV2021/html/Li_ODAM_Object_Detection_Association_and_Mapping_Using_Posed_RGB_Video.html
- PMVC reprojection consistency: https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_PMVC_Promoting_Multi-View_Consistency_for_3D_Scene_Reconstruction_WACV_2024_paper.pdf
