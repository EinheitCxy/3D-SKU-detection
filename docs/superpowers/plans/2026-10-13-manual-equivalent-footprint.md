# Manual-equivalent horizontal outlines Implementation Plan

> **For agentic workers:** Use subagent-driven-development. Read only your task brief and the shared contract below; coordinator owns integration and Git operations.

**Goal:** Implement an executable, explicitly uncalibrated automatic stack-outline candidate and Viewer manual/candidate comparison, preserving the confirmed outer-boundary area semantics.

**Architecture:** Independent metric raster geometry; a cache-only candidate stage reusing the existing DA3 input and support geometry; local JSON exchange with the Viewer. Existing production OBB metric is not silently replaced before human reference validation.

**Tech stack:** Existing NumPy/SciPy/Shapely, Python via uv, TypeScript/Three/Vitest. No new packages.

**Spec:** `docs/superpowers/specs/2026-10-13-manual-equivalent-footprint-design.md` (approved for implementation by Rick's 2026-10-13 instruction to invoke subagents and implement).

## Global constraints and common contract

- Worktree `/home/xingyu/3D_Recognization/runtime/worktrees/footprint-area-fix`, branch `feat/manual-equivalent-footprint`, baseline `278bf22`. Root dirty work and other worktrees are out of write scope.
- Workers are not alone. Preserve other changes; do not spawn subagents, commit, merge, push, install packages, run GPU inference, or modify original NPZ/masks/images. Coordinator handles local commits and reviews.
- Python: `UV_PROJECT_ENVIRONMENT=/home/xingyu/3D_Recognization/.venv UV_CACHE_DIR=/tmp/footprint-area-uv-cache uv run --no-sync python ...`. Tests use `-m pytest -q -p no:cacheprovider`. Use minimal relevant tests; existing 196-pass baseline applies to unchanged base.
- No actual human reference polygon exists yet. Require explicit CLI `gap_radius_m` and `max_hole_area_m2`; neither is a calibrated default. Candidate grid is 0.005 m by default. Production `value_m2` remains null in candidate artifacts, even when geometry is complete. This implements spec stages 1–3; production promotion requires future human-reference evidence.
- Projection must use the target Viewer's shared rotation; refined support normal controls height filtering only. No old per-ID OBB success gate, D1 coverage gate, 20% largest-component pruning, or OBB shadow evidence in candidate geometry.
- Local interchange JSON (manual and automatic) is `format: "viewer-horizontal-outlines", version: 1, coordinate_unit: "m", kind: "manual" | "candidate"`.
- `scene` contains exact target manifest `dataset_name`, `backend`, `frame_count`, `world_to_view` (16 row-major numbers), `display_bounds` (6 numbers). Compare descriptors and transform numerically with 1e-6 tolerance in Viewer; these are practical bundle checks, not proof of source identity. No new hashes.
- `regions` is a list of `{id: string, vertices_world: [number,number,number][], area_m2: number}`. Store an open vertex list: first vertex is not repeated at the end. Polygons must be simple, finite, coplanar with Viewer XZ and pairwise without positive-area overlap. Frontend recomputes areas instead of trusting input values; coordinate conversions apply row-major transform once.
- `total_area_m2` is the sum of represented non-overlapping region areas. Manual status `reference`, metric `manual_horizontal_outline`, `complete: true`; candidate status `candidate` (complete nonempty geometry) or `needs_review` (incomplete), with `value_m2: null`, metric `da3_manual_equivalent_stack_outline_candidate`, explicit parameters and diagnostics. `complete: false` plus `needs_review` means represented area is a partial subtotal, not whole-scene area. Rejected input has no meaningful candidate total. An empty incomplete candidate displays no valid region rather than suggesting measured zero area.
- Backend candidate consumes explicit `--viewer-manifest` so identity and projection match the target bundle. Verify DA3 dataset/backend/frame count and rotation against the cache's shared orientation; mismatches fail with a useful message to re-export the bundle. World vertices can be lifted onto a constant viewer Y plane; projection area is height-independent.
- All morphology/grid values and omitted components/ambiguous holes must be reported. Do not silently fill a large courtyard, manufacture depth from a line, discard a small real heap, or label candidate output accepted.

### Task 1: Pure metric outline geometry

**Ownership:** create `utils/stack_outline.py`, `test/test_stack_outline.py` only.

**Interfaces:** `OutlineOptions(*, gap_radius_m: float, max_hole_area_m2: float, grid_resolution_m: float = 0.005, max_grid_cells: int = 4_000_000)`; `extract_stack_outlines(points_xz: np.ndarray, *, options: OutlineOptions, frame_ids: np.ndarray | None = None) -> OutlineResult`. Result has `polygons: tuple[shapely.Polygon, ...]`, `diagnostics: dict`, `issues: tuple[str, ...]`; incomplete geometry reports issues. Python return objects need not themselves be JSON, but diagnostics must be serializable.

- [x] Implement targeted regression cases: filled 2×3 m rectangle/rotation; 3×3 L minus 2×2 corner ~5 m²; U entrance; small seam versus two heaps; small secondary heap; collinear facade and isolated point; newly sealed large hole; invalid numeric options and huge sparse extent bounded before allocation.
- [x] Rasterize with floor-based cell indices anchored deterministically in metric coordinates, padded for morphology. Preserve raw cell/frame diagnostics; no density threshold relative to largest heap. Remove no valid separate heap merely for being small.
- [x] Close with a Euclidean disk of explicit radius, then inspect complementary holes. Fill holes only up to explicit max_hole_area_m2. A larger hole makes its component unresolved (record why; do not flatten holes into accepted exteriors). A line/isolated point without 2D support is unresolved, not a synthetic-width rectangle. Retain other valid components and report partial status via issues.
- [x] Extract exact occupied-cell boundaries with vectorization suitable for millions of grid cells (row-run rectangle unions or a similarly bounded method; never one Shapely object per raw point). Return simple hole-free polygons; report positive-area coverage and removed/unresolved cells. Fix diagonal/pixel-centre conventions explicitly. No simplification that changes topology.
- [x] Tests must check geometry/area/topology, not just nonempty output. Allow a perimeter-times-cell-width discretization tolerance; show shrinking cell size reduces rectangle discretization error. Exercise frame IDs and deterministic ordering.
- [x] Run only new geometry test file; write report and hand off for task review.

### Task 2: Cache-only candidate entrypoint and interchange

**Ownership:** `src/da3_footprint_stage.py`, `utils/ground_stack_footprint.py` (minimal shared-input extraction), create `src/stack_outline_stage.py`, `utils/outline_exchange.py`, `src/stack_outline_compare.py`, `test/test_stack_outline_stage.py`, `test/test_outline_exchange.py`; targeted existing stage tests only if required by changed public helper.

**Interfaces:** consume Task 1. Provide executable `uv run --no-sync python -m src.stack_outline_stage --dataset PATH --save-root PATH --sam3-mask-cache-root PATH --viewer-manifest PATH --output PATH --gap-radius-m FLOAT --max-hole-area-m2 FLOAT [--grid-resolution-m FLOAT]`. No main.py/default-mode changes. Output the common JSON contract to explicit output path and a sibling top-down diagnostic PNG. Refuse overwriting an existing output unless an explicit overwrite option is supplied; production CURRENT untouched.

- [x] Extract the smallest shared DA3 loading/mask preparation seam, preserve existing production stage/report behavior, and avoid duplicate input-validation logic. Candidate skips memory-heavy EvidenceObservation copies when unnecessary.
- [x] Separate elevated world-point selection from plane projection; use all valid mapped observations, including weak observations, retaining frame labels. Support-height fitting may reuse refined plane/offsets; project candidates using the supplied manifest's rigid rotation only.
- [x] Validate the target manifest's metric/orientation and scene descriptors. World→viewer→world lift must roundtrip with nonzero translation/tilt and produce the same area in JS. Candidate coordinates must not accidentally receive centering twice.
- [x] Wire geometry, parameters, per-input counts, component issues, source-frame counts, complete/partial candidate statuses and JSON/PNG. No formal acceptance or OBB evidence. Domain/input failures explicit; programming KeyErrors propagate.
- [x] Put local interchange validation/conversion and candidate-vs-human polygon comparison in `utils/outline_exchange.py`. Add `python -m src.stack_outline_compare --candidate PATH --reference PATH --output PATH`. Require candidate/manual kinds and matching scene descriptors; compute count, relative area difference, union IoU and bidirectional sampled boundary-distance p95 with sampling spacing 0.01 m reported explicitly. Use full polygon boundary and sample by arclength; metric is sampled agreement, not physical accuracy. Test identical, translated equal-area, multiple heaps and wrong-scene cases. This CLI consumes saved references when the user has drawn them; synthetic fixtures are not human truth.
- [x] Smoke via existing metric fixtures; assert candidate does not call OBB, shadow-evidence builder or SAM model; test orientation mismatch, input failure, partial result and valid JSON structure. Cover one unchanged production smoke after preparation extraction, plus focused helpers' regressions.
- [x] Run relevant tests once after fixes; report commands/results and hand off for task review.

### Task 3: Viewer manual references and candidate overlay

**Ownership:** create `modules/viewer_web/src/footprint-comparison.ts`, `test/viewer_web/footprint-comparison.test.ts`; edit `modules/viewer_web/src/{area-measurement-panel.ts,scene.ts,style.css}` and `modules/viewer_web/README.md`. Modify `main.ts` and its focused bootstrap tests only for necessary controller wiring and explicit point-mode loading. Keep existing `AreaMeasurement` draft behavior.

**Interfaces:** common JSON contract; functions/types for local file validation, world/view conversion, saved manual regions, candidate import and controller callbacks. No bundle Manifest/ZIP/server protocol changes.

- [x] Validate all imported regions before replacing state; enforce metric/fitted/rigid availability, scene match, simple polygons, horizontal coplanarity, positive finite areas, declared/recomputed area agreement and no positive-area overlap. Preserve current state after failed import. No file-source HTML insertion.
- [x] Add explicit “保存当前货堆”, next/new draft behavior, clear saved references, manual JSON export/import and candidate JSON import. Multiple saved manual regions survive redraw/start/stop. Draft does not double-count into saved total. Release Blob URLs.
- [x] Overlay manual and candidate boundaries in distinct colors in measurement view. Imported candidates should be inspectable by entering measurement mode without clearing saved regions. Display per-region and total areas; always label automatic output candidate, and incomplete output partial subtotal with diagnostics. Never label geometry an accepted measurement.
- [x] Fix directly blocking bootstrap behavior: `render=points` is already accepted but still unconditionally requests Surfel assets. Load the explicit requested mode so fresh point-only CPU preview bundles can be opened; keep explicit surfel mode errors intact, no silent rendering fallback. Add one bootstrap regression.
- [x] Test tilted/nonzero-translation roundtrip, rectangle area parity, multiple regions, overlaps/nesting, wrong scene, malformed/forged area, invalid status, rejected file import preserving state, draft/save separation.
- [x] Run new pure-logic tests + existing area/scene tests, then one production frontend build. Document operation and limitations; report for task review.

### Task 4: Integrated validation and documentation

**Ownership:** coordinator, bounded experiment worker if warranted. New runtime receipts under `runtime/manual-outline-20261013/`, docs `docs/manual_stack_outline.md`; update design status and this plan. No unrelated README/config/camera edits.

- [x] Use synthetic backend output as a real frontend import fixture and verify Python/JS totals and transformed boundary alignment. Browser smoke should exercise save/export/import, two regions, candidate overlay and failure handling if available locally.
- [x] Run a bounded CPU replay on fd4/fd5 using full cache/mask inputs and correctly oriented newly exported local Viewer manifests as necessary. Freeze one explicitly illustrative parameter setting before results, do not tune to claim success. fd6–fd8 remain unscored without locked human reference boundaries.
- [x] Retain JSON, PNG, command/status, timings and failure/partial reasons. Clearly distinguish algorithm smoke from manual agreement and physical accuracy. No new inference or deployment.
- [x] Document executable commands, exact status semantics, examples, known missing human reference validation, and why production OBB was not promoted/replaced.
- [ ] Independent final code/spec/evidence review; root fixes and targeted retests, then concise handoff with runnable artifact links. No push/merge/deploy without a new request.

## Review focus

1. A closing-created hole must not silently turn a U entrance into filled floor area (Task 1).
2. A minority heap must remain represented or explicitly unresolved, never largest-component-pruned (Task 1).
3. Refined support normal must not change the shared Viewer projection (Task 2).
4. Forged/overlapping/manual imports must not corrupt a good state or inflate totals (Task 3).
5. Uncalibrated or incomplete area must remain visibly distinct from a formal whole-scene result (Tasks 2–4).
