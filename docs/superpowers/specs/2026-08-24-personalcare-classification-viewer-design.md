# Personalcare classification and viewer integration design

## Status and objective

Rick approved this design in chat on 2026-08-24 with these final decisions:

- The personalcare classifier runs after detection and overlaps DA3 reconstruction and SKU matching.
- Every valid bbox receives normalized SKU ID, SKU name, and model confidence in an enriched detection JSON.
- Classification propagates into the final global mapping and viewer bundle.
- If observations under one `global_id` disagree, every distinct SKU candidate is retained and ordered by aggregate confidence.
- Viewer details show all ordered SKU candidates but do not display confidence values.
- Total and SKU counts assign a conflicting object only to its top-ranked candidate, so one physical object is counted once.
- Selected objects keep the existing point/footprint highlight behavior; SKU classes do not introduce new colors.
- The SKU master-data mapping function exists as a stable placeholder. V1 does not infer manufacturer, brand, or category from product names.
- V1 adds no classification hash, signature, encryption, or content fingerprint. Efficiency takes priority.

The end state is a race-free pipeline that produces one auditable classification per bbox, one ordered candidate list per physical object, and a viewer that supports product-point selection, SKU totals, SKU filtering, and selected-object details.

## Scope

V1 includes:

- A local, no-HTTP classifier CLI in `modules/personalcare_classifier`.
- One classifier process and one model load per dataset run.
- A derived enriched-detection artifact that preserves the original detection schema and object order.
- Concurrent classification and core reconstruction/matching.
- Classification-aware dedup/global mapping and viewer export.
- Global candidate aggregation and deterministic ranking.
- Point-cloud product picking, SKU totals/filtering, and selected-object SKU details.
- Disabled placeholders for manufacturer, brand, category, POSM, price tag, and empty position.
- Focused Python and TypeScript regression tests plus README updates.

V1 does not include:

- A SKU master-data file or heuristic parsing of manufacturer, brand, or category.
- POSM, price-tag, or empty-position detection.
- Confidence values in the viewer UI.
- SKU-specific colors or replacement of the current selection highlight.
- A network service, BSON API, compatibility wrapper, or legacy fallback.
- New hash, signature, encryption, checkpoint conversion, or classification-cache fingerprinting.
- Removal or redesign of the viewer's existing provenance checks.

## Current classifier contract

The recovered classifier emits a per-image vocabulary under `classes.cls`. Each value is a combined `sku_id^sku_name` label. Every bbox stores an integer `classes.cls` index into that vocabulary and a float `confidences.cls` value.

V1 preserves those raw classifier fields and adds a normalized object-level record so downstream code never needs to repeat vocabulary lookup or split `^` strings.

```json
{
  "position": [2036, 2472, 2754, 3442],
  "classes": {"det": 0, "cls": 0},
  "confidences": {"det": 0.93, "cls": 0.97},
  "classification": {
    "schema_version": "1.0.0",
    "source": "personalcare",
    "project_id": 51,
    "status": "resolved",
    "sku_id": "430085",
    "sku_name": "立白大师香氛梦幻格拉斯玫瑰洗衣液瓶装2000克",
    "confidence": 0.97,
    "metadata": {
      "status": "master_data_pending",
      "manufacturer": null,
      "brand": null,
      "category": null,
      "object_kind": null
    }
  }
}
```

`classification.status` is `resolved` for a valid prediction and `unavailable` for an invalid bbox that cannot be classified. Invalid bbox coordinates are reported and preserved; they do not cause a crop fallback. Missing images, image/detection index mismatch, model-load failure, or incomplete output fail the classification stage.

## SKU metadata placeholder

The stable mapping boundary accepts normalized classifier output and returns a typed result:

```python
lookup_sku_metadata(sku_id: str, sku_name: str) -> SkuMetadata
```

In V1 it always returns:

```json
{
  "status": "master_data_pending",
  "manufacturer": null,
  "brand": null,
  "category": null,
  "object_kind": null
}
```

The function performs no file lookup, network call, regular expression classification, name heuristic, fallback, or caching. When the authoritative mapping arrives, its implementation can change without changing detection, global, exporter, or viewer schemas.

## Pipeline architecture

The original `detections_results/` directory is immutable pipeline input. Classification writes a derived copy so matching never reads a file while another process rewrites it.

```text
detector completes
       |
dataset validation
       |
       +---------------- classifier subprocess ----------------+
       |  images + original detections                          |
       |  one model load, batched bbox inference                |
       |  enriched detection JSON in unique run directory       |
       |                                                        |
       +-- DA3 reconstruction/cache -- SKU matching -- analysis-+
                                                                |
                                                        join results
                                                                |
                                dedup reads enriched detections -+
                                                                |
                              global mapping -> footprint -> viewer
```

The classifier stage publishes under:

```text
<save_root>/<dataset>/personalcare_classification/
├── runs/<run_id>/detections/<frame>.json
├── runs/<run_id>/result.json
└── CURRENT
```

`run_id` is `<unix_time_ns>-<pid>`, not a content digest. The stage writes a unique temporary sibling directory, validates frame/object counts, renames it into `runs/<run_id>`, and atomically replaces the small `CURRENT` pointer. It does not hash, encrypt, fsync every file, or add a content-addressed cache. The pipeline uses the exact returned run path rather than re-resolving `CURRENT`, avoiding cross-run ambiguity.

The core orchestrator starts classification immediately after dataset validation. Reconstruction and matching continue from original bbox JSON. Before dedup, the orchestrator joins the classifier task and passes the enriched detection directory explicitly to dedup. Classification failure leaves completed reconstruction/matching artifacts available but stops dedup, global mapping, footprint, and viewer publication.

The maintained video workflow invokes this same root pipeline behavior after the detector. Direct `main.py --mode pipeline` runs also classify existing detections, so there is one implementation rather than a shell-only path.

## Runtime and performance contract

The classifier remains a separate `uv` project and subprocess. It does not run FastAPI or import model code into the root orchestrator. Model paths resolve from the classifier module, not the caller's working directory.

Performance rules:

- Load and decode the checkpoint once per classifier process.
- Keep one model instance for the entire dataset.
- Preserve batched crop inference; do not invoke the model once per bbox.
- Read each source image and detection JSON once.
- Preserve bbox order so no object remapping pass is needed.
- Build enriched JSON in memory and publish each completed file once.
- Do not calculate new hashes, encrypt fields, duplicate point geometry, or serialize classifier feature vectors.
- Do not retain the classifier's deep feature output because no approved consumer uses it.

GPU device selection is explicit. There is no automatic CPU fallback or alternate-model fallback. If the chosen classifier device cannot load or run the model, classification fails clearly. Resource validation must measure the real classifier plus DA3 overlap before claiming that concurrent execution is safe on a given GPU.

## Global aggregation

Every `global_mapping.json` observation retains its normalized classification record alongside `image_id`, `object_id`, `bbox`, and `removed`. Removed observations still contribute classification evidence because `removed` affects physical counting, not whether the crop is informative.

For each `global_id`, candidates are grouped by exact `(sku_id, sku_name)`. Candidate ranking uses:

1. Descending sum of observation confidence.
2. Descending supporting-observation count.
3. Descending maximum observation confidence.
4. Ascending `sku_id` and then `sku_name` as deterministic tie-breakers.

The global status is:

- `unavailable`: no valid classification observation.
- `resolved`: exactly one distinct SKU candidate.
- `conflict`: two or more distinct SKU candidates.

All candidates remain in ranked order. The first candidate is `primary` and is the only candidate used for SKU facet counts. Consequently each `global_id` contributes exactly one to Total and, when a primary candidate exists, exactly one to a SKU count. Lower-ranked conflicting candidates appear in selected-object details but do not inflate totals.

The viewer-facing object entry contains the ordered aggregate:

```json
{
  "classification": {
    "status": "conflict",
    "primary_sku_id": "430085",
    "candidates": [
      {
        "sku_id": "430085",
        "sku_name": "产品A",
        "confidence_sum": 1.82,
        "support_count": 2,
        "max_confidence": 0.94
      },
      {
        "sku_id": "428987",
        "sku_name": "产品B",
        "confidence_sum": 0.88,
        "support_count": 1,
        "max_confidence": 0.88
      }
    ],
    "metadata": {
      "status": "master_data_pending",
      "manufacturer": null,
      "brand": null,
      "category": null,
      "object_kind": null
    }
  }
}
```

Confidence fields remain in the machine contract for deterministic ordering and tests but are not rendered in the UI.

## Viewer behavior

The existing three-column layout remains:

- Left: Total, visible count, SKU facet counts, global-object list, and disabled future facets.
- Center: existing point cloud and footprint scene.
- Right: selected global ID, ordered SKU candidates, formal footprint evidence, and observation thumbnails.

V1 product picking adds the point cloud to the existing click-release path. The Three.js intersection point index is mapped to a `global_id` through existing continuous `point_index_range` values. Footprint picking remains available. Both paths call the existing `selectGlobalId()` flow.

Selection continues to use the current magenta point highlight and footprint emphasis. No classification palette, manufacturer color, or candidate-specific highlight is added.

SKU filtering uses only each object's primary candidate and changes both the visible global-ID list and scene visibility. A conflict object still shows every candidate after selection. Manufacturer, brand, and category controls display `主数据待接入` and are disabled. POSM, price tag, and empty position display `检测能力待接入` and are disabled.

Scene filtering does not duplicate positions, colors, normals, or point objects. The viewer creates one client-side byte visibility attribute aligned with the existing point arrays. Filtering updates the ranges owned by affected global IDs; the point shader discards hidden points, footprint meshes toggle their existing `visible` flag, and point picking ignores intersections belonging to hidden IDs. The visibility array is runtime UI state and is not added to the bundle.

The right drawer displays candidate SKU ID and SKU name in ranked order. It does not display confidence, confidence sum, support count, or max confidence.

## Error handling

- An invalid bbox receives `classification.status = unavailable` and an explicit reason; no alternate crop is synthesized.
- Missing or mismatched frame inputs fail classification before publication.
- Model or CUDA failure fails classification; no CPU fallback is attempted.
- Partial classifier output never becomes the dedup input.
- Global aggregation rejects malformed candidate fields, non-finite/out-of-range confidence, or a class index outside its vocabulary.
- Viewer bundle validation remains strict. Invalid classification schema fails bundle loading with a clear re-export message.
- Existing viewer provenance validation remains unchanged. This feature does not add another digest or encryption layer.

## Testing and validation

Python tests do not load the real model unless running the explicit GPU smoke.

Required focused coverage:

- Parse combined `sku_id^sku_name` labels and reject malformed/index-mismatched output.
- Preserve detection shape, object order, raw detector fields, and raw classifier fields.
- Handle valid, empty, and invalid bboxes without silent fallback.
- Verify the metadata placeholder always returns `master_data_pending` with null fields.
- Verify classification and reconstruction/matching overlap, and dedup waits for both.
- Verify partial classification never reaches dedup/global publication.
- Verify deterministic candidate aggregation and ordering under input permutation.
- Verify conflict candidates all survive while Total and SKU counts use only primary.
- Verify exporter and TypeScript exact-key validation for resolved, conflict, and unavailable objects.
- Verify point picking maps intersection indices to the correct global ID.
- Verify selection reuses current highlight colors and behavior.
- Verify SKU filters change list and scene visibility while disabled facets remain inert.
- Verify viewer details omit confidence values.

Validation commands:

```bash
uv run --offline pytest -q \
  tests/test_personalcare_classifier.py \
  tests/test_classification_aggregation.py \
  tests/test_main_pipeline.py \
  tests/test_web_viewer_export.py
(cd modules/viewer_web && npm test -- --run && npm run build)
bash -n modules/video_to_dedup/*.sh
```

The explicit real-model smoke uses the classifier module's `uv` environment and a small approved dataset/frame. It records classification completeness, latency, peak VRAM, and whether overlap with DA3 succeeds on the selected GPU. A model-loading smoke alone is not evidence that parallel execution is safe.

## Implementation ownership

After implementation planning:

- A Terra agent owns classifier CLI/runtime, derived artifacts, orchestration, and Python tests.
- A second Terra agent owns detection/global aggregation, exporter schema, and Python tests.
- A Luna agent owns TypeScript contracts, point picking, filtering, UI, and frontend tests.
- The coordinator owns schema integration, README changes, conflict review, focused/full verification, and final evidence.

Agents work on disjoint files and must preserve the active root-layout migration and all unrelated user changes. Production edits begin only after Rick reviews this written spec and approves the implementation plan transition.
