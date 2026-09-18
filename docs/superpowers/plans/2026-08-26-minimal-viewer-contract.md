# Minimal Viewer Product Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 发布精简 schema 3.0.0 Viewer bundle，并保留 Dataset、SKU/Global ID 选择、magenta、Focus 和 Selected Object。

**Architecture:** Python 仍负责 DA3 点云过滤、SAM3 label 传播和 global classification 聚合，但在发布前投影成极简 objects。TypeScript loader 使用固定文件名并只做 shape/range 检查；UI 与 scene 不再加载 footprint、thumbnail 或 evidence。

**Tech Stack:** Python 3.11/NumPy/uv、TypeScript、Three.js、Vitest、Vite。

**Spec:** `docs/superpowers/specs/2026-08-26-minimal-viewer-contract-design.md`

## Global Constraints

- Schema 固定为 `3.0.0`，旧 bundle 无 fallback。
- canonical other 精确为 `56642^其他品类`。
- 不新增测试文件，不提交 commit，不覆盖无关 dirty changes。
- Viewer 不发布或校验 confidence、hash、provenance、footprint、thumbnail 或 observation evidence。
- 必须保留数组 shape 与 point range 边界/不重叠检查。
- SKU group 与单 Global ID 共用现有 magenta 增量颜色更新，不复制 geometry。

---

### Task 1: Python aggregation and minimal exporter

**Files:**
- Modify: `utils/classification_aggregation.py`
- Modify: `tests/test_classification_aggregation.py`
- Modify: `src/web_viewer_export.py`
- Modify: `tests/test_web_viewer_export.py`
- Modify: `main.py`

**Interfaces:**
- Produces `export_web_viewer_bundle(..., dataset_name: str, ...)` without `footprint_root`.
- Publishes CURRENT with only `run_id`, minimal manifest, fixed binary files and minimal objects.

- [ ] Add existing-file tests where other confidence exceeds a non-other observation, but non-other remains first; cover all-other and multiple non-other.
- [ ] Run targeted aggregation tests and confirm the legacy confidence-only order fails.
- [ ] Implement an exact other sort prefix:

```python
OTHER_SKU = ("56642", "其他品类")

def candidate_sort_key(candidate):
    identity = (candidate["sku_id"], candidate["sku_name"])
    return (identity == OTHER_SKU, -candidate["confidence_sum"], -candidate["support_count"], -candidate["max_confidence"], *identity)
```

- [ ] Replace rich Viewer object publication with:

```python
{"ordered_skus": [{"sku_id": c["sku_id"], "sku_name": c["sku_name"]} for c in candidates],
 "point_ranges": non_empty_instance_ranges}
```

- [ ] Remove thumbnail generation, footprint loading, provenance/hash fields and duplicate aggregate validation from exporter; retain point labeling and atomic run publication.
- [ ] Update `main.py` to pass `dataset.name` and stop passing footprint input.
- [ ] Run existing aggregation and exporter tests with uv; confirm green.

### Task 2: Minimal TypeScript contract and loader

**Files:**
- Modify: `modules/viewer_web/src/contracts.ts`
- Modify: `modules/viewer_web/src/contracts.test.ts`
- Modify: `modules/viewer_web/src/bundle-loader.ts`
- Modify: `modules/viewer_web/src/bundle-loader.test.ts`
- Modify: `modules/viewer_web/src/sku-filters.ts`
- Modify: `modules/viewer_web/src/sku-filters.test.ts`

**Interfaces:**
- `ObjectIndexEntry = { ordered_skus: readonly OrderedSku[]; point_ranges: readonly PointRange[] }`.
- Loader returns manifest, objects, positions, colors, normals and derived `pointCount` only.

- [ ] Rewrite existing fixtures to the minimal schema and run focused Vitest tests to confirm the old validators fail.
- [ ] Implement minimal validators that read required fields but ignore unknown fields.
- [ ] Fetch fixed binary filenames, derive point count from positions and check color/normal shapes.
- [ ] Validate point ranges are bounded and globally non-overlapping; delete source/footprint/thumbnail/aggregate recomputation.
- [ ] Adapt SKU facets to `ordered_skus[0]` and preserve candidate order.
- [ ] Run contract, loader and SKU filter tests; confirm green.

### Task 3: Point-only scene and product UI

**Files:**
- Modify: `modules/viewer_web/src/main.ts`
- Modify: `modules/viewer_web/src/main.test.ts`
- Modify: `modules/viewer_web/src/scene.ts`
- Modify: `modules/viewer_web/src/scene.test.ts`
- Delete: `modules/viewer_web/src/footprints.ts`
- Modify: `modules/viewer_web/src/point-picking.ts`
- Modify: `modules/viewer_web/src/point-picking.test.ts`
- Modify: `modules/viewer_web/src/presentation.ts`
- Modify: `modules/viewer_web/src/presentation.test.ts`
- Modify: `modules/viewer_web/src/style.css`

**Interfaces:**
- Scene selection/focus/visibility consume `entry.point_ranges`.
- UI Selected Object consumes `entry.ordered_skus`.

- [ ] Adapt existing tests to the minimal object type and add state tests for SKU/global mutual exclusion and canvas pick switching.
- [ ] Run focused tests and confirm they fail against rich-object/footprint code.
- [ ] Remove footprint creation, raycast, opacity and focus branches; keep point picking and point-range focus.
- [ ] Remove Evidence drawer/summary/thumbnails and render `Selected Object` with Global ID plus ordered SKU names.
- [ ] Keep Dataset badge, selection mode buttons, independent scrolling and collapsed View Controls; remove footprint opacity and Backend/Footprint badges.
- [ ] Run all frontend tests and Vite build; confirm green.

### Task 4: Workflow, docs and fd6 publication

**Files:**
- Modify: `scripts/3d/pipeline/video_to_viewer.sh`
- Modify: `scripts/3d/pipeline/README.md`
- Modify: `README.md`
- Modify: `docs/3d_core.md`
- Modify: `modules/viewer_web/README.md`
- Modify: `modules/viewer_web/src/edl.ts` (remove obsolete footprint-only comment)

- [ ] Remove the mandatory footprint stage from video-to-viewer orchestration while leaving standalone `ground-stack-area` intact.
- [ ] Document minimal schema3, product UI, non-other priority and removed evidence/provenance.
- [ ] Run shell syntax, Python exporter/aggregation tests, full frontend tests, build and diff check.
- [ ] Re-export fd6 from the verified output root and confirm CURRENT points to a minimal schema3 run.
- [ ] Verify the running local Viewer returns HTTP 200 and hot-loads the new fd6 bundle.
