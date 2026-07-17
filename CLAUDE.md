# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **multi-component workspace** (not a single Python package) for **3D SKU (商品) cross-image matching, deduplication, and global-ID counting** in retail shelf / floor-display scenes. Given multiple photos of the same scene plus per-image SKU detection boxes, the system determines which boxes are the *same physical object* across images, deduplicates them via union-find, and assigns a cross-image-unique `global_id` so each physical item is counted once.

The root has no top-level package code; it orchestrates independent sub-projects sharing one pipeline:

```
video -> frame_sampler (抽帧) -> images + per-image SKU detections
      -> code/ (3D reconstruction + SAM3 mask sampling + point matching -> correspondences + dedup + global_id)
```

Python **3.11** everywhere. **GPU (CUDA) is required** for matching/reconstruction. **`uv`** is the only Python tooling (see global rules). Top-level `README.md` describes the workspace; `code/` has its own README.

## Components

| Path | Role | Type |
|---|---|---|
| `code/` | Core R&D system: 3D SKU detection, matching, dedup, reconstruction, viewer. Own `pyproject.toml` + `.venv`. | Config-driven CLI (`main.py`, class `SKUDetectionMain`) |
| `frame_sampler/` | Standalone video frame-extraction FastAPI service (also CLI). | Docker (port 80) / CLI |
| `sam3/`, `Pi3/`, `Depth-Anything-3/` | Vendored model libs at root — **core source tracked in git** (env/weights/assets gitignored). | Vendored (tracked) |
| `vggt-main/` | VGGT model lib — fully gitignored (backend disabled); re-clone if needed. | Vendored (untracked) |
| `auto-research-loop/` | Autoresearch loop work state (`program*.md`, `progress*.md`, `results*.tsv`). Local-only, gitignored. | Work state |
| `imdata/` | Datasets: `floor_display1..12/`, each with `images/` + `detections_results/`. | Data |

> **Note**: Docker service wrappers (`Global-ID-Mapping/`, `Dockered_GlobalIDMapping/`, `docker_template/`) were removed in the 2026-07-16 repo cleanup (commit `9d9503f`). The canonical pipeline now lives only in `code/`; the old Docker BSON `/api` deployment is no longer in-tree.

### The three model libraries and their roles

All three are local source trees (not pip packages), injected onto `sys.path` by `code/`. Since 2026-07-17 their **core source is tracked in this repo** (nested `.git` removed; `.venv/`, `checkpoints/`, weights, `assets/`, `examples/` and other non-core content stay gitignored — weights must be re-fetched per each lib's README):

- **SAM3** (`sam3/`): segmentation. **Mask-guided point sampling** inside detection boxes (sample matching points from the object mask, not the whole bbox). Optional, gated by `inference.enable_sam3_mask_sampling`. Weights: local `sam3/checkpoints/sam3.pt`. Path injection: `utils/sam3_utils.py:_ensure_sam3_in_path()`. Entry: `sam3/inference.py` (standalone demo).
- **VGGT** (`vggt-main/`): 3D reconstruction, **real-time / flexible but slower** (re-infers every run). Produces point cloud + camera poses → `.glb`. Also used for 2D point-tracking matches. HF repo `facebook/VGGT-1B`. Path injection: `utils/__init__.py:_resolve_vggt_root()`. **Currently commented out in `modules/__init__.py`** — not the active backend.
- **Pi3** (`Pi3/`): 3D reconstruction, **precomputed-cache / fast / batch-friendly** — the **active backend**. Infers once, caches `pi3_cache/predictions.npz`; the matching stage then loads **no model** and reads depth/world_points/extrinsic/intrinsic from cache. HF repo `yyfz233/Pi3`. Path injection: `modules/pi3_3d_reconstructor.py:PI3_ROOT`. Entry: `Pi3/example.py`. Carries a **local patch** (`pi3/utils/basic.py`: `load_images_as_tensor` also accepts a list of image paths for frame-aligned loading) — preserved in-repo since 2026-07-17.
- **Depth-Anything-3** (`Depth-Anything-3/`): 3D reconstruction, **multi-view / higher-precision / subprocess-isolated**. DA3 depends on `omegaconf/addict/e3nn/evo` etc. not in `code/`'s venv (both code/ and DA3 use numpy<2 -- no numpy conflict), so `modules/da3_3d_reconstructor.py` runs it via **subprocess** invoking `Depth-Anything-3/.venv/bin/python modules/da3_runner.py` (self-contained, does not import `code/`). DA3 outputs depth+extrinsics(w2c)+intrinsics; `da3_runner.py` back-projects to `world_points` and writes `da3_cache/predictions.npz` (schema identical to Pi3). HF repo `depth-anything/DA3NESTED-GIANT-LARGE` (6.3GB, metric, **CC BY-NC 4.0**). Registered via `@register_reconstructor("da3")`.

`code/` selects backends via `--recon_backend` (reconstruction stage) and `--match_backend` (matching stage data source), each `vggt` | `pi3` | `da3`. New backends: subclass `ReconstructorBase` + `@register_reconstructor("<name>")` + import in `modules/__init__.py` (registry mechanism; CLI `choices` lists still need updating).

## Common Commands

### `code/` — the core system

Run from `code/` (own `.venv` + `pyproject.toml`).

```bash
cd code
uv sync                        # base deps
uv sync --extra dev            # + pytest/black/isort/flake8
uv sync --extra gpu            # + faiss-gpu / cupy (CUDA 12.x; see pyproject notes)
uv sync --extra rendering      # + nvdiffrast (mesh rendering, CUDA)

# Full pipeline on one dataset (PI3 backend, 3D matching — recommended)
uv run python main.py --mode pipeline --dataset ../imdata/floor_display2 \
    --algorithm 3d --match_backend pi3 --recon_backend pi3
# --floor N is a shortcut: --floor 2 == --dataset ../imdata/floor_display2

# Other modes
uv run python main.py --mode interactive   # menu-driven
uv run python main.py --mode reconstruct   # 3D reconstruction only -> .glb
uv run python main.py --mode viewer        # viser 3D viewer (default port 8080)
uv run python main.py --mode analyzer      # SKU count analysis
uv run python main.py --mode dedup         # cross-image dedup only
uv run python main.py --mode concise       # match + evaluate only

# Installed console script (same entry: main:main)
uv run sku-detection --mode pipeline --dataset ../imdata/floor_display2

# Run a single pipeline stage directly as a module
uv run python -m modules.inference --image_folder <path> --detection_dir <path> --algorithm 3d --backend pi3
uv run python -m modules.pi3_3d_reconstructor --input_dir <images> --output_file <out.glb>

# Batch over all floor_display datasets
bash batch_run_pipeline.sh         # PI3 3D over floor_display2..12
bash scripts/k.sh                          # batch run + batch accuracy eval (sets CUDA_VISIBLE_DEVICES=1)

# Accuracy vs human benchmark (imdata/picture_mapping_benchmark.csv)
uv run python accuracy_annotation.py
bash batch_accuracy_evaluation.sh floor_display2
```

**`--mode`**: `interactive` | `pipeline` | `concise` | `analyzer` | `dedup` | `reconstruct` | `viewer`.
**`--algorithm`**: `point_tracking` (2D feature-point trajectories, no 3D needed) | `3d` (3D-2D projection) | `both` (compare).
**`--match_backend` / `--recon_backend`**: `vggt` | `pi3`.
Defaults come from `code/config.yaml`; CLI overrides. `--config` selects an alternate YAML.

### Lint / format / test (`code/`)

```bash
cd code
uv run black .          # line-length 88, py38-py311
uv run isort .          # profile=black
uv run flake8 .
uv run pytest                          # all tests
uv run pytest test_api.py              # single file
uv run pytest test_api.py::test_name   # single test
```
Note: `code/test_api.py` is a BSON client for the now-removed Dockered service (`localhost:8000`, returns `detection_with_global_id`). The `Global-ID-Mapping/test_api.py` client is also gone (service removed in commit `9d9503f`). `code/test_api.py` is retained for reference only.

### Docker services

Only `frame_sampler/` remains as a Docker service (the Global-ID-Mapping service wrappers were removed in the 2026-07-16 cleanup, commit `9d9503f`).

```bash
# frame_sampler (CLI form)
cd frame_sampler && uv run python main.py video.mp4 --fps 2.0 --output custom_output/
```

| Entry | Port |
|---|---|
| `frame_sampler` Dockerfile | 80 |

## Architecture: `code/` (the core)

`code/main.py` (53k+) is a config-driven orchestrator. The `SKUDetectionMain` class parses `config.yaml` (sections: `main`, `reconstruction`, `inference`, `visualization`, `deduplication`, `accuracy`, `batch_accuracy`), wires colorlog logging (one `run_<timestamp>.log` per run under `save_root`), and dispatches by `--mode`.

```
code/
├── main.py                      # CLI entry + SKUDetectionMain + mode dispatch
├── config.yaml                  # all tunable params (Chinese-commented)
├── accuracy_annotation.py       # eval vs human benchmark CSV -> Precision/Recall/F1
├── aggregate_model_performance.py
├── modules/                     # pipeline stages
│   ├── inference.py             # SKU matching entry; builds SKUMatchingSystem -> process_images()
│   ├── reconstructor_base.py    # ReconstructorBase template (load_model->load_images->infer->export_glb->cache)
│   ├── pi3_3d_reconstructor.py  # Pi3 backend (active) — infers + caches predictions.npz
│   ├── da3_3d_reconstructor.py  # Depth-Anything-3 backend - subprocess to DA3 venv (dependency-set isolation)
│   ├── da3_runner.py            # DA3 inference script (runs in Depth-Anything-3/.venv; writes da3_cache npz)
│   ├── vggt_3d_reconstructor.py # VGGT backend (real-time) — currently disabled in __init__.py
│   ├── deduplicate_detections.py# sequential dedup + union-find global_id assignment
│   ├── improved_sku_analyzer.py # SKU count analysis (resolves one-to-many matches)
│   ├── draw_detection_boxes.py  # bbox visualization
│   └── viewer_runner.py         # viewer launcher
├── utils/                       # building blocks
│   ├── config.py                # SKUMatchingConfig dataclass + for_point_tracking()/for_3d_mapping() defaults
│   ├── sku_matching_system.py   # SKUMatchingSystem — end-to-end matching orchestration
│   ├── matching_algorithms.py   # find_correspondences_3d_mapping / _point_tracking
│   ├── sam3_utils.py            # SAM3 integration + mask-interior point sampling
│   ├── geometry_3d.py           # 3D sampling, 3D->2D projection, 3D match validation
│   ├── transforms.py            # VGGTImageTransform (518×518 crop) / Pi3ImageTransform (dynamic resize)
│   ├── global_id_mapper.py      # GlobalIDMapper — query global_mapping.json
│   ├── frame_alignment.py       # ReconstructionDetectionAligner — keep image/detection indices aligned
│   ├── nn_search.py, kdtree_utils.py   # KD-Tree / FAISS nearest-neighbour for point matching
│   ├── bbox_utils.py, bbox_3d_extractor.py, point_utils.py, mesh_utils.py
│   ├── data_utils.py, visualization.py, extract_frames.py, process_image_orientation.py
├── viewer/                      # Viser-based 3D viewer subsystem (runtime/datasource/indexer/id_assign/cache/types)
├── scripts/                     # shell batch/eval drivers
└── Output/                      # run outputs (per-dataset, per-backend)
```

### End-to-end data flow (`--mode pipeline`, 6 steps)

1. **Input** (`--dataset ../imdata/floor_displayN`): `images/<i>.JPG` (numbered) + `detections_results/<i>.json` — per-image boxes: `{skus:[{classes, objects:[{position:[x1,y1,x2,y2], confidences}]}]}`. `ReconstructionDetectionAligner` enforces image↔detection index alignment.
2. **3D reconstruction** (`modules/pi3_3d_reconstructor.py`): Pi3 infers over the image set → `Output/<dataset>/pi3_cache/predictions.npz` (depth, world_points, extrinsic, intrinsic, image_ids) + `reconstruction_pi3.glb`. **Skipped if the GLB/cache already exists** (reuse). VGGT path re-infers every run.
3. **Mask-guided sampling + matching** (`modules/inference.py` → `utils/sku_matching_system.py` → `matching_algorithms.py`): for each reference image, load 3D scene from cache; SAM3 (optional) generates a mask per ref bbox; sample 3D points from inside the mask (`sample_3d_points_from_mask`); project ref 3D points onto target images (`project_3d_to_2d`); find the target bbox each projection lands in (3D geometric validation + greedy uniqueness assignment with fallback: `find_best_matching_bbox_with_3d_validation` returns all validated candidates ranked by `combined_score`; `apply_uniqueness_constraint` assigns greedily so an evicted ref falls back to its next-best non-conflicting target box instead of being dropped). With `batch_all_refs=true` every image is used as reference in turn → `Output/<dataset>/output_3dmapping_<backend>/<ref>/matching_summary.txt` (+ `correspondences.json` when `save_json`). Gating: `pairing_3d: all|next`, `min_hit_ratio`, `confidence_threshold`, `min_confident_points`.
4. **Count analysis** (`modules/improved_sku_analyzer.py`): parse all `matching_summary.txt`, resolve one-to-many matches by double-filtering on ref+target → `output_reports/report_*.txt`.
5. **Dedup + global ID** (`modules/deduplicate_detections.py`): **union-find** clusters transitively-matched objects into connected components; each component = one `global_id` → `dedup_detections/{<i>.json, global_mapping.json, global_skus.json}`. Visualizations: `imgs_w_bboxes/`, `dedup_imgs_w_bboxes/`.
6. **Accuracy** (`accuracy_annotation.py`): compare `matching_summary.txt` against `imdata/picture_mapping_benchmark.csv` → `accuracy_evaluation/` (Precision/Recall/F1).

### Key design points

- **Pi3-first caching**: Pi3 infers once and caches `predictions.npz`; the matching stage loads **no model** (reads cache). VGGT re-infers each run. This is why Pi3 is preferred for batch.
- **Config override layers**: `SKUMatchingConfig` defaults (`for_point_tracking()` / `for_3d_mapping()`) ← `config.yaml` `inference:` section ← CLI args.
- **Model path injection**: VGGT/Pi3/SAM3 source trees must sit at the repo root (or the paths `code/` computes). They are added to `sys.path` at import time by the functions named above.
- **Union-find global ID**: cross-image matches cluster transitively; one connected component = one `global_id`.
- **Performance caching (da3 matching, 2026-07 speed opt)**: `--mode concise` batch_all_refs (N refs serial) was optimized 720s→180s (-75%, fd5/6/7, R/P equivalent via per-stage profiling `utils/profiling.py` + `--enable_profiling`). Bottleneck-located + fixed via 4 module-level read-only caches (pattern from `PI3_SCENE_CACHE`): `_DA3_IMAGE_CACHE` (image tensor), `_DA3_TRANSFORMS_CACHE` (transforms_info) in `utils/sku_matching_system.py`; `sam3_max_batch_size` config (5→32, avoids N×forward when >5 bbox/ref); removed per-ref `PI3_SCENE_CACHE.clear()` (scene_data read-only, reuse). Remaining bottleneck `sam3_mask` 64% is GPU compute-bound (single forward/ref, can't parallelize - CUDA serializes across threads). `--parallel_refs` is **ineffective** for SAM3 (GPU-bound) and breaks equivalence (RNG non-determinism). Enable profiling: `--enable_profiling` (zero-overhead no-op when off, byte-identical output verified).

## `Global-ID-Mapping/` service architecture (removed)

The `Global-ID-Mapping/` Docker service (and the older `Dockered_GlobalIDMapping/`) were removed in the 2026-07-16 repo cleanup (commit `9d9503f`). The canonical pipeline now lives only in `code/`. Historical reference: it wrapped a copy of `code/` and exposed the same pipeline over a BSON `/api` FastAPI endpoint (`uvicorn api:app`, port 80), with `processor.py` driving `SKUDetectionMain` in-process and returning `{"global_skus": [...]}`.

## Conventions

- `code/` is now the single canonical copy (the Docker-bundled `code/` copies were removed with the services in commit `9d9503f`).
- Vendored model libs (`sam3/`, `Pi3/`, `Depth-Anything-3/`) live at root; their core source is tracked in git (env/weights/assets gitignored) and root copies are canonical. `vggt-main/` is fully gitignored (disabled backend).
- Autoresearch loop state (`program*.md`, `progress*.md`, `results*.tsv`) lives in `auto-research-loop/` (gitignored, local-only); it was moved out of the repo root and untracked on 2026-07-17.
- `config.yaml` is the single source of tunable params for `code/`; `utils/config.py` holds `SKUMatchingConfig` defaults that the `inference:` section overrides.
- `imdata/floor_display*/` is the canonical benchmark dataset family; `picture_mapping_benchmark.csv` is the human-labeled ground truth for accuracy eval.
- Run logs: `code/main.py` writes one `run_<timestamp>.log` per run under `save_root` (default `Output/`).
