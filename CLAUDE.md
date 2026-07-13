# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **multi-component workspace** (not a single Python package) for **3D SKU (商品) cross-image matching, deduplication, and global-ID counting** in retail shelf / floor-display scenes. Given multiple photos of the same scene plus per-image SKU detection boxes, the system determines which boxes are the *same physical object* across images, deduplicates them via union-find, and assigns a cross-image-unique `global_id` so each physical item is counted once.

The root has no top-level package code; it orchestrates independent sub-projects sharing one pipeline:

```
video -> frame_sampler (抽帧) -> images + per-image SKU detections
      -> code/ (3D reconstruction + SAM3 mask sampling + point matching -> correspondences + dedup + global_id)
      -> Global-ID-Mapping (Dockerized FastAPI exposing the same pipeline over a BSON /api)
```

Python **3.11** everywhere. **GPU (CUDA) is required** for matching/reconstruction. **`uv`** is the only Python tooling (see global rules). No top-level `README.md`; each sub-project has its own.

## Components

| Path | Role | Type |
|---|---|---|
| `code/` | Core R&D system: 3D SKU detection, matching, dedup, reconstruction, viewer. Own `pyproject.toml` + `.venv`. | Config-driven CLI (`main.py`, class `SKUDetectionMain`) |
| `Global-ID-Mapping/` | **Current** production FastAPI service. Wraps a **copy of `code/`** + vendored `sam3/`, `vggt-main/`, `Pi3/`. BSON `/api`. | Docker (image `global-id-mapping:3.1.0`, port 80→host 8011) |
| `Dockered_GlobalIDMapping/` | **Older** alternate build (VGGT+point_tracking, subprocess-driven). Kept as history. | Docker (port 8000, dev 9999) |
| `frame_sampler/` | Standalone video frame-extraction FastAPI service (also CLI). | Docker (port 80) / CLI |
| `docker_template/` | Template for new Docker services (mirrors Global-ID-Mapping layout). | Template |
| `sam3/`, `vggt-main/`, `Pi3/` | Vendored model libs at root. **Copied (not symlinked)** into service dirs for self-contained images. | Vendored |
| `imdata/` | Datasets: `floor_display1..12/`, each with `images/` + `detections_results/`. | Data |

### The three model libraries and their roles

All three are local source trees (not pip packages), injected onto `sys.path` by `code/`:

- **SAM3** (`sam3/`): segmentation. **Mask-guided point sampling** inside detection boxes (sample matching points from the object mask, not the whole bbox). Optional, gated by `inference.enable_sam3_mask_sampling`. Weights: local `sam3/checkpoints/sam3.pt`. Path injection: `utils/sam3_utils.py:_ensure_sam3_in_path()`. Entry: `sam3/inference.py` (standalone demo).
- **VGGT** (`vggt-main/`): 3D reconstruction, **real-time / flexible but slower** (re-infers every run). Produces point cloud + camera poses → `.glb`. Also used for 2D point-tracking matches. HF repo `facebook/VGGT-1B`. Path injection: `utils/__init__.py:_resolve_vggt_root()`. **Currently commented out in `modules/__init__.py`** — not the active backend.
- **Pi3** (`Pi3/`): 3D reconstruction, **precomputed-cache / fast / batch-friendly** — the **active backend**. Infers once, caches `pi3_cache/predictions.npz`; the matching stage then loads **no model** and reads depth/world_points/extrinsic/intrinsic from cache. HF repo `yyfz233/Pi3`. Path injection: `modules/pi3_3d_reconstructor.py:PI3_ROOT`. Entry: `Pi3/example.py`.
- **Depth-Anything-3** (`Depth-Anything-3/`): 3D reconstruction, **multi-view / higher-precision / subprocess-isolated**. DA3 requires `numpy<2` + `omegaconf/addict/e3nn`, conflicting with `code/`'s `numpy>=2` venv, so `modules/da3_3d_reconstructor.py` runs it via **subprocess** invoking `Depth-Anything-3/.venv/bin/python modules/da3_runner.py` (self-contained, does not import `code/`). DA3 outputs depth+extrinsics(w2c)+intrinsics; `da3_runner.py` back-projects to `world_points` and writes `da3_cache/predictions.npz` (schema identical to Pi3). HF repo `depth-anything/DA3NESTED-GIANT-LARGE` (6.3GB, metric, **CC BY-NC 4.0**). Registered via `@register_reconstructor("da3")`.

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
bash scripts/batch_run_pipeline.sh         # PI3 3D over floor_display2..12
bash scripts/k.sh                          # batch run + batch accuracy eval (sets CUDA_VISIBLE_DEVICES=1)

# Accuracy vs human benchmark (imdata/picture_mapping_benchmark.csv)
uv run python accuracy_annotation.py
bash scripts/batch_accuracy_evaluation.sh floor_display2
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
Note: `code/test_api.py` is a BSON client for the **old Dockered** service (`localhost:8000`, returns `detection_with_global_id`). The current-service client is `Global-ID-Mapping/test_api.py` (`localhost:8011`, returns `global_skus`).

### Docker services

Each has a `build.sh` that builds, tags, and pushes to the private harbor (`harbor-cn.lingmouai.com/asu/<service>:<edition>`); all require `--gpus all`.

```bash
# Global-ID-Mapping: build + run (host 8011 -> container 80) + push
cd Global-ID-Mapping && bash build.sh

# frame_sampler (CLI form)
cd frame_sampler && uv run python main.py video.mp4 --fps 2.0 --output custom_output/
```

**Port map (easy to get wrong):**

| Entry | Port |
|---|---|
| `Global-ID-Mapping/api.py` `__main__` (local dev) | 8010 |
| `Global-ID-Mapping/build.sh` host mapping | 8011 → container 80 |
| `Global-ID-Mapping/Dockerfile` CMD | 80 |
| `Global-ID-Mapping/test_api.py` target | 8011 |
| `Global-ID-Mapping/test_connection.py` target | 8010 |
| `Dockered_GlobalIDMapping/main.py` `__main__` | 9999 |
| `Dockered_GlobalIDMapping/Dockerfile` | 8000 |
| `frame_sampler` Dockerfile | 80 |

## Architecture: `code/` (the core)

`code/main.py` (53k+) is a config-driven orchestrator. The `SKUDetectionMain` class parses `config.yaml` (sections: `main`, `reconstruction`, `inference`, `visualization`, `deduplication`, `accuracy`, `batch_accuracy`), wires colorlog logging (one `run_<timestamp>.log` per run under `save_root`), and dispatches by `--mode`.

```
code/
├── main.py                      # CLI entry + SKUDetectionMain + mode dispatch
├── config.yaml                  # all tunable params (Chinese-commented)
├── accuracy_annotation.py       # eval vs human benchmark CSV -> Precision/Recall/F1
├── aggregate_model_performance.py
├── interactive_3d_viewer.py     # LEGACY standalone viewer (plotly/dash) — superseded by viewer/
├── modules/                     # pipeline stages
│   ├── inference.py             # SKU matching entry; builds SKUMatchingSystem -> process_images()
│   ├── reconstructor_base.py    # ReconstructorBase template (load_model->load_images->infer->export_glb->cache)
│   ├── pi3_3d_reconstructor.py  # Pi3 backend (active) — infers + caches predictions.npz
│   ├── da3_3d_reconstructor.py  # Depth-Anything-3 backend - subprocess to DA3 venv (numpy<2 isolation)
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
3. **Mask-guided sampling + matching** (`modules/inference.py` → `utils/sku_matching_system.py` → `matching_algorithms.py`): for each reference image, load 3D scene from cache; SAM3 (optional) generates a mask per ref bbox; sample 3D points from inside the mask (`sample_3d_points_from_mask`); project ref 3D points onto target images (`project_3d_to_2d`); find the target bbox each projection lands in (3D geometric validation + uniqueness). With `batch_all_refs=true` every image is used as reference in turn → `Output/<dataset>/output_3dmapping_<backend>/<ref>/matching_summary.txt` (+ `correspondences.json` when `save_json`). Gating: `pairing_3d: all|next`, `min_hit_ratio`, `confidence_threshold`, `min_confident_points`.
4. **Count analysis** (`modules/improved_sku_analyzer.py`): parse all `matching_summary.txt`, resolve one-to-many matches by double-filtering on ref+target → `output_reports/report_*.txt`.
5. **Dedup + global ID** (`modules/deduplicate_detections.py`): **union-find** clusters transitively-matched objects into connected components; each component = one `global_id` → `dedup_detections/{<i>.json, global_mapping.json, global_skus.json}`. Visualizations: `imgs_w_bboxes/`, `dedup_imgs_w_bboxes/`.
6. **Accuracy** (`accuracy_annotation.py`): compare `matching_summary.txt` against `imdata/picture_mapping_benchmark.csv` → `accuracy_evaluation/` (Precision/Recall/F1).

### Key design points

- **Pi3-first caching**: Pi3 infers once and caches `predictions.npz`; the matching stage loads **no model** (reads cache). VGGT re-infers each run. This is why Pi3 is preferred for batch.
- **Config override layers**: `SKUMatchingConfig` defaults (`for_point_tracking()` / `for_3d_mapping()`) ← `config.yaml` `inference:` section ← CLI args.
- **Model path injection**: VGGT/Pi3/SAM3 source trees must sit at the repo root (or the paths `code/` computes). They are added to `sys.path` at import time by the functions named above.
- **Union-find global ID**: cross-image matches cluster transitively; one connected component = one `global_id`.

## `Global-ID-Mapping/` service architecture

`processor.py` inserts `Global-ID-Mapping/code/` onto `sys.path` and imports `from main import SKUDetectionMain` — it **drives the same `code/` system in-process** (not via CLI, not via subprocess). `process(inputs)` runs three steps: `SKUDetectionMain.run_reconstruction(backend='pi3')` → `run_sku_matching(algorithm='3d', batch_all_refs=True, backend='pi3')` → `run_dedup_sequence()`, then reads `dedup_detections/global_skus.json` and returns `{"global_skus": [json_str, ...]}`.

`api.py` (FastAPI) exposes `POST /api` taking a **BSON** body (`{"images":[bytes...], "skus":[json_str...]}`) and returning BSON. The Dockerfile runs `uvicorn api:app` on port 80. Inputs may be a directory of per-image JSONs or an aggregate `skus.json` array.

**README vs. reality:** the README describes a *CLI workflow-node mode* (`docker run ... --images ... --skus ... --output ...`), but `build.sh` actually runs the **HTTP API service** (`uvicorn api:app`). The CLI args in the README correspond to `code/main.py` capabilities, not to `api.py`. The actual HTTP output is `dedup_detections/{global_skus.json, global_mapping.json}`, returned as `{"global_skus": [...]}` — not the `skus/` + `all_images_with_global_id.json` layout the README's CLI mode describes.

**Hardcoded paths:** `processor.py` hardcodes the Pi3 model path `/app/Pi3/checkpoints/snapshots/.../model.safetensors`; SAM3 path comes from `config.yaml` (`../sam3/checkpoints/sam3.pt`). Moving deployment locations requires updating these.

**Base image** is a private Harbor image (`harbor-cn.lingmouai.com/asu/pricetag_ocr_recognition:...`) with CUDA 12.1 + PyTorch — needs intranet access.

### `Global-ID-Mapping/` vs `Dockered_GlobalIDMapping/`

| | `Global-ID-Mapping/` (current, 3.1.0) | `Dockered_GlobalIDMapping/` (old, 1.0.0) |
|---|---|---|
| Entry | `api.py` | `main.py` |
| Backend | Pi3, `3d` algorithm | VGGT, `point_tracking` |
| processor→engine | **in-process import** of `SKUDetectionMain` | **subprocess** to `modules.inference` / `deduplicate_detections` |
| Bundled models | `Pi3/` + `sam3/` + `vggt-main/` | only `vggt-main/` (no Pi3/SAM3) |
| `code/` copy | full `code/` (incl. `main.py`) | only `modules/`+`utils/`+`viewer/`+`config.yaml` (no `main.py`) |
| Return shape | `{"global_skus": [...]}` | `{"detection_with_global_id":[...], "global_mapping":{...}}` |

Both speak BSON over HTTP `/api`, but return structures differ.

## Conventions

- **Two `code/` copies exist** (`code/` at root; `code/` inside `Global-ID-Mapping/` and `Dockered_GlobalIDMapping/`). When changing pipeline logic, decide whether to update R&D `code/` only or also sync the Docker-bundled copies. They are **not linked** — changes must be copied manually.
- **Vendored model libs are duplicated** (`sam3/`, `vggt-main/`, `Pi3/` at root AND inside service dirs). Root copies are canonical for local dev; service-internal copies are baked into images.
- `config.yaml` is the single source of tunable params for `code/`; `utils/config.py` holds `SKUMatchingConfig` defaults that the `inference:` section overrides.
- `imdata/floor_display*/` is the canonical benchmark dataset family; `picture_mapping_benchmark.csv` is the human-labeled ground truth for accuracy eval.
- Run logs: `code/main.py` writes one `run_<timestamp>.log` per run under `save_root` (default `Output/`).
