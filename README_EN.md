## 3D Shelf Reconstruction and Cross-Image SKU Matching

This repository implements a pipeline to reconstruct shelves in 3D from multiple images, map 2D detections into 3D, and analyze cross-image SKU correspondences. Utilities include visualization, accuracy analysis, and GLB exports.

### Architecture Overview
- Ingestion: numeric-named images and per-image detection JSONs.
- Preprocessing: `VGGTImageTransform` maps original coordinates to VGGT input space (resize/crop/pad) and back.
- Matching algorithms:
  - Point Tracking: track sampled points across frames; vectorized point-in-box voting; unique-target constraint.
  - 3D-2D Projection: sample 3D points, project with extrinsics/intrinsics, score with spatial/depth checks.
- Visualization: overlays and matching summaries; consistent per-object colors (local RNG, no global pollution).
- Export: 3D reconstruction via VGGT; GLB compatible with Blender/Three.js.

### Repository Layout
- `sku_count/` – core code and scripts
  - `module/`
    - `config.py` – device/dtype auto selection; `SKUMatchingConfig`
    - `data_utils.py` – detection loading (multi-format), bbox extraction
    - `transforms.py` – `VGGTImageTransform` mapping (final/original)
    - `geometry_3d.py` – 3D sampling, projection, geometric validation
    - `matching_algorithms.py` – point tracking + 3D-2D matching
    - `visualization.py` – drawing and summary writer
  - `inference.py` – matching entrypoint (point/3D/both)
  - `draw_detection_boxes.py` – draws bboxes (now reuses module APIs)
  - `vggt_3d_reconstructor.py` – VGGT-based GLB reconstruction
  - `sku_count_analyzer.py`, `improved_sku_analyzer.py` – analysis tools
- `ultralytics/`, `vggt-main/` – vendored code (do not modify unless necessary)
- `imdata/`, `imdata0911/` – sample/input datasets (ignored by Git)

### Setup
- Requirements: Python 3.8–3.12; optional CUDA GPU; `uv` recommended.
- Base deps: `uv pip install -r requirements.txt`
- Project env (sku_count): `cd sku_count && uv sync`

### Data & Formats
- Images: numeric filenames (e.g., `1.jpg`); non-numeric files are skipped.
- Detection JSONs (one per image index): supports any of:
  - `[{"classes": {...}, "objects": [...]}]` (list-of-one)
  - `{ "classes": {...}, "objects": [...] }` (dict)
  - `{ "skus": [{"classes": {...}, "objects": [...]}] }` (wrapper)
- Object entry fields:
  - `position: [x1, y1, x2, y2]`, `confidences.det: float`, `classes.det: int or label`.

### Usage
- Matching (point tracking + 3D projection):
  - `cd sku_count`
  - `uv run python inference.py --algorithm both \
      --image_folder ../imdata/floor_display2/images \
      --detection_dir ../imdata/floor_display2/detections_results \
      --output_dir ../imdata/floor_display2`
  - Common flags: `--max_points_per_bbox`, `--confidence_threshold`, `--min_confident_points`, `--save_json`
- Draw boxes (reuses data/visualization modules):
  - `uv run python draw_detection_boxes.py \
      --image_dir ../imdata/floor_display12/images \
      --detection_dir ../imdata/floor_display12/detections_results \
      --output_dir ../imdata/floor_display12/imdata_with_bbox \
      --confidence_threshold 0.3`
- 3D reconstruction (GLB):
  - `uv run python vggt_3d_reconstructor.py \
      --input_dir ../imdata/floor_display2/images \
      --output_file ../imdata/floor_display2/reconstruction.glb`
- Analyze results:
  - Count/uniqueness report: `uv run python sku_count_analyzer.py --floor_display floor_display2`
  - Filter one-to-many: `uv run python improved_sku_analyzer.py`

### Key Configuration (SKUMatchingConfig)
- Detection: `detection_confidence_threshold`, `min_bbox_area`, `max_bboxes`
- Point sampling: `max_points_per_bbox`, `max_total_points`
- Matching thresholds: `confidence_threshold`, `min_confident_points`, `correspondence_threshold`
- 3D options: `enable_3d_projection_matching`, `max_3d_distance`, `max_depth_difference`, `min_depth_consistency`
- System: `device='auto'`, `dtype (auto)`, `use_autocast`
- Output: `output_dir`, `save_json`, `json_filename`

### Performance & Precision
- Device/dtype auto: CUDA → bfloat16 (cap >= 8) or float16; CPU → float32
- AMP: enabled on CUDA if dtype ∈ {float16, bfloat16}
- Matching: point-in-box voting is vectorized (M×N bool mask). Example M=100, N=5000 uses < 2 MB.
- Tips: reduce `max_points_per_bbox`; filter with `detection_confidence_threshold` to shrink N and M.

### Outputs
- Visualizations and summaries under `output*/` within `output_dir`.
- Matching summary lines include the real `reference_image_idx`.
- GLB exports (e.g., `reconstruction.glb`) compatible with Blender/Three.js.

### Troubleshooting
- “No images matched with detection files”: ensure numeric filenames and matching JSON indices exist.
- VGGT import error: verify `vggt-main` exists at repo root; path injection is centralized in `module/__init__.py`.
- CUDA OOM: reduce `--max_images` or `--max_points_per_bbox`; AMP is enabled on supported GPUs.
- Empty detections: check JSON format and thresholds (`detection_confidence_threshold`, `min_bbox_area`).

### Contributing
- See `AGENTS.md` for repository guidelines (style, structure, PR requirements).
- Keep changes focused; avoid editing vendored code unless necessary.

### Notes
- Large datasets and binaries are ignored by `.gitignore`.
- For extremely large M×N, consider batching boxes/points to bound memory.
