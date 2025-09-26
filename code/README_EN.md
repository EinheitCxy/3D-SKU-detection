# SKU Matching & Sequence Dedup (Modular)

This repository provides cross-image SKU matching on retail shelves and sequence-level deduplication, with a unified CLI, robust log parsing, and global ID aggregation.

## Project Structure

```
code/
├─ main.py                      # Unified CLI (interactive/pipeline/concise/analyzer/dedup)
├─ modules/                     # Executable / pipeline scripts
│  ├─ inference.py              # Matcher (point-tracking / 3D / both)
│  ├─ draw_detection_boxes.py   # Detection box visualization
│  ├─ improved_sku_analyzer.py  # One-to-one filtering (resolve many-to-one/one-to-many)
│  ├─ deduplicate_detections.py # Sequence dedup + global ID aggregation
│  ├─ analyze_accuracy_metrics.py# Batch metrics aggregator
│  └─ vggt_3d_reconstructor.py  # 3D reconstruction helper
├─ utils/                       # Reusable library modules
│  ├─ config.py, data_utils.py, transforms.py, point_utils.py
│  ├─ geometry_3d.py, matching_algorithms.py, visualization.py
│  ├─ sku_matching_system.py, bbox_utils.py
│  └─ process_image_orientation.py
├─ batch_run_inference.sh       # Batch run matcher across reference indices
├─ batch_accuracy_evaluation.sh # Batch accuracy evaluation
├─ output_viz/, output_logs/, output_dedup/  # Run artifacts
└─ README.md / README_EN.md
```

Sample datasets and results are expected under `imdata/` (not versioned). See `imdata/floor_display*/` structure (images + detections_results) in examples below.

## Quick Start

- Install deps (prefer uv):
  - At repo root: `uv pip install -r requirements.txt` (if present)
  - In `code/`: `uv sync`

- Interactive mode:
  - `uv run python code/main.py --mode interactive`

- Full pipeline (validate → visualize → match → analyze → dedup → evaluate):
  - `uv run python code/main.py --mode pipeline --dataset imdata/floor_display2 --save_root ./Output`

- Concise run (match only):
  - `uv run python code/main.py --mode concise --dataset imdata/floor_display2 --algorithm both --save_root ./Output`

- Analyzer only (one-to-one filtering report):
  - `uv run python code/main.py --mode analyzer --dataset imdata/floor_display2 --save_root ./Output`

- Dedup only (same-named outputs 1.json..X.json):
  - `uv run python code/main.py --mode dedup --dataset imdata/floor_display2 --save_root ./Output`

- Batch matching (reference idx 0..N):
  - `bash code/batch_run_inference.sh floor_display2 4`

Key passthrough params (to `modules/inference.py`):
```
uv run python code/main.py \
  --mode concise \
  --dataset imdata/floor_display2 \
  --algorithm both \
  --reference_idx 0 \
  --max_images 20 \
  --device cuda \
  --save_json \
  --save_root ./Output
```

## Outputs

- Matching summaries: `<dataset>/output_pt/<ref_idx>/`
- Visualization export: `code/output_viz/<dataset_name>/` (or `--save_root/output_viz/<dataset_name>/`)
- Analyzer reports: `code/output_reports/<dataset_name>/report_*.txt` (or `--save_root/output_reports/<dataset_name>/`)
- Sequence dedup (same-named): `<save_root>/<dataset_name>/1.json..X.json`
- Global IDs (union-find on matches): `<save_root>/<dataset_name>/global_mapping.json`

## Algorithms

1) Point tracking based SKU matching
   - Fast and memory-friendly, may degrade under large viewpoint changes
   - Output: `<dataset>/output_pt/<ref_idx>/`

2) 3D→2D projection matching
   - Uses VGGT depth/pose + geometry validation; stronger under large viewpoint changes
   - Output: `<dataset>/output_3dmapping/<ref_idx>/`

## Modules (utils/)

- `config.py`: runtime config and defaults
- `data_utils.py`: detection IO & normalization
- `transforms.py`: VGGT transforms
- `point_utils.py`: point sampling utilities
- `geometry_3d.py`: 3D geometry helpers
- `matching_algorithms.py`: matching core (requires VGGT)
- `visualization.py`: result visualization
- `sku_matching_system.py`: system wrapper (requires VGGT)
- `bbox_utils.py`, `process_image_orientation.py`

Check optional dependency availability:
```
from utils import check_dependencies
print(check_dependencies())  # {'vggt_modules': False, 'visualization': True}
```

## Testing

- Import check: `uv run python -c "from utils import SKUMatchingConfig; print('ok')"`
- Smoke test: `uv run python code/modules/inference.py --algorithm both --max_images 2`

## Notes

- Keep inputs under `imdata/`; outputs under `code/output*/` or the `--save_root` you provide.
- Avoid committing datasets or large binaries.
- Prefer `pathlib.Path`, keep configs out of the code, and use logging (no prints in libraries).

