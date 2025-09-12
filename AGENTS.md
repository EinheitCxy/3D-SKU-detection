# Repository Guidelines

This repository contains 3D shelf reconstruction and cross-image SKU matching. Core code lives in `sku_count/`; datasets and outputs are kept outside version control.

## Project Structure & Module Organization
- `sku_count/` – core Python code and CLI.
  - `inference.py` – main entry for matching workflows.
  - `module/` – reusable modules (`config.py`, `geometry_3d.py`, `matching_algorithms.py`, `visualization.py`, etc.). Add new logic here.
  - `output*/` – generated artifacts (ignored).
- `imdata/`, `imdata0911/` – sample/input images (ignored by Git).
- `sku_detection.json` – detection input format example.
- `ultralytics/`, `vggt-main/` – vendored third‑party code; avoid modifying unless absolutely necessary.

## Build, Test, and Development Commands
- Python 3.8–3.12. Prefer `uv`; `pip` also works.
- Install base deps (root): `uv pip install -r requirements.txt`
- Sync `sku_count` deps: `cd sku_count && uv sync` (uses `pyproject.toml`/`uv.lock`).
- Run a quick smoke: `cd sku_count && uv run python inference.py --algorithm both --max_images 2`
- Optional vendor tooling lives under `ultralytics/` and `vggt-main/` (manage in those folders if needed).

## Coding Style & Naming Conventions
- PEP 8, 4‑space indent, type hints required for new/edited functions.
- Names: files/modules `snake_case.py`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Keep orchestration in `inference.py`; put algorithms/utilities in `sku_count/module/`.
- Use `logging` (no print in libraries). Avoid hardcoded absolute paths; prefer `pathlib.Path`.
- No enforced linters; locally use Black (line length 100) and Ruff if available.

## Testing Guidelines
- Prefer `pytest`. Place tests in `sku_count/tests/` as `test_*.py`.
- Keep tests small and deterministic (`seed` in `SKUMatchingConfig`).
- Minimal checks:
  - Import: `uv run python -c "from module import SKUMatchingConfig; print('ok')"`
  - CLI: `uv run python inference.py --algorithm point_tracking --max_images 1 --save_json`

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise; optional scope, e.g., `sku_count: fix 3d thresholding`.
- PRs must include: summary, linked issues, reproduction command(s), before/after artifacts or counts, and notes on data used.
- Do not commit large binaries, datasets, or outputs (`*.glb`, `output*/`, `imdata/` are ignored). Do not modify vendored code unless required; prefer wrappers.

## Security & Configuration Tips
- Keep credentials and API keys out of code and Git.
- Pin dependencies when updating; edit root `requirements.txt` and/or `sku_count/pyproject.toml` together.
- Ensure all scripts write under `sku_count/output*/` and never mutate source data.

