"""One isolated DA3 mapping pipeline invocation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import PROJECT_ROOT as MAIN_PROJECT_ROOT, SKUDetectionMain
from src.web_viewer_export import export_web_viewer_bundle

_REQUIRED_STAGES = (
    "validation",
    "reconstruction",
    "matching",
    "improved_analysis",
    "classification",
    "dedup",
)


class RequestRunnerError(RuntimeError):
    """A pipeline stage failed for the current request only."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def run_mapping_request(
    dataset_dir: Path,
    output_root: Path,
    viewer_root: Path,
    model_path: str,
) -> dict[str, object]:
    """Run a fresh external-classification DA3 pipeline and export its Viewer."""
    dataset_dir = Path(dataset_dir)
    output_root = Path(output_root)
    viewer_root = Path(viewer_root)

    app = SKUDetectionMain()
    app.save_root = output_root
    app.match_backend = "da3"
    app.classifier_enabled = False
    app.config_path = MAIN_PROJECT_ROOT / "config.yaml"

    summary = app.run_complete_pipeline(
        str(dataset_dir), algorithm="3d", model_path=model_path
    )
    _require_complete_summary(summary)

    dataset_output = output_root / dataset_dir.name
    try:
        export_web_viewer_bundle(
            dataset_name=dataset_dir.name,
            da3_cache_path=dataset_output / "da3_cache" / "predictions.npz",
            global_mapping_path=dataset_output
            / "dedup_detections"
            / "global_mapping.json",
            output_dir=viewer_root,
            source_images_dir=dataset_dir / "images",
            sam3_mask_cache_root=dataset_output / "sam3_mask_cache" / "v2",
        )
    except Exception as error:
        raise RequestRunnerError("viewer", str(error)) from error

    return {
        "summary": dict(summary),
        "global_skus_path": str(
            dataset_output / "dedup_detections" / "global_skus.json"
        ),
        "viewer_root": str(viewer_root),
    }


def _require_complete_summary(summary: Mapping[str, Any]) -> None:
    for stage in _REQUIRED_STAGES:
        if summary.get(stage) is not True:
            raise RequestRunnerError(stage, f"pipeline stage {stage} did not succeed")
