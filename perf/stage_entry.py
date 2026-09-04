"""Subprocess entrypoint for benchmark stages that call the existing pipeline.

This adapter intentionally contains no reconstruction, matching, filtering, or
rendering logic. It isolates command construction and writes a small receipt so
the outer harness can measure each existing production boundary independently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol


class PipelineApp(Protocol):
    """The narrow SKUDetectionMain surface used by the performance harness."""

    def run_reconstruction(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    def run_sku_matching(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    def run_personalcare_classification(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]: ...

    def run_improved_sku_analysis(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]: ...

    def run_dedup_sequence(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


def dispatch_stage(
    *,
    stage: str,
    app: PipelineApp,
    dataset: Path,
    save_root: Path,
    viewer_output: Path | None = None,
    classification_result_path: Path | None = None,
    exporter: Callable[..., dict[str, Any]] | None = None,
    footprint_runner: Callable[[str, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one user-visible dependency of a new viewer bundle."""

    dataset_text = str(dataset)
    if stage == "classification":
        return app.run_personalcare_classification(dataset_text)
    if stage == "reconstruction":
        reconstruction_kwargs: dict[str, Any] = {"device": "cuda", "backend": "da3"}
        model_path = os.environ.get("DA3_MODEL_PATH")
        if model_path:
            reconstruction_kwargs["model_path"] = model_path
        return app.run_reconstruction(dataset_text, **reconstruction_kwargs)
    if stage == "matching":
        return app.run_sku_matching(
            dataset_text,
            "3d",
            batch_all_refs=True,
            backend="da3",
            enable_profiling=True,
        )
    if stage == "analysis_dedup":
        analysis = app.run_improved_sku_analysis(
            dataset_text, algorithm="3d", backend="da3"
        )
        if not analysis.get("success", False):
            return {"success": False, "analysis": analysis}
        detection_dir = _classification_detection_dir(classification_result_path)
        dedup = app.run_dedup_sequence(
            dataset_text,
            algorithm="3d",
            backend="da3",
            detection_dir=detection_dir,
        )
        return {
            "success": bool(dedup.get("success", False)),
            "analysis": analysis,
            "dedup": dedup,
        }
    if stage == "footprint":
        if footprint_runner is None:
            from src.da3_footprint_stage import run_da3_footprint

            footprint_runner = run_da3_footprint
        result = footprint_runner(dataset_text, save_root)
        status = result.get("status")
        report_path = result.get("report_path")
        if (
            status in {"accepted", "rejected"}
            and isinstance(report_path, str)
            and Path(report_path).is_file()
        ):
            return {
                **result,
                "success": True,
                "formal_status": status,
                "formal_success": bool(result.get("success", False)),
            }
        return result
    if stage == "viewer_export":
        if exporter is None:
            from src.web_viewer_export import export_web_viewer_bundle

            exporter = export_web_viewer_bundle

        dataset_output = save_root / dataset.name
        result = exporter(
            dataset_name=dataset.name,
            da3_cache_path=dataset_output / "da3_cache" / "predictions.npz",
            global_mapping_path=dataset_output
            / "dedup_detections"
            / "global_mapping.json",
            output_dir=viewer_output or save_root.parent / "viewer-data",
            source_images_dir=dataset / "images",
            sam3_mask_cache_root=dataset_output / "sam3_mask_cache" / "v2",
            voxel_size_m=0.005,
            max_points=1_500_000,
        )
        return {**result, "success": True}
    raise ValueError(f"unknown benchmark stage: {stage}")


def _classification_detection_dir(result_path: Path | None) -> Path:
    if result_path is None or not result_path.is_file():
        raise ValueError("classification stage result is missing")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = payload.get("result") if isinstance(payload, dict) else None
    detection_dir = result.get("detection_dir") if isinstance(result, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("success") is not True
        or not isinstance(result, dict)
        or result.get("success") is not True
        or not isinstance(detection_dir, str)
    ):
        raise ValueError("classification stage result is incomplete")
    path = Path(detection_dir)
    if not path.is_dir():
        raise ValueError(f"classified detection directory is missing: {path}")
    return path


def project_root_for_entry() -> Path:
    """Return the canonical root 3D core directory for this checkout."""

    return Path(__file__).resolve().parents[1]


def _build_app(save_root: Path) -> PipelineApp:
    project_root = project_root_for_entry()
    if not (project_root / "src").is_dir():
        raise RuntimeError(f"root src package is missing: {project_root / 'src'}")
    config_path = project_root / "config.yaml"
    if not config_path.is_file():
        raise RuntimeError(f"root config is missing: {config_path}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from main import SKUDetectionMain

    app = SKUDetectionMain()
    app.save_root = save_root
    app.match_backend = "da3"
    app.config_path = config_path
    return app


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one DA3 viewer benchmark stage")
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "classification",
            "reconstruction",
            "matching",
            "analysis_dedup",
            "footprint",
            "viewer_export",
        ),
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--save-root", required=True, type=Path)
    parser.add_argument("--viewer-output", required=True, type=Path)
    parser.add_argument("--classification-result", type=Path, default=None)
    parser.add_argument("--payload-path", required=True, type=Path)
    args = parser.parse_args()

    started_at = datetime.now(UTC).isoformat()
    try:
        result = dispatch_stage(
            stage=args.stage,
            app=_build_app(args.save_root),
            dataset=args.dataset,
            save_root=args.save_root,
            viewer_output=args.viewer_output,
            classification_result_path=args.classification_result,
        )
        success = bool(result.get("success", False))
        _write_payload(
            args.payload_path,
            {
                "stage": args.stage,
                "started_at": started_at,
                "completed_at": datetime.now(UTC).isoformat(),
                "success": success,
                "result": result,
            },
        )
    except Exception as error:
        _write_payload(
            args.payload_path,
            {
                "stage": args.stage,
                "started_at": started_at,
                "completed_at": datetime.now(UTC).isoformat(),
                "success": False,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
