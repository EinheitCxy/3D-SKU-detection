"""Subprocess entrypoint for benchmark stages that call the existing pipeline.

This adapter intentionally contains no reconstruction, matching, filtering, or
rendering logic. It isolates command construction and writes a small receipt so
the outer harness can measure each existing production boundary independently.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol


class PipelineApp(Protocol):
    """The narrow SKUDetectionMain surface used by the performance harness."""

    def run_reconstruction(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    def run_sku_matching(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

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
    exporter: Callable[..., dict[str, Any]] | None = None,
    footprint_runner: Callable[[str, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one user-visible dependency of a new viewer bundle."""

    dataset_text = str(dataset)
    if stage == "reconstruction":
        return app.run_reconstruction(dataset_text, device="cuda", backend="da3")
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
        dedup = app.run_dedup_sequence(dataset_text, algorithm="3d", backend="da3")
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
            da3_cache_path=dataset_output / "da3_cache" / "predictions.npz",
            global_mapping_path=dataset_output
            / "dedup_detections"
            / "global_mapping.json",
            footprint_root=dataset_output / "ground_stack_footprint",
            output_dir=viewer_output or save_root.parent / "viewer-data",
            source_images_dir=dataset / "images",
            sam3_mask_cache_root=dataset_output / "sam3_mask_cache" / "v1",
            voxel_size_m=0.005,
            max_points=1_500_000,
        )
        return {**result, "success": True}
    raise ValueError(f"unknown benchmark stage: {stage}")


def project_root_for_entry() -> Path:
    """Return the canonical root 3D core directory for this checkout."""

    return Path(__file__).resolve().parents[1]


def _build_app(save_root: Path) -> PipelineApp:
    project_root = project_root_for_entry()
    if not (project_root / "src").is_dir():
        raise RuntimeError(f"root src package is missing: {project_root / 'src'}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from main import SKUDetectionMain

    app = SKUDetectionMain()
    app.save_root = save_root
    app.match_backend = "da3"
    app.config_path = None
    return app


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one DA3 viewer benchmark stage")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("reconstruction", "matching", "analysis_dedup", "footprint", "viewer_export"),
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--save-root", required=True, type=Path)
    parser.add_argument("--viewer-output", required=True, type=Path)
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
