"""Opt-in DA3 geometry refinement into a new pipeline output root."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import numpy as np

from da3_defaults import DEFAULT_PROCESS_RES, PREPROCESS_METHOD, validate_batch_grid
from src.da3_3d_reconstructor import _validate_da3_runner_cache
from src.da3_runner import _depth_to_world_points
from utils.geometry_refinement import refine_geometry
from utils.refinement_correspondences import (
    CorrespondenceExtractionError,
    extract_correspondences,
)


_FRAME_KEYS = {
    "depth",
    "depth_conf",
    "world_points_conf",
    "extrinsic",
    "intrinsic",
    "images",
    "image_ids",
    "source_image_sizes",
    "source_to_processed_affine",
    "source_image_sha256",
}
_ALIGNMENT_KEYS = {
    "frame_alignment_sorted_indices",
    "frame_alignment_map_keys",
    "frame_alignment_map_values",
}


@contextmanager
def _record_failure(report: dict, report_path: Path):
    """Persist the failing stage at the CLI boundary, then preserve its exception."""
    try:
        yield
    except Exception as error:
        report.update(accepted=False, reason=f"{type(error).__name__}: {error}")
        if isinstance(error, CorrespondenceExtractionError):
            report["correspondences"] = error.diagnostics
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        raise


def refine_cache(
    cache_path: Path,
    output_root: Path,
    *,
    mode: str = "pose",
    frame_count: int | None = None,
    pair_window: int = 2,
    max_nfev: int = 60,
    seed: int = 42,
) -> dict:
    """Write a report and publish a new cache only if held-out gates pass.

    A frame subset is taken from the existing joint prediction, not reinferred.
    The original cache and any previous pipeline output are never overwritten.
    """
    started = time.monotonic()
    cache_path = Path(cache_path).resolve()
    output_root = Path(output_root).resolve()
    _validate_da3_runner_cache(cache_path)
    if cache_path.parent.name != "da3_cache":
        raise ValueError("输入缓存必须位于 <数据集名>/da3_cache/predictions.npz")
    if output_root.exists():
        raise FileExistsError(
            f"输出根目录必须是新目录，避免混入旧匹配结果：{output_root}"
        )
    with np.load(cache_path, allow_pickle=False) as source:
        if (
            int(source["preprocess_resolution"]) != DEFAULT_PROCESS_RES
            or str(source["preprocess_method"]) != PREPROCESS_METHOD
        ):
            raise ValueError(
                "输入缓存预处理与当前 pipeline 默认不一致；下游会重建并覆盖优化，拒绝发布。"
            )
        validate_batch_grid(source["source_image_sizes"], DEFAULT_PROCESS_RES)
        original_ids = source["image_ids"]
        original_count = len(original_ids)
        count = original_count if frame_count is None else frame_count
        if count < 3 or count > original_count:
            raise ValueError(f"frame_count 必须在 3 到 {original_count} 之间")
        if len(np.unique(original_ids)) != original_count:
            raise ValueError("缓存 image_ids 必须唯一")
        arrays = {
            key: (
                source[key][:count].copy() if key in _FRAME_KEYS else source[key].copy()
            )
            for key in source.files
            if key != "world_points" and key not in _ALIGNMENT_KEYS
        }
    ids = arrays["image_ids"]
    arrays["frame_alignment_sorted_indices"] = np.argsort(ids)
    arrays["frame_alignment_map_keys"] = ids.copy()
    arrays["frame_alignment_map_values"] = np.arange(count, dtype=np.int32)
    extrinsics = arrays["extrinsic"]
    if extrinsics.shape[1:] == (4, 4):
        if not np.allclose(extrinsics[:, 3], [0, 0, 0, 1]):
            raise ValueError("4×4 外参的最后一行必须为 [0,0,0,1]")
        extrinsics = extrinsics[:, :3]
    dataset_dir = output_root / cache_path.parent.parent.name
    output_root.mkdir(parents=True, exist_ok=False)
    dataset_dir.mkdir()
    report_path = dataset_dir / "geometry_refinement.json"
    report = {
        "source_cache": str(cache_path),
        "mode": mode,
        "source_joint_frame_count": original_count,
        "selected_image_ids": ids.tolist(),
        "subset_of_existing_joint_prediction": count != original_count,
        "confidence_semantics": "原 DA3 置信度保留，未重新估计或校准",
        "scale_factor_semantics": "原 DA3 全局尺度元数据；逐帧修正见 optimization.correction.depth_scale",
        "stage": "correspondences",
        "accepted": False,
        "cache_published": False,
    }
    with _record_failure(report, report_path):
        match_started = time.monotonic()
        matches, diagnostics = extract_correspondences(
            arrays["images"],
            arrays["intrinsic"],
            pair_window=pair_window,
            seed=seed,
        )
        report["correspondences"] = diagnostics
        report["correspondence_seconds"] = time.monotonic() - match_started
        report["stage"] = "optimization"
        optimized_depth, optimized_extrinsics, optimization = refine_geometry(
            arrays["depth"][..., 0],
            arrays["intrinsic"],
            extrinsics,
            matches,
            mode=mode,
            max_nfev=max_nfev,
            seed=seed,
        )
        report["optimization"] = optimization
        report["accepted"] = bool(optimization["accepted"])
        report["stage"] = "heldout_validation"
        if report["accepted"]:
            report["stage"] = "cache_publication"
            arrays["depth"] = optimized_depth[..., None].astype(np.float32)
            arrays["extrinsic"] = optimized_extrinsics.astype(np.float32)
            arrays["world_points"] = _depth_to_world_points(
                arrays["depth"][..., 0],
                arrays["intrinsic"],
                arrays["extrinsic"],
            )
            if not np.isfinite(arrays["world_points"]).all():
                raise ValueError("优化后的 world_points 包含非有限值，拒绝发布")
            arrays["geometry_refinement"] = np.asarray(
                json.dumps(
                    {
                        "mode": mode,
                        "source_cache": str(cache_path),
                        "source_joint_frame_count": original_count,
                        "selected_image_ids": ids.tolist(),
                        "optimization": optimization,
                    },
                    ensure_ascii=False,
                )
            )
            output_dir = dataset_dir / "da3_cache"
            output_dir.mkdir()
            destination = output_dir / "predictions.npz"
            partial = destination.with_name("predictions.npz.partial")
            try:
                with partial.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                _validate_da3_runner_cache(partial)
                os.replace(partial, destination)
            finally:
                partial.unlink(missing_ok=True)
            report["cache_published"] = True
            report["output_cache"] = str(destination)
        report["stage"] = "complete" if report["cache_published"] else "rejected"
        report["elapsed_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DA3 位姿/深度尺度优化：输出到全新流程目录"
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("pose", "pose-scale"), default="pose")
    parser.add_argument(
        "--frame-count", type=int, help="仅优化联合缓存的前 N 帧；不重新推理"
    )
    parser.add_argument("--pair-window", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = refine_cache(
        args.cache,
        args.output_root,
        mode=args.mode,
        frame_count=args.frame_count,
        pair_window=args.pair_window,
        max_nfev=args.max_nfev,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("accepted", "cache_published", "elapsed_seconds")
            },
            ensure_ascii=False,
        )
    )
    if not result["accepted"]:
        print("留出验证未通过：仅保存诊断报告，未发布优化缓存。")
        return 2
    print(f"优化缓存：{result['output_cache']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
