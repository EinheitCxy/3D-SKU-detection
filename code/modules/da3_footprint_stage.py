"""Auditable DA3/SAM3 ground-support footprint measurement stage."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import shapely
import torch
from matplotlib import pyplot as plt
from PIL import Image
from shapely import set_precision
from shapely.geometry import mapping
from shapely.ops import unary_union

from utils.detection_objects import flatten_detection_objects
from utils.ground_stack_footprint import (
    FootprintError,
    SupportPlaneSelectionError,
    carton_footprint_polygon_from_projected,
    project_to_plane,
    select_support_plane,
    union_footprints,
    voxel_balance_projected,
)
from utils.sam3_mask_cache import (
    DetectionPrompt,
    FrameMaskCacheRequest,
    FrameMaskCacheResult,
    load_or_compute_frame_masks,
)
from utils.sam3_utils import (
    checkpoint_sha256,
    normalize_device,
    sam3_masks_from_bboxes_predict_inst,
)

_CACHE_FIELDS = {
    "world_points",
    "world_points_conf",
    "image_ids",
    "source_image_sizes",
    "source_to_processed_affine",
    "cache_schema_version",
    "source_model",
    "source_image_sha256",
    "affine_convention",
    "preprocess_resolution",
    "preprocess_method",
}
_MODEL_ID = re.compile(r"^[A-Za-z0-9._/-]+$")
_SAM3_CHECKPOINT = str(
    Path(__file__).resolve().parents[2] / "sam3" / "checkpoints" / "sam3.pt"
)  # 相对仓库根解析，不依赖 CWD（与 sam3_utils._ensure_sam3_in_path 一致）
_SAM3_DEVICE = "cuda"
_PATCH_SIZE = 14
_PREPROCESS_METHOD = "upper_bound_resize"
_PREDICT_INST_CONTRACT = {
    "api": "predict_inst",
    "builder": {"enable_inst_interactivity": True, "load_from_HF": False},
    "empty_mask_retry": "bbox_center_positive_point",
    "mask_postprocess": "best_iou_then_clip_to_bbox",
    "predict": {
        "multimask_output": True,
        "normalize_coords": True,
        "return_logits": False,
    },
    "processor": {"confidence_threshold": 0.0},
    "prompts": {"positive_exemplar": None, "text_prompt": None},
    "source_mask": {"coordinate_space": "source_pixels", "dtype": "bool"},
}
_GENERATION_ARTIFACTS = {
    "measurement_report": "measurement_report.json",
    "footprints_geojson": "footprints.geojson",
    "top_down_footprint_png": "top_down_footprint.png",
}
_GENERATION_FILES = frozenset({*_GENERATION_ARTIFACTS.values(), "manifest.json"})
_LOGGER = logging.getLogger(__name__)


class FootprintStageError(ValueError):
    """Raised for a strict input or measurement contract violation."""


def run_da3_footprint(dataset_path: str, save_root: Path) -> dict[str, object]:
    """Measure the union of every mapped carton's support-plane OBB footprint."""
    dataset = Path(dataset_path)
    output_dir = Path(save_root) / dataset.name / "ground_stack_footprint"
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "metric": "da3_ground_footprint_union",
        "unit": "m2",
        "status": "rejected",
        "value_m2": None,
        "cache": {},
        "sam3_mask_cache": {"cache_root": "sam3_mask_cache/v1", "frames": []},
        "plane": {"candidates": [], "selected": None},
        "per_global_id": {},
        "union": {},
        "library_versions": {"numpy": np.__version__, "shapely": shapely.__version__},
    }
    polygons: dict[str, Any] = {}
    union = None
    try:
        cache_path = Path(save_root) / dataset.name / "da3_cache" / "predictions.npz"
        mapping_path = Path(save_root) / dataset.name / "dedup_detections" / "global_mapping.json"
        cache = _load_cache(cache_path)
        image_paths = _image_paths(dataset / "images")
        detection_paths = _detection_paths(dataset / "detections_results")
        _validate_complete_source_ids(cache["image_ids"], image_paths, detection_paths)
        report["cache"] = _validate_cache(cache, image_paths)
        detections = _load_detections(detection_paths)
        global_mapping = _load_mapping(mapping_path, detections)
        sam3_checkpoint = Path(_SAM3_CHECKPOINT)
        sam3_checkpoint_sha256 = checkpoint_sha256(sam3_checkpoint)
        sam3_code_fingerprint = _sam3_code_fingerprint()
        sam3_runtime_fingerprint = _sam3_runtime_fingerprint(_SAM3_DEVICE)
        masked_observations, background_points, background_frames = _masked_observations(
            cache,
            image_paths,
            detections,
            global_mapping,
            report["per_global_id"],
            Path(save_root) / dataset.name / "sam3_mask_cache" / "v1",
            sam3_checkpoint,
            sam3_checkpoint_sha256,
            sam3_code_fingerprint,
            sam3_runtime_fingerprint,
            report["sam3_mask_cache"]["frames"],
        )
        if checkpoint_sha256(sam3_checkpoint) != sam3_checkpoint_sha256:
            raise FootprintStageError("SAM3 checkpoint changed during mask cache access")
        all_object_points = [
            points for observations in masked_observations.values() for points in observations if len(points)
        ]
        if not all_object_points:
            raise FootprintStageError("no mapped observations produced valid object points")
        try:
            plane, plane_diagnostics = select_support_plane(
                background_points, background_frames, all_object_points
            )
        except SupportPlaneSelectionError as error:
            report["plane"] = {"selected": None, **error.diagnostics}
            raise
        report["plane"] = {
            "selected": {
                "point": plane.point.tolist(),
                "normal": plane.normal.tolist(),
                "u_axis": plane.u_axis.tolist(),
                "v_axis": plane.v_axis.tolist(),
                "inlier_count": plane.inlier_count,
                "inlier_fraction": plane.inlier_fraction,
                "p95_residual_m": plane.p95_residual_m,
            },
            **plane_diagnostics,
        }
        for global_id, observation_points in masked_observations.items():
            per_id = report["per_global_id"][global_id]
            accepted = [points for points in observation_points if len(points) >= 32]
            if len(accepted) != len(observation_points):
                per_id["rejection"] = "observation has fewer than 32 valid masked points"
                continue
            fused = np.concatenate(accepted)
            heights = (fused - plane.point) @ plane.normal
            elevated = fused[heights > 0.015]
            if len(elevated) < 64:
                per_id["rejection"] = "fused global id has fewer than 64 valid points"
                continue
            projected = project_to_plane(elevated, plane)
            balanced_projected = voxel_balance_projected(projected, voxel_size_m=0.005)
            per_id["projected_voxel_point_count"] = int(len(balanced_projected))
            if len(balanced_projected) < 64:
                per_id["rejection"] = "fused global id has fewer than 64 projected voxel points"
                continue
            try:
                polygon, metrics = carton_footprint_polygon_from_projected(balanced_projected)
            except FootprintError as error:
                per_id["rejection"] = str(error)
                per_id["component_diagnostics"] = error.diagnostics
                continue
            per_id.update(metrics)
            per_id["height_median_m"] = float(np.median(heights))
            per_id["observations_used"] = len(accepted)
            polygons[global_id] = polygon
        missing = sorted(set(masked_observations) - set(polygons), key=_global_id_key)
        if missing:
            raise FootprintStageError("one or more global ids were rejected: " + ", ".join(missing))
        precise_polygons = {key: set_precision(polygon, 0.0001) for key, polygon in polygons.items()}
        union = union_footprints(list(precise_polygons.values()))
        coarse_union = unary_union([set_precision(polygon, 0.001) for polygon in polygons.values()])
        accepted_area = float(union.area)
        coarse_area = float(coarse_union.area)
        sensitivity = abs(accepted_area - coarse_area)
        tolerance = max(0.005 * accepted_area, 1e-4)
        report["union"] = {
            "polygon_count": len(precise_polygons),
            "precision_grid_m": 0.0001,
            "sensitivity_grid_m": 0.001,
            "area_0_1mm_m2": accepted_area,
            "area_1mm_m2": coarse_area,
            "sensitivity_abs_m2": sensitivity,
            "sensitivity_tolerance_m2": tolerance,
        }
        if sensitivity > tolerance:
            raise FootprintStageError("footprint union precision sensitivity exceeds tolerance")
        polygons = precise_polygons
        report["status"] = "accepted"
        report["value_m2"] = accepted_area
    except (FootprintStageError, FootprintError, OSError, ValueError, KeyError) as error:
        report["rejection_reason"] = str(error)
        polygons = {}
        union = None
    artifact_paths = _publish_generation(output_dir, report, polygons, union)
    report_path = Path(artifact_paths["measurement_report"])
    return {
        "success": report["status"] == "accepted",
        "status": report["status"],
        "report_path": str(report_path),
    }


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FootprintStageError(f"DA3 cache is missing: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        missing = sorted(_CACHE_FIELDS - set(loaded.files))
        if missing:
            raise FootprintStageError("DA3 cache missing required schema-v2 fields: " + ", ".join(missing))
        return {key: loaded[key].copy() for key in _CACHE_FIELDS}


def _validate_cache(cache: dict[str, np.ndarray], images: dict[int, Path]) -> dict[str, object]:
    schema = cache["cache_schema_version"]
    if schema.shape != () or int(schema) != 2:
        raise FootprintStageError("DA3 cache schema version must be exactly 2")
    model = _safe_text(cache["source_model"], "source_model")
    convention = _safe_text(cache["affine_convention"], "affine_convention")
    preprocess_resolution = _safe_scalar_integer(
        cache["preprocess_resolution"], "preprocess_resolution"
    )
    preprocess_method = _safe_text(cache["preprocess_method"], "preprocess_method")
    if not _MODEL_ID.fullmatch(model):
        raise FootprintStageError("DA3 cache source_model is unsafe")
    if convention != "pixel_center_v1":
        raise FootprintStageError("DA3 cache affine convention must be pixel_center_v1")
    if preprocess_resolution <= 0:
        raise FootprintStageError("DA3 cache preprocess_resolution must be positive")
    if preprocess_method != _PREPROCESS_METHOD:
        raise FootprintStageError("DA3 cache preprocess_method is unsupported")
    points = cache["world_points"]
    confidence = cache["world_points_conf"]
    image_ids = cache["image_ids"]
    sizes = cache["source_image_sizes"]
    affine = cache["source_to_processed_affine"]
    hashes = cache["source_image_sha256"]
    if points.ndim != 4 or points.shape[-1] != 3 or confidence.shape != points.shape[:3]:
        raise FootprintStageError("DA3 world point/cache confidence shapes are inconsistent")
    frame_count, height, width, _ = points.shape
    if image_ids.shape != (frame_count,) or image_ids.dtype.kind not in "iu":
        raise FootprintStageError("DA3 cache image_ids must be an integer vector")
    if len(set(int(value) for value in image_ids)) != frame_count:
        raise FootprintStageError("DA3 cache image_ids must be unique")
    if sizes.shape != (frame_count, 2) or affine.shape != (frame_count, 2, 3):
        raise FootprintStageError("DA3 cache source size or affine shape is invalid")
    if affine.dtype.kind not in "fiu":
        raise FootprintStageError("DA3 cache source_to_processed_affine dtype must be numeric")
    if sizes.dtype.kind not in "iu":
        raise FootprintStageError("DA3 cache source_image_sizes dtype must be integer")
    if hashes.shape != (frame_count,) or hashes.dtype.kind != "U":
        raise FootprintStageError("DA3 cache source_image_sha256 must be safe unicode")
    if not np.isfinite(affine).all() or np.any(sizes <= 0):
        raise FootprintStageError("DA3 cache source sizes or affine values are invalid")
    _validate_affine_linear_parts(affine)
    for index, image_id in enumerate(image_ids):
        image_id = int(image_id)
        if image_id not in images:
            raise FootprintStageError(f"image {image_id} is absent from current dataset")
        with Image.open(images[image_id]) as image:
            if tuple(sizes[index].tolist()) != image.size:
                raise FootprintStageError(f"source image size mismatch for image {image_id}")
        current_hash = _sha256(images[image_id])
        if str(hashes[index]) != current_hash:
            raise FootprintStageError(f"source image SHA256 mismatch for image {image_id}")
        expected_affine = _expected_pixel_center_affine(
            image.width, image.height, preprocess_resolution, height, width
        )
        if not np.allclose(affine[index], expected_affine, rtol=0.0, atol=1e-6):
            raise FootprintStageError(
                f"DA3 cache affine provenance mismatch for image {image_id}"
            )
    return {
        "schema_version": 2,
        "source_model": model,
        "affine_convention": convention,
        "preprocess_resolution": preprocess_resolution,
        "preprocess_method": preprocess_method,
        "frame_count": int(frame_count),
        "processed_size": [int(width), int(height)],
        "image_ids": [int(value) for value in image_ids],
        "source_image_sha256": [str(value) for value in hashes],
    }


def _image_paths(images_dir: Path) -> dict[int, Path]:
    if not images_dir.is_dir():
        raise FootprintStageError(f"images directory is missing: {images_dir}")
    paths: dict[int, Path] = {}
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} and path.stem.isdigit():
            image_id = int(path.stem)
            if image_id in paths:
                raise FootprintStageError(f"multiple source images have id {image_id}")
            paths[image_id] = path
    return paths


def _detection_paths(detections_dir: Path) -> dict[int, Path]:
    if not detections_dir.is_dir():
        raise FootprintStageError(f"detections directory is missing: {detections_dir}")
    paths: dict[int, Path] = {}
    for path in detections_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".json" and path.stem.isdigit():
            image_id = int(path.stem)
            if image_id in paths:
                raise FootprintStageError(f"multiple detection JSON files have id {image_id}")
            paths[image_id] = path
    return paths


def _validate_complete_source_ids(
    cache_ids: np.ndarray, image_paths: dict[int, Path], detection_paths: dict[int, Path]
) -> None:
    numeric_cache_ids = {int(value) for value in cache_ids}
    numeric_image_ids = set(image_paths)
    numeric_detection_ids = set(detection_paths)
    if numeric_cache_ids != numeric_image_ids:
        raise FootprintStageError(
            "numeric source image ids must exactly equal cache image ids "
            f"(cache={sorted(numeric_cache_ids)}, images={sorted(numeric_image_ids)})"
        )
    if numeric_cache_ids != numeric_detection_ids:
        raise FootprintStageError(
            "numeric detection JSON ids must exactly equal cache image ids "
            f"(cache={sorted(numeric_cache_ids)}, detections={sorted(numeric_detection_ids)})"
        )


def _load_detections(detection_paths: dict[int, Path]) -> dict[int, list[dict[str, Any]]]:
    detections: dict[int, list[dict[str, Any]]] = {}
    for image_id, path in sorted(detection_paths.items()):
        try:
            objects = flatten_detection_objects(json.loads(path.read_text()))
        except (json.JSONDecodeError, ValueError) as error:
            raise FootprintStageError(f"invalid detection JSON for image {image_id}: {error}") from error
        checked: list[dict[str, Any]] = []
        for object_id, obj in enumerate(objects):
            bbox = _bbox(obj.get("position", obj.get("bbox")), image_id, object_id)
            checked.append({"image_id": image_id, "object_id": object_id, "bbox": bbox})
        detections[image_id] = checked
    return detections


def _load_mapping(path: Path, detections: dict[int, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise FootprintStageError(f"global_mapping.json is missing: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise FootprintStageError(f"invalid global_mapping.json: {error}") from error
    if not isinstance(raw, dict) or not raw:
        raise FootprintStageError("global_mapping.json must be a nonempty object")
    expected = {(image_id, item["object_id"]): item for image_id, items in detections.items() for item in items}
    seen: set[tuple[int, int]] = set()
    mapping_by_id: dict[str, list[dict[str, Any]]] = {}
    for global_id, entries in raw.items():
        if not isinstance(entries, list) or not entries:
            raise FootprintStageError(f"global id {global_id!r} has no observations")
        string_id = str(global_id)
        checked: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise FootprintStageError(f"global id {string_id} has invalid observation")
            key = (entry.get("image_id"), entry.get("object_id"))
            if not all(isinstance(value, int) for value in key) or key not in expected or key in seen:
                raise FootprintStageError("mapping must contain each detection (frame, object_id) exactly once")
            bbox = _bbox(entry.get("bbox"), key[0], key[1])
            if not np.allclose(bbox, expected[key]["bbox"], rtol=0.0, atol=0.0):
                raise FootprintStageError(f"mapping bbox mismatch for image {key[0]} object {key[1]}")
            seen.add(key)
            checked.append({**expected[key], "global_id": string_id})
        mapping_by_id[string_id] = checked
    if seen != set(expected):
        raise FootprintStageError("mapping observations do not exactly equal the full detection collection")
    return mapping_by_id


def _masked_observations(
    cache: dict[str, np.ndarray],
    image_paths: dict[int, Path],
    detections: dict[int, list[dict[str, Any]]],
    mapping_by_id: dict[str, list[dict[str, Any]]],
    per_global_id: dict[str, Any],
    mask_cache_root: Path,
    sam3_checkpoint: Path,
    sam3_checkpoint_sha256: str,
    sam3_code_fingerprint: dict[str, object],
    sam3_runtime_fingerprint: dict[str, object],
    mask_cache_frames: list[dict[str, object]],
) -> tuple[dict[str, list[np.ndarray]], np.ndarray, np.ndarray]:
    point_clouds = cache["world_points"]
    confidence = cache["world_points_conf"]
    frame_for_id = {int(image_id): index for index, image_id in enumerate(cache["image_ids"])}
    observations = {global_id: [] for global_id in mapping_by_id}
    background_points: list[np.ndarray] = []
    background_frames: list[np.ndarray] = []
    lookup = {(item["image_id"], item["object_id"]): item["global_id"] for entries in mapping_by_id.values() for item in entries}
    for image_id, frame_detections in detections.items():
        frame_index = frame_for_id[image_id]
        height, width = point_clouds.shape[1:3]
        with Image.open(image_paths[image_id]) as image:
            source_image_hw = (image.height, image.width)
        request = FrameMaskCacheRequest(
            cache_root=mask_cache_root,
            image_id=image_id,
            image_path=image_paths[image_id],
            detections=tuple(
                DetectionPrompt.from_bbox(item["object_id"], item["bbox"])
                for item in frame_detections
            ),
            checkpoint_path=sam3_checkpoint,
            checkpoint_sha256=sam3_checkpoint_sha256,
            code_fingerprint=sam3_code_fingerprint,
            runtime_fingerprint=sam3_runtime_fingerprint,
            inference_contract=_PREDICT_INST_CONTRACT,
            output_shape_hw=source_image_hw,
        )
        try:
            cache_result = load_or_compute_frame_masks(
                request,
                compute_masks=lambda: _compute_verified_sam3_masks(
                    image_paths[image_id],
                    frame_detections,
                    sam3_checkpoint,
                    sam3_checkpoint_sha256,
                ),
            )
        except RuntimeError as error:
            raise FootprintStageError(f"SAM3 failed for image {image_id}: {error}") from error
        mask_cache_frames.append(_mask_cache_report_entry(image_id, cache_result))
        masks = cache_result.masks
        if len(masks) != len(frame_detections):
            raise FootprintStageError(f"SAM3 did not return one mask per detection for image {image_id}")
        all_masks = np.zeros((height, width), dtype=bool)
        for item, mask in zip(frame_detections, masks):
            source_mask = np.asarray(mask, dtype=bool)
            with Image.open(image_paths[image_id]) as image:
                if source_mask.shape != (image.height, image.width):
                    raise FootprintStageError(f"SAM3 mask source dimensions mismatch for image {image_id}")
            warped = _warp_mask_to_da3_grid(cache["source_to_processed_affine"][frame_index], source_mask, height, width)
            global_id = lookup[(image_id, item["object_id"])]
            diagnostic = {"image_id": image_id, "object_id": item["object_id"], "valid_point_count": 0}
            per_global_id.setdefault(global_id, {"observations": []})["observations"].append(diagnostic)
            if not warped.any():
                diagnostic["rejection"] = "warped SAM3 mask is empty"
                observations[global_id].append(np.empty((0, 3), dtype=float))
            else:
                valid = _valid_points(point_clouds[frame_index], confidence[frame_index]) & warped
                points = point_clouds[frame_index][valid]
                diagnostic["valid_point_count"] = int(len(points))
                if len(points) < 32:
                    diagnostic["rejection"] = "mask has fewer than 32 valid DA3 points"
                observations[global_id].append(points)
            all_masks |= cv2.dilate(warped.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool)
        valid_background = _valid_points(point_clouds[frame_index], confidence[frame_index]) & ~all_masks
        frame_background = point_clouds[frame_index][valid_background]
        background_points.append(frame_background)
        background_frames.append(np.full(len(frame_background), image_id, dtype=np.int32))
    return observations, np.concatenate(background_points), np.concatenate(background_frames)


def _warp_mask_to_da3_grid(affine: np.ndarray, mask: np.ndarray, height: int, width: int) -> np.ndarray:
    warped = cv2.warpAffine(mask.astype(np.uint8), affine.astype(np.float32), (width, height), flags=cv2.INTER_NEAREST)
    return warped.astype(bool)


def _valid_points(points: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) > 0) & np.isfinite(confidence) & (confidence >= 1.0)


def _compute_verified_sam3_masks(
    image_path: Path,
    frame_detections: list[dict[str, Any]],
    checkpoint: Path,
    expected_checkpoint_sha256: str,
) -> list[np.ndarray]:
    if checkpoint_sha256(checkpoint) != expected_checkpoint_sha256:
        raise RuntimeError("SAM3 checkpoint digest changed before mask production")
    masks = sam3_masks_from_bboxes_predict_inst(
        str(image_path),
        [item["bbox"] for item in frame_detections],
        str(checkpoint),
        _SAM3_DEVICE,
    )
    if checkpoint_sha256(checkpoint) != expected_checkpoint_sha256:
        raise RuntimeError("SAM3 checkpoint changed during mask production")
    return masks


def _mask_cache_report_entry(
    image_id: int, result: FrameMaskCacheResult
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "key": result.key,
        "events": list(result.events),
        "payload_sha256": result.payload_sha256,
        "checkpoint_sha256": result.checkpoint_sha256,
        "code_fingerprint": dict(result.code_fingerprint),
        "invalid_reason": result.invalid_reason,
    }


def _sam3_code_fingerprint() -> dict[str, object]:
    code_root = Path(__file__).resolve().parents[1]
    paths = (
        code_root / "modules" / "da3_footprint_stage.py",
        code_root / "utils" / "sam3_mask_cache.py",
        code_root / "utils" / "sam3_utils.py",
    )
    return {
        "algorithm": "sha256",
        "files": {
            path.relative_to(code_root).as_posix(): _sha256(path)
            for path in sorted(paths, key=lambda candidate: candidate.as_posix())
        },
    }


def _sam3_runtime_fingerprint(device: str) -> dict[str, object]:
    try:
        normalized_device = normalize_device(device)
    except RuntimeError as error:
        raise FootprintStageError(f"SAM3 runtime device is unavailable: {error}") from error
    sam3_init = Path(__file__).resolve().parents[2] / "sam3" / "sam3" / "__init__.py"
    version_match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        sam3_init.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if version_match is None:
        raise FootprintStageError("SAM3 package version is missing")
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "sam3": {
            "source": "local",
            "version": version_match.group(1),
            "init_sha256": _sha256(sam3_init),
        },
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": normalized_device,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32": {
            "cuda_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "autocast": {"enabled": False, "dtype": None},
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _write_artifacts(output_dir: Path, report: dict[str, Any], polygons: dict[str, Any], union: Any) -> None:
    geojson_path = output_dir / "footprints.geojson"
    figure_path = output_dir / "top_down_footprint.png"
    accepted = report["status"] == "accepted"
    features: list[dict[str, Any]] = []
    if accepted:
        for global_id, polygon in polygons.items():
            features.append({"type": "Feature", "geometry": mapping(polygon), "properties": {"coordinate_space": "local_support_plane_meters", "global_id": global_id, "area_m2": float(polygon.area), "observations_used": report["per_global_id"][global_id].get("observations_used", 0)}})
        if union is not None:
            features.append({"type": "Feature", "geometry": mapping(union), "properties": {"coordinate_space": "local_support_plane_meters", "global_id": "union", "area_m2": float(union.area)}})
    geojson_path.write_text(json.dumps({"type": "FeatureCollection", "coordinate_space": "local_support_plane_meters", "status": report["status"], "measurement_complete": accepted, "features": features}, indent=2) + "\n")
    figure, axis = plt.subplots(figsize=(6, 6))
    if accepted:
        for global_id, polygon in polygons.items():
            x_values, y_values = polygon.exterior.xy
            axis.plot(x_values, y_values, label=f"global id {global_id}")
        if union is not None:
            geometries = getattr(union, "geoms", [union])
            for geometry in geometries:
                x_values, y_values = geometry.exterior.xy
                axis.fill(x_values, y_values, alpha=0.2, color="black", label="union")
    else:
        axis.text(0.5, 0.5, "REJECTED\nmeasurement incomplete", ha="center", va="center", color="firebrick", transform=axis.transAxes, fontsize=16)
    axis.set_xlabel("support-plane u (m)")
    axis.set_ylabel("support-plane v (m)")
    axis.set_aspect("equal", adjustable="box")
    if accepted and polygons:
        axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced_generation(
    runs_root: Path,
    run_id: str,
    report: dict[str, Any],
    polygons: dict[str, Any],
    union: Any,
) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    _fsync_directory(runs_root.parent)
    generation = runs_root / run_id
    if generation.exists():
        raise FileExistsError(f"artifact generation already exists: {generation}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
    try:
        _write_artifacts(temporary, report, polygons, union)
        report_path = temporary / "measurement_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifact_names = tuple(_GENERATION_ARTIFACTS.values())
        for name in artifact_names:
            _fsync_file(temporary / name)
        manifest = {
            "complete": True,
            "run_id": run_id,
            "sha256": {name: _sha256(temporary / name) for name in artifact_names},
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _fsync_file(manifest_path)
        _fsync_directory(temporary)
        os.rename(temporary, generation)
        _fsync_directory(runs_root)
        return generation
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _atomic_replace_current(current_path: Path, pointer: dict[str, object]) -> None:
    payload = (json.dumps(pointer, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".CURRENT.", dir=current_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, current_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    try:
        _fsync_directory(current_path.parent)
    except OSError as error:
        _LOGGER.warning(
            "CURRENT was replaced with a complete generation, but directory durability "
            "could not be confirmed: %s",
            error,
        )


def _artifact_paths_from_current(output_root: Path) -> dict[str, str]:
    try:
        pointer = json.loads((output_root / "CURRENT").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OSError(f"cannot resolve ground-stack artifact CURRENT: {error}") from error
    run_id = pointer.get("run_id") if isinstance(pointer, dict) else None
    if (
        not isinstance(pointer, dict)
        or pointer.get("complete") is not True
        or not isinstance(run_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
    ):
        raise OSError("ground-stack artifact CURRENT is invalid")
    generation = output_root / "runs" / run_id
    if not generation.is_dir() or {path.name for path in generation.iterdir()} != _GENERATION_FILES:
        raise OSError("ground-stack artifact generation is incomplete")
    try:
        manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OSError(f"cannot read ground-stack artifact manifest: {error}") from error
    expected_sha256 = {
        name: _sha256(generation / name) for name in _GENERATION_ARTIFACTS.values()
    }
    if (
        not isinstance(manifest, dict)
        or manifest.get("complete") is not True
        or manifest.get("run_id") != run_id
        or manifest.get("sha256") != expected_sha256
    ):
        raise OSError("ground-stack artifact manifest validation failed")
    return {
        key: str(generation / relative_path)
        for key, relative_path in _GENERATION_ARTIFACTS.items()
    }


def _publish_generation(
    output_root: Path,
    report: dict[str, Any],
    polygons: dict[str, Any],
    union: Any,
) -> dict[str, str]:
    run_id = uuid.uuid4().hex
    report["artifacts"] = dict(_GENERATION_ARTIFACTS)
    _write_fsynced_generation(output_root / "runs", run_id, report, polygons, union)
    _atomic_replace_current(output_root / "CURRENT", {"run_id": run_id, "complete": True})
    return _artifact_paths_from_current(output_root)


def _bbox(value: Any, image_id: int, object_id: int) -> list[float]:
    if not isinstance(value, list) or len(value) != 4 or not all(isinstance(item, (int, float)) for item in value):
        raise FootprintStageError(f"invalid bbox for image {image_id} object {object_id}")
    x1, y1, x2, y2 = (float(item) for item in value)
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        raise FootprintStageError(f"nonpositive bbox for image {image_id} object {object_id}")
    return [x1, y1, x2, y2]


def _safe_text(value: np.ndarray, field: str) -> str:
    if value.shape != () or value.dtype.kind != "U":
        raise FootprintStageError(f"DA3 cache {field} must be a scalar safe unicode string")
    return str(value.item())


def _safe_scalar_integer(value: np.ndarray, field: str) -> int:
    if value.shape != () or value.dtype.kind not in "iu":
        raise FootprintStageError(f"DA3 cache {field} must be a scalar integer")
    return int(value.item())


def _validate_affine_linear_parts(affine: np.ndarray) -> None:
    linear = affine[:, :, :2]
    if not np.allclose(linear[:, 0, 1], 0.0, rtol=0.0, atol=1e-8) or not np.allclose(
        linear[:, 1, 0], 0.0, rtol=0.0, atol=1e-8
    ):
        raise FootprintStageError("DA3 cache affine must be axis aligned")
    scale_x = linear[:, 0, 0]
    scale_y = linear[:, 1, 1]
    if np.any(scale_x <= 0.0) or np.any(scale_y <= 0.0):
        raise FootprintStageError("DA3 cache affine scales must be positive")
    determinants = np.linalg.det(linear)
    if np.any(determinants <= 0.0):
        raise FootprintStageError("DA3 cache affine determinant must be positive")
    if any(np.linalg.matrix_rank(matrix) != 2 for matrix in linear):
        raise FootprintStageError("DA3 cache affine linear part must have rank two")


def _nearest_patch_multiple(value: int) -> int:
    down = (value // _PATCH_SIZE) * _PATCH_SIZE
    up = down + _PATCH_SIZE
    return max(1, up if abs(up - value) <= abs(value - down) else down)


def _expected_pixel_center_affine(
    original_width: int,
    original_height: int,
    preprocess_resolution: int,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    scale = preprocess_resolution / float(max(original_width, original_height))
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    processed_width = _nearest_patch_multiple(resized_width)
    processed_height = _nearest_patch_multiple(resized_height)
    if processed_width < output_width or processed_height < output_height:
        raise FootprintStageError(
            "DA3 cache preprocess dimensions are smaller than the final world grid"
        )
    crop_left = (processed_width - output_width) // 2
    crop_top = (processed_height - output_height) // 2
    scale_x = processed_width / float(original_width)
    scale_y = processed_height / float(original_height)
    return np.asarray(
        [
            [scale_x, 0.0, (scale_x - 1.0) / 2.0 - crop_left],
            [0.0, scale_y, (scale_y - 1.0) / 2.0 - crop_top],
        ],
        dtype=np.float64,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _global_id_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):020d}") if value.isdigit() else (1, value)
