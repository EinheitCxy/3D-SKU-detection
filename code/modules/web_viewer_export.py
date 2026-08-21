"""Strict schema-v3 DA3 to bundle-v1 static web-viewer exporter."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import struct
import tempfile
import uuid
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from modules.da3_footprint_stage import resolve_current_footprint_artifacts
from utils.da3_cache_validation import (
    integer_scalar,
    unicode_scalar,
    validate_affine_linear_parts,
)
from utils.global_id_mapper import GlobalIDMapper
from utils.global_object_index import build_global_object_index
from utils.pointcloud_filter import PointCloudFilterConfig, filter_scene_points

logger = logging.getLogger(__name__)

_BUNDLE_FILES = (
    "manifest.json",
    "positions.f32.bin",
    "colors.u8.bin",
    "normals.i8.bin",
    "objects.json",
    "footprints.json",
)
_REQUIRED_CACHE_FIELDS = frozenset(
    {
        "cache_schema_version",
        "source_model",
        "image_ids",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "source_image_sizes",
        "source_to_processed_affine",
        "source_image_sha256",
        "affine_convention",
        "preprocess_resolution",
        "preprocess_method",
        "is_metric",
        "scale_factor",
    }
)
_MODEL_ID = re.compile(r"^[A-Za-z0-9._/-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID_STEM = re.compile(r"(\d+)")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_THUMB_DIR = "thumbs"
_THUMB_LONG_EDGE = 256
_THUMB_PADDING = 0.10
_THUMB_JPEG_QUALITY = 85


class WebViewerExportError(ValueError):
    """Raised when an input cannot satisfy the strict bundle-v1 contract."""


def export_web_viewer_bundle(
    *,
    da3_cache_path: Path,
    global_mapping_path: Path,
    footprint_root: Path,
    output_dir: Path,
    source_images_dir: Path,
    sam3_mask_cache_root: Path,
    voxel_size_m: float = 0.01,
    max_points: int = 500_000,
    filter_config: PointCloudFilterConfig | None = None,
) -> dict[str, object]:
    """Export verified formal artifacts and sampled DA3 points as bundle-v1."""
    voxel_size = _validate_export_options(voxel_size_m, max_points)
    actual_filter_config = filter_config or PointCloudFilterConfig()
    cache = _load_da3_cache(Path(da3_cache_path))
    artifacts = resolve_current_footprint_artifacts(Path(footprint_root))
    footprint, report_cache, expected_mapping_sha256 = _load_footprint(artifacts)
    if report_cache != cache["provenance"]:
        raise WebViewerExportError(
            "DA3 cache provenance does not match formal footprint report"
        )
    mapping_path = Path(global_mapping_path)
    mapping_sha256_before = _mapping_sha256(mapping_path)
    objects = build_global_object_index(GlobalIDMapper(str(mapping_path)))
    _validate_object_index_for_export(objects)
    mapping_sha256_after = _mapping_sha256(mapping_path)
    if (
        mapping_sha256_before != expected_mapping_sha256
        or mapping_sha256_after != expected_mapping_sha256
    ):
        raise WebViewerExportError(
            "global mapping changed or does not match formal footprint report"
        )
    if footprint["status"] == "accepted" and set(objects) != set(
        footprint["per_global_id"]
    ):
        raise WebViewerExportError(
            "accepted formal footprint object-index and geometry ID sets must match"
        )
    images_by_id = _resolve_source_images(Path(source_images_dir), cache)
    thumbs = _generate_thumbnails(objects, images_by_id)
    sampled = _sample_points(
        cache,
        objects,
        mask_cache_root=Path(sam3_mask_cache_root),
        voxel_size=voxel_size,
        max_points=max_points,
        filter_config=actual_filter_config,
    )
    _attach_point_index_ranges(
        objects, sampled["instance_labels"], sampled["label_keys"]
    )
    world_to_view = _compute_world_to_view(
        sampled["filtered_points"],
        sampled["level_rotation"],
    )
    output_path = Path(output_dir)

    manifest = {
        "schema_version": "1.0.0",
        "coordinate_space": "da3_world_meters",
        "point_count": int(len(sampled["positions"])),
        "display_bounds": _robust_display_bounds(sampled["positions"]),
        "arrays": {
            "positions": {
                "path": "positions.f32.bin",
                "dtype": "float32",
                "components": 3,
                "byte_length": int(sampled["positions"].nbytes),
            },
            "colors": {
                "path": "colors.u8.bin",
                "dtype": "uint8",
                "components": 3,
                "byte_length": int(sampled["colors"].nbytes),
            },
            "normals": {
                "path": "normals.i8.bin",
                "dtype": "int8",
                "components": 3,
                "byte_length": int(sampled["normals"].nbytes),
            },
        },
        "world_to_view": [float(value) for value in world_to_view.reshape(-1)],
        "coordinate_convention": (
            "bundle arrays store DA3 native OpenCV world coordinates "
            "(x-right, y-down, z-forward, first-camera anchored); world_to_view is a "
            "row-major 4x4 matrix mapping them into the viewer Y-up frame "
            "(CV->glTF axis flip, RANSAC ground leveling, per-axis median centering)"
        ),
        "objects_path": "objects.json",
        "footprints_path": "footprints.json",
        "source": {
            "da3_cache": {
                **cache["provenance"],
            },
            "footprint": {
                "run_id": footprint["run_id"],
                "status": footprint["status"],
            },
            "export": {
                "voxel_size_m": voxel_size,
                "max_points": max_points,
                "filter_config": asdict(actual_filter_config),
                "exporter_source_sha256": _file_sha256(Path(__file__)),
                "global_mapping_sha256": mapping_sha256_after,
                "sam3_mask_entries": sampled["sam3_mask_entries"],
            },
        },
        "capabilities": {
            "point_picking": False,
            "footprint_picking": True,
            "formal_ground_footprint": True,
        },
    }
    generation = _publish_bundle(
        output_path, manifest, objects, footprint, sampled, thumbs
    )
    return {
        "output_dir": str(output_path),
        "manifest_path": str(generation / "manifest.json"),
        "point_count": int(len(sampled["positions"])),
        "footprint_status": footprint["status"],
        "thumbnail_count": len(thumbs),
    }


def _validate_export_options(voxel_size_m: float, max_points: int) -> float:
    if isinstance(voxel_size_m, bool):
        raise WebViewerExportError("voxel_size_m must be a positive finite number")
    try:
        voxel_size = float(voxel_size_m)
    except (TypeError, ValueError) as error:
        raise WebViewerExportError(
            "voxel_size_m must be a positive finite number"
        ) from error
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise WebViewerExportError("voxel_size_m must be a positive finite number")
    if (
        isinstance(max_points, bool)
        or not isinstance(max_points, int)
        or max_points <= 0
    ):
        raise WebViewerExportError("max_points must be a positive integer")
    return voxel_size


def _validate_object_index_for_export(objects: dict[str, Any]) -> None:
    """Reject mapping-derived objects that the strict browser contract would reject."""
    for global_id, entry in objects.items():
        if not isinstance(global_id, str) or _GLOBAL_ID.fullmatch(global_id) is None or not isinstance(entry, dict):
            raise WebViewerExportError("object index global ID is invalid")
        instances = entry.get("instances")
        if not isinstance(instances, list):
            raise WebViewerExportError("object index instances are invalid")
        active = removed = 0
        images: set[int] = set()
        object_ids: list[int] = []
        for instance in instances:
            if not isinstance(instance, dict): raise WebViewerExportError("object index instance is invalid")
            image_id, object_id, bbox, is_removed = instance.get("image_id"), instance.get("object_id"), instance.get("bbox"), instance.get("removed")
            if isinstance(image_id, bool) or not isinstance(image_id, int) or abs(image_id) > 2**53 - 1 or isinstance(object_id, bool) or not isinstance(object_id, int) or abs(object_id) > 2**53 - 1 or not isinstance(is_removed, bool): raise WebViewerExportError("object index instance identity is invalid")
            if not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in bbox) or bbox[0] > bbox[2] or bbox[1] > bbox[3]: raise WebViewerExportError("object index instance bbox is invalid")
            images.add(image_id); object_ids.append(object_id); removed += int(is_removed); active += int(not is_removed)
        if entry.get("images") != sorted(images) or entry.get("objects") != sorted(object_ids) or entry.get("active_count") != active or entry.get("removed_count") != removed or entry.get("total_count") != len(instances): raise WebViewerExportError("object index counts or derived IDs are invalid")


def _load_da3_cache(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            missing = sorted(_REQUIRED_CACHE_FIELDS - set(loaded.files))
            if missing:
                raise WebViewerExportError(
                    "DA3 cache missing required schema-v3 fields: " + ", ".join(missing)
                )
            cache = {field: loaded[field].copy() for field in _REQUIRED_CACHE_FIELDS}
    except WebViewerExportError:
        raise
    except (OSError, ValueError) as error:
        if "allow_pickle" in str(error) or "Object arrays" in str(error):
            raise WebViewerExportError(
                f"cannot load DA3 cache {path}: {error} - object-dtype scalar fields "
                "from an outdated cache writer; regenerate the DA3 cache"
            ) from error
        raise WebViewerExportError(f"cannot load DA3 cache {path}: {error}") from error

    schema = cache["cache_schema_version"]
    if schema.shape != () or schema.dtype.kind not in "iu" or int(schema.item()) != 3:
        raise WebViewerExportError("DA3 cache schema version must be exactly 3")
    is_metric = integer_scalar(
        cache["is_metric"], "is_metric", error=WebViewerExportError
    )
    if is_metric != 1:
        raise WebViewerExportError(
            f"DA3 cache is_metric must be 1 (metric model), got {is_metric}"
        )
    scale_factor = cache["scale_factor"]
    if scale_factor.shape != () or scale_factor.dtype.kind != "f":
        raise WebViewerExportError("DA3 cache scale_factor must be a float scalar")
    if not np.isnan(float(scale_factor)) and float(scale_factor) <= 0:
        raise WebViewerExportError("DA3 cache scale_factor must be positive")
    source_model = unicode_scalar(
        cache["source_model"], "source_model", error=WebViewerExportError
    )
    if not _MODEL_ID.fullmatch(source_model):
        raise WebViewerExportError("DA3 cache source_model is unsafe")
    points = cache["world_points"]
    confidence = cache["world_points_conf"]
    images = cache["images"]
    image_ids = cache["image_ids"]
    if (
        points.dtype != np.dtype(np.float32)
        or points.ndim != 4
        or points.shape[-1] != 3
    ):
        raise WebViewerExportError(
            "DA3 cache world_points must be float32 with shape (N, H, W, 3)"
        )
    frame_count, height, width, _ = points.shape
    if frame_count < 1 or height < 1 or width < 1:
        raise WebViewerExportError("DA3 cache world_points grid must be nonempty")
    if confidence.dtype != np.dtype(np.float32) or confidence.shape != points.shape[:3]:
        raise WebViewerExportError(
            "DA3 cache world_points_conf must be float32 and align with world_points"
        )
    if images.dtype != np.dtype(np.uint8) or images.shape != (
        frame_count,
        height,
        width,
        3,
    ):
        raise WebViewerExportError(
            "DA3 cache images must be uint8 and align with world_points"
        )
    extrinsic = cache["extrinsic"]
    if (
        extrinsic.dtype.kind != "f"
        or extrinsic.ndim != 3
        or extrinsic.shape[0] != frame_count
    ):
        raise WebViewerExportError(
            "DA3 cache extrinsic must be a float per-frame w2c transform stack"
        )
    if extrinsic.shape[1:] == (4, 4):
        extrinsic = extrinsic[:, :3, :4]
    if extrinsic.shape[1:] != (3, 4) or not np.isfinite(extrinsic).all():
        raise WebViewerExportError(
            "DA3 cache extrinsic must be finite with shape (N, 3, 4) or (N, 4, 4)"
        )
    if image_ids.dtype.kind not in "iu" or image_ids.shape != (frame_count,):
        raise WebViewerExportError(
            "DA3 cache image_ids must be an integer vector aligned with frames"
        )
    if len({int(value) for value in image_ids}) != frame_count:
        raise WebViewerExportError("DA3 cache image_ids must be unique")
    integer_info = np.iinfo(np.int32)
    if (image_ids.dtype.kind == "i" and np.any(image_ids < integer_info.min)) or np.any(
        image_ids > integer_info.max
    ):
        raise WebViewerExportError("DA3 cache image_ids cannot be represented as int32")
    sizes = cache["source_image_sizes"]
    affine = cache["source_to_processed_affine"]
    hashes = cache["source_image_sha256"]
    convention = unicode_scalar(
        cache["affine_convention"], "affine_convention", error=WebViewerExportError
    )
    resolution = integer_scalar(
        cache["preprocess_resolution"],
        "preprocess_resolution",
        error=WebViewerExportError,
    )
    method = unicode_scalar(
        cache["preprocess_method"], "preprocess_method", error=WebViewerExportError
    )
    if (
        sizes.dtype.kind not in "iu"
        or sizes.shape != (frame_count, 2)
        or np.any(sizes <= 0)
    ):
        raise WebViewerExportError(
            "DA3 cache source_image_sizes must be positive integer pairs"
        )
    if (
        affine.dtype.kind not in "fiu"
        or affine.shape != (frame_count, 2, 3)
        or not np.isfinite(affine).all()
    ):
        raise WebViewerExportError("DA3 cache source_to_processed_affine is invalid")
    validate_affine_linear_parts(affine, error=WebViewerExportError)
    if (
        hashes.dtype.kind != "U"
        or hashes.shape != (frame_count,)
        or any(_SHA256.fullmatch(str(value)) is None for value in hashes)
    ):
        raise WebViewerExportError("DA3 cache source_image_sha256 is invalid")
    if (
        convention != "pixel_center_v1"
        or resolution <= 0
        or method != "upper_bound_resize"
    ):
        raise WebViewerExportError("DA3 cache formal preprocessing metadata is invalid")
    provenance = {
        "schema_version": 2,
        "source_model": source_model,
        "affine_convention": convention,
        "preprocess_resolution": resolution,
        "preprocess_method": method,
        "frame_count": int(frame_count),
        "processed_size": [int(width), int(height)],
        "image_ids": [int(value) for value in image_ids],
        "source_image_sha256": [str(value) for value in hashes],
    }
    return {
        "image_ids": image_ids.astype(np.int32, copy=False),
        "points": points,
        "confidence": confidence,
        "images": images,
        "affine": affine.astype(np.float64, copy=False),
        "extrinsic": extrinsic.astype(np.float64, copy=False),
        "source_image_sizes": sizes,
        "source_image_sha256": [str(value) for value in hashes],
        "provenance": provenance,
    }


def _sample_points(
    cache: dict[str, Any],
    objects: dict[str, Any],
    *,
    mask_cache_root: Path,
    voxel_size: float,
    max_points: int,
    filter_config: PointCloudFilterConfig | None = None,
) -> dict[str, Any]:
    points = cache["points"].reshape(-1, 3)
    confidence = cache["confidence"].reshape(-1)
    colors = cache["images"].reshape(-1, 3)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.any(points != 0, axis=1)
        & np.isfinite(confidence)
    )
    valid_indices = np.flatnonzero(valid)
    valid_points = points[valid].astype(np.float64, copy=False)
    labels, label_keys, sam3_mask_entries = _instance_labels(
        cache, objects, valid_indices, points.shape[0], mask_cache_root
    )
    # 仅有效 SAM3 mask 覆盖点（label >= 0）硬保留：它们已经通过 valid 的
    # finite/nonzero 基础校验，但不参加 SOR、DBSCAN、ground-plane 或 sky-line
    # 裁剪。bbox 内 mask 外点仍保持背景过滤，以免把框内噪声带入 bundle。
    protected = labels >= 0
    keep_filter = filter_scene_points(
        valid_points, filter_config, protect_mask=protected
    )
    filtered_points = valid_points[keep_filter]
    points_v = valid_points
    confidence_v = confidence[valid]
    colors_v = colors[valid]
    labels_f = labels[keep_filter]
    points = points_v[keep_filter]
    confidence = confidence_v[keep_filter]
    colors = colors_v[keep_filter]
    level_rotation = _fit_level_rotation(valid_points, cache["extrinsic"])
    # 天空线裁剪：摆平坐标系中以实例标签点高度 p99.9 + margin 为界，裁掉界
    # 以上的非 mask 点（天花板/天空薄片与主簇稠密相连，SOR/DBSCAN 无法剔除）。
    aligned, leveled = level_rotation
    if leveled and (labels_f >= 0).any():
        heights = (filtered_points @ aligned.T)[:, 1]
        subject_top = float(
            np.percentile(heights[labels_f >= 0], _SKY_SUBJECT_PERCENTILE)
        )
        sky_line = subject_top + _SKY_MARGIN_M
        sky_keep = (heights <= sky_line) | (labels_f >= 0)
        dropped = int((~sky_keep).sum())
        if dropped and sky_keep.sum() >= (1.0 - _SKY_MAX_DROP_RATIO) * len(heights):
            logger.info(
                "sky cut: dropping %d points above subject top %.3f m + margin",
                dropped,
                subject_top,
            )
            points = points[sky_keep]
            confidence = confidence[sky_keep]
            colors = colors[sky_keep]
            labels_f = labels_f[sky_keep]
            filtered_points = filtered_points[sky_keep]
    if len(points):
        scaled = np.floor(points.astype(np.float64, copy=False) / voxel_size)
        integer_info = np.iinfo(np.int64)
        if (
            not np.isfinite(scaled).all()
            or np.any(scaled < integer_info.min)
            or np.any(scaled > integer_info.max)
        ):
            raise WebViewerExportError("DA3 voxel keys exceed int64 representation")
        selected: dict[tuple[int, int, int, int], int] = {}
        protected_voxels: set[tuple[int, int, int]] = set()
        for index, key_values in enumerate(scaled.astype(np.int64)):
            label = int(labels_f[index])
            voxel = tuple(int(value) for value in key_values)
            if label >= 0:
                protected_voxels.add(voxel)
            key = (*voxel, label)
            previous = selected.get(key)
            if previous is None or confidence[index] > confidence[previous]:
                selected[key] = index
        for voxel in protected_voxels:
            selected.pop((*voxel, -1), None)
        keep = np.fromiter(selected.values(), dtype=np.int64, count=len(selected))
        protected_keep = keep[labels_f[keep] >= 0]
        if len(protected_keep) > max_points:
            raise WebViewerExportError(
                "protected representatives "
                f"({len(protected_keep)}) exceed max_points ({max_points}); "
                "refusing to silently discard SAM3 SKU geometry"
            )
        if len(keep) > max_points:
            rng = np.random.default_rng(42)
            background_keep = keep[labels_f[keep] < 0]
            sampled_background = rng.choice(
                background_keep,
                size=max_points - len(protected_keep),
                replace=False,
            )
            keep = np.sort(np.concatenate([protected_keep, sampled_background]))
        points = points[keep]
        colors = colors[keep]
        labels_final = labels_f[keep]
    else:
        labels_final = labels_f
    normals = _estimate_scene_normals(points, cache["extrinsic"])
    # Cluster each instance's points into one contiguous index range (stable sort:
    # unlabeled points first, then instances in ascending label order).
    order = np.argsort(labels_final, kind="stable")
    labels_sorted = labels_final[order]
    positions = np.ascontiguousarray(points[order], dtype="<f4")
    if not np.isfinite(positions).all():
        raise WebViewerExportError(
            "valid DA3 points must be representable as finite float32"
        )
    return {
        "positions": positions,
        "colors": np.ascontiguousarray(colors[order], dtype=np.uint8),
        "normals": np.ascontiguousarray(
            np.rint(normals[order] * 127.0).clip(-127, 127), dtype=np.int8
        ),
        "instance_labels": labels_sorted,
        "label_keys": label_keys,
        "sam3_mask_entries": sam3_mask_entries,
        "valid_points": valid_points,
        "filtered_points": filtered_points,
        "level_rotation": level_rotation,
    }


_NORMAL_KNN = 30
_NORMAL_SUBSAMPLE = 200_000
_NORMAL_FALLBACK = np.asarray([0.0, 0.0, 1.0])


def _estimate_scene_normals(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Unit normals for final selected representatives, oriented to face the mean camera.

    Estimated after voxel/max-point representative selection and before instance
    sorting so the same permutation path as colors keeps row alignment; the response radius
    adapts to the cloud's median nearest-neighbour distance. Degenerate
    neighbourhoods fall back to a constant unit vector instead of NaN.
    """
    try:
        import open3d as o3d
    except ImportError as error:
        raise ImportError("Open3D required: pip install open3d") from error

    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    rotation_w2c = extrinsic[:, :, :3]
    translation_w2c = extrinsic[:, :, 3]
    camera_mean = -np.einsum("nji,nj->ni", rotation_w2c, translation_w2c).mean(axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    subsample = (
        points
        if len(points) <= _NORMAL_SUBSAMPLE
        else points[np.linspace(0, len(points) - 1, _NORMAL_SUBSAMPLE).astype(np.int64)]
    )
    probe = o3d.geometry.PointCloud()
    probe.points = o3d.utility.Vector3dVector(subsample)
    median_nn = float(np.median(probe.compute_nearest_neighbor_distance()))
    radius = float(np.clip(median_nn * 4.0, 0.005, 0.2))
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=_NORMAL_KNN)
    )
    pcd.orient_normals_towards_camera_location(camera_mean)
    normals = np.asarray(pcd.normals, dtype=np.float64).reshape(len(points), 3)
    finite = np.isfinite(normals).all(axis=1)
    normals[~finite] = _NORMAL_FALLBACK
    return normals


_SAM3_MASK_SCHEMA = "sam3_frame_mask_cache_v2_canonical_bbox_clip"


def _sam3_mask_index(
    mask_cache_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Scan SAM3 frame-mask cache entries, indexed by source image SHA-256.

    返回按 source image SHA-256 索引的完整 immutable entry 元数据。每个 entry
    都先验证 content-addressed key；实际被 viewer 使用时还会验证 payload、每个
    mask digest/count 与 canonical bbox clipping，任何不一致均 fail-closed。
    """
    if not mask_cache_root.is_dir():
        raise WebViewerExportError(
            f"SAM3 mask cache not found: {mask_cache_root} "
            "(run --mode ground-stack-area before viewer-web export)"
        )
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry_dir in sorted(mask_cache_root.glob("entries/*")):
        manifest_path = entry_dir / "manifest.json"
        if not entry_dir.is_dir() or not manifest_path.is_file():
            raise WebViewerExportError(
                f"SAM3 mask cache entry missing manifest: {entry_dir}"
            )
        manifest = _read_json(manifest_path)
        if (
            not isinstance(manifest, dict)
            or set(manifest)
            != {"complete", "key", "key_payload", "payload_sha256", "masks"}
            or manifest.get("complete") is not True
        ):
            raise WebViewerExportError(f"SAM3 mask cache entry incomplete: {entry_dir}")
        key_payload = manifest.get("key_payload")
        if not isinstance(key_payload, dict) or (
            key_payload.get("schema") != _SAM3_MASK_SCHEMA
        ):
            raise WebViewerExportError(
                f"SAM3 mask cache entry schema mismatch: {entry_dir}"
            )
        canonical_key = hashlib.sha256(
            json.dumps(
                key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ).hexdigest()
        if (
            entry_dir.name != canonical_key
            or manifest.get("key") != canonical_key
            or not isinstance(manifest.get("key"), str)
        ):
            raise WebViewerExportError(
                f"SAM3 mask cache entry key does not match canonical payload: {entry_dir}"
            )
        image = key_payload.get("image")
        detections = key_payload.get("detections")
        sha = image.get("sha256") if isinstance(image, dict) else None
        image_id = image.get("image_id") if isinstance(image, dict) else None
        payload_sha256 = manifest.get("payload_sha256")
        if (
            not isinstance(sha, str)
            or _SHA256.fullmatch(sha) is None
            or isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or not isinstance(payload_sha256, str)
            or _SHA256.fullmatch(payload_sha256) is None
            or not isinstance(detections, list)
            or not all(isinstance(item, dict) for item in detections)
            or not isinstance(manifest.get("masks"), list)
            or len(manifest["masks"]) != len(detections)
        ):
            raise WebViewerExportError(
                f"SAM3 mask cache entry manifest invalid: {entry_dir}"
            )
        index[sha].append(
            {
                "entry_dir": entry_dir,
                "key": canonical_key,
                "image_id": image_id,
                "payload_sha256": payload_sha256,
                "detections": detections,
                "mask_metadata": manifest["masks"],
                "key_payload": key_payload,
            }
        )
    return dict(index)


def _canonical_bbox_from_detection(
    detection: dict[str, Any],
) -> tuple[float, float, float, float]:
    encoded = detection.get("bbox_xyxy_f64be_hex")
    if (
        not isinstance(encoded, list)
        or len(encoded) != 4
        or not all(isinstance(value, str) for value in encoded)
    ):
        raise WebViewerExportError("SAM3 mask cache detection bbox is invalid")
    values: list[float] = []
    for value in encoded:
        try:
            raw = bytes.fromhex(value)
            decoded = struct.unpack(">d", raw)[0]
        except (ValueError, struct.error) as error:
            raise WebViewerExportError(
                "SAM3 mask cache bbox encoding is invalid"
            ) from error
        canonical = struct.pack(">d", 0.0 if decoded == 0.0 else decoded).hex()
        if len(raw) != 8 or not math.isfinite(decoded) or value != canonical:
            raise WebViewerExportError("SAM3 mask cache bbox encoding is not canonical")
        values.append(decoded)
    if values[2] < values[0] or values[3] < values[1]:
        raise WebViewerExportError("SAM3 mask cache bbox is reversed")
    return tuple(values)  # type: ignore[return-value]


def _sam3_entry_matches_instances(
    entry: dict[str, Any], instances: list[tuple[int, list[float], int]]
) -> bool:
    by_object: dict[int, tuple[float, float, float, float]] = {}
    try:
        for detection in entry["detections"]:
            object_id = detection.get("object_id")
            if isinstance(object_id, bool) or not isinstance(object_id, int):
                return False
            if object_id in by_object:
                return False
            by_object[object_id] = _canonical_bbox_from_detection(detection)
    except WebViewerExportError:
        return False
    return all(
        by_object.get(object_id) == tuple(bbox) for _label, bbox, object_id in instances
    )


def _mask_is_within_bbox(
    mask: np.ndarray, bbox: tuple[float, float, float, float]
) -> bool:
    x1, y1, x2, y2 = bbox
    height, width = mask.shape
    xi1 = int(max(0, min(width - 1, round(x1))))
    yi1 = int(max(0, min(height - 1, round(y1))))
    xi2 = int(max(xi1 + 1, min(width, round(x2))))
    yi2 = int(max(yi1 + 1, min(height, round(y2))))
    clipped = mask.copy()
    clipped[:yi1, :] = False
    clipped[yi2:, :] = False
    clipped[:, :xi1] = False
    clipped[:, xi2:] = False
    return bool(np.array_equal(mask, clipped))


def _load_verified_sam3_masks(
    entry: dict[str, Any],
    *,
    image_id: int,
    source_shape: tuple[int, int],
) -> np.ndarray:
    if entry["image_id"] != image_id:
        raise WebViewerExportError(
            f"SAM3 mask cache image ID does not match cache frame {image_id}"
        )
    payload_path = entry["entry_dir"] / "masks.npz"
    if _file_sha256(payload_path) != entry["payload_sha256"]:
        raise WebViewerExportError(
            "SAM3 mask cache payload SHA-256 does not match manifest"
        )
    try:
        with np.load(payload_path, allow_pickle=False) as loaded:
            if loaded.files != ["masks"]:
                raise WebViewerExportError(
                    "SAM3 mask cache payload must contain exactly masks"
                )
            masks = loaded["masks"]
    except WebViewerExportError:
        raise
    except (OSError, ValueError) as error:
        raise WebViewerExportError(
            f"cannot load SAM3 mask cache payload: {error}"
        ) from error
    detections = entry["detections"]
    if masks.dtype != np.bool_ or masks.shape != (len(detections), *source_shape):
        raise WebViewerExportError("SAM3 mask cache masks shape/dtype invalid")
    output_contract = entry["key_payload"].get("output_contract")
    if (
        not isinstance(output_contract, dict)
        or output_contract.get("dtype") != "bool"
        or output_contract.get("shape_hw") != list(source_shape)
    ):
        raise WebViewerExportError("SAM3 mask cache output contract is invalid")
    for detection, mask, metadata in zip(detections, masks, entry["mask_metadata"]):
        bbox = _canonical_bbox_from_detection(detection)
        expected = {
            "sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
            "true_pixel_count": int(mask.sum()),
        }
        if not isinstance(metadata, dict) or metadata != expected:
            raise WebViewerExportError(
                "SAM3 mask cache per-mask digest or pixel count does not match"
            )
        if not _mask_is_within_bbox(mask, bbox):
            raise WebViewerExportError(
                "SAM3 mask cache contains true pixels outside its canonical bbox"
            )
    return masks


def _instance_labels(
    cache: dict[str, Any],
    objects: dict[str, Any],
    valid_indices: np.ndarray,
    flat_count: int,
    mask_cache_root: Path,
) -> tuple[np.ndarray, list[tuple[str, int]], list[dict[str, object]]]:
    """Map each DA3 grid point to the instance whose **SAM3 mask** covers its source pixel.

    与 bbox 矩形不同，mask 已被生产端 clip 到 bbox 内并紧贴商品轮廓：label（即
    染色/选中的点集与过滤保护范围）只含 mask 内的点，bbox 内 mask 外的噪点不参与。
    Returns ``(labels, label_keys)`` where ``labels`` aligns with ``valid_indices``
    (-1 = unlabeled) and ``label_keys[label]`` is ``(global_id, instance_index)``.
    """
    frame_count, height, width, _ = cache["points"].shape
    frame_for_image = {
        int(value): index for index, value in enumerate(cache["image_ids"])
    }
    by_image: dict[int, list[tuple[str, int, list[float]]]] = defaultdict(list)
    label_keys: list[tuple[str, int]] = []
    for global_id in sorted(objects, key=int):
        for instance_index, instance in enumerate(objects[global_id]["instances"]):
            bbox = instance["bbox"]
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in bbox
                )
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                raise WebViewerExportError(
                    "global mapping instance bbox is invalid for global ID "
                    f"{global_id} instance {instance_index}"
                )
            label_keys.append((global_id, instance_index))
            by_image[int(instance["image_id"])].append(
                (
                    len(label_keys) - 1,
                    [float(value) for value in bbox],
                    int(instance["object_id"]),
                )
            )
    mask_entries = _sam3_mask_index(mask_cache_root)
    used_entries: dict[int, dict[str, object]] = {}
    grid = np.full(flat_count, -1, dtype=np.int32).reshape(frame_count, height, width)
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    for image_id, instances in by_image.items():
        frame = frame_for_image.get(image_id)
        if frame is None:
            raise WebViewerExportError(
                f"global mapping instance references image {image_id} absent from cache"
            )
        affine = cache["affine"][frame]
        linear = affine[:, :2]
        translation = affine[:, 2]
        determinant = linear[0, 0] * linear[1, 1] - linear[0, 1] * linear[1, 0]
        if abs(determinant) < 1e-12:
            raise WebViewerExportError(
                f"source_to_processed_affine for frame {frame} is singular"
            )
        candidates = mask_entries.get(cache["source_image_sha256"][frame], [])
        matching_entries = [
            entry
            for entry in candidates
            if _sam3_entry_matches_instances(entry, instances)
        ]
        if not matching_entries:
            raise WebViewerExportError(
                f"SAM3 mask cache has no entry for image {image_id} "
                "(run --mode ground-stack-area before viewer-web export)"
            )
        if len(matching_entries) != 1:
            raise WebViewerExportError(
                f"SAM3 mask cache has ambiguous entries for image {image_id}"
            )
        entry = matching_entries[0]
        detections = entry["detections"]
        mask_for_object: dict[int, tuple[int, tuple[float, float, float, float]]] = {}
        for mask_index, detection in enumerate(detections):
            object_id = detection.get("object_id")
            encoded = detection.get("bbox_xyxy_f64be_hex")
            if (
                isinstance(object_id, bool)
                or not isinstance(object_id, int)
                or not isinstance(encoded, list)
                or len(encoded) != 4
                or not all(isinstance(value, str) for value in encoded)
            ):
                raise WebViewerExportError(
                    f"SAM3 mask cache detections invalid for image {image_id}"
                )
            try:
                decoded = tuple(
                    struct.unpack(">d", bytes.fromhex(value))[0] for value in encoded
                )
            except ValueError as error:
                raise WebViewerExportError(
                    f"SAM3 mask cache bbox encoding invalid for image {image_id}"
                ) from error
            if object_id in mask_for_object:
                raise WebViewerExportError(
                    f"SAM3 mask cache has duplicate object {object_id} for image {image_id}"
                )
            mask_for_object[object_id] = (mask_index, decoded)
        source_width, source_height = (
            int(value) for value in cache["source_image_sizes"][frame]
        )
        masks = _load_verified_sam3_masks(
            entry,
            image_id=image_id,
            source_shape=(source_height, source_width),
        )
        used_entries[image_id] = {
            "image_id": image_id,
            "key": entry["key"],
            "payload_sha256": entry["payload_sha256"],
        }
        # cv2.warpAffine semantics: processed (x, y) samples source at A^-1 (x, y).
        local_x = xs - translation[0]
        local_y = ys - translation[1]
        source_x = (linear[1, 1] * local_x - linear[0, 1] * local_y) / determinant
        source_y = (linear[0, 0] * local_y - linear[1, 0] * local_x) / determinant
        # mask 定义在源图像素网格上：取最近源像素（clamp 到图内）做点态采样。
        sample_x = np.clip(np.rint(source_x), 0, source_width - 1).astype(np.intp)
        sample_y = np.clip(np.rint(source_y), 0, source_height - 1).astype(np.intp)
        frame_grid = grid[frame]
        for label, bbox, object_id in instances:
            hit = mask_for_object.get(object_id)
            if hit is None:
                raise WebViewerExportError(
                    f"instance object {object_id} of image {image_id} absent "
                    "from SAM3 mask cache"
                )
            mask_index, mask_bbox = hit
            if mask_bbox != tuple(bbox):
                raise WebViewerExportError(
                    f"instance bbox disagrees with SAM3 mask cache for image "
                    f"{image_id} object {object_id}"
                )
            covered = masks[mask_index][sample_y, sample_x]
            frame_grid[covered & (frame_grid < 0)] = label
        del masks
    return (
        grid.reshape(-1)[valid_indices],
        label_keys,
        [used_entries[image_id] for image_id in sorted(used_entries)],
    )


def _attach_point_index_ranges(
    objects: dict[str, Any],
    labels_sorted: np.ndarray,
    label_keys: list[tuple[str, int]],
) -> None:
    for label, (global_id, instance_index) in enumerate(label_keys):
        start = int(np.searchsorted(labels_sorted, label, side="left"))
        end = int(np.searchsorted(labels_sorted, label, side="right"))
        objects[global_id]["instances"][instance_index]["point_index_range"] = [
            start,
            end,
        ]


def _resolve_source_images(images_dir: Path, cache: dict[str, Any]) -> dict[int, Path]:
    """Resolve cache image IDs to source image files, fail-closed on any mismatch.

    与 ``da3_runner.py`` 同一约定：文件名 stem 中的数字即 image_id，且逐文件
    SHA-256 必须与 DA3 cache 的 ``source_image_sha256`` 完全一致——字节级一致
    保证缩略图裁剪坐标系（raw 未转置像素空间，与 bbox/affine 相同）与 DA3
    推理时读到的图像一致，EXIF 方向问题因此被显式排除而非隐式假设。
    """
    if not images_dir.is_dir():
        raise WebViewerExportError(f"source images directory not found: {images_dir}")
    by_id: dict[int, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in _IMAGE_EXTS or not path.is_file():
            continue
        match = _IMAGE_ID_STEM.search(path.stem)
        if match is None:
            raise WebViewerExportError(
                f"source image {path.name} has no numeric image id in its stem"
            )
        image_id = int(match.group(1))
        if image_id in by_id:
            raise WebViewerExportError(
                f"source image id {image_id} is ambiguous in {images_dir}"
            )
        by_id[image_id] = path
    resolved: dict[int, Path] = {}
    for frame, image_id in enumerate(cache["image_ids"]):
        path = by_id.get(int(image_id))
        if path is None:
            raise WebViewerExportError(
                f"source image for image {int(image_id)} not found in {images_dir}"
            )
        if _file_sha256(path) != cache["source_image_sha256"][frame]:
            raise WebViewerExportError(
                f"source image {path.name} does not match DA3 cache provenance"
            )
        resolved[int(image_id)] = path
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generate_thumbnails(
    objects: dict[str, Any], images_by_id: dict[int, Path]
) -> dict[str, bytes]:
    """Crop one JPEG thumbnail per instance (active + removed) into memory.

    bbox 为源图像素坐标（检测器输出空间，与 DA3 affine 的 source 空间一致），
    四周加 10% padding 并 clamp 到图内，长边缩到 <=256px、JPEG q85。返回
    ``{"thumbs/<globalId>_<instanceIndex>.jpg": bytes}`` 并把相对路径写回
    ``instance["thumbnail"]``，随后与 bundle 其余文件一起原子发布。
    """
    thumbs: dict[str, bytes] = {}
    loaded: dict[int, Image.Image] = {}
    try:
        for global_id in sorted(objects, key=int):
            for instance_index, instance in enumerate(objects[global_id]["instances"]):
                image_id = int(instance["image_id"])
                path = images_by_id.get(image_id)
                if path is None:
                    raise WebViewerExportError(
                        f"global mapping instance references image {image_id} "
                        "absent from cache"
                    )
                image = loaded.get(image_id)
                if image is None:
                    image = Image.open(path).convert("RGB")
                    loaded[image_id] = image
                width, height = image.size
                x1, y1, x2, y2 = (float(value) for value in instance["bbox"])
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    raise WebViewerExportError(
                        f"global mapping instance bbox for global ID {global_id} "
                        f"instance {instance_index} exceeds source image bounds "
                        f"{width}x{height}"
                    )
                pad_x = (x2 - x1) * _THUMB_PADDING
                pad_y = (y2 - y1) * _THUMB_PADDING
                crop = image.crop(
                    (
                        max(0, math.floor(x1 - pad_x)),
                        max(0, math.floor(y1 - pad_y)),
                        min(width, math.ceil(x2 + pad_x)),
                        min(height, math.ceil(y2 + pad_y)),
                    )
                )
                crop.thumbnail(
                    (_THUMB_LONG_EDGE, _THUMB_LONG_EDGE),
                    Image.Resampling.LANCZOS,
                )
                buffer = io.BytesIO()
                crop.save(buffer, format="JPEG", quality=_THUMB_JPEG_QUALITY)
                relative = f"{_THUMB_DIR}/{global_id}_{instance_index}.jpg"
                thumbs[relative] = buffer.getvalue()
                instance["thumbnail"] = relative
    finally:
        for image in loaded.values():
            image.close()
    return thumbs


_PLANE_SUBSAMPLE = 200_000
_PLANE_MIN_INLIER_RATIO = 0.05
_PLANE_MAX_TILT_DEG = 60.0
_PLANE_MAX_CANDIDATES = 8
_PLANE_MAX_BELOW_RATIO = 0.15
_SKY_SUBJECT_PERCENTILE = 99.9
_SKY_MARGIN_M = 0.15
_SKY_MAX_DROP_RATIO = 0.3


def _fit_level_rotation(
    valid_points: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """RANSAC 地平面拟合 -> CV->glTF 翻转 + 摆平旋转。

    返回 ``(rotation, leveled)``：rotation 把 DA3 world 点映射到 Y-up 摆平
    坐标系；找不到可信地平面时 leveled=False，只做 CV->glTF 翻转。
    与天空线裁剪共用同一旋转，保证两处使用同一个"上"方向。

    地堆/货架场景里最大的平面常是竖直的墙或货架面，因此迭代剔除已拟合
    平面、最多尝试 ``_PLANE_MAX_CANDIDATES`` 个候选，取第一个同时满足：
    内点比、倾角门（近水平）、地板性门（几乎无点位于平面下方--地面从下方
    支撑场景，斜穿主体的平面必然有大比例点在其下方）的平面作为地面。
    """
    if len(valid_points) < 3:
        raise WebViewerExportError("DA3 cache has too few points to orient the scene")
    try:
        import open3d as o3d
    except ImportError as error:
        raise ImportError("Open3D required: pip install open3d") from error

    rotation_w2c = extrinsic[:, :, :3]
    translation_w2c = extrinsic[:, :, 3]
    camera_centers = -np.einsum("nji,nj->ni", rotation_w2c, translation_w2c)
    camera_mean = camera_centers.mean(axis=0)

    subsample = (
        valid_points
        if len(valid_points) <= _PLANE_SUBSAMPLE
        else valid_points[
            np.linspace(0, len(valid_points) - 1, _PLANE_SUBSAMPLE).astype(np.int64)
        ]
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(subsample)
    median_nn = float(np.median(pcd.compute_nearest_neighbor_distance()))
    distance_threshold = float(np.clip(median_nn * 3.0, 0.02, 0.15))
    total = len(subsample)
    min_inliers = _PLANE_MIN_INLIER_RATIO * total
    m_flip = np.diag([1.0, -1.0, -1.0, 1.0])
    remaining = pcd
    for _ in range(_PLANE_MAX_CANDIDATES):
        if len(remaining.points) < max(min_inliers, 3):
            break
        plane, inliers = remaining.segment_plane(
            distance_threshold, ransac_n=3, num_iterations=1000
        )
        inliers = np.asarray(inliers)
        plane_coeffs = np.asarray(plane[:4], dtype=np.float64)
        norm = float(np.linalg.norm(plane_coeffs[:3]))
        if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
            logger.warning("RANSAC returned a degenerate plane; skipping candidate")
            remaining = remaining.select_by_index(inliers, invert=True)
            continue
        normal = plane_coeffs[:3] / norm
        plane_d = plane_coeffs[3] / norm
        if normal @ (camera_mean - subsample.mean(axis=0)) < 0:
            normal = -normal
            plane_d = -plane_d
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -normal[1]))))
        below_ratio = float(np.mean(subsample @ normal + plane_d < -0.1))
        if (
            len(inliers) >= min_inliers
            and tilt_deg <= _PLANE_MAX_TILT_DEG
            and below_ratio <= _PLANE_MAX_BELOW_RATIO
        ):
            r_level = _shortest_arc_to_y_up(m_flip[:3, :3] @ normal)
            return (r_level @ m_flip)[:3, :3], True
        remaining = remaining.select_by_index(inliers, invert=True)
    return m_flip[:3, :3].copy(), False


def _compute_world_to_view(
    filtered_points: np.ndarray,
    level_rotation: tuple[np.ndarray, bool],
) -> np.ndarray:
    """Row-major 4x4 M = T_center @ R_level @ M_flip (column-vector convention).

    R_level 来自 ``_fit_level_rotation``（在过滤前的有效点集上拟合，因为导出
    过滤按设计会剔除地面）；T_center 把摆平后过滤点云的逐轴 median 移到原点。
    """
    if len(filtered_points) < 1:
        raise WebViewerExportError("DA3 cache has too few points to orient the scene")
    aligned, _leveled = level_rotation
    leveled = filtered_points @ aligned.T
    world_to_view = np.eye(4)
    world_to_view[:3, :3] = aligned
    world_to_view[:3, 3] = -np.median(leveled, axis=0)
    return world_to_view


def _robust_display_bounds(positions: np.ndarray) -> list[float]:
    """Compute final-position source-coordinate p01/p99 bounds for initial framing."""
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise WebViewerExportError("cannot export display_bounds without final positions")
    if not np.isfinite(positions).all():
        raise WebViewerExportError("cannot export display_bounds from non-finite positions")
    bounds = np.percentile(positions.astype(np.float64, copy=False), [1.0, 99.0], axis=0)
    return [float(value) for value in (*bounds[0], *bounds[1])]


def _shortest_arc_to_y_up(normal: np.ndarray) -> np.ndarray:
    """4x4 rotation mapping ``normal`` onto +Y via the shortest arc."""
    target = np.asarray([0.0, 1.0, 0.0])
    axis = np.cross(normal, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(normal @ target, -1.0, 1.0))
    rotation = np.eye(4)
    if sine < 1e-9:
        if cosine > 0:
            return rotation
        rotation[:3, :3] = np.diag([1.0, -1.0, -1.0])
        return rotation
    axis /= sine
    angle = math.atan2(sine, cosine)
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation[:3, :3] = (
        np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    )
    return rotation


def _load_footprint(
    artifacts: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    try:
        report_path = Path(artifacts["measurement_report"])
        report = _read_json(report_path)
        geojson = _read_json(Path(artifacts["footprints_geojson"]))
        generation_manifest = _read_json(report_path.with_name("manifest.json"))
    except KeyError as error:
        raise WebViewerExportError(
            "footprint resolver omitted a required artifact"
        ) from error
    if (
        not isinstance(report, dict)
        or not isinstance(geojson, dict)
        or not isinstance(generation_manifest, dict)
    ):
        raise WebViewerExportError(
            "formal footprint artifacts must contain JSON objects"
        )
    report_cache = report.get("cache")
    if not isinstance(report_cache, dict):
        raise WebViewerExportError(
            "formal footprint report cache provenance is invalid"
        )
    mapping_sha256 = report.get("global_mapping_sha256")
    if not isinstance(mapping_sha256, str) or _SHA256.fullmatch(mapping_sha256) is None:
        raise WebViewerExportError(
            "formal footprint report global mapping digest is invalid"
        )
    run_id = generation_manifest.get("run_id")
    if not isinstance(run_id, str):
        raise WebViewerExportError("formal footprint manifest run_id is invalid")
    status = report.get("status")
    if (
        report.get("metric") != "da3_ground_footprint_union"
        or report.get("unit") != "m2"
    ):
        raise WebViewerExportError("formal footprint report metric or unit is invalid")
    if status not in {"accepted", "rejected"}:
        raise WebViewerExportError("formal footprint report status is invalid")
    accepted = status == "accepted"
    value = report.get("value_m2")
    if accepted:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise WebViewerExportError(
                "accepted formal footprint value_m2 must be finite and non-negative"
            )
        if report.get("rejection_reason") is not None:
            raise WebViewerExportError(
                "accepted formal footprint rejection_reason must be null"
            )
    elif value is not None:
        raise WebViewerExportError("rejected formal footprint value_m2 must be null")
    elif (
        not isinstance(report.get("rejection_reason"), str)
        or not report["rejection_reason"].strip()
    ):
        raise WebViewerExportError(
            "rejected formal footprint rejection_reason must be non-empty"
        )
    if (
        geojson.get("type") != "FeatureCollection"
        or geojson.get("coordinate_space") != "local_support_plane_meters"
        or geojson.get("status") != status
        or geojson.get("measurement_complete") is not accepted
        or not isinstance(geojson.get("features"), list)
    ):
        raise WebViewerExportError("formal footprint GeoJSON contract is invalid")
    if not accepted:
        if geojson["features"]:
            raise WebViewerExportError(
                "rejected formal footprint must not contain geometry"
            )
        return (
            {
                "metric": report["metric"],
                "unit": report["unit"],
                "status": status,
                "value_m2": None,
                "rejection_reason": report.get("rejection_reason"),
                "run_id": run_id,
                "support_plane": None,
                "per_global_id": {},
                "union": None,
            },
            report_cache,
            mapping_sha256,
        )
    plane = _support_plane(report)
    per_global_id: dict[str, Any] = {}
    union: dict[str, Any] | None = None
    for feature in geojson["features"]:
        global_id, geometry = _parse_feature(feature)
        if global_id == "union":
            if union is not None:
                raise WebViewerExportError(
                    "formal footprint GeoJSON has multiple union features"
                )
            union = geometry
        elif global_id in per_global_id:
            raise WebViewerExportError(
                "formal footprint GeoJSON has duplicate global_id features"
            )
        else:
            per_global_id[global_id] = geometry
    if union is None:
        raise WebViewerExportError(
            "accepted formal footprint GeoJSON is missing union geometry"
        )
    if not per_global_id:
        raise WebViewerExportError(
            "accepted formal footprint GeoJSON is missing per-ID geometry"
        )
    if union["properties"]["area_m2"] != float(value):
        raise WebViewerExportError(
            "accepted formal footprint union area_m2 must equal value_m2"
        )
    return (
        {
            "metric": report["metric"],
            "unit": report["unit"],
            "status": status,
            "value_m2": float(value),
            "rejection_reason": report.get("rejection_reason"),
            "run_id": run_id,
            "support_plane": plane,
            "per_global_id": per_global_id,
            "union": union,
        },
        report_cache,
        mapping_sha256,
    )


def _mapping_sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise WebViewerExportError(f"cannot hash global mapping: {error}") from error


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, json.JSONDecodeError) as error:
        raise WebViewerExportError(
            f"cannot read JSON artifact {path}: {error}"
        ) from error


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _support_plane(report: dict[str, Any]) -> dict[str, list[float]]:
    plane = report.get("plane")
    selected = plane.get("selected") if isinstance(plane, dict) else None
    if not isinstance(selected, dict):
        raise WebViewerExportError(
            "accepted formal footprint report is missing support plane"
        )
    result: dict[str, list[float]] = {}
    for field in ("point", "u_axis", "v_axis", "normal"):
        value = selected.get(field)
        if not isinstance(value, list) or len(value) != 3:
            raise WebViewerExportError(
                f"formal footprint support plane {field} is invalid"
            )
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            for item in value
        ):
            raise WebViewerExportError(
                f"formal footprint support plane {field} is invalid"
            )
        result[field] = [float(item) for item in value]
    return result


def _parse_feature(feature: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(feature, dict) or feature.get("type") != "Feature":
        raise WebViewerExportError("formal footprint GeoJSON feature is invalid")
    properties = feature.get("properties")
    if not isinstance(properties, dict) or not isinstance(
        properties.get("global_id"), str
    ):
        raise WebViewerExportError("formal footprint GeoJSON global_id is invalid")
    global_id = properties["global_id"]
    if global_id == "union":
        expected_keys = {"coordinate_space", "global_id", "area_m2"}
        if set(properties) != expected_keys:
            raise WebViewerExportError("formal union footprint properties are invalid")
        normalized_properties = {
            "coordinate_space": "local_support_plane_meters",
            "global_id": "union",
            "area_m2": _footprint_area(properties.get("area_m2"), "union"),
        }
    else:
        expected_keys = {
            "coordinate_space",
            "global_id",
            "area_m2",
            "observations_used",
        }
        if set(properties) != expected_keys or _GLOBAL_ID.fullmatch(global_id) is None:
            raise WebViewerExportError("formal per-ID footprint properties are invalid")
        normalized_properties = {
            "coordinate_space": "local_support_plane_meters",
            "global_id": global_id,
            "area_m2": _footprint_area(properties.get("area_m2"), global_id),
            "observations_used": _observations_used(
                properties.get("observations_used")
            ),
        }
    if properties.get("coordinate_space") != "local_support_plane_meters":
        raise WebViewerExportError(
            "formal footprint GeoJSON coordinate_space is invalid"
        )
    if not global_id:
        raise WebViewerExportError("formal footprint GeoJSON global_id is invalid")
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") not in {
        "Polygon",
        "MultiPolygon",
    }:
        raise WebViewerExportError("formal footprint GeoJSON geometry is invalid")
    polygons = _parse_polygons(geometry["type"], geometry.get("coordinates"))
    return global_id, {"rings": polygons, "properties": normalized_properties}


_GLOBAL_ID = re.compile(r"^(0|[1-9][0-9]*)$")


def _footprint_area(value: Any, global_id: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise WebViewerExportError(
            f"formal footprint area_m2 is invalid for global ID {global_id}"
        )
    return float(value)


def _observations_used(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not float(value).is_integer()
        or value < 0
        or value > 2**53 - 1
    ):
        raise WebViewerExportError(
            "formal per-ID footprint observations_used is invalid"
        )
    return int(value)


def _parse_polygons(
    geometry_type: str, coordinates: Any
) -> list[list[list[list[float]]]]:
    raw_polygons = [coordinates] if geometry_type == "Polygon" else coordinates
    if not isinstance(raw_polygons, list) or not raw_polygons:
        raise WebViewerExportError("formal footprint GeoJSON coordinates are invalid")
    polygons: list[list[list[list[float]]]] = []
    for raw_polygon in raw_polygons:
        if not isinstance(raw_polygon, list) or not raw_polygon:
            raise WebViewerExportError(
                "formal footprint GeoJSON polygon rings are invalid"
            )
        rings: list[list[list[float]]] = []
        for raw_ring in raw_polygon:
            if not isinstance(raw_ring, list) or len(raw_ring) < 4:
                raise WebViewerExportError("formal footprint GeoJSON ring is invalid")
            ring: list[list[float]] = []
            for coordinate in raw_ring:
                if (
                    not isinstance(coordinate, list)
                    or len(coordinate) != 2
                    or any(
                        isinstance(item, bool)
                        or not isinstance(item, (int, float))
                        or not math.isfinite(item)
                        for item in coordinate
                    )
                ):
                    raise WebViewerExportError(
                        "formal footprint GeoJSON coordinate is invalid"
                    )
                ring.append([float(coordinate[0]), float(coordinate[1])])
            if ring[0] != ring[-1]:
                raise WebViewerExportError(
                    "formal footprint GeoJSON ring must be closed"
                )
            rings.append(ring)
        polygons.append(rings)
    return polygons


def _publish_bundle(
    output_dir: Path,
    manifest: dict[str, Any],
    objects: dict[str, Any],
    footprint: dict[str, Any],
    arrays: dict[str, np.ndarray],
    thumbs: dict[str, bytes],
) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = output_dir / "runs"
    runs_root.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex
    generation = runs_root / run_id
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
    try:
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "positions.f32.bin").write_bytes(
            arrays["positions"].tobytes(order="C")
        )
        (temporary / "colors.u8.bin").write_bytes(arrays["colors"].tobytes(order="C"))
        (temporary / "normals.i8.bin").write_bytes(arrays["normals"].tobytes(order="C"))
        _write_json(temporary / "objects.json", objects)
        _write_json(temporary / "footprints.json", footprint)
        thumbs_dir = temporary / _THUMB_DIR
        thumbs_dir.mkdir()
        for relative, payload in thumbs.items():
            (temporary / relative).write_bytes(payload)
        os.rename(temporary, generation)
        _atomic_replace_current(
            output_dir / "CURRENT",
            {"complete": True, "run_id": run_id, "schema_version": "1.0.0"},
        )
        return generation
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _atomic_replace_current(current_path: Path, pointer: dict[str, object]) -> None:
    payload = (
        json.dumps(pointer, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".CURRENT.", dir=current_path.parent
    )
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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
