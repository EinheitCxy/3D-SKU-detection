"""Minimal schema-3 DA3 static web-viewer bundle exporter."""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import shutil
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.da3_cache_validation import validate_affine_linear_parts
from utils.global_id_mapper import GlobalIDMapper
from utils.global_object_index import build_global_object_index
from utils.pointcloud_filter import PointCloudFilterConfig, filter_scene_points
from utils.sam3_mask_cache import (
    FrameMaskCacheError,
    FrameMaskCacheRequest,
    ProcessedDetectionPrompt,
    SCHEMA as SAM3_MASK_CACHE_SCHEMA,
    load_complete_frame_masks,
    map_source_bbox_to_processed,
)

_REQUIRED_CACHE_FIELDS = frozenset(
    {
        "cache_schema_version",
        "image_ids",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "source_image_sizes",
        "source_to_processed_affine",
    }
)
_GLOBAL_ID = re.compile(r"^(0|[1-9][0-9]*)$")
_IMAGE_ID_STEM = re.compile(r"(\d+)")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_THUMB_DIR = "thumbs"
_THUMB_LONG_EDGE = 256
_THUMB_PADDING = 0.10
_THUMB_JPEG_QUALITY = 85
logger = logging.getLogger(__name__)


class WebViewerExportError(ValueError):
    """Raised when an input cannot produce a safe minimal viewer bundle."""


def export_web_viewer_bundle(
    *,
    dataset_name: str,
    da3_cache_path: Path,
    global_mapping_path: Path,
    output_dir: Path,
    source_images_dir: Path,
    sam3_mask_cache_root: Path,
    voxel_size_m: float = 0.01,
    max_points: int = 500_000,
    filter_config: PointCloudFilterConfig | None = None,
) -> dict[str, object]:
    """Publish point cloud arrays and selection data in an atomic schema-3 run."""
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise WebViewerExportError("dataset_name must be a non-empty string")
    voxel_size = _validate_export_options(voxel_size_m, max_points)
    cache = _load_da3_cache(Path(da3_cache_path))
    objects = build_global_object_index(GlobalIDMapper(str(Path(global_mapping_path))))
    thumbnails = _generate_thumbnails(
        objects, _resolve_source_images(Path(source_images_dir))
    )
    sampled = _sample_points(
        cache,
        objects,
        mask_cache_root=Path(sam3_mask_cache_root),
        voxel_size=voxel_size,
        max_points=max_points,
        filter_config=filter_config or PointCloudFilterConfig(),
    )
    _attach_point_index_ranges(
        objects, sampled["instance_labels"], sampled["label_keys"]
    )
    manifest = {
        "schema_version": "3.0.0",
        "backend": "DA3",
        "dataset_name": dataset_name,
        "frame_count": int(len(cache["image_ids"])),
        "display_bounds": _robust_display_bounds(sampled["positions"]),
        "world_to_view": [
            float(value)
            for value in _compute_world_to_view(
                sampled["filtered_points"], sampled["level_rotation"]
            ).reshape(-1)
        ],
    }
    generation = _publish_bundle(
        Path(output_dir),
        manifest,
        _minimal_objects(objects, point_count=len(sampled["positions"])),
        sampled,
        thumbnails,
    )
    return {
        "output_dir": str(Path(output_dir)),
        "manifest_path": str(generation / "manifest.json"),
        "point_count": int(len(sampled["positions"])),
        "thumbnail_count": len(thumbnails),
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


def _load_da3_cache(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            missing = sorted(_REQUIRED_CACHE_FIELDS - set(loaded.files))
            if missing:
                raise WebViewerExportError(
                    "DA3 cache missing required fields: " + ", ".join(missing)
                )
            cache = {field: loaded[field].copy() for field in _REQUIRED_CACHE_FIELDS}
    except WebViewerExportError:
        raise
    except (OSError, ValueError) as error:
        raise WebViewerExportError(f"cannot load DA3 cache {path}: {error}") from error

    schema = cache["cache_schema_version"]
    if schema.shape != () or schema.dtype.kind not in "iu" or int(schema.item()) != 3:
        raise WebViewerExportError("DA3 cache schema version must be exactly 3")
    points, confidence, images = (
        cache["world_points"],
        cache["world_points_conf"],
        cache["images"],
    )
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
        raise WebViewerExportError("DA3 cache images must align with world_points")
    image_ids = cache["image_ids"]
    if image_ids.dtype.kind not in "iu" or image_ids.shape != (frame_count,):
        raise WebViewerExportError("DA3 cache image_ids must align with frames")
    if len({int(value) for value in image_ids}) != frame_count:
        raise WebViewerExportError("DA3 cache image_ids must be unique")
    int32 = np.iinfo(np.int32)
    if (image_ids.dtype.kind == "i" and np.any(image_ids < int32.min)) or np.any(
        image_ids > int32.max
    ):
        raise WebViewerExportError("DA3 cache image_ids cannot be represented as int32")
    extrinsic = cache["extrinsic"]
    if (
        extrinsic.dtype.kind != "f"
        or extrinsic.ndim != 3
        or extrinsic.shape[0] != frame_count
    ):
        raise WebViewerExportError(
            "DA3 cache extrinsic must be a per-frame transform stack"
        )
    if extrinsic.shape[1:] == (4, 4):
        extrinsic = extrinsic[:, :3, :4]
    if extrinsic.shape[1:] != (3, 4) or not np.isfinite(extrinsic).all():
        raise WebViewerExportError("DA3 cache extrinsic is invalid")
    sizes, affine = cache["source_image_sizes"], cache["source_to_processed_affine"]
    if (
        sizes.dtype.kind not in "iu"
        or sizes.shape != (frame_count, 2)
        or np.any(sizes <= 0)
    ):
        raise WebViewerExportError("DA3 cache source_image_sizes is invalid")
    if (
        affine.dtype.kind not in "fiu"
        or affine.shape != (frame_count, 2, 3)
        or not np.isfinite(affine).all()
    ):
        raise WebViewerExportError("DA3 cache source_to_processed_affine is invalid")
    validate_affine_linear_parts(affine, error=WebViewerExportError)
    return {
        "image_ids": image_ids.astype(np.int32, copy=False),
        "points": points,
        "confidence": confidence,
        "images": images,
        "affine": affine.astype(np.float64, copy=False),
        "extrinsic": extrinsic.astype(np.float64, copy=False),
        "source_image_sizes": sizes,
    }


def _sample_points(
    cache: dict[str, Any],
    objects: dict[str, Any],
    *,
    mask_cache_root: Path,
    voxel_size: float,
    max_points: int,
    filter_config: PointCloudFilterConfig,
) -> dict[str, Any]:
    flat_points = cache["points"].reshape(-1, 3)
    flat_confidence = cache["confidence"].reshape(-1)
    flat_colors = cache["images"].reshape(-1, 3)
    valid = (
        np.isfinite(flat_points).all(axis=1)
        & np.any(flat_points != 0, axis=1)
        & np.isfinite(flat_confidence)
    )
    valid_indices = np.flatnonzero(valid)
    valid_points = flat_points[valid].astype(np.float64, copy=False)
    level_rotation = _fit_level_rotation(valid_points, cache["extrinsic"])
    keep_filter = filter_scene_points(valid_points, filter_config)
    filtered_points = valid_points[keep_filter]
    if len(filtered_points) == 0:
        raise WebViewerExportError("DA3 point filtering removed every valid point")
    labels, label_keys = _instance_labels_v2(
        cache, objects, valid_indices[keep_filter], len(flat_points), mask_cache_root
    )
    points = valid_points[keep_filter]
    confidence = flat_confidence[valid][keep_filter]
    colors = flat_colors[valid][keep_filter]
    scaled = np.floor(points / voxel_size)
    limits = np.iinfo(np.int64)
    if (
        not np.isfinite(scaled).all()
        or np.any(scaled < limits.min)
        or np.any(scaled > limits.max)
    ):
        raise WebViewerExportError("DA3 voxel keys exceed int64 representation")
    selected: dict[tuple[int, int, int], int] = {}
    for index, values in enumerate(scaled.astype(np.int64)):
        voxel = tuple(int(value) for value in values)
        previous = selected.get(voxel)
        if previous is None or confidence[index] > confidence[previous]:
            selected[voxel] = index
    keep = np.fromiter(selected.values(), dtype=np.int64, count=len(selected))
    if len(keep) > max_points:
        keep = np.sort(
            np.random.default_rng(42).choice(keep, size=max_points, replace=False)
        )
    points, colors, labels_final = points[keep], colors[keep], labels[keep]
    normals = _estimate_scene_normals(points, cache["extrinsic"])
    order = np.argsort(labels_final, kind="stable")
    positions = np.ascontiguousarray(points[order], dtype="<f4")
    if not np.isfinite(positions).all():
        raise WebViewerExportError("valid DA3 points must be finite float32")
    return {
        "positions": positions,
        "colors": np.ascontiguousarray(colors[order], dtype=np.uint8),
        "normals": np.ascontiguousarray(
            np.rint(normals[order] * 127.0).clip(-127, 127), dtype=np.int8
        ),
        "instance_labels": labels_final[order],
        "label_keys": label_keys,
        "filtered_points": filtered_points,
        "level_rotation": level_rotation,
    }


def _estimate_scene_normals(points: np.ndarray, _extrinsic: np.ndarray) -> np.ndarray:
    """Use a deterministic safe fallback normal for the fixed normals array."""
    return np.tile(np.asarray([0.0, 0.0, 1.0]), (len(points), 1))


def _instance_labels_v2(
    cache: dict[str, Any],
    objects: dict[str, Any],
    valid_indices: np.ndarray,
    flat_count: int,
    mask_cache_root: Path,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Propagate canonical SAM3 masks into filtered DA3 point labels."""
    frame_count, height, width, _ = cache["points"].shape
    frame_for_image = {
        int(image_id): frame for frame, image_id in enumerate(cache["image_ids"])
    }
    by_image: dict[int, list[tuple[int, list[float], int]]] = defaultdict(list)
    label_keys: list[tuple[str, int]] = []
    for global_id in sorted(objects, key=int):
        for instance_index, instance in enumerate(objects[global_id]["instances"]):
            bbox = instance.get("bbox")
            if not _valid_bbox(bbox):
                raise WebViewerExportError(
                    f"global mapping bbox is invalid for global ID {global_id}"
                )
            label_keys.append((global_id, instance_index))
            by_image[int(instance["image_id"])].append(
                (
                    len(label_keys) - 1,
                    [float(value) for value in bbox],
                    int(instance["object_id"]),
                )
            )
    grid = np.full(flat_count, -1, dtype=np.int32).reshape(frame_count, height, width)
    contract = {
        "api": "self_exemplar",
        "threshold": 0.5,
        "image_size": 1008,
        "max_batch_size": 32,
        "max_dets_per_query": 1,
        "clip_to_bbox": True,
    }
    for image_id, instances in by_image.items():
        frame = frame_for_image.get(image_id)
        if frame is None:
            raise WebViewerExportError(
                f"global mapping image {image_id} is absent from cache"
            )
        affine = cache["affine"][frame]
        prompts = tuple(
            ProcessedDetectionPrompt(
                object_id=object_id,
                source_bbox_xyxy=tuple(bbox),
                processed_bbox_xyxy=map_source_bbox_to_processed(
                    bbox, affine, (height, width)
                ),
            )
            for _label, bbox, object_id in sorted(instances, key=lambda item: item[2])
        )
        request = FrameMaskCacheRequest(
            cache_root=mask_cache_root,
            image_id=image_id,
            image_path=Path(str(image_id)),
            source_size_wh=tuple(
                int(value) for value in cache["source_image_sizes"][frame]
            ),
            processed_shape_hw=(height, width),
            source_to_processed_affine=affine,
            detections=prompts,
            inference_contract=contract,
        )
        try:
            result = load_complete_frame_masks(request)
        except FrameMaskCacheError as error:
            raise WebViewerExportError(
                "canonical SAM3 mask cache is incomplete"
            ) from error
        if result.schema != SAM3_MASK_CACHE_SCHEMA:
            raise WebViewerExportError("canonical SAM3 mask cache schema is invalid")
        for label, _bbox, object_id in instances:
            mask = result.masks_by_object_id.get(object_id)
            if mask is None or mask.shape != (height, width) or mask.dtype != np.bool_:
                raise WebViewerExportError(
                    "canonical SAM3 mask shape or dtype is invalid"
                )
            grid[frame][mask & (grid[frame] < 0)] = label
    return grid.reshape(-1)[valid_indices], label_keys


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            for item in value
        )
        and value[2] > value[0]
        and value[3] > value[1]
    )


def _attach_point_index_ranges(
    objects: dict[str, Any], labels: np.ndarray, label_keys: list[tuple[str, int]]
) -> None:
    for label, (global_id, instance_index) in enumerate(label_keys):
        start = int(np.searchsorted(labels, label, side="left"))
        end = int(np.searchsorted(labels, label, side="right"))
        objects[global_id]["instances"][instance_index]["point_index_range"] = [
            start,
            end,
        ]


def _resolve_source_images(images_dir: Path) -> dict[int, Path]:
    """Resolve numeric source-image stems without provenance or hash metadata."""
    if not images_dir.is_dir():
        raise WebViewerExportError(f"source images directory not found: {images_dir}")
    images_by_id: dict[int, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
            continue
        match = _IMAGE_ID_STEM.fullmatch(path.stem)
        if match is None:
            raise WebViewerExportError(
                f"source image {path.name} must have a numeric image-id stem"
            )
        image_id = int(match.group(1))
        if image_id in images_by_id:
            raise WebViewerExportError(
                f"source image ID {image_id} is ambiguous in {images_dir}"
            )
        images_by_id[image_id] = path
    return images_by_id


def _generate_thumbnails(
    objects: dict[str, Any], images_by_id: dict[int, Path]
) -> dict[str, bytes]:
    """Crop every active and removed rich instance into a <=256px JPEG."""
    thumbnails: dict[str, bytes] = {}
    loaded: dict[int, Image.Image] = {}
    try:
        for global_id in sorted(objects, key=int):
            for instance_index, instance in enumerate(objects[global_id]["instances"]):
                image_id = int(instance["image_id"])
                image_path = images_by_id.get(image_id)
                if image_path is None:
                    raise WebViewerExportError(
                        f"source image for observation {image_id} not found"
                    )
                image = loaded.get(image_id)
                if image is None:
                    with Image.open(image_path) as source:
                        image = source.convert("RGB")
                    loaded[image_id] = image
                width, height = image.size
                bbox = instance.get("bbox")
                if not _valid_bbox(bbox):
                    raise WebViewerExportError("global mapping bbox is invalid")
                x1, y1, x2, y2 = (float(value) for value in bbox)
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    raise WebViewerExportError(
                        f"global mapping bbox exceeds source image bounds {width}x{height}"
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
                    (_THUMB_LONG_EDGE, _THUMB_LONG_EDGE), Image.Resampling.LANCZOS
                )
                buffer = io.BytesIO()
                crop.save(buffer, format="JPEG", quality=_THUMB_JPEG_QUALITY)
                relative = f"{_THUMB_DIR}/{global_id}_{instance_index}.jpg"
                thumbnails[relative] = buffer.getvalue()
                instance["thumbnail"] = relative
    finally:
        for image in loaded.values():
            image.close()
    return thumbnails


def _minimal_objects(
    objects: dict[str, Any], *, point_count: int
) -> dict[str, dict[str, Any]]:
    """Publish ordered SKUs, point ranges, and minimal product observations."""
    result: dict[str, dict[str, Any]] = {}
    all_ranges: list[tuple[int, int]] = []
    for global_id, entry in objects.items():
        if not isinstance(global_id, str) or _GLOBAL_ID.fullmatch(global_id) is None:
            raise WebViewerExportError("object global ID is invalid")
        classification = entry.get("classification")
        candidates = (
            classification.get("candidates")
            if isinstance(classification, dict)
            else None
        )
        if not isinstance(candidates, list):
            raise WebViewerExportError("object SKU candidates are invalid")
        ordered_skus = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise WebViewerExportError("object SKU candidate is invalid")
            sku_id, sku_name = candidate.get("sku_id"), candidate.get("sku_name")
            if (
                not isinstance(sku_id, str)
                or not sku_id
                or not isinstance(sku_name, str)
                or not sku_name
            ):
                raise WebViewerExportError("object SKU candidate is invalid")
            ordered_skus.append({"sku_id": sku_id, "sku_name": sku_name})
        ranges = []
        observations = []
        for instance in entry.get("instances", []):
            image_id = instance.get("image_id")
            object_id = instance.get("object_id")
            removed = instance.get("removed")
            thumbnail = instance.get("thumbnail")
            if (
                isinstance(image_id, bool)
                or not isinstance(image_id, int)
                or isinstance(object_id, bool)
                or not isinstance(object_id, int)
                or not isinstance(removed, bool)
                or not isinstance(thumbnail, str)
                or not thumbnail.startswith(f"{_THUMB_DIR}/")
            ):
                raise WebViewerExportError("object observation is invalid")
            observations.append(
                {
                    "image_id": image_id,
                    "object_id": object_id,
                    "removed": removed,
                    "thumbnail": thumbnail,
                }
            )
            point_range = instance.get("point_index_range")
            if point_range is None:
                continue
            if not isinstance(point_range, list) or len(point_range) != 2:
                raise WebViewerExportError("object point range is invalid")
            start, end = point_range
            if (
                any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in point_range
                )
                or start < 0
                or end < start
                or end > point_count
            ):
                raise WebViewerExportError("object point range is invalid")
            if end > start:
                ranges.append([start, end])
                all_ranges.append((start, end))
        result[global_id] = {
            "ordered_skus": ordered_skus,
            "point_ranges": ranges,
            "observations": observations,
        }
    previous_end = 0
    for start, end in sorted(all_ranges):
        if start < previous_end:
            raise WebViewerExportError("object point ranges overlap")
        previous_end = end
    return result


_PLANE_SUBSAMPLE = 200_000
_PLANE_MIN_INLIER_RATIO = 0.05
_PLANE_MAX_TILT_DEG = 60.0
_PLANE_MAX_CANDIDATES = 8
_PLANE_MAX_BELOW_RATIO = 0.15


def _fit_level_rotation(
    valid_points: np.ndarray, extrinsic: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Fit a floor plane from unfiltered points and map it into viewer Y-up."""
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
    min_inliers = _PLANE_MIN_INLIER_RATIO * len(subsample)
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
    filtered_points: np.ndarray, level_rotation: tuple[np.ndarray, bool]
) -> np.ndarray:
    """Center post-filter points after applying the fitted world-to-view rotation."""
    if len(filtered_points) == 0:
        raise WebViewerExportError("DA3 cache has too few points to orient the scene")
    rotation, _leveled = level_rotation
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -np.median(filtered_points @ rotation.T, axis=0)
    return transform


def _shortest_arc_to_y_up(normal: np.ndarray) -> np.ndarray:
    """Return the homogeneous shortest-arc rotation from ``normal`` to +Y."""
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


def _robust_display_bounds(positions: np.ndarray) -> list[float]:
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or len(positions) == 0
        or not np.isfinite(positions).all()
    ):
        raise WebViewerExportError("cannot export finite display_bounds")
    bounds = np.percentile(
        positions.astype(np.float64, copy=False), [1.0, 99.0], axis=0
    )
    return [float(value) for value in (*bounds[0], *bounds[1])]


def _publish_bundle(
    output_dir: Path,
    manifest: dict[str, Any],
    objects: dict[str, Any],
    arrays: dict[str, np.ndarray],
    thumbnails: dict[str, bytes],
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
        thumbs_dir = temporary / _THUMB_DIR
        thumbs_dir.mkdir()
        for relative, payload in thumbnails.items():
            (temporary / relative).write_bytes(payload)
        os.rename(temporary, generation)
        _atomic_replace_current(output_dir / "CURRENT", {"run_id": run_id})
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
