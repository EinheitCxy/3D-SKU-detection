"""Read-only DA3 metric area stage for deduplicated detection boxes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.ground_stack_area import BBoxAreaError, validate_bbox_within_image_bounds


METRIC_NAME = "da3_metric_bbox_area_sum"
SCHEMA_VERSION = "1.0"
MIN_VALID_POINTS = 64
MIN_VALID_FRACTION = 0.5
MIN_CONFIDENCE = 1.0
MAX_NORMAL_SPAN_M = 0.06
CORE_FRACTION = 0.25
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class DA3MetricAreaError(BBoxAreaError):
    """Raised when a DA3 observation cannot support a metric area estimate."""


def _image_path(images_dir: Path, image_id: int) -> Path:
    for suffix in IMAGE_EXTENSIONS:
        path = images_dir / f"{image_id}{suffix}"
        if path.is_file():
            return path
    raise DA3MetricAreaError(f"source image is missing for frame {image_id}")


def _image_size(images_dir: Path, image_id: int) -> tuple[int, int]:
    image = cv2.imread(str(_image_path(images_dir, image_id)))
    if image is None:
        raise DA3MetricAreaError(f"source image cannot be read for frame {image_id}")
    height, width = image.shape[:2]
    return width, height


def _frame_index(image_ids: np.ndarray, image_id: int) -> int:
    matches = np.flatnonzero(image_ids == image_id)
    if len(matches) != 1:
        raise DA3MetricAreaError(f"DA3 cache has no unique frame for image_id {image_id}")
    return int(matches[0])


def _cached_source_image_size(
    source_image_sizes: np.ndarray, frame: int
) -> tuple[int, int]:
    size = np.asarray(source_image_sizes[frame], dtype=np.float64)
    if size.shape != (2,) or not np.isfinite(size).all():
        raise DA3MetricAreaError("DA3 cache source_image_sizes is invalid")
    if np.any(size <= 0) or not np.allclose(size, np.rint(size)):
        raise DA3MetricAreaError("DA3 cache source_image_sizes must be positive integers")
    width, height = np.rint(size).astype(np.int64)
    return int(width), int(height)


def _roi_bounds(
    bbox: Any,
    source_width: int,
    source_height: int,
    source_to_processed_affine: np.ndarray,
    point_width: int,
    point_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = validate_bbox_within_image_bounds(
        bbox, source_width, source_height
    )
    affine = np.asarray(source_to_processed_affine, dtype=np.float64)
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        raise DA3MetricAreaError("DA3 source-to-processed affine is invalid")
    corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    mapped = corners @ affine[:, :2].T + affine[:, 2]
    tolerance = 1e-6
    u1, v1 = np.floor(mapped.min(axis=0) + tolerance).astype(int)
    u2, v2 = np.ceil(mapped.max(axis=0) - tolerance).astype(int)
    if u1 < 0 or v1 < 0 or u2 > point_width or v2 > point_height:
        raise DA3MetricAreaError("bbox is outside the DA3 processed image")
    if u2 <= u1 or v2 <= v1:
        raise DA3MetricAreaError("bbox becomes empty in DA3 point-cloud coordinates")
    return u1, v1, u2, v2


def _measure_roi(points: np.ndarray, confidence: np.ndarray) -> dict[str, float]:
    roi_height, roi_width = points.shape[:2]
    full_valid = (
        np.isfinite(points).all(axis=-1)
        & (np.linalg.norm(points, axis=-1) > 0.0)
        & np.isfinite(confidence)
        & (confidence >= MIN_CONFIDENCE)
    )
    full_valid_points = int(full_valid.sum())
    full_valid_fraction = float(full_valid.mean())
    if full_valid_points < MIN_VALID_POINTS or full_valid_fraction < MIN_VALID_FRACTION:
        raise DA3MetricAreaError(
            "DA3 bbox lacks valid coverage "
            f"({full_valid_points} points, {full_valid_fraction:.2%})"
        )
    core_y1 = int(np.floor(roi_height * CORE_FRACTION))
    core_y2 = int(np.ceil(roi_height * (1.0 - CORE_FRACTION)))
    core_x1 = int(np.floor(roi_width * CORE_FRACTION))
    core_x2 = int(np.ceil(roi_width * (1.0 - CORE_FRACTION)))
    core_points = points[core_y1:core_y2, core_x1:core_x2]
    core_confidence = confidence[core_y1:core_y2, core_x1:core_x2]
    if min(core_points.shape[:2]) < 3:
        raise DA3MetricAreaError("DA3 bbox center is too small for local metric scale")

    flat_points = core_points.reshape(-1, 3)
    flat_confidence = core_confidence.reshape(-1)
    core_valid = full_valid[core_y1:core_y2, core_x1:core_x2]
    valid = core_valid.reshape(-1)
    flat_points = flat_points[valid]
    flat_confidence = flat_confidence[valid]
    core_valid_fraction = float(valid.mean())
    if len(flat_points) < 16 or core_valid_fraction < MIN_VALID_FRACTION:
        raise DA3MetricAreaError(
            "DA3 bbox center lacks valid coverage "
            f"({len(flat_points)} points, {core_valid_fraction:.2%})"
        )

    center = np.median(flat_points, axis=0)
    centered = flat_points - center
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    normal = vectors[2]
    normal_offsets = centered @ normal
    normal_span_m = float(
        np.quantile(normal_offsets, 0.95) - np.quantile(normal_offsets, 0.05)
    )
    if normal_span_m > MAX_NORMAL_SPAN_M:
        raise DA3MetricAreaError(
            f"DA3 bbox is not planar enough (normal span {normal_span_m:.3f} m)"
        )

    residual = np.abs(normal_offsets)
    mad = float(np.median(np.abs(residual - np.median(residual))))
    inlier_limit = max(0.005, 3.0 * 1.4826 * mad)
    inliers = residual <= inlier_limit
    inlier_ratio = float(np.mean(inliers))
    if int(inliers.sum()) < 16 or inlier_ratio < 0.6:
        raise DA3MetricAreaError("DA3 bbox has insufficient planar inliers")

    inlier_grid = np.zeros(core_points.shape[:2], dtype=bool)
    inlier_grid.reshape(-1)[np.flatnonzero(valid)] = inliers
    delta_x = core_points[:, 1:] - core_points[:, :-1]
    delta_y = core_points[1:, :] - core_points[:-1, :]
    cell_areas = np.linalg.norm(
        np.cross(delta_x[:-1, :], delta_y[:, :-1]), axis=-1
    )
    valid_cells = (
        inlier_grid[:-1, :-1]
        & inlier_grid[1:, :-1]
        & inlier_grid[:-1, 1:]
        & inlier_grid[1:, 1:]
        & np.isfinite(cell_areas)
    )
    if int(valid_cells.sum()) < 16:
        raise DA3MetricAreaError("DA3 bbox center has insufficient planar grid cells")
    local_cell_area_m2 = float(np.median(cell_areas[valid_cells]))
    area_m2 = local_cell_area_m2 * (roi_height - 1) * (roi_width - 1)
    if not np.isfinite(area_m2) or area_m2 <= 0:
        raise DA3MetricAreaError("DA3 metric area is not positive and finite")
    quality_score = (
        min(1.0, full_valid_points / 1000.0)
        * inlier_ratio
        * max(0.0, 1.0 - normal_span_m / MAX_NORMAL_SPAN_M)
        * min(1.0, float(np.median(flat_confidence)) / 10.0)
    )
    return {
        "area_m2": area_m2,
        "valid_points": full_valid_points,
        "valid_point_fraction": full_valid_fraction,
        "core_valid_points": int(len(flat_points)),
        "core_valid_fraction": core_valid_fraction,
        "inlier_ratio": inlier_ratio,
        "normal_span_m": normal_span_m,
        "median_confidence": float(np.median(flat_confidence)),
        "local_cell_area_m2": local_cell_area_m2,
        "quality_score": quality_score,
    }


def _annotate(
    images_dir: Path, output_dir: Path, instances: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    annotations_dir = output_dir / "annotated_frames"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        by_frame[int(instance["image_id"])].append(instance)

    paths: list[str] = []
    warnings: list[str] = []
    for image_id, frame_instances in sorted(by_frame.items()):
        try:
            image = cv2.imread(str(_image_path(images_dir, image_id)))
            if image is None:
                raise DA3MetricAreaError(f"source image cannot be read for frame {image_id}")
            for instance in frame_instances:
                x1, y1, x2, y2 = (round(value) for value in instance["bbox"])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
                cv2.putText(
                    image,
                    f"gid={instance['global_id']} {instance['area_m2']:.3f} m2",
                    (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 120, 0),
                    1,
                    cv2.LINE_AA,
                )
            path = annotations_dir / f"{image_id}.jpg"
            if not cv2.imwrite(str(path), image):
                raise DA3MetricAreaError(f"annotation write failed for frame {image_id}")
            paths.append(str(path))
        except DA3MetricAreaError as exc:
            warnings.append(str(exc))
    return paths, warnings


def run_da3_metric_area(dataset_path: str, save_root: Path) -> dict[str, Any]:
    """Sum one quality-selected DA3 metric bbox area per global ID."""
    dataset_dir = Path(dataset_path)
    output_dir = Path(save_root) / dataset_dir.name / "ground_stack_area"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "measurement_report.json"
    instances_path = output_dir / "selected_instances.json"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "rejected",
        "metric": METRIC_NAME,
        "unit": {"instance": "m2", "total": "m2"},
        "value_m2": None,
        "accepted_global_ids": 0,
        "rejected_global_ids": 0,
        "rejections": [],
        "calibration": {"method": "DA3 metric point cloud", "reference_required": False},
        "quality_gate": {
            "min_valid_points": MIN_VALID_POINTS,
            "min_valid_fraction": MIN_VALID_FRACTION,
            "min_confidence": MIN_CONFIDENCE,
            "max_normal_span_m": MAX_NORMAL_SPAN_M,
        },
        "warnings": [],
        "artifacts": {"instances": "selected_instances.json", "annotated_frames": []},
        "source": {"dataset": str(dataset_dir), "global_mapping": None, "da3_cache": None},
    }
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    try:
        mapping_path = Path(save_root) / dataset_dir.name / "dedup_detections" / "global_mapping.json"
        cache_path = Path(save_root) / dataset_dir.name / "da3_cache" / "predictions.npz"
        if not mapping_path.is_file():
            raise DA3MetricAreaError(f"global mapping does not exist: {mapping_path}")
        if not cache_path.is_file():
            raise DA3MetricAreaError(f"DA3 cache does not exist: {cache_path}")
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise DA3MetricAreaError("global mapping must be a JSON object")
        cache = np.load(cache_path, allow_pickle=False)
        points = cache["world_points"]
        confidence = cache["world_points_conf"]
        image_ids = cache["image_ids"]
        source_image_sizes = cache["source_image_sizes"]
        source_to_processed_affine = cache["source_to_processed_affine"]
        if points.ndim != 4 or points.shape[-1] != 3 or confidence.shape != points.shape[:3]:
            raise DA3MetricAreaError("DA3 cache geometry arrays have incompatible shapes")
        if image_ids.ndim != 1 or len(image_ids) != len(points):
            raise DA3MetricAreaError("DA3 cache image_ids do not match point-cloud frames")
        if source_image_sizes.shape != (len(points), 2):
            raise DA3MetricAreaError(
                "DA3 cache source_image_sizes does not match point-cloud frames"
            )
        if source_to_processed_affine.shape != (len(points), 2, 3):
            raise DA3MetricAreaError(
                "DA3 cache source_to_processed_affine does not match point-cloud frames"
            )
        report["source"]["global_mapping"] = str(mapping_path)
        report["source"]["da3_cache"] = str(cache_path)
        images_dir = dataset_dir / "images"
        point_height, point_width = points.shape[1:3]

        for global_id, observations in sorted(mapping.items(), key=lambda item: str(item[0])):
            candidates: list[dict[str, Any]] = []
            errors: list[str] = []
            diagnostics: list[dict[str, Any]] = []
            for observation_index, observation in enumerate(observations):
                try:
                    if isinstance(observation["image_id"], bool) or isinstance(observation["object_id"], bool):
                        raise DA3MetricAreaError("observation indexes must be integers")
                    image_id = int(observation["image_id"])
                    object_id = int(observation["object_id"])
                    if image_id != observation["image_id"] or object_id != observation["object_id"]:
                        raise DA3MetricAreaError("observation indexes must be integers")
                    source_width, source_height = _image_size(images_dir, image_id)
                    frame = _frame_index(image_ids, image_id)
                    cached_width, cached_height = _cached_source_image_size(
                        source_image_sizes, frame
                    )
                    if (source_width, source_height) != (cached_width, cached_height):
                        raise DA3MetricAreaError(
                            f"source image size changed for frame {image_id}; rebuild DA3 cache"
                        )
                    bbox = validate_bbox_within_image_bounds(
                        observation["bbox"], source_width, source_height
                    )
                    u1, v1, u2, v2 = _roi_bounds(
                        bbox,
                        source_width,
                        source_height,
                        source_to_processed_affine[frame],
                        point_width,
                        point_height,
                    )
                    measurement = _measure_roi(
                        points[frame, v1:v2, u1:u2], confidence[frame, v1:v2, u1:u2]
                    )
                    candidate = {
                        "global_id": str(global_id),
                        "image_id": image_id,
                        "object_id": object_id,
                        "bbox": list(bbox),
                        "observation_index": observation_index,
                        **measurement,
                    }
                    candidates.append(candidate)
                    diagnostics.append(
                        {
                            "observation_index": observation_index,
                            "status": "eligible",
                            **candidate,
                        }
                    )
                except (DA3MetricAreaError, KeyError, TypeError, ValueError) as exc:
                    reason = str(exc)
                    errors.append(reason)
                    diagnostics.append(
                        {
                            "observation_index": observation_index,
                            "status": "rejected",
                            "reason": reason,
                        }
                    )
            if not candidates:
                rejected.append(
                    {
                        "global_id": str(global_id),
                        "reason": errors[0] if errors else "no observations",
                        "observation_diagnostics": diagnostics,
                    }
                )
                continue
            candidates.sort(key=lambda item: (-item["quality_score"], item["image_id"], item["object_id"]))
            selected_candidate = candidates[0]
            selected_candidate["selected_observation_index"] = selected_candidate.pop(
                "observation_index"
            )
            selected_candidate["observation_diagnostics"] = diagnostics
            selected.append(selected_candidate)

        report["accepted_global_ids"] = len(selected)
        report["rejected_global_ids"] = len(rejected)
        report["rejections"] = rejected
        rejection_warnings = [
            f"global_id {item['global_id']}: {item['reason']}" for item in rejected
        ]
        if selected:
            report["value_m2"] = float(sum(item["area_m2"] for item in selected))
            report["status"] = "accepted_with_warnings" if rejected else "accepted"
            annotations, warnings = _annotate(images_dir, output_dir, selected)
            report["artifacts"]["annotated_frames"] = annotations
            report["warnings"] = rejection_warnings + warnings
            if warnings and report["status"] == "accepted":
                report["status"] = "accepted_with_warnings"
        else:
            report["warnings"] = rejection_warnings or [
                "no global ID has a valid DA3 metric measurement"
            ]
    except (DA3MetricAreaError, OSError, ValueError, TypeError, json.JSONDecodeError, KeyError) as exc:
        report["warnings"] = [str(exc)]

    instances_path.write_text(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "metric": METRIC_NAME, "instances": selected, "rejected": rejected},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"success": bool(selected), "status": report["status"], "report_path": str(report_path), "instances_path": str(instances_path)}
