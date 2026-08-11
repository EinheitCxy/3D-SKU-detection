"""Read-only stage for calibrated sums of ground-stack detection boxes."""

from __future__ import annotations

import json
from collections import defaultdict
from math import isfinite
from pathlib import Path
from typing import Any

import cv2

from utils.detection_objects import flatten_detection_objects
from utils.ground_stack_area import (
    BBoxAreaError,
    RejectedInstance,
    SelectedInstance,
    calibrated_bbox_area_cm2,
    calibrate_from_anchor,
    select_best_instances,
    validate_bbox,
    validate_bbox_within_image_bounds,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SCHEMA_VERSION = "1.0"
METRIC_NAME = "calibrated_bbox_area_sum"


def _image_path(images_dir: Path, image_id: int) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        for candidate in (
            images_dir / f"{image_id}{extension}",
            images_dir / f"{image_id}{extension.upper()}",
        ):
            if candidate.is_file():
                return candidate
    return None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _source_image_bounds(images_dir: Path, image_id: int) -> tuple[int, int]:
    source_path = _image_path(images_dir, image_id)
    if source_path is None:
        raise BBoxAreaError(f"source image is missing for frame {image_id}")
    image = cv2.imread(str(source_path))
    if image is None:
        raise BBoxAreaError(f"source image cannot be read for frame {image_id}")
    height, width = image.shape[:2]
    return width, height


def _finite_report_value(value: float) -> float | None:
    return value if isfinite(value) else None


def _load_anchor_bbox(dataset_dir: Path, frame_id: int, object_id: int) -> tuple[float, ...]:
    detection_path = dataset_dir / "detections_results" / f"{frame_id}.json"
    if not detection_path.is_file():
        raise BBoxAreaError(f"anchor detection does not exist: {detection_path}")
    detection = json.loads(detection_path.read_text(encoding="utf-8"))
    objects = flatten_detection_objects(detection)
    if object_id < 0 or object_id >= len(objects):
        raise BBoxAreaError(
            f"anchor object index {object_id} is outside detection objects ({len(objects)})"
        )
    bbox = objects[object_id].get("position")
    if bbox is None:
        raise BBoxAreaError("anchor object does not have a position bbox")
    return tuple(bbox)


def _anchor_mapping_status(
    global_mapping: dict[str, list[dict[str, Any]]],
    frame_id: int,
    object_id: int,
    anchor_bbox: tuple[float, ...],
) -> str:
    matching_identity = False
    for observations in global_mapping.values():
        for observation in observations:
            try:
                if (
                    int(observation["image_id"]) == frame_id
                    and int(observation["object_id"]) == object_id
                ):
                    matching_identity = True
                    if validate_bbox(observation.get("bbox")) == validate_bbox(anchor_bbox):
                        return "matched"
            except (KeyError, TypeError, ValueError):
                continue
    return "bbox_mismatch" if matching_identity else "missing"


def _instance_payload(instance: SelectedInstance, area_cm2: float) -> dict[str, Any]:
    return {
        "global_id": instance.global_id,
        "image_id": instance.image_id,
        "object_id": instance.object_id,
        "bbox": list(instance.bbox),
        "source_area_px2": instance.source_area_px2,
        "area_cm2": area_cm2,
    }


def _rejection_payload(instance: RejectedInstance) -> dict[str, str]:
    return {"global_id": instance.global_id, "reason": instance.reason}


def _write_annotations(
    images_dir: Path, output_dir: Path, instances: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    annotations_dir = output_dir / "annotated_frames"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        by_image[int(instance["image_id"])].append(instance)

    artifacts: list[str] = []
    warnings: list[str] = []
    for image_id, image_instances in sorted(by_image.items()):
        source_path = _image_path(images_dir, image_id)
        if source_path is None:
            warnings.append(f"source image is missing for frame {image_id}")
            continue
        image = cv2.imread(str(source_path))
        if image is None:
            warnings.append(f"source image cannot be read for frame {image_id}")
            continue

        for instance in image_instances:
            x1, y1, x2, y2 = (round(value) for value in instance["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
            label = f"gid={instance['global_id']} {instance['area_cm2']:.2f} cm2"
            cv2.putText(
                image,
                label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 120, 0),
                1,
                cv2.LINE_AA,
            )

        output_path = annotations_dir / f"{image_id}.jpg"
        if cv2.imwrite(str(output_path), image):
            artifacts.append(str(output_path))
        else:
            warnings.append(f"annotation write failed for frame {image_id}")
    return artifacts, warnings


def _base_report(
    dataset_dir: Path,
    anchor_frame: int,
    anchor_object: int,
    anchor_width_cm: float,
    anchor_height_cm: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "rejected",
        "metric": METRIC_NAME,
        "unit": {"instance": "cm2", "total": "m2"},
        "value_cm2": None,
        "value_m2": None,
        "accepted_global_ids": 0,
        "rejected_global_ids": 0,
        "calibration": {
            "anchor_frame": anchor_frame,
            "anchor_object": anchor_object,
            "anchor_width_cm": _finite_report_value(anchor_width_cm),
            "anchor_height_cm": _finite_report_value(anchor_height_cm),
            "method": "axis_aligned_bbox_scale_same_frame",
        },
        "warnings": [],
        "artifacts": {
            "instances": "selected_instances.json",
            "annotated_frames_dir": "annotated_frames",
        },
        "source": {
            "dataset": str(dataset_dir),
            "global_mapping": None,
        },
    }


def run_ground_stack_area(
    dataset_path: str,
    save_root: Path,
    anchor_frame: int,
    anchor_object: int,
    anchor_width_cm: float,
    anchor_height_cm: float,
) -> dict[str, Any]:
    """Measure one calibrated bbox per global ID without mutating inputs."""
    dataset_dir = Path(dataset_path)
    output_dir = Path(save_root) / dataset_dir.name / "ground_stack_area"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "measurement_report.json"
    instances_path = output_dir / "selected_instances.json"
    report = _base_report(
        dataset_dir,
        anchor_frame,
        anchor_object,
        anchor_width_cm,
        anchor_height_cm,
    )

    selected_payload: list[dict[str, Any]] = []
    rejected_payload: list[dict[str, str]] = []
    try:
        mapping_path = (
            Path(save_root)
            / dataset_dir.name
            / "dedup_detections"
            / "global_mapping.json"
        )
        if not mapping_path.is_file():
            raise BBoxAreaError(f"global mapping does not exist: {mapping_path}")
        global_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(global_mapping, dict):
            raise BBoxAreaError("global mapping must be a JSON object")
        report["source"]["global_mapping"] = str(mapping_path)

        images_dir = dataset_dir / "images"
        anchor_bbox = _load_anchor_bbox(dataset_dir, anchor_frame, anchor_object)
        anchor_width_px, anchor_height_px = _source_image_bounds(images_dir, anchor_frame)
        validate_bbox_within_image_bounds(
            anchor_bbox, anchor_width_px, anchor_height_px
        )
        anchor_mapping_status = _anchor_mapping_status(
            global_mapping, anchor_frame, anchor_object, anchor_bbox
        )
        if anchor_mapping_status == "missing":
            raise BBoxAreaError("anchor object is not represented in global mapping")
        if anchor_mapping_status == "bbox_mismatch":
            raise BBoxAreaError("anchor bbox does not match global mapping")
        calibration = calibrate_from_anchor(
            anchor_bbox, anchor_width_cm, anchor_height_cm
        )
        selected, rejected = select_best_instances(
            global_mapping, required_image_id=anchor_frame
        )
        rejected_payload = [_rejection_payload(instance) for instance in rejected]
        image_bounds: dict[int, tuple[int, int]] = {}

        for instance in selected:
            try:
                image_bounds.setdefault(
                    instance.image_id,
                    _source_image_bounds(images_dir, instance.image_id),
                )
                width, height = image_bounds[instance.image_id]
                validate_bbox_within_image_bounds(instance.bbox, width, height)
                area_cm2 = calibrated_bbox_area_cm2(instance.bbox, calibration)
                selected_payload.append(_instance_payload(instance, area_cm2))
            except BBoxAreaError as exc:
                rejected_payload.append(
                    {"global_id": instance.global_id, "reason": str(exc)}
                )

        report["accepted_global_ids"] = len(selected_payload)
        report["rejected_global_ids"] = len(rejected_payload)
        if not selected_payload:
            report["warnings"] = ["no global ID has a valid measured bbox"]
        else:
            total_cm2 = sum(instance["area_cm2"] for instance in selected_payload)
            report["value_cm2"] = total_cm2
            report["value_m2"] = total_cm2 / 10_000.0
            report["status"] = (
                "accepted_with_warnings" if rejected_payload else "accepted"
            )
            annotations, annotation_warnings = _write_annotations(
                dataset_dir / "images", output_dir, selected_payload
            )
            report["artifacts"]["annotated_frames"] = annotations
            report["warnings"] = annotation_warnings
            if annotation_warnings and report["status"] == "accepted":
                report["status"] = "accepted_with_warnings"
    except (BBoxAreaError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        report["warnings"] = [str(exc)]

    _write_json(
        instances_path,
        {
            "schema_version": SCHEMA_VERSION,
            "metric": METRIC_NAME,
            "instances": selected_payload,
            "rejected": rejected_payload,
        },
    )
    _write_json(report_path, report)
    return {
        "success": report["status"] != "rejected",
        "status": report["status"],
        "report_path": str(report_path),
        "instances_path": str(instances_path),
    }
