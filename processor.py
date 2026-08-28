"""Strict external-classifier request preparation and Viewer packaging."""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from utils.classification_aggregation import build_resolved_classification

_FRAME_KEYS = frozenset({"classes", "objects"})
_CLASS_KEYS = frozenset({"det", "cls"})
_VIEWER_FILES = (
    "manifest.json",
    "positions.f32.bin",
    "colors.u8.bin",
    "normals.i8.bin",
    "objects.json",
)
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class PreparedRequest:
    dataset_dir: Path


def prepare_request(inputs: Mapping[str, Any], work_root: Path) -> PreparedRequest:
    """Validate BSON-decoded input and write a numeric canonical dataset."""
    if not isinstance(inputs, Mapping):
        raise ValueError("request must be an object")
    try:
        images = inputs["images"]
        skus = inputs["skus"]
    except KeyError as error:
        raise ValueError("request requires images and skus") from error
    if not isinstance(images, list) or not images:
        raise ValueError("images must be a non-empty list")
    if not isinstance(skus, list) or len(skus) != len(images):
        raise ValueError("images and skus count must match")

    decoded_images = [
        _validate_image(payload, index) for index, payload in enumerate(images)
    ]
    canonical_frames = [
        _canonical_frame(payload, index) for index, payload in enumerate(skus)
    ]

    root = Path(work_root)
    dataset_dir = root / "dataset"
    images_dir = dataset_dir / "images"
    detections_dir = dataset_dir / "detections_results"
    images_dir.mkdir(parents=True, exist_ok=False)
    detections_dir.mkdir()
    for index, (image_bytes, frame) in enumerate(zip(decoded_images, canonical_frames)):
        (images_dir / f"{index}.jpg").write_bytes(image_bytes)
        (detections_dir / f"{index}.json").write_text(
            json.dumps(frame, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
    return PreparedRequest(dataset_dir=dataset_dir)


def _validate_image(payload: Any, index: int) -> bytes:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError(f"image {index} must be non-empty bytes")
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise ValueError(f"image {index} is invalid")
    return payload


def _canonical_frame(payload: Any, index: int) -> dict[str, Any]:
    if not isinstance(payload, str) or not payload:
        raise ValueError(f"sku frame {index} must be a JSON string")
    try:
        frame = json.loads(payload, parse_constant=_reject_nonfinite)
    except (TypeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"sku frame {index} is invalid JSON: {error}") from error
    if not isinstance(frame, dict) or set(frame) != _FRAME_KEYS:
        raise ValueError("sku frame top-level keys must be exactly classes and objects")
    classes = frame["classes"]
    objects = frame["objects"]
    if not isinstance(classes, dict) or set(classes) != _CLASS_KEYS:
        raise ValueError("sku frame classes must contain exactly det and cls")
    det_labels = classes["det"]
    labels = classes["cls"]
    _validate_labels(det_labels, "det", split=False)
    _validate_labels(labels, "cls", split=True)
    if not isinstance(objects, list):
        raise ValueError("sku frame objects must be a list")

    canonical_objects = [
        _canonical_object(obj, det_labels, labels, index, object_index)
        for object_index, obj in enumerate(objects)
    ]
    return {
        "skus": [
            {
                "classes": deepcopy(classes),
                "objects": canonical_objects,
            }
        ]
    }


def _canonical_object(
    obj: Any,
    det_labels: list[Any],
    labels: list[Any],
    frame_index: int,
    object_index: int,
) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(
            f"sku frame {frame_index} object {object_index} must be an object"
        )
    if "features" in obj:
        raise ValueError("features are not accepted")
    position = obj.get("position")
    if (
        not isinstance(position, list)
        or len(position) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in position
        )
        or not all(math.isfinite(float(value)) for value in position)
    ):
        raise ValueError(
            f"sku frame {frame_index} object {object_index} position is invalid"
        )
    object_classes = obj.get("classes")
    object_confidences = obj.get("confidences")
    if not isinstance(object_classes, dict) or set(object_classes) != _CLASS_KEYS:
        raise ValueError("sku object classes must contain exactly det and cls")
    detector_index = object_classes["det"]
    if (
        isinstance(detector_index, bool)
        or not isinstance(detector_index, int)
        or detector_index < 0
        or detector_index >= len(det_labels)
    ):
        raise ValueError("sku object det index is out of range")
    class_index = object_classes["cls"]
    if isinstance(class_index, bool) or not isinstance(class_index, int):
        raise ValueError("sku object cls index must be an integer")
    if class_index < 0 or class_index >= len(labels):
        raise ValueError("sku object cls index is out of range")
    if not isinstance(object_confidences, dict) or set(object_confidences) != _CLASS_KEYS:
        raise ValueError("sku object confidences must contain exactly det and cls")
    detector_confidence = object_confidences["det"]
    _validate_confidence(detector_confidence, "det")
    confidence = object_confidences["cls"]
    _validate_confidence(confidence, "cls")

    output = deepcopy(obj)
    output["classification"] = build_resolved_classification(
        51, labels[class_index], float(confidence)
    )
    return output


def _validate_labels(value: Any, name: str, *, split: bool) -> None:
    if not isinstance(value, list):
        raise ValueError(f"sku frame classes.{name} must be a list")
    for label in value:
        if not isinstance(label, str):
            raise ValueError(f"sku frame classes.{name} labels must be strings")
        if split:
            build_resolved_classification(51, label, 0.0)


def _validate_confidence(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(
            f"sku object {name} confidence must be finite and within [0, 1]"
        )


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value is invalid: {value}")


def pack_viewer_bundle(viewer_root: Path) -> bytes:
    """Pack only CURRENT and its selected fixed Viewer generation."""
    viewer_root = Path(viewer_root)
    current_path = viewer_root / "CURRENT"
    if current_path.is_symlink() or not current_path.is_file():
        raise ValueError("viewer CURRENT is missing")
    try:
        pointer = json.loads(
            current_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"viewer CURRENT is invalid: {error}") from error
    run_id = pointer.get("run_id") if isinstance(pointer, dict) else None
    if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("viewer CURRENT run_id is invalid")
    run_root = viewer_root / "runs" / run_id
    if run_root.is_symlink() or not run_root.is_dir():
        raise ValueError("viewer CURRENT run is missing")
    viewer_root_resolved = viewer_root.resolve()
    run_root_resolved = run_root.resolve()
    if not _is_relative_to(run_root_resolved, viewer_root_resolved):
        raise ValueError("viewer CURRENT run is outside viewer root")
    paths = [run_root / name for name in _VIEWER_FILES]
    for path in paths:
        if (
            path.is_symlink()
            or not path.is_file()
            or not _is_relative_to(path.resolve(), run_root_resolved)
        ):
            raise ValueError("viewer run is missing a required file")
    thumbs_root = run_root / "thumbs"
    if (
        thumbs_root.is_symlink()
        or not thumbs_root.is_dir()
        or not _is_relative_to(thumbs_root.resolve(), run_root_resolved)
    ):
        raise ValueError("viewer run thumbs directory is missing")
    thumbs = []
    for path in sorted(thumbs_root.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise ValueError("viewer thumbnail symlink is not allowed")
        if not _safe_thumbnail_name(path.name):
            raise ValueError("viewer thumbnail name is invalid")
        if (
            path.suffix == ".jpg"
            and path.is_file()
            and _is_relative_to(path.resolve(), run_root_resolved)
        ):
            thumbs.append(path)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("CURRENT", current_path.read_bytes())
        for path, name in zip(paths, _VIEWER_FILES):
            archive.writestr(f"runs/{run_id}/{name}", path.read_bytes())
        for path in thumbs:
            archive.writestr(f"runs/{run_id}/thumbs/{path.name}", path.read_bytes())
    return buffer.getvalue()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_thumbnail_name(name: str) -> bool:
    return (
        bool(name)
        and "\\" not in name
        and all(ord(character) >= 32 and ord(character) != 127 for character in name)
    )


def build_success_response(global_skus_path: Path, viewer_root: Path) -> dict[str, Any]:
    """Read published global SKU strings and return the exact success envelope."""
    try:
        global_skus = json.loads(
            Path(global_skus_path).read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"global_skus is invalid: {error}") from error
    if not isinstance(global_skus, list):
        raise ValueError("global_skus must be a list")
    for index, item in enumerate(global_skus):
        if not isinstance(item, str):
            raise ValueError(f"global_skus[{index}] must be a JSON string")
        try:
            decoded = json.loads(item, parse_constant=_reject_nonfinite)
        except (TypeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"global_skus[{index}] is invalid JSON: {error}") from error
        if not isinstance(decoded, dict):
            raise ValueError(f"global_skus[{index}] must decode to an object")
    return {
        "global_skus": global_skus,
        "viewer_bundle": pack_viewer_bundle(viewer_root),
    }
