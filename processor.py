"""Strict external-classifier request preparation and Viewer packaging."""

from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import PROJECT_ROOT as MAIN_PROJECT_ROOT, SKUDetectionMain
from src.web_viewer_export import export_web_viewer_bundle
from utils.classification_aggregation import build_resolved_classification
from utils.matching_algorithms import PI3_SCENE_CACHE
from utils.sku_matching_system import _DA3_IMAGE_CACHE, _DA3_TRANSFORMS_CACHE

_FRAME_KEYS = frozenset({"classes", "objects"})
_CLASS_KEYS = frozenset({"det", "cls"})
_VIEWER_FILES = (
    "manifest.json",
    "positions.f32.bin",
    "colors.u8.bin",
    "normals.i8.bin",
    "objects.json",
)
_REQUIRED_STAGES = (
    "validation",
    "reconstruction",
    "matching",
    "improved_analysis",
    "classification",
    "dedup",
)


@dataclass(frozen=True)
class PreparedRequest:
    dataset_dir: Path


def process(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Run one complete mapping request and return its BSON-ready result."""
    with tempfile.TemporaryDirectory(prefix="global-id-mapping-") as work_dir:
        try:
            work_root = Path(work_dir)
            prepared = prepare_request(inputs, work_root)
            result = run_mapping_request(
                prepared.dataset_dir,
                work_root / "outputs",
                work_root / "viewer",
                os.environ["DA3_MODEL_PATH"],
            )
            return build_success_response(
                Path(result["global_skus_path"]), Path(result["viewer_dir"])
            )
        finally:
            PI3_SCENE_CACHE.clear()
            _DA3_IMAGE_CACHE.clear()
            _DA3_TRANSFORMS_CACHE.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
    if (
        not isinstance(object_confidences, dict)
        or set(object_confidences) != _CLASS_KEYS
    ):
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


def pack_viewer_bundle(generation_dir: Path) -> bytes:
    """Pack one selected Viewer generation with flat archive member names."""
    generation_dir = Path(generation_dir)
    paths = [generation_dir / name for name in _VIEWER_FILES]
    thumbs = sorted((generation_dir / "thumbs").glob("*.jpg"))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for path, name in zip(paths, _VIEWER_FILES):
            archive.writestr(name, path.read_bytes())
        for path in thumbs:
            archive.writestr(f"thumbs/{path.name}", path.read_bytes())
    return buffer.getvalue()


def build_success_response(
    global_skus_path: Path, generation_dir: Path
) -> dict[str, Any]:
    """Read published global SKU strings and return the exact success envelope."""
    global_skus = json.loads(
        Path(global_skus_path).read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(global_skus, list):
        raise ValueError("global_skus must be a list")
    for index, item in enumerate(global_skus):
        if not isinstance(item, str):
            raise ValueError(f"global_skus[{index}] must be a JSON string")
        decoded = json.loads(item, parse_constant=_reject_nonfinite)
        if not isinstance(decoded, dict):
            raise ValueError(f"global_skus[{index}] must decode to an object")
    return {
        "global_skus": global_skus,
        "viewer_bundle": pack_viewer_bundle(generation_dir),
    }


def run_mapping_request(
    dataset_dir: Path,
    output_root: Path,
    viewer_root: Path,
    model_path: str,
) -> dict[str, str]:
    """Run DA3 mapping and export the selected Viewer generation."""
    dataset_dir = Path(dataset_dir)
    output_root = Path(output_root)
    viewer_root = Path(viewer_root)

    pipeline = SKUDetectionMain()
    pipeline.save_root = output_root
    pipeline.match_backend = "da3"
    pipeline.classifier_enabled = False
    pipeline.config_path = MAIN_PROJECT_ROOT / "config.yaml"

    summary = pipeline.run_complete_pipeline(
        str(dataset_dir), algorithm="3d", model_path=model_path
    )
    for stage in _REQUIRED_STAGES:
        if summary.get(stage) is not True:
            raise RuntimeError(f"pipeline stage {stage} did not succeed")

    dataset_output = output_root / dataset_dir.name
    export_result = export_web_viewer_bundle(
        dataset_name=dataset_dir.name,
        da3_cache_path=dataset_output / "da3_cache" / "predictions.npz",
        global_mapping_path=dataset_output / "dedup_detections" / "global_mapping.json",
        output_dir=viewer_root,
        source_images_dir=dataset_dir / "images",
        sam3_mask_cache_root=dataset_output / "sam3_mask_cache" / "v2",
    )
    return {
        "global_skus_path": str(
            dataset_output / "dedup_detections" / "global_skus.json"
        ),
        "viewer_dir": str(Path(export_result["manifest_path"]).parent),
    }
