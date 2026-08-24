from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2
import numpy as np

from modules.personalcare_classifier.source.contracts import resolved_classification
from modules.personalcare_classifier.source.processor import PersonalcarePredictor


class BatchPredictor(Protocol):
    project_id: int

    def predict(self, crops: list[np.ndarray]) -> list[tuple[str, float]]: ...


@dataclass(frozen=True)
class ClassificationRunResult:
    run_id: str
    detection_dir: Path
    result_path: Path
    frame_count: int
    object_count: int
    unavailable_count: int

    def to_cli_payload(self) -> dict[str, object]:
        return {
            "success": True,
            "run_id": self.run_id,
            "detection_dir": str(self.detection_dir),
            "result_path": str(self.result_path),
            "frame_count": self.frame_count,
            "object_count": self.object_count,
            "unavailable_count": self.unavailable_count,
        }


@dataclass
class _ObjectReference:
    sku: dict[str, object]
    object_data: dict[str, object]


def classify_dataset(
    dataset: Path,
    output_root: Path,
    device: str,
    predictor: BatchPredictor,
) -> ClassificationRunResult:
    del device
    dataset = Path(dataset)
    frame_paths = _matching_frame_paths(dataset)
    run_id = f"{time.time_ns()}-{os.getpid()}"
    publication_root = output_root / dataset.name / "personalcare_classification"
    runs_dir = publication_root / "runs"
    temporary_run = runs_dir / f".{run_id}.tmp"
    run_dir = runs_dir / run_id
    detection_dir = run_dir / "detections"
    result_path = run_dir / "result.json"
    temporary_detections = temporary_run / "detections"
    runs_dir.mkdir(parents=True, exist_ok=True)
    temporary_run.mkdir()
    temporary_detections.mkdir()

    object_count = 0
    unavailable_count = 0
    for frame_id, image_path, detection_path in frame_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to read image: {image_path}")
        payload = json.loads(detection_path.read_text(encoding="utf-8"))
        references = _object_references(payload)
        object_count += len(references)
        valid_crops: list[np.ndarray] = []
        valid_references: list[_ObjectReference] = []
        for reference in references:
            crop = _valid_crop(image, reference.object_data.get("position"))
            if crop is None:
                reference.object_data["classification"] = _unavailable_classification(
                    predictor.project_id
                )
                unavailable_count += 1
                continue
            valid_crops.append(crop)
            valid_references.append(reference)

        predictions: list[tuple[str, float]] = []
        for start in range(0, len(valid_crops), 32):
            predictions.extend(predictor.predict(valid_crops[start : start + 32]))
        if len(predictions) != len(valid_references):
            raise ValueError("predictor returned a different number of predictions")
        for reference, (label, confidence) in zip(valid_references, predictions):
            _attach_raw_classification(reference, label, confidence)
            reference.object_data["classification"] = resolved_classification(
                predictor.project_id, label, confidence
            )
        (temporary_detections / f"{frame_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    frame_count = len(frame_paths)
    _validate_output_counts(
        temporary_detections, frame_count, object_count, unavailable_count
    )
    result = ClassificationRunResult(
        run_id=run_id,
        detection_dir=detection_dir,
        result_path=result_path,
        frame_count=frame_count,
        object_count=object_count,
        unavailable_count=unavailable_count,
    )
    (temporary_run / "result.json").write_text(
        json.dumps(result.to_cli_payload(), ensure_ascii=False), encoding="utf-8"
    )
    temporary_run.rename(run_dir)
    _replace_current(publication_root / "CURRENT", run_id)
    return result


def _matching_frame_paths(dataset: Path) -> list[tuple[int, Path, Path]]:
    image_paths = _numeric_files(dataset / "images", image=True)
    detection_paths = _numeric_files(dataset / "detections_results", image=False)
    if image_paths.keys() != detection_paths.keys():
        raise ValueError("image/detection frame IDs differ")
    return [
        (frame_id, image_paths[frame_id], detection_paths[frame_id])
        for frame_id in sorted(image_paths)
    ]


def _numeric_files(directory: Path, image: bool) -> dict[int, Path]:
    if not directory.is_dir():
        raise ValueError(f"missing directory: {directory}")
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp"} if image else {".json"}
    paths: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in allowed_extensions:
            continue
        if not path.stem.isdigit():
            continue
        frame_id = int(path.stem)
        if frame_id in paths:
            raise ValueError(f"duplicate frame ID: {frame_id}")
        paths[frame_id] = path
    return paths


def _object_references(payload: dict[str, object]) -> list[_ObjectReference]:
    skus = payload.get("skus")
    if not isinstance(skus, list):
        raise ValueError("detection payload skus must be a list")
    references: list[_ObjectReference] = []
    for sku in skus:
        if not isinstance(sku, dict):
            raise ValueError("detection sku must be an object")
        objects = sku.get("objects")
        if not isinstance(objects, list):
            raise ValueError("detection sku objects must be a list")
        for object_data in objects:
            if not isinstance(object_data, dict):
                raise ValueError("detection object must be an object")
            references.append(_ObjectReference(sku=sku, object_data=object_data))
    return references


def _valid_crop(image: np.ndarray, position: object) -> np.ndarray | None:
    if not isinstance(position, list) or len(position) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in position):
        return None
    if not all(math.isfinite(float(value)) for value in position):
        return None
    x1, y1, x2, y2 = (int(value) for value in position)
    height, width = image.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return None
    if x1 >= x2 or y1 >= y2:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _attach_raw_classification(
    reference: _ObjectReference, label: str, confidence: float
) -> None:
    classes = reference.sku.setdefault("classes", {})
    if not isinstance(classes, dict):
        raise ValueError("detection sku classes must be an object")
    class_names = classes.setdefault("cls", [])
    if not isinstance(class_names, list):
        raise ValueError("detection sku classes.cls must be a list")
    try:
        class_index = class_names.index(label)
    except ValueError:
        class_names.append(label)
        class_index = len(class_names) - 1
    object_classes = reference.object_data.setdefault("classes", {})
    confidences = reference.object_data.setdefault("confidences", {})
    if not isinstance(object_classes, dict) or not isinstance(confidences, dict):
        raise ValueError("detection object classes and confidences must be objects")
    object_classes["cls"] = class_index
    confidences["cls"] = float(confidence)


def _unavailable_classification(project_id: int) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "source": "personalcare",
        "project_id": project_id,
        "status": "unavailable",
        "reason": "invalid_bbox",
    }


def _validate_output_counts(
    detection_dir: Path,
    frame_count: int,
    object_count: int,
    unavailable_count: int,
) -> None:
    output_files = sorted(detection_dir.glob("*.json"))
    if len(output_files) != frame_count:
        raise ValueError("published frame count does not match input")
    if unavailable_count > object_count:
        raise ValueError("unavailable object count exceeds object count")


def _replace_current(current_path: Path, run_id: str) -> None:
    temporary_current = current_path.with_name(f".{current_path.name}.{run_id}.tmp")
    temporary_current.write_text(
        json.dumps({"run_id": run_id, "complete": True}), encoding="utf-8"
    )
    temporary_current.replace(current_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify personalcare detections locally")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = classify_dataset(
            args.dataset,
            args.output_root,
            args.device,
            PersonalcarePredictor(args.device),
        )
    except Exception as error:
        print(f"classification failed: {error}", file=sys.stderr)
        print(json.dumps({"success": False, "error": str(error)}))
        return 1
    print(json.dumps(result.to_cli_payload(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
