import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.personalcare_classifier.source.classify_dataset import classify_dataset
from modules.personalcare_classifier.source.contracts import (
    lookup_sku_metadata,
    resolved_classification,
    split_sku_label,
)
from modules.personalcare_classifier.source.processor import PersonalcarePredictor


def test_split_sku_label_preserves_name_suffix() -> None:
    assert split_sku_label("430085^产品^限定版") == ("430085", "产品^限定版")


def test_mapping_placeholder_is_explicit_and_empty() -> None:
    assert lookup_sku_metadata("430085", "产品A") == {
        "status": "master_data_pending",
        "manufacturer": None,
        "brand": None,
        "category": None,
        "object_kind": None,
    }


def test_resolved_classification_has_stable_schema() -> None:
    assert resolved_classification(51, "430085^产品A", 0.75) == {
        "schema_version": "1.0.0",
        "source": "personalcare",
        "project_id": 51,
        "status": "resolved",
        "sku_id": "430085",
        "sku_name": "产品A",
        "confidence": 0.75,
        "metadata": lookup_sku_metadata("430085", "产品A"),
    }


@pytest.mark.parametrize("label", ["^产品A", "430085", "430085^"])
def test_split_sku_label_rejects_malformed_values(label: str) -> None:
    with pytest.raises(ValueError, match="sku_id\\^sku_name"):
        split_sku_label(label)


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.01, 1.01])
def test_resolved_classification_rejects_invalid_confidence(
    confidence: float,
) -> None:
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        resolved_classification(51, "430085^产品A", confidence)


def make_dataset(root: Path, positions: list[list[int]]) -> Path:
    dataset = root / "dataset"
    images = dataset / "images"
    detections = dataset / "detections_results"
    images.mkdir(parents=True)
    detections.mkdir()
    image = np.full((24, 24, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(images / "0.jpg"), image)
    objects = [
        {"position": position, "classes": {"det": 0}, "confidences": {"det": 0.9}}
        for position in positions
    ]
    (detections / "0.json").write_text(
        json.dumps(
            {
                "skus": [
                    {"classes": {"det": ["sku"]}, "objects": objects},
                ],
            }
        ),
        encoding="utf-8",
    )
    return dataset


def test_classify_dataset_preserves_order_and_publishes_enriched_json(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path, positions=[[0, 0, 10, 10], [10, 0, 20, 10]])

    class FakePredictor:
        project_id = 51

        def predict(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
            assert len(crops) == 2
            return [("430085^产品A", 0.9), ("428987^产品B", 0.8)]

    result = classify_dataset(dataset, tmp_path / "Output", "cuda:0", FakePredictor())
    enriched = json.loads((result.detection_dir / "0.json").read_text())
    objects = enriched["skus"][0]["objects"]
    assert [item["position"] for item in objects] == [[0, 0, 10, 10], [10, 0, 20, 10]]
    assert [item["classification"]["sku_id"] for item in objects] == [
        "430085",
        "428987",
    ]
    assert result.run_id.split("-")[0].isdigit()


def test_invalid_bbox_is_unavailable_and_features_are_not_published(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path, positions=[[5, 5, 5, 12]])

    class PredictorThatMustNotRun:
        project_id = 51

        def predict(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
            raise AssertionError("invalid bbox must not reach predictor")

    result = classify_dataset(
        dataset,
        tmp_path / "Output",
        "cuda:0",
        PredictorThatMustNotRun(),
    )
    enriched = json.loads((result.detection_dir / "0.json").read_text())
    classification = enriched["skus"][0]["objects"][0]["classification"]
    assert classification == {
        "schema_version": "1.0.0",
        "source": "personalcare",
        "project_id": 51,
        "status": "unavailable",
        "reason": "invalid_bbox",
    }
    assert "features" not in json.dumps(enriched)
    assert result.unavailable_count == 1


def test_missing_frame_does_not_replace_current(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, positions=[[0, 0, 10, 10]])
    (dataset / "images" / "0.jpg").rename(dataset / "images" / "1.jpg")
    current = tmp_path / "Output" / dataset.name / "personalcare_classification" / "CURRENT"
    current.parent.mkdir(parents=True)
    current.write_text('{"run_id":"old","complete":true}', encoding="utf-8")

    class PredictorThatMustNotRun:
        project_id = 51

        def predict(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
            raise AssertionError("mismatched frames must not reach predictor")

    with pytest.raises(ValueError, match="image/detection frame IDs differ"):
        classify_dataset(dataset, tmp_path / "Output", "cuda:0", PredictorThatMustNotRun())
    assert json.loads(current.read_text(encoding="utf-8"))["run_id"] == "old"


def test_canonical_checkpoint_decodes_to_a_state_dict() -> None:
    model_path = (
        Path(__file__).parents[1]
        / "modules"
        / "personalcare_classifier"
        / "source"
        / "model"
        / "model.bin"
    )
    state_dict = PersonalcarePredictor._load_state_dict(model_path)
    assert "features.0.0.weight" in state_dict
