from __future__ import annotations

import pytest

from src.deduplicate_detections import (
    add_global_id_to_jsons,
    build_global_mapping,
    resolve_dataset_paths,
)
from utils.classification_aggregation import aggregate_classifications
from utils.global_id_mapper import GlobalIDMapper, InstanceInfo
from utils.global_object_index import build_global_object_index


def resolved(sku_id: str, sku_name: str, confidence: float) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "source": "personalcare",
        "project_id": 51,
        "status": "resolved",
        "sku_id": sku_id,
        "sku_name": sku_name,
        "confidence": confidence,
        "metadata": {
            "status": "master_data_pending",
            "manufacturer": None,
            "brand": None,
            "category": None,
            "object_kind": None,
        },
    }


def test_conflicts_keep_all_candidates_but_primary_is_highest_sum() -> None:
    result = aggregate_classifications(
        [
            resolved("A", "产品A", 0.60),
            resolved("B", "产品B", 0.95),
            resolved("A", "产品A", 0.50),
        ]
    )

    assert result["status"] == "conflict"
    assert result["primary_sku_id"] == "A"
    assert [item["sku_id"] for item in result["candidates"]] == ["A", "B"]
    assert result["candidates"][0]["confidence_sum"] == pytest.approx(1.10)


def test_aggregation_is_permutation_stable() -> None:
    inputs = [resolved("2", "乙", 0.8), resolved("1", "甲", 0.8)]

    assert aggregate_classifications(inputs) == aggregate_classifications(
        list(reversed(inputs))
    )
    assert aggregate_classifications(inputs)["primary_sku_id"] == "1"


def test_unavailable_observations_produce_no_primary() -> None:
    unavailable = {
        "schema_version": "1.0.0",
        "source": "personalcare",
        "project_id": 51,
        "status": "unavailable",
        "reason": "invalid_bbox",
    }

    assert aggregate_classifications([unavailable]) == {
        "status": "unavailable",
        "primary_sku_id": None,
        "candidates": [],
        "metadata": {
            "status": "master_data_pending",
            "manufacturer": None,
            "brand": None,
            "category": None,
            "object_kind": None,
        },
    }


def test_invalid_confidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="classification confidence"):
        aggregate_classifications([resolved("A", "产品A", float("nan"))])


def test_non_integer_project_id_is_rejected() -> None:
    classification = resolved("A", "产品A", 0.8)
    classification["project_id"] = 51.0

    with pytest.raises(ValueError, match="project_id"):
        aggregate_classifications([classification])


def test_same_id_with_different_names_remains_distinct_and_deterministic() -> None:
    result = aggregate_classifications(
        [resolved("A", "产品乙", 0.8), resolved("A", "产品甲", 0.8)]
    )

    assert [(item["sku_id"], item["sku_name"]) for item in result["candidates"]] == [
        ("A", "产品乙"),
        ("A", "产品甲"),
    ]


def test_global_object_index_aggregates_removed_observation_and_keeps_provenance() -> None:
    mapper = GlobalIDMapper()
    mapper.data = {
        "1": [
            InstanceInfo(1, 0, [0.0, 0.0, 1.0, 1.0], False, resolved("A", "产品A", 0.6)),
            InstanceInfo(2, 0, [0.0, 0.0, 1.0, 1.0], True, resolved("B", "产品B", 0.9)),
        ]
    }

    index = build_global_object_index(mapper)

    assert [inst["classification"]["sku_id"] for inst in index["1"]["instances"]] == [
        "A",
        "B",
    ]
    assert [item["sku_id"] for item in index["1"]["classification"]["candidates"]] == [
        "B",
        "A",
    ]


def test_global_mapping_copies_classification_for_removed_observation() -> None:
    first = {"position": [0.0, 0.0, 1.0, 1.0], "classification": resolved("A", "产品A", 0.6)}
    removed = {"position": [0.0, 0.0, 1.0, 1.0], "classification": resolved("B", "产品B", 0.9)}
    mapping = build_global_mapping(
        [{"ref_idx": 1, "ref_id": 0, "target_idx": 2, "target_id": 0}],
        {1: {0}, 2: set()},
        {1: [first], 2: [removed]},
        [1, 2],
    )

    assert mapping["1"][1]["removed"] is True
    assert mapping["1"][1]["classification"] == removed["classification"]
    assert mapping["1"][1]["classification"] is not removed["classification"]


def test_dataset_paths_can_explicitly_use_classified_detection_directory(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    classified = tmp_path / "classified"

    paths = resolve_dataset_paths(dataset, detections_dir=classified)

    assert paths.dataset_dir == dataset
    assert paths.detections_dir == classified


def test_global_id_publication_requires_an_explicit_detection_directory() -> None:
    with pytest.raises(ValueError, match="detections_dir"):
        add_global_id_to_jsons(global_mapping={}, indices=[])
