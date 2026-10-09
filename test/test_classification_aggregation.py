from __future__ import annotations

import json

import pytest

import src.deduplicate_detections as dedup_module
from src.deduplicate_detections import (
    add_global_id_to_jsons,
    build_global_mapping,
    deduplicate_sequence,
    resolve_dataset_paths,
)
from utils.classification_aggregation import (
    aggregate_classifications,
    build_resolved_classification,
)
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


def test_build_resolved_classification_uses_canonical_schema() -> None:
    assert build_resolved_classification(51, "430085^产品A", 0.75) == resolved(
        "430085", "产品A", 0.75
    )


@pytest.mark.parametrize(
    "project_id,label,confidence",
    [
        (50, "430085^产品A", 0.75),
        (51, "430085", 0.75),
        (51, "^产品A", 0.75),
        (51, "430085^", 0.75),
        (51, "430085^产品A", float("nan")),
        (51, "430085^产品A", float("inf")),
    ],
)
def test_build_resolved_classification_rejects_invalid_input(
    project_id: int, label: str, confidence: float
) -> None:
    with pytest.raises(ValueError):
        build_resolved_classification(project_id, label, confidence)


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


def test_non_other_candidate_wins_over_higher_confidence_other_observation() -> None:
    other = resolved("56642", "其他品类", 0.99)
    product = resolved("A", "产品A", 0.20)
    original = [other.copy(), product.copy()]

    result = aggregate_classifications([other, product])

    assert result["status"] == "conflict"
    assert result["primary_sku_id"] == "A"
    assert [item["sku_id"] for item in result["candidates"]] == ["A", "56642"]
    assert [other, product] == original


def test_all_other_observations_allow_other_as_primary() -> None:
    result = aggregate_classifications(
        [resolved("56642", "其他品类", 0.40), resolved("56642", "其他品类", 0.90)]
    )

    assert result["status"] == "resolved"
    assert result["primary_sku_id"] == "56642"
    assert result["candidates"] == [
        {
            "sku_id": "56642",
            "sku_name": "其他品类",
            "confidence_sum": pytest.approx(1.30),
            "support_count": 2,
            "max_confidence": 0.90,
        }
    ]


def test_multiple_non_other_candidates_keep_confidence_order_before_other() -> None:
    result = aggregate_classifications(
        [
            resolved("56642", "其他品类", 0.99),
            resolved("B", "产品B", 0.75),
            resolved("A", "产品A", 0.40),
            resolved("A", "产品A", 0.40),
        ]
    )

    assert [item["sku_id"] for item in result["candidates"]] == ["A", "B", "56642"]
    assert result["primary_sku_id"] == "A"


def test_unavailable_observations_are_ignored_when_other_and_non_other_exist() -> None:
    unavailable = {
        "schema_version": "1.0.0",
        "source": "personalcare",
        "project_id": 51,
        "status": "unavailable",
        "reason": "invalid_bbox",
    }

    result = aggregate_classifications(
        [unavailable, resolved("56642", "其他品类", 0.99), resolved("A", "产品A", 0.20)]
    )

    assert result["primary_sku_id"] == "A"
    assert [item["sku_id"] for item in result["candidates"]] == ["A", "56642"]


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


def test_global_object_index_aggregates_removed_observation_and_keeps_provenance() -> (
    None
):
    mapper = GlobalIDMapper()
    mapper.data = {
        "1": [
            InstanceInfo(
                1, 0, [0.0, 0.0, 1.0, 1.0], False, resolved("A", "产品A", 0.6)
            ),
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
    first = {
        "position": [0.0, 0.0, 1.0, 1.0],
        "classification": resolved("A", "产品A", 0.6),
    }
    removed = {
        "position": [0.0, 0.0, 1.0, 1.0],
        "classification": resolved("B", "产品B", 0.9),
    }
    mapping = build_global_mapping(
        [{"ref_idx": 1, "ref_id": 0, "target_idx": 2, "target_id": 0}],
        {1: {0}, 2: set()},
        {1: [first], 2: [removed]},
        [1, 2],
    )

    assert mapping["1"][1]["removed"] is True
    assert mapping["1"][1]["classification"] == removed["classification"]
    assert mapping["1"][1]["classification"] is not removed["classification"]


def test_dataset_paths_can_explicitly_use_classified_detection_directory(
    tmp_path,
) -> None:
    dataset = tmp_path / "dataset"
    classified = tmp_path / "classified"

    paths = resolve_dataset_paths(dataset, detections_dir=classified)

    assert paths.dataset_dir == dataset
    assert paths.detections_dir == classified


def test_global_id_publication_requires_an_explicit_detection_directory() -> None:
    with pytest.raises(ValueError, match="detections_dir"):
        add_global_id_to_jsons(global_mapping={}, indices=[])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda _: None,
        lambda value: {key: item for key, item in value.items() if key != "sku_name"},
        lambda value: {**value, "status": "illegal"},
        lambda value: {**value, "confidence": float("inf")},
        lambda value: {**value, "confidence": float("-inf")},
        lambda value: {**value, "confidence": -0.01},
        lambda value: {**value, "confidence": 1.01},
        lambda value: {**value, "confidence": True},
        lambda value: {**value, "project_id": True},
        lambda value: {
            "schema_version": "1.0.0",
            "source": "personalcare",
            "project_id": 51,
            "status": "unavailable",
        },
        lambda value: {
            "schema_version": "1.0.0",
            "source": "personalcare",
            "project_id": 51,
            "status": "unavailable",
            "reason": 1,
        },
        lambda value: {
            "schema_version": "1.0.0",
            "source": "personalcare",
            "project_id": 51,
            "status": "unavailable",
            "reason": "",
        },
    ],
)
def test_dedup_rejects_invalid_classification_without_publishing_global_artifacts(
    tmp_path, mutate
) -> None:
    dataset = tmp_path / "dataset"
    detections_dir = dataset / "detections_results"
    detections_dir.mkdir(parents=True)
    classification = mutate(resolved("A", "产品A", 0.8))
    (detections_dir / "1.json").write_text(
        json.dumps(
            {
                "skus": [
                    {
                        "classes": {},
                        "objects": [
                            {
                                "position": [0.0, 0.0, 1.0, 1.0],
                                "classification": classification,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "Output"
    publication_dir = output_root / dataset.name / "dedup_detections"
    publication_dir.mkdir(parents=True)
    (publication_dir / "global_mapping.json").write_text("stale", encoding="utf-8")
    (publication_dir / "global_skus.json").write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="classification"):
        deduplicate_sequence(
            resolve_dataset_paths(dataset),
            output_root=output_root,
            output_subdir="dedup_detections",
        )

    assert not (publication_dir / "global_mapping.json").exists()
    assert not (publication_dir / "global_skus.json").exists()
    assert not (publication_dir / "1_dedup.json").exists()


def _write_single_classified_detection(dataset, classification) -> None:
    detections_dir = dataset / "detections_results"
    detections_dir.mkdir(parents=True)
    (detections_dir / "1.json").write_text(
        json.dumps(
            {
                "skus": [
                    {
                        "classes": {},
                        "objects": [
                            {
                                "position": [0.0, 0.0, 1.0, 1.0],
                                "classification": classification,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_dedup_cleans_both_global_files_when_second_publish_replace_fails(
    tmp_path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset"
    _write_single_classified_detection(dataset, resolved("A", "产品A", 0.8))
    output_root = tmp_path / "Output"
    publication_dir = output_root / dataset.name / "dedup_detections"
    original_replace = type(publication_dir).replace
    replaces = 0

    def fail_second_replace(path, target):
        nonlocal replaces
        replaces += 1
        if replaces == 2:
            raise OSError("second publish replace failed")
        return original_replace(path, target)

    monkeypatch.setattr(type(publication_dir), "replace", fail_second_replace)

    with pytest.raises(OSError, match="second publish replace failed"):
        deduplicate_sequence(
            resolve_dataset_paths(dataset),
            output_root=output_root,
            output_subdir="dedup_detections",
        )

    assert not (publication_dir / "global_mapping.json").exists()
    assert not (publication_dir / "global_skus.json").exists()
    assert not list(publication_dir.glob(".*.tmp"))


def test_dedup_cleans_both_global_files_when_second_temp_write_fails(
    tmp_path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset"
    _write_single_classified_detection(dataset, resolved("A", "产品A", 0.8))
    output_root = tmp_path / "Output"
    publication_dir = output_root / dataset.name / "dedup_detections"
    write_temp = dedup_module._write_global_publication_temp
    writes = 0

    def fail_second_write(path, payload):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("second temporary write failed")
        return write_temp(path, payload)

    monkeypatch.setattr(
        dedup_module, "_write_global_publication_temp", fail_second_write
    )

    with pytest.raises(OSError, match="second temporary write failed"):
        deduplicate_sequence(
            resolve_dataset_paths(dataset),
            output_root=output_root,
            output_subdir="dedup_detections",
        )

    assert not (publication_dir / "global_mapping.json").exists()
    assert not (publication_dir / "global_skus.json").exists()
    assert not list(publication_dir.glob(".*.tmp"))


def test_dedup_publishes_a_complete_valid_global_pair(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    _write_single_classified_detection(dataset, resolved("A", "产品A", 0.8))
    output_root = tmp_path / "Output"

    deduplicate_sequence(
        resolve_dataset_paths(dataset),
        output_root=output_root,
        output_subdir="dedup_detections",
    )

    publication_dir = output_root / dataset.name / "dedup_detections"
    mapping = json.loads((publication_dir / "global_mapping.json").read_text())
    global_skus = json.loads((publication_dir / "global_skus.json").read_text())
    assert mapping["1"][0]["classification"]["sku_id"] == "A"
    assert json.loads(global_skus[0])["objects"][0]["classification"]["sku_id"] == "A"


def _write_dedup_dataset(tmp_path, frame_objects: dict[int, int], edges=()):
    dataset = tmp_path / "dataset"
    detections_dir = dataset / "detections_results"
    detections_dir.mkdir(parents=True)
    for image_id, object_count in frame_objects.items():
        objects = [
            {
                "position": [float(index), 0.0, float(index + 1), 1.0],
                "classification": resolved(f"sku-{image_id}-{index}", "产品", 0.8),
            }
            for index in range(object_count)
        ]
        (detections_dir / f"{image_id}.json").write_text(
            json.dumps({"skus": [{"classes": {"frame": image_id}, "objects": objects}]}),
            encoding="utf-8",
        )

    output_root = tmp_path / "Output"
    summary_root = output_root / dataset.name / "output_3dmapping_da3"
    for ref_image, ref_object, target_image, target_object, score in edges:
        summary_dir = summary_root / str(ref_image)
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary = summary_dir / "matching_summary.txt"
        with summary.open("a", encoding="utf-8") as handle:
            handle.write(
                f"Matching objects between reference image {ref_image} and target image {target_image}\n"
                f"Matched ref {ref_object} -> target {target_object} "
                f"(hit ratio: {score:.2f} 10/10)\n"
                f"Found 1 matches in image {target_image}\n"
            )
    return dataset, output_root


def _assert_frame_outputs_match_global_artifacts(
    publication_dir, frame_ids: list[int], same_names: bool = False
) -> None:
    mapping = json.loads((publication_dir / "global_mapping.json").read_text())
    global_skus = [
        json.loads(item)
        for item in json.loads((publication_dir / "global_skus.json").read_text())
    ]
    mapped_retained = {
        (entry["image_id"], entry["object_id"]): int(global_id)
        for global_id, entries in mapping.items()
        for entry in entries
        if not entry["removed"]
    }
    sku_retained = {
        (image_id, object_id): obj["global_id"]
        for image_id, image in zip(frame_ids, global_skus)
        for object_id, obj in enumerate(image["objects"])
        if not obj["is_deduplicated"]
    }
    assert mapped_retained == sku_retained

    for image_id, image in zip(frame_ids, global_skus):
        filename = f"{image_id}.json" if same_names else f"{image_id}_dedup.json"
        frame = json.loads((publication_dir / filename).read_text())
        frame_skus = [
            obj["classification"]["sku_id"] for obj in frame["objects"]
        ]
        expected_skus = [
            obj["classification"]["sku_id"]
            for object_id, obj in enumerate(image["objects"])
            if (image_id, object_id) in mapped_retained
        ]
        assert frame_skus == expected_skus
        assert len(frame["objects"]) == sum(
            mapped_image == image_id for mapped_image, _ in mapped_retained
        )
        assert frame["classes"] == {"frame": image_id}


@pytest.mark.parametrize(
    "frame_ids,same_names",
    [([0, 1, 2], True), ([1, 2, 3], False)],
)
@pytest.mark.parametrize("dedup_mode", ["any", "best"])
def test_conflicting_ring_keeps_frame_json_mapping_and_global_skus_consistent(
    tmp_path, frame_ids, same_names, dedup_mode
) -> None:
    first, middle, last = frame_ids
    dataset, output_root = _write_dedup_dataset(
        tmp_path,
        {first: 1, middle: 1, last: 2},
        [
            (first, 0, middle, 0, 0.95),
            (middle, 0, last, 0, 0.80),
            (last, 1, first, 0, 0.90),
        ],
    )

    deduplicate_sequence(
        resolve_dataset_paths(dataset),
        output_root=output_root,
        output_subdir="dedup_detections",
        same_names=same_names,
        dedup_mode=dedup_mode,
        algorithm="3d_mapping",
        backend="da3",
    )

    publication_dir = output_root / dataset.name / "dedup_detections"
    _assert_frame_outputs_match_global_artifacts(
        publication_dir, frame_ids, same_names=same_names
    )
    filename = f"{last}.json" if same_names else f"{last}_dedup.json"
    last_frame = json.loads((publication_dir / filename).read_text())
    assert len(last_frame["objects"]) == 1
    assert last_frame["objects"][0]["classification"]["sku_id"] == f"sku-{last}-0"


def test_successful_match_and_isolated_objects_keep_expected_dedup_results(
    tmp_path,
) -> None:
    dataset, output_root = _write_dedup_dataset(
        tmp_path,
        {1: 1, 2: 2, 3: 1},
        [(1, 0, 2, 0, 0.9)],
    )

    deduplicate_sequence(
        resolve_dataset_paths(dataset),
        output_root=output_root,
        output_subdir="dedup_detections",
        algorithm="3d_mapping",
        backend="da3",
    )

    publication_dir = output_root / dataset.name / "dedup_detections"
    _assert_frame_outputs_match_global_artifacts(publication_dir, [1, 2, 3])
    assert len(json.loads((publication_dir / "1_dedup.json").read_text())["objects"]) == 1
    frame2 = json.loads((publication_dir / "2_dedup.json").read_text())
    assert [obj["classification"]["sku_id"] for obj in frame2["objects"]] == [
        "sku-2-1"
    ]
    assert len(json.loads((publication_dir / "3_dedup.json").read_text())["objects"]) == 1


def test_zero_matches_keeps_every_object_as_an_isolated_global_id(tmp_path) -> None:
    dataset, output_root = _write_dedup_dataset(tmp_path, {1: 1, 2: 2})

    deduplicate_sequence(
        resolve_dataset_paths(dataset),
        output_root=output_root,
        output_subdir="dedup_detections",
        algorithm="3d_mapping",
        backend="da3",
    )

    publication_dir = output_root / dataset.name / "dedup_detections"
    _assert_frame_outputs_match_global_artifacts(publication_dir, [1, 2])
    assert len(json.loads((publication_dir / "1_dedup.json").read_text())["objects"]) == 1
    assert len(json.loads((publication_dir / "2_dedup.json").read_text())["objects"]) == 2
