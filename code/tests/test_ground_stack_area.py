import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.deduplicate_detections import build_global_mapping, load_detection_objects
from modules.ground_stack_area_stage import run_ground_stack_area
from utils.data_utils import load_detections

from utils.ground_stack_area import (
    BBoxAreaError,
    calibrated_bbox_area_cm2,
    calibrate_from_anchor,
    select_best_instances,
)


def test_anchor_bbox_maps_to_known_physical_area():
    calibration = calibrate_from_anchor([10, 20, 110, 70], 20.0, 10.0)

    assert calibrated_bbox_area_cm2([10, 20, 110, 70], calibration) == pytest.approx(
        200.0
    )


def test_select_best_instances_counts_each_global_id_once():
    selected, rejected = select_best_instances(
        {
            "1": [
                {"image_id": 0, "object_id": 0, "bbox": [0, 0, 10, 10]},
                {"image_id": 1, "object_id": 2, "bbox": [0, 0, 20, 20]},
            ],
            "2": [{"image_id": 1, "object_id": 3, "bbox": [0, 0, 5, 10]}],
        }
    )

    assert [(item.global_id, item.bbox) for item in selected] == [
        ("1", (0.0, 0.0, 20.0, 20.0)),
        ("2", (0.0, 0.0, 5.0, 10.0)),
    ]
    assert rejected == []


@pytest.mark.parametrize("bbox", ([0, 0, 0, 1], [0, 0, float("nan"), 1]))
def test_invalid_bbox_is_rejected(bbox):
    with pytest.raises(BBoxAreaError, match="bbox"):
        calibrate_from_anchor(bbox, 20.0, 10.0)


def make_measurement_fixture(tmp_path):
    dataset = tmp_path / "stack"
    images = dataset / "images"
    detections = dataset / "detections_results"
    images.mkdir(parents=True)
    detections.mkdir()
    assert cv2.imwrite(
        str(images / "0.jpg"), np.full((100, 100, 3), 255, dtype=np.uint8)
    )

    detection_path = detections / "0.json"
    detection_path.write_text(
        json.dumps(
            {
                "skus": [
                    {
                        "objects": [
                            {"position": [10, 10, 30, 20]},
                            {"position": [40, 10, 60, 20]},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    save_root = tmp_path / "Output"
    mapping_path = save_root / "stack" / "dedup_detections" / "global_mapping.json"
    mapping_path.parent.mkdir(parents=True)
    mapping_path.write_text(
        json.dumps(
            {
                "1": [
                    {
                        "image_id": 0,
                        "object_id": 0,
                        "bbox": [10, 10, 30, 20],
                        "removed": False,
                    }
                ],
                "2": [
                    {
                        "image_id": 0,
                        "object_id": 1,
                        "bbox": [40, 10, 60, 20],
                        "removed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return dataset, save_root, detection_path, mapping_path


def test_stage_writes_report_without_mutating_inputs(tmp_path):
    dataset, save_root, detection_path, mapping_path = make_measurement_fixture(tmp_path)
    mapping_before = mapping_path.read_bytes()
    detection_before = detection_path.read_bytes()

    result = run_ground_stack_area(
        str(dataset), save_root, 0, 0, 20.0, 10.0
    )

    report_path = Path(result["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["success"] is True
    assert report["status"] == "accepted"
    assert report["value_cm2"] == pytest.approx(400.0)
    assert report["value_m2"] == pytest.approx(0.04)
    assert report["accepted_global_ids"] == 2
    assert report["rejected_global_ids"] == 0
    assert mapping_path.read_bytes() == mapping_before
    assert detection_path.read_bytes() == detection_before
    assert (report_path.parent / "selected_instances.json").is_file()
    assert (report_path.parent / "annotated_frames" / "0.jpg").is_file()


def test_stage_uses_only_anchor_frame_observations_for_calibration(tmp_path):
    dataset, save_root, _, mapping_path = make_measurement_fixture(tmp_path)
    assert cv2.imwrite(
        str(dataset / "images" / "1.jpg"),
        np.full((100, 100, 3), 255, dtype=np.uint8),
    )
    mapping_path.write_text(
        json.dumps(
            {
                "1": [
                    {"image_id": 0, "object_id": 0, "bbox": [10, 10, 30, 20]},
                ],
                "2": [
                    {"image_id": 0, "object_id": 1, "bbox": [40, 10, 50, 15]},
                    {"image_id": 1, "object_id": 0, "bbox": [0, 0, 20, 20]},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_ground_stack_area(str(dataset), save_root, 0, 0, 20.0, 10.0)

    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    instances = json.loads(Path(result["instances_path"]).read_text(encoding="utf-8"))
    assert result["success"] is True
    assert report["value_cm2"] == pytest.approx(250.0)
    assert instances["instances"][1]["image_id"] == 0
    assert instances["instances"][1]["area_cm2"] == pytest.approx(50.0)


def test_stage_rejects_bbox_outside_source_image_bounds(tmp_path):
    dataset, save_root, _, mapping_path = make_measurement_fixture(tmp_path)
    mapping_path.write_text(
        json.dumps(
            {
                "1": [{"image_id": 0, "object_id": 0, "bbox": [10, 10, 30, 20]}],
                "2": [{"image_id": 0, "object_id": 1, "bbox": [-1, 10, 10, 20]}],
            }
        ),
        encoding="utf-8",
    )

    result = run_ground_stack_area(str(dataset), save_root, 0, 0, 20.0, 10.0)

    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    instances = json.loads(Path(result["instances_path"]).read_text(encoding="utf-8"))
    assert result["success"] is True
    assert report["status"] == "accepted_with_warnings"
    assert report["value_cm2"] == pytest.approx(200.0)
    assert report["rejected_global_ids"] == 1
    assert "outside source image bounds" in instances["rejected"][0]["reason"]


def test_stage_writes_rejected_report_for_nonfinite_anchor_dimension(tmp_path):
    dataset, save_root, _, _ = make_measurement_fixture(tmp_path)

    result = run_ground_stack_area(str(dataset), save_root, 0, 0, float("nan"), 10.0)

    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert result["success"] is False
    assert report["status"] == "rejected"
    assert report["calibration"]["anchor_width_cm"] is None
    assert "width_cm must be a positive finite value" in report["warnings"]


def test_stage_rejects_anchor_when_mapping_bbox_does_not_match_detection(tmp_path):
    dataset, save_root, _, mapping_path = make_measurement_fixture(tmp_path)
    mapping_path.write_text(
        json.dumps(
            {
                "1": [{"image_id": 0, "object_id": 0, "bbox": [10, 10, 50, 20]}],
                "2": [{"image_id": 0, "object_id": 1, "bbox": [40, 10, 60, 20]}],
            }
        ),
        encoding="utf-8",
    )

    result = run_ground_stack_area(str(dataset), save_root, 0, 0, 20.0, 10.0)

    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert result["success"] is False
    assert report["status"] == "rejected"
    assert "anchor bbox does not match global mapping" in report["warnings"]


def test_select_best_instances_rejects_nonintegral_observation_indexes():
    selected, rejected = select_best_instances(
        {"1": [{"image_id": 0.5, "object_id": 0, "bbox": [0, 0, 10, 10]}]}
    )

    assert selected == []
    assert rejected[0].reason == "observation must contain integer image_id and object_id"


def test_second_sku_group_can_be_anchor_and_global_mapping_member(tmp_path):
    dataset = tmp_path / "multi_sku_stack"
    images = dataset / "images"
    detections = dataset / "detections_results"
    images.mkdir(parents=True)
    detections.mkdir()
    assert cv2.imwrite(
        str(images / "0.jpg"), np.full((100, 100, 3), 255, dtype=np.uint8)
    )
    detection_path = detections / "0.json"
    detection_path.write_text(
        json.dumps(
            {
                "skus": [
                    {"objects": [{"position": [10, 10, 30, 20]}]},
                    {"objects": [{"position": [40, 10, 60, 20]}]},
                ]
            }
        ),
        encoding="utf-8",
    )
    matching_detections = load_detections(str(detections), return_index_map=True)
    assert [
        object_["position"] for object_ in matching_detections[0][1]["objects"]
    ] == [[10, 10, 30, 20], [40, 10, 60, 20]]
    _, objects = load_detection_objects(detection_path)
    mapping = build_global_mapping(
        matches=[],
        survivors_by_image={0: set(range(len(objects)))},
        objects_by_image={0: objects},
        image_indices=[0],
    )
    save_root = tmp_path / "Output"
    mapping_path = (
        save_root / "multi_sku_stack" / "dedup_detections" / "global_mapping.json"
    )
    mapping_path.parent.mkdir(parents=True)
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    result = run_ground_stack_area(str(dataset), save_root, 0, 1, 20.0, 10.0)

    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert result["success"] is True
    assert report["accepted_global_ids"] == 2
    assert report["value_cm2"] == pytest.approx(400.0)


def test_main_ground_stack_area_mode_runs_stage(tmp_path):
    dataset, save_root, _, _ = make_measurement_fixture(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--mode",
            "ground-stack-area",
            "--dataset",
            str(dataset),
            "--save_root",
            str(save_root),
            "--area-anchor-frame",
            "0",
            "--area-anchor-object",
            "0",
            "--area-anchor-width-cm",
            "20",
            "--area-anchor-height-cm",
            "10",
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        save_root / "stack" / "ground_stack_area" / "measurement_report.json"
    ).is_file()
