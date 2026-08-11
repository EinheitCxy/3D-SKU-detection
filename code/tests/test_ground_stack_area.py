import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.ground_stack_area_stage import run_ground_stack_area

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
