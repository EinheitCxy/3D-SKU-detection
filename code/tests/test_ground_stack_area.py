import pytest

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
