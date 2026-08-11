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
            "--area-mode",
            "calibrated_bbox",
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


def test_main_da3_metric_area_sums_each_global_id_once_without_anchor(tmp_path):
    dataset = tmp_path / "metric_stack"
    images = dataset / "images"
    images.mkdir(parents=True)
    for image_id in (0, 1):
        assert cv2.imwrite(
            str(images / f"{image_id}.jpg"),
            np.full((100, 100, 3), 255, dtype=np.uint8),
        )

    u, v = np.meshgrid(np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32))
    plane = np.stack((u * 0.1, v * 0.2, np.ones_like(u)), axis=-1)
    world_points = np.stack((plane, plane), axis=0)
    save_root = tmp_path / "Output"
    cache_dir = save_root / "metric_stack" / "da3_cache"
    cache_dir.mkdir(parents=True)
    np.savez_compressed(
        cache_dir / "predictions.npz",
        world_points=world_points,
        world_points_conf=np.ones((2, 10, 10), dtype=np.float32),
        image_ids=np.array([0, 1], dtype=np.int32),
        source_image_sizes=np.array([[100, 100], [100, 100]], dtype=np.int32),
        source_to_processed_affine=np.array(
            [[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]] * 2, dtype=np.float32
        ),
    )
    mapping_path = (
        save_root / "metric_stack" / "dedup_detections" / "global_mapping.json"
    )
    mapping_path.parent.mkdir(parents=True)
    mapping_path.write_text(
        json.dumps(
            {
                "1": [
                    {"image_id": 0, "object_id": 0, "bbox": [0, 0, 100, 100]},
                    {"image_id": 1, "object_id": 0, "bbox": [0, 0, 100, 110]},
                ],
                "2": [{"image_id": 0, "object_id": 1, "bbox": [0, 0, 100, 100]}],
                "3": [{"image_id": 1, "object_id": 1, "bbox": [0, 0, 100, 110]}],
            }
        ),
        encoding="utf-8",
    )
    mapping_before = mapping_path.read_bytes()

    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--mode",
            "ground-stack-area",
            "--area-mode",
            "da3_metric",
            "--dataset",
            str(dataset),
            "--save_root",
            str(save_root),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(
        (
            save_root / "metric_stack" / "ground_stack_area" / "measurement_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["metric"] == "da3_metric_bbox_area_sum"
    assert report["accepted_global_ids"] == 2
    assert report["rejected_global_ids"] == 1
    assert report["status"] == "accepted_with_warnings"
    assert report["warnings"][0].startswith("global_id 3: ")
    assert report["rejections"] == [
        {
            "global_id": "3",
            "reason": "bbox is outside source image bounds",
            "observation_diagnostics": [
                {
                    "observation_index": 0,
                    "status": "rejected",
                    "reason": "bbox is outside source image bounds",
                }
            ],
        }
    ]
    assert report["value_m2"] == pytest.approx(2 * 0.9 * 1.8)
    instances = json.loads(
        (
            save_root / "metric_stack" / "ground_stack_area" / "selected_instances.json"
        ).read_text(encoding="utf-8")
    )
    first = instances["instances"][0]
    assert first["selected_observation_index"] == 0
    assert [item["status"] for item in first["observation_diagnostics"]] == [
        "eligible",
        "rejected",
    ]
    assert mapping_path.read_bytes() == mapping_before


def test_main_da3_metric_area_uses_center_scale_when_bbox_edge_is_background(tmp_path):
    dataset = tmp_path / "edge_background_stack"
    images = dataset / "images"
    images.mkdir(parents=True)
    assert cv2.imwrite(
        str(images / "0.jpg"), np.full((100, 100, 3), 255, dtype=np.uint8)
    )

    u, v = np.meshgrid(np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32))
    plane = np.stack((u * 0.1, v * 0.2, np.ones_like(u)), axis=-1)
    plane[[0, -1], :, 0] += 5.0
    plane[:, [0, -1], 0] += 5.0
    save_root = tmp_path / "Output"
    cache_dir = save_root / "edge_background_stack" / "da3_cache"
    cache_dir.mkdir(parents=True)
    np.savez_compressed(
        cache_dir / "predictions.npz",
        world_points=plane[None, ...],
        world_points_conf=np.ones((1, 10, 10), dtype=np.float32),
        image_ids=np.array([0], dtype=np.int32),
        source_image_sizes=np.array([[100, 100]], dtype=np.int32),
        source_to_processed_affine=np.array(
            [[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]], dtype=np.float32
        ),
    )
    mapping_path = (
        save_root
        / "edge_background_stack"
        / "dedup_detections"
        / "global_mapping.json"
    )
    mapping_path.parent.mkdir(parents=True)
    mapping_path.write_text(
        json.dumps(
            {"1": [{"image_id": 0, "object_id": 0, "bbox": [0, 0, 100, 100]}]}
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--mode",
            "ground-stack-area",
            "--area-mode",
            "da3_metric",
            "--dataset",
            str(dataset),
            "--save_root",
            str(save_root),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(
        (
            save_root
            / "edge_background_stack"
            / "ground_stack_area"
            / "measurement_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["value_m2"] == pytest.approx(0.9 * 1.8)


def test_main_da3_metric_area_uses_cached_source_to_processed_affine(tmp_path):
    dataset = tmp_path / "mixed_size_stack"
    images = dataset / "images"
    images.mkdir(parents=True)
    assert cv2.imwrite(
        str(images / "0.jpg"), np.full((100, 200, 3), 255, dtype=np.uint8)
    )

    u, v = np.meshgrid(np.arange(20, dtype=np.float32), np.arange(10, dtype=np.float32))
    points = np.stack((u, v, np.ones_like(u)), axis=-1)
    points[:, 10:, 0] = (u[:, 10:] - 10) * 0.1
    points[:, 10:, 1] = v[:, 10:] * 0.2
    save_root = tmp_path / "Output"
    cache_dir = save_root / "mixed_size_stack" / "da3_cache"
    cache_dir.mkdir(parents=True)
    np.savez_compressed(
        cache_dir / "predictions.npz",
        world_points=points[None, ...],
        world_points_conf=np.ones((1, 10, 20), dtype=np.float32),
        image_ids=np.array([0], dtype=np.int32),
        source_image_sizes=np.array([[200, 100]], dtype=np.int32),
        source_to_processed_affine=np.array(
            [[[0.1, 0.0, 10.0], [0.0, 0.1, 0.0]]], dtype=np.float32
        ),
    )
    mapping_path = (
        save_root / "mixed_size_stack" / "dedup_detections" / "global_mapping.json"
    )
    mapping_path.parent.mkdir(parents=True)
    mapping_path.write_text(
        json.dumps(
            {"1": [{"image_id": 0, "object_id": 0, "bbox": [0, 0, 100, 100]}]}
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--mode",
            "ground-stack-area",
            "--area-mode",
            "da3_metric",
            "--dataset",
            str(dataset),
            "--save_root",
            str(save_root),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(
        (
            save_root / "mixed_size_stack" / "ground_stack_area" / "measurement_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["value_m2"] == pytest.approx(0.9 * 1.8)


def test_main_da3_metric_area_rejects_zero_confidence_padding(tmp_path):
    dataset = tmp_path / "invalid_depth_stack"
    images = dataset / "images"
    images.mkdir(parents=True)
    assert cv2.imwrite(
        str(images / "0.jpg"), np.full((200, 200, 3), 255, dtype=np.uint8)
    )

    u, v = np.meshgrid(np.arange(20, dtype=np.float32), np.arange(20, dtype=np.float32))
    points = np.zeros((20, 20, 3), dtype=np.float32)
    points[5:15, 5:15] = np.stack(
        (u[5:15, 5:15] * 0.1, v[5:15, 5:15] * 0.2, np.ones((10, 10))),
        axis=-1,
    )
    confidence = np.zeros((20, 20), dtype=np.float32)
    confidence[5:15, 5:15] = 1.0
    save_root = tmp_path / "Output"
    cache_dir = save_root / "invalid_depth_stack" / "da3_cache"
    cache_dir.mkdir(parents=True)
    np.savez_compressed(
        cache_dir / "predictions.npz",
        world_points=points[None, ...],
        world_points_conf=confidence[None, ...],
        image_ids=np.array([0], dtype=np.int32),
        source_image_sizes=np.array([[200, 200]], dtype=np.int32),
        source_to_processed_affine=np.array(
            [[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]], dtype=np.float32
        ),
    )
    mapping_path = (
        save_root / "invalid_depth_stack" / "dedup_detections" / "global_mapping.json"
    )
    mapping_path.parent.mkdir(parents=True)
    mapping_path.write_text(
        json.dumps(
            {"1": [{"image_id": 0, "object_id": 0, "bbox": [0, 0, 200, 200]}]}
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--mode",
            "ground-stack-area",
            "--area-mode",
            "da3_metric",
            "--dataset",
            str(dataset),
            "--save_root",
            str(save_root),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    report = json.loads(
        (
            save_root / "invalid_depth_stack" / "ground_stack_area" / "measurement_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "rejected"
    assert report["accepted_global_ids"] == 0


def test_main_da3_metric_area_rejects_stale_source_image_size(tmp_path):
    dataset = tmp_path / "stale_source_stack"
    images = dataset / "images"
    images.mkdir(parents=True)
    assert cv2.imwrite(
        str(images / "0.jpg"), np.full((100, 100, 3), 255, dtype=np.uint8)
    )

    u, v = np.meshgrid(np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32))
    plane = np.stack((u * 0.1, v * 0.2, np.ones_like(u)), axis=-1)
    save_root = tmp_path / "Output"
    cache_dir = save_root / "stale_source_stack" / "da3_cache"
    cache_dir.mkdir(parents=True)
    np.savez_compressed(
        cache_dir / "predictions.npz",
        world_points=plane[None, ...],
        world_points_conf=np.ones((1, 10, 10), dtype=np.float32),
        image_ids=np.array([0], dtype=np.int32),
        source_image_sizes=np.array([[200, 100]], dtype=np.int32),
        source_to_processed_affine=np.array(
            [[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]], dtype=np.float32
        ),
    )
    mapping_path = (
        save_root / "stale_source_stack" / "dedup_detections" / "global_mapping.json"
    )
    mapping_path.parent.mkdir(parents=True)
    mapping_path.write_text(
        json.dumps(
            {"1": [{"image_id": 0, "object_id": 0, "bbox": [0, 0, 100, 100]}]}
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--mode",
            "ground-stack-area",
            "--area-mode",
            "da3_metric",
            "--dataset",
            str(dataset),
            "--save_root",
            str(save_root),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    report = json.loads(
        (
            save_root / "stale_source_stack" / "ground_stack_area" / "measurement_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "rejected"
    assert report["warnings"] == [
        "global_id 1: source image size changed for frame 0; rebuild DA3 cache"
    ]
    assert report["rejections"][0]["observation_diagnostics"][0]["status"] == "rejected"
