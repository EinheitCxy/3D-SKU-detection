"""Regression cases for invalid inputs, original exceptions and summary grouping."""

import importlib.util
import json
import sys
import os
import subprocess
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np

import main
from src.improved_sku_analyzer import ImprovedSKUCountAnalyzer
from utils.config import SKUMatchingConfig
from utils.data_utils import load_detections, save_correspondences_json
from utils import matching_algorithms as matching
from utils.sku_matching_system import SKUMatchingSystem
from utils.frame_alignment import FrameAlignmentError


@pytest.mark.parametrize("contents", ["{", '{"objects": 42}'])
def test_bad_detection_file_is_not_skipped(tmp_path, contents):
    (tmp_path / "0.json").write_text('{"objects": []}')
    (tmp_path / "1.json").write_text(contents)
    with pytest.raises(ValueError):
        load_detections(str(tmp_path))


def test_missing_detection_frame_is_rejected(tmp_path):
    images = tmp_path / "images"
    detections = tmp_path / "detections_results"
    images.mkdir()
    detections.mkdir()
    for frame in (0, 1):
        (images / f"{frame}.jpg").touch()
    (detections / "0.json").write_text('{"objects": []}')
    system = SKUMatchingSystem(SKUMatchingConfig.for_3d_mapping(device="cpu"))
    with pytest.raises(FrameAlignmentError):
        system._load_data(str(images), str(detections), 50)


def test_json_serialization_preserves_original_exception(tmp_path):
    config = SimpleNamespace(output_dir=tmp_path, json_filename="matches.json")
    with pytest.raises(TypeError, match="not JSON serializable"):
        save_correspondences_json({}, {0: object()}, config)


def test_detector_failure_does_not_publish_empty_detections(tmp_path, monkeypatch):
    image = tmp_path / "0.jpg"
    image.touch()
    error = RuntimeError("detector inference failed")

    def predict(*args, **kwargs):
        raise error

    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=lambda _: SimpleNamespace(predict=predict)),
    )
    script = Path(__file__).parents[1] / "modules/sku_detector/bbox_gen.py"
    spec = importlib.util.spec_from_file_location("detector_fail_fast", script)
    detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector)
    monkeypatch.setattr(sys, "argv", [str(script), str(image), "-o", str(tmp_path)])
    with pytest.raises(RuntimeError) as caught:
        detector.main()
    assert caught.value is error
    assert not (tmp_path / "detections_results/0.json").exists()


def test_summary_keeps_distinct_target_images(tmp_path):
    reference = tmp_path / "0"
    reference.mkdir()
    (reference / "matching_summary.txt").write_text(
        "Matched ref 0 → target 0 (hit ratio: 1.00 5/5)\n"
        "Matching objects between reference image 0 and target image 1\n"
        "Found 1 matches in image 1\n\n"
        "Matched ref 0 → target 0 (hit ratio: 1.00 5/5)\n"
        "Matching objects between reference image 0 and target image 2\n"
        "Found 1 matches in image 2\n",
        encoding="utf-8",
    )
    result = ImprovedSKUCountAnalyzer(
        str(tmp_path), str(tmp_path)
    ).analyze_with_filtering()
    assert result["original_matches"] == result["filtered_matches"] == 2
    assert {pair["target_idx"] for pair in result["pairs"]} == {1, 2}


def test_zero_match_report_is_complete(tmp_path):
    app = main.SKUDetectionMain()
    app.save_root = tmp_path / "output"
    (app.save_root / "dataset/output_pt").mkdir(parents=True)
    result = app.run_improved_sku_analysis(str(tmp_path / "dataset"))
    report = Path(result["details"]["report_file"]).read_text()
    assert "0 个冗余匹配" in report
    assert "【详细匹配结果】" in report


def test_single_matching_propagates_inference_exception(tmp_path, monkeypatch):
    from src import inference

    (tmp_path / "images").mkdir()
    (tmp_path / "detections_results").mkdir()
    error = RuntimeError("matching failed")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(inference, "run_3d_mapping", fail)
    with pytest.raises(RuntimeError) as caught:
        main.SKUDetectionMain()._run_single_matching(
            str(tmp_path), "3d", 0, 5, "cpu", False, "da3"
        )
    assert caught.value is error


def test_visualization_rejects_bad_detection_and_restores_argv(tmp_path):
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "detections_results").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "images/0.jpg")
    (tmp_path / "detections_results/0.json").write_text("{")
    original = list(sys.argv)
    app = main.SKUDetectionMain()
    app.save_root = tmp_path / "output"
    with pytest.raises(json.JSONDecodeError):
        app.run_detection_visualization(str(tmp_path))
    assert sys.argv == original


def test_accuracy_cli_preserves_missing_input_error(tmp_path, monkeypatch):
    import accuracy_annotation

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accuracy_annotation.py",
            "--benchmark-csv",
            str(tmp_path / "missing.csv"),
            "--vggt-result",
            str(tmp_path / "missing.txt"),
        ],
    )
    with pytest.raises(FileNotFoundError):
        accuracy_annotation.main()


def test_accuracy_report_write_failure_propagates(tmp_path):
    from accuracy_annotation import AccuracyAnnotator

    annotator = AccuracyAnnotator()
    annotator.ground_truth = {"1_to_2": [{"reference_id": 0, "target_id": 0}]}
    annotator.vggt_results = {"1_to_2": []}
    with pytest.raises(IsADirectoryError):
        annotator.generate_report(str(tmp_path))


def test_accuracy_rejects_missing_comparable_data(tmp_path):
    from accuracy_annotation import AccuracyAnnotator

    with pytest.raises(ValueError, match="可比较"):
        AccuracyAnnotator().generate_report(str(tmp_path / "report.txt"))
    assert not (tmp_path / "report.txt").exists()


@pytest.mark.parametrize("summary_text", [
    "Found 0 matches in image 1\n",
    "3D Mapping complete. Found correspondences in 0 images.\nFound 0 matches across 0 images\n",
])
def test_accuracy_zero_predictions_are_valid(tmp_path, summary_text):
    from accuracy_annotation import AccuracyAnnotator

    reference = tmp_path / "0"
    reference.mkdir()
    summary = reference / "matching_summary.txt"
    summary.write_text("Reference image file ID: 0\n" + summary_text)
    annotator = AccuracyAnnotator()
    annotator.ground_truth = {"1_to_2": [{"reference_id": 0, "target_id": 0}]}
    annotator.load_vggt_results(str(summary))
    report = annotator.generate_report(str(tmp_path / "report.txt"))
    assert "总体召回率 (Recall): 0.00% (0/1)" in report


def test_accuracy_shell_stops_on_failed_evaluator(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    executable = binaries / "uv"
    executable.write_text("#!/bin/sh\nexit 23\n")
    executable.chmod(0o755)
    match_dir = tmp_path / "sample/output_pt/0"
    match_dir.mkdir(parents=True)
    (match_dir / "matching_summary.txt").touch()
    script = Path(__file__).parents[1] / "scripts/3d/evaluation/accuracy_evaluation.sh"
    result = subprocess.run(
        ["bash", str(script), "sample", "--save-root", str(tmp_path)],
        env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 23, result.stdout + result.stderr


def test_accuracy_shell_preserves_successful_reports(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    executable = binaries / "uv"
    executable.write_text(
        """#!/bin/sh
while [ "$1" != "--output" ]; do shift; done
printf '图片对 1_to_2 详细分析:\\n总体召回率: 100.00%%\\n模型有效率: 100.00%%\\nReference ID映射准确率: 100.00%%\\n' > "$2"
"""
    )
    executable.chmod(0o755)
    match_dir = tmp_path / "sample/output_pt/0"
    match_dir.mkdir(parents=True)
    (match_dir / "matching_summary.txt").touch()
    script = Path(__file__).parents[1] / "scripts/3d/evaluation/accuracy_evaluation.sh"
    result = subprocess.run(
        ["bash", str(script), "sample", "--save-root", str(tmp_path)],
        env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "sample/accuracy_evaluation_pt"
    assert (output / "1_to_2.txt").is_file()
    assert "成功评估数量: 1" in (output / "summary.txt").read_text()
    assert "模型有效率: 100.00%" in (output / "summary.txt").read_text()


def test_root_cli_failure_exits_nonzero(tmp_path):
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            str(root / "main.py"),
            "--mode",
            "reconstruct",
            "--dataset",
            str(tmp_path / "missing"),
            "--recon_backend",
            "da3",
            "--save_root",
            str(tmp_path / "output"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "数据集路径不存在" in result.stderr


def test_concise_uses_matching_backend_and_forwards_options(tmp_path, monkeypatch):
    app = main.SKUDetectionMain()
    app.match_backend = "pi3x"
    monkeypatch.setattr(app, "validate_dataset", lambda _: True)
    calls = []
    monkeypatch.setattr(
        app, "run_sku_matching", lambda *a, **kw: calls.append(kw) or {"success": True}
    )
    monkeypatch.setattr(
        app,
        "run_accuracy_evaluation",
        lambda *a, **kw: calls.append(kw) or {"success": True},
    )
    app.run_concise_pipeline(str(tmp_path), "3d", reference_idx=2, device="cpu")
    assert calls[0]["reference_idx"] == 2
    assert calls[0]["backend"] == calls[1]["backend"] == "pi3x"


def test_invalid_reference_object_is_not_no_match():
    with pytest.raises(KeyError, match="point_indices"):
        matching._process_single_ref_object(
            0, {}, [], torch.zeros(1, 1, 2), torch.zeros(1, 1), 0, SKUMatchingConfig()
        )


def test_parallel_timeout_does_not_rerun_objects(monkeypatch):
    calls = []

    def process(ref_id, *args):
        calls.append(ref_id)
        return [{"object_id": ref_id}], {}

    class Executor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, fn, *args):
            future = Future()
            future.set_result(fn(*args))
            return future

    def completed(futures, **kwargs):
        yield next(iter(futures))
        raise TimeoutError("reference deadline")

    monkeypatch.setattr(matching, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(matching, "as_completed", completed)
    monkeypatch.setattr(matching, "_process_single_ref_object", process)
    with pytest.raises(TimeoutError, match="reference deadline"):
        matching.match_objects_by_correspondence(
            torch.zeros(1, 1, 2),
            torch.ones(1, 1),
            torch.ones(1, 1),
            {i: {} for i in range(3)},
            {"objects": [{"position": [0, 0, 100, 100], "confidences": {"det": 1.0}}]},
            0,
            0,
            SKUMatchingConfig(),
        )
    assert calls == [0, 1, 2]


def _pi3_cache(tmp_path, monkeypatch, backend="pi3"):
    config = SKUMatchingConfig.for_3d_mapping(backend=backend, device="cpu")
    config.enable_sam3_mask_sampling = False
    config.output_dir = str(tmp_path / f"output_3dmapping_{backend}/0")
    cache = tmp_path / f"{backend}_cache/predictions.npz"
    cache.parent.mkdir()
    np.savez(
        cache,
        image_ids=[0, 1],
        depth=np.ones((2, 4, 4, 1), dtype=np.float32),
        depth_conf=np.ones((2, 4, 4), dtype=np.float32),
        world_points=np.ones((2, 4, 4, 3), dtype=np.float32),
        world_points_conf=np.ones((2, 4, 4), dtype=np.float32),
        extrinsic=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        intrinsic=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
    )
    monkeypatch.setattr(matching, "PI3_SCENE_CACHE", {})
    return config


@pytest.mark.parametrize("backend", ["pi3", "pi3x"])
def test_cache_missing_requested_frame_is_rejected(tmp_path, monkeypatch, backend):
    config = _pi3_cache(tmp_path, monkeypatch, backend)
    transforms = [SimpleNamespace(image_id=0), SimpleNamespace(image_id=3)]
    with pytest.raises(KeyError, match="3"):
        matching.find_correspondences_3d_mapping(
            None,
            [{"objects": []}] * 2,
            torch.zeros(2, 3, 4, 4),
            config,
            transforms_info=transforms,
        )


def test_scene_cache_does_not_bypass_frame_alignment(tmp_path, monkeypatch):
    config = _pi3_cache(tmp_path, monkeypatch)
    transforms = [SimpleNamespace(image_id=0), SimpleNamespace(image_id=1)]
    inputs = (None, [{"objects": []}] * 2, torch.zeros(2, 3, 4, 4), config)
    matching.find_correspondences_3d_mapping(*inputs, transforms_info=transforms)
    transforms[1].image_id = 3
    with pytest.raises(KeyError, match="3"):
        matching.find_correspondences_3d_mapping(*inputs, transforms_info=transforms)


def test_target_boxes_are_transformed_once_for_multiple_references(
    tmp_path, monkeypatch
):
    config = _pi3_cache(tmp_path, monkeypatch)
    config.min_bbox_area = 0
    config.min_3d_sample_points = 1
    calls = []

    class Transform:
        def __init__(self, image_id):
            self.image_id = image_id

        def map_bbox_to_final(self, bbox):
            calls.append(self.image_id)
            return list(bbox)

    bbox = {"position": [0, 0, 4, 4], "confidences": {"det": 1.0}}
    monkeypatch.setattr(
        matching,
        "sample_3d_points_from_non_overlap_regions",
        lambda *a, **k: torch.ones(5, 3),
    )
    monkeypatch.setattr(matching, "project_3d_to_2d", lambda *a, **k: torch.ones(5, 2))
    validated_boxes = []

    def no_match(points, boxes, *args, **kwargs):
        validated_boxes.append(boxes)
        return None

    monkeypatch.setattr(
        matching, "find_best_matching_bbox_with_3d_validation", no_match
    )
    result, _ = matching.find_correspondences_3d_mapping(
        None,
        [{"objects": [bbox, bbox]}, {"objects": [bbox]}],
        torch.zeros(2, 3, 4, 4),
        config,
        transforms_info=[Transform(0), Transform(1)],
    )
    assert result == {}
    assert len(validated_boxes) == 2
    assert validated_boxes[0][0]["bbox"] == [0, 0, 4, 4]
    assert calls.count(1) == 1
