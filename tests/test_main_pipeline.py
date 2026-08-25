"""Regression coverage for DA3 pipeline artifact routing."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from src import inference


def test_accuracy_evaluation_invokes_da3_report_for_the_current_save_root(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "datasets" / "sample"
    output_root = tmp_path / "runtime-output"
    (output_root / dataset.name / "output_3dmapping_da3").mkdir(parents=True)
    app = main.SKUDetectionMain()
    app.save_root = output_root
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args[0]
        captured["cwd"] = kwargs["cwd"]
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = app.run_accuracy_evaluation(str(dataset), backend="da3")

    script = (
        main.PROJECT_ROOT
        / "scripts"
        / "3d"
        / "evaluation"
        / "accuracy_evaluation.sh"
    )
    assert result["success"] is True
    assert captured["args"] == [
        "bash",
        str(script),
        dataset.name,
        "--backend",
        "da3",
        "--save-root",
        str(output_root),
    ]
    assert captured["cwd"] == str(main.PROJECT_ROOT)


def test_da3_pipeline_reuses_only_metric_schema_v3_predictions_cache(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "datasets" / "sample"
    output_root = tmp_path / "runtime-output"
    cache_path = output_root / dataset.name / "da3_cache" / "predictions.npz"
    cache_path.parent.mkdir(parents=True)
    np.savez_compressed(
        cache_path,
        cache_schema_version=np.asarray(3, dtype=np.int32),
        is_metric=np.asarray(1, dtype=np.int32),
    )
    app = main.SKUDetectionMain()
    app.save_root = output_root
    app.match_backend = "da3"
    calls: list[str] = []

    monkeypatch.setattr(app, "validate_dataset", lambda _path: True)
    monkeypatch.setattr(
        app,
        "run_reconstruction",
        lambda *_args, **_kwargs: calls.append("reconstruct") or {"success": True},
    )
    monkeypatch.setattr(
        app,
        "run_detection_visualization",
        lambda *_args, **_kwargs: {"success": True},
    )
    monkeypatch.setattr(
        app,
        "run_sku_matching",
        lambda *_args, **_kwargs: {"success": True},
    )
    monkeypatch.setattr(
        app,
        "run_improved_sku_analysis",
        lambda *_args, **_kwargs: {"success": True},
    )
    monkeypatch.setattr(
        app,
        "run_dedup_sequence",
        lambda *_args, **_kwargs: {"success": False},
    )
    monkeypatch.setattr(
        app,
        "run_accuracy_evaluation",
        lambda *_args, **_kwargs: {"success": True},
    )

    summary = app.run_complete_pipeline(str(dataset), algorithm="3d")

    assert calls == []
    assert summary["reconstruction"] is True


def test_inference_main_accepts_explicit_argv(monkeypatch, tmp_path: Path) -> None:
    """Embedding inference must not require callers to mutate process argv."""
    images = tmp_path / "images"
    detections = tmp_path / "detections"
    images.mkdir()
    detections.mkdir()
    observed: list[int] = []

    def fake_3d(args):
        observed.append(args.reference_idx)
        return {}

    monkeypatch.setattr(inference, "run_3d_mapping", fake_3d)

    inference.main(
        [
            "--image_folder",
            str(images),
            "--detection_dir",
            str(detections),
            "--reference_idx",
            "7",
            "--device",
            "cpu",
        ]
    )

    assert observed == [7]


def test_parallel_refs_keep_explicit_reference_argv_serialized(
    monkeypatch, tmp_path: Path
) -> None:
    """parallel_refs schedules work but serializes global-RNG matching calls."""
    dataset = tmp_path / "datasets" / "sample"
    images = dataset / "images"
    detections = dataset / "detections_results"
    images.mkdir(parents=True)
    detections.mkdir()
    for frame_id in (0, 1):
        (images / f"{frame_id}.JPG").write_bytes(b"not-read-by-fake-match")
        (detections / f"{frame_id}.json").write_text(
            json.dumps({"objects": [{"position": [0, 0, 4, 4]}]})
        )

    app = main.SKUDetectionMain()
    output_root = tmp_path / "runtime-output"
    app.save_root = output_root
    active = 0
    max_active = 0
    guard = threading.Lock()
    observed: list[tuple[int, str]] = []

    def fake_3d(args):
        nonlocal active, max_active
        config = inference.create_config_from_args(args, "3d_mapping")
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with guard:
            observed.append((args.reference_idx, config.output_dir))
            active -= 1
        return {}

    monkeypatch.setattr(inference, "run_3d_mapping", fake_3d)

    result = app.run_sku_matching(
        str(dataset),
        algorithm="3d",
        max_images=2,
        device="cpu",
        batch_all_refs=True,
        backend="da3",
        parallel_refs=2,
    )

    assert result["success"] is True
    assert max_active == 1
    assert sorted(observed) == [
        (0, str(output_root / dataset.name / "output_3dmapping_da3" / "0")),
        (1, str(output_root / dataset.name / "output_3dmapping_da3" / "1")),
    ]


def test_root_parallel_refs_publish_real_complete_frame_cache(
    monkeypatch, tmp_path: Path
) -> None:
    """The root scheduler serializes real producer/cache calls for every frame."""
    from PIL import Image

    from utils.data_utils import extract_bboxes_from_detections, load_detections
    from utils.sam3_mask_cache import FrameMaskCacheError, load_complete_frame_masks
    from utils import sam3_utils
    from utils.transforms import Pi3ImageTransform

    dataset = tmp_path / "datasets" / "sample"
    images = dataset / "images"
    detections_dir = dataset / "detections_results"
    images.mkdir(parents=True)
    detections_dir.mkdir()
    frames = [
        {"objects": [{"position": [0.0, 0.0, 4.0, 4.0]}]},
        {"objects": [{"position": [4.0, 0.0, 5.0, 1.0]}]},
        {"objects": []},
    ]
    for frame_id, frame in enumerate(frames):
        Image.new("RGB", (8, 6)).save(images / f"{frame_id}.JPG")
        (detections_dir / f"{frame_id}.json").write_text(json.dumps(frame))
    config_path = tmp_path / "matching.yaml"
    config_path.write_text(
        "inference:\n"
        "  enable_sam3_mask_sampling: true\n"
        "  sam3_checkpoint_path: unused-by-test.pt\n"
    )

    active = 0
    max_active = 0
    guard = threading.Lock()
    records: list[tuple[int, str, list[str]]] = []
    producer_calls: list[list[list[float]]] = []

    def fake_self_exemplar(*, bboxes_xyxy, **_kwargs):
        producer_calls.append(bboxes_xyxy)
        return [np.ones((3, 4), dtype=bool) for _ in bboxes_xyxy]

    monkeypatch.setattr(sam3_utils, "sam3_masks_self_exemplar", fake_self_exemplar)

    class FakeSystem:
        def __init__(self, config):
            self.config = config

        def process_images(
            self, image_folder, detection_dir, reference_image_idx, max_images
        ):
            nonlocal active, max_active
            image_paths = [str(images / f"{frame_id}.JPG") for frame_id in range(3)]
            detections = load_detections(detection_dir)
            transforms = []
            for frame_id in range(3):
                transform = Pi3ImageTransform(8, 6, 4, 3)
                transform.image_id = frame_id
                transforms.append(transform)
            events: list[str] = []
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            try:
                for frame_id, frame in enumerate(detections):
                    request = sam3_utils._processed_frame_request(
                        cache_root=Path(self.config.sam3_mask_cache_root),
                        image_path=Path(image_paths[frame_id]),
                        image_id=frame_id,
                        frame_detections=frame["objects"],
                        transform=transforms[frame_id],
                    )
                    try:
                        load_complete_frame_masks(request)
                        events.append("hit")
                    except FrameMaskCacheError:
                        events.append("miss")
                    ref_bboxes = extract_bboxes_from_detections(
                        detections, frame_id, self.config
                    )
                    sam3_utils.get_self_exemplar_masks_for_reference(
                        self.config,
                        image_path=Path(image_paths[frame_id]),
                        image_id=frame_id,
                        frame_detections=frame["objects"],
                        matching_object_ids=[
                            int(bbox["object_id"]) for bbox in ref_bboxes
                        ],
                        transform=transforms[frame_id],
                    )
                records.append(
                    (reference_image_idx, self.config.output_dir, events)
                )
                return {}
            finally:
                with guard:
                    active -= 1

        def cleanup(self):
            return None

    monkeypatch.setattr(inference, "SKUMatchingSystem", FakeSystem)
    global_argv = ["pytest-sentinel"]
    monkeypatch.setattr(main.sys, "argv", global_argv)
    app = main.SKUDetectionMain()
    output_root = tmp_path / "runtime-output"
    app.save_root = output_root
    app.config_path = config_path

    result = app.run_sku_matching(
        str(dataset),
        algorithm="3d",
        max_images=3,
        device="cpu",
        batch_all_refs=True,
        backend="da3",
        parallel_refs=2,
    )

    assert result["success"] is True
    assert main.sys.argv is global_argv
    assert max_active == 1
    assert sorted((reference_idx, output_dir) for reference_idx, output_dir, _ in records) == [
        (0, str(output_root / dataset.name / "output_3dmapping_da3" / "0")),
        (1, str(output_root / dataset.name / "output_3dmapping_da3" / "1")),
        (2, str(output_root / dataset.name / "output_3dmapping_da3" / "2")),
    ]
    event_sequences = [events for _reference_idx, _output_dir, events in records]
    assert event_sequences.count(["miss", "miss", "miss"]) == 1
    assert event_sequences.count(["hit", "hit", "hit"]) == 2
    assert len(producer_calls) == 2
    for frame_id, frame in enumerate(frames):
        transform = Pi3ImageTransform(8, 6, 4, 3)
        transform.image_id = frame_id
        request = sam3_utils._processed_frame_request(
            cache_root=output_root / dataset.name / "sam3_mask_cache" / "v2",
            image_path=images / f"{frame_id}.JPG",
            image_id=frame_id,
            frame_detections=frame["objects"],
            transform=transform,
        )
        assert set(load_complete_frame_masks(request).masks_by_object_id) == set(
            range(len(frame["objects"]))
        )
