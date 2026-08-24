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
