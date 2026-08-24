"""Regression coverage for DA3 pipeline artifact routing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


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
