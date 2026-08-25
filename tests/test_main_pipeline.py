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
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from src import inference


def _pipeline_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[main.SKUDetectionMain, Path]:
    """Build a pipeline whose non-classification stages are deterministic."""
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "detections_results").mkdir()
    app = main.SKUDetectionMain()
    app.save_root = tmp_path / "Output"
    app.match_backend = "da3"
    monkeypatch.setattr(app, "validate_dataset", lambda _path: True)
    monkeypatch.setattr(
        app, "run_reconstruction", lambda *_args, **_kwargs: {"success": True}
    )
    monkeypatch.setattr(
        app,
        "run_detection_visualization",
        lambda *_args, **_kwargs: {"success": True},
    )
    monkeypatch.setattr(
        app,
        "run_improved_sku_analysis",
        lambda *_args, **_kwargs: {"success": True},
    )
    monkeypatch.setattr(
        app,
        "run_accuracy_evaluation",
        lambda *_args, **_kwargs: {"success": True},
    )
    return app, dataset


def _published_classifier_payload(
    dataset: Path, output_root: Path, *, run_id: str = "123456789-4321"
) -> tuple[dict[str, object], Path, Path]:
    run_dir = (
        output_root
        / dataset.name
        / "personalcare_classification"
        / "runs"
        / run_id
    )
    detection_dir = run_dir / "detections"
    detection_dir.mkdir(parents=True)
    result_path = run_dir / "result.json"
    payload: dict[str, object] = {
        "success": True,
        "run_id": run_id,
        "detection_dir": str(detection_dir),
        "result_path": str(result_path),
        "frame_count": 1,
        "object_count": 2,
        "unavailable_count": 0,
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    (run_dir.parents[1] / "CURRENT").write_text(
        json.dumps({"run_id": run_id, "complete": True}), encoding="utf-8"
    )
    return payload, detection_dir, result_path


def test_personalcare_classification_accepts_only_successful_json_with_existing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A malformed, failed, or unpublished classifier stage cannot enter dedup."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    app = main.SKUDetectionMain()
    app.save_root = tmp_path / "Output"
    app.classifier_device = "cuda:7"
    payload, detection_dir, result_path = _published_classifier_payload(
        dataset, app.save_root
    )
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = app.run_personalcare_classification(str(dataset))

    assert result["success"] is True
    assert result["detection_dir"] == str(detection_dir)
    assert payload["result_path"] == str(result_path)
    assert captured["command"] == [
        "uv",
        "run",
        "--project",
        str(main.CLASSIFIER_ROOT),
        "python",
        str(main.CLASSIFIER_SCRIPT),
        "--dataset",
        str(dataset),
        "--output-root",
        str(app.save_root),
        "--device",
        "cuda:7",
    ]
    assert captured["kwargs"] == {
        "capture_output": True,
        "text": True,
        "check": False,
    }


@pytest.mark.parametrize(
    ("returncode", "payload", "make_detection_dir"),
    [
        (1, {"success": True}, False),
        (0, {"success": False}, False),
        (0, {"success": True, "detection_dir": "/missing"}, False),
    ],
)
def test_personalcare_classification_rejects_nonpublished_subprocess_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    returncode: int,
    payload: dict[str, object],
    make_detection_dir: bool,
) -> None:
    """Changing any subprocess success guard must block classified input."""
    detection_dir = tmp_path / "classified"
    if make_detection_dir:
        detection_dir.mkdir()
        payload["detection_dir"] = str(detection_dir)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=returncode,
            stdout=json.dumps(payload),
            stderr="classifier stderr",
        ),
    )

    result = main.SKUDetectionMain().run_personalcare_classification(
        str(tmp_path / "dataset")
    )

    assert result["success"] is False
    assert "classifier stderr" in result["error"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload, _run_dir, _publication_root: payload.update(run_id="invalid"),
        lambda payload, run_dir, _publication_root: payload.update(
            detection_dir=str(run_dir.parent / "unrelated"),
        ),
        lambda payload, _run_dir, _publication_root: payload.update(extra="forbidden"),
        lambda payload, _run_dir, _publication_root: payload.update(frame_count=True),
        lambda payload, _run_dir, _publication_root: payload.pop("object_count"),
        lambda payload, run_dir, _publication_root: (run_dir / "result.json").write_text(
            json.dumps({**payload, "object_count": 99}), encoding="utf-8"
        ),
        lambda _payload, _run_dir, publication_root: (publication_root / "CURRENT").write_text(
            json.dumps({"run_id": "123456789-4321", "complete": False}),
            encoding="utf-8",
        ),
    ],
)
def test_personalcare_classification_rejects_unbound_publication_payloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation
) -> None:
    """Any stale, incomplete, or mismatched Task2 publication must be rejected."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output_root = tmp_path / "Output"
    payload, _detection_dir, _result_path = _published_classifier_payload(
        dataset, output_root
    )
    run_dir = output_root / dataset.name / "personalcare_classification" / "runs" / payload["run_id"]
    publication_root = run_dir.parents[1]
    (run_dir.parent / "unrelated").mkdir()
    mutation(payload, run_dir, publication_root)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr="classifier stderr"
        ),
    )
    app = main.SKUDetectionMain()
    app.save_root = output_root

    result = app.run_personalcare_classification(str(dataset))

    assert result["success"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload, run_dir, _publication_root: (run_dir / "result.json").write_text(
            json.dumps({**payload, "frame_count": True}), encoding="utf-8"
        ),
        lambda payload, run_dir, _publication_root: (run_dir / "result.json").write_text(
            json.dumps({**payload, "extra": "forbidden"}), encoding="utf-8"
        ),
        lambda _payload, _run_dir, publication_root: (publication_root / "CURRENT").write_text(
            json.dumps({"run_id": "123456789-4321", "complete": 1}),
            encoding="utf-8",
        ),
        lambda _payload, _run_dir, publication_root: (publication_root / "CURRENT").write_text(
            json.dumps(
                {"run_id": "123456789-4321", "complete": True, "extra": 1}
            ),
            encoding="utf-8",
        ),
    ],
)
def test_personalcare_classification_rejects_type_confused_publication_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation
) -> None:
    """Task2 disk records must satisfy the same exact typed contract as stdout."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output_root = tmp_path / "Output"
    payload, _detection_dir, _result_path = _published_classifier_payload(
        dataset, output_root
    )
    run_dir = (
        output_root
        / dataset.name
        / "personalcare_classification"
        / "runs"
        / "123456789-4321"
    )
    mutation(payload, run_dir, run_dir.parents[1])
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr="classifier stderr"
        ),
    )
    app = main.SKUDetectionMain()
    app.save_root = output_root

    result = app.run_personalcare_classification(str(dataset))

    assert result["success"] is False


def test_personalcare_classification_rejects_nonunique_json_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Accepting an additional JSON value would make the subprocess boundary ambiguous."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"success": true}\n{"success": true}',
            stderr="classifier stderr",
        ),
    )

    result = main.SKUDetectionMain().run_personalcare_classification(
        str(tmp_path / "dataset")
    )

    assert result["success"] is False
    assert "one JSON object" in result["error"]


def test_run_dedup_sequence_forwards_the_classified_directory_to_task3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dropping the explicit Task 3 argument would regress classified global output."""
    from src import deduplicate_detections

    dataset = tmp_path / "dataset"
    classified = tmp_path / "classified"
    dataset.mkdir()
    classified.mkdir()
    app = main.SKUDetectionMain()
    app.save_root = tmp_path / "Output"
    captured: dict[str, object] = {}

    def fake_deduplicate(paths, **kwargs):
        captured["paths"] = paths
        captured["detections_dir"] = kwargs["detections_dir"]
        return {}

    monkeypatch.setattr(deduplicate_detections, "deduplicate_sequence", fake_deduplicate)

    result = app.run_dedup_sequence(
        str(dataset), algorithm="3d", backend="da3", detection_dir=classified
    )

    assert result["success"] is True
    assert captured["paths"].detections_dir == classified
    assert captured["detections_dir"] == classified


def test_matching_failure_joins_classification_and_stops_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A matching failure must preserve classification evidence but stop later stages."""
    app, dataset = _pipeline_fixture(monkeypatch, tmp_path)
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def classify(_dataset: str) -> dict[str, object]:
        started.set()
        assert release.wait(timeout=2)
        calls.append("classification_done")
        return {"success": True, "detection_dir": str(tmp_path / "classified")}

    def matching(*_args, **_kwargs):
        assert started.wait(timeout=2)
        calls.append("matching_failed")
        release.set()
        return {"success": False}

    monkeypatch.setattr(app, "run_personalcare_classification", classify)
    monkeypatch.setattr(app, "run_sku_matching", matching)
    monkeypatch.setattr(
        app,
        "run_improved_sku_analysis",
        lambda *_args, **_kwargs: calls.append("analysis") or {"success": True},
    )
    monkeypatch.setattr(
        app,
        "run_dedup_sequence",
        lambda *_args, **_kwargs: calls.append("dedup") or {"success": True},
    )
    monkeypatch.setattr(
        app,
        "run_accuracy_evaluation",
        lambda *_args, **_kwargs: calls.append("accuracy") or {"success": True},
    )

    summary = app.run_complete_pipeline(str(dataset), algorithm="3d")

    assert summary["matching"] is False
    assert summary["classification"] is True
    assert summary["improved_analysis"] is False
    assert summary["dedup"] is False
    assert summary["dedup_visualization"] is False
    assert summary["accuracy_evaluation"] is False
    assert calls.index("matching_failed") < calls.index("classification_done")
    assert calls == ["matching_failed", "classification_done"]


def test_pipeline_exception_waits_for_classifier_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An exception before join must still wait for the classifier executor."""
    app, dataset = _pipeline_fixture(monkeypatch, tmp_path)
    finished = threading.Event()
    shutdown_waits: list[bool] = []
    real_executor = main.ThreadPoolExecutor

    class TrackingExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self._executor = real_executor(*args, **kwargs)

        def submit(self, *args, **kwargs):
            return self._executor.submit(*args, **kwargs)

        def shutdown(self, *, wait: bool) -> None:
            shutdown_waits.append(wait)
            self._executor.shutdown(wait=wait)

    def classify(_dataset: str) -> dict[str, object]:
        time.sleep(0.05)
        finished.set()
        return {"success": True, "detection_dir": str(tmp_path / "classified")}

    monkeypatch.setattr(main, "ThreadPoolExecutor", TrackingExecutor)
    monkeypatch.setattr(app, "run_personalcare_classification", classify)
    monkeypatch.setattr(
        app,
        "run_detection_visualization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("viz failed")),
    )

    with pytest.raises(RuntimeError, match="viz failed"):
        app.run_complete_pipeline(str(dataset), algorithm="3d")

    assert finished.is_set()
    assert shutdown_waits == [True]


def test_pipeline_submit_exception_still_shuts_down_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Submitting the classifier is inside the executor lifetime boundary."""
    app, dataset = _pipeline_fixture(monkeypatch, tmp_path)
    shutdown_waits: list[bool] = []

    class SubmitRaisesExecutor:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def submit(self, *_args, **_kwargs):
            raise RuntimeError("submit failed")

        def shutdown(self, *, wait: bool) -> None:
            shutdown_waits.append(wait)

    monkeypatch.setattr(main, "ThreadPoolExecutor", SubmitRaisesExecutor)

    with pytest.raises(RuntimeError, match="submit failed"):
        app.run_complete_pipeline(str(dataset), algorithm="3d")

    assert shutdown_waits == [True]


def test_pipeline_cli_defaults_classifier_device_to_cuda_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the explicit device option would silently change CUDA ownership."""
    captured: dict[str, object] = {}

    class FakeApp:
        def __init__(self) -> None:
            self.default_dataset = ""
            self.save_root = None
            self.match_backend = ""
            self.classifier_device = ""
            self.config_path = None

        def run_complete_pipeline(self, *_args, **_kwargs) -> None:
            captured["classifier_device"] = self.classifier_device

    monkeypatch.setattr(main, "SKUDetectionMain", FakeApp)
    monkeypatch.setattr(main, "_configure_logging_to_save_root", lambda _path: None)
    monkeypatch.setattr(
        main.sys,
        "argv",
        ["main.py", "--mode", "pipeline"],
    )

    main.main()

    assert captured["classifier_device"] == "cuda:0"


def test_pipeline_joins_classification_only_before_dedup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dedup must observe the completed classified copy, never the in-flight run."""
    app, dataset = _pipeline_fixture(monkeypatch, tmp_path)
    entered = threading.Event()
    release = threading.Event()
    classified = tmp_path / "classified"
    calls: list[str] = []

    def classify(_dataset: str) -> dict[str, object]:
        entered.set()
        assert release.wait(timeout=2)
        calls.append("classification_done")
        return {"success": True, "detection_dir": str(classified)}

    def reconstruct(*_args, **_kwargs):
        assert entered.wait(timeout=2)
        return {"success": True}

    def matching(*_args, **_kwargs):
        release.set()
        calls.append("matching_done")
        return {"success": True}

    monkeypatch.setattr(app, "run_personalcare_classification", classify)
    monkeypatch.setattr(app, "run_reconstruction", reconstruct)
    monkeypatch.setattr(app, "run_sku_matching", matching)
    monkeypatch.setattr(
        app,
        "run_dedup_sequence",
        lambda *_args, **kwargs: calls.append(f"dedup:{kwargs['detection_dir']}")
        or {"success": True},
    )

    summary = app.run_complete_pipeline(str(dataset), algorithm="3d")

    assert summary["classification"] is True
    assert calls.index("classification_done") < calls.index(f"dedup:{classified}")
    assert calls.index("matching_done") < calls.index(f"dedup:{classified}")


def test_classification_failure_stops_before_dedup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed classifier retains core artifacts but blocks global publication."""
    app, dataset = _pipeline_fixture(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        app,
        "run_personalcare_classification",
        lambda _path: {"success": False, "error": "model failed"},
    )
    monkeypatch.setattr(
        app,
        "run_sku_matching",
        lambda *_args, **_kwargs: calls.append("matching") or {"success": True},
    )
    monkeypatch.setattr(
        app,
        "run_dedup_sequence",
        lambda *_args, **_kwargs: calls.append("dedup") or {"success": True},
    )

    summary = app.run_complete_pipeline(str(dataset), algorithm="3d")

    assert summary["reconstruction"] is True
    assert summary["matching"] is True
    assert summary["classification"] is False
    assert summary["dedup"] is False
    assert summary["dedup_visualization"] is False
    assert summary["accuracy_evaluation"] is False
    assert calls == ["matching"]


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
        "run_personalcare_classification",
        lambda _path: {"success": True, "detection_dir": str(tmp_path / "classified")},
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


def test_batch_matching_serial_failure_is_returned_with_reference_identity(
    monkeypatch, tmp_path: Path
) -> None:
    """A failed per-reference StepResult must make the serial batch fail."""
    dataset = tmp_path / "datasets" / "sample"
    detections = dataset / "detections_results"
    detections.mkdir(parents=True)
    app = main.SKUDetectionMain()
    calls: list[int] = []

    monkeypatch.setattr(
        "utils.data_utils.load_detections",
        lambda *_args, **_kwargs: [(17, {}), (23, {})],
    )

    def fake_single(*_args, **_kwargs):
        reference_idx = _args[2]
        calls.append(reference_idx)
        if reference_idx == 0:
            return {"success": False, "error": "reference mask failed"}
        return {"success": True}

    monkeypatch.setattr(app, "_run_single_matching", fake_single)

    result = app.run_sku_matching(str(dataset), batch_all_refs=True, parallel_refs=1)

    assert calls == [0, 1]
    assert result["success"] is False
    assert result["failed_references"] == [
        {
            "reference_idx": 0,
            "image_index": 17,
            "error": "reference mask failed",
        }
    ]


def test_serial_batch_matching_collects_exception_and_continues_references(
    monkeypatch, tmp_path: Path
) -> None:
    """A direct serial exception fails that reference without aborting the batch."""
    dataset = tmp_path / "datasets" / "sample"
    detections = dataset / "detections_results"
    detections.mkdir(parents=True)
    app = main.SKUDetectionMain()
    completed: list[int] = []

    monkeypatch.setattr(
        "utils.data_utils.load_detections",
        lambda *_args, **_kwargs: [(17, {}), (23, {}), (31, {})],
    )

    def fake_single(*_args, **_kwargs):
        reference_idx = _args[2]
        if reference_idx == 1:
            raise RuntimeError("serial worker exploded")
        completed.append(reference_idx)
        return {"success": True}

    monkeypatch.setattr(app, "_run_single_matching", fake_single)

    result = app.run_sku_matching(str(dataset), batch_all_refs=True, parallel_refs=1)

    assert completed == [0, 2]
    assert result["success"] is False
    assert result["failed_references"] == [
        {
            "reference_idx": 1,
            "image_index": 23,
            "error": "serial worker exploded",
        }
    ]


def test_parallel_batch_matching_collects_step_failures_and_future_exceptions(
    monkeypatch, tmp_path: Path
) -> None:
    """Parallel scheduling reports every failed reference without cancelling peers."""
    dataset = tmp_path / "datasets" / "sample"
    detections = dataset / "detections_results"
    detections.mkdir(parents=True)
    app = main.SKUDetectionMain()
    completed: list[int] = []

    monkeypatch.setattr(
        "utils.data_utils.load_detections",
        lambda *_args, **_kwargs: [(17, {}), (23, {}), (31, {})],
    )

    def fake_single(*_args, **_kwargs):
        reference_idx = _args[2]
        if reference_idx == 0:
            return {"success": False, "error": "reference mask failed"}
        if reference_idx == 1:
            raise RuntimeError("worker exploded")
        completed.append(reference_idx)
        return {"success": True}

    monkeypatch.setattr(app, "_run_single_matching", fake_single)

    result = app.run_sku_matching(str(dataset), batch_all_refs=True, parallel_refs=2)

    assert completed == [2]
    assert result["success"] is False
    assert result["failed_references"] == [
        {
            "reference_idx": 0,
            "image_index": 17,
            "error": "reference mask failed",
        },
        {
            "reference_idx": 1,
            "image_index": 23,
            "error": "worker exploded",
        },
    ]


def test_complete_pipeline_matching_failure_stops_later_publication(
    monkeypatch, tmp_path: Path
) -> None:
    """A matching failure is terminal after classifier join."""
    dataset = tmp_path / "datasets" / "sample"
    (dataset / "images").mkdir(parents=True)
    (dataset / "detections_results").mkdir()
    app = main.SKUDetectionMain()

    monkeypatch.setattr(app, "validate_dataset", lambda _path: True)
    monkeypatch.setattr(
        app, "run_detection_visualization", lambda *_args, **_kwargs: {"success": True}
    )
    monkeypatch.setattr(
        app,
        "run_sku_matching",
        lambda *_args, **_kwargs: {
            "success": False,
            "failed_references": [
                {"reference_idx": 0, "image_index": 0, "error": "failed"}
            ],
        },
    )
    monkeypatch.setattr(
        app, "run_improved_sku_analysis", lambda *_args, **_kwargs: {"success": True}
    )
    monkeypatch.setattr(
        app, "run_dedup_sequence", lambda *_args, **_kwargs: {"success": True}
    )
    monkeypatch.setattr(
        app,
        "run_personalcare_classification",
        lambda _path: {"success": True, "detection_dir": str(tmp_path / "classified")},
    )
    monkeypatch.setattr(
        app, "run_accuracy_evaluation", lambda *_args, **_kwargs: {"success": True}
    )

    summary = app.run_complete_pipeline(str(dataset), algorithm="point_tracking")

    assert summary["matching"] is False
    assert summary["classification"] is True
    assert summary["improved_analysis"] is False
    assert summary["dedup"] is False
    assert summary["dedup_visualization"] is False
    assert summary["accuracy_evaluation"] is False


def test_root_parallel_refs_publish_real_complete_frame_cache(
    monkeypatch, tmp_path: Path
) -> None:
    """The root scheduler serializes real producer/cache calls for every frame."""
    from PIL import Image

    from utils.data_utils import extract_bboxes_from_detections, load_detections
    from utils.sam3_mask_cache import FrameMaskCacheError, load_complete_frame_masks
    from utils import sam3_utils
    from utils.transforms import DA3ImageTransform

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

    def cache_bound_transform(frame_id: int) -> DA3ImageTransform:
        transform = DA3ImageTransform(8, 6, 4, 3)
        transform.bind_da3_cache_geometry(
            np.asarray([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]), (3, 4)
        )
        transform.image_id = frame_id
        return transform

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
                transforms.append(cache_bound_transform(frame_id))
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
        transform = cache_bound_transform(frame_id)
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
