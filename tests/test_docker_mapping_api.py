from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

import bson
import httpx
import pytest

from docker import api, request_runner


def _exit_child(*_args) -> dict[str, object]:
    os._exit(17)


async def _post_api_async(body: bytes) -> httpx.Response:
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        return await client.post(
            "/api",
            content=body,
            headers={"content-type": "application/bson"},
        )


def _post_api(body: bytes) -> httpx.Response:
    return asyncio.run(_post_api_async(body))


class _Pipeline:
    last: "_Pipeline | None" = None

    def __init__(self) -> None:
        type(self).last = self
        self.save_root: Path | None = None
        self.match_backend = ""
        self.classifier_enabled = True
        self.config_path: Path | None = None

    def run_complete_pipeline(self, dataset: str, algorithm: str, model_path: str):
        self.dataset = dataset
        self.algorithm = algorithm
        self.model_path = model_path
        return {
            "validation": True,
            "reconstruction": True,
            "visualization": False,
            "matching": True,
            "improved_analysis": True,
            "classification": True,
            "dedup": True,
            "dedup_visualization": False,
            "accuracy_evaluation": False,
        }


def test_runner_fixes_da3_requires_complete_summary_and_exports_viewer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exported: dict[str, object] = {}
    monkeypatch.setattr(request_runner, "SKUDetectionMain", _Pipeline)
    monkeypatch.setattr(
        request_runner,
        "export_web_viewer_bundle",
        lambda **kwargs: exported.update(kwargs) or {"manifest_path": "manifest.json"},
    )

    result = request_runner.run_mapping_request(
        tmp_path / "dataset", tmp_path / "outputs", tmp_path / "viewer", "/models/da3"
    )

    pipeline = _Pipeline.last
    assert pipeline is not None
    assert pipeline.match_backend == "da3"
    assert pipeline.classifier_enabled is False
    assert pipeline.save_root == tmp_path / "outputs"
    assert pipeline.algorithm == "3d"
    assert pipeline.model_path == "/models/da3"
    assert exported == {
        "dataset_name": "dataset",
        "da3_cache_path": tmp_path / "outputs" / "dataset" / "da3_cache" / "predictions.npz",
        "global_mapping_path": tmp_path / "outputs" / "dataset" / "dedup_detections" / "global_mapping.json",
        "output_dir": tmp_path / "viewer",
        "source_images_dir": tmp_path / "dataset" / "images",
        "sam3_mask_cache_root": tmp_path / "outputs" / "dataset" / "sam3_mask_cache" / "v2",
    }
    assert result["viewer_root"] == str(tmp_path / "viewer")
    assert result["global_skus_path"] == str(
        tmp_path / "outputs" / "dataset" / "dedup_detections" / "global_skus.json"
    )


def test_runner_reports_the_first_missing_required_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _FailedPipeline(_Pipeline):
        def run_complete_pipeline(self, *_args, **_kwargs):
            return {"validation": True, "reconstruction": False}

    monkeypatch.setattr(request_runner, "SKUDetectionMain", _FailedPipeline)

    with pytest.raises(request_runner.RequestRunnerError, match="reconstruction"):
        request_runner.run_mapping_request(
            tmp_path / "dataset", tmp_path / "outputs", tmp_path / "viewer", "/models/da3"
        )


def test_api_round_trips_exact_success_bson_and_pipeline_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DA3_MODEL_PATH", "/models/da3")
    monkeypatch.setattr(
        api,
        "prepare_request",
        lambda _inputs, root: type("Prepared", (), {"dataset_dir": root / "dataset"})(),
    )
    monkeypatch.setattr(
        api,
        "build_success_response",
        lambda _skus, _viewer: {"global_skus": ['{"skus":[]}'], "viewer_bundle": b"zip"},
    )
    monkeypatch.setattr(
        api,
        "execute_mapping_child",
        lambda *_args: {"global_skus_path": "global_skus.json", "viewer_root": "viewer"},
    )
    response = _post_api(bson.dumps({"images": [], "skus": [], "project_id": 51}))

    assert response.status_code == 200
    assert bson.loads(response.content) == {
        "global_skus": ['{"skus":[]}'],
        "viewer_bundle": b"zip",
    }

    monkeypatch.setattr(
        api,
        "execute_mapping_child",
        lambda *_args: (_ for _ in ()).throw(api.RequestExecutionError("matching", "boom")),
    )
    response = _post_api(bson.dumps({"images": [], "skus": [], "project_id": 51}))
    assert response.status_code == 500
    assert bson.loads(response.content) == {"stage": "matching", "message": "boom"}


def test_api_returns_contract_errors_as_bson_400() -> None:
    response = _post_api(b"not bson")

    assert response.status_code == 400
    payload = bson.loads(response.content)
    assert set(payload) == {"stage", "message"}
    assert payload["stage"] == "contract"


def test_api_returns_bson_400_when_decoder_raises_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode_bson = bson.loads
    monkeypatch.setattr(
        api.bson,
        "loads",
        lambda _body: (_ for _ in ()).throw(UnboundLocalError("unsupported BSON")),
    )

    response = _post_api(b"valid body is irrelevant")
    monkeypatch.setattr(api.bson, "loads", decode_bson)

    assert response.status_code == 400
    assert decode_bson(response.content)["stage"] == "contract"


def test_api_returns_bson_500_when_request_workspace_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenTemporaryDirectory:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            raise OSError("workspace unavailable")

        def __exit__(self, *_args) -> None:
            pass

    monkeypatch.setattr(api.tempfile, "TemporaryDirectory", _BrokenTemporaryDirectory)

    response = _post_api(bson.dumps({"images": [], "skus": [], "project_id": 51}))

    assert response.status_code == 500
    assert bson.loads(response.content) == {
        "stage": "request",
        "message": "workspace unavailable",
    }


def test_api_serializes_concurrent_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DA3_MODEL_PATH", "/models/da3")
    monkeypatch.setattr(
        api,
        "prepare_request",
        lambda _inputs, root: type("Prepared", (), {"dataset_dir": root / "dataset"})(),
    )
    monkeypatch.setattr(
        api,
        "build_success_response",
        lambda _skus, _viewer: {"global_skus": [], "viewer_bundle": b"zip"},
    )
    active = 0
    maximum = 0
    guard = threading.Lock()

    def blocking_child(*_args):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return {"global_skus_path": "global_skus.json", "viewer_root": "viewer"}

    monkeypatch.setattr(api, "execute_mapping_child", blocking_child)
    body = bson.dumps({"images": [], "skus": [], "project_id": 51})
    statuses: list[int] = []

    def call() -> None:
        statuses.append(_post_api(body).status_code)

    threads = [threading.Thread(target=call), threading.Thread(target=call)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert statuses == [200, 200]
    assert maximum == 1


def test_child_execution_requests_spawn_context(tmp_path: Path) -> None:
    contexts: list[str] = []

    class _Sender:
        def __init__(self) -> None:
            self.payload = None
            self.closed = False

        def send(self, payload):
            self.payload = payload

        def close(self) -> None:
            self.closed = True

    class _Receiver:
        def __init__(self, sender: _Sender) -> None:
            self.sender = sender

        def poll(self) -> bool:
            return self.sender.payload is not None

        def recv(self):
            return self.sender.payload

        def close(self) -> None:
            pass

    class _Process:
        sentinel = object()

        def __init__(self, target, args) -> None:
            self.target = target
            self.args = args
            self.exitcode = 0
            self.join_calls = 0
            self.terminated = False
            self.alive = True

        def start(self) -> None:
            self.target(*self.args)
            self.alive = False

        def join(self) -> None:
            self.join_calls += 1

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

    class _Context:
        def Pipe(self, duplex: bool):
            assert duplex is False
            sender = _Sender()
            return _Receiver(sender), sender

        def Process(self, *, target, args):
            return _Process(target, args)

    def context_factory(method: str):
        contexts.append(method)
        return _Context()

    result = api.execute_mapping_child(
        tmp_path / "dataset",
        tmp_path / "outputs",
        tmp_path / "viewer",
        "/models/da3",
        context_factory=context_factory,
        runner=lambda *_args: {"summary": {"dedup": True}},
        waiter=lambda connections: [connections[0]],
    )

    assert contexts == ["spawn"]
    assert result == {"summary": {"dedup": True}}


def test_child_exit_is_reported_and_reaped(tmp_path: Path) -> None:
    with pytest.raises(api.RequestExecutionError, match="17") as error:
        api.execute_mapping_child(
            tmp_path / "dataset",
            tmp_path / "outputs",
            tmp_path / "viewer",
            "/models/da3",
            runner=_exit_child,
        )

    assert error.value.stage == "child"
