from __future__ import annotations

import asyncio
import io
import sys
import threading
import time
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import bson
import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from docker import api, processor, test_api
from utils import matching_algorithms, sku_matching_system


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
            "matching": True,
            "improved_analysis": True,
            "classification": True,
            "dedup": True,
        }


def test_processor_runs_da3_pipeline_and_exports_viewer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exported: dict[str, object] = {}
    monkeypatch.setattr(processor, "SKUDetectionMain", _Pipeline)
    monkeypatch.setattr(
        processor,
        "export_web_viewer_bundle",
        lambda **kwargs: exported.update(kwargs)
        or {
            "manifest_path": str(
                tmp_path / "viewer" / "runs" / "run-1" / "manifest.json"
            )
        },
    )

    result = processor.run_mapping_request(
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
        "da3_cache_path": tmp_path
        / "outputs"
        / "dataset"
        / "da3_cache"
        / "predictions.npz",
        "global_mapping_path": tmp_path
        / "outputs"
        / "dataset"
        / "dedup_detections"
        / "global_mapping.json",
        "output_dir": tmp_path / "viewer",
        "source_images_dir": tmp_path / "dataset" / "images",
        "sam3_mask_cache_root": tmp_path
        / "outputs"
        / "dataset"
        / "sam3_mask_cache"
        / "v2",
    }
    assert result["viewer_dir"] == str(tmp_path / "viewer" / "runs" / "run-1")
    assert result["global_skus_path"] == str(
        tmp_path / "outputs" / "dataset" / "dedup_detections" / "global_skus.json"
    )


def test_processor_does_not_wrap_pipeline_status_as_stage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _FailedPipeline(_Pipeline):
        def run_complete_pipeline(self, *_args, **_kwargs):
            return {"validation": True, "reconstruction": False}

    monkeypatch.setattr(processor, "SKUDetectionMain", _FailedPipeline)
    monkeypatch.setattr(
        processor,
        "export_web_viewer_bundle",
        lambda **_kwargs: {
            "manifest_path": str(tmp_path / "viewer" / "manifest.json")
        },
    )

    result = processor.run_mapping_request(
        tmp_path / "dataset",
        tmp_path / "outputs",
        tmp_path / "viewer",
        "/models/da3",
    )

    assert result["viewer_dir"] == str(tmp_path / "viewer")


def test_processor_propagates_viewer_errors_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    monkeypatch.setattr(processor, "SKUDetectionMain", _Pipeline)
    monkeypatch.setattr(
        processor,
        "export_web_viewer_bundle",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("viewer failed")),
    )
    with pytest.raises(ValueError, match="viewer failed"):
        processor.run_mapping_request(
            tmp_path / "dataset",
            tmp_path / "outputs",
            tmp_path / "viewer",
            "/models/da3",
        )


def test_process_directly_composes_request_pipeline_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setenv("DA3_MODEL_PATH", "/models/da3")

    def prepare(inputs, work_root):
        calls["inputs"] = inputs
        calls["work_root"] = work_root
        return SimpleNamespace(dataset_dir=work_root / "dataset", taskid="task-01")

    def run(dataset_dir, output_root, viewer_root, model_path):
        calls["run"] = (dataset_dir, output_root, viewer_root, model_path)
        return {
            "global_skus_path": str(output_root / "dataset" / "global_skus.json"),
            "viewer_dir": str(viewer_root / "generation"),
        }

    def build(global_skus_path, viewer_dir):
        calls["build"] = (global_skus_path, viewer_dir)
        return {"global_skus": ['{"objects":[]}'], "viewer_bundle": b"zip"}

    class _CosUploadConfig:
        @classmethod
        def from_env(cls):
            calls["config"] = True
            return cls()

    def upload(taskid, viewer_bundle, config):
        calls["upload"] = (taskid, viewer_bundle, config)
        return {
            "viewer_bundle_url": "https://cos.example/task-01/viewer_bundle.zip",
        }

    monkeypatch.setattr(processor, "prepare_request", prepare)
    monkeypatch.setattr(processor, "run_mapping_request", run)
    monkeypatch.setattr(processor, "build_success_response", build)
    monkeypatch.setattr(processor, "CosUploadConfig", _CosUploadConfig, raising=False)
    monkeypatch.setattr(processor, "upload_viewer_bundle", upload, raising=False)
    matching_algorithms.PI3_SCENE_CACHE["old-scene"] = {"tensor": object()}
    sku_matching_system._DA3_IMAGE_CACHE["old-images"] = object()
    sku_matching_system._DA3_TRANSFORMS_CACHE["old-transforms"] = [object()]

    inputs = {"taskID": "task-01", "images": [b"image"], "skus": ["{}"]}
    result = processor.process(inputs)

    work_root = calls["work_root"]
    assert calls["inputs"] is inputs
    assert calls["run"] == (
        work_root / "dataset",
        work_root / "outputs",
        work_root / "viewer",
        "/models/da3",
    )
    assert calls["build"] == (
        work_root / "outputs" / "dataset" / "global_skus.json",
        work_root / "viewer" / "generation",
    )
    assert calls["config"] is True
    assert calls["upload"] == (
        "task-01",
        b"zip",
        calls["upload"][2],
    )
    assert result == {
        "global_skus": ['{"objects":[]}'],
        "viewer_bundle": b"zip",
        "cos": {
            "taskID": "task-01",
            "viewer_bundle_url": "https://cos.example/task-01/viewer_bundle.zip",
        },
    }
    assert matching_algorithms.PI3_SCENE_CACHE == {}
    assert sku_matching_system._DA3_IMAGE_CACHE == {}
    assert sku_matching_system._DA3_TRANSFORMS_CACHE == {}


def test_process_clears_request_caches_when_pipeline_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DA3_MODEL_PATH", "/models/da3")
    monkeypatch.setattr(
        processor,
        "prepare_request",
        lambda _inputs, root: SimpleNamespace(dataset_dir=root / "dataset"),
    )
    monkeypatch.setattr(
        processor,
        "run_mapping_request",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("pipeline failed")),
    )
    matching_algorithms.PI3_SCENE_CACHE["failed-scene"] = {"tensor": object()}
    sku_matching_system._DA3_IMAGE_CACHE["failed-images"] = object()
    sku_matching_system._DA3_TRANSFORMS_CACHE["failed-transforms"] = [object()]

    with pytest.raises(RuntimeError, match="pipeline failed"):
        processor.process({"images": [b"image"], "skus": ["{}"]})

    assert matching_algorithms.PI3_SCENE_CACHE == {}
    assert sku_matching_system._DA3_IMAGE_CACHE == {}
    assert sku_matching_system._DA3_TRANSFORMS_CACHE == {}


@pytest.mark.parametrize(
    "members",
    [
        [
            "manifest.json",
            "manifest.json",
            "positions.f32.bin",
            "colors.u8.bin",
            "normals.i8.bin",
            "objects.json",
        ],
        [
            "manifest.json",
            "positions.f32.bin",
            "colors.u8.bin",
            "normals.i8.bin",
            "objects.json",
            "thumbs/nested/0.jpg",
        ],
    ],
)
def test_client_rejects_duplicate_or_nested_viewer_bundle_members(
    members: list[str],
) -> None:
    bundle = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(bundle, "w") as archive:
            for member in members:
                archive.writestr(member, b"x")

    with pytest.raises(ValueError):
        test_api._verify_viewer_bundle(bundle.getvalue())


def test_client_requires_task_id_and_validates_cos_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    classifier_result = tmp_path / "classifier"
    (dataset / "images").mkdir(parents=True)
    classifier_result.mkdir()
    (dataset / "images" / "0.jpg").write_bytes(b"image")
    (classifier_result / "0.json").write_text("{}", encoding="utf-8")

    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w") as archive:
        for name in (
            "manifest.json",
            "positions.f32.bin",
            "colors.u8.bin",
            "normals.i8.bin",
            "objects.json",
        ):
            archive.writestr(name, b"x")

    class _Response:
        content = bson.dumps(
            {
                "global_skus": ['{"objects":[]}'],
                "viewer_bundle": bundle.getvalue(),
                "cos": {
                    "taskID": "task-01",
                    "viewer_bundle_url": "https://cos.example/task-01/viewer_bundle.zip",
                },
            }
        )

        def raise_for_status(self) -> None:
            pass

    sent: dict[str, object] = {}

    def post(_url, **kwargs):
        sent.update(kwargs)
        return _Response()

    monkeypatch.setattr(test_api.requests, "post", post)
    test_api.main(
        [
            "--dataset",
            str(dataset),
            "--classifier-result",
            str(classifier_result),
            "--taskID",
            "task-01",
        ]
    )

    assert bson.loads(sent["data"])["taskID"] == "task-01"
    assert (dataset / "docker_mapping_response" / "global_skus.json").is_file()
    with pytest.raises(SystemExit):
        test_api.parse_args(
            ["--dataset", str(dataset), "--classifier-result", str(classifier_result)]
        )


def test_api_round_trips_success_bson_and_returns_tracebacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api,
        "process",
        lambda inputs: {
            "global_skus": ['{"skus":[]}'],
            "viewer_bundle": b"zip",
            "cos": {
                "taskID": inputs["taskID"],
                "viewer_bundle_url": "https://cos.example/task-01/viewer_bundle.zip",
            },
        },
    )
    response = _post_api(bson.dumps({"taskID": "task-01", "images": [], "skus": []}))
    assert response.status_code == 200
    assert bson.loads(response.content) == {
        "global_skus": ['{"skus":[]}'],
        "viewer_bundle": b"zip",
        "cos": {
            "taskID": "task-01",
            "viewer_bundle_url": "https://cos.example/task-01/viewer_bundle.zip",
        },
    }

    monkeypatch.setattr(
        api,
        "process",
        lambda _inputs: (_ for _ in ()).throw(RuntimeError("pipeline failed")),
    )
    response = _post_api(bson.dumps({"taskID": "task-01", "images": [], "skus": []}))
    assert response.status_code == 500
    assert "RuntimeError: pipeline failed" in response.text

    response = _post_api(b"not bson")
    assert response.status_code == 500
    assert "bson" in response.text.lower()

    response = _post_api(b"")
    assert response.status_code == 500
    assert "bson" in response.text.lower()


def test_api_serializes_concurrent_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum = 0
    guard = threading.Lock()

    def blocking_process(_inputs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return {"global_skus": [], "viewer_bundle": b"zip"}

    monkeypatch.setattr(api, "process", blocking_process)
    body = bson.dumps({"images": [], "skus": []})
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
