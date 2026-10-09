"""The GLB is returned as BSON binary; only the Viewer ZIP is uploaded."""
import asyncio

import bson
import httpx

import pytest

import api
import processor


@pytest.mark.parametrize("failure", [None, "bake", "upload"])
def test_process_reuses_generation_and_propagates_errors(tmp_path, monkeypatch, failure):
    events = []
    roots = []
    config = object()
    monkeypatch.setenv("DA3_MODEL_PATH", "/model")
    monkeypatch.setattr(processor.torch.cuda, "is_available", lambda: False)

    def prepare(inputs, root):
        roots.append(root)
        return processor.PreparedRequest(root / "dataset", "task-01")

    def mapping(dataset, output, viewer, model):
        assert model == "/model"
        events.append("pipeline")
        return {"global_skus_path": output / "global_skus.json",
                "viewer_dir": viewer / "runs" / "generation"}

    def bake(source, destination):
        assert source == roots[0] / "viewer/runs/generation"
        assert destination == roots[0] / "scene.glb"
        events.append("bake")
        if failure == "bake":
            raise RuntimeError("bake failed")
        destination.write_bytes(b"glTF")

    def upload_zip(taskid, data, actual_config):
        assert (taskid, data, actual_config) == ("task-01", b"PK", config)
        assert (roots[0] / "scene.glb").is_file()
        events.append("zip")
        if failure == "upload":
            raise RuntimeError("upload failed")

    monkeypatch.setattr(processor, "prepare_request", prepare)
    monkeypatch.setattr(processor, "run_mapping_request", mapping)
    monkeypatch.setattr(processor, "build_success_response", lambda *args: {
        "global_skus": ["frame"], "viewer_bundle": b"PK"})
    monkeypatch.setattr(processor, "export_scene_glb", bake)
    monkeypatch.setattr(processor.CosUploadConfig, "from_env", lambda: config)
    monkeypatch.setattr(processor, "upload_viewer_bundle", upload_zip)
    if failure:
        with pytest.raises(RuntimeError, match=failure + " failed"):
            processor.process({})
    else:
        result = processor.process({})
        assert bson.loads(bson.dumps(result)) == {"global_skus": ["frame"], "scene_glb": b"glTF"}
    assert events == (["pipeline", "bake"] if failure == "bake" else
                      ["pipeline", "bake", "zip"])
    assert not roots[0].exists()


def test_api_returns_scene_as_bson_binary(monkeypatch):
    scene = b"glTF\x00\xff\x80binary"
    expected = {"global_skus": ["frame"], "scene_glb": scene}
    monkeypatch.setattr(api, "process", lambda inputs: expected)

    async def post():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),
                                     base_url="http://test") as client:
            return await client.post("/api", content=bson.dumps({}),
                                     headers={"content-type": "application/bson"})

    response = asyncio.run(post())
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/bson"
    result = bson.loads(response.content)
    assert result == expected
    assert isinstance(result["scene_glb"], bytes)
