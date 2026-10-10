import json

import numpy as np
import pytest

import utils.scene_orientation as orientation


def _cache(path, height=1.0):
    np.savez(
        path,
        world_points=np.array(
            [[[[0.0, height, 1.0], [1.0, height, 1.0], [0.0, height, 2.0]]]]
        ),
        world_points_conf=np.ones((1, 1, 3)),
        extrinsic=np.array([np.c_[np.eye(3), np.zeros(3)]]),
    )


def _result():
    return orientation.SceneOrientation(
        "fitted", np.diag([1.0, -1.0, -1.0]), np.array([0.0, -1.0, 0.0]), 1.0, {}
    )


def test_shared_orientation_cache_hit_rebuild_and_malformed(tmp_path, monkeypatch):
    path = tmp_path / "predictions.npz"
    _cache(path)
    calls = []
    monkeypatch.setattr(
        orientation,
        "fit_scene_orientation",
        lambda *args: calls.append(args) or _result(),
    )
    first, event = orientation.get_scene_orientation(path)
    assert event == "computed"
    second, event = orientation.get_scene_orientation(path)
    assert event == "hit"
    assert len(calls) == 1
    np.testing.assert_array_equal(first.normal_world, second.normal_world)
    _cache(path, 2.0)
    assert orientation.get_scene_orientation(path)[1] == "stale_recomputed"
    assert len(calls) == 2
    sidecar = tmp_path / "scene_orientation.json"
    data = json.loads(sidecar.read_text())
    data["normal_world"] = [1.0, 0.0, 0.0]
    sidecar.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="malformed"):
        orientation.get_scene_orientation(path)
    assert len(calls) == 2


def test_cache_rebuilt_during_fit_is_not_published(tmp_path, monkeypatch):
    path = tmp_path / "predictions.npz"
    _cache(path)

    def rebuild(*args):
        _cache(path, 2.0)
        return _result()

    monkeypatch.setattr(orientation, "fit_scene_orientation", rebuild)
    with pytest.raises(ValueError, match="changed while computing"):
        orientation.get_scene_orientation(path)
    assert not (tmp_path / "scene_orientation.json").exists()


def test_not_found_cannot_claim_ground_plane(tmp_path, monkeypatch):
    path = tmp_path / "predictions.npz"
    _cache(path)
    result = orientation.SceneOrientation(
        "not_found", np.diag([1.0, -1.0, -1.0]), None, None, {}
    )
    monkeypatch.setattr(orientation, "fit_scene_orientation", lambda *args: result)
    assert orientation.get_scene_orientation(path)[0].normal_world is None
    assert orientation.get_scene_orientation(path)[1] == "hit"
    data = json.loads((tmp_path / "scene_orientation.json").read_text())
    data["plane_offset_m"] = 1.0
    (tmp_path / "scene_orientation.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="cannot contain a plane"):
        orientation.get_scene_orientation(path)


def test_fitted_tilted_plane_keeps_nonzero_offset():
    x, z = np.meshgrid(np.linspace(0, 2, 21), np.linspace(0, 2, 21))
    points = np.c_[x.ravel(), (0.15 * x + 0.1 * z + 0.7).ravel(), z.ravel()]
    extrinsic = np.array([np.c_[np.eye(3), [0.0, 2.0, 0.0]]])
    result = orientation.fit_scene_orientation(points, extrinsic)
    assert result.status == "fitted"
    assert abs(result.plane_offset_m) > 0.5
    np.testing.assert_allclose(
        points @ result.normal_world + result.plane_offset_m, 0, atol=1e-6
    )
    np.testing.assert_allclose(
        result.rotation @ result.normal_world, [0, 1, 0], atol=1e-6
    )
