"""Geometry correction must generalize and never publish a rejected candidate."""

from pathlib import Path

import numpy as np
import pytest

from utils.geometry_refinement import _bilinear_valid, refine_geometry


def synthetic_scene():
    rng = np.random.default_rng(7)
    k = np.tile([[90.0, 0, 63], [0, 90, 47], [0, 0, 1]], (3, 1, 1))
    e = np.tile(np.c_[np.eye(3), np.zeros(3)], (3, 1, 1))
    e[:, 0, 3] = [0, -0.1, 0.1]
    fi, fj, xi, xj = [], [], [], []
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        xy = rng.uniform([20, 20], [105, 75], (60, 2))
        other = xy.copy()
        other[:, 0] += 90 * (e[b, 0, 3] - e[a, 0, 3]) / 3
        fi.extend([a] * len(xy))
        fj.extend([b] * len(xy))
        xi.extend(xy)
        xj.extend(other)
    observed = e.copy()
    observed[1, :2, 3] += [0.03, 0.01]
    observed[2, :2, 3] += [-0.02, 0.005]
    matches = {
        "frame_i": np.array(fi),
        "frame_j": np.array(fj),
        "xy_i": np.array(xi),
        "xy_j": np.array(xj),
    }
    return np.full((3, 96, 128), 3.0), k, observed, matches


@pytest.mark.parametrize("mode", ["pose", "pose-scale"])
def test_independent_heldout_geometry_improves(mode):
    depth, k, e, matches = synthetic_scene()
    scales = np.array([1, 0.94, 1.06]) if mode == "pose-scale" else np.ones(3)
    depth /= scales[:, None, None]
    depth[:, 0, 0] = 0  # DA3's invalid pixel is not a reason to reject the scene.
    corrected, poses, report = refine_geometry(depth, k, e, matches, mode=mode)
    heldout = report["metrics"]["heldout"]
    assert report["accepted"]
    assert heldout["after"]["pixel_rmse"] < heldout["before"]["pixel_rmse"] * 0.1
    np.testing.assert_array_equal(poses[0], e[0])
    np.testing.assert_array_equal(corrected[:, 0, 0], 0)
    np.testing.assert_allclose(report["correction"]["depth_scale"], scales, atol=0.015)


def test_depth_edge_is_not_interpolated_into_a_fake_surface():
    depth = np.array([[1.0, 3.0], [1.0, 3.0]])
    _, valid, rejected_edge = _bilinear_valid(depth, np.array([[0.5, 0.5]]))
    assert not valid[0]
    assert rejected_edge[0]


def test_disconnected_correspondences_fail():
    depth, k, e, matches = synthetic_scene()
    only_first_pair = matches["frame_j"] == 1
    matches = {key: value[only_first_pair] for key, value in matches.items()}
    with pytest.raises(ValueError, match="connect"):
        refine_geometry(depth, k, e, matches)


def test_relative_improvement_cannot_publish_large_absolute_error():
    depth, k, e, matches = synthetic_scene()
    matches["xy_j"][:, 0] += 30
    _, _, report = refine_geometry(depth, k, e, matches)
    assert not report["accepted"]
    assert "heldout_pixel_absolute" in report["rejection_reasons"]


def test_frame_indices_are_not_silently_truncated():
    depth, k, e, matches = synthetic_scene()
    matches["frame_i"] = matches["frame_i"].astype(float) + 0.25
    with pytest.raises(ValueError, match="integer"):
        refine_geometry(depth, k, e, matches)


@pytest.mark.parametrize("accepted,reusable", [(False, False), (True, False), (True, True)])
def test_pipeline_does_not_replace_rejected_refinement(tmp_path, accepted, reusable):
    import json
    from main import _check_geometry_refinement_report

    (tmp_path / "geometry_refinement.json").write_text(json.dumps({
        "accepted": accepted, "cache_published": accepted,
    }))
    if accepted and reusable:
        _check_geometry_refinement_report(tmp_path, reusable)
    else:
        with pytest.raises(ValueError, match="拒绝重建覆盖"):
            _check_geometry_refinement_report(tmp_path, reusable)


def test_rejected_candidate_is_not_published(tmp_path, monkeypatch):
    import src.da3_geometry_refinement as cli

    cache = tmp_path / "source" / "dataset" / "da3_cache" / "predictions.npz"
    cache.parent.mkdir(parents=True)
    depth, k, e, matches = synthetic_scene()
    np.savez(
        cache,
        depth=depth[..., None],
        intrinsic=k,
        extrinsic=e,
        images=np.zeros((3, 96, 128, 3), dtype=np.uint8),
        image_ids=np.array([9, 4, 7], dtype=np.int32),
        source_image_sizes=np.tile([504, 378], (3, 1)),
        preprocess_resolution=cli.DEFAULT_PROCESS_RES,
        preprocess_method=cli.PREPROCESS_METHOD,
    )
    monkeypatch.setattr(cli, "_validate_da3_runner_cache", lambda _: None)
    monkeypatch.setattr(cli, "extract_correspondences", lambda *a, **kw: (matches, {}))
    monkeypatch.setattr(
        cli, "refine_geometry", lambda *a, **kw: (depth, e, {"accepted": False})
    )
    output = tmp_path / "rejected"
    report = cli.refine_cache(cache, output)
    assert not report["cache_published"]
    assert (output / "dataset" / "geometry_refinement.json").is_file()
    assert not list(output.rglob("predictions.npz"))
    with pytest.raises(FileExistsError):
        cli.refine_cache(cache, output)
