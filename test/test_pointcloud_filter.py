"""Unit tests for utils.pointcloud_filter (synthetic scenes, no GPU/data needed)."""

import numpy as np

from utils.pointcloud_filter import (
    PointCloudFilterConfig,
    apply_mask,
    filter_scene_points,
)


def _make_scene(seed=0, n_cluster=8000, n_ground=8000, n_outlier=300):
    rng = np.random.default_rng(seed)
    cluster = rng.normal(loc=[0.0, 1.2, 0.0], scale=0.3, size=(n_cluster, 3))
    cluster = cluster[cluster[:, 1] > 0.05]  # subject sits above the floor
    n_cluster = len(cluster)
    ground = np.column_stack(
        [
            rng.uniform(-3, 3, n_ground),
            np.zeros(n_ground),
            rng.uniform(-3, 3, n_ground),
        ]
    )
    outliers = rng.uniform(-30, 30, size=(n_outlier, 3))
    return np.vstack([cluster, ground, outliers]), n_cluster, n_ground


def test_removes_sparse_outliers_and_ground():
    points, n_cluster, n_ground = _make_scene()
    mask = filter_scene_points(points)
    kept_idx = np.flatnonzero(mask)

    kept_cluster = np.sum(kept_idx < n_cluster)
    kept_ground = np.sum((kept_idx >= n_cluster) & (kept_idx < n_cluster + n_ground))
    kept_outlier = np.sum(kept_idx >= n_cluster + n_ground)

    assert kept_cluster > 0.8 * n_cluster, "main cluster must survive"
    assert kept_ground < 0.1 * n_ground, "ground plane should be removed"
    assert kept_outlier < 0.05 * 300, "sparse outliers should be removed"


def test_keeps_ground_when_disabled():
    points, n_cluster, n_ground = _make_scene()
    cfg = PointCloudFilterConfig(remove_ground=False)
    mask = filter_scene_points(points, cfg)
    kept_idx = np.flatnonzero(mask)
    kept_ground = np.sum((kept_idx >= n_cluster) & (kept_idx < n_cluster + n_ground))
    assert kept_ground > 0.5 * n_ground


def test_base_mask_drops_nonfinite_and_zero_rows():
    points = np.array(
        [[np.nan, 0.0, 0.0], [np.inf, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
    )
    cfg = PointCloudFilterConfig(min_points=10)  # skip geometric stages
    mask = filter_scene_points(points, cfg)
    assert mask.tolist() == [False, False, False, True]


def test_protected_mask_points_bypass_geometry_while_background_outliers_drop():
    """保护点跳过 SOR/DBSCAN/ground，未标注远端点仍按原过滤规则删除。"""
    rng = np.random.default_rng(23)
    subject = rng.normal(loc=[0.0, 1.0, 0.0], scale=0.05, size=(400, 3))
    protected = np.asarray([[10.0, 10.0, 10.0]])
    nearby_noise = np.asarray([[10.01, 10.0, 10.0]])
    remote_noise = np.asarray([[30.0, 30.0, 30.0]])
    points = np.vstack([subject, protected, nearby_noise, remote_noise])
    protect_mask = np.zeros(len(points), dtype=bool)
    protect_mask[len(subject)] = True

    keep = filter_scene_points(
        points,
        PointCloudFilterConfig(min_points=100, remove_ground=False),
        protect_mask=protect_mask,
    )

    assert keep[len(subject)]
    assert not keep[len(subject) + 1]
    assert not keep[len(subject) + 2]


def test_protected_majority_still_filters_small_unprotected_outlier_set():
    """保护点参与模型估计，不能令少量 mask 外离群点绕过 SOR。"""
    rng = np.random.default_rng(47)
    protected = rng.normal(loc=[0.0, 1.0, 0.0], scale=0.03, size=(100, 3))
    background_outliers = np.asarray([[20.0, 20.0, 20.0], [40.0, 40.0, 40.0]])
    points = np.vstack([protected, background_outliers])
    protect_mask = np.zeros(len(points), dtype=bool)
    protect_mask[: len(protected)] = True

    keep = filter_scene_points(
        points,
        PointCloudFilterConfig(min_points=10, remove_ground=False),
        protect_mask=protect_mask,
    )

    assert keep[: len(protected)].all()
    assert not keep[len(protected) :].any()


def test_protect_mask_is_not_mutated_when_invalid_points_are_unprotectable():
    points = np.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    protect_mask = np.asarray([True, True])

    filter_scene_points(points, PointCloudFilterConfig(min_points=10), protect_mask)

    assert protect_mask.tolist() == [True, True]


def test_small_scene_returned_unfiltered():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(500, 3))
    mask = filter_scene_points(points)
    assert mask.all()


def test_empty_input():
    mask = filter_scene_points(np.empty((0, 3)))
    assert mask.size == 0


def test_apply_mask_aligns_aux_arrays():
    points = np.arange(12, dtype=float).reshape(4, 3)
    colors = np.arange(4)
    mask = np.array([True, False, True, False])
    out_points, out_colors, out_none = apply_mask(mask, points, colors, None)
    assert out_points.shape == (2, 3)
    assert out_colors.tolist() == [0, 2]
    assert out_none is None
