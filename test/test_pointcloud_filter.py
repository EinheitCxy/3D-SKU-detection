"""Unit tests for utils.pointcloud_filter (synthetic scenes, no GPU/data needed)."""

import numpy as np
import pytest

from utils.pointcloud_filter import (
    PointCloudFilterConfig,
    apply_mask,
    filter_scene_points,
    prepare_scene_filter,
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


def test_invalid_protect_mask_rejected_before_preparation(monkeypatch):
    import utils.pointcloud_filter as filtering

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid protect_mask must not prepare NN/SOR")

    monkeypatch.setattr(filtering, "prepare_scene_filter", unexpected)
    with pytest.raises(ValueError, match="protect_mask"):
        filtering.filter_scene_points(
            np.ones((1000, 3)), protect_mask=np.ones((1000, 1), dtype=bool),
        )


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


def _fake_open3d(monkeypatch, *, raw_sor, median=1.0, cluster=None,
                 ground_plane=(0.0, 1.0, 0.0, 0.0), ground_inliers=None):
    """Instrument label-independent calls and selections, without random RANSAC."""
    import sys
    from types import SimpleNamespace

    calls = {"nn": 0, "sor": 0, "clusters": [], "planes": []}

    class Cloud:
        def __init__(self):
            self.points = np.empty((0, 3))

        def compute_nearest_neighbor_distance(self):
            calls["nn"] += 1
            return np.full(len(self.points), median)

        def remove_statistical_outlier(self, neighbors, std):
            calls["sor"] += 1
            return None, raw_sor

        def select_by_index(self, indices):
            result = Cloud()
            result.points = self.points[indices]
            return result

        def segment_plane(self, threshold, *, ransac_n, num_iterations):
            calls["planes"].append((self.points.copy(), threshold, ransac_n, num_iterations))
            indices = (np.flatnonzero(np.abs(self.points @ np.asarray(ground_plane[:3]) + ground_plane[3]) < threshold)
                       if ground_inliers is None else ground_inliers)
            return ground_plane, indices

        def cluster_dbscan(self, eps, min_points):
            calls["clusters"].append((self.points.copy(), eps, min_points))
            return np.zeros(len(self.points), dtype=int) if cluster is None else cluster(self.points)

    monkeypatch.setitem(sys.modules, "open3d", SimpleNamespace(
        geometry=SimpleNamespace(PointCloud=Cloud),
        utility=SimpleNamespace(Vector3dVector=np.asarray),
    ))
    return calls


@pytest.mark.parametrize("protect, expected", [
    ([False] * 5, [True] * 5),  # 0/5 raw mask: guard skips SOR.
    ([False, False, True, False, False], [False, False, True, False, False]),
])
def test_prepared_raw_sor_applies_protection_before_guard(monkeypatch, protect, expected):
    calls = _fake_open3d(monkeypatch, raw_sor=[])
    points = np.arange(15, dtype=float).reshape(-1, 3) + 1
    config = PointCloudFilterConfig(
        min_points=1, sor_nb_neighbors=1, keep_main_clusters=False, remove_ground=False,
    )
    prepared = prepare_scene_filter(points, config)
    assert not prepared.sor_mask.any()
    assert calls["nn"] == calls["sor"] == 1
    assert filter_scene_points(points, config, protect, prepared=prepared).tolist() == expected
    assert calls["nn"] == calls["sor"] == 1
    assert filter_scene_points(points, config, protect).tolist() == expected


@pytest.mark.parametrize("config, median, expected_calls", [
    (PointCloudFilterConfig(enabled=False), 1, (0, 0)),
    (PointCloudFilterConfig(min_points=10), 1, (0, 0)),
    (PointCloudFilterConfig(min_points=1), 0, (1, 0)),
    (PointCloudFilterConfig(min_points=1, remove_ground=False), 1, (1, 0)),
])
def test_prepared_skip_branches_preserve_base(monkeypatch, config, median, expected_calls):
    calls = _fake_open3d(monkeypatch, raw_sor=[], median=median)
    points = np.asarray([[1, 2, 3], [0, 0, 0], [np.nan, 1, 1]], dtype=float)
    prepared = prepare_scene_filter(points, config)
    keep = filter_scene_points(points, config, prepared=prepared)
    assert keep.tolist() == [True, False, False]
    assert keep.flags.writeable
    assert (calls["nn"], calls["sor"]) == expected_calls


def test_prepared_rejects_reordered_points_and_changed_config():
    points = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=float)
    config = PointCloudFilterConfig(enabled=False)
    prepared = prepare_scene_filter(points, config)
    with pytest.raises(ValueError, match="domain"):
        filter_scene_points(points[::-1], config, prepared=prepared)
    config.sor_std_ratio = 3.0
    with pytest.raises(ValueError, match="config"):
        filter_scene_points(points, config, prepared=prepared)


def test_prepared_raw_sor_matches_original_float64_base_order(monkeypatch):
    calls = _fake_open3d(monkeypatch, raw_sor=[0, 2])
    points = np.asarray([[1, 0, 0], [0, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    config = PointCloudFilterConfig(
        min_points=1, sor_nb_neighbors=1, keep_main_clusters=False, remove_ground=False,
    )
    prepared = prepare_scene_filter(points, config)
    assert prepared.points.dtype == np.float64
    assert np.asarray(prepared.pcd.points).tolist() == [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
    assert prepared.sor_mask.tolist() == [True, False, True]
    assert filter_scene_points(points, config, prepared=prepared).tolist() == [True, False, False, True]
    assert calls["sor"] == 1


@pytest.mark.parametrize("shared_neighbors", [False, True])
def test_prepared_raw_sor_matches_open3d_base_cloud(shared_neighbors):
    import open3d as o3d

    points = np.random.default_rng(901).normal(size=(128, 3)).astype(np.float32)
    points = np.vstack([points, [0, 0, 0], [np.nan, 0, 0]])
    config = PointCloudFilterConfig(min_points=10, remove_ground=False)
    prepared = prepare_scene_filter(points, config, shared_neighbors=shared_neighbors)
    cloud = o3d.geometry.PointCloud()
    base = np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
    cloud.points = o3d.utility.Vector3dVector(points[base].astype(np.float64))
    _, indices = cloud.remove_statistical_outlier(config.sor_nb_neighbors, config.sor_std_ratio)
    expected = np.zeros(int(base.sum()), dtype=bool)
    expected[np.asarray(indices, dtype=np.int64)] = True
    assert np.array_equal(prepared.base_mask, base)
    assert np.array_equal(prepared.sor_mask, expected)
    assert prepared.median_nn == np.median(cloud.compute_nearest_neighbor_distance())
    assert not prepared.points.flags.writeable
    assert not prepared.base_mask.flags.writeable
    assert not prepared.sor_mask.flags.writeable
    assert np.array_equal(np.asarray(prepared.pcd.points), points[base])


@pytest.mark.parametrize("points, config, has_cloud", [
    (np.asarray([[1., 1., 1.], [2., 1., 1.]]),
     PointCloudFilterConfig(enabled=False), False),
    (np.asarray([[1., 1., 1.], [2., 1., 1.]]),
     PointCloudFilterConfig(min_points=3), False),
    (np.ones((4, 3)), PointCloudFilterConfig(min_points=1), True),
    (np.asarray([[1., 1., 1.], [2., 1., 1.], [4., 1., 1.]]),
     PointCloudFilterConfig(min_points=1), True),
])
def test_shared_prepared_preserves_original_sor_skip_contract(points, config, has_cloud):
    original = prepare_scene_filter(points, config)
    shared = prepare_scene_filter(points, config, shared_neighbors=True)
    assert (original.pcd is not None) == (shared.pcd is not None) == has_cloud
    assert original.sor_mask is shared.sor_mask is None
    assert original.median_nn == shared.median_nn
    np.testing.assert_array_equal(original.base_mask, shared.base_mask)
    np.testing.assert_array_equal(
        filter_scene_points(points, config, prepared=original),
        filter_scene_points(points, config, prepared=shared),
    )


def test_proxy_cluster_ratio_counts_original_members(monkeypatch):
    # Two representatives: first has 9 members, second has 1. At a 20% size
    # threshold only the 9-member cluster survives, even though proxy sizes tie.
    points = np.asarray([[1 + i * .001, 1, 1] for i in range(9)] + [[10, 1, 1]])
    calls = _fake_open3d(
        monkeypatch, raw_sor=list(range(10)), cluster=lambda p: np.arange(len(p)),
    )
    config = PointCloudFilterConfig(
        min_points=1, sor_nb_neighbors=1, cluster_min_points=2,
        cluster_proxy_voxel_scale=1, min_cluster_ratio=.2, remove_ground=False,
    )
    keep = filter_scene_points(points, config)
    assert keep.tolist() == [True] * 9 + [False]
    proxy, eps, min_points = calls["clusters"][0]
    assert proxy.tolist() == points[[0, 9]].tolist()
    assert eps == 5.0
    assert min_points == 2  # counts proxy neighbors; deliberately not member-weighted.


@pytest.mark.parametrize("scale", [-1, np.nan, np.inf])
def test_proxy_scale_rejects_invalid_values(scale):
    with pytest.raises(ValueError, match="cluster_proxy_voxel_scale"):
        PointCloudFilterConfig(cluster_proxy_voxel_scale=scale)


@pytest.mark.parametrize("cap", [-1, 1, 2, True, 3.5, None])
def test_ground_proposal_cap_rejects_invalid_values(cap):
    with pytest.raises(ValueError, match="ground_proposal_max_points"):
        PointCloudFilterConfig(ground_proposal_max_points=cap)


def _ground_scene(heights):
    return np.column_stack((np.arange(1, len(heights) + 1), heights, np.ones(len(heights))))


def _ground_config(**kwargs):
    return PointCloudFilterConfig(min_points=3, sor_nb_neighbors=1,
                                 keep_main_clusters=False, **kwargs)


def test_ground_proposal_uses_full_support_and_protects_products(monkeypatch, caplog):
    points = _ground_scene([0] * 8 + [.5] * 2)
    # The proposal's returned inlier count is deliberately zero; only full-domain
    # re-evaluation may decide whether this normalized plane passes the 8% rule.
    calls = _fake_open3d(monkeypatch, raw_sor=list(range(10)), median=.001,
                         ground_plane=(0., 2., 0., 0.), ground_inliers=[])
    protected = np.zeros(10, dtype=bool)
    protected[0] = True
    with caplog.at_level("INFO"):
        keep = filter_scene_points(points, _ground_config(ground_proposal_max_points=3), protected)
    assert keep.tolist() == [True] + [False] * 7 + [True, True]
    sampled, threshold, ransac_n, iterations = calls["planes"][0]
    expected = np.sort(np.random.default_rng(42).choice(10, size=3, replace=False))
    np.testing.assert_array_equal(sampled, points[expected])
    assert (threshold, ransac_n, iterations) == (.02, 3, 1000)
    assert len(calls["planes"]) == 1  # no full-cloud retry
    assert "'proposal_points': 3" in caplog.text and "'reason': 'applied'" in caplog.text


def test_ground_proposal_does_not_change_global_numpy_rng(monkeypatch):
    points = _ground_scene([0] * 8 + [.5] * 2)
    calls = _fake_open3d(monkeypatch, raw_sor=list(range(10)), median=.001)
    before = np.random.get_state()
    config = _ground_config(ground_proposal_max_points=3)
    filter_scene_points(points, config)
    filter_scene_points(points, config)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(calls["planes"][0][0], calls["planes"][1][0])


def test_ground_proposal_samples_post_sor_cluster_domain(monkeypatch):
    points = _ground_scene([.5] * 12)
    calls = _fake_open3d(monkeypatch, raw_sor=list(range(10)), median=.001,
                         cluster=lambda p: np.asarray([0] * 8 + [1] * 2))
    config = PointCloudFilterConfig(min_points=3, sor_nb_neighbors=1, cluster_min_points=2,
                                   min_cluster_ratio=.3, ground_proposal_max_points=3)
    keep = filter_scene_points(points, config)
    expected = np.sort(np.random.default_rng(42).choice(8, size=3, replace=False))
    np.testing.assert_array_equal(calls["planes"][0][0], points[expected])
    assert keep.tolist() == [True] * 8 + [False] * 4


def test_ground_refined_plane_recount_uses_strict_threshold(monkeypatch):
    points = _ground_scene([0, .02, .5, .5])
    _fake_open3d(monkeypatch, raw_sor=list(range(4)), median=.001,
                 ground_plane=(0., 2., 0., 0.), ground_inliers=[])
    keep = filter_scene_points(points, _ground_config(ground_proposal_max_points=4))
    assert keep.tolist() == [False, True, True, True]


def test_ground_control_keeps_native_pre_refinement_inliers(monkeypatch):
    points = _ground_scene([0, 0, .5, .5])
    calls = _fake_open3d(monkeypatch, raw_sor=list(range(4)), median=.001, ground_inliers=[0])
    keep = filter_scene_points(points, _ground_config(ground_proposal_max_points=0))
    assert keep.tolist() == [False, True, True, True]
    np.testing.assert_array_equal(calls["planes"][0][0], points)


@pytest.mark.parametrize("heights, plane, reason", [
    ([0] + [.5] * 19, (0., 1., 0., 0.), "insufficient_full_support"),
    ([0] * 3 + [.5] * 2 + [-.5] * 2, (0., 1., 0., 0.), "plane_slices_subject"),
    ([0] * 9 + [.5], (0., 1., 0., 0.), "min_remaining_guard"),
    ([0] * 8 + [.5] * 2, (0., 0., 0., 0.), "degenerate_plane"),
    ([0] * 8 + [.5] * 2, (0., np.nan, 0., 0.), "degenerate_plane"),
])
def test_ground_candidate_rejections_keep_current_cloud(monkeypatch, caplog, heights, plane, reason):
    points = _ground_scene(heights)
    calls = _fake_open3d(monkeypatch, raw_sor=list(range(len(points))), median=.001,
                         ground_plane=plane, ground_inliers=[])
    with caplog.at_level("INFO"):
        keep = filter_scene_points(points, _ground_config(ground_proposal_max_points=3))
    assert keep.all()
    assert len(calls["planes"]) == 1
    assert "'reason': '" + reason + "'" in caplog.text


def test_ground_proposal_default_stays_full_control():
    assert PointCloudFilterConfig().ground_proposal_max_points == 0
    assert PointCloudFilterConfig(ground_proposal_max_points=50_000).ground_proposal_max_points == 50_000
