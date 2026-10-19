"""Shared NN/SOR statistics boundaries against native Open3D."""

import numpy as np
import pytest

from utils.pointcloud_neighbors import prepare_neighbor_statistics


@pytest.mark.parametrize("k", [1, 2, 20])
def test_shared_knn_matches_open3d_with_duplicates_and_short_domain(k):
    o3d = pytest.importorskip("open3d")
    rng = np.random.default_rng(34)
    points = np.vstack([rng.normal(size=(9, 3)), [[1., 2., 3.], [1., 2., 3.]]])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    nearest = np.asarray(cloud.compute_nearest_neighbor_distance())
    _, indices = cloud.remove_statistical_outlier(k, 2.)
    expected = np.zeros(len(points), dtype=bool)
    expected[np.asarray(indices, dtype=np.int64)] = True
    result = prepare_neighbor_statistics(points, k=k, chunk_points=3, workers=1)
    np.testing.assert_allclose(result.nearest_distances, nearest, rtol=1e-15, atol=0)
    np.testing.assert_array_equal(result.sor_mask, expected)
    assert result.median_nn == pytest.approx(float(np.median(nearest)), rel=1e-15)


def test_shared_knn_degenerate_and_empty_domains():
    result = prepare_neighbor_statistics(np.ones((5, 3)), k=20, workers=1)
    assert not result.sor_mask.any()
    assert result.median_nn == 0
    assert not result.nearest_distances.flags.writeable
    empty = prepare_neighbor_statistics(np.empty((0, 3)), workers=1)
    assert empty.median_nn == 0
    assert empty.sor_mask.shape == (0,)
    single = prepare_neighbor_statistics(np.ones((1, 3)), k=1, workers=1)
    assert single.nearest_distances.tolist() == [0.]
    assert single.sor_mask.tolist() == [False]
