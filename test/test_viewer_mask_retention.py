"""商品 mask 的全部有效源点必须穿过 Viewer 的过滤与采样。"""
from pathlib import Path
import numpy as np
import pytest
from src import web_viewer_export as exporter
from utils.pointcloud_filter import PointCloudFilterConfig


def sample(monkeypatch, points, labels, *, background_limit=None, filter_config=None):
    if background_limit is not None:
        monkeypatch.setattr(exporter, "MAX_BACKGROUND_POINTS", background_limit)
    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    n = len(points)
    cache = {
        'points': points.reshape(1, 1, n, 3),
        'confidence': np.arange(1, n + 1, dtype=np.float32).reshape(1, 1, n),
        'images': np.tile(np.arange(n, dtype=np.uint8)[:, None], (1, 3)).reshape(1, 1, n, 3),
        'extrinsic': np.eye(4)[None],
    }
    # Isolate mask-file I/O; filtering, voxel selection and slot sorting are real.
    monkeypatch.setattr(exporter, '_instance_labels_v2', lambda cache, objects, indices, *args: (labels[indices], [('1', 0), ('2', 0)]))
    monkeypatch.setattr(exporter, '_fit_level_rotation', lambda *args: (np.eye(3), False))
    return exporter._sample_points(cache, {}, mask_cache_root=Path('/unused'), voxel_size=.005,
        filter_config=filter_config or PointCloudFilterConfig(enabled=False))


def test_mask_points_survive_voxel_collision_and_exceed_budget(monkeypatch):
    result = sample(monkeypatch, [[0, 0, 1], [.001, 0, 1], [.002, 0, 1], [2, 0, 1]],
        [0, 1, -1, -1], background_limit=1)
    assert {0, 1}.issubset(result['source_indices'])
    assert len(result['source_indices']) == 3
    assert set(result['instance_labels']) == {-1, 0, 1}
    assert np.array_equal(result['colors'][:, 0], result['source_indices'])


def test_all_mask_pixels_survive_even_for_same_object_same_voxel(monkeypatch):
    result = sample(monkeypatch, [[0, 0, 1], [.001, 0, 1], [2, 0, 1], [3, 0, 1]],
        [0, 0, -1, -1], background_limit=3)
    assert {0, 1}.issubset(result['source_indices'])
    assert len(result['source_indices']) == 4


def test_protected_small_cluster_survives_but_invalid_and_background_drop(monkeypatch):
    rng = np.random.default_rng(3)
    points = np.concatenate([rng.normal(0, .03, (2000, 3)) + [0, 0, 2],
        rng.normal(0, .001, (8, 3)) + [1, 0, 2],
        rng.normal(0, .001, (8, 3)) + [-1, 0, 2], [[np.nan, 0, 2]]])
    labels = np.full(len(points), -1); labels[2000:2008] = 0; labels[-1] = 0
    result = sample(monkeypatch, points, labels, background_limit=10000,
        filter_config=PointCloudFilterConfig(remove_ground=False))
    assert set(range(2000, 2008)).issubset(result['source_indices'])
    assert not np.isin(result['source_indices'], np.arange(2008, len(points))).any()


def test_background_is_capped_at_500000_without_spending_product_points(monkeypatch):
    points = np.zeros((500_005, 3), dtype=np.float32)
    points[:, 0] = np.arange(len(points), dtype=np.float32) * .01
    points[:, 2] = 1
    labels = np.full(len(points), -1, dtype=np.int32)
    labels[:2] = 0
    result = sample(monkeypatch, points, labels)
    assert np.count_nonzero(result['instance_labels'] < 0) == 500_000
    assert np.count_nonzero(result['instance_labels'] >= 0) == 2


def test_product_budget_is_shared_by_global_id_across_frames(monkeypatch):
    monkeypatch.setattr(exporter, 'MAX_PRODUCT_POINTS', 6, raising=False)
    labels = np.array([0] * 5 + [1] * 5 + [2] * 10)
    indices = np.arange(len(labels))
    keys = [('1', 0), ('1', 1), ('2', 0)]
    selected = exporter._sample_product_points(indices, labels, keys)
    gids = np.array([int(keys[label][0]) for label in labels[selected]])
    assert len(selected) == len(set(selected)) == 6
    assert np.count_nonzero(gids == 1) == np.count_nonzero(gids == 2) == 3


def test_small_products_keep_all_points_and_redistribute_unused_quota(monkeypatch):
    monkeypatch.setattr(exporter, 'MAX_PRODUCT_POINTS', 9, raising=False)
    labels = np.array([0] + [1] * 10 + [2] * 10)
    selected = exporter._sample_product_points(np.arange(len(labels)), labels, [('1', 0), ('2', 0), ('3', 0)])
    assert len(selected) == 9
    assert np.bincount(labels[selected]).tolist() == [1, 4, 4]
    assert np.array_equal(selected, exporter._sample_product_points(np.arange(len(labels)), labels, [('1', 0), ('2', 0), ('3', 0)]))


def test_product_cap_keeps_source_slots_and_background_budget_independent(monkeypatch):
    monkeypatch.setattr(exporter, 'MAX_PRODUCT_POINTS', 4, raising=False)
    result = sample(monkeypatch, [[i*.01,0,1] for i in range(14)], [0]*6+[1]*6+[-1]*2, background_limit=2)
    assert np.count_nonzero(result['instance_labels'] == 0) == 2
    assert np.count_nonzero(result['instance_labels'] == 1) == 2
    assert np.count_nonzero(result['instance_labels'] < 0) == 2
    assert np.array_equal(result['colors'][:,0],result['source_indices'])
