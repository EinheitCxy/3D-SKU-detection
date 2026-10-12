"""Small camera grids for the locked Viewer export quality rules."""
import numpy as np
import pytest

from src.viewer_export_tuning import ViewerExportTuning
from src.viewer_point_quality import (
    camera_centers, pixel_quality_drop, product_spill_keep, viewer_quality_keep,
)


def _grid(depth, confidence=None, u=None, v=None):
    depth = np.asarray(depth, dtype=np.float32)
    points = np.zeros((*depth.shape, 3), dtype=np.float32)
    points[..., 2] = depth
    tangent_u = np.zeros_like(points)
    tangent_v = np.zeros_like(points)
    tangent_u[..., 0] = 1
    tangent_v[..., 1] = 1
    return dict(depth=depth, valid=np.ones(depth.shape, dtype=bool),
                confidence=np.ones(depth.shape, dtype=np.float32) if confidence is None else confidence,
                u=tangent_u if u is None else u, v=tangent_v if v is None else v,
                points=points, centers=np.zeros((depth.shape[0], 3)), spacing=0.002)


def test_middle_of_depth_ramp_drops_but_flat_plane_remains():
    depth = np.ones((1, 5, 7), dtype=np.float32)
    depth[:, :, 3] = 1.2
    depth[:, :, 4:] = 1.4
    drop, counts = pixel_quality_drop(**_grid(depth))
    assert drop[0, 2, 3]
    assert not drop[0, 2, 1]
    assert not drop[0, 2, 5]
    assert counts["mid_slope"] >= 1


def test_low_confidence_product_label_is_fully_retained():
    drop = np.zeros((1, 2, 5), dtype=bool)
    drop.ravel()[[0, 1, 2, 3, 4, 5]] = True
    indices = np.arange(10)
    labels = np.array([0, 0, 0, 0, -1, -1, 1, 1, 1, 1])
    keep = viewer_quality_keep(indices=indices, labels=labels, drop=drop)
    assert keep[:4].all()
    assert not keep[4:6].any()
    assert keep[6:].all()


def test_85_degree_background_plane_is_grazing():
    depth = np.ones((1, 3, 3), dtype=np.float32)
    inputs = _grid(depth)
    angle = np.deg2rad(85)
    inputs["u"][..., 0] = np.cos(angle)
    inputs["u"][..., 2] = -np.sin(angle)
    drop, counts = pixel_quality_drop(**inputs)
    assert drop.all()
    assert counts["grazing"] == depth.size


@pytest.mark.parametrize("far_count,expected_drop", [(1, 1), (2, 0)])
def test_product_spill_group_retention(far_count, expected_drop):
    depth = np.ones((1, 1, 5), dtype=np.float32)
    depth.ravel()[-far_count:] = 1.2
    keep = product_spill_keep(indices=np.arange(5), labels=np.zeros(5, dtype=int), depth=depth)
    assert (~keep).sum() == expected_drop


def test_product_spill_groups_by_frame_and_ignores_background():
    depth = np.array([[[1, 1, 1, 1.2, 2]], [[2, 2, 2, 2.4, 3]]], dtype=np.float32)
    labels = np.array([0, 0, 0, 0, -1] * 2)
    keep = product_spill_keep(indices=np.arange(10), labels=labels, depth=depth)
    assert np.array_equal(np.flatnonzero(~keep), [3, 8])


def test_camera_centers_from_rotated_world_to_camera():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    center = np.array([2, 3, 4])
    extrinsic = np.eye(4)[None]
    extrinsic[0, :3, :3] = rotation
    extrinsic[0, :3, 3] = -rotation @ center
    assert np.allclose(camera_centers(extrinsic), center[None])
    assert np.allclose(camera_centers(extrinsic[:, :3]), center[None])


def test_tuning_rejects_invalid_weight_and_patch_values():
    with pytest.raises(ValueError):
        ViewerExportTuning(background_edge_weights=(2, 1, 0))
    with pytest.raises(ValueError):
        ViewerExportTuning(owner_patch_cells=17)
