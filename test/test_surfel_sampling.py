"""Boundary coverage, fixed budgets and pre-thinned candidate semantics."""
import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from src.surfel_sampling import EdgeAdaptiveSampler


@pytest.mark.parametrize("budget", [0, 1, 97, 646])
def test_exact_deterministic_budget_on_irregular_frame_grid(budget):
    valid = np.ones((2, 17, 19), dtype=bool)
    depth = np.ones(valid.shape)
    sampler = EdgeAdaptiveSampler(depth, valid, 0.002)
    indices = np.arange(depth.size)
    positions, scales = sampler.select(indices, budget)
    again, again_scales = sampler.select(indices, budget)
    assert len(positions) == len(np.unique(positions)) == budget
    assert np.array_equal(positions, again)
    assert np.array_equal(scales, again_scales)
    assert scales.shape == (budget, 2)
    assert ((scales >= 0.5) & (scales <= 8)).all()


def test_depth_edge_gets_detail_while_smooth_surface_keeps_coverage():
    y, x = np.indices((96, 128))
    depth = np.where(x < 64, 2.0, 3.0)[None]
    sampler = EdgeAdaptiveSampler(depth, np.ones_like(depth, bool), 0.002)
    indices, scales = sampler.select(np.arange(depth.size), depth.size // 16)
    selected = np.zeros(depth.shape, dtype=bool)
    selected.ravel()[indices] = True
    edge = (abs(x - 63.5) < 2) & (y >= 8) & (y < 88)
    interior = (abs(x - 63.5) > 8) & (x >= 8) & (x < 120) & (y >= 8) & (y < 88)
    assert selected[0, edge].mean() > 1.8 * selected[0, interior].mean()
    assert distance_transform_edt(~selected[0])[interior].max() < 7
    _, sy, sx = np.unravel_index(indices, depth.shape)
    on_step = (abs(sx - 63.5) < 1) & (sy >= 8) & (sy < 88)
    assert on_step.any() and (scales[on_step, 0] == 0.5).all()
    slope = (2 + 0.02 * x)[None]
    smooth = EdgeAdaptiveSampler(slope, np.ones_like(slope, bool), 0.002)
    assert (smooth.weights.reshape(slope.shape)[0, 8:-8, 8:-8] == 1).all()


def test_voxel_candidates_do_not_create_false_depth_holes_or_lose_density():
    depth = np.ones((1, 64, 64))
    valid = np.ones_like(depth, dtype=bool)
    sampler = EdgeAdaptiveSampler(depth, valid, 0.002)
    y, x = np.indices(depth.shape[1:])
    candidates = np.flatnonzero((x % 4 == 0) & (y % 4 == 0))
    positions, scales = sampler.select(candidates, len(candidates), support_indices=np.arange(depth.size))
    assert np.array_equal(positions, np.arange(len(candidates)))
    interior = ((x.ravel()[candidates] >= 16) & (x.ravel()[candidates] < 48)
                & (y.ravel()[candidates] >= 16) & (y.ravel()[candidates] < 48))
    assert np.allclose(scales[interior], 4)
    assert (sampler.weights.reshape(depth.shape)[0, 8:-8, 8:-8] == 1).all()


def test_same_depth_product_boundary_limits_footprint():
    depth = np.ones((1, 32, 64))
    regions = np.zeros(depth.shape, dtype=np.int32)
    regions[:, :, 32:] = 1
    sampler = EdgeAdaptiveSampler(depth, np.ones_like(depth, bool), 0.002, regions)
    assert (sampler.caps.reshape(*depth.shape, 2)[0, 8:-8, 31:33, 0] == 0.5).all()
    indices = np.flatnonzero(regions.ravel() == 0)
    selected, _ = sampler.select(indices, 128)
    assert len(np.unique(selected)) == 128
    assert (regions.ravel()[indices[selected]] == 0).all()
