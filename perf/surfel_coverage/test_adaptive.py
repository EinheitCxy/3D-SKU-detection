"""Behavioral synthetic checks for the fixed-budget adaptive candidate."""

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from perf.surfel_coverage.adaptive import select


def test_irregular_observations_keep_exact_valid_unique_budget_and_mapping():
    yy, xx = np.indices((47, 61))
    valid = np.stack((
        (xx > 2) & ~((xx > 21) & (xx < 35) & (yy > 9) & (yy < 33)),
        (yy < 42) & ((xx + 2 * yy) % 11 != 0),
    ))
    depth = np.where(valid, np.where(xx < 30, 2.0, 3.0), np.nan)
    original = depth.copy()
    data = {"valid": valid, "depth": depth, "spacing": 0.002}
    result = select(data, 347)
    repeat = select(data, 347)
    indices, scales = result["indices"], result["scales"]
    assert indices.dtype == np.int64
    assert len(indices) == len(np.unique(indices)) == 347
    assert valid.ravel()[indices].all()
    frame, y, x = np.unravel_index(indices, valid.shape)
    assert np.array_equal(indices, (frame * 47 + y) * 61 + x)
    assert np.array_equal(indices, repeat["indices"])
    assert np.array_equal(scales, repeat["scales"])
    assert scales.shape == (347, 2)
    assert np.isfinite(scales).all() and (scales >= 0.5).all()
    assert np.array_equal(depth, original, equal_nan=True)
    assert sum(result["stats"]["per_frame_quota"]) == 347
    assert result["stats"]["covered_tiles"] == result["stats"]["nonempty_tiles"]


def test_step_gets_more_detail_without_leaving_flat_surface_holes():
    height, width = 96, 128
    yy, xx = np.indices((height, width))
    depth = np.where(xx < 64, 2.0, 3.0)[None]
    valid = np.ones_like(depth, dtype=bool)
    result = select({"valid": valid, "depth": depth, "spacing": 0.002}, height * width // 16)
    selected = np.zeros_like(valid)
    selected.ravel()[result["indices"]] = True
    band = (abs(xx - 63.5) < 2) & (yy >= 8) & (yy < height - 8)
    interior = (abs(xx - 63.5) > 8) & (xx >= 8) & (xx < width - 8) & (yy >= 8) & (yy < height - 8)
    assert selected[0, band].mean() > 1.8 * selected[0, interior].mean()
    assert selected[0, abs(xx - 63.5) >= 2].sum() > 0.75 * len(result["indices"])
    # Independent image-space coverage proxy: even the worst interior valid
    # pixel remains within a small neighborhood of a selected observation.
    assert distance_transform_edt(~selected[0])[interior].max() < 7
    for y in range(8, height - 8, 8):
        for x in range(8, width - 8, 8):
            assert selected[0, y:y + 8, x:x + 8].any()
    _, sy, sx = np.unravel_index(result["indices"], valid.shape)
    on_step = (abs(sx - 63.5) < 1) & (sy >= 8) & (sy < height - 8)
    assert on_step.sum() > 0
    assert (result["scales"][on_step, 0] == 0.5).all()
    assert np.median(result["scales"][on_step, 1]) > 1.5


def test_smooth_depth_slope_does_not_become_a_dense_discontinuity():
    _, xx = np.indices((48, 64))
    depth = (2 + 0.02 * xx)[None]
    result = select({"valid": np.ones_like(depth, bool), "depth": depth, "spacing": 0.002}, 192)
    assert result["stats"]["per_frame_depth_jump_links"] == [0]
    assert result["stats"]["scale_u"]["p50"] > 2


@pytest.mark.parametrize("budget", [1, 43, 320, 646])
def test_sparse_and_saturated_budgets_are_exact(budget):
    depth = np.ones((2, 17, 19))
    result = select({"valid": np.ones_like(depth, bool), "depth": depth, "spacing": 0.002}, budget)
    assert len(np.unique(result["indices"])) == budget
    assert np.isfinite(result["scales"]).all()
    assert (result["scales"] >= 0.5).all()
    if budget == depth.size:
        assert np.array_equal(result["indices"], np.arange(depth.size))


@pytest.mark.parametrize("budget", [0, 10, 1.5, True])
def test_unfulfillable_or_noninteger_budget_is_rejected(budget):
    depth = np.ones((1, 3, 3))
    with pytest.raises(ValueError, match="budget"):
        select({"valid": np.ones_like(depth, bool), "depth": depth, "spacing": 0.002}, budget)
