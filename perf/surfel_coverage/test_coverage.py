"""Behavior checks for the isolated coverage-budget candidate."""

import numpy as np
import pytest

from perf.surfel_coverage.coverage import select


def planes(offsets, height, width):
    """Repeated observations, with a translated camera for each separate plane."""
    frames = len(offsets)
    spacing = 1 / 64
    yy, xx = np.indices((height, width))
    points = np.empty((frames, height, width, 3), dtype=np.float32)
    points[..., 0] = (xx - width // 2) * spacing
    points[..., 1] = (yy - height // 2) * spacing
    points[..., 2] = 1
    points[..., 0] += np.asarray(offsets)[:, None, None]
    u, v = np.zeros_like(points), np.zeros_like(points)
    u[..., 0], v[..., 1] = spacing, spacing
    pose = np.tile(np.eye(4), (frames, 1, 1))
    pose[:, 0, 3] = -np.asarray(offsets)
    intrinsic = np.tile(np.eye(3), (frames, 1, 1))
    intrinsic[:, 0, 0] = intrinsic[:, 1, 1] = 1 / spacing
    intrinsic[:, 0, 2], intrinsic[:, 1, 2] = width // 2, height // 2
    valid = np.ones((frames, height, width), dtype=bool)
    baseline = np.flatnonzero((valid & (yy % 4 == 0) & (xx % 4 == 0)).reshape(-1))
    return dict(points_grid=points, u_grid=u, v_grid=v, valid=valid,
                depth=np.ones(valid.shape, dtype=np.float32),
                confidence_grid=np.full(valid.shape, 2, dtype=np.float32),
                K=intrinsic, E=pose, spacing=spacing, baseline_indices=baseline)


def test_exact_official_budget_has_unique_valid_original_indices_and_fixed_scales():
    data = planes([0, 0, 4, 4], 256, 256)
    data["valid"][:, 30:32, 47:50] = False
    data["baseline_indices"] = data["baseline_indices"][
        data["valid"].reshape(-1)[data["baseline_indices"]]
    ]
    before = data["points_grid"].copy()
    result = select(data, 217049)
    selected = result["indices"]
    assert selected.dtype == np.int64
    assert len(selected) == len(np.unique(selected)) == 217049
    assert data["valid"].reshape(-1)[selected].all()
    assert result["scales"].shape == (217049, 2)
    np.testing.assert_array_equal(result["scales"], 4)
    np.testing.assert_array_equal(data["points_grid"], before)
    assert sum(result["stats"]["per_frame_selected"]) == 217049


def test_rare_observed_surface_gets_budget_from_repeated_surface():
    data = planes([0] * 8 + [4], 64, 64)
    budget = len(data["baseline_indices"])
    result = select(data, budget)
    selected_frames = result["indices"] // (64 * 64)
    baseline_rare = np.count_nonzero(data["baseline_indices"] // (64 * 64) == 8)
    selected_rare = np.count_nonzero(selected_frames == 8)
    # The rare plane has equal physical extent but only one of nine views.
    # A useful distribution must overcome the baseline's 8:1 view-count skew.
    assert selected_rare >= 3 * baseline_rare
    assert budget * 0.35 <= selected_rare <= budget * 0.65
    assert result["stats"]["coverage_proxy"]["selected_cells"] > 0


def test_impossible_budget_fails_without_changing_candidate_validity():
    data = planes([0], 8, 8)
    with pytest.raises(ValueError, match="number of valid observations"):
        select(data, 65)
