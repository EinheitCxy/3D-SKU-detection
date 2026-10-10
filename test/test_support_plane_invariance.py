"""Geometric invariance and convergence of support-plane selection."""

import numpy as np
import pytest

import utils.ground_stack_footprint as footprint_geometry
from utils.ground_stack_footprint import (
    SupportPlaneSelectionError,
    _refine_support_plane,
    select_support_plane,
)


def plane_grid(width, length, nx=120, ny=100):
    x, y = np.meshgrid(
        np.linspace(-width / 2, width / 2, nx),
        np.linspace(-length / 2, length / 2, ny),
    )
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


def test_support_plane_selection_is_rigid_transform_invariant():
    points = plane_grid(0.8, 1.4)
    objects = plane_grid(0.2, 0.4, 10, 10) + [0, 0, 0.03]
    frames = np.arange(len(points)) % 3
    rotation, _ = np.linalg.qr(
        np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 1.0], [2.0, 1.0, 5.0]])
    )
    translation = np.array([31.0, -17.0, 8.0])
    original, before = select_support_plane(points, frames, objects)
    transformed, after = select_support_plane(
        points @ rotation.T + translation,
        frames,
        objects @ rotation.T + translation,
    )
    first = before["candidates"][before["selected_index"]]
    second = after["candidates"][after["selected_index"]]
    assert first["gates"] == second["gates"]
    assert second["spans_m"] == pytest.approx([0.8, 1.4])
    assert second["hull_area_m2"] == pytest.approx(first["hull_area_m2"])
    assert abs(np.dot(transformed.normal, rotation @ original.normal)) == pytest.approx(
        1
    )
    assert abs(np.dot(transformed.point - translation, transformed.normal)) < 1e-10


@pytest.mark.parametrize("angle", [0.0, np.pi / 4])
def test_narrow_support_is_rejected_even_when_diagonal(angle):
    points = plane_grid(0.2, 2.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    points = points @ rotation.T
    objects = plane_grid(0.05, 0.1, 10, 10) + [0.0, 0.0, 0.03]
    # The old coordinate-axis spans both exceed 0.30 m for the diagonal case.
    if angle:
        assert np.all(np.ptp(points[:, :2], axis=0) > 0.30)
    with pytest.raises(SupportPlaneSelectionError) as raised:
        select_support_plane(points, np.arange(len(points)) % 3, objects)
    candidate = raised.value.diagnostics["candidates"][0]
    assert candidate["spans_m"] == pytest.approx([0.2, 2.0])
    assert candidate["gates"]["in_plane_span"] is False
    assert candidate["gates"]["hull_area"] is True


def test_refinement_converges_after_more_than_three_trims():
    base = plane_grid(2.0, 2.0)
    tail_grid = plane_grid(2.0, 2.0, 10, 10)
    heights = [
        0.010130,
        0.010411,
        0.010417,
        0.010423,
        0.010426,
        0.011142,
        0.011174,
        0.012480,
        0.014565,
        0.014849,
        0.015041,
        0.015674,
        0.015871,
        0.016401,
        0.016727,
    ]
    points = np.vstack([base] + [tail_grid + [0, 0, h] for h in heights])
    plane, retained = _refine_support_plane(
        points,
        len(points),
        return_retained_indices=True,
    )
    # Three trims leave 200 tail points; their p95 residual still passes because
    # the 12,000 true plane points dominate. Convergence must remove the tails.
    assert plane.inlier_count == len(base)
    np.testing.assert_array_equal(retained, np.arange(len(base)))
    assert np.max(np.abs((points[retained] - plane.point) @ plane.normal)) <= 0.010


def test_refinement_budget_keeps_last_plane_and_support_consistent(monkeypatch):
    base = plane_grid(2.0, 2.0)
    tail_grid = plane_grid(2.0, 2.0, 10, 10)
    # Each symmetric 100-point layer sits just outside the fit containing it
    # and all lower layers. This removes one layer per fit for 15 iterations.
    heights = []
    for layer in range(1, 16):
        heights.append(
            (0.010 * (120 + layer) + sum(heights)) / (120 + layer - 1) + 0.000001
        )
    points = np.vstack([base] + [tail_grid + [0, 0, h] for h in heights])
    fits = []
    original_fit = footprint_geometry._fit_plane_svd

    def record_fit(support):
        result = original_fit(support)
        fits.append((len(support), result))
        return result

    monkeypatch.setattr(footprint_geometry, "_fit_plane_svd", record_fit)
    plane, retained = _refine_support_plane(
        points, len(points), return_retained_indices=True
    )

    assert len(fits) == 10
    assert fits[-1][0] == 12_600
    np.testing.assert_array_equal(plane.point, fits[-1][1][0])
    np.testing.assert_array_equal(plane.normal, fits[-1][1][1])
    residuals = np.abs((points - plane.point) @ plane.normal)
    np.testing.assert_array_equal(retained, np.flatnonzero(residuals <= 0.010))
    assert plane.inlier_count == len(retained) == 12_500
    assert plane.inlier_fraction == pytest.approx(len(retained) / len(points))
    assert plane.p95_residual_m == pytest.approx(np.percentile(residuals[retained], 95))
    next_point, next_normal = original_fit(points[retained])
    assert np.any(np.abs((points[retained] - next_point) @ next_normal) > 0.010)
