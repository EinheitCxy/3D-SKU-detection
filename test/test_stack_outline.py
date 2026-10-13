import json

import numpy as np
import pytest
from shapely.geometry import Point, box

from utils.stack_outline import OutlineOptions, extract_stack_outlines


def samples(bounds, step=0.02):
    x0, z0, x1, z1 = bounds
    x, z = np.meshgrid(np.arange(x0 + step / 2, x1, step),
                       np.arange(z0 + step / 2, z1, step))
    return np.column_stack((x.ravel(), z.ravel()))


def run(points, *, gap=0, hole=0, resolution=0.02, frames=None):
    return extract_stack_outlines(points, options=OutlineOptions(
        gap_radius_m=gap, max_hole_area_m2=hole,
        grid_resolution_m=resolution), frame_ids=frames)


def area(result):
    return sum(p.area for p in result.polygons)


def assert_simple(result):
    assert all(p.is_valid and p.exterior.is_simple and not p.interiors
               and p.area > 0 for p in result.polygons)
    assert all(a.intersection(b).area == 0 for i, a in enumerate(result.polygons)
               for b in result.polygons[i + 1:])
    json.dumps(result.diagnostics, allow_nan=False)


def test_rectangle_rotation_and_resolution_convergence():
    errors = []
    for resolution in (0.04, 0.02):
        points = samples((0.013, -0.017, 2.013, 2.983), resolution / 2)
        result = run(points, resolution=resolution)
        assert not result.issues
        assert len(result.polygons) == 1
        assert area(result) == pytest.approx(6, abs=10 * resolution)
        errors.append(abs(area(result) - 6))
        assert_simple(result)
    assert errors[1] < errors[0]
    theta = 0.43
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    rotated = run(samples((0, 0, 2, 3), 0.01) @ rotation.T,
                  gap=0.025, hole=0.01)
    assert len(rotated.polygons) == 1
    assert area(rotated) == pytest.approx(6, abs=0.2)
    assert_simple(rotated)


def test_l_shape_keeps_missing_corner():
    points = np.vstack([samples((0, 0, 1, 3)), samples((1, 0, 3, 1))])
    result = run(points, gap=0.04)
    assert len(result.polygons) == 1
    assert area(result) == pytest.approx(5, abs=0.2)
    assert not result.polygons[0].covers(Point(2, 2))
    assert_simple(result)


def test_u_entrance_stays_open():
    points = np.vstack([samples((0, 0, 0.5, 3)), samples((2.5, 0, 3, 3)),
                        samples((0.5, 0, 2.5, 0.5))])
    result = run(points, gap=0.06, hole=0.01)
    assert not result.issues
    assert len(result.polygons) == 1
    assert not result.polygons[0].covers(Point(1.5, 1.5))
    assert area(result) == pytest.approx(4, abs=0.24)


def test_small_seam_closes_but_two_heaps_remain_separate():
    seam = np.vstack([samples((0, 0, 1, 1)), samples((1.04, 0, 2.04, 1))])
    assert len(run(seam).polygons) == 2
    closed = run(seam, gap=0.06)
    assert len(closed.polygons) == 1
    assert area(closed) == pytest.approx(2.04, abs=0.1)
    heaps = np.vstack([samples((0, 0, 1, 1)), samples((1.5, 0, 2.5, 1))])
    assert len(run(heaps, gap=0.06).polygons) == 2


def test_small_secondary_heap_and_frame_diagnostics_are_preserved():
    main = samples((0, 0, 2, 3))
    small = samples((4, 0, 4.2, 0.2))
    points = np.vstack([main, small])
    frames = np.r_[np.zeros(len(main), dtype=int), np.full(len(small), 7)]
    result = run(points, frames=frames)
    assert len(result.polygons) == 2
    assert sorted(p.area for p in result.polygons) == pytest.approx([0.04, 6])
    assert result.diagnostics['raw_frame_count'] == 2
    assert result.diagnostics['raw_occupied_cells'] == len(points)
    assert result.diagnostics['frame_cell_counts'] == {'0': len(main), '7': len(small)}
    shuffled = run(points[::-1], frames=frames[::-1])
    assert [p.wkt for p in result.polygons] == [p.wkt for p in shuffled.polygons]
    assert_simple(result)


@pytest.mark.parametrize('degenerate', [np.array([[4., 0.]]),
    np.column_stack((np.arange(4., 5., 0.02), np.zeros(50))),
    np.column_stack((np.arange(4., 5., 0.02), np.arange(4., 5., 0.02)))])
def test_degenerate_support_is_unresolved_without_fabricated_width(degenerate):
    result = run(np.vstack([samples((0, 0, 1, 1)), degenerate]), gap=0.04)
    assert result.issues
    assert not result.diagnostics['complete']
    assert area(result) == pytest.approx(1, abs=0.04)
    assert result.diagnostics['unresolved_cells'] > 0


def test_small_hole_fills_and_large_newly_sealed_hole_is_unresolved():
    solid = samples((0, 0, 1, 1))
    small_hole = solid[~((solid[:, 0] > 0.4) & (solid[:, 0] < 0.5)
                         & (solid[:, 1] > 0.4) & (solid[:, 1] < 0.5))]
    filled = run(small_hole, hole=0.011)
    assert not filled.issues
    assert area(filled) == pytest.approx(1)
    assert filled.diagnostics['filled_hole_cells'] == 25
    ring = np.vstack([samples((0, 0, 0.4, 3)), samples((2.6, 0, 3, 3)),
                      samples((0.4, 0, 2.6, 0.4)), samples((0.4, 2.6, 1.48, 3)),
                      samples((1.52, 2.6, 2.6, 3))])
    open_result = run(ring, hole=0.02)
    assert not open_result.issues
    result = run(np.vstack([ring, samples((5, 0, 5.2, 0.2))]), gap=0.06, hole=0.02)
    assert result.issues
    assert any('hole' in reason for reason in result.issues)
    assert len(result.polygons) == 1
    assert area(result) == pytest.approx(0.04)
    assert result.diagnostics['unresolved_cells'] > 0


@pytest.mark.parametrize('kwargs', [dict(gap_radius_m=-1), dict(gap_radius_m=np.nan),
    dict(max_hole_area_m2=np.inf), dict(max_hole_area_m2=-0.1),
    dict(grid_resolution_m=0), dict(grid_resolution_m=np.nan),
    dict(max_grid_cells=0), dict(max_grid_cells=1.5)])
def test_invalid_options(kwargs):
    options = dict(gap_radius_m=0, max_hole_area_m2=0)
    options.update(kwargs)
    with pytest.raises(ValueError):
        OutlineOptions(**options)


def test_huge_sparse_extent_and_morphology_padding_are_bounded():
    with pytest.raises(ValueError, match='max_grid_cells'):
        run(np.array([[0, 0], [1e6, 1e6]]))
    with pytest.raises(ValueError, match='max_grid_cells'):
        run(samples((0, 0, 0.1, 0.1)), gap=1e6)


def test_empty_and_invalid_input():
    result = run(np.empty((0, 2)))
    assert not result.polygons and result.issues
    assert not result.diagnostics['complete']
    with pytest.raises(ValueError):
        run(np.array([[np.nan, 0]]))
    with pytest.raises(ValueError):
        run(np.ones((3, 3)))
    with pytest.raises(ValueError):
        run(samples((0, 0, 1, 1)), frames=np.zeros(3))


def test_diagonal_cells_are_separate_and_negative_floor_anchor_is_exact():
    # Two squares meet only at a corner; no self-touching exterior is allowed.
    result = run(np.vstack([samples((-0.04, -0.04, 0, 0), 0.01),
                            samples((0, 0, 0.04, 0.04), 0.01)]))
    assert len(result.polygons) == 2
    assert_simple(result)
    assert result.polygons[0].equals(box(-0.04, -0.04, 0, 0))


def test_repeated_multiframe_line_never_creates_two_dimensional_support():
    line = np.column_stack((np.arange(0.01, 1, 0.02), np.full(50, 0.01)))
    result = run(np.tile(line, (10, 1)), frames=np.repeat(np.arange(10), 50), gap=0.06)
    assert not result.polygons
    assert result.issues
    assert result.diagnostics['raw_frame_count'] == 10
    assert result.diagnostics['raw_occupied_cells'] == 50
    assert result.diagnostics['unresolved_cells'] == 50


def test_finite_opposing_extremes_rejected_before_integer_allocation():
    with pytest.raises(ValueError, match='max_grid_cells'):
        run(np.array([[-1e306, -1e306], [1e306, 1e306]]))


def test_two_million_cell_rectangle_uses_exact_cell_edges():
    result = run(samples((0, 0, 1, 2), 0.001), resolution=0.001)
    assert not result.issues
    assert result.diagnostics['raw_occupied_cells'] == 2_000_000
    assert result.diagnostics['grid_cells'] <= 4_000_000
    assert len(result.polygons) == 1
    assert area(result) == pytest.approx(2, abs=1e-10)
    assert_simple(result)
