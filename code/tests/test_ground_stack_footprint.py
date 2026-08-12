import numpy as np
import pytest

from utils.ground_stack_footprint import (
    FootprintError,
    SupportPlane,
    carton_footprint_polygon,
    fit_support_plane,
    union_footprints,
)


def make_plane_grid(point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return a dense metric plane grid for deterministic fitting tests."""
    del normal  # These task fixtures exercise the horizontal support-table case.
    coordinates = np.linspace(-1.0, 1.0, 101)
    x_values, y_values = np.meshgrid(coordinates, coordinates, indexing="xy")
    return np.column_stack(
        [
            point[0] + x_values.ravel(),
            point[1] + y_values.ravel(),
            np.full(x_values.size, point[2]),
        ]
    )


def test_fit_support_plane_recovers_metric_table_with_outliers():
    table = make_plane_grid(
        point=np.array([0.0, 0.0, 1.0]), normal=np.array([0.0, 0.0, 1.0])
    )
    random_outliers = np.random.default_rng(7).uniform(-5.0, 5.0, size=(1_000, 3))

    plane = fit_support_plane(np.vstack([table, random_outliers]))

    assert abs(np.dot(plane.normal, [0.0, 0.0, 1.0])) == pytest.approx(
        1.0, abs=1e-3
    )
    assert plane.inlier_count >= 10_000


def test_fit_support_plane_rejects_insufficient_background():
    with pytest.raises(FootprintError, match="support plane"):
        fit_support_plane(np.zeros((9_999, 3)))


def horizontal_support_plane() -> SupportPlane:
    return SupportPlane(
        point=np.zeros(3),
        normal=np.array([0.0, 0.0, 1.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        inlier_count=10_000,
        inlier_fraction=1.0,
        p95_residual_m=0.0,
    )


def carton_points(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    height: float,
) -> np.ndarray:
    """Sample two opposing carton faces, providing two fused views."""
    x_values = np.linspace(*x_range, 40)
    y_values = np.linspace(*y_range, 40)
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    first_face = np.column_stack(
        [x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, height)]
    )
    second_face = np.column_stack(
        [x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, height - 0.01)]
    )
    return np.vstack([first_face, second_face])


def test_two_sampled_carton_faces_fuse_to_one_obb_footprint():
    polygon, metrics = carton_footprint_polygon(
        carton_points(x_range=(0.0, 1.0), y_range=(0.0, 1.0), height=0.2),
        horizontal_support_plane(),
    )

    assert polygon.area == pytest.approx(1.0)
    assert metrics["input_point_count"] == 3_200


def test_upper_carton_points_produce_their_ground_projected_footprint():
    polygon, _ = carton_footprint_polygon(
        carton_points(x_range=(0.5, 1.5), y_range=(0.0, 1.0), height=0.8),
        horizontal_support_plane(),
    )

    assert polygon.area == pytest.approx(1.0)


def test_overlapping_carton_footprints_use_union_not_sum():
    plane = horizontal_support_plane()
    first = carton_points(x_range=(0.0, 1.0), y_range=(0.0, 1.0), height=0.2)
    upper = carton_points(x_range=(0.5, 1.5), y_range=(0.0, 1.0), height=0.8)

    first_polygon, _ = carton_footprint_polygon(first, plane)
    upper_polygon, _ = carton_footprint_polygon(upper, plane)

    assert union_footprints([first_polygon, upper_polygon]).area == pytest.approx(1.5)


def test_line_like_carton_points_are_rejected():
    line_points = np.column_stack(
        [np.linspace(0.0, 1.0, 640), np.zeros(640), np.full(640, 0.2)]
    )

    with pytest.raises(FootprintError, match="OBB"):
        carton_footprint_polygon(line_points, horizontal_support_plane())
