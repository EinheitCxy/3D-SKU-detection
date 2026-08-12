import numpy as np
import pytest

from utils.ground_stack_footprint import (
    FootprintError,
    SupportPlane,
    _refine_support_plane,
    _sample_ransac_triplet,
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


def test_support_plane_rejects_inlier_fraction_below_ten_percent():
    with pytest.raises(FootprintError, match="insufficient inliers"):
        _refine_support_plane(np.zeros((10_000, 3)), total_points=100_001)


def test_support_plane_rejects_p95_residual_above_threshold():
    coordinates = np.linspace(-1.0, 1.0, 100)
    x_values, y_values = np.meshgrid(coordinates, coordinates, indexing="xy")
    inliers = np.column_stack(
        [x_values.ravel(), y_values.ravel(), np.zeros(x_values.size)]
    )
    inliers[:9_400, 2] = 0.012
    inliers[9_400:, 2] = -0.012

    with pytest.raises(FootprintError, match="residual"):
        _refine_support_plane(inliers, total_points=len(inliers))


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


def boundary_preserving_carton_points() -> np.ndarray:
    """Return a dense 0.050 m x 1 m footprint surviving percentile trimming."""
    x_values = np.concatenate(
        [np.zeros(12), np.linspace(0.005, 0.045, 9), np.full(12, 0.05)]
    )
    y_values = np.concatenate(
        [np.zeros(12), np.linspace(0.025, 0.975, 39), np.ones(12)]
    )
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    first_face = np.column_stack(
        [x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, 0.2)]
    )
    second_face = np.column_stack(
        [x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, 0.19)]
    )
    return np.vstack([first_face, second_face])


def connected_component_points(count: int, x_offset: float) -> np.ndarray:
    """Create one DBSCAN-connected component with dimensions above 0.05 m."""
    indices = np.arange(count)
    return np.column_stack(
        [
            x_offset + (indices % 7) * 0.02,
            (indices // 7) * 0.02,
            np.full(count, 0.2),
        ]
    )


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


def test_narrow_carton_obb_is_rejected():
    narrow_points = carton_points(
        x_range=(0.0, 0.049), y_range=(0.0, 1.0), height=0.2
    )

    with pytest.raises(FootprintError, match="OBB"):
        carton_footprint_polygon(narrow_points, horizontal_support_plane())


def test_exactly_five_centimeter_carton_obb_is_accepted():
    polygon, _ = carton_footprint_polygon(
        boundary_preserving_carton_points(), horizontal_support_plane()
    )
    side_lengths = np.sort(
        np.linalg.norm(np.diff(np.asarray(polygon.exterior.coords), axis=0), axis=1)
    )

    assert side_lengths[0] == pytest.approx(0.05, abs=1e-10)
    assert side_lengths[-1] == pytest.approx(1.0, abs=1e-10)


def test_carton_with_multiple_substantial_components_is_rejected():
    first_component = carton_points(
        x_range=(0.0, 1.0), y_range=(0.0, 1.0), height=0.2
    )
    second_component = carton_points(
        x_range=(3.0, 4.0), y_range=(0.0, 1.0), height=0.2
    )

    with pytest.raises(FootprintError, match="multiple substantial components"):
        carton_footprint_polygon(
            np.vstack([first_component, second_component]), horizontal_support_plane()
        )


def test_low_count_substantial_second_component_is_rejected():
    first_component = connected_component_points(35, x_offset=0.0)
    second_component = connected_component_points(29, x_offset=1.0)

    with pytest.raises(FootprintError, match="multiple substantial components"):
        carton_footprint_polygon(
            np.vstack([first_component, second_component]), horizontal_support_plane()
        )


def test_absolute_32_point_second_component_is_rejected():
    first_component = connected_component_points(168, x_offset=0.0)
    second_component = connected_component_points(32, x_offset=1.0)

    with pytest.raises(FootprintError, match="multiple substantial components"):
        carton_footprint_polygon(
            np.vstack([first_component, second_component]), horizontal_support_plane()
        )


def test_ransac_triplet_sampling_is_distinct_and_seed_deterministic():
    first = _sample_ransac_triplet(np.random.default_rng(13), population_size=10)
    second = _sample_ransac_triplet(np.random.default_rng(13), population_size=10)

    assert tuple(first) == (8, 7, 9)
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 3
