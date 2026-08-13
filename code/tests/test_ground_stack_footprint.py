import numpy as np
import pytest
import utils.ground_stack_footprint as footprint_geometry

from utils.ground_stack_footprint import (
    FootprintError,
    RansacOutcome,
    SupportPlane,
    SupportPlaneSelectionError,
    _adaptive_ransac_plane,
    _refine_support_plane,
    _sample_ransac_triplet,
    carton_footprint_polygon,
    fit_support_plane,
    project_to_plane,
    select_footprint_component,
    select_support_plane,
    union_footprints,
    voxel_balance_projected,
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


def _reference_adaptive_ransac(
    points: np.ndarray, threshold_m: float, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], int]:
    """Preserve the pre-optimization RANSAC contract for parity checks."""
    generator = np.random.default_rng(seed)
    best_count, best_candidate, trial, target_trials = 0, None, 0, 10_000
    while trial < target_trials:
        indices = _sample_ransac_triplet(generator, population_size=len(points))
        first, second, third = points[indices]
        normal = np.cross(second - first, third - first)
        norm = np.linalg.norm(normal)
        if norm == 0:
            trial += 1
            continue
        normal /= norm
        count = int(np.count_nonzero(np.abs((points - first) @ normal) <= threshold_m))
        if count > best_count:
            best_count, best_candidate = count, (first, normal)
            ratio = min(max(count / len(points), 1e-9), 1.0 - 1e-12)
            target_trials = min(
                10_000,
                max(128, int(np.ceil(np.log(0.001) / np.log(1.0 - ratio**3)))),
            )
        trial += 1
    assert best_candidate is not None
    return best_candidate, trial


def test_adaptive_ransac_workspace_matches_reference_candidate_exactly():
    table = make_plane_grid(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    points = np.vstack([table, np.random.default_rng(4).uniform(-2.0, 2.0, (900, 3))])

    expected, expected_trials = _reference_adaptive_ransac(points, 0.012, 13)
    actual = _adaptive_ransac_plane(points, 0.012, 13)

    np.testing.assert_array_equal(actual.point, expected[0])
    np.testing.assert_array_equal(actual.normal, expected[1])
    assert actual.trial_count == expected_trials
    assert actual.early_exit is False


def test_perfect_candidate_returns_on_first_non_degenerate_trial():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    outcome = _adaptive_ransac_plane(points, 0.012, 13)

    assert outcome.early_exit is True
    assert outcome.trial_count == 1


def test_adaptive_ransac_keeps_threshold_plus_minus_one_ulp_behavior():
    threshold = np.float64(0.012)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.2, threshold],
        ]
    )

    for candidate_threshold in (
        np.nextafter(threshold, 0.0),
        np.nextafter(threshold, np.inf),
    ):
        expected, expected_trials = _reference_adaptive_ransac(
            points, candidate_threshold, 13
        )
        actual = _adaptive_ransac_plane(points, candidate_threshold, 13)
        np.testing.assert_array_equal(actual.point, expected[0])
        np.testing.assert_array_equal(actual.normal, expected[1])
        assert actual.trial_count <= expected_trials


def test_adaptive_ransac_tie_preserves_first_strictly_better_triplet():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )

    expected, expected_trials = _reference_adaptive_ransac(points, 0.012, 13)
    outcome = _adaptive_ransac_plane(points, 0.012, 13)

    np.testing.assert_array_equal(outcome.point, expected[0])
    np.testing.assert_array_equal(outcome.normal, expected[1])
    assert outcome.trial_count == expected_trials


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


def test_support_plane_refinement_trims_ransac_tail_without_relaxing_ten_mm_gate():
    coordinates = np.linspace(-1.0, 1.0, 120)
    x_values, y_values = np.meshgrid(coordinates, coordinates, indexing="xy")
    table = np.column_stack(
        [x_values.ravel(), y_values.ravel(), np.zeros(x_values.size)]
    )
    tail = table[::12].copy()
    tail[:, 2] = 0.011

    plane = _refine_support_plane(
        np.vstack([table, tail]),
        total_points=len(table) + len(tail),
        max_residual_m=0.010,
    )

    assert plane.inlier_count == len(table)
    assert plane.inlier_fraction > 0.90
    assert plane.p95_residual_m <= 0.010


def test_support_plane_refinement_rejects_when_trim_drops_below_ten_percent():
    coordinates = np.linspace(-1.0, 1.0, 100)
    x_values, y_values = np.meshgrid(coordinates, coordinates, indexing="xy")
    table = np.column_stack(
        [x_values.ravel(), y_values.ravel(), np.zeros(x_values.size)]
    )
    tail = table[::10].copy()
    tail[:, 2] = 0.011

    with pytest.raises(FootprintError, match="after residual refinement"):
        _refine_support_plane(
            np.vstack([table, tail]),
            total_points=110_000,
            max_residual_m=0.010,
        )


def test_support_plane_gates_use_final_ten_mm_support_points(monkeypatch):
    coordinates = np.linspace(-1.0, 1.0, 120)
    x_values, y_values = np.meshgrid(coordinates, coordinates, indexing="xy")
    table = np.column_stack(
        [x_values.ravel(), y_values.ravel(), np.zeros(x_values.size)]
    )
    tail_coordinates = np.linspace(5.0, 9.0, 40)
    tail_x, tail_y = np.meshgrid(tail_coordinates, tail_coordinates, indexing="xy")
    tail = np.column_stack(
        [
            tail_x.ravel(),
            tail_y.ravel(),
            np.where((np.indices(tail_x.shape).sum(axis=0) % 2) == 0, 0.0119, -0.0119).ravel(),
        ]
    )
    background = np.vstack([table, tail])
    frames = np.arange(len(background)) % 3
    observations = [carton_points((0.0, 0.2), (0.0, 0.2), 0.02)]

    monkeypatch.setattr(
        footprint_geometry,
        "_adaptive_ransac_plane",
        lambda *_args, **_kwargs: RansacOutcome(
            np.zeros(3), np.array([0.0, 0.0, 1.0]), 0, False
        ),
    )

    _, diagnostics = select_support_plane(background, frames, observations)
    candidate = diagnostics["candidates"][0]

    assert candidate["spans_m"] == pytest.approx([2.0, 2.0])
    assert candidate["hull_area_m2"] == pytest.approx(4.0)
    assert candidate["ransac"] == {"trial_count": 0, "early_exit": False}


def test_support_plane_refinement_rejects_empty_residual_trim():
    coordinates = np.linspace(-1.0, 1.0, 100)
    x_values, y_values = np.meshgrid(coordinates, coordinates, indexing="xy")
    base = np.column_stack(
        [x_values.ravel(), y_values.ravel(), np.zeros(x_values.size)]
    )
    upper = base.copy()
    lower = base.copy()
    upper[:, 2] = 0.011
    lower[:, 2] = -0.011

    with pytest.raises(FootprintError, match="after residual refinement"):
        _refine_support_plane(
            np.vstack([upper, lower]),
            total_points=len(upper) + len(lower),
            max_residual_m=0.010,
        )


def test_support_plane_preserves_refinement_rejection_diagnostics(monkeypatch):
    table = make_plane_grid(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    frame_ids = np.arange(len(table)) % 3
    observations = [carton_points((0.0, 0.2), (0.0, 0.2), 0.02)]

    def reject_refinement(*_args, **_kwargs):
        raise FootprintError("support plane residual exceeds 0.010 m")

    monkeypatch.setattr(footprint_geometry, "_refine_support_plane", reject_refinement)

    with pytest.raises(
        SupportPlaneSelectionError,
        match="no support-plane candidate passed refinement gates",
    ) as raised:
        select_support_plane(table, frame_ids, observations)

    candidates = raised.value.diagnostics["candidates"]
    assert candidates
    assert all(candidate["refinement"]["passed"] is False for candidate in candidates)
    assert all(candidate["refinement"]["reason"].startswith("support plane residual") for candidate in candidates)
    assert all(candidate["raw_inlier_count"] >= 10_000 for candidate in candidates)
    assert all(candidate["raw_inlier_fraction"] >= 0.10 for candidate in candidates)


def test_support_plane_refinement_rejection_keeps_exact_ransac_diagnostics(
    monkeypatch,
):
    table = make_plane_grid(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    frame_ids = np.arange(len(table)) % 3
    observations = [carton_points((0.0, 0.2), (0.0, 0.2), 0.02)]

    monkeypatch.setattr(
        footprint_geometry,
        "_adaptive_ransac_plane",
        lambda *_args, **_kwargs: RansacOutcome(
            np.zeros(3), np.array([0.0, 0.0, 1.0]), 37, True
        ),
    )

    def reject_refinement(*_args, **_kwargs):
        raise FootprintError("support plane residual exceeds 0.010 m")

    monkeypatch.setattr(
        footprint_geometry,
        "_refine_support_plane",
        reject_refinement,
    )

    with pytest.raises(SupportPlaneSelectionError) as raised:
        select_support_plane(table, frame_ids, observations)

    candidate = raised.value.diagnostics["candidates"][0]
    assert candidate["refinement"]["passed"] is False
    assert candidate["ransac"] == {"trial_count": 37, "early_exit": True}


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
    x_values = np.linspace(*x_range, 80)
    y_values = np.linspace(*y_range, 80)
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
        [np.zeros(12), np.linspace(0.01, 0.99, 99), np.ones(12)]
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
            x_offset + (indices % 7) * 0.01,
            (indices // 7) * 0.01,
            np.full(count, 0.2),
        ]
    )


def test_two_sampled_carton_faces_fuse_to_one_obb_footprint():
    polygon, metrics = carton_footprint_polygon(
        carton_points(x_range=(0.0, 1.0), y_range=(0.0, 1.0), height=0.2),
        horizontal_support_plane(),
    )

    assert polygon.area == pytest.approx(1.0)
    assert metrics["input_point_count"] == 12_800


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


def test_select_support_plane_prefers_table_over_larger_wall():
    coordinates = np.linspace(-1.0, 1.0, 120)
    x_values, y_values = np.meshgrid(coordinates, coordinates, indexing="xy")
    table = np.column_stack([x_values.ravel(), y_values.ravel(), np.zeros(x_values.size)])
    wall_y, wall_z = np.meshgrid(np.linspace(-2.0, 2.0, 160), np.linspace(0.0, 2.0, 160), indexing="xy")
    wall = np.column_stack([np.full(wall_y.size, 3.0), wall_y.ravel(), wall_z.ravel()])
    object_points = carton_points((0.0, 0.8), (0.0, 0.8), 0.02)
    frame_ids = np.concatenate(
        [np.arange(len(table)) % 3, np.arange(len(wall)) % 3]
    )

    plane, diagnostics = select_support_plane(
        np.vstack([table, wall]), frame_ids, object_points
    )

    assert abs(np.dot(plane.normal, [0.0, 0.0, 1.0])) == pytest.approx(1.0, abs=1e-3)
    assert diagnostics["selected_index"] is not None


def test_projected_voxel_balance_ignores_height_density():
    plane = horizontal_support_plane()
    points = np.asarray(
        [[0.001, 0.001, 0.02], [0.001, 0.001, 0.80], [0.006, 0.001, 0.30]]
    )

    balanced = voxel_balance_projected(project_to_plane(points, plane), voxel_size_m=0.005)

    assert balanced.shape == (2, 2)


def test_component_uses_exact_twenty_millimetre_dbscan_radius():
    y_values = np.linspace(0.0, 0.08, 17)
    first = np.column_stack([np.full(len(y_values), 0.0), y_values])
    at_twenty_mm = np.column_stack([np.full(len(y_values), 0.020), y_values])
    at_thirty_mm = np.column_stack([np.full(len(y_values), 0.030), y_values])

    component, diagnostics = select_footprint_component(np.vstack([first, at_twenty_mm]))

    assert len(component) == 34
    assert diagnostics["eps_m"] == pytest.approx(0.020)
    assert diagnostics["min_samples"] == 4
    with pytest.raises(FootprintError, match="multiple substantial components"):
        select_footprint_component(np.vstack([first, at_thirty_mm]))


def test_support_plane_rejects_when_four_of_five_observation_centres_are_outside_table():
    table = make_plane_grid(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    frame_ids = np.arange(len(table)) % 3
    observations = [
        carton_points((0.0, 0.2), (0.0, 0.2), 0.02),
        carton_points((3.0, 3.2), (0.0, 0.2), 0.02),
        carton_points((0.0, 0.2), (3.0, 3.2), 0.02),
        carton_points((-3.2, -3.0), (0.0, 0.2), 0.02),
        carton_points((0.0, 0.2), (-3.2, -3.0), 0.02),
    ]

    with pytest.raises(SupportPlaneSelectionError) as raised:
        select_support_plane(table, frame_ids, observations)

    assert raised.value.diagnostics["candidates"]
    assert raised.value.diagnostics["candidates"][0]["gates"]["object_hull"] is False


def test_ambiguous_wall_and_table_preserve_candidate_diagnostics():
    coordinates = np.linspace(-1.0, 1.0, 120)
    x_values, y_values = np.meshgrid(coordinates, coordinates, indexing="xy")
    table = np.column_stack([x_values.ravel(), y_values.ravel(), np.zeros(x_values.size)])
    wall_y, wall_z = np.meshgrid(coordinates, coordinates, indexing="xy")
    wall = np.column_stack([np.zeros(wall_y.size), wall_y.ravel(), wall_z.ravel()])
    objects = [carton_points((0.02, 0.08), (0.0, 0.2), 0.02)]
    frame_ids = np.concatenate([np.arange(len(table)) % 3, np.arange(len(wall)) % 3])

    with pytest.raises(SupportPlaneSelectionError, match="ambiguous") as raised:
        select_support_plane(np.vstack([table, wall]), frame_ids, objects)

    assert len(raised.value.diagnostics["candidates"]) >= 2
