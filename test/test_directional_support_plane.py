import numpy as np
import pytest

from utils.ground_stack_footprint import (
    SupportPlaneSelectionError,
    select_support_plane_with_direction,
)


def grid(width=1., length=1., height=0.):
    x, y = np.meshgrid(np.linspace(-width / 2, width / 2, 110),
                       np.linspace(-length / 2, length / 2, 100))
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, height)])


def select(points, normals=None, frames=None, objects=None, up=None):
    return select_support_plane_with_direction(
        points,
        np.arange(len(points)) % 3 if frames is None else frames,
        [grid(.2, .2, 1.)[::100]] if objects is None else objects,
        up_direction=np.array([0., 0., 1.]) if up is None else up,
        background_normals=np.tile([0., 0., 1.], (len(points), 1)) if normals is None else normals,
    )


def test_floor_below_ten_percent_background_is_not_hidden_by_wall():
    floor = grid()
    wall = np.tile(grid(3., 3.)[:, [2, 0, 1]] + [2., 0., 1.5], (11, 1))
    points = np.vstack([floor, wall])
    normals = np.vstack([np.tile([0., 0., 1.], (len(floor), 1)),
                         np.tile([1., 0., 0.], (len(wall), 1))])
    plane, diagnostics = select(points, normals)
    assert plane.point[2] == pytest.approx(0.)
    assert plane.inlier_fraction < .10
    assert diagnostics["horizontal_point_count"] == len(floor)
    assert diagnostics["method"] == "direction_constrained_height_modes"


def test_lowest_observed_plane_beats_more_populous_shelves():
    points = np.vstack([grid(), np.tile(grid(height=.5), (2, 1)),
                        np.tile(grid(height=1.), (3, 1))])
    plane, diagnostics = select(points, objects=[grid(.2, .2, 2.)[::100]])
    assert plane.point[2] == pytest.approx(0.)
    assert len(diagnostics["candidates"]) == 3
    assert diagnostics["semantics"] == "lowest_observed_horizontal_support"


def test_walls_without_horizontal_support_are_rejected():
    walls = np.vstack([grid()[:, [2, 0, 1]], grid()[:, [0, 2, 1]]])
    normals = np.vstack([np.tile([1., 0., 0.], (11000, 1)),
                         np.tile([0., 1., 0.], (11000, 1))])
    with pytest.raises(SupportPlaneSelectionError, match="no horizontal") as raised:
        select(walls, normals)
    assert raised.value.diagnostics["candidates"] == []


@pytest.mark.parametrize("kind", ["narrow", "single_frame"])
def test_inadequate_support_is_rejected(kind):
    points = grid(.2, 2.) if kind == "narrow" else grid()
    frames = np.zeros(len(points), dtype=int) if kind == "single_frame" else None
    with pytest.raises(SupportPlaneSelectionError) as raised:
        select(points, frames=frames)
    gate = "in_plane_span" if kind == "narrow" else "frame_span"
    assert raised.value.diagnostics["candidates"][0]["gates"][gate] is False


def test_high_objects_and_side_visible_floor_do_not_need_contact_or_hull_overlap():
    plane, diagnostics = select(grid(), objects=[grid(.2, .2, 2.)[::100] + [3., 0., 0.]])
    candidate = diagnostics["candidates"][diagnostics["selected_index"]]
    assert plane.point[2] == pytest.approx(0.)
    assert candidate["object_centres_inside_fraction"] == 0.
    assert candidate["object_above_fraction"] == 1.


def test_directional_selection_is_rigid_transform_invariant():
    points = np.vstack([grid(), grid(height=.5)])
    objects = [grid(.2, .2, 1.)[::100]]
    rotation, _ = np.linalg.qr(np.array([[1., 2., 3.], [4., 2., 1.], [2., 1., 5.]]))
    translation = np.array([31., -17., 8.])
    up = rotation @ np.array([0., 0., 1.])
    original, _ = select(points, objects=objects)
    transformed, diagnostics = select(
        points @ rotation.T + translation,
        normals=np.tile(up, (len(points), 1)),
        objects=[objects[0] @ rotation.T + translation], up=up,
    )
    assert abs((transformed.point - translation) @ up) < 1e-10
    assert np.dot(transformed.normal, up) == pytest.approx(1.)
    assert transformed.inlier_count == original.inlier_count
    candidate = diagnostics["candidates"][diagnostics["selected_index"]]
    assert candidate["spans_m"] == pytest.approx([1., 1.])


def test_object_above_gate_weights_observations_equally():
    objects = [np.tile(grid(.2, .2, 1.)[::100], (100, 1)), grid(.2, .2, -.5)[::100]]
    with pytest.raises(SupportPlaneSelectionError) as raised:
        select(grid(), objects=objects)
    assert raised.value.diagnostics["candidates"][0]["object_above_fraction"] == .5


def test_refined_tail_mode_is_deduplicated_without_merging_separate_layers():
    floor = grid()
    tails = []
    for layer in range(1, 13):
        count = int(1500 * .75 ** (layer - 1))
        indices = np.linspace(0, len(floor) - 1, count, dtype=int)
        tails.append(floor[indices] + [0., 0., .003 * layer])
    _, diagnostics = select(np.vstack([floor, *tails]))
    candidates = diagnostics["candidates"]
    duplicates = [candidate for candidate in candidates if "duplicate_of" in candidate]
    assert duplicates
    for duplicate in duplicates:
        original = candidates[duplicate["duplicate_of"]]
        assert abs(duplicate["raw_offset_m"] - original["raw_offset_m"]) > .024
        assert abs(duplicate["offset_m"] - original["offset_m"]) <= .024
    unique_eligible = [candidate for candidate in candidates
                       if all(candidate["gates"].values()) and "duplicate_of" not in candidate]
    assert len(unique_eligible) == 1
    assert diagnostics["selected_index"] == unique_eligible[0]["index"]
    assert len(candidates) <= 8

    _, separate = select(np.vstack([floor, floor + [0., 0., .0368]]))
    separate_eligible = [candidate for candidate in separate["candidates"]
                         if all(candidate["gates"].values()) and "duplicate_of" not in candidate]
    assert len(separate_eligible) == 2
    offsets = sorted(candidate["offset_m"] for candidate in separate_eligible)
    assert offsets[1] - offsets[0] == pytest.approx(.0368)
