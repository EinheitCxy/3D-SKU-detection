"""Focused source-preservation, continuity and strict-visibility checks."""

import numpy as np
import pytest

from perf.roi_fusion.texture import face_view_candidates
from perf.surfel_coverage import texture


def scene_data(views=3):
    return {
        "K": np.repeat(np.array([[[200., 0., 64.], [0., 200., 64.], [0., 0., 1.]]]), views, axis=0),
        "E": np.repeat(np.eye(4)[None], views, axis=0),
        "depth": np.full((views, 129, 129), 2., dtype=np.float32),
        "mask": np.ones((views, 129, 129), dtype=bool),
        "spacing": .006,
    }


def triangles_at(x_positions, radius=.001):
    triangle = np.array([[-radius, -radius, 2.], [0., radius, 2.], [radius, -radius, 2.]])
    vertices = np.concatenate([triangle + [x, 0., 0.] for x in x_positions])
    return vertices, np.arange(len(vertices)).reshape(-1, 3)


def test_valid_original_is_frozen_even_when_another_source_scores_higher():
    data = scene_data(2)
    data["E"][0, 2, 3] = -.2
    data["depth"][0] = 1.8
    vertices, faces = triangles_at([0.])
    _, quality, _, _, _ = face_view_candidates(vertices, faces, data)
    assert quality[0, 0] > quality[1, 0]
    sources, stats = texture.choose_sources(vertices, faces, data, preferred_sources=[1])
    np.testing.assert_array_equal(sources, [1])
    assert sources.dtype == np.int32
    assert stats["preserved_original_faces"] == 1
    assert stats["switched_valid_original_faces"] == 0
    assert stats["recovered_faces"] == 0
    assert stats["restored_original_source_area_fraction"] == pytest.approx(1.)


def test_residual_patch_uses_frozen_boundary_source_and_keeps_primary_faces():
    data = scene_data()
    data["mask"][0] = False
    # Frame 2 is higher quality, but frame 1 anchors both residual boundaries.
    data["E"][2, 2, 3] = -.1
    data["depth"][2] = 1.9
    vertices, faces = triangles_at(np.arange(10) * .01 - .05)
    preferred = np.array([1] + [0] * 8 + [1])
    _, quality, _, _, _ = face_view_candidates(vertices, faces, data)
    assert quality[2, 1:9].sum() > quality[1, 1:9].sum()
    sources, stats = texture.choose_sources(vertices, faces, data, preferred_sources=preferred)
    np.testing.assert_array_equal(sources, np.ones(10, dtype=np.int32))
    assert stats["preserved_original_faces"] == 2
    assert stats["recovered_faces"] == 8
    assert stats["assigned_patches"] == 1
    assert stats["anchored_assigned_patches"] == 1
    assert stats["alternative_source_transitions"] == {"0->1": 8}
    assert stats["chosen_sources_valid"]
    assert stats["untextured_area_fraction"] == 0.
    assert stats["recovered_area_m2"] == pytest.approx(stats["initial_untextured_area_m2"])


def test_conflicting_views_do_not_create_a_per_face_mosaic():
    data = scene_data()
    data["mask"][0] = False
    vertices, faces = triangles_at(np.arange(8) * .01 - .04)
    data["mask"][1, :, 64:] = False
    data["mask"][2, :, :64] = False
    visibility, _, _, _, _ = face_view_candidates(vertices, faces, data)
    assert visibility.any(axis=0).all()
    assert not visibility.all(axis=1).any()
    sources, stats = texture.choose_sources(vertices, faces, data, preferred_sources=np.zeros(8, dtype=int))
    np.testing.assert_array_equal(sources, np.full(8, -1))
    assert stats["deferred_no_common_view_patches"] == 1
    assert stats["observable_but_unassigned_faces"] == 8
    assert stats["untextured_area_fraction"] == pytest.approx(1.)


@pytest.mark.parametrize("failure", ["occlusion", "vertex_mask", "backface"])
def test_alternative_source_must_pass_unchanged_strict_gates(failure):
    data = scene_data(2)
    data["mask"][0] = False
    vertices, faces = triangles_at([0.] * 4, radius=.015)
    sources, _ = texture.choose_sources(vertices, faces, data, preferred_sources=np.zeros(4, dtype=int))
    np.testing.assert_array_equal(sources, np.ones(4))
    if failure == "occlusion":
        data["depth"][1] = 1.9
    elif failure == "vertex_mask":
        # Centroid remains supported; one vertex must reject the complete face.
        data["mask"][1, 62, 62] = False
    else:
        faces = faces[:, ::-1]
    sources, stats = texture.choose_sources(vertices, faces, data, preferred_sources=np.zeros(4, dtype=int))
    np.testing.assert_array_equal(sources, np.full(4, -1))
    assert stats["no_strict_view_faces"] == 4
    assert stats["invalid_chosen_source_faces"] == 0


def test_projection_batches_preserve_visibility_and_reject_tiny_islands(monkeypatch):
    data = scene_data(2)
    data["mask"][0] = False
    vertices, faces = triangles_at([0., .01, .02, .03, .3])
    original = texture.face_view_candidates
    sizes = []

    def recording_candidates(vertices, faces, data):
        sizes.append(len(faces))
        return original(vertices, faces, data)

    monkeypatch.setattr(texture, "_FACE_BATCH", 2)
    monkeypatch.setattr(texture, "face_view_candidates", recording_candidates)
    sources, stats = texture.choose_sources(vertices, faces, data, preferred_sources=np.zeros(5, dtype=int))
    assert sizes == [2, 2, 1]
    np.testing.assert_array_equal(sources, [1, 1, 1, 1, -1])
    assert stats["deferred_small_patches"] == 1
    assert stats["observable_but_unassigned_faces"] == 1
    assert stats["candidate_arrays_bytes"] == 5 * (2 * 5 + 7 * 8)


def test_invalid_primary_index_is_rejected_before_selection():
    data = scene_data(2)
    vertices, faces = triangles_at([0.])
    with pytest.raises(ValueError, match="invalid frame index"):
        texture.choose_sources(vertices, faces, data, preferred_sources=[2])
