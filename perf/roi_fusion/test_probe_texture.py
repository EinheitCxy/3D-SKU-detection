"""Material risks of the experimental visibility and static source optimizer."""
import numpy as np
import trimesh

from perf.roi_fusion.probe_texture import label_faces, mesh_view_candidates
from perf.roi_fusion.test_texture import scene_data


def test_mesh_visibility_rejects_hidden_layer_and_excessive_depth_error():
    data = scene_data()
    triangle = np.array([[-.2, -.2, 2.], [0., .2, 2.], [.2, -.2, 2.]])
    vertices = np.concatenate((triangle, triangle + [0, 0, .01]))
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    visible, _, _ = mesh_view_candidates(vertices, faces, data, .04)
    assert visible.tolist() == [[True, False]]
    # A hidden layer inside the numerical hit tolerance must still be hidden.
    vertices[3:] = triangle + [0, 0, .0001]
    visible, _, _ = mesh_view_candidates(vertices, faces, data, .04)
    assert visible.tolist() == [[True, False]]
    data['depth'][:] = 1.
    visible, _, _ = mesh_view_candidates(vertices, faces, data, .04)
    assert not visible.any()


def test_mesh_visibility_keeps_mask_and_front_constraints():
    data = scene_data()
    vertices = np.array([[-.2, -.2, 2.], [0., .2, 2.], [.2, -.2, 2.]])
    faces = np.array([[0, 1, 2]])
    data['mask'][0, 9, 9] = False
    assert not mesh_view_candidates(vertices, faces, data, .04)[0].any()
    data['mask'][:] = True
    assert not mesh_view_candidates(vertices, faces[:, ::-1], data, .04)[0].any()


def test_static_labeling_covers_valid_faces_without_invalid_source_or_energy_increase():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]])
    mesh = trimesh.Trimesh(vertices, [[0, 1, 2], [0, 2, 3], [1, 4, 2]], process=False)
    visible = np.array([[True, False, False], [False, True, False]])
    scores = visible.astype(float)
    labels, stats = label_faces(mesh, visible, scores)
    assert labels.tolist() == [0, 1, -1]
    assert np.all(np.diff(stats['energy_history']) <= 1e-8)
    visible[:] = True
    scores = np.array([[1., .99, 1.], [.99, 1., .99]])
    labels, stats = label_faces(mesh, visible, scores)
    assert len(set(labels)) == 1
    assert np.all(np.diff(stats['energy_history']) <= 1e-8)
    assert stats['energy_history'][-1] < stats['energy_history'][0]
