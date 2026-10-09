"""Small geometric examples for the isolated fusion experiment."""

import numpy as np
import pytest

from perf.roi_fusion.fusion import disc_mesh, fuse_surfels


def observations():
    points = np.array([[0, 0, 2], [0, 0, 2.002], [0.08, 0, 2]])
    return {
        "points": points,
        "u": np.tile([0.01, 0, 0], (3, 1)),
        "v": np.tile([0, 0.01, 0], (3, 1)),
        "normals": np.tile([0, 0, -1], (3, 1)),
        "confidence": np.ones(3),
        "frame": np.array([0, 1, 0]),
        "xy": np.array([[8, 8], [8, 8], [12, 8]]),
        "depth": np.stack([np.full((17, 17), 2), np.full((17, 17), 2.002)]),
        "mask": np.ones((2, 17, 17), dtype=bool),
        "K": np.tile([[100, 0, 8], [0, 100, 8], [0, 0, 1]], (2, 1, 1)),
        "E": np.tile(np.eye(4), (2, 1, 1)),
        "spacing": np.array(0.01),
    }


def test_repeated_surface_fuses_and_unsupported_point_is_preserved():
    data = observations()
    result, stats = fuse_surfels(data)
    assert len(result["points"]) == 2
    assert sorted(result["support"].tolist()) == [1, 2]
    supported = result["support"] == 2
    np.testing.assert_allclose(result["points"][supported], [[0, 0, 2.001]])
    np.testing.assert_allclose(result["points"][~supported], [[0.08, 0, 2]])
    assert result["membership"][0] == result["membership"][1]
    assert result["membership"][0] != result["membership"][2]
    assert stats["unsupported_observations_preserved"] == 1


@pytest.mark.parametrize("obstruction", ["occluded", "other_product", "opposite_face"])
def test_geometric_or_product_boundary_prevents_merge(obstruction):
    data = observations()
    if obstruction == "occluded":
        data["depth"][1, 8, 8] = 1.5
    elif obstruction == "other_product":
        data["mask"][1, 8, 8] = False
    else:
        # A second camera sees the back of a thin object. Euclidean distance
        # alone would collapse its physically distinct front and back faces.
        data["E"][1] = np.diag([-1.0, 1.0, -1.0, 1.0])
        data["E"][1, 2, 3] = 4
        data["depth"][1] = 1.998
        data["normals"][1] = [0, 0, 1]
    result, _ = fuse_surfels(data)
    assert len(result["points"]) == 3
    np.testing.assert_allclose(result["points"], data["points"])


def test_fused_tangents_and_mesh_winding_follow_surface_normal():
    data = observations()
    data["u"][1] = [0.01, 0, 0.001]
    result, _ = fuse_surfels(data)
    cross = np.cross(result["u"], result["v"])
    cross /= np.linalg.norm(cross, axis=1)[:, None]
    np.testing.assert_allclose(cross, result["normals"], atol=1e-10)
    vertices, faces = disc_mesh(result["points"], result["u"], result["v"])
    triangle_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                vertices[faces[:, 2]] - vertices[faces[:, 0]])
    assert np.all(np.einsum("ij,ij->i", triangle_normals,
                            np.repeat(result["normals"], 8, axis=0)) > 0)


def test_same_frame_neighbors_cannot_be_collapsed_via_a_third_observation():
    data = observations()
    data["points"][2] = [0.01, 0, 2]
    data["xy"][2] = [9, 8]
    result, _ = fuse_surfels(data)
    assert len(result["points"]) == 2
    assert result["membership"][0] != result["membership"][2]


def test_transitive_pairs_do_not_bridge_beyond_normal_distance_gate():
    data = observations()
    data["points"] = np.array([[0, 0, 2], [0, 0, 2.014], [0, 0, 2.028]])
    data["frame"] = np.arange(3)
    data["xy"] = np.tile([8, 8], (3, 1))
    data["K"] = np.repeat(data["K"][:1], 3, axis=0)
    data["E"] = np.repeat(data["E"][:1], 3, axis=0)
    data["mask"] = np.ones((3, 17, 17), dtype=bool)
    data["depth"] = np.stack([np.full((17, 17), z) for z in [2, 2.014, 2.028]])
    result, stats = fuse_surfels(data)
    assert len(result["points"]) == 2
    assert result["membership"][0] != result["membership"][2]
    assert stats["rejected_cluster_edges"] == 1
