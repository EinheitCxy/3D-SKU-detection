"""Projection, visibility and GLB boundary checks for the isolated experiment."""
import json
import struct

import numpy as np
from PIL import Image

from perf.roi_fusion.texture import choose_sources, export_textured_mesh, project_uv


def scene_data():
    return {
        "K": np.array([[[10., 0., 10.], [0., 10., 10.], [0., 0., 1.]]]),
        "E": np.eye(4)[None],
        "affine": np.eye(3)[None],
        "depth": np.full((1, 21, 21), 2., dtype=np.float32),
        "mask": np.ones((1, 21, 21), dtype=bool),
        "spacing": np.array(.01),
    }


def test_uv_undoes_crop_resize_and_uses_pixel_centers():
    data = scene_data()
    data["affine"][0] = [[.5, 0., -10.], [0., .5, -20.], [0., 0., 1.]]
    # World origin projects to processed (10,10), original (40,60).
    uv = project_uv(np.array([[0., 0., 2.]]), 0, data, (100, 200))
    np.testing.assert_allclose(uv, [[.405, .6975]])


def test_visibility_rejects_occlusion_and_backfaces():
    data = scene_data()
    vertices = np.array([[-.02, -.02, 2.], [0., .02, 2.], [.02, -.02, 2.]])
    faces = np.array([[0, 1, 2]])
    sources, _ = choose_sources(vertices, faces, data)
    assert sources.tolist() == [0]
    data["depth"][:] = 1.
    sources, _ = choose_sources(vertices, faces, data)
    assert sources.tolist() == [-1]
    data["depth"][:] = 2.
    sources, _ = choose_sources(vertices, faces[:, ::-1], data)
    assert sources.tolist() == [-1]


def test_disconnected_neighbors_keep_one_patch_source():
    data = scene_data()
    for key in ("K", "E", "affine", "depth", "mask"):
        data[key] = np.repeat(data[key], 2, axis=0)
    data["spacing"] = np.array(.2)
    data["depth"][0, :, 11:] = 1.
    data["depth"][1, :, :10] = 1.
    triangle = np.array([[-.01, -.01, 2.], [0., .01, 2.], [.01, -.01, 2.]])
    vertices = np.concatenate((triangle + [-.2, 0, 0], triangle + [.2, 0, 0]))
    sources, stats = choose_sources(vertices, [[0, 1, 2], [3, 4, 5]], data)
    assert stats["patches"] == 1
    assert np.count_nonzero(sources == -1) == 1
    assert len(set(sources[sources >= 0])) == 1


def test_mask_rejects_triangle_with_center_inside_but_vertex_outside():
    data = scene_data()
    vertices = np.array([[-.2, -.2, 2.], [0., .2, 2.], [.2, -.2, 2.]])
    data["mask"][0, 9, 9] = False
    sources, _ = choose_sources(vertices, [[0, 1, 2]], data)
    assert sources.tolist() == [-1]


def test_original_sources_are_retained_without_visible_alternative():
    data = scene_data()
    for key in ("K", "E", "affine", "depth", "mask"):
        data[key] = np.repeat(data[key], 2, axis=0)
    vertices = np.array([[-.02, -.02, 2.], [0., .02, 2.], [.02, -.02, 2.]])
    sources, stats = choose_sources(vertices, [[0, 1, 2]], data, preferred_sources=[1])
    assert sources.tolist() == [1]
    assert stats["selection_mode"] == "original-source"
    data["mask"][1] = False
    sources, _ = choose_sources(vertices, [[0, 1, 2]], data, preferred_sources=[1])
    assert sources.tolist() == [-1]


def test_export_embeds_jpeg_and_unlit_material_with_uv(tmp_path):
    data = scene_data()
    source = tmp_path / "source.jpg"
    Image.new("RGB", (21, 21), (255, 100, 0)).save(source)
    data["image_paths"] = np.array([str(source)])
    vertices = np.array([[-.02, -.02, 2.], [0., .02, 2.], [.02, -.02, 2.]])
    output = tmp_path / "mesh.glb"
    stats = export_textured_mesh(vertices, [[0, 1, 2]], [0], data, output)
    blob = output.read_bytes()
    length = struct.unpack_from("<I", blob, 12)[0]
    tree = json.loads(blob[20:20 + length])
    assert tree["images"][0]["mimeType"] == "image/jpeg"
    assert "KHR_materials_unlit" in tree["materials"][0]["extensions"]
    assert tree["materials"][0]["name"] == "frame_0"
    assert "TEXCOORD_0" in tree["meshes"][0]["primitives"][0]["attributes"]
    assert stats["textured_faces"] == 1
