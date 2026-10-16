"""Portable GLB layout and the external bundle boundary."""
from io import BytesIO
import json
import struct
import zipfile

import numpy as np
from PIL import Image
import pytest

from src.surfel_glb import export_surfel_glb


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "generation"
    root.mkdir()
    matrix = np.eye(4)
    matrix[:3, 3] = [1, 2, 3]
    manifest = dict(schema_version="3.0.0", backend="DA3", frame_count=1,
                    world_to_view=matrix.reshape(-1).tolist())
    frame = dict(extrinsic=np.eye(4)[:3].tolist(), intrinsic=np.eye(3).tolist(),
                 processed_to_source=np.eye(3).tolist(), source_size=[2, 2],
                 texture_size=[2, 2], texture="surfel-texture-0.jpg")
    for name, value in {
        "manifest.json": manifest,
        "objects.json": {"7": {"point_ranges": [[1, 2]]}},
        "surfel.json": dict(version=3, point_count=2, grid_size=[2, 2], frames=[frame]),
    }.items():
        (root / name).write_text(json.dumps(value))
    for name, value, dtype in [
        ("positions.f32.bin", [[0, 0, 1], [1, 0, 1]], "<f4"),
        ("surfel-u.f16.bin", [[1, 0, 0]] * 2, "<f2"),
        ("surfel-v.f16.bin", [[0, 1, 0]] * 2, "<f2"),
        ("surfel-scale.f16.bin", [[1, 1]] * 2, "<f2"),
        ("surfel-frame.u8.bin", [0, 0], "u1"),
        ("surfel-depth.f16.bin", np.ones((1, 2, 2)), "<f2"),
    ]:
        (root / name).write_bytes(np.asarray(value, dtype=dtype).tobytes())
    Image.new("RGB", (2, 2), "red").save(root / frame["texture"])
    return root


@pytest.fixture
def fake_baker(monkeypatch):
    captured = {}

    def bake(**kwargs):
        captured.update(kwargs)
        stream = BytesIO()
        Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(stream, format="PNG")
        for center in kwargs["positions"]:
            image = (dict(image_rgba=np.full((4, 4, 4), [255, 0, 0, 255], dtype=np.uint8))
                     if kwargs.get("image_format") == "rgba"
                     else dict(image_png=stream.getvalue()))
            yield dict(positions=np.array([center] * 4, dtype="<f4"),
                       uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="<f4"),
                       indices=np.array([[0, 1, 2], [0, 2, 3]], dtype="<u4"),
                       **image, surfel_count=1)

    monkeypatch.setattr("src.surfel_mesh.bake_surfel_pages", bake)
    return captured


def parse(path):
    data = path.read_bytes()
    assert struct.unpack_from("<4sII", data) == (b"glTF", 2, len(data))
    length, kind = struct.unpack_from("<I4s", data, 12)
    assert kind == b"JSON" and length % 4 == 0
    doc = json.loads(data[20:20 + length])
    bin_length, kind = struct.unpack_from("<I4s", data, 20 + length)
    binary = data[28 + length:]
    assert kind == b"BIN\0" and bin_length == len(binary)
    assert len(binary) % 4 == 0
    return doc, binary


@pytest.mark.parametrize("zipped", [False, True])
def test_roundtrip_layout_and_selection(bundle, tmp_path, fake_baker, zipped):
    source = bundle
    if zipped:
        source = tmp_path / "bundle.zip"
        with zipfile.ZipFile(source, "w") as archive:
            for path in bundle.iterdir():
                archive.write(path, path.name)
    output = tmp_path / "scene.glb"
    report = export_surfel_glb(source, output, global_id=7)
    assert report["point_count"] == 1 and report["triangle_count"] == 2
    assert report["atlas_pages"] == 1
    np.testing.assert_array_equal(fake_baker["positions"], [[1, 0, 1]])
    assert fake_baker["radius"] == 2.0
    doc, binary = parse(output)
    assert doc["nodes"][0]["matrix"][12:15] == [1, 2, 3]
    assert doc["extensionsUsed"] == ["KHR_materials_unlit"]
    primitive = doc["meshes"][0]["primitives"][0]
    assert primitive["mode"] == 4
    position = doc["accessors"][primitive["attributes"]["POSITION"]]
    view = doc["bufferViews"][position["bufferView"]]
    values = np.frombuffer(binary, dtype="<f4", count=12, offset=view["byteOffset"])
    np.testing.assert_array_equal(values.reshape(4, 3), [[1, 0, 1]] * 4)
    for view in doc["bufferViews"]:
        assert view["byteOffset"] % 4 == 0
        assert view["byteOffset"] + view["byteLength"] <= len(binary)
    image_view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    raw = binary[image_view["byteOffset"]:image_view["byteOffset"] + image_view["byteLength"]]
    with Image.open(BytesIO(raw)) as image:
        assert image.mode == "RGBA" and image.size == (4, 4)
    material = doc["materials"][0]
    assert material["alphaMode"] == "MASK" and material["doubleSided"]
    assert material["alphaCutoff"] == .5
    with pytest.raises(FileExistsError):
        export_surfel_glb(source, output)


def test_all_points_and_v2_unit_scales(bundle, tmp_path, fake_baker):
    metadata = json.loads((bundle / "surfel.json").read_text())
    metadata["version"] = 2
    (bundle / "surfel.json").write_text(json.dumps(metadata))
    (bundle / "surfel-scale.f16.bin").unlink()
    report = export_surfel_glb(bundle, tmp_path / "all.glb")
    assert report["point_count"] == 2 and report["atlas_pages"] == 2
    np.testing.assert_array_equal(fake_baker["scales"], np.ones((2, 2)))


def test_lod_selection_counts_are_distinct_from_invalid_geometry(bundle, tmp_path, fake_baker, monkeypatch):
    def select(**inputs):
        np.testing.assert_array_equal(inputs["object_ids"], [-1, 0])
        return np.array([1]), {"removed": 1}

    monkeypatch.setattr("src.surfel_lod.select_surfel_lod", select)
    report = export_surfel_glb(bundle, tmp_path / "lod.glb", lod=True)
    assert report["input_point_count"] == 2
    assert report["selected_point_count"] == report["baked_surfel_count"] == 1
    assert report["lod_removed_point_count"] == 1
    assert report["skipped_surfel_count"] == 0
    np.testing.assert_array_equal(fake_baker["positions"], [[1, 0, 1]])


@pytest.mark.parametrize("problem", ["length", "nonfinite", "missing", "escape", "frame", "range"])
def test_rejects_bad_bundle(bundle, tmp_path, fake_baker, problem):
    if problem == "length":
        (bundle / "surfel-u.f16.bin").write_bytes(b"x")
    elif problem == "nonfinite":
        (bundle / "positions.f32.bin").write_bytes(np.full((2, 3), np.nan, "<f4").tobytes())
    elif problem == "missing":
        (bundle / "surfel-texture-0.jpg").unlink()
    elif problem == "escape":
        path = bundle / "surfel.json"
        metadata = json.loads(path.read_text())
        metadata["frames"][0]["texture"] = "../outside.jpg"
        path.write_text(json.dumps(metadata))
    elif problem == "frame":
        (bundle / "surfel-frame.u8.bin").write_bytes(bytes([0, 1]))
    else:
        (bundle / "objects.json").write_text(json.dumps({"7": {"point_ranges": [[0, 3]]}}))
    output = tmp_path / "bad.glb"
    with pytest.raises((ValueError, FileNotFoundError)):
        export_surfel_glb(bundle, output)
    assert not output.exists()
    assert not fake_baker


@pytest.mark.parametrize("visible_count", [0, 1])
def test_invisible_surfels_are_skipped_or_empty_export_rejected(
    bundle, tmp_path, fake_baker, monkeypatch, visible_count
):
    from src import surfel_mesh

    original_baker = surfel_mesh.bake_surfel_pages

    def filtered_baker(**kwargs):
        for index, page in enumerate(original_baker(**kwargs)):
            if index < visible_count:
                yield page

    monkeypatch.setattr(surfel_mesh, "bake_surfel_pages", filtered_baker)
    output = tmp_path / "visible.glb"
    if visible_count == 0:
        with pytest.raises(ValueError, match="no visible, nondegenerate geometry"):
            export_surfel_glb(bundle, output)
        assert not output.exists()
    else:
        report = export_surfel_glb(bundle, output)
        assert report["input_point_count"] == report["point_count"] == 2
        assert report["baked_surfel_count"] == 1
        assert report["skipped_surfel_count"] == 1
        assert report["triangle_count"] == 2
        doc, _ = parse(output)
        assert len(doc["meshes"][0]["primitives"]) == 1


def test_unsupported_backend_rejected(bundle, tmp_path, fake_baker):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["backend"] = "Pi3X"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "pi3x.glb"
    with pytest.raises(ValueError, match="supports only backend DA3"):
        export_surfel_glb(bundle, output)
    assert not output.exists()
    assert not fake_baker


def test_scene_delivery_bakes_and_compacts_existing_assets(bundle, tmp_path):
    from src.surfel_glb import export_scene_glb

    output = tmp_path / "delivery" / "scene.glb"
    report = export_scene_glb(bundle, output)
    doc, binary = parse(output)
    assert report["glb_bytes"] == output.stat().st_size
    assert report["bake"]["triangle_count"] > 0
    assert "KHR_mesh_quantization" in doc["extensionsRequired"]
    for mesh in doc["meshes"]:
        for primitive in mesh["primitives"]:
            assert doc["accessors"][primitive["attributes"]["POSITION"]]["componentType"] == 5123
    for entry in doc["images"]:
        view = doc["bufferViews"][entry["bufferView"]]
        start = view.get("byteOffset", 0)
        with Image.open(BytesIO(binary[start:start + view["byteLength"]])) as image:
            assert image.mode == "P"
    assert list(output.parent.iterdir()) == [output]
    with pytest.raises(FileExistsError):
        export_scene_glb(bundle, output)


def test_scene_delivery_accepts_unquantized_custom_bake_parameters(bundle, tmp_path):
    from src.surfel_glb import export_scene_glb

    output = tmp_path / "custom" / "scene.glb"
    report = export_scene_glb(
        bundle,
        output,
        radius=1.05,
        tile_size=10,
        lod=False,
        palette_colors=None,
    )

    assert report["radius"] == 1.05
    assert report["tile_size"] == 10
    assert report["lod"] is False
    assert report["palette_colors"] is None
    assert report["bake"]["radius"] == 1.05
    assert report["bake"]["tile_size"] == 10
    doc, binary = parse(output)
    assert doc["asset"]["version"] == "2.0"
    image_view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    start = image_view.get("byteOffset", 0)
    raw = binary[start : start + image_view["byteLength"]]
    with Image.open(BytesIO(raw)) as image:
        assert image.mode == "RGBA"
        assert image.size == (24, 12)


@pytest.mark.parametrize("stage", ["encode", "write"])
def test_scene_delivery_leaves_no_partial_output_on_failure(bundle, tmp_path, monkeypatch, stage):
    from src.surfel_glb import export_scene_glb

    def fail(*args, **kwargs):
        if stage == "write":
            args[1].write(b"partial")
        raise RuntimeError("export failed")

    target = ("src.surfel_glb_texture.encode_atlas" if stage == "encode"
              else "src.surfel_glb_compact.shutil.copyfileobj")
    monkeypatch.setattr(target, fail)
    output = tmp_path / "delivery" / "scene.glb"
    with pytest.raises(RuntimeError, match="export failed"):
        export_scene_glb(bundle, output)
    assert list(output.parent.iterdir()) == []


def assert_same_compact_assets(left_path, right_path):
    """Compare geometry and PNG payloads independently of BIN chunk packing order."""
    left, left_binary = parse(left_path)
    right, right_binary = parse(right_path)

    def payload(document, binary, index):
        view = document["bufferViews"][index]
        start = view["byteOffset"]
        metadata = {key: value for key, value in view.items() if key != "byteOffset"}
        return metadata, binary[start:start + view["byteLength"]]

    for key in set(left) - {"accessors", "bufferViews", "images", "buffers"}:
        assert left[key] == right[key], key
    assert len(left["accessors"]) == len(right["accessors"])
    for a, b in zip(left["accessors"], right["accessors"]):
        assert {k: v for k, v in a.items() if k != "bufferView"} == {
            k: v for k, v in b.items() if k != "bufferView"}
        assert payload(left, left_binary, a["bufferView"]) == payload(right, right_binary, b["bufferView"])
    assert len(left["images"]) == len(right["images"])
    for a, b in zip(left["images"], right["images"]):
        assert a["mimeType"] == b["mimeType"] == "image/png"
        assert payload(left, left_binary, a["bufferView"]) == payload(right, right_binary, b["bufferView"])


@pytest.mark.parametrize("palette_colors", [None, 128])
def test_streamed_delivery_matches_two_stage_geometry_and_textures(
    bundle, tmp_path, fake_baker, palette_colors
):
    from functools import partial
    from src.surfel_glb import export_scene_glb
    from src.surfel_glb_compact import compact_surfel_glb
    from src.surfel_glb_texture import compact_png, rgba_png

    baked, old, direct = (tmp_path / name for name in ("baked.glb", "old.glb", "direct.glb"))
    export_surfel_glb(bundle, baked, lod=True)
    transform = rgba_png if palette_colors is None else partial(compact_png, colors=palette_colors)
    before = compact_surfel_glb(baked, old, image_transform=transform)
    report = export_scene_glb(bundle, direct, palette_colors=palette_colors)
    assert_same_compact_assets(old, direct)
    for key in ("geometry_bytes_before", "geometry_bytes_after", "texture_bytes_after",
                "max_position_error", "max_uv_error", "triangle_count", "primitive_count"):
        assert report["compact"][key] == before[key]
    assert report["bake"]["glb_bytes"] is None
    assert report["compact"]["source_bytes"] is None
    assert report["compact"]["texture_bytes_before"] is None
    assert report["delivery_mode"] == "streamed"
