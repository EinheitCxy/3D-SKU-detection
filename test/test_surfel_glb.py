"""Portable GLB layout and the external bundle boundary."""
from io import BytesIO
import json
import struct
from threading import Event, Thread, current_thread, main_thread
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


@pytest.mark.parametrize('mode', ['source_world', 'camera_basis'])
def test_scene_projection_mode_propagates_to_baker_and_reports(bundle, tmp_path, fake_baker, mode):
    from src.surfel_glb import export_scene_glb

    report = export_scene_glb(bundle, tmp_path / (mode+'.glb'), lod=False, projection_mode=mode)
    assert fake_baker['projection_mode'] == mode
    assert report['projection_mode'] == report['bake']['projection_mode'] == mode
    assert report['compact']['projection_mode'] == mode


def test_plain_projection_default_and_invalid_modes(bundle, tmp_path, fake_baker):
    report = export_surfel_glb(bundle, tmp_path / 'plain.glb')
    assert fake_baker['projection_mode'] == report['projection_mode'] == 'source_world'
    with pytest.raises(ValueError, match='projection_mode'):
        export_surfel_glb(bundle, tmp_path / 'invalid.glb', projection_mode='unknown')


def test_scene_invalid_projection_creates_no_parent(bundle, tmp_path):
    from src.surfel_glb import export_scene_glb

    parent = tmp_path / 'absent' / 'nested'
    with pytest.raises(ValueError, match='projection_mode'):
        export_scene_glb(bundle, parent / 'scene.glb', projection_mode='unknown')
    assert not parent.exists()
    assert not parent.parent.exists()


@pytest.mark.parametrize('fixed_shape', [False, True])
def test_scene_device_propagates_without_initializing_cuda(bundle, tmp_path, fake_baker, fixed_shape):
    from src.surfel_glb import export_scene_glb

    report = export_scene_glb(bundle, tmp_path / 'gpu.glb', lod=False,
                              bake_device='cuda:0', bake_fixed_shape=fixed_shape,
                              encoder_workers=4)
    assert fake_baker['bake_device'] == report['bake_device'] == 'cuda:0'
    assert fake_baker['bake_fixed_shape'] == report['bake_fixed_shape'] == fixed_shape
    assert report['compact']['encoder_workers'] == 4


@pytest.mark.parametrize('options', [dict(bake_device='auto'),
    dict(bake_device='cuda:0', tile_size=4),
    dict(bake_device='cuda:0', projection_mode='camera_basis'),
    dict(bake_fixed_shape=True)])
def test_scene_invalid_device_creates_no_parent(bundle, tmp_path, options):
    from src.surfel_glb import export_scene_glb

    parent = tmp_path / 'absent'
    with pytest.raises(ValueError):
        export_scene_glb(bundle, parent / 'invalid.glb', **options)
    assert not parent.exists()


def test_compact_delivery_current_is_fixed_and_exports(bundle, tmp_path, fake_baker, capsys):
    from src.surfel_glb_delivery import current_generation, main

    root = tmp_path / 'viewer'
    (root / 'runs').mkdir(parents=True)
    destination = root / 'runs' / 'fixed'
    bundle.rename(destination)
    (root / 'CURRENT').write_text(json.dumps({'run_id': 'fixed'}))
    assert current_generation(root) == destination
    output = tmp_path / 'delivery.glb'
    main(['--viewer-root', str(root), '--output', str(output), '--encoder-workers', '4'])
    assert output.exists()
    assert json.loads(capsys.readouterr().out)['bake_device'] == 'cpu'
    (root / 'CURRENT').write_text(json.dumps({'run_id': '..'}))
    with pytest.raises(ValueError):
        current_generation(root)


def test_scene_default_two_encoder_output(bundle, tmp_path, fake_baker):
    from src.surfel_glb import export_scene_glb

    output = tmp_path / 'default.glb'
    report = export_scene_glb(bundle, output, lod=False)
    assert output.stat().st_size == report['glb_bytes'] > 0
    assert report['encoder_workers'] == report['compact']['encoder_workers'] == 2
    assert report['compact']['encoder_queue_pages'] == 2
    assert report['projection_mode'] == fake_baker['projection_mode'] == 'source_world'


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


@pytest.mark.parametrize("backend", ["DA3", "PromptDA"])
def test_scene_delivery_bakes_and_compacts_existing_assets(bundle, tmp_path, backend):
    from src.surfel_glb import export_scene_glb

    if backend != "DA3":
        path = bundle / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["backend"] = backend
        path.write_text(json.dumps(manifest))
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


@pytest.fixture
def three_page_bundle(bundle, fake_baker, monkeypatch, request):
    """Reuse the existing bundle/baker for three (or requested five) RGBA pages."""
    from src import surfel_mesh

    metadata = json.loads((bundle / "surfel.json").read_text())
    count = getattr(request, "param", 3)
    metadata["point_count"] = count
    (bundle / "surfel.json").write_text(json.dumps(metadata))
    for name, dtype, width in [
        ("positions.f32.bin", "<f4", 3), ("surfel-u.f16.bin", "<f2", 3),
        ("surfel-v.f16.bin", "<f2", 3), ("surfel-scale.f16.bin", "<f2", 2),
        ("surfel-frame.u8.bin", "u1", 1),
    ]:
        path = bundle / name
        values = np.frombuffer(path.read_bytes(), dtype=dtype).reshape(-1, width)
        path.write_bytes(np.concatenate((values, np.repeat(values[:1], count - 2, axis=0))).tobytes())
    original = surfel_mesh.bake_surfel_pages
    state = {"produced": 0, "written": 0, "bounded": False, "pages": [],
             "queue_limit": 2, "max_unwritten": 0}

    def bake(**kwargs):
        for index, page in enumerate(original(**kwargs)):
            if state["bounded"]:
                assert state["produced"] - state["written"] < state["queue_limit"]
            image = page["image_rgba"]
            image[:, :, :3] = [index + 1, 89, 173]
            image[::2, ::2, 3] = 0  # Retain nonblack transparent RGB.
            state["produced"] += 1
            state["max_unwritten"] = max(state["max_unwritten"],
                                          state["produced"] - state["written"])
            state["pages"].append(page)
            yield page

    monkeypatch.setattr(surfel_mesh, "bake_surfel_pages", bake)
    return bundle, state


@pytest.mark.parametrize("palette_colors", [None, 128])
@pytest.mark.parametrize("encoder_workers,three_page_bundle", [(2, 3), (4, 5)],
                         indirect=["three_page_bundle"])
def test_parallel_atlas_fifo_is_bounded_and_byte_identical(
    three_page_bundle, tmp_path, monkeypatch, palette_colors, encoder_workers
):
    from src import surfel_glb_texture
    from src.surfel_glb import export_scene_glb
    from src.surfel_glb_compact import _CompactWriter

    bundle, state = three_page_bundle
    serial, parallel = tmp_path / "serial.glb", tmp_path / "parallel.glb"
    export_scene_glb(bundle, serial, lod=False, palette_colors=palette_colors, encoder_workers=1)
    count = 3 if encoder_workers == 2 else 5
    state.update(produced=0, written=0, bounded=True, pages=[], queue_limit=encoder_workers,
                 max_unwritten=0)
    second_done = Event()
    completion = []
    original_encode, original_image = surfel_glb_texture.encode_atlas, _CompactWriter.image
    original_geometry = _CompactWriter.geometry
    write_order = []

    def encode(rgba, colors):
        assert current_thread() is not main_thread()
        assert not rgba.flags.writeable
        index = int(rgba[0, 0, 0]) - 1
        if index == 0:
            assert second_done.wait(5), "second encode did not complete"
        raw = original_encode(rgba, colors=colors)
        completion.append(index)
        if index == 1:
            second_done.set()
        return raw

    def geometry(self, *args):
        assert current_thread() is main_thread()
        write_order.append(("geometry", args[-1]))
        return original_geometry(self, *args)

    def image(self, raw):
        assert current_thread() is main_thread()
        index = state["written"]
        write_order.append(("image", index))
        result = original_image(self, raw)
        state["written"] += 1
        return result

    monkeypatch.setattr(surfel_glb_texture, "encode_atlas", encode)
    monkeypatch.setattr(_CompactWriter, "geometry", geometry)
    monkeypatch.setattr(_CompactWriter, "image", image)
    report = export_scene_glb(bundle, parallel, lod=False, palette_colors=palette_colors,
                              encoder_workers=encoder_workers)
    assert sorted(completion) == list(range(count))
    assert completion.index(1) < completion.index(0)
    if encoder_workers == 2:
        assert completion == [1, 0, 2]
    assert state["produced"] == state["written"] == count
    assert state["max_unwritten"] == encoder_workers
    assert write_order == [(kind, i) for i in range(count) for kind in ("geometry", "image")]
    assert all(not value.flags.writeable for page in state["pages"]
               for value in page.values() if isinstance(value, np.ndarray))
    assert parallel.read_bytes() == serial.read_bytes()
    assert_same_compact_assets(serial, parallel)
    assert report["encoder_workers"] == report["compact"]["encoder_workers"] == encoder_workers
    assert report["compact"]["encoder_queue_pages"] == encoder_workers
    compact = report["compact"]
    assert compact["elapsed_seconds"] == pytest.approx(
        compact["encoding_join_seconds"] + compact["main_thread_write_seconds"])


@pytest.mark.parametrize("stage", ["produce", "encode", "geometry", "image", "write"])
@pytest.mark.parametrize("encoder_workers", [1, 2])
def test_parallel_atlas_failure_propagates_and_drains(
    three_page_bundle, tmp_path, monkeypatch, stage, encoder_workers
):
    from src import surfel_glb_texture, surfel_mesh
    from src.surfel_glb import export_scene_glb
    from src.surfel_glb_compact import _CompactWriter

    bundle, state = three_page_bundle
    original_bake = surfel_mesh.bake_surfel_pages
    original_encode = surfel_glb_texture.encode_atlas
    started, finished, producer_closed = [], [], []
    failure = RuntimeError("atlas " + stage + " failed")

    def encode(rgba, colors):
        index = int(rgba[0, 0, 0]) - 1
        started.append(index)
        try:
            if stage == "encode" and index == 0:
                raise failure
            return original_encode(rgba, colors=colors)
        finally:
            finished.append(index)

    def bake(**kwargs):
        try:
            for index, page in enumerate(original_bake(**kwargs)):
                if stage == "produce" and index == 1:
                    raise failure
                yield page
        finally:
            producer_closed.append(True)

    def fail(*args, **kwargs):
        if stage == "write":
            args[1].write(b"partial")
        raise failure

    monkeypatch.setattr(surfel_glb_texture, "encode_atlas", encode)
    monkeypatch.setattr(surfel_mesh, "bake_surfel_pages", bake)
    if stage in ("geometry", "image"):
        monkeypatch.setattr(_CompactWriter, stage, fail)
    if stage == "write":
        monkeypatch.setattr("src.surfel_glb_compact.shutil.copyfileobj", fail)
    output = tmp_path / "delivery" / "scene.glb"
    with pytest.raises(RuntimeError) as caught:
        export_scene_glb(bundle, output, lod=False, encoder_workers=encoder_workers)
    assert caught.value is failure
    assert caught.value.__traceback__ is not None  # Keep the failed export alive.
    assert producer_closed == [True]  # Cleanup cannot depend on traceback GC.
    assert sorted(started) == sorted(finished)  # No worker survives the failing export.
    assert list(output.parent.iterdir()) == []
    if stage != "write":
        assert state["produced"] <= 2  # Stop before producing a third page.


@pytest.mark.parametrize("workers", [0, 3, 8, True, 1.5])
def test_scene_delivery_rejects_invalid_encoder_workers(bundle, tmp_path, workers):
    from src.surfel_glb import export_scene_glb

    output = tmp_path / "invalid-workers.glb"
    with pytest.raises(ValueError, match="encoder_workers"):
        export_scene_glb(bundle, output, encoder_workers=workers)
    assert not output.exists()


def test_parallel_writer_failure_waits_for_running_encoder(
    three_page_bundle, tmp_path, monkeypatch
):
    from src import surfel_glb_texture
    from src.surfel_glb import export_scene_glb
    from src.surfel_glb_compact import _CompactWriter

    bundle, state = three_page_bundle
    running, release, writer_failed, returned = Event(), Event(), Event(), Event()
    failure = RuntimeError("geometry write failed")
    errors, finished = [], []
    original_encode = surfel_glb_texture.encode_atlas

    def encode(rgba, colors):
        running.set()
        assert release.wait(5), "test did not release encoder"
        try:
            return original_encode(rgba, colors=colors)
        finally:
            finished.append(True)

    def geometry(*args):
        assert running.wait(5), "encoder did not start"
        writer_failed.set()
        raise failure

    output = tmp_path / "drain.glb"

    def export():
        try:
            export_scene_glb(bundle, output, lod=False, encoder_workers=2)
        except BaseException as error:
            errors.append(error)
        finally:
            returned.set()

    monkeypatch.setattr(surfel_glb_texture, "encode_atlas", encode)
    monkeypatch.setattr(_CompactWriter, "geometry", geometry)
    thread = Thread(target=export)
    thread.start()
    try:
        assert writer_failed.wait(5), "writer did not fail"
        assert not returned.is_set()  # shutdown must drain the blocked running encode.
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert errors == [failure]
    assert finished
    assert state["produced"] == 2
    assert not output.exists()
