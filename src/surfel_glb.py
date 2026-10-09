"""Bake an existing DA3 Viewer Surfel bundle into a portable textured GLB.

Only DA3 source projection is supported; other backends are rejected explicitly.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
import shutil
import struct
import tempfile
import time
import zipfile

import numpy as np


def _asset_name(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Invalid relative asset name")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"Invalid relative asset name: {name}")
    return name


@contextmanager
def _assets(source):
    source = Path(source)
    if source.is_dir():
        root = source.resolve()

        def read(name):
            path = (root / _asset_name(name)).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Asset escapes generation directory")
            return path.read_bytes()

        yield read
    else:
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("ZIP contains duplicate asset names")

            def read(name):
                return archive.read(_asset_name(name))

            yield read


def _json(read, name):
    value = json.loads(read(name))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Invalid {label}")
    return value


def _finite(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid finite shape for {label}; expected {shape}")
    return array


def _array(read, name, dtype, shape):
    raw = read(name)
    if len(raw) != int(np.prod(shape)) * np.dtype(dtype).itemsize:
        raise ValueError(f"Invalid byte length for {name}")
    result = np.frombuffer(raw, dtype=dtype).reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite values in {name}")
    return result


def _load(read, global_id):
    manifest = _json(read, "manifest.json")
    if manifest.get("backend") != "DA3":
        raise ValueError("Portable Surfel GLB export supports only backend DA3")
    if manifest.get("schema_version") != "3.0.0":
        raise ValueError("Expected Viewer schema 3.0.0")
    matrix = _finite(manifest.get("world_to_view"), (16,), "world_to_view").reshape(4, 4)
    objects = _json(read, "objects.json")
    metadata = _json(read, "surfel.json")
    version = metadata.get("version")
    if type(version) is not int or version not in (2, 3):
        raise ValueError("Expected Surfel version 2 or 3")
    count = _integer(metadata.get("point_count"), "point_count", 1)
    grid = metadata.get("grid_size")
    if not isinstance(grid, list) or len(grid) != 2:
        raise ValueError("Invalid grid_size")
    width, height = [_integer(x, "grid_size", 1) for x in grid]
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 256:
        raise ValueError("Expected 1..256 source frames")
    if manifest.get("frame_count") != len(frames):
        raise ValueError("manifest frame_count differs from Surfel frames")
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Invalid source frame")
        _finite(frame.get("extrinsic"), (3, 4), "extrinsic")
        intrinsic = _finite(frame.get("intrinsic"), (3, 3), "intrinsic")
        if np.linalg.det(intrinsic) == 0:
            raise ValueError("Singular intrinsic")
        _finite(frame.get("processed_to_source"), (3, 3), "processed_to_source")
        for key in ("source_size", "texture_size"):
            size = frame.get(key)
            if not isinstance(size, list) or len(size) != 2:
                raise ValueError(f"Invalid {key}")
            for value in size:
                _integer(value, key, 1)
        read(_asset_name(frame.get("texture")))
    positions = _array(read, "positions.f32.bin", "<f4", (count, 3))
    u = _array(read, "surfel-u.f16.bin", "<f2", (count, 3))
    v = _array(read, "surfel-v.f16.bin", "<f2", (count, 3))
    scales = (_array(read, "surfel-scale.f16.bin", "<f2", (count, 2))
              if version == 3 else np.ones((count, 2), dtype=np.float16))
    if np.any((scales < .5) | (scales > 8)):
        raise ValueError("Surfel scales must be within [0.5, 8]")
    frame_indices = _array(read, "surfel-frame.u8.bin", "u1", (count,))
    if np.any(frame_indices >= len(frames)):
        raise ValueError("Surfel frame index out of range")
    depths = _array(read, "surfel-depth.f16.bin", "<f2", (len(frames), height, width))
    if np.any(depths < 0):
        raise ValueError("Negative source depth")
    all_ranges = []
    for entry in objects.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("point_ranges"), list):
            raise ValueError("Invalid object point_ranges")
        for pair in entry["point_ranges"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Invalid object point range")
            start, end = [_integer(x, "point range") for x in pair]
            if not start <= end <= count:
                raise ValueError("Object point range outside point array")
            if start != end:
                all_ranges.append((start, end))
    ordered = sorted(all_ranges)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Object point ranges overlap")
    selection = slice(None)
    if global_id is not None:
        key = str(global_id)
        if key not in objects:
            raise ValueError(f"Unknown global_id: {key}")
        mask = np.zeros(count, dtype=bool)
        for start, end in objects[key]["point_ranges"]:
            mask[start:end] = True
        if not mask.any():
            raise ValueError(f"global_id {key} has no Surfel points")
        selection = mask
    return matrix, dict(positions=positions[selection], u=u[selection], v=v[selection],
                       scales=scales[selection], frame_indices=frame_indices[selection],
                       depths=depths, frames=frames, read_asset=read)


def export_surfel_glb(source, output, *, global_id=None, tile_size=8,
                      atlas_size=2048, radius=2.0, lod=False):
    """Export selected DA3 surfels without overwriting an existing destination."""
    from src.surfel_mesh import bake_surfel_pages

    started = time.monotonic()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    _integer(tile_size, "tile_size", 1)
    _integer(atlas_size, "atlas_size", tile_size + 2)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    if lod and radius != 2.0:
        raise ValueError("Surfel LOD currently requires radius 2.0")
    document = {"asset": {"version": "2.0", "generator": "Surfel portable GLB exporter"},
                "extensionsUsed": ["KHR_materials_unlit"],
                "scene": 0, "scenes": [{"nodes": [0]}],
                "nodes": [], "meshes": [{"primitives": []}],
                "bufferViews": [], "accessors": [], "images": [],
                "textures": [], "materials": [],
                "samplers": [{"magFilter": 9729, "minFilter": 9729,
                              "wrapS": 33071, "wrapT": 33071}]}
    page_count = triangle_count = baked_count = 0
    with _assets(source) as read, tempfile.TemporaryFile() as binary:
        matrix, inputs = _load(read, global_id)
        point_count = len(inputs["positions"])
        lod_report = None
        if lod:
            from src.surfel_lod import select_surfel_lod

            owners = np.full(point_count, -1, dtype=np.int32)
            if global_id is not None:
                owners[:] = 0
            else:
                for owner, entry in enumerate(_json(read, "objects.json").values()):
                    for start, end in entry["point_ranges"]:
                        owners[start:end] = owner
            chosen, lod_report = select_surfel_lod(
                **{key: inputs[key] for key in ("positions", "u", "v", "scales",
                                               "frame_indices", "frames", "depths")},
                object_ids=owners,
            )
            for key in ("positions", "u", "v", "scales", "frame_indices"):
                inputs[key] = inputs[key][chosen]
        selected_count = len(inputs["positions"])
        document["nodes"] = [{"mesh": 0, "matrix": matrix.T.reshape(-1).tolist()}]

        def buffer_view(data, target=None):
            padding = (-binary.tell()) % 4
            binary.write(b"\0" * padding)
            offset = binary.tell()
            binary.write(data)
            view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
            if target is not None:
                view["target"] = target
            document["bufferViews"].append(view)
            return len(document["bufferViews"]) - 1

        def accessor(values, kind, component, target, bounds=False):
            index = len(document["accessors"])
            record = {"bufferView": buffer_view(values.tobytes(), target),
                      "componentType": component, "count": len(values), "type": kind}
            if bounds:
                record.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
            document["accessors"].append(record)
            return index

        for page in bake_surfel_pages(**inputs, radius=radius, tile_size=tile_size,
                                      atlas_size=atlas_size):
            positions = np.asarray(page["positions"], dtype="<f4")
            uv = np.asarray(page["uv"], dtype="<f4")
            indices = np.asarray(page["indices"], dtype="<u4").reshape(-1)
            position_accessor = accessor(positions, "VEC3", 5126, 34962, True)
            uv_accessor = accessor(uv, "VEC2", 5126, 34962)
            index_accessor = accessor(indices, "SCALAR", 5125, 34963)
            image_view = buffer_view(page["image_png"])
            document["images"].append({"bufferView": image_view, "mimeType": "image/png"})
            document["textures"].append({"source": page_count, "sampler": 0})
            document["materials"].append({
                "doubleSided": True, "alphaMode": "MASK", "alphaCutoff": .5,
                "extensions": {"KHR_materials_unlit": {}},
                "pbrMetallicRoughness": {"baseColorTexture": {"index": page_count},
                                         "metallicFactor": 0, "roughnessFactor": 1}})
            document["meshes"][0]["primitives"].append({
                "attributes": {"POSITION": position_accessor, "TEXCOORD_0": uv_accessor},
                "indices": index_accessor, "material": page_count, "mode": 4})
            triangle_count += len(indices) // 3
            baked_count += page["surfel_count"]
            page_count += 1
        if page_count == 0 or baked_count == 0:
            raise ValueError("Selected Surfels have no visible, nondegenerate geometry to export")
        if not 0 <= baked_count <= selected_count:
            raise ValueError("Baker emitted an invalid Surfel count")
        binary.write(b"\0" * ((-binary.tell()) % 4))
        binary_length = binary.tell()
        document["buffers"] = [{"byteLength": binary_length}]
        encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
        encoded += b" " * ((-len(encoded)) % 4)
        length = 12 + 8 + len(encoded) + 8 + binary_length
        if length > 0xFFFFFFFF:
            raise ValueError("GLB exceeds the 4 GiB format size limit; export an explicit global-id subset")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects concurrent writers; only remove our own partial file.
        stream = output.open("xb")
        try:
            with stream:
                stream.write(struct.pack("<4sII", b"glTF", 2, length))
                stream.write(struct.pack("<I4s", len(encoded), b"JSON"))
                stream.write(encoded)
                stream.write(struct.pack("<I4s", binary_length, b"BIN\0"))
                binary.seek(0)
                shutil.copyfileobj(binary, stream)
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    return {"source": str(source), "output": str(output), "global_id": global_id,
            "input_point_count": point_count, "point_count": point_count,
            "selected_point_count": selected_count, "lod": lod_report,
            "lod_removed_point_count": point_count - selected_count,
            "baked_surfel_count": baked_count,
            "skipped_surfel_count": selected_count - baked_count,
            "triangle_count": triangle_count,
            "atlas_pages": page_count, "glb_bytes": length,
            "elapsed_seconds": time.monotonic() - started,
            "tile_size": tile_size, "atlas_size": atlas_size, "radius": radius,
            "appearance": "Static source-texture baking; view-dependent Gaussian blending is not reproduced"}


def export_scene_glb(source, output):
    """Build the compact LOD delivery file from existing Viewer assets on CPU.

    The intermediate full-precision GLB is request-local and always removed.
    No inference or Viewer bundle repacking is needed.
    """
    from src.surfel_glb_compact import compact_surfel_glb
    from src.surfel_glb_texture import compact_png

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="surfel-glb-", dir=output.parent) as work:
        intermediate = Path(work) / "baked.glb"
        bake = export_surfel_glb(source, intermediate, lod=True)
        compact = compact_surfel_glb(intermediate, output, image_transform=compact_png)
    # Do not expose paths to intermediates that have already been removed.
    bake.pop("output")
    compact.pop("source")
    return {"output": str(output), "glb_bytes": output.stat().st_size,
            "elapsed_seconds": time.monotonic() - started,
            "bake": bake, "compact": compact}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--global-id", type=int)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--atlas-size", type=int, default=2048)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--lod", action="store_true",
                        help="Thin covered continuous 2x2 source-pixel interiors")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.report is not None and args.report.exists():
        raise FileExistsError(args.report)
    report = export_surfel_glb(args.source, args.output, global_id=args.global_id,
                              tile_size=args.tile_size, atlas_size=args.atlas_size,
                              radius=args.radius, lod=args.lod)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
