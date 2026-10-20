"""Adapt the existing Scanner PromptDA prediction to a metric scene cache.

This CPU-only boundary reuses the saved native depth, processed RGB and trusted
Scanner cameras. It does not load a model, infer confidence or rescale geometry.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from utils.scene_backend import prepare_promptda_rgb, promptda_processed_size, resize_pixel_center_affine


def convert_promptda_cache(
    prediction_path: Path, source_images_dir: Path, output_path: Path,
) -> dict[str, object]:
    """Write schema-v3 geometry, preserving the prediction's declared RGB policy."""
    prediction_path, source_images_dir, output_path = map(
        Path, (prediction_path, source_images_dir, output_path),
    )
    fields = {"depth", "processed_images", "intrinsics", "extrinsics", "frame_ids", "is_metric"}
    with np.load(prediction_path, allow_pickle=False) as archive:
        if "conf" in archive.files or "scale_factor" in archive.files:
            raise ValueError("Selected PromptDA prediction has no native conf/scale_factor; do not pass a DA3 prediction")
        missing = fields - set(archive.files)
        if missing:
            raise ValueError(f"PromptDA prediction missing fields: {sorted(missing)}")
        optional = {"preprocess_method", "preprocess_resolution", "network_image_sizes"} & set(archive.files)
        prediction = {key: archive[key] for key in fields | optional}
    method = prediction.get("preprocess_method", np.array("promptda_resize"))
    if (method.shape != () or method.dtype.kind != "U"
            or method.item() not in {"promptda_resize", "promptda_native_pad14"}):
        raise ValueError("Unknown PromptDA preprocess_method")
    native = method.item() == "promptda_native_pad14"
    resolution = prediction.get("preprocess_resolution")
    if resolution is not None and (resolution.shape != () or resolution.dtype.kind not in "iu" or resolution.item() <= 0):
        raise ValueError("PromptDA preprocess_resolution must be a positive integer scalar")
    if not native:
        resolution = 504 if resolution is None else int(resolution.item())
        if resolution not in {504, 1008}:
            raise ValueError("PromptDA resize preprocess_resolution must be 504 or 1008")
        if resolution != 504 and "preprocess_method" not in prediction:
            raise ValueError("PromptDA non-504 resize requires explicit preprocess_method")
    metric = prediction["is_metric"]
    if metric.shape != () or metric.dtype.kind not in "iu" or metric.item() != 1:
        raise ValueError("PromptDA prediction must declare native metric depth")
    depth = np.asarray(prediction["depth"], dtype=np.float32)
    if depth.ndim != 3 or min(depth.shape) <= 0 or not np.all(np.isfinite(depth) & (depth > 0)):
        raise ValueError("Selected PromptDA depths must be finite positive frame grids")
    count, height, width = depth.shape
    frame_ids = prediction["frame_ids"]
    if (frame_ids.shape != (count,) or frame_ids.dtype.kind not in "iu"
            or len(np.unique(frame_ids)) != count):
        raise ValueError("PromptDA frame IDs must be unique integers aligned with depth")
    rgb = prediction["processed_images"]
    if rgb.shape != (count, height, width, 3) or rgb.dtype != np.uint8:
        raise ValueError("PromptDA processed RGB must be uint8 aligned with depth")
    intrinsic = np.asarray(prediction["intrinsics"], dtype=np.float32)
    extrinsic = np.asarray(prediction["extrinsics"], dtype=np.float32)
    if (intrinsic.shape != (count, 3, 3) or extrinsic.shape != (count, 3, 4)
            or not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all()):
        raise ValueError("PromptDA requires finite per-frame K and 3x4 world-to-camera E")
    if np.any(intrinsic[:, 0, 0] <= 0) or np.any(intrinsic[:, 1, 1] <= 0):
        raise ValueError("PromptDA focal lengths must be positive")
    paths: dict[int, Path] = {}
    for path in source_images_dir.iterdir():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        if not path.stem.isdigit():
            raise ValueError(f"PromptDA source image must have a numeric frame ID: {path.name}")
        image_id = int(path.stem)
        if image_id in paths:
            raise ValueError(f"Duplicate PromptDA source frame ID: {image_id}")
        paths[image_id] = path
    if set(paths) != set(map(int, frame_ids)):
        raise ValueError("PromptDA prediction frame IDs must match the source image set")
    sizes, affines, hashes = [], [], []
    for index, image_id in enumerate(frame_ids):
        path = paths[int(image_id)]
        with Image.open(path) as image:
            source_width, source_height = image.size
            source_rgb = np.asarray(image.convert("RGB"))
        expected_size = (source_width, source_height) if native else promptda_processed_size(source_width, source_height, resolution)
        if expected_size != (width, height):
            raise ValueError(f"PromptDA processed grid differs from the selected RGB policy: {image_id}")
        expected_rgb = source_rgb if native else prepare_promptda_rgb(source_rgb, (width, height))
        if not np.array_equal(expected_rgb, rgb[index]):
            raise ValueError(f"PromptDA source RGB does not match the saved prediction: {image_id}")
        sizes.append((source_width, source_height))
        affines.append(resize_pixel_center_affine(source_width, source_height, width, height))
        # Footprint's existing schema-v3 image binding requires this field.
        with path.open("rb") as stream:
            hashes.append(hashlib.file_digest(stream, "sha256").hexdigest())
    if native:
        source_resolution = max(max(size) for size in sizes)
        if resolution is not None and resolution.item() != source_resolution:
            raise ValueError("PromptDA native preprocess_resolution must equal the source image longest edge")
        resolution = source_resolution
    y, x = np.indices((height, width), dtype=np.float32)
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    world_points = np.empty((count, height, width, 3), dtype=np.float32)
    for index in range(count):
        camera = (pixels @ np.linalg.inv(intrinsic[index]).T) * depth[index, ..., None]
        pose = np.eye(4, dtype=np.float32)
        pose[:3] = extrinsic[index]
        camera_to_world = np.linalg.inv(pose)
        world_points[index] = camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    if not np.isfinite(world_points).all():
        raise ValueError("PromptDA cameras produced nonfinite world coordinates")
    # PromptDA has no learned confidence head. One denotes a valid measurement,
    # not certainty; matching uses a separate >0 validity threshold for this backend.
    validity = np.ones(depth.shape, dtype=np.float32)
    cache = dict(
        depth=depth[..., None], depth_conf=validity, world_points=world_points,
        world_points_conf=validity, images=rgb, intrinsic=intrinsic, extrinsic=extrinsic,
        image_ids=frame_ids.astype(np.int64), source_image_sizes=np.asarray(sizes, dtype=np.int64),
        source_to_processed_affine=np.asarray(affines, dtype=np.float64),
        cache_schema_version=np.array(3, dtype=np.int64), source_model=np.array("promptda/vitl"),
        source_image_sha256=np.asarray(hashes), affine_convention=np.array("pixel_center_v1"),
        preprocess_resolution=np.array(resolution, dtype=np.int64),
        preprocess_method=method.copy(),
        is_metric=np.array(1, dtype=np.int64), scale_factor=np.array(1., dtype=np.float64),
        confidence_kind=np.array("validity"),
    )
    if "network_image_sizes" in prediction:
        network = prediction["network_image_sizes"]
        expected_network = np.tile((np.asarray((width, height), dtype=np.int64) + 13) // 14 * 14, (count, 1))
        if network.shape != (count, 2) or network.dtype.kind not in "iu" or not np.array_equal(network, expected_network):
            raise ValueError("PromptDA network_image_sizes must match the declared preprocessing policy")
        cache["network_image_sizes"] = network.astype(np.int64, copy=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        try:
            np.savez(stream, **cache)
        except BaseException:
            output_path.unlink()
            raise
    return {"backend": "PromptDA", "cache_path": str(output_path), "frame_count": count,
            "grid_size": [width, height], "confidence_kind": "validity"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--source-images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(convert_promptda_cache(args.prediction, args.source_images_dir, args.output))


if __name__ == "__main__":
    main()
