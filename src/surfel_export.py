"""Depth-grid oriented surfels, in the exact schema-3 selection slot order."""
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image


def grid_tangents(points, extrinsic):
    """One-pixel tangents; use the shorter one-sided derivative at depth edges.

    Invalid/isolated samples have zero area. A step > 3% source depth is
    prohibited, so footprints cannot join foreground and background surfaces.
    """
    depth = np.einsum("fij,fhwj->fhwi", extrinsic[:, :3, :3], points)[..., 2]
    depth += extrinsic[:, None, None, 2, 3]
    valid = np.isfinite(points).all(-1) & np.any(points != 0, axis=-1) & np.isfinite(depth) & (depth > 0)
    tangents = []
    for axis in (2, 1):
        forward = np.roll(points, -1, axis=axis) - points
        backward = points - np.roll(points, 1, axis=axis)
        f_ok = valid & np.roll(valid, -1, axis=axis)
        b_ok = valid & np.roll(valid, 1, axis=axis)
        edge = [slice(None)] * 3
        edge[axis] = -1
        f_ok[tuple(edge)] = False
        edge[axis] = 0
        b_ok[tuple(edge)] = False
        limit = np.maximum(depth * 0.03, 0.001)
        fl, bl = np.linalg.norm(forward, axis=-1), np.linalg.norm(backward, axis=-1)
        f_ok &= fl < limit
        b_ok &= bl < limit
        use_f = f_ok & (~b_ok | (fl <= bl))
        tangent = np.where(use_f[..., None], forward, backward)
        tangent[~(f_ok | b_ok)] = 0
        tangents.append(tangent.astype("<f4"))
    return *tangents, np.where(valid, depth, 0).astype("<f4")


def prepare_surfels(cache, indices, cache_path: Path, images_dir: Path, texture_edge: int):
    from src.web_viewer_export import _resolve_source_images

    if not 256 <= texture_edge <= 1920:
        raise ValueError("surfel texture longest edge must be in [256, 1920] (1080p cap)")
    with np.load(cache_path, allow_pickle=False) as archive:
        intrinsic = archive["intrinsic"]
    n, h, w, _ = cache["points"].shape
    if not 1 <= n <= 256:
        raise ValueError("Surfel 支持 1..256 个来源帧（Uint8 帧编号 0..255）")
    if intrinsic.shape != (n, 3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("Surfel requires finite per-frame 3x3 intrinsic")
    u, v, depth = grid_tangents(cache["points"], cache["extrinsic"])
    frame_ids = indices // (h * w)
    def half_bytes(values, name):
        if not np.isfinite(values).all() or np.any(np.abs(values) > 65504):
            raise ValueError(f"Surfel {name} exceeds finite Float16 range")
        encoded = values.astype("<f2")
        if name == "depth" and np.any((values > 0) & (encoded == 0)):
            raise ValueError("Surfel positive depth underflows Float16")
        return encoded.tobytes()

    files = {
        "surfel-u.f16.bin": half_bytes(u.reshape(-1, 3)[indices], "u"),
        "surfel-v.f16.bin": half_bytes(v.reshape(-1, 3)[indices], "v"),
        "surfel-frame.u8.bin": frame_ids.astype("u1").tobytes(),
        "surfel-depth.f16.bin": half_bytes(depth, "depth"),
    }
    frames = []
    images = _resolve_source_images(images_dir)
    for f, image_id in enumerate(cache["image_ids"]):
        with Image.open(images[int(image_id)]) as original:
            image = original.convert("RGB")
        source_w, source_h = map(int, cache["source_image_sizes"][f])
        if image.size != (source_w, source_h):
            raise ValueError(f"Source image {image_id} dimensions differ from cache")
        short_edge = min(texture_edge, 1080)
        bounds = (texture_edge, short_edge) if source_w >= source_h else (short_edge, texture_edge)
        image.thumbnail(bounds, Image.Resampling.LANCZOS)
        filename = f"surfel-texture-{f}.jpg"
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=95, subsampling=0)
        files[filename] = stream.getvalue()
        affine = np.eye(3)
        affine[:2] = cache["affine"][f]
        frames.append({
            "image_id": int(image_id), "texture": filename,
            "texture_size": list(image.size), "source_size": [source_w, source_h],
            "processed_to_source": np.linalg.inv(affine).tolist(),
            "intrinsic": intrinsic[f].tolist(), "extrinsic": cache["extrinsic"][f].tolist(),
        })
    files["surfel.json"] = json.dumps({
        "version": 2, "point_count": len(indices), "grid_size": [w, h],
        "texture_longest_edge": texture_edge, "frames": frames,
        "coverage_radius_pixels": 1.05, "source_depth_relative_tolerance": 0.015,
    }, allow_nan=False).encode()
    return files
