"""Prepared Stray Scanner inputs and shared colored POINTS GLB export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def load_prepared(base: Path) -> tuple[list[str], dict]:
    sensor = read_npz(base / "camera_inputs.npz")
    frame_ids = sensor["frame_ids"]
    count = len(frame_ids)
    if not count or len(np.unique(frame_ids)) != count:
        raise ValueError("Prepared inputs need nonempty, unique frame IDs")
    if sensor["intrinsics"].shape != (count, 3, 3) or sensor["w2c"].shape != (
        count,
        4,
        4,
    ):
        raise ValueError("Expected per-frame source K and homogeneous Scanner w2c")
    images = [str(base / "frames" / f"{int(frame):06d}.png") for frame in frame_ids]
    for image in images:
        if not Path(image).is_file():
            raise FileNotFoundError(image)
    return images, sensor


def resize_intrinsics(source_k: np.ndarray, source_size, target_size) -> np.ndarray:
    """Use OpenCV resize pixel centers for the official PromptDA RGB loader."""
    sx, sy = np.asarray(target_size) / np.asarray(source_size)
    affine = np.array(
        [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    return affine @ source_k


def prediction_mask(prediction: dict) -> tuple[np.ndarray, float | None]:
    depth = prediction["depth"]
    valid = np.isfinite(depth) & (depth > 0)
    if "conf" not in prediction:
        return valid, None
    conf = prediction["conf"]
    valid &= np.isfinite(conf)
    if not np.any(valid):
        raise ValueError("No valid predicted depth/confidence")
    # Preserve DA3's native adaptive confidence rule used by the accepted GLB.
    p40, p90 = np.percentile(conf[valid], [40, 90])
    threshold = float(min(max(1.05, p40), p90))
    return valid & (conf >= threshold), threshold


def backproject(depth, intrinsics, c2w, images, masks):
    points, colors = [], []
    for d, k, pose, rgb, mask in zip(
        depth, intrinsics, c2w, images, masks, strict=True
    ):
        y, x = np.nonzero(mask & np.isfinite(d) & (d > 0))
        pixels = np.column_stack((x, y, np.ones(len(x)))).astype(np.float32)
        camera = (pixels @ np.linalg.inv(k).T) * d[y, x, None]
        points.append(camera @ pose[:3, :3].T + pose[:3, 3])
        colors.append(rgb[y, x])
    return np.concatenate(points).astype(np.float32), np.concatenate(colors)


def export_prediction(
    prediction: dict,
    sensor: dict,
    output_glb: Path,
    max_points: int = 1_000_000,
    seed: int = 42,
) -> dict:
    import trimesh

    if max_points < 1:
        raise ValueError("max_points must be positive")
    if not np.array_equal(prediction["frame_ids"], sensor["frame_ids"]):
        raise ValueError("Prediction and Scanner frame IDs differ")
    mask, threshold = prediction_mask(prediction)
    # Both selected candidates explicitly retain the supplied Scanner poses.
    points, colors = backproject(
        prediction["depth"],
        prediction["intrinsics"],
        np.linalg.inv(sensor["w2c"]),
        prediction["processed_images"],
        mask,
    )
    before = len(points)
    if not before or not np.isfinite(points).all():
        raise ValueError("Cannot export an empty or nonfinite point cloud")
    if before > max_points:
        indices = np.random.default_rng(seed).choice(before, max_points, replace=False)
        points, colors = points[indices], colors[indices]
    display = np.diag([1, -1, -1, 1]).astype(np.float32) @ sensor["w2c"][0]
    points = points @ display[:3, :3].T + display[:3, 3]
    cloud = trimesh.points.PointCloud(points.astype(np.float32), colors=colors)
    scene = trimesh.Scene()
    scene.add_geometry(cloud, geom_name=output_glb.stem, node_name=output_glb.stem)
    scene.export(str(output_glb))
    # Validate the external binary format once, at the export boundary.
    loaded = trimesh.load(str(output_glb), force="scene", process=False)
    geometries = list(loaded.geometry.values())
    if len(geometries) != 1 or not isinstance(geometries[0], trimesh.points.PointCloud):
        raise ValueError("GLB must contain exactly one PointCloud")
    restored = geometries[0]
    if (
        len(restored.vertices) != len(points)
        or not np.isfinite(restored.vertices).all()
        or len(restored.colors) != len(points)
    ):
        raise ValueError("Invalid GLB point/color readback")
    return {
        "file": output_glb.name,
        "points_before_limit": before,
        "points": len(points),
        "bytes": output_glb.stat().st_size,
        "bounds_m": restored.bounds.tolist(),
        "confidence_threshold": threshold,
        "point_limit": max_points,
        "seed": seed,
        "display_transform": display.tolist(),
        "glb_readback": "PASS",
    }
