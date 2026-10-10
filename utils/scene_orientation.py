"""Shared scene orientation, retaining the Viewer plane and its cache identity."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import tempfile

import numpy as np

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1
ALGORITHM_VERSION = "viewer-plane-v1"


@dataclass(frozen=True)
class SceneOrientation:
    status: str
    rotation: np.ndarray
    normal_world: np.ndarray | None
    plane_offset_m: float | None
    diagnostics: dict


_PLANE_SUBSAMPLE = 200_000
_PLANE_MIN_INLIER_RATIO = 0.05
_PLANE_MAX_TILT_DEG = 60.0
_PLANE_MAX_CANDIDATES = 8
_PLANE_MAX_BELOW_RATIO = 0.15


def fit_scene_orientation(
    valid_points: np.ndarray, extrinsic: np.ndarray
) -> SceneOrientation:
    """Fit a floor plane from unfiltered points and map it into viewer Y-up."""
    if len(valid_points) < 3:
        raise ValueError("DA3 cache has too few points to orient the scene")
    try:
        import open3d as o3d
    except ImportError as error:
        raise ImportError("Open3D required: pip install open3d") from error

    rotation_w2c = extrinsic[:, :, :3]
    translation_w2c = extrinsic[:, :, 3]
    camera_centers = -np.einsum("nji,nj->ni", rotation_w2c, translation_w2c)
    camera_mean = camera_centers.mean(axis=0)
    subsample = (
        valid_points
        if len(valid_points) <= _PLANE_SUBSAMPLE
        else valid_points[
            np.linspace(0, len(valid_points) - 1, _PLANE_SUBSAMPLE).astype(np.int64)
        ]
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(subsample)
    median_nn = float(np.median(pcd.compute_nearest_neighbor_distance()))
    distance_threshold = float(np.clip(median_nn * 3.0, 0.02, 0.15))
    min_inliers = _PLANE_MIN_INLIER_RATIO * len(subsample)
    m_flip = np.diag([1.0, -1.0, -1.0, 1.0])
    remaining = pcd
    for _ in range(_PLANE_MAX_CANDIDATES):
        if len(remaining.points) < max(min_inliers, 3):
            break
        plane, inliers = remaining.segment_plane(
            distance_threshold, ransac_n=3, num_iterations=1000
        )
        inliers = np.asarray(inliers)
        plane_coeffs = np.asarray(plane[:4], dtype=np.float64)
        norm = float(np.linalg.norm(plane_coeffs[:3]))
        if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
            logger.warning("RANSAC returned a degenerate plane; skipping candidate")
            remaining = remaining.select_by_index(inliers, invert=True)
            continue
        normal = plane_coeffs[:3] / norm
        plane_d = plane_coeffs[3] / norm
        if normal @ (camera_mean - subsample.mean(axis=0)) < 0:
            normal = -normal
            plane_d = -plane_d
        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -normal[1]))))
        below_ratio = float(np.mean(subsample @ normal + plane_d < -0.1))
        if (
            len(inliers) >= min_inliers
            and tilt_deg <= _PLANE_MAX_TILT_DEG
            and below_ratio <= _PLANE_MAX_BELOW_RATIO
        ):
            r_level = _shortest_arc_to_y_up(m_flip[:3, :3] @ normal)
            return SceneOrientation(
                "fitted",
                (r_level @ m_flip)[:3, :3],
                normal,
                float(plane_d),
                {
                    "distance_threshold_m": distance_threshold,
                    "inlier_count": int(len(inliers)),
                    "sample_count": int(len(subsample)),
                    "tilt_deg": tilt_deg,
                    "below_ratio": below_ratio,
                },
            )
        remaining = remaining.select_by_index(inliers, invert=True)
    return SceneOrientation(
        "not_found",
        m_flip[:3, :3].copy(),
        None,
        None,
        {
            "distance_threshold_m": distance_threshold,
            "sample_count": int(len(subsample)),
        },
    )


def _shortest_arc_to_y_up(normal: np.ndarray) -> np.ndarray:
    """Return the homogeneous shortest-arc rotation from ``normal`` to +Y."""
    target = np.asarray([0.0, 1.0, 0.0])
    axis = np.cross(normal, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.clip(normal @ target, -1.0, 1.0))
    rotation = np.eye(4)
    if sine < 1e-9:
        if cosine > 0:
            return rotation
        rotation[:3, :3] = np.diag([1.0, -1.0, -1.0])
        return rotation
    axis /= sine
    angle = math.atan2(sine, cosine)
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation[:3, :3] = (
        np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    )
    return rotation


def _cache_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "resolved_path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _validate_result(result: SceneOrientation) -> None:
    rotation = np.asarray(result.rotation)
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-7, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7, rtol=0)
    ):
        raise ValueError("scene orientation rotation must be proper orthonormal")
    if not isinstance(result.diagnostics, dict):
        raise ValueError("scene orientation diagnostics must be an object")
    if result.status == "fitted":
        normal = np.asarray(result.normal_world)
        if (
            normal.shape != (3,)
            or not np.isfinite(normal).all()
            or not np.isclose(np.linalg.norm(normal), 1, atol=1e-7, rtol=0)
            or not np.allclose(rotation @ normal, [0, 1, 0], atol=1e-7, rtol=0)
            or not isinstance(result.plane_offset_m, (float, int))
            or not np.isfinite(result.plane_offset_m)
        ):
            raise ValueError("scene orientation fitted plane is invalid")
    elif result.status == "not_found":
        if result.normal_world is not None or result.plane_offset_m is not None:
            raise ValueError("not_found scene orientation cannot contain a plane")
    else:
        raise ValueError("scene orientation status is invalid")


def _read_sidecar(path: Path) -> tuple[SceneOrientation, dict]:
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or set(data) != {
            "schema_version",
            "algorithm_version",
            "cache_identity",
            "status",
            "rotation",
            "normal_world",
            "plane_offset_m",
            "diagnostics",
        }:
            raise ValueError("invalid scene orientation sidecar fields")
        identity = data["cache_identity"]
        if (
            not isinstance(identity, dict)
            or set(identity)
            != {"resolved_path", "size", "mtime_ns", "ctime_ns", "device", "inode"}
            or not isinstance(identity["resolved_path"], str)
            or any(
                type(identity[key]) is not int
                for key in identity
                if key != "resolved_path"
            )
            or type(data["schema_version"]) is not int
            or not isinstance(data["algorithm_version"], str)
        ):
            raise ValueError("invalid scene orientation cache identity or version")
        result = SceneOrientation(
            data["status"],
            np.asarray(data["rotation"], dtype=float),
            (
                None
                if data["normal_world"] is None
                else np.asarray(data["normal_world"], dtype=float)
            ),
            data["plane_offset_m"],
            data["diagnostics"],
        )
        _validate_result(result)
        # JSON diagnostics must also be finite even though they are not acceptance gates.
        json.dumps(data, allow_nan=False)
        return result, data
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(
            f"malformed scene orientation sidecar: {path}: {error}"
        ) from error


def get_scene_orientation(cache_path: Path) -> tuple[SceneOrientation, str]:
    """Load or compute orientation tied to the current NPZ file stat identity.

    The fitted plane follows ``normal_world @ point + plane_offset_m = 0``.
    Viewer centering translations are deliberately not part of this result.
    """
    cache_path = Path(cache_path).resolve()
    before = _cache_identity(cache_path)
    sidecar = cache_path.parent / "scene_orientation.json"
    event = "computed"
    if sidecar.exists():
        result, data = _read_sidecar(sidecar)
        if (
            data["schema_version"] == SCHEMA_VERSION
            and data["algorithm_version"] == ALGORITHM_VERSION
            and data["cache_identity"] == before
        ):
            if _cache_identity(cache_path) != before:
                raise ValueError("DA3 cache changed while loading scene orientation")
            return result, "hit"
        event = "stale_recomputed"
    with np.load(cache_path, allow_pickle=False) as cache:
        points = np.asarray(cache["world_points"], dtype=float)
        confidence = np.asarray(cache["world_points_conf"], dtype=float)
        extrinsic = np.asarray(cache["extrinsic"], dtype=float)
    if (
        points.ndim != 4
        or points.shape[-1] != 3
        or confidence.shape != points.shape[:-1]
        or extrinsic.shape != (points.shape[0], 3, 4)
        or not np.isfinite(extrinsic).all()
    ):
        raise ValueError("invalid DA3 geometry for scene orientation")
    valid = (
        np.isfinite(points).all(axis=-1)
        & np.any(points != 0, axis=-1)
        & np.isfinite(confidence)
    )
    result = fit_scene_orientation(points[valid], extrinsic)
    _validate_result(result)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "cache_identity": before,
        "status": result.status,
        "rotation": result.rotation.tolist(),
        "normal_world": (
            None if result.normal_world is None else result.normal_world.tolist()
        ),
        "plane_offset_m": result.plane_offset_m,
        "diagnostics": result.diagnostics,
    }
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(
        prefix=".scene_orientation.", suffix=".tmp", dir=cache_path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(encoded)
        if _cache_identity(cache_path) != before:
            raise ValueError("DA3 cache changed while computing scene orientation")
        os.replace(temporary, sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return result, event
