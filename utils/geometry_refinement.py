"""Small CPU bundle adjustment for camera poses and per-frame depth scales.

The optimizer consumes measured cross-frame pixel correspondences.  It never
manufactures correspondences by reprojecting the input prediction itself.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


_ROTATION_BOUND_RAD = np.deg2rad(5.0)
_TRANSLATION_DEPTH_FRACTION = 0.10
_SCALE_BOUND = 1.10
_PIXEL_ACCEPT_TOLERANCE = 1.01
_DEPTH_ACCEPT_TOLERANCE = 1.05
_DEPTH_ACCEPT_ABSOLUTE_TOLERANCE = 1e-4
_DEPTH_EDGE_RELATIVE_RANGE = 0.05
_HELDOUT_PIXEL_RMSE_MAX = 3.0
_HELDOUT_DEPTH_RELATIVE_RMSE_MAX = 0.05


def _bilinear_valid(
    depth: np.ndarray, xy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample valid, locally smooth 2x2 footprints without crossing depth edges."""
    h, w = depth.shape
    x, y = xy[:, 0], xy[:, 1]
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (y >= 0)
        & (x < w - 1)
        & (y < h - 1)
    )
    values = np.full(len(xy), np.nan, dtype=np.float64)
    candidates = np.flatnonzero(inside)
    if not len(candidates):
        empty = np.zeros(len(xy), dtype=bool)
        return values, empty, empty
    xc, yc = x[candidates], y[candidates]
    x0, y0 = np.floor(xc).astype(int), np.floor(yc).astype(int)
    neighbourhood = np.stack(
        (depth[y0, x0], depth[y0, x0 + 1], depth[y0 + 1, x0], depth[y0 + 1, x0 + 1]),
        axis=1,
    )
    footprint_valid = np.isfinite(neighbourhood).all(axis=1) & (neighbourhood > 0).all(
        axis=1
    )
    local_median = np.median(neighbourhood, axis=1)
    relative_range = np.ptp(neighbourhood, axis=1) / np.maximum(local_median, 1e-12)
    smooth_local = footprint_valid & (relative_range <= _DEPTH_EDGE_RELATIVE_RANGE)
    edge_rejected = np.zeros(len(xy), dtype=bool)
    edge_rejected[candidates] = footprint_valid & ~smooth_local
    valid_local = smooth_local
    good = candidates[valid_local]
    if len(good):
        dx = x[good] - np.floor(x[good])
        dy = y[good] - np.floor(y[good])
        v = neighbourhood[valid_local]
        values[good] = (
            v[:, 0] * (1 - dx) * (1 - dy)
            + v[:, 1] * dx * (1 - dy)
            + v[:, 2] * (1 - dx) * dy
            + v[:, 3] * dx * dy
        )
    valid = np.isfinite(values) & (values > 0)
    return values, valid, edge_rejected


def _validate_inputs(depth, intrinsics, extrinsics, correspondences):
    depth = np.asarray(depth, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if depth.ndim != 3:
        raise ValueError(f"depth must have shape (N,H,W), got {depth.shape}")
    n = depth.shape[0]
    if intrinsics.shape != (n, 3, 3) or extrinsics.shape != (n, 3, 4):
        raise ValueError("intrinsics/extrinsics must have shapes (N,3,3)/(N,3,4)")
    if not (np.isfinite(intrinsics).all() and np.isfinite(extrinsics).all()):
        raise ValueError("camera matrices must be finite")
    required = ("frame_i", "frame_j", "xy_i", "xy_j")
    if any(key not in correspondences for key in required):
        raise ValueError(f"correspondences must contain {required}")
    fi_input = np.asarray(correspondences["frame_i"])
    fj_input = np.asarray(correspondences["frame_j"])
    if (
        fi_input.ndim != 1
        or fj_input.ndim != 1
        or not np.issubdtype(fi_input.dtype, np.integer)
        or not np.issubdtype(fj_input.dtype, np.integer)
    ):
        raise ValueError("frame_i/frame_j must be one-dimensional integer arrays")
    fi = fi_input.astype(np.int64, copy=False)
    fj = fj_input.astype(np.int64, copy=False)
    xi = np.asarray(correspondences["xy_i"], dtype=np.float64)
    xj = np.asarray(correspondences["xy_j"], dtype=np.float64)
    m = len(fi)
    if fj.shape != (m,) or xi.shape != (m, 2) or xj.shape != (m, 2):
        raise ValueError("correspondence arrays have inconsistent shapes")
    if m == 0 or (fi < 0).any() or (fj < 0).any() or (fi >= n).any() or (fj >= n).any():
        raise ValueError("correspondences are empty or contain invalid frame indices")
    if (fi == fj).any() or not (np.isfinite(xi).all() and np.isfinite(xj).all()):
        raise ValueError("correspondences must be finite and cross-frame")
    return depth, intrinsics, extrinsics, fi, fj, xi, xj


def _connected(n: int, fi: np.ndarray, fj: np.ndarray) -> bool:
    seen, stack = {0}, [0]
    adjacency = [set() for _ in range(n)]
    for a, b in zip(fi, fj):
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    while stack:
        for neighbour in adjacency[stack.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return len(seen) == n


def _split_indices(fi, fj, seed):
    """Hold out approximately 20%, while keeping every observed edge in training."""
    rng = np.random.default_rng(seed)
    heldout = np.zeros(len(fi), dtype=bool)
    pairs = np.stack((np.minimum(fi, fj), np.maximum(fi, fj)), axis=1)
    for pair in np.unique(pairs, axis=0):
        indices = np.flatnonzero((pairs == pair).all(axis=1))
        shuffled = rng.permutation(indices)
        count = min(int(round(0.2 * len(indices))), max(0, len(indices) - 1))
        heldout[shuffled[:count]] = True
    return np.flatnonzero(~heldout), np.flatnonzero(heldout)


def refine_geometry(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    correspondences: Dict[str, np.ndarray],
    *,
    mode: str = "pose",
    max_nfev: int = 60,
    seed: int = 42,
):
    """Refine w2c poses and optionally one positive depth scale per frame.

    Camera 0 and depth scale 0 define the gauge and remain fixed.  Acceptance
    requires the input valid-depth mask to remain finite and positive, held-out
    pixel RMSE no worse than 1%, and held-out relative-depth RMSE no worse than
    5% plus an absolute 1e-4 numerical tolerance.  It also requires absolute
    held-out pixel/depth RMSE at most 3 px/0.05.  Original zero/NaN invalid
    depths remain invalid.  The result is never reverted automatically.
    """
    if mode not in {"pose", "pose-scale"}:
        raise ValueError("mode must be 'pose' or 'pose-scale'")
    if max_nfev < 1:
        raise ValueError("max_nfev must be positive")
    depth, k, e0, fi, fj, xi, xj = _validate_inputs(
        depth, intrinsics, extrinsics, correspondences
    )
    n = len(depth)
    di = np.empty(len(fi))
    dj = np.empty(len(fi))
    valid = np.ones(len(fi), dtype=bool)
    edge_rejected = np.zeros(len(fi), dtype=bool)
    for frame in range(n):
        mask_i, mask_j = fi == frame, fj == frame
        if mask_i.any():
            di[mask_i], ok, edge = _bilinear_valid(depth[frame], xi[mask_i])
            valid[mask_i] &= ok
            edge_rejected[mask_i] |= edge
        if mask_j.any():
            dj[mask_j], ok, edge = _bilinear_valid(depth[frame], xj[mask_j])
            valid[mask_j] &= ok
            edge_rejected[mask_j] |= edge
    input_count = len(valid)
    rejected_edge_count = int(edge_rejected.sum())
    rejected_other_count = int((~valid & ~edge_rejected).sum())
    fi, fj, xi, xj, di, dj = (array[valid] for array in (fi, fj, xi, xj, di, dj))
    if len(fi) < max(n - 1, 2) or not _connected(n, fi, fj):
        raise ValueError("valid depth correspondences do not connect all cameras")
    train, heldout = _split_indices(fi, fj, seed)
    if not _connected(n, fi[train], fj[train]):
        raise ValueError("training correspondences do not connect all cameras")

    scene_depth = float(np.median(np.concatenate((di, dj))))
    translation_bound = max(scene_depth * _TRANSLATION_DEPTH_FRACTION, 1e-6)
    pose_size = 6 * (n - 1)
    scale_size = n - 1 if mode == "pose-scale" else 0
    x0 = np.zeros(pose_size + scale_size, dtype=np.float64)
    lower = np.r_[
        np.tile([-_ROTATION_BOUND_RAD] * 3 + [-translation_bound] * 3, n - 1),
        np.full(scale_size, -np.log(_SCALE_BOUND)),
    ]
    upper = -lower

    inv_k = np.linalg.inv(k)

    def unpack(parameters):
        refined = e0.copy()
        for frame in range(1, n):
            offset = 6 * (frame - 1)
            rd = Rotation.from_rotvec(parameters[offset : offset + 3]).as_matrix()
            refined[frame, :, :3] = rd @ e0[frame, :, :3]
            refined[frame, :, 3] = (
                rd @ e0[frame, :, 3] + parameters[offset + 3 : offset + 6]
            )
        scales = np.ones(n)
        if scale_size:
            scales[1:] = np.exp(parameters[pose_size:])
        return refined, scales

    def backproject(frame, xy, z, refined):
        homogeneous = np.c_[xy, np.ones(len(xy))]
        rays = np.einsum("mij,mj->mi", inv_k[frame], homogeneous)
        camera = rays * z[:, None]
        r, t = refined[frame, :, :3], refined[frame, :, 3]
        return np.einsum("mj,mjk->mk", camera - t, r)

    def directed(
        source,
        target,
        source_xy,
        target_xy,
        source_depth,
        target_depth,
        refined,
        scales,
    ):
        world = backproject(source, source_xy, source_depth * scales[source], refined)
        camera = (
            np.einsum("mij,mj->mi", refined[target, :, :3], world)
            + refined[target, :, 3]
        )
        z = camera[:, 2]
        safe_z = np.maximum(z, scene_depth * 1e-6)
        projected_h = np.einsum("mij,mj->mi", k[target], camera)
        pixel = projected_h[:, :2] / safe_z[:, None]
        reprojection = pixel - target_xy
        depth_error = (z - target_depth * scales[target]) / scene_depth
        behind = np.minimum(z / scene_depth, 0.0)
        return reprojection, depth_error, behind

    def data_terms(parameters, indices):
        refined, scales = unpack(parameters)
        a = directed(
            fi[indices],
            fj[indices],
            xi[indices],
            xj[indices],
            di[indices],
            dj[indices],
            refined,
            scales,
        )
        b = directed(
            fj[indices],
            fi[indices],
            xj[indices],
            xi[indices],
            dj[indices],
            di[indices],
            refined,
            scales,
        )
        return a, b

    def residual(parameters):
        a, b = data_terms(parameters, train)
        # Pixel units dominate reprojection; depth and cheirality receive explicit weights.
        data = np.r_[
            a[0].ravel(), b[0].ravel(), 5.0 * a[1], 5.0 * b[1], 20.0 * a[2], 20.0 * b[2]
        ]
        pose_prior = parameters[:pose_size].reshape(-1, 6).copy()
        pose_prior[:, :3] /= _ROTATION_BOUND_RAD
        pose_prior[:, 3:] /= translation_bound
        prior = 0.25 * pose_prior.ravel()
        scale_prior = 0.25 * parameters[pose_size:] / np.log(_SCALE_BOUND)
        return np.r_[data, prior, scale_prior]

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=max_nfev,
    )
    refined_e, scales = unpack(result.x)
    refined_depth = depth * scales[:, None, None]

    def metrics(parameters, indices):
        if not len(indices):
            return {
                "count": 0,
                "pixel_rmse": None,
                "pixel_euclidean_median": None,
                "pixel_euclidean_p90": None,
                "depth_relative_rmse": None,
                "positive_fraction": None,
            }
        a, b = data_terms(parameters, indices)
        pixels = np.r_[a[0].ravel(), b[0].ravel()]
        pixel_euclidean = np.r_[
            np.linalg.norm(a[0], axis=1), np.linalg.norm(b[0], axis=1)
        ]
        depths = np.r_[a[1], b[1]]
        positive = np.r_[a[2] == 0, b[2] == 0]
        return {
            "count": int(len(indices)),
            "pixel_rmse": float(np.sqrt(np.mean(pixels**2))),
            "pixel_euclidean_median": float(np.median(pixel_euclidean)),
            "pixel_euclidean_p90": float(np.percentile(pixel_euclidean, 90)),
            "depth_relative_rmse": float(np.sqrt(np.mean(depths**2))),
            "positive_fraction": float(positive.mean()),
        }

    before_train, after_train = metrics(x0, train), metrics(result.x, train)
    before_holdout, after_holdout = metrics(x0, heldout), metrics(result.x, heldout)
    original_valid_depth = np.isfinite(depth) & (depth > 0)
    refined_valid_depth = np.isfinite(refined_depth) & (refined_depth > 0)
    invalid_mask_preserved = bool(
        np.array_equal(original_valid_depth, refined_valid_depth)
    )
    valid_depth_finite_positive = bool(
        np.isfinite(refined_e).all()
        and np.isfinite(refined_depth[original_valid_depth]).all()
        and (refined_depth[original_valid_depth] > 0).all()
    )
    has_holdout = after_holdout["count"] > 0
    checks = {
        "optimizer_converged": bool(result.success),
        "valid_depth_finite_positive": valid_depth_finite_positive,
        "invalid_depth_mask_preserved": invalid_mask_preserved,
        "heldout_available": has_holdout,
    }
    if has_holdout:
        checks.update(
            {
                "heldout_positive_depth": after_holdout["positive_fraction"] == 1.0,
                "heldout_pixel_relative": after_holdout["pixel_rmse"]
                <= before_holdout["pixel_rmse"] * _PIXEL_ACCEPT_TOLERANCE,
                "heldout_depth_relative": after_holdout["depth_relative_rmse"]
                <= before_holdout["depth_relative_rmse"] * _DEPTH_ACCEPT_TOLERANCE
                + _DEPTH_ACCEPT_ABSOLUTE_TOLERANCE,
                "heldout_pixel_absolute": after_holdout["pixel_rmse"]
                <= _HELDOUT_PIXEL_RMSE_MAX,
                "heldout_depth_absolute": after_holdout["depth_relative_rmse"]
                <= _HELDOUT_DEPTH_RELATIVE_RMSE_MAX,
            }
        )
    accepted = bool(all(checks.values()))
    rejection_reasons = [name for name, passed in checks.items() if not passed]
    pose_parameters = result.x[:pose_size].reshape(-1, 6) if n > 1 else np.empty((0, 6))
    rotation_deg = np.r_[
        0.0, np.linalg.norm(pose_parameters[:, :3], axis=1) * 180 / np.pi
    ]
    translation = np.r_[0.0, np.linalg.norm(pose_parameters[:, 3:], axis=1)]
    report = {
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "acceptance_checks": checks,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "mode": mode,
        "metrics": {
            "train": {"before": before_train, "after": after_train},
            "heldout": {"before": before_holdout, "after": after_holdout},
        },
        "metric_definitions": {
            "pixel_rmse": "RMSE over individual x/y reprojection residual components in pixels",
            "pixel_euclidean_median": "median Euclidean reprojection distance per directed observation in pixels",
            "pixel_euclidean_p90": "90th percentile Euclidean reprojection distance per directed observation in pixels",
            "depth_relative_rmse": "RMSE of target-depth residual normalized by scene median depth",
        },
        "constraint_counts": {
            "input": int(input_count),
            "valid": int(len(fi)),
            "rejected_depth_edge": rejected_edge_count,
            "rejected_invalid_or_boundary": rejected_other_count,
            "train": int(len(train)),
            "heldout": int(len(heldout)),
        },
        "correction": {
            "rotation_deg": rotation_deg.tolist(),
            "translation": translation.tolist(),
            "depth_scale": scales.tolist(),
            "max_rotation_deg": float(rotation_deg.max()),
            "max_translation": float(translation.max()),
        },
        "validity": {
            "valid_depth_finite_positive": valid_depth_finite_positive,
            "invalid_depth_mask_preserved": invalid_mask_preserved,
            "input_invalid_depth_count": int((~original_valid_depth).sum()),
            "output_all_finite": bool(np.isfinite(refined_depth).all()),
            "heldout_available": has_holdout,
        },
        "thresholds": {
            "rotation_component_bound_deg": 5.0,
            "translation_component_bound_depth_fraction": 0.10,
            "depth_edge_relative_range_max": _DEPTH_EDGE_RELATIVE_RANGE,
            "depth_scale_range": [1 / _SCALE_BOUND, _SCALE_BOUND],
            "heldout_pixel_ratio_max": _PIXEL_ACCEPT_TOLERANCE,
            "heldout_depth_ratio_max": _DEPTH_ACCEPT_TOLERANCE,
            "heldout_depth_absolute_tolerance": _DEPTH_ACCEPT_ABSOLUTE_TOLERANCE,
            "heldout_pixel_rmse_absolute_max": _HELDOUT_PIXEL_RMSE_MAX,
            "heldout_depth_relative_rmse_absolute_max": _HELDOUT_DEPTH_RELATIVE_RMSE_MAX,
        },
        "split": {
            "seed": int(seed),
            "heldout_target_fraction": 0.20,
            "strategy": "per-frame-pair",
        },
    }
    return refined_depth, refined_e, report
