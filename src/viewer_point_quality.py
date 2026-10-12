"""Quality rules for Viewer surfels on the original camera grid."""
import numpy as np

from src.surfel_sampling import _depth_breaks


# Fixed A/B rules. Keep these thresholds unchanged while comparing export arms.
CONFIDENCE_LOW_PERCENTILE = 20
CONFIDENCE_EDGE_PERCENTILE = 40
GRAZING_COSINE = 0.15
SLOPE_DEPTH_FRACTION = 0.005
SLOPE_SPACING_MULTIPLIER = 2


def camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    """w2c (F,3,4) or (F,4,4) -> camera centers (F,3) = -R^T t."""
    extrinsic = np.asarray(extrinsic)
    if extrinsic.ndim != 3 or extrinsic.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError("extrinsic must have shape (F,3,4) or (F,4,4)")
    if not np.isfinite(extrinsic).all():
        raise ValueError("extrinsic must be finite")
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return -np.einsum("fji,fj->fi", rotation, translation)


def _mid_slope(depth, valid, spacing, axis):
    left = np.roll(depth, 1, axis=axis)
    right = np.roll(depth, -1, axis=axis)
    eligible = valid & np.roll(valid, 1, axis=axis) & np.roll(valid, -1, axis=axis)
    boundary = [slice(None)] * depth.ndim
    boundary[axis] = 0
    eligible[tuple(boundary)] = False
    boundary[axis] = -1
    eligible[tuple(boundary)] = False
    dl = depth - left
    dr = right - depth
    tolerance = np.maximum(SLOPE_SPACING_MULTIPLIER * spacing,
                           SLOPE_DEPTH_FRACTION * depth)
    return eligible & (np.abs(dl) > tolerance) & (np.abs(dr) > tolerance) & (np.sign(dl) == np.sign(dr))


def pixel_quality_drop(*, depth, valid, confidence, u, v, points, centers, spacing) -> tuple[np.ndarray, dict[str, int]]:
    """(F,H,W) bool: pixels a viewer should not draw; counts per rule."""
    depth = np.asarray(depth)
    valid = np.asarray(valid, dtype=bool)
    confidence = np.asarray(confidence)
    u = np.asarray(u)
    v = np.asarray(v)
    points = np.asarray(points)
    centers = np.asarray(centers)
    if (depth.ndim != 3 or not all(depth.shape) or valid.shape != depth.shape
            or confidence.shape != depth.shape or u.shape != (*depth.shape, 3)
            or v.shape != u.shape or points.shape != u.shape
            or centers.shape != (depth.shape[0], 3)):
        raise ValueError("quality arrays must align with the camera grid")
    if not np.isfinite(spacing) or spacing < 0:
        raise ValueError("spacing must be finite and nonnegative")
    if (np.any(valid & (~np.isfinite(depth) | (depth <= 0) | ~np.isfinite(confidence)))
            or not np.isfinite(points[valid]).all() or not np.isfinite(u[valid]).all()
            or not np.isfinite(v[valid]).all() or not np.isfinite(centers).all()):
        raise ValueError("valid quality observations must be finite with positive depth")
    if not valid.any():
        return np.zeros(depth.shape, dtype=bool), dict(low_confidence=0, edge_confidence=0,
                                                        grazing=0, mid_slope=0, total=0)

    conf_p20, conf_p40 = np.percentile(
        confidence[valid], (CONFIDENCE_LOW_PERCENTILE, CONFIDENCE_EDGE_PERCENTILE)
    )
    edge = np.zeros(depth.shape, dtype=bool)
    for frame in range(depth.shape[0]):
        bx = _depth_breaks(depth[frame], valid[frame], spacing, 1)
        by = _depth_breaks(depth[frame], valid[frame], spacing, 0)
        edge[frame, :, :-1] |= bx
        edge[frame, :, 1:] |= bx
        edge[frame, :-1, :] |= by
        edge[frame, 1:, :] |= by

    normal = np.cross(u, v)
    ray = points - centers[:, None, None, :]
    normal_length = np.linalg.norm(normal, axis=-1)
    ray_length = np.linalg.norm(ray, axis=-1)
    cosine = np.divide(np.abs(np.sum(normal * ray, axis=-1)),
                       normal_length * ray_length,
                       out=np.ones(depth.shape, dtype=np.float64),
                       where=(normal_length > 0) & (ray_length > 0))
    low_confidence = valid & (confidence < conf_p20)
    edge_confidence = valid & edge & (confidence < conf_p40)
    grazing = valid & (normal_length > 0) & (ray_length > 0) & (cosine < GRAZING_COSINE)
    mid_slope = _mid_slope(depth, valid, spacing, 2) | _mid_slope(depth, valid, spacing, 1)
    drop = low_confidence | edge_confidence | grazing | mid_slope
    counts = {"low_confidence": int(low_confidence.sum()),
              "edge_confidence": int(edge_confidence.sum()),
              "grazing": int(grazing.sum()),
              "mid_slope": int(mid_slope.sum()), "total": int(drop.sum())}
    return drop, counts


def viewer_quality_keep(*, indices, labels, drop, min_label_retention=0.5) -> np.ndarray:
    """Keep mask aligned with flat grid ``indices`` and product ``labels``."""
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    drop = np.asarray(drop, dtype=bool)
    if (drop.ndim != 3 or indices.ndim != 1 or labels.shape != indices.shape
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= drop.size)):
        raise ValueError("indices and labels must align with the drop grid")
    if not np.isfinite(min_label_retention) or not 0 <= min_label_retention <= 1:
        raise ValueError("min_label_retention must be in [0, 1]")
    keep = ~drop.ravel()[indices]
    for label in np.unique(labels[labels >= 0]):
        group = labels == label
        if keep[group].mean() < min_label_retention:
            keep[group] = True
    return keep


def product_spill_keep(*, indices, labels, depth, ratio=1.10, max_drop_fraction=0.30) -> np.ndarray:
    """Drop product points behind their (label, frame) median camera depth."""
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    depth = np.asarray(depth)
    if (depth.ndim != 3 or indices.ndim != 1 or labels.shape != indices.shape
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= depth.size)):
        raise ValueError("indices and labels must align with the depth grid")
    if not np.isfinite(ratio) or ratio <= 0 or not np.isfinite(max_drop_fraction) or not 0 <= max_drop_fraction <= 1:
        raise ValueError("invalid product spill thresholds")
    keep = np.ones(len(indices), dtype=bool)
    frames = indices // (depth.shape[1] * depth.shape[2])
    values = depth.ravel()[indices]
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("product spill depths must be finite and positive")
    for label in np.unique(labels[labels >= 0]):
        for frame in np.unique(frames[labels == label]):
            group = (labels == label) & (frames == frame)
            spill = values[group] > ratio * np.median(values[group])
            if spill.mean() <= max_drop_fraction:
                keep[group] = ~spill
    return keep
