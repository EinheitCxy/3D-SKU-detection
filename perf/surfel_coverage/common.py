"""Read-only inputs shared by the three isolated Surfel coverage trials."""
from pathlib import Path

import numpy as np

from src.surfel_export import grid_tangents

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "runtime/roi-fusion-video3-gid3-v2/shelf-view"
OUT = ROOT / "runtime/surfel-coverage-20260924"
CACHE = ROOT / "runtime/video3-resolution-review/504/outputs/dataset/da3_cache/predictions.npz"
IMAGES = ROOT / "runtime/video3-resolution-review/dataset/images"
EVAL_FRAMES = (0, 4, 9, 14, 19, 24, 28)


def load_sparse():
    with np.load(OLD / "input.npz", allow_pickle=False) as archive:
        return dict(archive)


def load_dense():
    sparse = load_sparse()
    with np.load(CACHE, allow_pickle=False) as archive:
        points = archive["world_points"]
        confidence = archive["depth_conf"]
    u, v, _ = grid_tangents(points, sparse["E"][:, :3])
    valid = sparse["mask"].astype(bool)
    f, h, w = valid.shape
    frame, y, x = np.indices((f, h, w))
    baseline = np.flatnonzero(valid & (y % 4 == 0) & (x % 4 == 0))
    expected = sparse["frame"] * (h * w) + sparse["xy"][:, 1] * w + sparse["xy"][:, 0]
    if not np.array_equal(baseline, expected):
        raise ValueError("Existing baseline is not the declared valid stride-4 selection")
    return dict(points_grid=points, u_grid=u, v_grid=v,
                confidence_grid=confidence, valid=valid, baseline_indices=baseline,
                **{key: sparse[key] for key in ("depth", "K", "E", "affine", "image_paths", "image_ids", "spacing")})


def selected_data(dense, selection):
    indices, scales = np.asarray(selection["indices"]), np.asarray(selection["scales"])
    n = len(dense["baseline_indices"])
    if indices.shape != (n,) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"Selection must have exactly {n} integer indices")
    if len(np.unique(indices)) != n or np.any(indices < 0) or np.any(indices >= dense["valid"].size):
        raise ValueError("Selection has duplicate/out-of-grid samples")
    if not dense["valid"].ravel()[indices].all():
        raise ValueError("Selection contains invalid depth samples")
    if scales.shape != (n, 2) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Selection requires finite positive U/V footprint scales")
    _, h, w = dense["depth"].shape
    frame, remainder = np.divmod(indices, h * w)
    y, x = np.divmod(remainder, w)
    return dict(points=dense["points_grid"].reshape(-1, 3)[indices],
                u=dense["u_grid"].reshape(-1, 3)[indices] * scales[:, :1],
                v=dense["v_grid"].reshape(-1, 3)[indices] * scales[:, 1:],
                confidence=dense["confidence_grid"].ravel()[indices], frame=frame,
                xy=np.column_stack((x, y)), mask=dense["valid"],
                **{key: dense[key] for key in ("depth", "K", "E", "affine", "image_paths", "image_ids", "spacing")})
