"""Choose one source frame for each local Viewer Surfel surface patch."""
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.stats import rankdata


def frame_sharpness_weights(image_paths: Sequence[Path]) -> np.ndarray:
    """Return normalized Laplacian-variance weights in the range [0.25, 1]."""
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    sharpness = np.empty(len(image_paths), dtype=np.float64)
    for index, path in enumerate(image_paths):
        with Image.open(path) as source:
            gray = source.convert("L")
            scale = 960 / max(gray.size)
            size = tuple(max(1, round(length * scale)) for length in gray.size)
            gray = gray.resize(size, Image.Resampling.LANCZOS)
            pixels = np.asarray(gray, dtype=np.float32)
        sharpness[index] = float(cv2.Laplacian(pixels, cv2.CV_32F).var())
    median = float(np.median(sharpness))
    if median <= 0:
        return np.where(sharpness > 0, 1, 0.25).astype(np.float32)
    return np.clip(sharpness / median, 0.25, 1).astype(np.float32)


def _owners(group, frames, scores, group_count, frame_count):
    pairs = group * frame_count + frames
    totals = np.bincount(
        pairs, weights=scores, minlength=group_count * frame_count
    ).reshape(group_count, frame_count)
    observed = np.zeros(totals.shape, dtype=bool)
    observed.ravel()[pairs] = True
    totals[~observed] = -np.inf
    return np.argmax(totals, axis=1)


def select_single_owner(
    *,
    points,
    frames,
    labels,
    normals,
    centers,
    confidence,
    sharpness,
    spacing,
    patch_cells,
) -> np.ndarray:
    """Keep one frame per surface patch, then refill newly empty fine voxels."""
    points = np.asarray(points)
    frames = np.asarray(frames)
    labels = np.asarray(labels)
    normals = np.asarray(normals)
    centers = np.asarray(centers)
    confidence = np.asarray(confidence)
    sharpness = np.asarray(sharpness)
    count = len(points)
    if (
        points.shape != (count, 3)
        or normals.shape != (count, 3)
        or frames.shape != (count,)
        or labels.shape != (count,)
        or confidence.shape != (count,)
        or centers.ndim != 2
        or centers.shape[1:] != (3,)
        or sharpness.shape != (len(centers),)
    ):
        raise ValueError("ownership arrays have incompatible shapes")
    if (
        not np.issubdtype(frames.dtype, np.integer)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any(frames < 0)
        or np.any(frames >= len(centers))
    ):
        raise ValueError("frames and labels must contain valid integer IDs")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(normals).all()
        or not np.isfinite(centers).all()
        or not np.isfinite(confidence).all()
        or not np.isfinite(sharpness).all()
        or np.any(sharpness < 0.25)
        or np.any(sharpness > 1)
    ):
        raise ValueError("ownership geometry and scores must be finite")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    if isinstance(patch_cells, (bool, np.bool_)) or not isinstance(
        patch_cells, (int, np.integer)
    ) or patch_cells <= 0:
        raise ValueError("patch_cells must be a positive integer")
    if count == 0:
        return np.empty(0, dtype=bool)

    rays = points - centers[frames]
    normal_length = np.linalg.norm(normals, axis=1)
    depth = np.linalg.norm(rays, axis=1)
    cosine = np.divide(
        np.abs(np.einsum("ij,ij->i", normals, rays)),
        normal_length * depth,
        out=np.zeros(count, dtype=np.float64),
        where=(normal_length > 0) & (depth > 0),
    )
    if count == 1:
        confidence_rank = np.ones(1, dtype=np.float64)
    else:
        confidence_rank = (rankdata(confidence, method="average") - 1) / (count - 1)
    # Euclidean camera distance approximates optical-axis depth without requiring
    # per-frame extrinsics; the view cosine already penalizes oblique rays.
    relative_depth = np.divide(
        np.median(depth),
        depth,
        out=np.zeros(count, dtype=np.float64),
        where=depth > 0,
    )
    scores = (
        cosine
        * relative_depth
        * (0.25 + 0.75 * confidence_rank)
        * sharpness[frames]
    )

    axes = np.argmax(np.abs(normals), axis=1)
    dominant = normals[np.arange(count), axes]
    normal_bin = axes * 2 + (dominant < 0)
    normal_bin[normal_length == 0] = 6

    patch = np.floor(points / (patch_cells * spacing)).astype(np.int64)
    patch_keys = np.column_stack((patch, normal_bin, labels))
    _, patch_group = np.unique(patch_keys, axis=0, return_inverse=True)
    patch_owner = _owners(
        patch_group,
        frames,
        scores,
        int(patch_group.max()) + 1,
        len(centers),
    )
    keep = frames == patch_owner[patch_group]

    fine = np.floor(points / spacing).astype(np.int64)
    fine_keys = np.column_stack((fine, normal_bin, labels))
    _, fine_group = np.unique(fine_keys, axis=0, return_inverse=True)
    fine_count = int(fine_group.max()) + 1
    fine_has_owner = np.bincount(
        fine_group, weights=keep, minlength=fine_count
    ).astype(bool)
    holes = ~fine_has_owner
    if holes.any():
        hole_member = holes[fine_group]
        hole_ids = np.cumsum(holes, dtype=np.int64) - 1
        compact_group = hole_ids[fine_group[hole_member]]
        hole_owner = _owners(
            compact_group,
            frames[hole_member],
            scores[hole_member],
            int(holes.sum()),
            len(centers),
        )
        keep[hole_member] = frames[hole_member] == hole_owner[compact_group]
    return keep
