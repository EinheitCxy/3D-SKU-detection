"""Independent pixel correspondences for multi-view pose/depth refinement.

The matches here come only from image appearance.  They deliberately do not
use reconstructed depth, projected DA3 points, SKU detections, or global IDs,
so a later optimizer can use them as an independent geometric constraint.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np


class CorrespondenceExtractionError(ValueError):
    """Raised when image-only correspondences cannot connect every frame."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics


def _validate_inputs(
    images: np.ndarray, intrinsics: np.ndarray
) -> tuple[int, int, int]:
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8:
        raise ValueError("images must be an NHWC uint8 RGB numpy array")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must have shape (N, H, W, 3)")
    frame_count, height, width, _ = images.shape
    if frame_count < 2 or height < 8 or width < 8:
        raise ValueError(
            "images must contain at least two RGB frames of size at least 8x8"
        )
    if not np.isfinite(images).all():  # kept explicit if callers pass a uint8 subclass
        raise ValueError("images must contain only finite values")
    if not isinstance(intrinsics, np.ndarray) or intrinsics.shape != (
        frame_count,
        3,
        3,
    ):
        raise ValueError("intrinsics must have shape (N, 3, 3) matching images")
    if (
        not np.issubdtype(intrinsics.dtype, np.number)
        or not np.isfinite(intrinsics).all()
    ):
        raise ValueError("intrinsics must be a finite numeric array")
    intrinsics64 = intrinsics.astype(np.float64, copy=False)
    if np.any(np.linalg.det(intrinsics64) <= 0.0):
        raise ValueError("each intrinsic matrix must have positive determinant")
    if np.any(intrinsics64[:, 0, 0] <= 0.0) or np.any(intrinsics64[:, 1, 1] <= 0.0):
        raise ValueError("each intrinsic matrix must have positive focal lengths")
    return frame_count, height, width


def _mutual_ratio_matches(
    descriptors_i: np.ndarray | None,
    descriptors_j: np.ndarray | None,
    *,
    ratio: float = 0.75,
) -> list[cv2.DMatch]:
    if descriptors_i is None or descriptors_j is None:
        return []
    if len(descriptors_i) < 2 or len(descriptors_j) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    forward = matcher.knnMatch(descriptors_i, descriptors_j, k=2)
    reverse = matcher.knnMatch(descriptors_j, descriptors_i, k=2)
    accepted_forward = {
        pair[0].queryIdx: pair[0]
        for pair in forward
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    }
    accepted_reverse = {
        pair[0].queryIdx: pair[0]
        for pair in reverse
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    }
    return [
        match
        for query_index, match in accepted_forward.items()
        if (reverse_match := accepted_reverse.get(match.trainIdx)) is not None
        and reverse_match.trainIdx == query_index
    ]


def _grid_balance(
    xy_i: np.ndarray,
    xy_j: np.ndarray,
    distances: np.ndarray,
    *,
    width: int,
    height: int,
    max_matches: int,
    grid_size: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cap per-cell selections before filling unused capacity by match quality."""
    if len(xy_i) <= max_matches:
        return xy_i, xy_j, distances
    cells_x = np.clip((xy_i[:, 0] * grid_size / width).astype(int), 0, grid_size - 1)
    cells_y = np.clip((xy_i[:, 1] * grid_size / height).astype(int), 0, grid_size - 1)
    order = np.argsort(distances, kind="stable")
    per_cell_cap = max(1, int(np.ceil(max_matches / (grid_size * grid_size))))
    counts = np.zeros((grid_size, grid_size), dtype=np.int32)
    selected: list[int] = []
    remainder: list[int] = []
    for index in order:
        cell_x, cell_y = cells_x[index], cells_y[index]
        if counts[cell_y, cell_x] < per_cell_cap:
            selected.append(int(index))
            counts[cell_y, cell_x] += 1
        else:
            remainder.append(int(index))
    if len(selected) < max_matches:
        selected.extend(remainder[: max_matches - len(selected)])
    keep = np.asarray(selected[:max_matches], dtype=np.intp)
    return xy_i[keep], xy_j[keep], distances[keep]


def _has_image_coverage(xy: np.ndarray, *, width: int, height: int) -> bool:
    """Reject localized repeated-texture clusters despite a large match count."""
    if len(xy) < 2:
        return False
    spread = np.ptp(xy, axis=0)
    return bool(spread[0] >= 0.15 * width and spread[1] >= 0.15 * height)


def _connected_components(
    frame_count: int, edges: list[tuple[int, int]]
) -> list[list[int]]:
    adjacency = [[] for _ in range(frame_count)]
    for frame_i, frame_j in edges:
        adjacency[frame_i].append(frame_j)
        adjacency[frame_j].append(frame_i)
    components: list[list[int]] = []
    remaining = set(range(frame_count))
    while remaining:
        start = remaining.pop()
        component = [start]
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def extract_correspondences(
    images: np.ndarray,
    intrinsics: np.ndarray,
    *,
    pair_window: int = 2,
    max_features: int = 2000,
    max_matches_per_pair: int = 250,
    min_matches_per_pair: int = 16,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract robust, image-only SIFT correspondences from nearby frame pairs.

    Args:
        images: Processed-grid RGB pixels with shape ``(N, H, W, 3)`` and
            dtype ``uint8``.
        intrinsics: Per-frame ``(N, 3, 3)`` camera matrices.  They are checked
            here to keep this module's input contract aligned with the later
            optimizer; geometric filtering intentionally uses only image
            coordinates through an independently fitted fundamental matrix.

    Returns:
        ``(correspondences, diagnostics)``.  Correspondences has ``frame_i``
        and ``frame_j`` int32 vectors, plus ``xy_i`` and ``xy_j`` float64
        arrays of shape ``(M, 2)``.  Diagnostics records every attempted pair,
        including rejected-pair reasons.

    Raises:
        CorrespondenceExtractionError: if no reliable connected graph covers
            every frame.  Its ``diagnostics`` attribute gives pair details.
    """
    frame_count, height, width = _validate_inputs(images, intrinsics)
    if pair_window < 1:
        raise ValueError("pair_window must be at least 1")
    if max_features < 2 or max_matches_per_pair < 8 or min_matches_per_pair < 8:
        raise ValueError("feature and match limits must be at least 8")
    if min_matches_per_pair > max_matches_per_pair:
        raise ValueError("min_matches_per_pair must not exceed max_matches_per_pair")

    cv2.setRNGSeed(int(seed))
    sift = cv2.SIFT_create(nfeatures=int(max_features))
    keypoints: list[list[cv2.KeyPoint]] = []
    descriptors: list[np.ndarray | None] = []
    for frame in images:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame_keypoints, frame_descriptors = sift.detectAndCompute(gray, None)
        keypoints.append(frame_keypoints)
        descriptors.append(frame_descriptors)

    diagnostics: dict[str, Any] = {
        "seed": int(seed),
        "frame_count": frame_count,
        "pair_window": int(pair_window),
        "feature_counts": [len(points) for points in keypoints],
        "pairs": [],
    }
    retained: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for frame_i in range(frame_count):
        for frame_j in range(frame_i + 1, min(frame_count, frame_i + pair_window + 1)):
            record: dict[str, Any] = {"frame_i": frame_i, "frame_j": frame_j}
            matches = _mutual_ratio_matches(descriptors[frame_i], descriptors[frame_j])
            record["mutual_ratio_matches"] = len(matches)
            if len(matches) < 8:
                record["accepted"] = False
                record["reason"] = "fewer than 8 mutual Lowe-ratio matches"
                diagnostics["pairs"].append(record)
                continue
            xy_i = np.asarray(
                [keypoints[frame_i][m.queryIdx].pt for m in matches], dtype=np.float64
            )
            xy_j = np.asarray(
                [keypoints[frame_j][m.trainIdx].pt for m in matches], dtype=np.float64
            )
            fundamental, inlier_mask = cv2.findFundamentalMat(
                xy_i,
                xy_j,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=1.0,
                confidence=0.999,
            )
            if fundamental is None or inlier_mask is None:
                record["accepted"] = False
                record["reason"] = "fundamental-matrix RANSAC could not fit a model"
                diagnostics["pairs"].append(record)
                continue
            inliers = inlier_mask.reshape(-1).astype(bool)
            xy_i, xy_j = xy_i[inliers], xy_j[inliers]
            distances = np.asarray(
                [m.distance for m, keep in zip(matches, inliers) if keep],
                dtype=np.float64,
            )
            record["ransac_inliers"] = len(xy_i)
            if len(xy_i) < min_matches_per_pair:
                record["accepted"] = False
                record["reason"] = f"fewer than {min_matches_per_pair} RANSAC inliers"
                diagnostics["pairs"].append(record)
                continue
            xy_i, xy_j, _ = _grid_balance(
                xy_i,
                xy_j,
                distances,
                width=width,
                height=height,
                max_matches=max_matches_per_pair,
            )
            record["retained_matches"] = len(xy_i)
            if not _has_image_coverage(
                xy_i, width=width, height=height
            ) or not _has_image_coverage(xy_j, width=width, height=height):
                record["accepted"] = False
                record["reason"] = "RANSAC inliers do not cover both image axes"
                diagnostics["pairs"].append(record)
                continue
            record["accepted"] = True
            diagnostics["pairs"].append(record)
            retained.append((frame_i, frame_j, xy_i, xy_j))

    components = _connected_components(
        frame_count, [(item[0], item[1]) for item in retained]
    )
    diagnostics["connected_components"] = components
    diagnostics["accepted_pair_count"] = len(retained)
    if len(components) != 1:
        raise CorrespondenceExtractionError(
            f"image-only correspondence graph is disconnected: {components}",
            diagnostics,
        )
    if not retained:
        raise CorrespondenceExtractionError(
            "no frame pair passed correspondence gates", diagnostics
        )

    return (
        {
            "frame_i": np.concatenate(
                [
                    np.full(len(xy_i), pair_i, dtype=np.int32)
                    for pair_i, _, xy_i, _ in retained
                ]
            ),
            "frame_j": np.concatenate(
                [
                    np.full(len(xy_j), pair_j, dtype=np.int32)
                    for _, pair_j, _, xy_j in retained
                ]
            ),
            "xy_i": np.concatenate([xy_i for _, _, xy_i, _ in retained]).astype(
                np.float64, copy=False
            ),
            "xy_j": np.concatenate([xy_j for _, _, _, xy_j in retained]).astype(
                np.float64, copy=False
            ),
        },
        diagnostics,
    )
