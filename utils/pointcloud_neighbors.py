"""Bounded CPU KNN reuse for Viewer geometry preparation.

SciPy's exact spatial query is not a promise of bitwise Open3D equivalence.
Only arrays/scalars are returned. Downstream protection and guards stay in the
original scene filter; the native Open3D preparation remains an explicit control.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeighborStatistics:
    nearest_distances: np.ndarray
    mean_distances: np.ndarray
    sor_mask: np.ndarray
    median_nn: float
    sor_threshold: float


def prepare_neighbor_statistics(
    points: np.ndarray, *, k: int = 20, std_ratio: float = 2.0,
    workers: int = 4, chunk_points: int = 50_000,
) -> NeighborStatistics:
    """Build one float64 cKDTree and query max(k, 2) neighbors in bounded chunks.

    ``points`` is the already selected finite base domain in original order.
    K includes self, as in Open3D SOR. With k=1 NN still needs the second
    neighbor. Sequential accumulation follows Open3D's reduction order; distance
    arithmetic and tied neighbors still require a comparison against Open3D.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite float64[N,3]")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(std_ratio) or std_ratio <= 0:
        raise ValueError("std_ratio must be positive and finite")
    if type(chunk_points) is not int or chunk_points < 1:
        raise ValueError("chunk_points must be a positive integer")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    count = len(points)
    nearest = np.zeros(count, dtype=np.float64)
    means = np.full(count, -1.0, dtype=np.float64)
    mask = np.zeros(count, dtype=bool)
    threshold = float("nan")
    if count:
        tree = cKDTree(points)
        query_k = min(count, max(k, 2))
        sor_k = min(count, k)
        for start in range(0, count, chunk_points):
            stop = min(count, start + chunk_points)
            distances, _ = tree.query(points[start:stop], k=query_k, eps=0,
                                      p=2, workers=workers)
            distances = distances.reshape(stop - start, query_k)
            if query_k >= 2:
                nearest[start:stop] = distances[:, 1]
            means[start:stop] = np.add.accumulate(
                distances[:, :sor_k], axis=1
            )[:, -1] / sor_k
        # Open3D counts successful queries even when their mean distance is zero.
        positive = means > 0
        mean = np.add.accumulate(np.where(positive, means, 0.0))[-1] / count
        squared = np.where(positive, (means - mean) ** 2, 0.0)
        if count > 1:
            deviation = np.sqrt(np.add.accumulate(squared)[-1] / (count - 1))
            threshold = float(mean + std_ratio * deviation)
            mask = positive & (means < threshold)
    for array in (nearest, means, mask):
        array.flags.writeable = False
    return NeighborStatistics(nearest, means, mask,
                              float(np.median(nearest)) if count else 0.0,
                              threshold)
