"""Scene point-cloud filtering: drop sparse outliers, sky/ground, keep the central subject.

Used by the web viewer export path (``src/web_viewer_export.py``) at its
point-cloud choke point. All operations return index/boolean masks so aligned
arrays (colors, global_ids, confidences, frame_indices) stay consistent with the points.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PointCloudFilterConfig:
    """Tunables for scene point-cloud filtering."""

    enabled: bool = True
    # Always-on base mask: non-finite and all-zero points are dropped regardless.
    sor_nb_neighbors: int = 20  # statistical outlier removal neighbours
    sor_std_ratio: float = 2.0  # smaller = more aggressive
    keep_main_clusters: bool = True  # DBSCAN, drop sparse peripheral clusters
    cluster_eps_scale: float = (
        5.0  # eps = median NN distance * scale (auto, scale-aware)
    )
    cluster_min_points: int = (
        10  # DBSCAN core-point threshold (keep low for sparse ground)
    )
    min_cluster_ratio: float = 0.01  # keep clusters holding >= 1% of points
    remove_ground: bool = True  # RANSAC largest-plane removal with guardrails
    ground_dist_scale: float = 3.0  # plane threshold = median NN distance * scale
    ground_min_inlier_ratio: float = 0.08  # plane must explain >= 8% to count as ground
    min_remaining_ratio: float = 0.2  # never let filtering keep less than 20%
    min_points: int = 1000  # skip geometric filtering below this size


def _median_nn_distance(pcd) -> float:
    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    if distances.size == 0:
        return 0.0
    return float(np.median(distances))


def filter_scene_points(
    points: np.ndarray,
    config: Optional[PointCloudFilterConfig] = None,
    protect_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a boolean mask of points to keep (same length as ``points``).

    Pipeline: finite/nonzero base mask -> statistical outlier removal ->
    DBSCAN main-cluster retention -> guarded ground-plane removal.
    ``protect_mask`` points still require base validity, but bypass every geometric
    stage. Guardrails ensure a degenerate scene is returned unfiltered rather than
    emptied.
    """
    config = config or PointCloudFilterConfig()
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = points.shape[0]

    base_mask = np.isfinite(points).all(axis=1) & (np.abs(points) > 0).any(axis=1)
    if protect_mask is None:
        protected = np.zeros(n, dtype=bool)
    else:
        protected = np.array(protect_mask, dtype=bool, copy=True)
        if protected.shape != (n,):
            raise ValueError(
                "protect_mask must be a boolean vector aligned with points"
            )
        protected &= base_mask
    if not config.enabled or int(base_mask.sum()) < config.min_points:
        return base_mask

    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D required: pip install open3d")

    keep = base_mask.copy()
    valid_idx = np.flatnonzero(base_mask)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[valid_idx])
    median_nn = _median_nn_distance(pcd)
    if median_nn <= 0:
        return base_mask

    def apply(sub_keep: np.ndarray, stage: str) -> None:
        """Fold a keep-mask over the current working set into ``keep``."""
        nonlocal keep, pcd, valid_idx
        stage_keep = sub_keep | protected[valid_idx]
        if stage_keep.sum() < config.min_remaining_ratio * len(valid_idx):
            logger.warning(
                "%s would keep only %d/%d points; stage skipped",
                stage,
                stage_keep.sum(),
                len(valid_idx),
            )
            return
        keep[valid_idx[~stage_keep]] = False
        valid_idx = valid_idx[stage_keep]
        pcd = pcd.select_by_index(np.flatnonzero(stage_keep))

    # 1. statistical outlier removal (sparse floaters, sky specks)
    if len(valid_idx) > config.sor_nb_neighbors:
        _, ind = pcd.remove_statistical_outlier(
            config.sor_nb_neighbors, config.sor_std_ratio
        )
        sor_mask = np.zeros(len(valid_idx), dtype=bool)
        sor_mask[np.asarray(ind)] = True
        apply(sor_mask, "statistical outlier removal")

    # 2. DBSCAN main-cluster retention (peripheral sparse clusters)
    if config.keep_main_clusters and len(valid_idx) >= config.cluster_min_points:
        eps = median_nn * config.cluster_eps_scale
        labels = np.asarray(pcd.cluster_dbscan(eps, config.cluster_min_points))
        cluster_mask = np.zeros(len(valid_idx), dtype=bool)
        if labels.size and labels.max() >= 0:
            counts = np.bincount(labels[labels >= 0])
            big = np.flatnonzero(counts >= config.min_cluster_ratio * len(valid_idx))
            if big.size == 0:
                big = np.array([counts.argmax()])
            cluster_mask[np.isin(labels, big)] = True
        apply(cluster_mask, "cluster retention")

    # 3. guarded ground-plane removal: the largest plane only counts as ground
    # if the remaining points sit (mostly) on ONE side of it; a plane slicing
    # through the subject (mass on both sides) is rejected.
    if config.remove_ground and len(valid_idx) >= max(3, config.min_points):
        dist = float(np.clip(median_nn * config.ground_dist_scale, 0.02, 0.15))
        plane, inliers = pcd.segment_plane(dist, ransac_n=3, num_iterations=1000)
        inliers = np.asarray(inliers)
        if inliers.size >= config.ground_min_inlier_ratio * len(valid_idx):
            ground_mask = np.ones(len(valid_idx), dtype=bool)
            ground_mask[inliers] = False
            normal = np.asarray(plane[:3])
            signed = np.asarray(pcd.points) @ normal + plane[3]
            rest = signed[ground_mask]
            on_pos = np.sum(rest > dist)
            on_neg = np.sum(rest < -dist)
            if min(on_pos, on_neg) > 0.15 * rest.size:
                logger.info(
                    "largest plane slices through the subject; ground removal skipped"
                )
            else:
                apply(ground_mask, "ground-plane removal")

    logger.info(
        "point-cloud filter: %d -> %d points (median NN %.4f m)",
        n,
        int(keep.sum()),
        median_nn,
    )
    return keep


def apply_mask(
    mask: np.ndarray, *arrays: Optional[np.ndarray]
) -> Tuple[Optional[np.ndarray], ...]:
    """Apply a keep-mask to points and any index-aligned arrays (None passthrough)."""
    return tuple(None if a is None else np.asarray(a)[mask] for a in arrays)
