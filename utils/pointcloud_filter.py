"""Scene point-cloud filtering: drop sparse outliers, sky/ground, keep the central subject.

Used by the web viewer export path (``src/web_viewer_export.py``) at its
point-cloud choke point. All operations return index/boolean masks so aligned
arrays (colors, global_ids, confidences, frame_indices) stay consistent with the points.
"""

import logging
from time import perf_counter
from dataclasses import dataclass, replace
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
    remove_ground: bool = True  # guarded RANSAC ground-plane removal
    ground_dist_scale: float = 3.0  # plane threshold = median NN distance * scale
    ground_min_inlier_ratio: float = 0.08  # plane must explain >= 8% to count as ground
    min_remaining_ratio: float = 0.2  # never let filtering keep less than 20%
    min_points: int = 1000  # skip geometric filtering below this size
    # Experimental DBSCAN only: voxel edge = original median NN * scale.
    # Cluster sizes use original member counts; min_points counts proxy neighbors.
    cluster_proxy_voxel_scale: float = 0.0  # 0 disables approximation
    # Uniform proposal only; acceptance always evaluates the complete current cloud.
    ground_proposal_max_points: int = 0  # sampled proposals require explicit opt-in

    def __post_init__(self):
        if (type(self.ground_proposal_max_points) is not int
                or self.ground_proposal_max_points < 0
                or 0 < self.ground_proposal_max_points < 3):
            raise ValueError("ground_proposal_max_points must be 0 or an integer >= 3")
        if (not np.isfinite(self.cluster_proxy_voxel_scale)
                or self.cluster_proxy_voxel_scale < 0):
            raise ValueError("cluster_proxy_voxel_scale must be finite and nonnegative")


def _median_nn_distance(pcd) -> float:
    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    if distances.size == 0:
        return 0.0
    return float(np.median(distances))


@dataclass(frozen=True)
class PreparedSceneFilter:
    """Label-independent full-domain NN/SOR result, before protection or guards.

    Arrays are immutable snapshots in original input order. ``pcd`` is the base
    cloud: selection always creates a new cloud and never changes this one.
    """

    points: np.ndarray
    config: PointCloudFilterConfig
    base_mask: np.ndarray
    median_nn: float
    sor_mask: Optional[np.ndarray]
    pcd: object


def prepare_scene_filter(
    points: np.ndarray, config: Optional[PointCloudFilterConfig] = None,
) -> PreparedSceneFilter:
    """Prepare original float64/base-domain NN and raw SOR, without labels.

    No protection, 20% guard, clustering or ground removal runs here. A missing
    SOR mask means that geometry was disabled, degenerate or too small for SOR.
    """
    config = replace(config or PointCloudFilterConfig())
    points = np.array(points, dtype=np.float64, copy=True).reshape(-1, 3)
    base_mask = np.isfinite(points).all(axis=1) & (np.abs(points) > 0).any(axis=1)
    median_nn, sor_mask, pcd = 0.0, None, None
    if config.enabled and int(base_mask.sum()) >= config.min_points:
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required: pip install open3d")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[base_mask])
        median_nn = _median_nn_distance(pcd)
        if median_nn > 0 and int(base_mask.sum()) > config.sor_nb_neighbors:
            _, ind = pcd.remove_statistical_outlier(
                config.sor_nb_neighbors, config.sor_std_ratio
            )
            sor_mask = np.zeros(int(base_mask.sum()), dtype=bool)
            sor_mask[np.asarray(ind, dtype=np.int64)] = True
            sor_mask.flags.writeable = False
    points.flags.writeable = False
    base_mask.flags.writeable = False
    return PreparedSceneFilter(points, config, base_mask, median_nn, sor_mask, pcd)


def _cluster_labels(pcd, median_nn: float, config: PointCloudFilterConfig) -> np.ndarray:
    """Return per-member labels; optional proxy changes DBSCAN neighborhood counts."""
    eps = median_nn * config.cluster_eps_scale
    if config.cluster_proxy_voxel_scale == 0:
        return np.asarray(pcd.cluster_dbscan(eps, config.cluster_min_points))
    import open3d as o3d

    points = np.asarray(pcd.points)
    scaled = np.floor(points / (median_nn * config.cluster_proxy_voxel_scale))
    limits = np.iinfo(np.int64)
    # Reject an unrepresentable proxy explicitly; do not silently run another mode.
    if (not np.isfinite(scaled).all() or np.any(scaled < limits.min)
            or np.any(scaled >= float(limits.max))):
        raise ValueError("DBSCAN proxy voxel keys exceed int64 representation")
    _, first, members = np.unique(
        scaled.astype(np.int64), axis=0, return_index=True, return_inverse=True
    )
    # First original member is the representative, ordered by first occurrence.
    order = np.argsort(first)
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    proxy = o3d.geometry.PointCloud()
    proxy.points = o3d.utility.Vector3dVector(points[first[order]])
    labels = np.asarray(proxy.cluster_dbscan(eps, config.cluster_min_points))
    return labels[remap[members]]


def _ground_filter_mask(pcd, median_nn: float, config: PointCloudFilterConfig):
    """Propose a plane, then evaluate ground rules on the original working cloud.

    Cap 0 is the original control, including Open3D's returned inlier indices.
    Positive caps recompute full-domain inliers from the returned refined plane;
    this deliberately changes the algorithm and is not an equivalent shortcut.
    """
    started = perf_counter()
    points = np.asarray(pcd.points)
    count = len(points)
    dist = float(np.clip(median_nn * config.ground_dist_scale, 0.02, 0.15))
    proposal = pcd
    cap = config.ground_proposal_max_points
    if cap and count > cap:
        # A local generator never changes NumPy's global sampling state. Sorting
        # the sampled indices preserves original order within the uniform subset.
        indices = np.sort(np.random.default_rng(42).choice(count, size=cap, replace=False))
        proposal = pcd.select_by_index(indices)
    sample_seconds = perf_counter() - started
    proposed_at = perf_counter()
    plane, proposal_inliers = proposal.segment_plane(dist, ransac_n=3, num_iterations=1000)
    proposal_seconds = perf_counter() - proposed_at
    evaluated_at = perf_counter()
    plane = np.asarray(plane, dtype=np.float64)
    info = dict(input_points=count, proposal_points=len(proposal.points), threshold_m=dist,
                proposal_inliers=len(proposal_inliers), sampling_s=sample_seconds,
                proposal_s=proposal_seconds, evaluation_s=0.0, inliers=0,
                support_ratio=0.0, reason="degenerate_plane", control=cap == 0,
                proposal_seed=42 if cap and count > cap else None)
    norm = float(np.linalg.norm(plane[:3])) if plane.shape == (4,) else 0.0
    if plane.shape != (4,) or not np.isfinite(plane).all() or not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        info["evaluation_s"] = perf_counter() - evaluated_at
        return None, info
    if cap:
        plane = plane / norm
        signed = points @ plane[:3] + plane[3]
        # Match Open3D's strict distance comparison, not <=. Its returned plane
        # is refined after final_inliers were found; recomputation is intentional.
        inliers = np.abs(signed) < dist
    else:
        # Preserve pre-refinement final_inliers for the full-cloud control.
        inliers = np.zeros(count, dtype=bool)
        inliers[np.asarray(proposal_inliers, dtype=np.int64)] = True
        signed = None
    inlier_count = int(np.count_nonzero(inliers))
    info.update(inliers=inlier_count, support_ratio=inlier_count / count)
    ground_mask = None
    if inlier_count < config.ground_min_inlier_ratio * count:
        info["reason"] = "insufficient_full_support"
    else:
        ground_mask = ~inliers
        if signed is None:
            signed = points @ plane[:3] + plane[3]
        rest = signed[ground_mask]
        on_pos, on_neg = int(np.count_nonzero(rest > dist)), int(np.count_nonzero(rest < -dist))
        info.update(rest_points=len(rest), positive=on_pos, negative=on_neg)
        if min(on_pos, on_neg) > 0.15 * len(rest):
            ground_mask = None
            info["reason"] = "plane_slices_subject"
        else:
            info["reason"] = "eligible_for_protected_apply"
    info["evaluation_s"] = perf_counter() - evaluated_at
    return ground_mask, info


def filter_scene_points(
    points: np.ndarray,
    config: Optional[PointCloudFilterConfig] = None,
    protect_mask: Optional[np.ndarray] = None,
    *,
    prepared: Optional[PreparedSceneFilter] = None,
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
    if protect_mask is None:
        protected = np.zeros(n, dtype=bool)
    else:
        protected = np.array(protect_mask, dtype=bool, copy=True)
        if protected.shape != (n,):
            raise ValueError(
                "protect_mask must be a boolean vector aligned with points"
            )
    if prepared is None:
        prepared = prepare_scene_filter(points, config)
    elif (not isinstance(prepared, PreparedSceneFilter)
          or prepared.config != config
          or not np.array_equal(prepared.points, points, equal_nan=True)):
        raise ValueError("prepared scene filter inputs/config differ from original domain")
    base_mask = prepared.base_mask
    protected &= base_mask
    if not config.enabled or int(base_mask.sum()) < config.min_points:
        return base_mask.copy()

    keep = base_mask.copy()
    valid_idx = np.flatnonzero(base_mask)
    pcd = prepared.pcd
    median_nn = prepared.median_nn
    if median_nn <= 0:
        return base_mask.copy()

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
    if prepared.sor_mask is not None:
        apply(prepared.sor_mask, "statistical outlier removal")

    # 2. DBSCAN main-cluster retention (peripheral sparse clusters)
    if config.keep_main_clusters and len(valid_idx) >= config.cluster_min_points:
        labels = _cluster_labels(pcd, median_nn, config)
        cluster_mask = np.zeros(len(valid_idx), dtype=bool)
        if labels.size and labels.max() >= 0:
            counts = np.bincount(labels[labels >= 0])
            big = np.flatnonzero(counts >= config.min_cluster_ratio * len(valid_idx))
            if big.size == 0:
                big = np.array([counts.argmax()])
            cluster_mask[np.isin(labels, big)] = True
        apply(cluster_mask, "cluster retention")

    # 3. guarded ground-plane removal: the proposed plane only counts as ground
    # if the remaining points sit (mostly) on ONE side of it; a plane slicing
    # through the subject (mass on both sides) is rejected.
    if config.remove_ground and len(valid_idx) >= max(3, config.min_points):
        ground_mask, ground_info = _ground_filter_mask(pcd, median_nn, config)
        if ground_mask is not None:
            before = len(valid_idx)
            protected_current = protected[valid_idx]
            ground_info["protected_rescued"] = int(np.count_nonzero(~ground_mask & protected_current))
            stage_count = before - ground_info["inliers"] + ground_info["protected_rescued"]
            guard_skipped = stage_count < config.min_remaining_ratio * before
            applied_at = perf_counter()
            apply(ground_mask, "ground-plane removal")
            ground_info["apply_s"] = perf_counter() - applied_at
            ground_info.update(reason="min_remaining_guard" if guard_skipped else "applied",
                               output_points=len(valid_idx), removed_points=before - len(valid_idx))
        else:
            ground_info.update(output_points=len(valid_idx), removed_points=0)
        logger.info("ground filter: %s", ground_info)

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
