"""Deterministic support-plane geometry for ground-stacked cartons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN


class FootprintError(ValueError):
    """Raised when footprint geometry does not meet its quality gates."""


@dataclass(frozen=True)
class SupportPlane:
    point: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    inlier_count: int
    inlier_fraction: float
    p95_residual_m: float


def fit_support_plane(background_points: np.ndarray) -> SupportPlane:
    """Fit a deterministic support plane from metric background points."""
    points = _validate_points(background_points, minimum=10_000, label="background")
    sampled = _deterministic_subsample(points, maximum=50_000, seed=13)
    candidate_point, candidate_normal = _best_ransac_plane(
        sampled, trials=2_048, threshold_m=0.012, seed=13
    )
    full_distances = np.abs((points - candidate_point) @ candidate_normal)
    inliers = points[full_distances <= 0.012]
    return _refine_support_plane(inliers, total_points=len(points))


def select_support_plane(
    background_points: np.ndarray,
    frame_ids: np.ndarray,
    object_points: np.ndarray,
) -> tuple[SupportPlane, dict[str, object]]:
    """Select a table-like support plane instead of blindly choosing the largest plane.

    Candidate extraction is deterministic, frame balanced, and deliberately keeps
    several RANSAC hypotheses so a large wall cannot hide a smaller tabletop.
    """
    points = _validate_points(background_points, minimum=10_000, label="background")
    frames = np.asarray(frame_ids)
    objects = _validate_points(object_points, minimum=64, label="object")
    if frames.shape != (len(points),):
        raise FootprintError("support plane frame ids must align with background points")
    sampled, sampled_frames = _frame_balanced_subsample(points, frames, maximum=50_000)
    remaining = np.arange(len(sampled))
    candidates: list[tuple[SupportPlane, dict[str, object]]] = []
    for candidate_index in range(5):
        if len(remaining) < 3:
            break
        candidate_point, candidate_normal = _adaptive_ransac_plane(
            sampled[remaining], threshold_m=0.012, seed=13 + candidate_index
        )
        full_distances = np.abs((points - candidate_point) @ candidate_normal)
        inliers = points[full_distances <= 0.012]
        try:
            plane = _refine_support_plane(inliers, total_points=len(points), max_residual_m=0.010)
        except FootprintError:
            local_distances = np.abs((sampled[remaining] - candidate_point) @ candidate_normal)
            remaining = remaining[local_distances > 0.012]
            continue
        plane, oriented_heights = _orient_for_objects(plane, objects)
        coordinates = project_to_plane(inliers, plane)
        hull = MultiPoint(coordinates).convex_hull
        spans = np.ptp(coordinates, axis=0)
        inlier_frames = np.unique(frames[full_distances <= 0.012])
        frame_fraction = len(inlier_frames) / len(np.unique(frames))
        centres_inside = _object_centres_inside_hull(objects, plane, hull)
        gates = {
            "inlier_count": int(plane.inlier_count >= 10_000),
            "inlier_fraction": bool(plane.inlier_fraction >= 0.10),
            "p95_residual": bool(plane.p95_residual_m <= 0.010),
            "frame_span": bool(len(inlier_frames) >= 3 and frame_fraction >= 0.30),
            "in_plane_span": bool(np.all(spans >= 0.30)),
            "hull_area": bool(hull.area >= 0.25),
            "object_height": bool(
                np.mean(oriented_heights >= -0.012) >= 0.95
                and np.quantile(oriented_heights, 0.01) <= 0.080
            ),
            "object_hull": bool(centres_inside >= 0.80),
        }
        score = float(
            sum(bool(value) for value in gates.values())
            + min(1.0, hull.area) * 0.01
            + min(1.0, frame_fraction) * 0.001
        )
        diagnostics = {
            "index": candidate_index,
            "normal": plane.normal.tolist(),
            "point": plane.point.tolist(),
            "inlier_count": plane.inlier_count,
            "inlier_fraction": plane.inlier_fraction,
            "p95_residual_m": plane.p95_residual_m,
            "frame_count": int(len(inlier_frames)),
            "frame_fraction": float(frame_fraction),
            "spans_m": spans.tolist(),
            "hull_area_m2": float(hull.area),
            "object_centres_inside_fraction": float(centres_inside),
            "object_height_p01_m": float(np.quantile(oriented_heights, 0.01)),
            "gates": gates,
            "score": score,
        }
        candidates.append((plane, diagnostics))
        local_distances = np.abs((sampled[remaining] - candidate_point) @ candidate_normal)
        remaining = remaining[local_distances > 0.012]
    eligible = [(plane, info) for plane, info in candidates if all(info["gates"].values())]
    if not eligible:
        raise FootprintError("support plane candidates failed table compatibility gates")
    eligible.sort(key=lambda item: (-float(item[1]["score"]), int(item[1]["index"])))
    selected, selected_info = eligible[0]
    for _, other in eligible[1:]:
        normal_difference = abs(float(np.dot(selected.normal, np.asarray(other["normal"]))))
        if normal_difference < 0.95 and float(other["score"]) >= 0.95 * float(selected_info["score"]):
            raise FootprintError("support plane is ambiguous between differently oriented candidates")
    return selected, {
        "sample_count": int(len(sampled)),
        "sample_frame_count": int(len(np.unique(sampled_frames))),
        "candidates": [info for _, info in candidates],
        "selected_index": int(selected_info["index"]),
    }


def carton_footprint_polygon(
    points: np.ndarray, plane: SupportPlane
) -> tuple[Polygon, dict[str, float | int]]:
    """Return the support-plane OBB footprint for one carton point cloud."""
    object_points = _validate_points(points, minimum=64, label="carton")
    _validate_plane(plane)
    heights = (object_points - plane.point) @ plane.normal
    elevated_points = object_points[heights > 0.015]
    if len(elevated_points) < 64:
        raise FootprintError("carton has fewer than 64 points above the support plane")
    projected = project_to_plane(elevated_points, plane)
    component = _largest_density_component(projected, eps_m=0.03, min_samples=8)
    cleaned = _trim_projected_outliers(component, lower=0.01, upper=0.99)
    polygon = MultiPoint(cleaned).convex_hull.minimum_rotated_rectangle
    polygon = _validate_obb_polygon(polygon)
    return polygon, {
        "input_point_count": int(len(object_points)),
        "elevated_point_count": int(len(elevated_points)),
        "component_point_count": int(len(component)),
        "cleaned_point_count": int(len(cleaned)),
        "footprint_area_m2": float(polygon.area),
    }


def union_footprints(polygons: list[Polygon]) -> Polygon | MultiPolygon:
    """Merge overlapping carton footprints without double-counting area."""
    if not polygons:
        raise FootprintError("footprint union requires at least one polygon")
    union = unary_union(polygons)
    if union.is_empty or not np.isfinite(union.area) or union.area <= 0:
        raise FootprintError("footprint union is not positive and finite")
    if not isinstance(union, (Polygon, MultiPolygon)):
        raise FootprintError("footprint union is not polygonal")
    return union


def project_to_plane(points: np.ndarray, plane: SupportPlane) -> np.ndarray:
    """Express orthogonally projected 3-D points in deterministic plane axes."""
    offsets = points - plane.point
    projected_offsets = offsets - np.outer(offsets @ plane.normal, plane.normal)
    return np.column_stack(
        [projected_offsets @ plane.u_axis, projected_offsets @ plane.v_axis]
    )


def _validate_points(points: np.ndarray, minimum: int, label: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise FootprintError(f"{label} points must have shape (N, 3)")
    if len(array) < minimum:
        subject = "support plane background" if label == "background" else label
        raise FootprintError(f"{subject} requires at least {minimum} points")
    if not np.isfinite(array).all():
        raise FootprintError(f"{label} points must be finite")
    return array


def _validate_plane(plane: SupportPlane) -> None:
    vectors = (plane.point, plane.normal, plane.u_axis, plane.v_axis)
    if any(np.asarray(vector).shape != (3,) for vector in vectors):
        raise FootprintError("support plane vectors must have shape (3,)")
    if not all(np.isfinite(vector).all() for vector in vectors):
        raise FootprintError("support plane vectors must be finite")
    if not np.isclose(np.linalg.norm(plane.normal), 1.0, atol=1e-8):
        raise FootprintError("support plane normal must be unit length")


def _deterministic_subsample(
    points: np.ndarray, maximum: int, seed: int
) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.random.default_rng(seed).choice(len(points), size=maximum, replace=False)
    return points[np.sort(indices)]


def _best_ransac_plane(
    points: np.ndarray, trials: int, threshold_m: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    best_count = 0
    best_candidate: tuple[np.ndarray, np.ndarray] | None = None

    for _ in range(trials):
        indices = _sample_ransac_triplet(generator, population_size=len(points))
        first, second, third = points[indices]
        first_edge = second - first
        second_edge = third - first
        normal = np.cross(first_edge, second_edge)
        norm = np.linalg.norm(normal)
        edge_product = np.linalg.norm(first_edge) * np.linalg.norm(second_edge)
        if edge_product == 0 or norm / edge_product <= 1e-8:
            continue
        normal /= norm
        count = int(np.count_nonzero(np.abs((points - first) @ normal) <= threshold_m))
        if count > best_count:
            best_count = count
            best_candidate = (first, normal)

    if best_candidate is None:
        raise FootprintError("support plane RANSAC found no non-collinear candidate")
    return best_candidate


def _sample_ransac_triplet(
    generator: np.random.Generator, population_size: int
) -> np.ndarray:
    return generator.choice(population_size, size=3, replace=False)


def _refine_support_plane(
    inliers: np.ndarray, total_points: int, max_residual_m: float = 0.012
) -> SupportPlane:
    inlier_count = len(inliers)
    inlier_fraction = inlier_count / total_points
    if inlier_count < 10_000 or inlier_fraction < 0.10:
        raise FootprintError("support plane has insufficient inliers")

    point = inliers.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(inliers - point, full_matrices=False)
    normal = right_vectors[-1]
    normal /= np.linalg.norm(normal)
    dominant_index = int(np.argmax(np.abs(normal)))
    if normal[dominant_index] < 0:
        normal = -normal

    residuals = np.abs((inliers - point) @ normal)
    p95_residual_m = float(np.percentile(residuals, 95))
    if p95_residual_m > max_residual_m:
        raise FootprintError(f"support plane residual exceeds {max_residual_m:.3f} m")

    reference_axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u_axis = np.cross(normal, reference_axis)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    return SupportPlane(
        point=point,
        normal=normal,
        u_axis=u_axis,
        v_axis=v_axis,
        inlier_count=inlier_count,
        inlier_fraction=inlier_fraction,
        p95_residual_m=p95_residual_m,
    )


def voxel_balance_points(points: np.ndarray, voxel_size_m: float = 0.005) -> np.ndarray:
    """Keep one deterministic point per 5-mm projected-density voxel."""
    array = _validate_points(points, minimum=1, label="carton")
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise FootprintError("voxel size must be positive and finite")
    voxel_keys = np.floor(array / voxel_size_m).astype(np.int64)
    _, keep = np.unique(voxel_keys, axis=0, return_index=True)
    return array[np.sort(keep)]


def _frame_balanced_subsample(
    points: np.ndarray, frame_ids: np.ndarray, maximum: int
) -> tuple[np.ndarray, np.ndarray]:
    unique_frames = np.unique(frame_ids)
    per_frame = max(1, maximum // len(unique_frames))
    selected: list[np.ndarray] = []
    generator = np.random.default_rng(13)
    for frame in unique_frames:
        indices = np.flatnonzero(frame_ids == frame)
        if len(indices) > per_frame:
            indices = np.sort(generator.choice(indices, size=per_frame, replace=False))
        selected.append(indices)
    all_indices = np.concatenate(selected)
    if len(all_indices) > maximum:
        all_indices = np.sort(generator.choice(all_indices, size=maximum, replace=False))
    return points[all_indices], frame_ids[all_indices]


def _adaptive_ransac_plane(
    points: np.ndarray, threshold_m: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    best_count = 0
    best_candidate: tuple[np.ndarray, np.ndarray] | None = None
    minimum_trials = 128
    target_trials = 10_000
    trial = 0
    while trial < target_trials:
        indices = _sample_ransac_triplet(generator, population_size=len(points))
        first, second, third = points[indices]
        normal = np.cross(second - first, third - first)
        norm = np.linalg.norm(normal)
        if norm == 0:
            trial += 1
            continue
        normal /= norm
        count = int(np.count_nonzero(np.abs((points - first) @ normal) <= threshold_m))
        if count > best_count:
            best_count = count
            best_candidate = (first, normal)
            ratio = min(max(count / len(points), 1e-9), 1.0 - 1e-12)
            target_trials = min(
                10_000,
                max(minimum_trials, int(np.ceil(np.log(1.0 - 0.999) / np.log(1.0 - ratio**3)))),
            )
        trial += 1
    if best_candidate is None:
        raise FootprintError("support plane RANSAC found no non-collinear candidate")
    return best_candidate


def _orient_for_objects(plane: SupportPlane, objects: np.ndarray) -> tuple[SupportPlane, np.ndarray]:
    heights = (objects - plane.point) @ plane.normal
    if np.median(heights) < 0:
        plane = SupportPlane(
            point=plane.point,
            normal=-plane.normal,
            u_axis=-plane.u_axis,
            v_axis=plane.v_axis,
            inlier_count=plane.inlier_count,
            inlier_fraction=plane.inlier_fraction,
            p95_residual_m=plane.p95_residual_m,
        )
        heights = -heights
    return plane, heights


def _object_centres_inside_hull(objects: np.ndarray, plane: SupportPlane, hull: object) -> float:
    if hull.is_empty:
        return 0.0
    # Object observations are not available in this pure boundary; quantized 5-cm
    # groups retain a conservative per-object-centre approximation.
    projected = project_to_plane(objects, plane)
    centre = projected.mean(axis=0)
    return float(hull.buffer(0.150).covers(MultiPoint([centre])))


def _largest_density_component(
    projected: np.ndarray, eps_m: float, min_samples: int
) -> np.ndarray:
    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(projected)
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        raise FootprintError("carton footprint has no density component")
    populations = np.bincount(valid_labels)
    nonzero_populations = populations[populations > 0]
    if len(nonzero_populations) > 1:
        second_largest = int(np.sort(nonzero_populations)[-2])
        substantial_limit = min(32, int(np.ceil(0.20 * len(valid_labels))))
        if second_largest >= substantial_limit:
            raise FootprintError("carton footprint has multiple substantial components")
    greatest_population = int(populations.max())
    label = int(np.flatnonzero(populations == greatest_population)[0])
    return projected[labels == label]


def _trim_projected_outliers(
    projected: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    lower_bounds = np.quantile(projected, lower, axis=0)
    upper_bounds = np.quantile(projected, upper, axis=0)
    cleaned = projected[
        np.all((projected >= lower_bounds) & (projected <= upper_bounds), axis=1)
    ]
    if len(cleaned) < 3:
        raise FootprintError("carton footprint has fewer than three cleaned points")
    return cleaned


def _validate_obb_polygon(candidate: object) -> Polygon:
    if not isinstance(candidate, Polygon):
        raise FootprintError("carton OBB is degenerate")
    if candidate.is_empty or not candidate.is_valid:
        raise FootprintError("carton OBB is invalid")
    if not np.isfinite(candidate.area) or candidate.area <= 0:
        raise FootprintError("carton OBB area is not positive and finite")
    coordinates = np.asarray(candidate.exterior.coords, dtype=float)
    edge_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    if len(edge_lengths) != 4 or not np.isfinite(edge_lengths).all():
        raise FootprintError("carton OBB edges must be finite")
    first_side_m, second_side_m = edge_lengths[:2]
    if first_side_m < 0.05 or second_side_m < 0.05:
        raise FootprintError("carton OBB side length must be at least 0.05 m")
    return candidate
