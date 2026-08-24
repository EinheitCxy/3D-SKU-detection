"""Deterministic support-plane geometry for ground-stacked cartons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN


class FootprintError(ValueError):
    """Raised when footprint geometry does not meet its quality gates."""

    def __init__(self, message: str, diagnostics: dict[str, object] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class SupportPlaneSelectionError(FootprintError):
    """A support-plane rejection that retains every evaluated candidate."""


@dataclass(frozen=True)
class SupportPlane:
    point: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    inlier_count: int
    inlier_fraction: float
    p95_residual_m: float


@dataclass(frozen=True)
class RansacOutcome:
    """One deterministic RANSAC candidate with audit-only trial diagnostics."""

    point: np.ndarray
    normal: np.ndarray
    trial_count: int
    early_exit: bool


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
    object_observations: np.ndarray | list[np.ndarray],
) -> tuple[SupportPlane, dict[str, object]]:
    """Select a table-like support plane instead of blindly choosing the largest plane.

    Candidate extraction is deterministic, frame balanced, and deliberately keeps
    several RANSAC hypotheses so a large wall cannot hide a smaller tabletop.
    """
    points = _validate_points(background_points, minimum=10_000, label="background")
    frames = np.asarray(frame_ids)
    observations = _normalise_object_observations(object_observations)
    objects = np.concatenate(observations)
    object_centres = np.asarray([observation.mean(axis=0) for observation in observations])
    if frames.shape != (len(points),):
        raise FootprintError("support plane frame ids must align with background points")
    sampled, sampled_frames = _frame_balanced_subsample(points, frames, maximum=50_000)
    remaining = np.arange(len(sampled))
    candidates: list[tuple[SupportPlane | None, dict[str, object]]] = []
    for candidate_index in range(5):
        if len(remaining) < 3:
            break
        ransac = _adaptive_ransac_plane(
            sampled[remaining], threshold_m=0.012, seed=13 + candidate_index
        )
        candidate_point, candidate_normal = ransac.point, ransac.normal
        full_distances = np.abs((points - candidate_point) @ candidate_normal)
        inliers = points[full_distances <= 0.012]
        raw_candidate = {
            "index": candidate_index,
            "raw_point": candidate_point.tolist(),
            "raw_normal": candidate_normal.tolist(),
            "raw_inlier_count": int(len(inliers)),
            "raw_inlier_fraction": float(len(inliers) / len(points)),
            "ransac": {
                "trial_count": ransac.trial_count,
                "early_exit": ransac.early_exit,
            },
        }
        try:
            plane, retained_indices = _refine_support_plane(
                inliers,
                total_points=len(points),
                max_residual_m=0.010,
                return_retained_indices=True,
            )
        except FootprintError as error:
            candidates.append(
                (
                    None,
                    {
                        **raw_candidate,
                        "refinement": {"passed": False, "reason": str(error)},
                    },
                )
            )
            local_distances = np.abs((sampled[remaining] - candidate_point) @ candidate_normal)
            remaining = remaining[local_distances > 0.012]
            continue
        plane, oriented_heights = _orient_for_objects(plane, objects)
        final_support_points = inliers[retained_indices]
        raw_inlier_frames = frames[full_distances <= 0.012]
        final_support_frames = raw_inlier_frames[retained_indices]
        coordinates = project_to_plane(final_support_points, plane)
        hull = MultiPoint(coordinates).convex_hull
        spans = np.ptp(coordinates, axis=0)
        inlier_frames = np.unique(final_support_frames)
        frame_fraction = len(inlier_frames) / len(np.unique(frames))
        centres_inside = _object_centres_inside_hull(object_centres, plane, hull)
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
            **raw_candidate,
            "normal": plane.normal.tolist(),
            "point": plane.point.tolist(),
            "inlier_count": plane.inlier_count,
            "inlier_fraction": plane.inlier_fraction,
            "p95_residual_m": plane.p95_residual_m,
            "refinement": {
                "passed": True,
                "retained_inlier_count": int(len(final_support_points)),
                "retained_inlier_fraction": float(len(final_support_points) / len(points)),
            },
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
    eligible = [
        (plane, info)
        for plane, info in candidates
        if plane is not None and all(info["gates"].values())
    ]
    diagnostics = {
        "sample_count": int(len(sampled)),
        "sample_frame_count": int(len(np.unique(sampled_frames))),
        "candidates": [info for _, info in candidates],
        "selected_index": None,
    }
    if not eligible:
        if not any(info["refinement"]["passed"] for _, info in candidates):
            raise SupportPlaneSelectionError(
                "no support-plane candidate passed refinement gates", diagnostics
            )
        raise SupportPlaneSelectionError(
            "support plane candidates failed table compatibility gates", diagnostics
        )
    eligible.sort(key=lambda item: (-float(item[1]["score"]), int(item[1]["index"])))
    selected, selected_info = eligible[0]
    for _, other in eligible[1:]:
        normal_difference = abs(float(np.dot(selected.normal, np.asarray(other["normal"]))))
        if normal_difference < 0.95 and float(other["score"]) >= 0.95 * float(selected_info["score"]):
            raise SupportPlaneSelectionError(
                "support plane is ambiguous between differently oriented candidates", diagnostics
            )
    diagnostics["selected_index"] = int(selected_info["index"])
    return selected, diagnostics


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
    polygon, metrics = carton_footprint_polygon_from_projected(projected)
    metrics["input_point_count"] = int(len(object_points))
    metrics["elevated_point_count"] = int(len(elevated_points))
    return polygon, metrics


def carton_footprint_polygon_from_projected(
    projected_points: np.ndarray,
) -> tuple[Polygon, dict[str, float | int | dict[str, object]]]:
    """Build a carton OBB from already projected, 2-D balanced support points."""
    projected = _validate_projected_points(projected_points, minimum=64, label="carton")
    component, component_diagnostics = select_footprint_component(projected)
    cleaned = _trim_projected_outliers(component, lower=0.01, upper=0.99)
    polygon = MultiPoint(cleaned).convex_hull.minimum_rotated_rectangle
    polygon = _validate_obb_polygon(polygon)
    return polygon, {
        "component_point_count": int(len(component)),
        "cleaned_point_count": int(len(cleaned)),
        "footprint_area_m2": float(polygon.area),
        "component_diagnostics": component_diagnostics,
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
    inliers: np.ndarray,
    total_points: int,
    max_residual_m: float = 0.010,
    return_retained_indices: bool = False,
) -> SupportPlane | tuple[SupportPlane, np.ndarray]:
    refined = inliers
    retained_indices = np.arange(len(inliers))
    inlier_count = len(refined)
    inlier_fraction = inlier_count / total_points
    if inlier_count < 10_000 or inlier_fraction < 0.10:
        raise FootprintError("support plane has insufficient inliers")

    for _ in range(3):
        point, normal = _fit_plane_svd(refined)
        residuals = np.abs((refined - point) @ normal)
        retained = residuals <= max_residual_m
        if retained.all():
            break
        refined = refined[retained]
        retained_indices = retained_indices[retained]
        inlier_count = len(refined)
        inlier_fraction = inlier_count / total_points
        if inlier_count < 10_000 or inlier_fraction < 0.10:
            raise FootprintError("support plane has insufficient inliers after residual refinement")

    inlier_count = len(refined)
    inlier_fraction = inlier_count / total_points
    if inlier_count < 10_000 or inlier_fraction < 0.10:
        raise FootprintError("support plane has insufficient inliers after residual refinement")

    point, normal = _fit_plane_svd(refined)
    residuals = np.abs((refined - point) @ normal)
    p95_residual_m = float(np.percentile(residuals, 95))
    if p95_residual_m > max_residual_m:
        raise FootprintError(f"support plane residual exceeds {max_residual_m:.3f} m")

    reference_axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u_axis = np.cross(normal, reference_axis)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    plane = SupportPlane(
        point=point,
        normal=normal,
        u_axis=u_axis,
        v_axis=v_axis,
        inlier_count=inlier_count,
        inlier_fraction=inlier_fraction,
        p95_residual_m=p95_residual_m,
    )
    if return_retained_indices:
        return plane, retained_indices
    return plane


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point = points.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(points - point, full_matrices=False)
    normal = right_vectors[-1]
    normal /= np.linalg.norm(normal)
    dominant_index = int(np.argmax(np.abs(normal)))
    if normal[dominant_index] < 0:
        normal = -normal
    return point, normal


def voxel_balance_projected(
    projected_points: np.ndarray, voxel_size_m: float = 0.005
) -> np.ndarray:
    """Keep one point per deterministic 5-mm support-plane (u, v) voxel."""
    array = _validate_projected_points(projected_points, minimum=1, label="carton")
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
) -> RansacOutcome:
    generator = np.random.default_rng(seed)
    best_count = 0
    best_candidate: tuple[np.ndarray, np.ndarray] | None = None
    minimum_trials = 128
    target_trials = 10_000
    trial = 0
    offsets = np.empty_like(points)
    distances = np.empty(len(points), dtype=points.dtype)
    while trial < target_trials:
        indices = _sample_ransac_triplet(generator, population_size=len(points))
        first, second, third = points[indices]
        normal = np.cross(second - first, third - first)
        norm = np.linalg.norm(normal)
        if norm == 0:
            trial += 1
            continue
        normal /= norm
        np.subtract(points, first, out=offsets)
        np.matmul(offsets, normal, out=distances)
        np.abs(distances, out=distances)
        count = int(np.count_nonzero(distances <= threshold_m))
        if count > best_count:
            best_count = count
            best_candidate = (first, normal)
            ratio = min(max(count / len(points), 1e-9), 1.0 - 1e-12)
            target_trials = min(
                10_000,
                max(minimum_trials, int(np.ceil(np.log(1.0 - 0.999) / np.log(1.0 - ratio**3)))),
            )
        trial += 1
        if count == len(points):
            return RansacOutcome(first, normal, trial, True)
    if best_candidate is None:
        raise FootprintError("support plane RANSAC found no non-collinear candidate")
    return RansacOutcome(*best_candidate, trial, False)


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


def _object_centres_inside_hull(centres: np.ndarray, plane: SupportPlane, hull: object) -> float:
    if hull.is_empty:
        return 0.0
    projected_centres = project_to_plane(centres, plane)
    buffered_hull = hull.buffer(0.150)
    return float(np.mean([buffered_hull.covers(MultiPoint([centre])) for centre in projected_centres]))


def select_footprint_component(
    projected_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select the one allowed 20-mm density component with audit diagnostics."""
    projected = _validate_projected_points(projected_points, minimum=4, label="carton")
    eps_m = 0.020
    min_samples = 4
    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(projected)
    valid_labels = labels[labels >= 0]
    populations = np.bincount(valid_labels) if len(valid_labels) else np.asarray([], dtype=int)
    nonzero_populations = populations[populations > 0]
    diagnostics: dict[str, object] = {
        "eps_m": eps_m,
        "min_samples": min_samples,
        "component_count": int(len(nonzero_populations)),
        "non_noise_point_count": int(len(valid_labels)),
        "component_populations": [int(value) for value in nonzero_populations],
        "substantial_component_threshold": None,
    }
    if len(valid_labels) == 0:
        raise FootprintError("carton footprint has no density component", diagnostics)
    threshold = min(32, int(np.ceil(0.20 * len(valid_labels))))
    diagnostics["substantial_component_threshold"] = threshold
    if len(nonzero_populations) > 1 and int(np.sort(nonzero_populations)[-2]) >= threshold:
        raise FootprintError("carton footprint has multiple substantial components", diagnostics)
    greatest_population = int(populations.max())
    label = int(np.flatnonzero(populations == greatest_population)[0])
    return projected[labels == label], diagnostics


def _normalise_object_observations(
    observations: np.ndarray | list[np.ndarray],
) -> list[np.ndarray]:
    if isinstance(observations, np.ndarray):
        return [_validate_points(observations, minimum=64, label="object")]
    if not observations:
        raise FootprintError("support plane requires object observations")
    return [_validate_points(observation, minimum=1, label="object") for observation in observations]


def _validate_projected_points(points: np.ndarray, minimum: int, label: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1:] != (2,):
        raise FootprintError(f"{label} projected points must have shape (N, 2)")
    if len(array) < minimum:
        raise FootprintError(f"{label} projected points require at least {minimum} points")
    if not np.isfinite(array).all():
        raise FootprintError(f"{label} projected points must be finite")
    return array


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
