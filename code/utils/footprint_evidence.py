"""Shadow-only multi-view evidence for frozen ground-stack footprints.

This module is intentionally independent from the formal measurement stage.  It
loads optional camera tensors, reports internal reprojection agreement, and
measures fixed-support-plane leave-one-observation-out sensitivity.  None of the
diagnostics in this module is a formal acceptance gate or calibrated accuracy
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from utils.ground_stack_footprint import (
    FootprintError,
    SupportPlane,
    carton_footprint_polygon_from_projected,
    project_to_plane,
    voxel_balance_projected,
)


DEPTH_TOLERANCE_M = 0.020
MAX_REPROJECTION_SAMPLES = 512
POINT_CONFIDENCE_THRESHOLD = 1.0
_CAMERA_FIELDS = (
    "world_points",
    "world_points_conf",
    "depth",
    "intrinsic",
    "extrinsic",
)


@dataclass(frozen=True)
class EvidenceObservation:
    """One processed-grid mask observation used only for shadow evidence."""

    global_id: str
    image_id: int
    object_id: int
    processed_mask: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class FormalSnapshot:
    """The already-decided formal result consumed without mutation."""

    status: str
    value_m2: float | None
    plane: SupportPlane | None
    polygons: Mapping[str, Polygon]
    union: Polygon | MultiPolygon | None
    rejection_reason: str | None


@dataclass(frozen=True)
class _CameraCache:
    frame_ids: np.ndarray
    world_points: np.ndarray
    confidence: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    extrinsic: np.ndarray


@dataclass(frozen=True)
class _PreparedObservation:
    observation: EvidenceObservation
    frame_index: int
    processed_mask: np.ndarray
    valid_mask: np.ndarray
    qualified_mask: np.ndarray


class EvidenceUnavailable(RuntimeError):
    """Expected absence of optional shadow evidence."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


class CameraContractError(ValueError):
    """Malformed or numerically invalid optional camera data."""


class EvidenceComputationError(ValueError):
    """Non-camera shadow computation produced invalid derived numerics."""


def build_shadow_evidence(
    cache_path: Path,
    *,
    cache_frame_ids: np.ndarray,
    observations: Sequence[EvidenceObservation],
    formal_snapshot: FormalSnapshot,
) -> dict[str, object]:
    """Return JSON-safe, non-authoritative evidence without raising.

    Camera data is deliberately loaded in a separate NPZ read.  Expected
    optional-data absence and camera-contract failures receive distinct statuses;
    every other ordinary evidence exception is isolated as ``failed_evidence``.
    """

    try:
        camera = _load_optional_camera_cache(Path(cache_path), cache_frame_ids)
        if formal_snapshot.plane is None:
            raise EvidenceUnavailable(
                "unavailable_no_formal_geometry",
                "fixed-plane evidence requires a frozen formal support plane",
            )
        return _build_valid_camera_evidence(camera, observations, formal_snapshot)
    except EvidenceUnavailable as error:
        return {"mode": "shadow", "status": error.status, "reason": str(error)}
    except CameraContractError as error:
        return {
            "mode": "shadow",
            "status": "failed_camera_contract",
            "reason": str(error),
        }
    except Exception as error:
        return {"mode": "shadow", "status": "failed_evidence", "reason": str(error)}


def _load_optional_camera_cache(
    cache_path: Path, cache_frame_ids: np.ndarray
) -> _CameraCache:
    try:
        with np.load(cache_path, allow_pickle=False) as loaded:
            missing = [field for field in _CAMERA_FIELDS if field not in loaded.files]
            if missing:
                raise EvidenceUnavailable(
                    "unavailable_missing_camera_fields",
                    "DA3 cache missing optional camera fields: " + ", ".join(missing),
                )
            arrays = {field: loaded[field].copy() for field in _CAMERA_FIELDS}
    except EvidenceUnavailable:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise CameraContractError(f"cannot read optional DA3 camera fields: {error}") from error

    for field, array in arrays.items():
        if np.asarray(array).dtype.kind not in "fiu":
            raise CameraContractError(f"DA3 camera field {field} must be numeric")

    world_points = np.asarray(arrays["world_points"], dtype=np.float64)
    confidence = np.asarray(arrays["world_points_conf"], dtype=np.float64)
    depth = np.asarray(arrays["depth"], dtype=np.float64)
    intrinsic = np.asarray(arrays["intrinsic"], dtype=np.float64)
    extrinsic = np.asarray(arrays["extrinsic"], dtype=np.float64)
    frame_ids = np.asarray(cache_frame_ids)

    if world_points.ndim != 4 or world_points.shape[-1] != 3:
        raise CameraContractError("world_points must have shape (N, H, W, 3)")
    frame_count, height, width, _ = world_points.shape
    if frame_count < 1 or height < 1 or width < 1:
        raise CameraContractError("camera tensors must have nonempty N, H, and W dimensions")
    expected_shapes = {
        "world_points_conf": (frame_count, height, width),
        "depth": (frame_count, height, width, 1),
        "intrinsic": (frame_count, 3, 3),
        "extrinsic": (frame_count, 3, 4),
    }
    actual_shapes = {
        "world_points_conf": confidence.shape,
        "depth": depth.shape,
        "intrinsic": intrinsic.shape,
        "extrinsic": extrinsic.shape,
    }
    for field, expected_shape in expected_shapes.items():
        if actual_shapes[field] != expected_shape:
            raise CameraContractError(
                f"{field} must have shape {expected_shape}, got {actual_shapes[field]}"
            )
    if frame_ids.shape != (frame_count,) or frame_ids.dtype.kind not in "iu":
        raise CameraContractError("cache_frame_ids must be an integer vector aligned to N")
    if len(np.unique(frame_ids)) != frame_count:
        raise CameraContractError("cache_frame_ids must be unique")
    if not all(
        np.isfinite(array).all()
        for array in (world_points, confidence, depth, intrinsic, extrinsic)
    ):
        raise CameraContractError("DA3 optional camera tensors must be finite")
    if np.any(intrinsic[:, 0, 0] <= 0.0) or np.any(intrinsic[:, 1, 1] <= 0.0):
        raise CameraContractError("camera focal lengths must be positive")

    identity = np.eye(3)
    for frame_index in range(frame_count):
        camera_matrix = intrinsic[frame_index]
        try:
            inverse_intrinsic = np.linalg.inv(camera_matrix)
        except np.linalg.LinAlgError as error:
            raise CameraContractError(
                f"intrinsic matrix is not invertible for frame index {frame_index}"
            ) from error
        if not np.isfinite(inverse_intrinsic).all():
            raise CameraContractError(
                f"intrinsic inverse is not finite for frame index {frame_index}"
            )

        rotation = extrinsic[frame_index, :, :3]
        if not np.allclose(
            rotation.T @ rotation, identity, rtol=0.0, atol=1e-4
        ):
            raise CameraContractError(
                f"world-to-camera rotation is not orthonormal for frame index {frame_index}"
            )
        determinant = float(np.linalg.det(rotation))
        if determinant <= 0.0:
            raise CameraContractError(
                f"world-to-camera rotation determinant must be positive for frame index {frame_index}"
            )
        world_to_camera = _homogeneous_world_to_camera(extrinsic[frame_index])
        try:
            camera_to_world = np.linalg.inv(world_to_camera)
        except np.linalg.LinAlgError as error:
            raise CameraContractError(
                f"world-to-camera transform is not invertible for frame index {frame_index}"
            ) from error
        if not np.isfinite(camera_to_world).all():
            raise CameraContractError(
                f"world-to-camera inverse is not finite for frame index {frame_index}"
            )

    return _CameraCache(
        frame_ids=frame_ids.astype(np.int64, copy=True),
        world_points=world_points,
        confidence=confidence,
        depth=depth,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )


def _build_valid_camera_evidence(
    camera: _CameraCache,
    observations: Sequence[EvidenceObservation],
    formal_snapshot: FormalSnapshot,
) -> dict[str, object]:
    plane = formal_snapshot.plane
    if plane is None:
        raise EvidenceUnavailable(
            "unavailable_no_formal_geometry",
            "fixed-plane evidence requires a frozen formal support plane",
        )
    camera_contract = _camera_contract_report(camera)
    prepared = _prepare_observations(camera, observations)
    grouped: dict[str, list[_PreparedObservation]] = {}
    for item in prepared:
        grouped.setdefault(str(item.observation.global_id), []).append(item)

    per_global_id: dict[str, object] = {}
    for global_id in sorted(grouped, key=_global_id_key):
        group = grouped[global_id]
        distinct_image_id_count = len(
            {int(item.observation.image_id) for item in group}
        )
        observation_reports = [
            _observation_report(camera, item, plane) for item in group
        ]
        if distinct_image_id_count < 2:
            cross_view_status = "single_observation_insufficient_cross_view_evidence"
            pairs: list[dict[str, object]] = []
        else:
            cross_view_status = "available"
            pairs = [
                _pairwise_reprojection(camera, source, target)
                for source in group
                for target in group
                if source.observation.image_id != target.observation.image_id
            ]
        if len(group) == 1:
            leave_one_out: list[dict[str, object]] = []
        else:
            leave_one_out = _leave_one_observation_out(
                camera,
                group,
                plane,
                formal_snapshot.polygons.get(global_id),
            )
        per_global_id[global_id] = {
            "distinct_image_id_count": distinct_image_id_count,
            "cross_view_status": cross_view_status,
            "observations": observation_reports,
            "pairs": pairs,
            "leave_one_observation_out": leave_one_out,
        }

    return {
        "mode": "shadow",
        "status": "available",
        "camera_contract": camera_contract,
        "per_global_id": per_global_id,
    }


def _prepare_observations(
    camera: _CameraCache, observations: Sequence[EvidenceObservation]
) -> list[_PreparedObservation]:
    frame_for_id = {
        int(image_id): index for index, image_id in enumerate(camera.frame_ids)
    }
    expected_shape = camera.world_points.shape[1:3]
    prepared: list[_PreparedObservation] = []
    for observation in observations:
        if not isinstance(observation.image_id, (int, np.integer)) or isinstance(
            observation.image_id, (bool, np.bool_)
        ):
            raise ValueError("evidence observation image_id must be an integer")
        if not isinstance(observation.object_id, (int, np.integer)) or isinstance(
            observation.object_id, (bool, np.bool_)
        ):
            raise ValueError("evidence observation object_id must be an integer")
        image_id = int(observation.image_id)
        if image_id not in frame_for_id:
            raise ValueError(f"evidence observation image {image_id} is absent from cache")
        processed_mask = np.asarray(observation.processed_mask, dtype=bool)
        valid_mask = np.asarray(observation.valid_mask, dtype=bool)
        if processed_mask.shape != expected_shape or valid_mask.shape != expected_shape:
            raise ValueError(
                "evidence observation masks must match the processed camera grid"
            )
        frame_index = frame_for_id[image_id]
        qualified = (
            processed_mask
            & valid_mask
            & _qualified_camera_grid(camera, frame_index)
        )
        prepared.append(
            _PreparedObservation(
                observation=observation,
                frame_index=frame_index,
                processed_mask=processed_mask,
                valid_mask=valid_mask,
                qualified_mask=qualified,
            )
        )
    return prepared


def _camera_contract_report(camera: _CameraCache) -> dict[str, object]:
    frame_reports: list[dict[str, object]] = []
    residual_parts: list[np.ndarray] = []
    for frame_index, image_id in enumerate(camera.frame_ids):
        reconstructed = _reconstruct_world_points(
            camera.depth[frame_index],
            camera.intrinsic[frame_index],
            camera.extrinsic[frame_index],
        )
        try:
            with np.errstate(over="raise", invalid="raise"):
                residual_vectors = reconstructed - camera.world_points[frame_index]
        except FloatingPointError as error:
            raise CameraContractError(
                f"source-world residual arithmetic failed for image {int(image_id)}"
            ) from error
        residuals = _stable_vector_norm(
            residual_vectors,
            error_type=CameraContractError,
            label=f"source-world residual for image {int(image_id)}",
        ).reshape(-1)
        residual_parts.append(residuals)
        frame_reports.append(
            {
                "image_id": int(image_id),
                **_residual_summary(
                    residuals,
                    error_type=CameraContractError,
                    label=f"source-world residual for image {int(image_id)}",
                ),
            }
        )
    all_residuals = np.concatenate(residual_parts)
    return {
        "status": "valid",
        "frame_count": int(len(camera.frame_ids)),
        "processed_shape_hw": [
            int(camera.world_points.shape[1]),
            int(camera.world_points.shape[2]),
        ],
        "rotation_orthonormal_atol": 1e-4,
        "depth_tolerance_m": DEPTH_TOLERANCE_M,
        "source_world_reconstruction_residual_m": {
            **_residual_summary(
                all_residuals,
                error_type=CameraContractError,
                label="source-world residual",
            ),
            "frames": frame_reports,
        },
    }


def _reconstruct_world_points(
    depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray
) -> np.ndarray:
    height, width, _ = depth.shape
    x_pixels, y_pixels = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
        indexing="xy",
    )
    pixels = np.stack(
        [x_pixels, y_pixels, np.ones((height, width), dtype=np.float64)], axis=-1
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            rays = pixels @ np.linalg.inv(intrinsic).T
            camera_points = rays * depth
            homogeneous_camera = np.concatenate(
                [camera_points, np.ones((height, width, 1), dtype=np.float64)],
                axis=-1,
            )
            camera_to_world = np.linalg.inv(_homogeneous_world_to_camera(extrinsic))
            reconstructed = (homogeneous_camera @ camera_to_world.T)[..., :3]
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise CameraContractError(
            "depth/intrinsic/extrinsic world reconstruction arithmetic failed"
        ) from error
    _require_finite(
        reconstructed,
        error_type=CameraContractError,
        label="reconstructed source-world points",
    )
    return reconstructed


def _homogeneous_world_to_camera(extrinsic: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3] = extrinsic
    return transform


def _observation_report(
    camera: _CameraCache, item: _PreparedObservation, plane: SupportPlane
) -> dict[str, object]:
    frame_index = item.frame_index
    masked_confidence = camera.confidence[frame_index][
        item.processed_mask & item.valid_mask
    ]
    points = camera.world_points[frame_index][item.qualified_mask]
    if len(points):
        try:
            with np.errstate(over="raise", invalid="raise"):
                heights = (points - plane.point) @ plane.normal
        except FloatingPointError as error:
            raise EvidenceComputationError(
                "observation support-plane height arithmetic failed"
            ) from error
        _require_finite(
            heights,
            error_type=EvidenceComputationError,
            label="observation support-plane heights",
        )
    else:
        heights = np.asarray([])
    camera_to_world = np.linalg.inv(
        _homogeneous_world_to_camera(camera.extrinsic[frame_index])
    )
    viewing_direction = camera_to_world[:3, :3] @ np.array([0.0, 0.0, 1.0])
    viewing_norm = _stable_vector_norm(
        viewing_direction,
        error_type=EvidenceComputationError,
        label="camera viewing direction",
    )
    if float(viewing_norm) <= 0.0:
        raise EvidenceComputationError("camera viewing direction has zero norm")
    viewing_direction /= float(viewing_norm)
    _require_finite(
        camera_to_world[:3, 3],
        viewing_direction,
        error_type=EvidenceComputationError,
        label="observation camera pose diagnostics",
    )
    return {
        "image_id": int(item.observation.image_id),
        "object_id": int(item.observation.object_id),
        "source_mask_pixel_count": int(np.count_nonzero(item.processed_mask)),
        "valid_point_count": int(len(points)),
        "confidence": _value_summary(masked_confidence),
        "elevated_point_fraction": (
            float(np.mean(heights > 0.015)) if len(heights) else None
        ),
        "camera_centre_world": camera_to_world[:3, 3].tolist(),
        "viewing_direction_world": viewing_direction.tolist(),
    }


def _pairwise_reprojection(
    camera: _CameraCache,
    source: _PreparedObservation,
    target: _PreparedObservation,
) -> dict[str, object]:
    source_indices = np.flatnonzero(source.qualified_mask.reshape(-1))[
        :MAX_REPROJECTION_SAMPLES
    ]
    source_points = camera.world_points[source.frame_index].reshape(-1, 3)[
        source_indices
    ]
    target_extrinsic = camera.extrinsic[target.frame_index]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            camera_points = (
                source_points @ target_extrinsic[:, :3].T
                + target_extrinsic[:, 3]
            )
            projected = camera_points @ camera.intrinsic[target.frame_index].T
    except FloatingPointError as error:
        raise EvidenceComputationError(
            "pairwise world-to-target projection arithmetic failed"
        ) from error
    _require_finite(
        camera_points,
        projected,
        error_type=EvidenceComputationError,
        label="pairwise world-to-target projection",
    )
    projected_depth = camera_points[:, 2]
    behind = projected_depth <= 0.0
    denominators = projected[:, 2]
    projectable = (~behind) & (denominators != 0.0)
    x_coordinates = np.zeros(len(source_points), dtype=np.float64)
    y_coordinates = np.zeros(len(source_points), dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            x_coordinates[projectable] = (
                projected[projectable, 0] / denominators[projectable]
            )
            y_coordinates[projectable] = (
                projected[projectable, 1] / denominators[projectable]
            )
    except FloatingPointError as error:
        raise EvidenceComputationError(
            "pairwise target-grid coordinate arithmetic failed"
        ) from error
    _require_finite(
        x_coordinates[projectable],
        y_coordinates[projectable],
        error_type=EvidenceComputationError,
        label="pairwise target-grid coordinates",
    )
    height, width = target.processed_mask.shape
    inside = (
        projectable
        & (x_coordinates >= 0.0)
        & (x_coordinates <= width - 1)
        & (y_coordinates >= 0.0)
        & (y_coordinates <= height - 1)
    )
    outside = (~behind) & ~inside
    inside_indices = np.flatnonzero(inside)
    x_pixels = np.rint(x_coordinates[inside_indices]).astype(np.int64)
    y_pixels = np.rint(y_coordinates[inside_indices]).astype(np.int64)
    target_depth = camera.depth[target.frame_index, y_pixels, x_pixels, 0]
    target_is_valid = (
        target.valid_mask[y_pixels, x_pixels]
        & _qualified_camera_grid(camera, target.frame_index)[y_pixels, x_pixels]
        & np.isfinite(target_depth)
        & (target_depth > 0.0)
    )
    valid_inside_indices = inside_indices[target_is_valid]
    valid_target_depth = target_depth[target_is_valid]
    try:
        with np.errstate(over="raise", invalid="raise"):
            signed_residual = (
                projected_depth[valid_inside_indices] - valid_target_depth
            )
    except FloatingPointError as error:
        raise EvidenceComputationError(
            "pairwise target-depth residual arithmetic failed"
        ) from error
    _require_finite(
        signed_residual,
        error_type=EvidenceComputationError,
        label="pairwise target-depth residual",
    )
    occluded = signed_residual > DEPTH_TOLERANCE_M
    foreground_conflict = signed_residual < -DEPTH_TOLERANCE_M
    visible = ~(occluded | foreground_conflict)
    target_support = target.processed_mask[
        y_pixels[target_is_valid], x_pixels[target_is_valid]
    ]
    visible_supported = visible & target_support
    visible_unsupported = visible & ~target_support
    absolute_residual = np.abs(signed_residual)
    residual_summary = _residual_summary(
        absolute_residual,
        error_type=EvidenceComputationError,
        label="pairwise target-depth residual",
    )
    return {
        "source_image_id": int(source.observation.image_id),
        "source_object_id": int(source.observation.object_id),
        "target_image_id": int(target.observation.image_id),
        "target_object_id": int(target.observation.object_id),
        "source_sample_count": int(len(source_indices)),
        "behind_camera_count": int(np.count_nonzero(behind)),
        "outside_grid_count": int(np.count_nonzero(outside)),
        "invalid_target_count": int(np.count_nonzero(~target_is_valid)),
        "eligible_count": int(len(valid_inside_indices)),
        "occluded_count": int(np.count_nonzero(occluded)),
        "foreground_conflict_count": int(np.count_nonzero(foreground_conflict)),
        "visible_consistent_count": int(np.count_nonzero(visible)),
        "visible_mask_supported_count": int(np.count_nonzero(visible_supported)),
        "visible_mask_unsupported_count": int(np.count_nonzero(visible_unsupported)),
        "depth_residual_p50_m": residual_summary["p50"],
        "depth_residual_p95_m": residual_summary["p95"],
    }


def _qualified_camera_grid(camera: _CameraCache, frame_index: int) -> np.ndarray:
    world_points = camera.world_points[frame_index]
    confidence = camera.confidence[frame_index]
    return (
        np.isfinite(world_points).all(axis=-1)
        & np.any(world_points != 0.0, axis=-1)
        & np.isfinite(confidence)
        & (confidence >= POINT_CONFIDENCE_THRESHOLD)
    )


def _leave_one_observation_out(
    camera: _CameraCache,
    observations: Sequence[_PreparedObservation],
    plane: SupportPlane,
    full_polygon: Polygon | None,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for removed_index, removed in enumerate(observations):
        identity = {
            "removed_observation_index": removed_index,
            "image_id": int(removed.observation.image_id),
            "object_id": int(removed.observation.object_id),
        }
        if full_polygon is None:
            results.append(
                {
                    **identity,
                    "status": "unavailable_missing_formal_polygon",
                }
            )
            continue
        remaining = [
            item for index, item in enumerate(observations) if index != removed_index
        ]
        try:
            polygon = _fixed_plane_polygon(camera, remaining, plane)
            results.append(
                {
                    **identity,
                    "status": "available",
                    **_polygon_change_metrics(full_polygon, polygon),
                }
            )
        except FootprintError as error:
            results.append(
                {
                    **identity,
                    "status": "insufficient_geometry_after_removal",
                    "reason": str(error),
                }
            )
    return results


def _fixed_plane_polygon(
    camera: _CameraCache,
    observations: Sequence[_PreparedObservation],
    plane: SupportPlane,
) -> Polygon:
    if not observations:
        raise FootprintError("leave-one-out has no remaining observations")
    point_parts: list[np.ndarray] = []
    for item in observations:
        points = camera.world_points[item.frame_index][item.qualified_mask]
        if len(points) < 32:
            raise FootprintError(
                "leave-one-out observation has fewer than 32 valid masked points"
            )
        point_parts.append(points)
    fused = np.concatenate(point_parts)
    try:
        with np.errstate(over="raise", invalid="raise"):
            heights = (fused - plane.point) @ plane.normal
    except FloatingPointError as error:
        raise EvidenceComputationError(
            "leave-one-out support-plane height arithmetic failed"
        ) from error
    _require_finite(
        heights,
        error_type=EvidenceComputationError,
        label="leave-one-out support-plane heights",
    )
    elevated = fused[heights > 0.015]
    if len(elevated) < 64:
        raise FootprintError(
            "leave-one-out global id has fewer than 64 elevated points"
        )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            projected = project_to_plane(elevated, plane)
            voxel_coordinates = projected / 0.005
    except FloatingPointError as error:
        raise EvidenceComputationError(
            "leave-one-out support-plane projection arithmetic failed"
        ) from error
    _require_finite(
        projected,
        voxel_coordinates,
        error_type=EvidenceComputationError,
        label="leave-one-out projected points",
    )
    integer_limit = float(np.iinfo(np.int64).max)
    if np.any(np.abs(voxel_coordinates) > integer_limit):
        raise EvidenceComputationError(
            "leave-one-out voxel coordinates exceed integer representation"
        )
    balanced = voxel_balance_projected(projected, voxel_size_m=0.005)
    if len(balanced) < 64:
        raise FootprintError(
            "leave-one-out global id has fewer than 64 projected voxel points"
        )
    polygon, _ = carton_footprint_polygon_from_projected(balanced)
    _require_finite(
        np.asarray(polygon.exterior.coords, dtype=np.float64),
        np.asarray([polygon.area], dtype=np.float64),
        error_type=EvidenceComputationError,
        label="leave-one-out polygon",
    )
    return polygon


def _polygon_change_metrics(full_polygon: Polygon, loo_polygon: Polygon) -> dict[str, object]:
    if not isinstance(full_polygon, Polygon):
        raise FootprintError("frozen full-data footprint polygon is invalid")
    _require_finite(
        np.asarray(full_polygon.exterior.coords, dtype=np.float64),
        error_type=EvidenceComputationError,
        label="frozen full-data footprint polygon",
    )
    if full_polygon.is_empty or not full_polygon.is_valid:
        raise FootprintError("frozen full-data footprint polygon is invalid")
    intersection_area = float(full_polygon.intersection(loo_polygon).area)
    union_area = float(full_polygon.union(loo_polygon).area)
    _require_finite(
        np.asarray([intersection_area, union_area], dtype=np.float64),
        error_type=EvidenceComputationError,
        label="leave-one-out overlap metrics",
    )
    if union_area <= 0.0:
        raise EvidenceComputationError(
            "leave-one-out polygon union area must be positive"
        )
    full_descriptor = _polygon_descriptor(full_polygon)
    loo_descriptor = _polygon_descriptor(loo_polygon)
    angle_difference = abs(loo_descriptor["angle_deg"] - full_descriptor["angle_deg"])
    angle_delta = min(angle_difference, 180.0 - angle_difference)
    centre_delta = _stable_vector_norm(
        np.asarray(loo_descriptor["centre"])
        - np.asarray(full_descriptor["centre"]),
        error_type=EvidenceComputationError,
        label="leave-one-out centre delta",
    )
    metrics = {
        "polygon_iou": intersection_area / union_area,
        "hausdorff_distance_m": float(full_polygon.hausdorff_distance(loo_polygon)),
        "centre_delta_m": float(centre_delta),
        "angle_delta_deg": float(angle_delta),
        "side_length_deltas_m": (
            np.asarray(loo_descriptor["side_lengths_m"])
            - np.asarray(full_descriptor["side_lengths_m"])
        ).tolist(),
        "area_delta_m2": float(loo_polygon.area - full_polygon.area),
        "full_centre": full_descriptor["centre"],
        "loo_centre": loo_descriptor["centre"],
        "full_angle_deg": full_descriptor["angle_deg"],
        "loo_angle_deg": loo_descriptor["angle_deg"],
        "full_side_lengths_m": full_descriptor["side_lengths_m"],
        "loo_side_lengths_m": loo_descriptor["side_lengths_m"],
        "full_area_m2": float(full_polygon.area),
        "loo_area_m2": float(loo_polygon.area),
    }
    _require_finite(
        np.asarray(
            [
                metrics["polygon_iou"],
                metrics["hausdorff_distance_m"],
                metrics["centre_delta_m"],
                metrics["angle_delta_deg"],
                metrics["area_delta_m2"],
                metrics["full_area_m2"],
                metrics["loo_area_m2"],
                *metrics["side_length_deltas_m"],
            ],
            dtype=np.float64,
        ),
        error_type=EvidenceComputationError,
        label="leave-one-out polygon change metrics",
    )
    return metrics


def _polygon_descriptor(polygon: Polygon) -> dict[str, object]:
    coordinates = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    if coordinates.shape != (4, 2):
        raise FootprintError("frozen or leave-one-out footprint is not a four-edge OBB")
    edges = np.roll(coordinates, -1, axis=0) - coordinates
    lengths = _stable_vector_norm(
        edges,
        error_type=EvidenceComputationError,
        label="footprint polygon edge lengths",
    )
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise FootprintError("frozen or leave-one-out footprint has invalid sides")
    ordered_lengths = np.sort(lengths)
    side_lengths = [
        float(np.mean(ordered_lengths[:2])),
        float(np.mean(ordered_lengths[2:])),
    ]
    longest = float(np.max(lengths))
    candidate_angles = [
        float(np.degrees(np.arctan2(edge[1], edge[0])) % 180.0)
        for edge, length in zip(edges, lengths)
        if np.isclose(length, longest, rtol=0.0, atol=1e-10)
    ]
    return {
        "centre": [float(polygon.centroid.x), float(polygon.centroid.y)],
        "angle_deg": min(candidate_angles),
        "side_lengths_m": side_lengths,
    }


def _require_finite(
    *arrays: np.ndarray,
    error_type: type[Exception],
    label: str,
) -> None:
    if not all(np.isfinite(np.asarray(array)).all() for array in arrays):
        raise error_type(f"{label} produced non-finite derived values")


def _stable_vector_norm(
    vectors: np.ndarray,
    *,
    error_type: type[Exception],
    label: str,
) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    _require_finite(array, error_type=error_type, label=label)
    absolute = np.abs(array)
    scale = np.max(absolute, axis=-1, keepdims=True)
    normalized = np.zeros_like(absolute)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            np.divide(absolute, scale, out=normalized, where=scale != 0.0)
            norm = scale[..., 0] * np.sqrt(np.sum(normalized * normalized, axis=-1))
    except FloatingPointError as error:
        raise error_type(f"{label} norm arithmetic failed") from error
    _require_finite(norm, error_type=error_type, label=label)
    return norm


def _residual_summary(
    values: np.ndarray,
    *,
    error_type: type[Exception],
    label: str,
) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    _require_finite(array, error_type=error_type, label=label)
    summary = {
        "count": int(len(array)),
        "p50": _finite_percentile(
            array, 50.0, error_type=error_type, label=label
        ),
        "p95": _finite_percentile(
            array, 95.0, error_type=error_type, label=label
        ),
        "max": float(np.max(array)),
    }
    _require_finite(
        np.asarray([summary["p50"], summary["p95"], summary["max"]]),
        error_type=error_type,
        label=f"{label} summary",
    )
    return summary


def _finite_percentile(
    values: np.ndarray,
    percentile: float,
    *,
    error_type: type[Exception],
    label: str,
) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    position = (len(ordered) - 1) * percentile / 100.0
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))
    fraction = position - lower_index
    try:
        with np.errstate(over="raise", invalid="raise"):
            value = (
                (1.0 - fraction) * ordered[lower_index]
                + fraction * ordered[upper_index]
            )
    except FloatingPointError as error:
        raise error_type(f"{label} percentile arithmetic failed") from error
    _require_finite(
        np.asarray([value]),
        error_type=error_type,
        label=f"{label} percentile",
    )
    return float(value)


def _value_summary(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    _require_finite(
        array,
        error_type=EvidenceComputationError,
        label="observation confidence",
    )
    summary = {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p50": _finite_percentile(
            array,
            50.0,
            error_type=EvidenceComputationError,
            label="observation confidence",
        ),
        "p95": _finite_percentile(
            array,
            95.0,
            error_type=EvidenceComputationError,
            label="observation confidence",
        ),
        "max": float(np.max(array)),
    }
    _require_finite(
        np.asarray(
            [summary["min"], summary["p50"], summary["p95"], summary["max"]]
        ),
        error_type=EvidenceComputationError,
        label="observation confidence summary",
    )
    return summary


def _global_id_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):020d}") if value.isdigit() else (1, value)
