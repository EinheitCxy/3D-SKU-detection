import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import box

from utils.footprint_evidence import (
    EvidenceObservation,
    FormalSnapshot,
    build_shadow_evidence,
)
from utils.ground_stack_footprint import SupportPlane


def _plane() -> SupportPlane:
    return SupportPlane(
        point=np.zeros(3),
        normal=np.array([0.0, 0.0, 1.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        inlier_count=10_000,
        inlier_fraction=1.0,
        p95_residual_m=0.0,
    )


def _snapshot(polygon=None) -> FormalSnapshot:
    polygon = polygon or box(0.0, 0.0, 1.0, 1.0)
    return FormalSnapshot(
        status="accepted",
        value_m2=float(polygon.area),
        plane=_plane(),
        polygons={"1": polygon},
        union=polygon,
        rejection_reason=None,
    )


def _identity_camera_arrays(
    frame_count: int,
    height: int,
    width: int,
    *,
    focal_length: float = 1.0,
) -> dict[str, np.ndarray]:
    depth = np.ones((frame_count, height, width, 1), dtype=np.float64)
    intrinsic = np.repeat(np.eye(3, dtype=np.float64)[None], frame_count, axis=0)
    intrinsic[:, 0, 0] = focal_length
    intrinsic[:, 1, 1] = focal_length
    extrinsic = np.repeat(
        np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)[None],
        frame_count,
        axis=0,
    )
    world_points = _world_points_for_identity_camera(depth, focal_length)
    return {
        "world_points": world_points,
        "world_points_conf": np.ones((frame_count, height, width), dtype=np.float64),
        "depth": depth,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
    }


def _world_points_for_identity_camera(
    depth: np.ndarray, focal_length: float
) -> np.ndarray:
    frame_count, height, width, _ = depth.shape
    x_pixels, y_pixels = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
        indexing="xy",
    )
    z_values = depth[..., 0]
    return np.stack(
        [
            np.broadcast_to(x_pixels, (frame_count, height, width))
            * z_values
            / focal_length,
            np.broadcast_to(y_pixels, (frame_count, height, width))
            * z_values
            / focal_length,
            z_values,
        ],
        axis=-1,
    )


def _write_npz(path: Path, fields: dict[str, np.ndarray]) -> Path:
    np.savez_compressed(path, **fields)
    return path


def _observation(
    image_id: int,
    mask: np.ndarray,
    *,
    object_id: int = 0,
    valid_mask: np.ndarray | None = None,
    source_mask: np.ndarray | None = None,
    source_to_processed_affine: np.ndarray | None = None,
) -> EvidenceObservation:
    return EvidenceObservation(
        global_id="1",
        image_id=image_id,
        object_id=object_id,
        source_mask=(mask.copy() if source_mask is None else source_mask),
        source_to_processed_affine=(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            if source_to_processed_affine is None
            else source_to_processed_affine
        ),
        processed_mask=mask,
        valid_mask=np.ones_like(mask, dtype=bool) if valid_mask is None else valid_mask,
    )


def _formal_only_geometry_fields(height: int, width: int) -> dict[str, np.ndarray]:
    x_pixels, y_pixels = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
        indexing="xy",
    )
    return {
        "world_points": np.stack(
            [x_pixels * 0.01, y_pixels * 0.01, np.full_like(x_pixels, 0.50)],
            axis=-1,
        )[None],
        "world_points_conf": np.ones((1, height, width), dtype=np.float64),
    }


def test_missing_camera_fields_returns_unavailable_without_raise(tmp_path):
    cache_path = _write_npz(
        tmp_path / "formal_only.npz",
        {
            "world_points": np.zeros((1, 2, 2, 3), dtype=np.float32),
            "world_points_conf": np.ones((1, 2, 2), dtype=np.float32),
        },
    )

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([0], dtype=np.int32),
        observations=(),
        formal_snapshot=_snapshot(),
    )

    assert evidence["mode"] == "shadow"
    assert evidence["status"] == "unavailable_missing_camera_fields"
    assert evidence["mask_robustness"]["status"] == "available"


def test_source_space_one_pixel_robustness_precedes_nonunit_affine_warp(tmp_path):
    """Catches eroding the enlarged processed mask instead of the source mask."""
    source_mask = np.zeros((12, 20), dtype=np.uint8)
    source_mask[2:10, 3:17] = 1
    affine = np.array([[3.0, 0.0, 1.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    processed_shape = (36, 60)
    processed_mask = cv2.warpAffine(
        source_mask,
        affine.astype(np.float32),
        (processed_shape[1], processed_shape[0]),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    cache_path = _write_npz(
        tmp_path / "formal_geometry_only.npz",
        _formal_only_geometry_fields(*processed_shape),
    )

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([5], dtype=np.int32),
        observations=(
            _observation(
                5,
                processed_mask,
                source_mask=source_mask.astype(bool),
                source_to_processed_affine=affine,
            ),
        ),
        formal_snapshot=_snapshot(box(0.0, 0.0, 0.59, 0.35)),
    )

    kernel = np.ones((3, 3), dtype=np.uint8)
    source_eroded = cv2.erode(source_mask, kernel, iterations=1)
    expected_processed = cv2.warpAffine(
        source_eroded,
        affine.astype(np.float32),
        (processed_shape[1], processed_shape[0]),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    wrong_processed = cv2.erode(
        processed_mask.astype(np.uint8), kernel, iterations=1
    ).astype(bool)
    eroded = evidence["mask_robustness"]["variants"]["eroded"]

    assert evidence["status"] == "unavailable_missing_camera_fields"
    assert evidence["mask_robustness"]["status"] == "available"
    assert eroded["mask_counts"][0] == {
        "image_id": 5,
        "object_id": 0,
        "source_mask_pixel_count": int(source_eroded.sum()),
        "processed_mask_pixel_count": int(expected_processed.sum()),
    }
    assert expected_processed.sum() != wrong_processed.sum()
    assert eroded["status"] == "accepted"
    json.dumps(evidence, allow_nan=False)


def test_thin_source_mask_erosion_reports_rejection_transition_without_partial_geometry(
    tmp_path,
):
    """Catches substituting partial polygons when a perturbation loses one ID."""
    source_mask = np.zeros((5, 30), dtype=np.uint8)
    source_mask[2:3, 1:29] = 1
    affine = np.array([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5]], dtype=np.float64)
    processed_shape = (20, 120)
    processed_mask = cv2.warpAffine(
        source_mask,
        affine.astype(np.float32),
        (processed_shape[1], processed_shape[0]),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    cache_path = _write_npz(
        tmp_path / "thin_source_mask.npz",
        _formal_only_geometry_fields(*processed_shape),
    )

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([11], dtype=np.int32),
        observations=(
            _observation(
                11,
                processed_mask,
                source_mask=source_mask.astype(bool),
                source_to_processed_affine=affine,
            ),
        ),
        formal_snapshot=_snapshot(box(0.0, 0.0, 1.19, 0.19)),
    )

    eroded = evidence["mask_robustness"]["variants"]["eroded"]
    robustness = evidence["mask_robustness"]
    assert eroded["status"] == "rejected"
    assert eroded["value_m2"] is None
    assert eroded["rejection_transition"] == "accepted_to_rejected"
    assert isinstance(eroded["reason"], str) and eroded["reason"]
    assert "polygons" not in eroded
    assert robustness["area_interval_m2"] is None
    json.dumps(evidence, allow_nan=False)


def test_camera_absence_does_not_disable_available_mask_robustness(tmp_path):
    source_mask = np.ones((20, 20), dtype=bool)
    cache_path = _write_npz(
        tmp_path / "no_camera_but_geometry.npz",
        _formal_only_geometry_fields(20, 20),
    )

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([4], dtype=np.int32),
        observations=(_observation(4, source_mask),),
        formal_snapshot=_snapshot(box(0.0, 0.0, 0.19, 0.19)),
    )

    assert evidence["status"] == "unavailable_missing_camera_fields"
    assert evidence["mask_robustness"]["status"] == "available"
    assert evidence["mask_robustness"]["variants"]["original"]["status"] == "accepted"
    assert evidence["mask_robustness"]["area_interval_m2"] is not None
    json.dumps(evidence, allow_nan=False)


def test_absent_formal_plane_keeps_all_shadow_geometry_unavailable(tmp_path):
    source_mask = np.ones((20, 20), dtype=bool)
    cache_path = _write_npz(
        tmp_path / "no_formal_plane.npz", _formal_only_geometry_fields(20, 20)
    )
    snapshot = FormalSnapshot(
        status="rejected",
        value_m2=None,
        plane=None,
        polygons={},
        union=None,
        rejection_reason="formal plane selection failed",
    )

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([4], dtype=np.int32),
        observations=(_observation(4, source_mask),),
        formal_snapshot=snapshot,
    )

    assert evidence["status"] == "unavailable_no_formal_geometry"
    assert evidence["mask_robustness"]["status"] == "unavailable_no_formal_geometry"
    json.dumps(evidence, allow_nan=False)


def test_bidirectional_reprojection_classifies_occluded_and_foreground_conflict(
    tmp_path,
):
    fields = _identity_camera_arrays(2, 2, 2)
    fields["depth"][1, 0, 0, 0] = 0.90
    fields["depth"][1, 0, 1, 0] = 1.10
    fields["world_points"] = _world_points_for_identity_camera(fields["depth"], 1.0)
    cache_path = _write_npz(tmp_path / "two_view.npz", fields)
    mask = np.array([[True, True], [False, False]])

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([0, 1], dtype=np.int32),
        observations=(_observation(0, mask), _observation(1, mask)),
        formal_snapshot=_snapshot(),
    )

    pairs = evidence["per_global_id"]["1"]["pairs"]
    assert [(pair["source_image_id"], pair["target_image_id"]) for pair in pairs] == [
        (0, 1),
        (1, 0),
    ]
    assert pairs[0]["occluded_count"] == 1
    assert pairs[0]["foreground_conflict_count"] == 1
    assert pairs[0]["visible_consistent_count"] == 0
    assert pairs[1]["occluded_count"] == 1
    assert pairs[1]["foreground_conflict_count"] == 1


def test_loo_metrics_change_when_the_only_side_view_is_removed(tmp_path):
    fields = _identity_camera_arrays(3, 101, 101, focal_length=100.0)
    cache_path = _write_npz(tmp_path / "three_view.npz", fields)
    first = np.zeros((101, 101), dtype=bool)
    first[20:81, 0:61] = True
    second = np.zeros((101, 101), dtype=bool)
    second[:, 20:81] = True
    side = np.zeros((101, 101), dtype=bool)
    side[20:81, 40:101] = True

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([0, 1, 2], dtype=np.int32),
        observations=(
            _observation(0, first),
            _observation(1, second),
            _observation(2, side),
        ),
        formal_snapshot=_snapshot(),
    )

    loo = evidence["per_global_id"]["1"]["leave_one_observation_out"]
    assert [item["status"] for item in loo] == ["available"] * 3
    assert max(item["polygon_iou"] for item in loo) < 1.0
    assert loo[2]["image_id"] == 2
    assert loo[2]["centre_delta_m"] > 0.0
    assert loo[2]["area_delta_m2"] < 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        "bad_depth_shape",
        "nonfinite_world_points",
        "nonpositive_focal",
        "singular_intrinsic",
        "nonorthonormal_rotation",
        "negative_rotation_determinant",
    ],
)
def test_invalid_camera_contract_returns_failed_without_raise(tmp_path, mutation):
    fields = _identity_camera_arrays(1, 3, 4)
    if mutation == "bad_depth_shape":
        fields["depth"] = fields["depth"][..., 0]
    elif mutation == "nonfinite_world_points":
        fields["world_points"][0, 0, 0, 0] = np.nan
    elif mutation == "nonpositive_focal":
        fields["intrinsic"][0, 0, 0] = 0.0
    elif mutation == "singular_intrinsic":
        fields["intrinsic"][0, 2] = 0.0
    elif mutation == "nonorthonormal_rotation":
        fields["extrinsic"][0, 0, 0] = 2.0
    else:
        fields["extrinsic"][0, 0, 0] = -1.0
    cache_path = _write_npz(tmp_path / f"{mutation}.npz", fields)

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([0], dtype=np.int32),
        observations=(),
        formal_snapshot=_snapshot(),
    )

    assert evidence["mode"] == "shadow"
    assert evidence["status"] == "failed_camera_contract"
    assert isinstance(evidence["reason"], str) and evidence["reason"]


def test_valid_evidence_is_json_serializable_and_reports_reconstruction_residual(
    tmp_path,
):
    fields = _identity_camera_arrays(2, 4, 5, focal_length=4.0)
    cache_path = _write_npz(tmp_path / "json_safe.npz", fields)
    mask = np.ones((4, 5), dtype=bool)

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([7, 9], dtype=np.int64),
        observations=(_observation(7, mask), _observation(9, mask)),
        formal_snapshot=_snapshot(),
    )

    assert evidence["status"] == "available"
    assert evidence["camera_contract"]["status"] == "valid"
    residual = evidence["camera_contract"]["source_world_reconstruction_residual_m"]
    assert residual["p50"] == pytest.approx(0.0)
    assert residual["p95"] == pytest.approx(0.0)
    json.dumps(evidence, allow_nan=False)


def test_pairwise_sampling_uses_first_512_qualified_flattened_indices(tmp_path):
    fields = _identity_camera_arrays(2, 24, 24, focal_length=100.0)
    cache_path = _write_npz(tmp_path / "sample_cap.npz", fields)
    source_mask = np.ones((24, 24), dtype=bool)
    target_mask = np.zeros((24, 24), dtype=bool)
    target_mask.reshape(-1)[:512] = True

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([0, 1], dtype=np.int32),
        observations=(
            _observation(0, source_mask),
            _observation(1, target_mask),
        ),
        formal_snapshot=_snapshot(box(0.0, 0.0, 0.23, 0.23)),
    )

    first_direction = evidence["per_global_id"]["1"]["pairs"][0]
    assert first_direction["source_sample_count"] == 512
    assert first_direction["visible_mask_supported_count"] == 512
    assert first_direction["visible_mask_unsupported_count"] == 0


def test_single_observation_is_clearly_marked_insufficient(tmp_path):
    fields = _identity_camera_arrays(1, 12, 12, focal_length=100.0)
    cache_path = _write_npz(tmp_path / "single_view.npz", fields)
    mask = np.ones((12, 12), dtype=bool)

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([3], dtype=np.int32),
        observations=(_observation(3, mask),),
        formal_snapshot=_snapshot(box(0.0, 0.0, 0.11, 0.11)),
    )

    per_id = evidence["per_global_id"]["1"]
    assert per_id["distinct_image_id_count"] == 1
    assert per_id["cross_view_status"] == (
        "single_observation_insufficient_cross_view_evidence"
    )
    assert per_id["pairs"] == []
    assert per_id["leave_one_observation_out"] == []


def test_duplicate_observations_from_one_image_are_not_cross_view_evidence(tmp_path):
    fields = _identity_camera_arrays(1, 12, 12, focal_length=100.0)
    cache_path = _write_npz(tmp_path / "duplicate_same_image.npz", fields)
    mask = np.ones((12, 12), dtype=bool)

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([3], dtype=np.int32),
        observations=(
            _observation(3, mask, object_id=0),
            _observation(3, mask, object_id=1),
        ),
        formal_snapshot=_snapshot(box(0.0, 0.0, 0.11, 0.11)),
    )

    per_id = evidence["per_global_id"]["1"]
    assert per_id["distinct_image_id_count"] == 1
    assert per_id["cross_view_status"] == (
        "single_observation_insufficient_cross_view_evidence"
    )
    assert per_id["pairs"] == []


def test_multiview_pairs_never_compare_observations_from_the_same_image(tmp_path):
    fields = _identity_camera_arrays(2, 12, 12, focal_length=100.0)
    cache_path = _write_npz(tmp_path / "multiview_with_duplicate.npz", fields)
    mask = np.ones((12, 12), dtype=bool)

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([3, 4], dtype=np.int32),
        observations=(
            _observation(3, mask, object_id=0),
            _observation(3, mask, object_id=1),
            _observation(4, mask, object_id=0),
        ),
        formal_snapshot=_snapshot(box(0.0, 0.0, 0.11, 0.11)),
    )

    pairs = evidence["per_global_id"]["1"]["pairs"]
    assert len(pairs) == 4
    assert all(pair["source_image_id"] != pair["target_image_id"] for pair in pairs)


def test_pairwise_reprojection_excludes_invalid_target_pixels_from_all_depth_counts(
    tmp_path,
):
    fields = _identity_camera_arrays(2, 1, 5)
    fields["world_points_conf"][1, 0, 1] = 0.5
    fields["world_points"][1, 0, 2] = 0.0
    fields["depth"][1, 0, 3, 0] = 0.0
    fields["world_points"][1, 0, 3] = 0.0
    cache_path = _write_npz(tmp_path / "invalid_target_pixels.npz", fields)
    mask = np.ones((1, 5), dtype=bool)
    target_valid_mask = mask.copy()
    target_valid_mask[0, 0] = False

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([0, 1], dtype=np.int32),
        observations=(
            _observation(0, mask),
            _observation(1, mask, valid_mask=target_valid_mask),
        ),
        formal_snapshot=_snapshot(),
    )

    forward, reverse = evidence["per_global_id"]["1"]["pairs"]
    assert forward["source_sample_count"] == 5
    assert forward["invalid_target_count"] == 4
    assert forward["eligible_count"] == 1
    assert forward["occluded_count"] == 0
    assert forward["foreground_conflict_count"] == 0
    assert forward["visible_consistent_count"] == 1
    assert forward["visible_mask_supported_count"] == 1
    assert forward["visible_mask_unsupported_count"] == 0
    assert reverse["source_sample_count"] == 1
    json.dumps(evidence, allow_nan=False)


def test_nonidentity_world_to_camera_rotation_and_translation_reconstruct_world(
    tmp_path,
):
    fields = _identity_camera_arrays(1, 2, 2, focal_length=2.0)
    fields["depth"][:] = 5.0
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = np.array([1.0, 2.0, 3.0])
    fields["extrinsic"][0, :, :3] = rotation
    fields["extrinsic"][0, :, 3] = translation
    x_pixels, y_pixels = np.meshgrid(
        np.arange(2, dtype=np.float64),
        np.arange(2, dtype=np.float64),
        indexing="xy",
    )
    camera_points = np.stack(
        [2.5 * x_pixels, 2.5 * y_pixels, np.full((2, 2), 5.0)], axis=-1
    )
    fields["world_points"][0] = (camera_points - translation) @ rotation
    cache_path = _write_npz(tmp_path / "nonidentity_camera.npz", fields)

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([8], dtype=np.int32),
        observations=(),
        formal_snapshot=_snapshot(),
    )

    residual = evidence["camera_contract"]["source_world_reconstruction_residual_m"]
    assert evidence["status"] == "available"
    assert residual["p50"] == pytest.approx(0.0, abs=1e-12)
    assert residual["p95"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("mutation", ["unknown_image_id", "malformed_mask_shape"])
def test_malformed_observation_returns_failed_evidence_without_raise(
    tmp_path, mutation
):
    fields = _identity_camera_arrays(1, 2, 2)
    cache_path = _write_npz(tmp_path / f"{mutation}.npz", fields)
    mask = np.ones((2, 2), dtype=bool)
    observation = (
        _observation(99, mask)
        if mutation == "unknown_image_id"
        else _observation(0, np.ones((1, 2), dtype=bool))
    )

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([0], dtype=np.int32),
        observations=(observation,),
        formal_snapshot=_snapshot(),
    )

    assert evidence["mode"] == "shadow"
    assert evidence["status"] == "failed_evidence"
    json.dumps(evidence, allow_nan=False)


def test_large_finite_world_residual_remains_json_safe(tmp_path):
    fields = _identity_camera_arrays(1, 1, 1)
    fields["world_points"][0, 0, 0] = 1e308
    cache_path = _write_npz(tmp_path / "large_finite_residual.npz", fields)

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([0], dtype=np.int32),
        observations=(),
        formal_snapshot=_snapshot(),
    )

    residual = evidence["camera_contract"]["source_world_reconstruction_residual_m"]
    assert evidence["status"] == "available"
    assert np.isfinite(residual["max"])
    assert residual["max"] > 1e308
    json.dumps(evidence, allow_nan=False)


def test_large_opposite_finite_confidences_have_json_safe_summary(tmp_path):
    fields = _identity_camera_arrays(1, 1, 2)
    fields["world_points_conf"][0, 0] = np.array([-1e308, 1e308])
    cache_path = _write_npz(tmp_path / "large_finite_confidence.npz", fields)

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([0], dtype=np.int32),
        observations=(_observation(0, np.ones((1, 2), dtype=bool)),),
        formal_snapshot=_snapshot(),
    )

    summary = evidence["per_global_id"]["1"]["observations"][0]["confidence"]
    assert evidence["status"] == "available"
    assert summary["p50"] == pytest.approx(0.0)
    assert summary["p95"] == pytest.approx(9e307)
    json.dumps(evidence, allow_nan=False)


def test_nonfinite_camera_reconstruction_math_returns_failed_camera_contract(tmp_path):
    fields = _identity_camera_arrays(1, 1, 2)
    fields["depth"][0, 0, 1, 0] = 1e200
    fields["intrinsic"][0, 0, 0] = 1e-200
    cache_path = _write_npz(tmp_path / "camera_math_overflow.npz", fields)

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([0], dtype=np.int32),
        observations=(),
        formal_snapshot=_snapshot(),
    )

    assert evidence["status"] == "failed_camera_contract"
    json.dumps(evidence, allow_nan=False)


def test_nonfinite_reprojection_math_returns_failed_evidence(tmp_path):
    fields = _identity_camera_arrays(2, 1, 1)
    fields["world_points"][0, 0, 0] = np.array([1e308, 0.0, 1.0])
    fields["intrinsic"][1, 0, 0] = 1e308
    cache_path = _write_npz(tmp_path / "reprojection_overflow.npz", fields)
    mask = np.ones((1, 1), dtype=bool)

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([0, 1], dtype=np.int32),
        observations=(_observation(0, mask), _observation(1, mask)),
        formal_snapshot=_snapshot(),
    )

    assert evidence["status"] == "failed_evidence"
    json.dumps(evidence, allow_nan=False)


def test_nonfinite_observation_geometry_math_returns_failed_evidence(tmp_path):
    fields = _identity_camera_arrays(1, 1, 1)
    cache_path = _write_npz(tmp_path / "geometry_overflow.npz", fields)
    normal = np.full(3, 1.0 / np.sqrt(3.0))
    u_axis = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    v_axis = np.cross(normal, u_axis)
    extreme_plane = SupportPlane(
        point=np.full(3, -1.7e308),
        normal=normal,
        u_axis=u_axis,
        v_axis=v_axis,
        inlier_count=10_000,
        inlier_fraction=1.0,
        p95_residual_m=0.0,
    )
    polygon = box(0.0, 0.0, 1.0, 1.0)
    snapshot = FormalSnapshot(
        status="accepted",
        value_m2=1.0,
        plane=extreme_plane,
        polygons={"1": polygon},
        union=polygon,
        rejection_reason=None,
    )

    evidence = build_shadow_evidence(
        cache_path,
        cache_frame_ids=np.array([0], dtype=np.int32),
        observations=(_observation(0, np.ones((1, 1), dtype=bool)),),
        formal_snapshot=snapshot,
    )

    assert evidence["status"] == "failed_evidence"
    json.dumps(evidence, allow_nan=False)
