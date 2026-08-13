import json
from pathlib import Path

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
) -> EvidenceObservation:
    return EvidenceObservation(
        global_id="1",
        image_id=image_id,
        object_id=object_id,
        processed_mask=mask,
        valid_mask=np.ones_like(mask, dtype=bool) if valid_mask is None else valid_mask,
    )


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
