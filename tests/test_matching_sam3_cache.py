"""Matching-owned SAM3 v2 cache boundary tests."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from utils.config import SKUMatchingConfig, build_matching_config_from_yaml
from utils.transforms import DA3ImageTransform, bind_da3_transforms_from_cache


def _frame_objects() -> list[dict[str, object]]:
    return [
        {"position": [0.0, 0.0, 4.0, 4.0], "confidences": {"det": 0.9}},
        {"position": [4.0, 0.0, 8.0, 4.0], "confidences": {"det": 0.9}},
        {"position": [0.0, 4.0, 4.0, 6.0], "confidences": {"det": 0.9}},
    ]


def _config(cache_root: Path) -> SKUMatchingConfig:
    return SKUMatchingConfig.for_3d_mapping(
        backend="da3",
        device="cpu",
        sam3_checkpoint_path="unused-by-injected-producer.pt",
        sam3_mask_cache_root=str(cache_root),
    )


def _bound_da3_transform(
    image_id: int = 0,
    *,
    source_size_wh: tuple[int, int] = (8, 6),
    processed_shape_hw: tuple[int, int] = (3, 4),
    affine: np.ndarray | None = None,
) -> DA3ImageTransform:
    source_width, source_height = source_size_wh
    processed_height, processed_width = processed_shape_hw
    transform = DA3ImageTransform(
        source_width, source_height, processed_width, processed_height
    )
    if affine is None:
        affine = np.asarray(
            [
                [processed_width / source_width, 0.0, 0.0],
                [0.0, processed_height / source_height, 0.0],
            ],
            dtype=np.float64,
        )
    transform.bind_da3_cache_geometry(affine, processed_shape_hw)
    transform.image_id = image_id
    return transform


def _snapshot_rng() -> tuple[object, tuple[object, ...], torch.Tensor]:
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )


def _assert_rng_equal(
    before: tuple[object, tuple[object, ...], torch.Tensor],
    after: tuple[object, tuple[object, ...], torch.Tensor],
) -> None:
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    assert np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


def test_removed_self_exemplar_key_is_rejected(tmp_path: Path) -> None:
    """A legacy YAML key must not silently select a different SAM3 producer."""
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text("inference:\n  sam3_use_self_exemplar: true\n")

    with pytest.raises(
        ValueError,
        match="sam3_use_self_exemplar was removed; self-exemplar is now the only SAM3 mode",
    ):
        build_matching_config_from_yaml(config_path, algorithm="3d", backend="da3")


def test_master_gate_defaults_true_and_cache_root_is_explicit() -> None:
    """The matching boundary has one default-on SAM3 gate and no mode switch."""
    config = SKUMatchingConfig.for_3d_mapping(
        sam3_mask_cache_root="/tmp/output/dataset/sam3_mask_cache/v2"
    )

    assert config.enable_sam3_mask_sampling is True
    assert not hasattr(config, "sam3_use_self_exemplar")
    assert config.sam3_mask_cache_root == "/tmp/output/dataset/sam3_mask_cache/v2"


def test_self_exemplar_clip_uses_the_pre_change_v2_pixel_boundary_rule() -> None:
    """Fractional self-exemplar boxes must clip exactly as a v2 cache entry."""
    from utils.sam3_utils import _clamp_bbox_xyxy_to_image

    assert _clamp_bbox_xyxy_to_image([0.6, 0.6, 3.6, 3.6], width=5, height=5) == (
        0,
        0,
        4,
        4,
    )


def test_da3_processed_request_uses_pixel_center_affine_exactly(
    tmp_path: Path,
) -> None:
    """The matching producer must publish the exact affine DA3 consumers request."""
    from utils import sam3_utils
    from utils.sam3_mask_cache import map_source_bbox_to_processed

    image_path = tmp_path / "0.JPG"
    Image.new("RGB", (3024, 4032)).save(image_path)
    transform = DA3ImageTransform(3024, 4032, 378, 504)
    source_bbox = [1143.0, 2198.0, 1322.0, 2612.0]
    expected_affine = np.asarray(
        [[0.125, 0.0, -0.4375], [0.0, 0.125, -0.4375]], dtype=np.float64
    )
    transform.bind_da3_cache_geometry(expected_affine, (504, 378))

    request = sam3_utils._processed_frame_request(
        cache_root=tmp_path / "sam3_mask_cache" / "v2",
        image_path=image_path,
        image_id=0,
        frame_detections=[{"position": source_bbox}],
        transform=transform,
    )

    expected_bbox = (142.4375, 274.3125, 164.8125, 326.0625)
    assert np.array_equal(request.source_to_processed_affine, expected_affine)
    assert request.source_size_wh == (3024, 4032)
    assert request.processed_shape_hw == (504, 378)
    assert request.detections[0].processed_bbox_xyxy == expected_bbox
    assert (
        map_source_bbox_to_processed(source_bbox, expected_affine, (504, 378))
        == expected_bbox
    )


def test_da3_processed_request_rejects_unbound_transform(tmp_path: Path) -> None:
    """A DA3 producer cannot synthesize an affine from resize dimensions."""
    from utils import sam3_utils

    image_path = tmp_path / "0.JPG"
    Image.new("RGB", (8, 6)).save(image_path)

    with pytest.raises(ValueError, match="explicit DA3 cache affine and processed shape"):
        sam3_utils._processed_frame_request(
            cache_root=tmp_path / "sam3_mask_cache" / "v2",
            image_path=image_path,
            image_id=0,
            frame_detections=[{"position": [0.0, 0.0, 2.0, 2.0]}],
            transform=DA3ImageTransform(8, 6, 4, 3),
        )


def test_da3_cache_binding_uses_non_aligned_affine_and_shape_exactly(
    tmp_path: Path,
) -> None:
    """A cache crop/rounding geometry is authoritative over process_res recomputation."""
    from utils import sam3_utils
    from utils.sam3_mask_cache import map_source_bbox_to_processed

    image_path = tmp_path / "9.JPG"
    Image.new("RGB", (1000, 700)).save(image_path)
    transform = DA3ImageTransform(1000, 700, 504, 353)
    transform.image_id = 9
    cache_affine = np.asarray([[0.51, 0.0, -3.245], [0.0, 0.49, -2.755]])
    bind_da3_transforms_from_cache(
        [transform],
        image_ids=np.asarray([9], dtype=np.int32),
        source_image_sizes=np.asarray([[1000, 700]], dtype=np.int32),
        source_to_processed_affine=np.asarray([cache_affine]),
        processed_shape_hw=(341, 497),
    )
    source_bbox = [10.0, 8.0, 998.0, 699.0]

    request = sam3_utils._processed_frame_request(
        cache_root=tmp_path / "sam3_mask_cache" / "v2",
        image_path=image_path,
        image_id=9,
        frame_detections=[{"position": source_bbox}],
        transform=transform,
    )

    assert request.processed_shape_hw == (341, 497)
    assert np.array_equal(request.source_to_processed_affine, cache_affine)
    assert request.detections[0].processed_bbox_xyxy == map_source_bbox_to_processed(
        source_bbox, cache_affine, (341, 497)
    )


def test_processed_bbox_mapping_clips_to_cache_grid_with_one_pixel_extent() -> None:
    """Prompts and manifests never retain raw negative processed coordinates."""
    from utils.sam3_mask_cache import map_source_bbox_to_processed

    assert map_source_bbox_to_processed(
        [0.0, 0.0, 1.0, 1.0],
        np.asarray([[0.125, 0.0, -0.4375], [0.0, 0.125, -0.4375]]),
        (504, 378),
    ) == (0.0, 0.0, 1.0, 1.0)


def test_matching_publishes_complete_frame_once_and_restores_rng(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A subset match publishes every frame object without perturbing sampling RNG."""
    from utils import sam3_utils

    image_path = tmp_path / "7.jpg"
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image_path)
    transform = _bound_da3_transform(7)
    config = _config(tmp_path / "sam3_mask_cache" / "v2")
    calls: list[list[tuple[float, float, float, float]]] = []

    def fake_self_exemplar(*, image_path, bboxes_xyxy, **_kwargs):
        random.random()
        np.random.random()
        torch.rand(1)
        calls.append([tuple(bbox) for bbox in bboxes_xyxy])
        masks: list[np.ndarray] = []
        for x1, y1, x2, y2 in bboxes_xyxy:
            mask = np.zeros((3, 4), dtype=bool)
            mask[int(y1) : int(y2), int(x1) : int(x2)] = True
            masks.append(mask)
        return masks

    monkeypatch.setattr(sam3_utils, "sam3_masks_self_exemplar", fake_self_exemplar)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    before = _snapshot_rng()

    masks = sam3_utils.get_self_exemplar_masks_for_reference(
        config,
        image_path=image_path,
        image_id=7,
        frame_detections=_frame_objects(),
        matching_object_ids=[2, 0],
        transform=transform,
    )

    _assert_rng_equal(before, _snapshot_rng())
    assert list(masks) == [0, 1, 2]
    assert calls == [[(0.0, 2.0, 2.0, 3.0), (0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 4.0, 2.0)]]
    assert (
        config.sam3_mask_cache_root
        and (Path(config.sam3_mask_cache_root) / "entries" / "7").is_dir()
    )

    warm_before = _snapshot_rng()
    warm_masks = sam3_utils.get_self_exemplar_masks_for_reference(
        config,
        image_path=image_path,
        image_id=7,
        frame_detections=_frame_objects(),
        matching_object_ids=[2, 0],
        transform=transform,
    )

    _assert_rng_equal(warm_before, _snapshot_rng())
    assert calls == [[(0.0, 2.0, 2.0, 3.0), (0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 4.0, 2.0)]]
    assert list(warm_masks) == [0, 1, 2]
    for object_id in (0, 1, 2):
        assert np.array_equal(masks[object_id], warm_masks[object_id])


def test_matching_never_inspects_a_v1_sibling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale v1 sibling cannot satisfy a v2 matching request."""
    from utils import sam3_utils

    image_path = tmp_path / "8.jpg"
    Image.new("RGB", (8, 6)).save(image_path)
    v1_entry = tmp_path / "sam3_mask_cache" / "v1" / "entries" / "8"
    v1_entry.mkdir(parents=True)
    (v1_entry / "sentinel").write_text("must remain unread")
    config = _config(tmp_path / "sam3_mask_cache" / "v2")
    calls = 0

    def fake_self_exemplar(*, bboxes_xyxy, **_kwargs):
        nonlocal calls
        calls += 1
        return [np.zeros((3, 4), dtype=bool) for _ in bboxes_xyxy]

    monkeypatch.setattr(sam3_utils, "sam3_masks_self_exemplar", fake_self_exemplar)
    masks = sam3_utils.get_self_exemplar_masks_for_reference(
        config,
        image_path=image_path,
        image_id=8,
        frame_detections=_frame_objects(),
        matching_object_ids=[0],
        transform=_bound_da3_transform(8),
    )

    assert calls == 1
    assert set(masks) == {0, 1, 2}
    assert (v1_entry / "sentinel").read_text() == "must remain unread"


def test_concurrent_references_share_one_complete_frame_producer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two matching references to one frame publish exactly one complete entry."""
    from utils import sam3_utils

    image_path = tmp_path / "9.jpg"
    Image.new("RGB", (8, 6)).save(image_path)
    config = _config(tmp_path / "sam3_mask_cache" / "v2")
    producer_calls = 0

    def fake_self_exemplar(*, bboxes_xyxy, **_kwargs):
        nonlocal producer_calls
        producer_calls += 1
        return [np.ones((3, 4), dtype=bool) for _ in bboxes_xyxy]

    monkeypatch.setattr(sam3_utils, "sam3_masks_self_exemplar", fake_self_exemplar)

    def load_for_reference() -> dict[int, np.ndarray]:
        return sam3_utils.get_self_exemplar_masks_for_reference(
            config,
            image_path=image_path,
            image_id=9,
            frame_detections=_frame_objects(),
            matching_object_ids=[1, 0],
            transform=_bound_da3_transform(9),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = list(
            executor.map(lambda _unused: load_for_reference(), range(2))
        )

    assert producer_calls == 1
    assert list(first) == [0, 1, 2]
    assert list(second) == [0, 1, 2]
    for object_id in (0, 1, 2):
        assert np.array_equal(first[object_id], second[object_id])


def test_disabled_sam3_gate_writes_no_cache(tmp_path: Path) -> None:
    """Disabling the diagnostic SAM3 gate leaves matching on its bbox path."""
    from utils import sam3_utils

    image_path = tmp_path / "10.jpg"
    Image.new("RGB", (8, 6)).save(image_path)
    cache_root = tmp_path / "sam3_mask_cache" / "v2"
    config = _config(cache_root)
    config.enable_sam3_mask_sampling = False

    assert (
        sam3_utils.get_self_exemplar_masks_for_reference(
            config,
            image_path=image_path,
            image_id=10,
            frame_detections=_frame_objects(),
            matching_object_ids=[0],
            transform=_bound_da3_transform(10),
        )
        == {}
    )
    assert not cache_root.exists()


def test_empty_and_filtered_reference_frames_publish_complete_v2_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every detection frame publishes once even when matching has no refs."""
    from utils import matching_algorithms, sam3_utils
    from utils.sam3_mask_cache import load_complete_frame_masks

    image_paths: list[str] = []
    transforms: list[DA3ImageTransform] = []
    for frame_id in range(3):
        image_path = tmp_path / f"{frame_id}.JPG"
        Image.new("RGB", (8, 6)).save(image_path)
        image_paths.append(str(image_path))
        transform = DA3ImageTransform(8, 6, 4, 3)
        transform.image_id = frame_id
        transforms.append(transform)
    detections = [
        {"objects": []},
        {"objects": [{"position": [0.0, 0.0, 1.0, 1.0]}]},
        {"objects": []},
    ]
    cache_root = tmp_path / "sam3_mask_cache" / "v2"
    config = _config(cache_root)
    config.output_dir = str(
        tmp_path / "Output" / "sample" / "output_3dmapping_da3" / "0"
    )
    cache_path = Path(config.output_dir).parent.parent / "da3_cache" / "predictions.npz"
    cache_path.parent.mkdir(parents=True)
    np.savez(
        cache_path,
        depth=np.zeros((3, 3, 4, 1), dtype=np.float32),
        depth_conf=np.ones((3, 3, 4), dtype=np.float32),
        world_points=np.zeros((3, 3, 4, 3), dtype=np.float32),
        world_points_conf=np.ones((3, 3, 4), dtype=np.float32),
            extrinsic=np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0),
            intrinsic=np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
            image_ids=np.asarray([0, 1, 2], dtype=np.int32),
            source_image_sizes=np.asarray([[8, 6], [8, 6], [8, 6]], dtype=np.int32),
            source_to_processed_affine=np.repeat(
                np.asarray([[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]]), 3, axis=0
            ),
    )
    producer_batches: list[list[list[float]]] = []

    def fake_self_exemplar(*, bboxes_xyxy, **_kwargs):
        producer_batches.append(bboxes_xyxy)
        return [np.ones((3, 4), dtype=bool) for _ in bboxes_xyxy]

    monkeypatch.setattr(sam3_utils, "sam3_masks_self_exemplar", fake_self_exemplar)
    images = torch.zeros((3, 3, 3, 4), dtype=torch.float32)
    for reference_idx in range(3):
        correspondences, points = matching_algorithms.find_correspondences_3d_mapping(
            None,
            detections,
            images,
            config,
            reference_image_idx=reference_idx,
            transforms_info=transforms,
            image_paths=image_paths,
        )
        assert correspondences == {}
        assert points is None

    assert producer_batches == [[[0.0, 0.0, 0.5, 0.5]]]
    for frame_id, frame in enumerate(detections):
        assert transforms[frame_id].processed_shape_hw == (3, 4)
        assert np.array_equal(
            transforms[frame_id].source_to_processed_affine,
            np.asarray([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]),
        )
        request = sam3_utils._processed_frame_request(
            cache_root=cache_root,
            image_path=Path(image_paths[frame_id]),
            image_id=frame_id,
            frame_detections=frame["objects"],
            transform=transforms[frame_id],
        )
        result = load_complete_frame_masks(request)
        assert set(result.masks_by_object_id) == set(range(len(frame["objects"])))

    disabled_root = tmp_path / "disabled" / "sam3_mask_cache" / "v2"
    disabled = _config(disabled_root)
    disabled.enable_sam3_mask_sampling = False
    disabled.output_dir = config.output_dir
    correspondences, points = matching_algorithms.find_correspondences_3d_mapping(
        None,
        detections,
        images,
        disabled,
        reference_image_idx=0,
        transforms_info=transforms,
        image_paths=image_paths,
    )
    assert correspondences == {}
    assert points is None
    assert not disabled_root.exists()
