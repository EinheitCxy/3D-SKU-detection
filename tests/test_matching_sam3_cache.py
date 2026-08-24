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
from utils.transforms import Pi3ImageTransform


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


def test_matching_publishes_complete_frame_once_and_restores_rng(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A subset match publishes every frame object without perturbing sampling RNG."""
    from utils import sam3_utils

    image_path = tmp_path / "7.jpg"
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image_path)
    transform = Pi3ImageTransform(8, 6, 4, 3)
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
        transform=Pi3ImageTransform(8, 6, 4, 3),
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
            transform=Pi3ImageTransform(8, 6, 4, 3),
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
            transform=Pi3ImageTransform(8, 6, 4, 3),
        )
        == {}
    )
    assert not cache_root.exists()
