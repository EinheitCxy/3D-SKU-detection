"""Next-reference SAM3 input prefetch: key hit, key mismatch, worker errors."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from utils import sam3_utils
from utils.config import SKUMatchingConfig
from utils.transforms import DA3ImageTransform


def _transform(image_id: int) -> DA3ImageTransform:
    transform = DA3ImageTransform(8, 6, 4, 3)
    affine = np.asarray([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float64)
    transform.bind_da3_cache_geometry(affine, (3, 4))
    transform.image_id = image_id
    return transform


def _config(cache_root: Path) -> SKUMatchingConfig:
    return SKUMatchingConfig.for_3d_mapping(
        backend="da3",
        device="cpu",
        sam3_checkpoint_path="unused-by-injected-producer.pt",
        sam3_mask_cache_root=str(cache_root),
    )


@pytest.fixture(autouse=True)
def _fresh_slot(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sam3_utils, "_SAM3_INPUT_PREFETCH_SLOT", None)


@pytest.fixture
def frames(tmp_path: Path) -> list[Path]:
    paths = []
    for index, color in enumerate([(10, 20, 30), (200, 100, 50)]):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (8, 6), color=color).save(path)
        paths.append(path)
    return paths


def _count_prepares(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the thread of every `_prepare_sam3_input_image` call."""
    real = sam3_utils._prepare_sam3_input_image
    threads: list[str] = []

    def counting(image, transform):
        threads.append(threading.current_thread().name)
        return real(image, transform)

    monkeypatch.setattr(sam3_utils, "_prepare_sam3_input_image", counting)
    return threads


def test_matching_key_hit_uses_prefetched_image(monkeypatch, frames, tmp_path):
    """A miss prefetches the next frame; the next miss consumes it, not inline."""
    threads = _count_prepares(monkeypatch)
    seen: list[bytes] = []

    def fake_self_exemplar(*, image_path, bboxes_xyxy, **_kwargs):
        seen.append(image_path.tobytes())
        return [np.ones((3, 4), dtype=bool) for _ in bboxes_xyxy]

    monkeypatch.setattr(sam3_utils, "sam3_masks_self_exemplar", fake_self_exemplar)
    config = _config(tmp_path / "cache" / "v2")
    detections = [{"position": [0.0, 0.0, 4.0, 4.0]}]
    transforms = [_transform(0), _transform(1)]

    sam3_utils.get_self_exemplar_masks_for_reference(
        config,
        image_path=frames[0],
        image_id=0,
        frame_detections=detections,
        matching_object_ids=[0],
        transform=transforms[0],
        prefetch_next=(frames[1], transforms[1]),
    )
    sam3_utils.get_self_exemplar_masks_for_reference(
        config,
        image_path=frames[1],
        image_id=1,
        frame_detections=detections,
        matching_object_ids=[0],
        transform=_transform(1),  # equal fields, different object
    )

    assert len(threads) == 2
    assert not threads[0].startswith("sam3-input-prefetch")
    assert threads[1].startswith("sam3-input-prefetch")
    assert sam3_utils._SAM3_INPUT_PREFETCH_SLOT is None
    for path, data in zip(frames, seen):
        with Image.open(path) as source:
            expected, _ = sam3_utils._prepare_sam3_input_image(
                source.convert("RGB"), _transform(0)
            )
        assert data == expected.tobytes()


def test_cache_hit_submits_no_prefetch(monkeypatch, frames, tmp_path):
    monkeypatch.setattr(
        sam3_utils,
        "sam3_masks_self_exemplar",
        lambda *, bboxes_xyxy, **_k: [np.ones((3, 4), dtype=bool) for _ in bboxes_xyxy],
    )
    config = _config(tmp_path / "cache" / "v2")
    kwargs = dict(
        image_path=frames[0],
        image_id=0,
        frame_detections=[{"position": [0.0, 0.0, 4.0, 4.0]}],
        matching_object_ids=[0],
        transform=_transform(0),
    )
    sam3_utils.get_self_exemplar_masks_for_reference(config, **kwargs)
    sam3_utils.get_self_exemplar_masks_for_reference(
        config, **kwargs, prefetch_next=(frames[1], _transform(1))
    )
    assert sam3_utils._SAM3_INPUT_PREFETCH_SLOT is None


def test_key_mismatch_prepares_inline(monkeypatch, frames):
    threads = _count_prepares(monkeypatch)
    sam3_utils._submit_sam3_input_prefetch(frames[0], _transform(0))
    sam3_utils._SAM3_INPUT_PREFETCH_SLOT[1].result()

    sam3_utils._take_sam3_input_image(frames[1], _transform(1))  # other path
    other_size = DA3ImageTransform(8, 6, 2, 2)
    other_size.bind_da3_cache_geometry(
        np.asarray([[0.25, 0.0, 0.0], [0.0, 1 / 3, 0.0]]), (2, 2)
    )
    image, _ = sam3_utils._take_sam3_input_image(frames[0], other_size)
    assert image.size == (2, 2)

    stat = frames[0].stat()
    os.utime(frames[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    sam3_utils._take_sam3_input_image(frames[0], _transform(0))  # changed mtime

    assert len(threads) == 4
    assert all(not name.startswith("sam3-input-prefetch") for name in threads[1:])


def test_worker_exception_propagates(monkeypatch, frames):
    def failing(image, transform):
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(sam3_utils, "_prepare_sam3_input_image", failing)
    sam3_utils._submit_sam3_input_prefetch(frames[0], _transform(0))
    with pytest.raises(RuntimeError, match="prepare failed"):
        sam3_utils._take_sam3_input_image(frames[0], _transform(0))
