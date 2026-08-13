"""Behavior tests for the audited immutable SAM3 frame-mask cache."""

from __future__ import annotations

import json
import multiprocessing
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from utils.sam3_mask_cache import (
    DetectionPrompt,
    FrameMaskCacheRequest,
    canonical_frame_mask_key,
    load_or_compute_frame_masks,
)


def _prompt(object_id: int, bbox_xyxy: tuple[float, float, float, float]) -> DetectionPrompt:
    return DetectionPrompt.from_bbox(object_id, bbox_xyxy)


def _request(
    tmp_path: Path,
    *,
    detections: tuple[tuple[int, tuple[float, float, float, float]], ...] = ((7, (1.0, 2.0, 8.0, 8.0)),),
) -> FrameMaskCacheRequest:
    image_path = tmp_path / "source.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    checkpoint_path = tmp_path / "sam3.ckpt"
    checkpoint_path.write_bytes(b"verified-checkpoint-content")
    return FrameMaskCacheRequest(
        cache_root=tmp_path / "sam3_mask_cache" / "v1",
        image_id=42,
        image_path=image_path,
        detections=tuple(_prompt(object_id, bbox) for object_id, bbox in detections),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256="a" * 64,
        code_fingerprint={"files": {"utils/sam3_utils.py": "b" * 64}},
        runtime_fingerprint={"python": "3.11", "device": "cuda"},
        inference_contract={"api": "predict_inst", "source_mask_dtype": "bool"},
        output_shape_hw=(8, 8),
    )


def _mutate_request(request: FrameMaskCacheRequest, mutation: str) -> FrameMaskCacheRequest:
    if mutation == "image":
        Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8)).save(request.image_path)
        return request
    if mutation == "bbox":
        return replace(request, detections=(_prompt(7, (1.0, 2.0, 7.0, 8.0)),))
    if mutation == "order":
        return replace(
            request,
            detections=(_prompt(8, (1.0, 2.0, 8.0, 8.0)), _prompt(7, (1.0, 2.0, 8.0, 8.0))),
        )
    if mutation == "checkpoint":
        return replace(request, checkpoint_sha256="c" * 64)
    if mutation == "contract":
        return replace(request, inference_contract={"api": "predict_inst", "threshold": 0.0})
    raise AssertionError(f"unknown mutation {mutation}")


def _producer_counting_calls() -> tuple[object, list[int]]:
    calls: list[int] = []

    def produce() -> list[np.ndarray]:
        calls.append(1)
        return [np.ones((8, 8), dtype=bool)]

    return produce, calls


def _producer_that_fails_if_called() -> list[np.ndarray]:
    raise AssertionError("valid cache hit must not call the mask producer")


def _concurrent_worker(cache_root: str, image_path: str, checkpoint_path: str, barrier: object, queue: object) -> None:
    request = FrameMaskCacheRequest(
        cache_root=Path(cache_root),
        image_id=42,
        image_path=Path(image_path),
        detections=(_prompt(7, (1.0, 2.0, 8.0, 8.0)),),
        checkpoint_path=Path(checkpoint_path),
        checkpoint_sha256="a" * 64,
        code_fingerprint={"files": {"utils/sam3_utils.py": "b" * 64}},
        runtime_fingerprint={"python": "3.11", "device": "cuda"},
        inference_contract={"api": "predict_inst", "source_mask_dtype": "bool"},
        output_shape_hw=(8, 8),
    )

    def producer() -> list[np.ndarray]:
        return [np.ones((8, 8), dtype=bool)]

    barrier.wait(timeout=10)
    result = load_or_compute_frame_masks(request, producer)
    queue.put((result.events, result.masks[0].tolist()))


def _concurrently_load_same_request(tmp_path: Path):
    request = _request(tmp_path)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    queue = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_worker,
            args=(str(request.cache_root), str(request.image_path), str(request.checkpoint_path), barrier, queue),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    return [queue.get(timeout=2) for _ in processes], request


def test_frame_cache_hit_does_not_call_mask_producer(tmp_path: Path) -> None:
    request = _request(tmp_path)
    producer, calls = _producer_counting_calls()
    first = load_or_compute_frame_masks(request, producer)
    second = load_or_compute_frame_masks(request, _producer_that_fails_if_called)
    assert calls == [1]
    assert np.array_equal(first.masks[0], second.masks[0])
    assert second.events == ("hit",)


@pytest.mark.parametrize("mutation", ["image", "bbox", "order", "checkpoint", "contract"])
def test_key_input_mutation_is_a_miss(tmp_path: Path, mutation: str) -> None:
    first_request = _request(tmp_path)
    load_or_compute_frame_masks(first_request, lambda: [np.ones((8, 8), dtype=bool)])
    changed_request = _mutate_request(first_request, mutation)
    result = load_or_compute_frame_masks(changed_request, lambda: [np.ones((8, 8), dtype=bool)] * len(changed_request.detections))
    assert result.events == ("miss", "written")


def test_corrupt_payload_is_quarantined_then_recomputed(tmp_path: Path) -> None:
    request = _request(tmp_path)
    first = load_or_compute_frame_masks(request, lambda: [np.ones((8, 8), dtype=bool)])
    (request.cache_root / "entries" / first.key / "masks.npz").write_bytes(b"truncated")
    second = load_or_compute_frame_masks(request, lambda: [np.ones((8, 8), dtype=bool)])
    assert second.events == ("invalid", "written")
    assert any((request.cache_root / "corrupt").iterdir())


def test_manifest_bbox_escape_is_quarantined_and_never_returned(tmp_path: Path) -> None:
    request = _request(tmp_path, detections=((7, (2.0, 2.0, 6.0, 6.0)),))
    first = load_or_compute_frame_masks(request, lambda: [np.ones((8, 8), dtype=bool) & False])
    payload = request.cache_root / "entries" / first.key / "masks.npz"
    escaped = np.zeros((1, 8, 8), dtype=bool)
    escaped[0, 0, 0] = True
    np.savez_compressed(payload, masks=escaped)
    manifest_path = request.cache_root / "entries" / first.key / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["payload_sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest["masks"][0]["sha256"] = hashlib.sha256(escaped[0].tobytes()).hexdigest()
    manifest["masks"][0]["true_pixel_count"] = 1
    manifest_path.write_text(json.dumps(manifest))
    result = load_or_compute_frame_masks(request, lambda: [np.zeros((8, 8), dtype=bool)])
    assert result.events == ("invalid", "written")
    assert result.masks[0].sum() == 0


def test_producer_mask_is_clipped_to_bbox_without_bbox_fallback(tmp_path: Path) -> None:
    request = _request(tmp_path, detections=((7, (2.0, 2.0, 6.0, 6.0)),))
    outside = np.zeros((8, 8), dtype=bool)
    outside[0, 0] = True
    result = load_or_compute_frame_masks(request, lambda: [outside])
    assert result.masks[0].sum() == 0


def test_non_finite_bbox_rejects_before_cache_operation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        _prompt(7, (1.0, 2.0, float("nan"), 8.0))


def test_negative_zero_has_same_canonical_key_as_positive_zero(tmp_path: Path) -> None:
    negative = _request(tmp_path, detections=((7, (-0.0, 0.0, 8.0, 8.0)),))
    positive = _request(tmp_path, detections=((7, (0.0, 0.0, 8.0, 8.0)),))
    assert canonical_frame_mask_key(negative) == canonical_frame_mask_key(positive)


def test_two_processes_publish_one_complete_bundle(tmp_path: Path) -> None:
    results, request = _concurrently_load_same_request(tmp_path)
    assert all(np.asarray(mask, dtype=bool).sum() == 42 for _, mask in results)
    assert sorted(events for events, _ in results) == [("hit",), ("miss", "written")]
    assert len(list((request.cache_root / "entries").iterdir())) == 1


def test_complete_empty_mask_is_cached_without_bbox_fallback(tmp_path: Path) -> None:
    result = load_or_compute_frame_masks(_request(tmp_path), lambda: [np.zeros((8, 8), dtype=bool)])
    assert result.masks[0].sum() == 0
    assert result.events == ("miss", "written")
