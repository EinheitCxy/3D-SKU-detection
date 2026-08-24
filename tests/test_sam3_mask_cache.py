"""Contract tests for the processed-space SAM3 frame-mask cache v2."""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from utils.sam3_mask_cache import (
    FrameMaskCacheError,
    FrameMaskCacheRequest,
    ProcessedDetectionPrompt,
    load_complete_frame_masks,
    load_or_compute_frame_masks,
)


def request(
    tmp_path: Path,
    detections: tuple[ProcessedDetectionPrompt, ...] = (
        ProcessedDetectionPrompt(7, (0.0, 0.0, 8.0, 8.0), (0.0, 0.0, 4.0, 4.0)),
    ),
) -> FrameMaskCacheRequest:
    image = tmp_path / "7.jpg"
    image.write_bytes(b"immutable-fixture")
    return FrameMaskCacheRequest(
        cache_root=tmp_path / "sam3_mask_cache" / "v2",
        image_id=7,
        image_path=image,
        source_size_wh=(8, 8),
        processed_shape_hw=(4, 4),
        source_to_processed_affine=np.asarray(
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float64
        ),
        detections=detections,
        inference_contract={
            "api": "self_exemplar",
            "threshold": 0.5,
            "image_size": 1008,
            "max_batch_size": 32,
            "max_dets_per_query": 1,
            "clip_to_bbox": True,
        },
    )


def changed_request(
    request_: FrameMaskCacheRequest, field: str
) -> FrameMaskCacheRequest:
    if field == "processed_shape_hw":
        return dataclasses.replace(request_, processed_shape_hw=(8, 4))
    if field == "source_to_processed_affine":
        return dataclasses.replace(
            request_,
            source_to_processed_affine=np.asarray(
                [[0.5, 0.0, 0.25], [0.0, 0.5, 0.0]], dtype=np.float64
            ),
        )
    if field == "detections":
        return dataclasses.replace(
            request_,
            detections=(
                ProcessedDetectionPrompt(7, (0.0, 0.0, 6.0, 8.0), (0.0, 0.0, 3.0, 4.0)),
            ),
        )
    if field == "inference_contract":
        return dataclasses.replace(
            request_,
            inference_contract={**request_.inference_contract, "threshold": 0.4},
        )
    raise AssertionError(f"unknown request field: {field}")


def _two_detection_request(tmp_path: Path) -> FrameMaskCacheRequest:
    return request(
        tmp_path,
        (
            ProcessedDetectionPrompt(7, (0.0, 0.0, 4.0, 8.0), (0.0, 0.0, 2.0, 4.0)),
            ProcessedDetectionPrompt(9, (4.0, 0.0, 8.0, 8.0), (2.0, 0.0, 4.0, 4.0)),
        ),
    )


def test_packbits_round_trip_is_byte_exact_and_object_keyed(tmp_path: Path) -> None:
    expected = np.asarray(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, True, False, False],
            [False, False, True, True],
        ],
        dtype=bool,
    )
    result = load_or_compute_frame_masks(request(tmp_path), lambda: {7: expected})
    loaded = load_complete_frame_masks(request(tmp_path))
    assert result.cache_event == "miss"
    assert loaded.cache_event == "hit"
    np.testing.assert_array_equal(loaded.masks_by_object_id[7], expected)
    assert loaded.masks_by_object_id[7].dtype == np.bool_


def test_non_byte_aligned_tail_bits_are_lossless_and_validated(tmp_path: Path) -> None:
    prompt = ProcessedDetectionPrompt(7, (0.0, 0.0, 6.0, 6.0), (0.0, 0.0, 3.0, 3.0))
    req = dataclasses.replace(
        request(tmp_path), processed_shape_hw=(3, 3), detections=(prompt,)
    )
    expected = np.eye(3, dtype=bool)
    load_or_compute_frame_masks(req, lambda: {7: expected})
    np.testing.assert_array_equal(
        load_complete_frame_masks(req).masks_by_object_id[7], expected
    )
    payload_path = req.cache_root / "entries" / "7" / "masks.npz"
    with np.load(payload_path, allow_pickle=False) as loaded:
        packed = loaded["packed_masks"].copy()
    packed[0, -1] |= np.uint8(0b10000000)
    np.savez(payload_path, packed_masks=packed)
    with pytest.raises(FrameMaskCacheError, match="tail bits"):
        load_complete_frame_masks(req)


def test_returned_masks_are_independent_processed_bool_arrays(tmp_path: Path) -> None:
    req = request(tmp_path)
    load_or_compute_frame_masks(req, lambda: {7: np.ones((4, 4), dtype=bool)})
    loaded = load_complete_frame_masks(req)
    loaded.masks_by_object_id[7][0, 0] = False
    assert load_complete_frame_masks(req).masks_by_object_id[7][0, 0]


def test_hit_never_calls_compute(tmp_path: Path) -> None:
    req = request(tmp_path)
    load_or_compute_frame_masks(req, lambda: {7: np.ones((4, 4), dtype=bool)})
    result = load_or_compute_frame_masks(
        req, lambda: (_ for _ in ()).throw(AssertionError("must not compute"))
    )
    assert result.cache_event == "hit"


@pytest.mark.parametrize(
    "field",
    [
        "processed_shape_hw",
        "source_to_processed_affine",
        "detections",
        "inference_contract",
    ],
)
def test_request_mismatch_is_not_a_hit(tmp_path: Path, field: str) -> None:
    original = request(tmp_path)
    load_or_compute_frame_masks(original, lambda: {7: np.ones((4, 4), dtype=bool)})
    with pytest.raises(FrameMaskCacheError):
        load_complete_frame_masks(changed_request(original, field))


def test_reordered_detections_are_not_misbound_by_payload_position(
    tmp_path: Path,
) -> None:
    req = _two_detection_request(tmp_path)
    left = np.zeros((4, 4), dtype=bool)
    left[:, :2] = True
    right = np.zeros((4, 4), dtype=bool)
    right[:, 2:] = True
    load_or_compute_frame_masks(req, lambda: {7: left, 9: right})
    reordered = dataclasses.replace(req, detections=tuple(reversed(req.detections)))
    with pytest.raises(FrameMaskCacheError, match="manifest"):
        load_complete_frame_masks(reordered)
    regenerated = load_or_compute_frame_masks(reordered, lambda: {9: right, 7: left})
    np.testing.assert_array_equal(regenerated.masks_by_object_id[7], left)
    np.testing.assert_array_equal(regenerated.masks_by_object_id[9], right)


def test_compute_requires_exact_complete_object_id_set(tmp_path: Path) -> None:
    req = _two_detection_request(tmp_path)
    with pytest.raises(FrameMaskCacheError, match="object IDs"):
        load_or_compute_frame_masks(req, lambda: {7: np.ones((4, 4), dtype=bool)})
    with pytest.raises(FrameMaskCacheError, match="object IDs"):
        load_or_compute_frame_masks(
            req,
            lambda: {
                7: np.ones((4, 4), dtype=bool),
                9: np.ones((4, 4), dtype=bool),
                11: np.ones((4, 4), dtype=bool),
            },
        )
    assert not (req.cache_root / "entries" / "7").exists()


@pytest.mark.parametrize(
    "mask",
    [np.ones((4, 4), dtype=np.uint8), np.ones((8, 8), dtype=bool)],
)
def test_compute_rejects_non_bool_or_source_shaped_masks(
    tmp_path: Path, mask: np.ndarray
) -> None:
    with pytest.raises(
        FrameMaskCacheError, match="boolean processed mask|processed shape"
    ):
        load_or_compute_frame_masks(request(tmp_path), lambda: {7: mask})


def test_producer_masks_are_canonically_clipped_to_processed_bbox(
    tmp_path: Path,
) -> None:
    req = request(
        tmp_path,
        (ProcessedDetectionPrompt(7, (2.0, 2.0, 6.0, 6.0), (1.0, 1.0, 3.0, 3.0)),),
    )
    result = load_or_compute_frame_masks(req, lambda: {7: np.ones((4, 4), dtype=bool)})
    expected = np.zeros((4, 4), dtype=bool)
    expected[1:3, 1:3] = True
    np.testing.assert_array_equal(result.masks_by_object_id[7], expected)


def test_payload_true_pixel_outside_processed_bbox_is_rejected(tmp_path: Path) -> None:
    req = request(
        tmp_path,
        (ProcessedDetectionPrompt(7, (2.0, 2.0, 6.0, 6.0), (1.0, 1.0, 3.0, 3.0)),),
    )
    load_or_compute_frame_masks(req, lambda: {7: np.zeros((4, 4), dtype=bool)})
    payload_path = req.cache_root / "entries" / "7" / "masks.npz"
    escaped = np.zeros((1, 4, 4), dtype=bool)
    escaped[0, 0, 0] = True
    np.savez(
        payload_path,
        packed_masks=np.packbits(escaped.reshape(1, -1), axis=1, bitorder="little"),
    )
    with pytest.raises(FrameMaskCacheError, match="outside"):
        load_complete_frame_masks(req)


def test_malformed_request_is_rejected_before_filesystem_access(tmp_path: Path) -> None:
    duplicate = ProcessedDetectionPrompt(7, (0.0, 0.0, 8.0, 8.0), (0.0, 0.0, 4.0, 4.0))
    malformed = dataclasses.replace(
        request(tmp_path), detections=(duplicate, duplicate)
    )
    with pytest.raises(FrameMaskCacheError, match="unique"):
        load_complete_frame_masks(malformed)
    non_v2 = dataclasses.replace(
        request(tmp_path), cache_root=tmp_path / "sam3_mask_cache" / "v1"
    )
    with pytest.raises(FrameMaskCacheError, match="v2"):
        load_complete_frame_masks(non_v2)
    assert not (tmp_path / "sam3_mask_cache" / "v1").exists()


def test_v1_sibling_is_never_read(tmp_path: Path) -> None:
    req = request(tmp_path)
    legacy = req.cache_root.parent / "v1" / "entries" / "7"
    legacy.mkdir(parents=True)
    (legacy / "manifest.json").write_text("not a v2 manifest")
    result = load_or_compute_frame_masks(req, lambda: {7: np.eye(4, dtype=bool)})
    assert result.cache_event == "miss"
    np.testing.assert_array_equal(
        load_complete_frame_masks(req).masks_by_object_id[7], np.eye(4, dtype=bool)
    )


def test_compute_exception_leaves_no_readable_entry_and_ignores_partial_temp(
    tmp_path: Path,
) -> None:
    req = request(tmp_path)
    partial = req.cache_root / "entries" / ".7.interrupted"
    partial.mkdir(parents=True)
    (partial / "masks.npz").write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="inference failed"):
        load_or_compute_frame_masks(
            req, lambda: (_ for _ in ()).throw(RuntimeError("inference failed"))
        )
    with pytest.raises(FrameMaskCacheError, match="missing"):
        load_complete_frame_masks(req)
    assert partial.is_dir()


def test_manifest_and_payload_require_exact_schema(tmp_path: Path) -> None:
    req = request(tmp_path)
    load_or_compute_frame_masks(req, lambda: {7: np.eye(4, dtype=bool)})
    manifest_path = req.cache_root / "entries" / "7" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["extra"] = "rejected"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FrameMaskCacheError, match="manifest"):
        load_complete_frame_masks(req)


def _concurrent_worker(
    cache_root: str, image_path: str, counter_path: str, barrier: object, queue: object
) -> None:
    req = FrameMaskCacheRequest(
        cache_root=Path(cache_root),
        image_id=7,
        image_path=Path(image_path),
        source_size_wh=(8, 8),
        processed_shape_hw=(4, 4),
        source_to_processed_affine=np.asarray([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]),
        detections=(
            ProcessedDetectionPrompt(7, (0.0, 0.0, 8.0, 8.0), (0.0, 0.0, 4.0, 4.0)),
        ),
        inference_contract={
            "api": "self_exemplar",
            "threshold": 0.5,
            "image_size": 1008,
            "max_batch_size": 32,
            "max_dets_per_query": 1,
            "clip_to_bbox": True,
        },
    )

    def compute() -> dict[int, np.ndarray]:
        counter = Path(counter_path)
        count = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(count + 1))
        return {7: np.ones((4, 4), dtype=bool)}

    barrier.wait(timeout=10)
    queue.put(load_or_compute_frame_masks(req, compute).cache_event)


def test_two_processes_compute_one_complete_frame(tmp_path: Path) -> None:
    req = request(tmp_path)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    queue = context.Queue()
    counter = tmp_path / "compute-count"
    workers = [
        context.Process(
            target=_concurrent_worker,
            args=(
                str(req.cache_root),
                str(req.image_path),
                str(counter),
                barrier,
                queue,
            ),
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0
    assert sorted(queue.get(timeout=2) for _ in workers) == ["hit", "miss"]
    assert counter.read_text() == "1"
