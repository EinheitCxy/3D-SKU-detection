"""Contract tests for the processed-space SAM3 frame-mask cache v2."""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import os
import struct
from pathlib import Path
import zipfile

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


def test_read_only_load_missing_root_never_creates_cache_paths(tmp_path: Path) -> None:
    req = request(tmp_path)

    with pytest.raises(FrameMaskCacheError, match="cache root"):
        load_complete_frame_masks(req)

    assert not req.cache_root.exists()
    assert not (req.cache_root.parent / "locks").exists()


def test_read_only_load_of_published_entry_never_mkdirs(tmp_path: Path, monkeypatch) -> None:
    req = request(tmp_path)
    load_or_compute_frame_masks(req, lambda: {7: np.eye(4, dtype=bool)})

    def forbidden_mkdir(*_args, **_kwargs):
        raise AssertionError("read-only load must not create directories")

    monkeypatch.setattr(Path, "mkdir", forbidden_mkdir)
    loaded = load_complete_frame_masks(req)

    np.testing.assert_array_equal(loaded.masks_by_object_id[7], np.eye(4, dtype=bool))


def test_read_only_load_missing_lock_is_a_typed_error_without_writes(
    tmp_path: Path,
) -> None:
    req = request(tmp_path)
    load_or_compute_frame_masks(req, lambda: {7: np.eye(4, dtype=bool)})
    (req.cache_root / "locks" / "7.lock").unlink()

    with pytest.raises(FrameMaskCacheError, match="lock"):
        load_complete_frame_masks(req)

    assert not (req.cache_root / "locks" / "7.lock").exists()


def test_read_only_load_succeeds_from_published_read_only_cache(
    tmp_path: Path,
) -> None:
    req = request(tmp_path)
    load_or_compute_frame_masks(req, lambda: {7: np.eye(4, dtype=bool)})
    entry = req.cache_root / "entries" / "7"
    lock = req.cache_root / "locks" / "7.lock"
    os.chmod(req.cache_root, 0o555)
    os.chmod(entry.parent, 0o555)
    os.chmod(entry, 0o555)
    os.chmod(lock, 0o444)
    try:
        loaded = load_complete_frame_masks(req)
    finally:
        os.chmod(req.cache_root, 0o755)
        os.chmod(entry.parent, 0o755)
        os.chmod(entry, 0o755)
        os.chmod(lock, 0o644)

    np.testing.assert_array_equal(loaded.masks_by_object_id[7], np.eye(4, dtype=bool))


def test_read_only_lock_permission_error_is_typed(tmp_path: Path, monkeypatch) -> None:
    req = request(tmp_path)
    load_or_compute_frame_masks(req, lambda: {7: np.eye(4, dtype=bool)})
    lock = req.cache_root / "locks" / "7.lock"
    original_open = Path.open

    def denied_open(path: Path, *args, **kwargs):
        if path == lock:
            raise PermissionError("denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)
    with pytest.raises(FrameMaskCacheError, match="cannot read cache lock"):
        load_complete_frame_masks(req)


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


def test_fractional_processed_bbox_uses_floor_ceil_pixel_slice(
    tmp_path: Path,
) -> None:
    req = dataclasses.replace(
        request(tmp_path),
        processed_shape_hw=(5, 5),
        detections=(
            ProcessedDetectionPrompt(
                7,
                (1.2, 1.2, 7.2, 7.2),
                (0.6, 0.6, 3.6, 3.6),
            ),
        ),
    )
    expected = np.zeros((5, 5), dtype=bool)
    expected[:4, :4] = True
    result = load_or_compute_frame_masks(req, lambda: {7: np.ones((5, 5), dtype=bool)})
    np.testing.assert_array_equal(result.masks_by_object_id[7], expected)

    escaped = expected.copy()
    escaped[4, 0] = True
    payload_path = req.cache_root / "entries" / "7" / "masks.npz"
    np.savez(
        payload_path,
        packed_masks=np.packbits(escaped.reshape(1, -1), axis=1, bitorder="little"),
    )
    with pytest.raises(FrameMaskCacheError, match="outside"):
        load_complete_frame_masks(req)


@pytest.mark.parametrize(
    ("source_bbox", "processed_bbox", "allowed_pixel", "outside_pixel"),
    [
        ((2.0, 2.0, 2.0, 4.0), (1.0, 1.0, 1.0, 2.0), (1, 1), (1, 2)),
        ((8.0, 8.0, 8.0, 8.0), (5.0, 5.0, 5.0, 5.0), (4, 4), (4, 3)),
        ((8.0, 8.0, 8.0, 8.0), (5.2, 5.2, 6.0, 6.0), (4, 4), (4, 3)),
    ],
)
def test_degenerate_processed_bbox_preserves_producer_one_pixel_slice(
    tmp_path: Path,
    source_bbox: tuple[float, float, float, float],
    processed_bbox: tuple[float, float, float, float],
    allowed_pixel: tuple[int, int],
    outside_pixel: tuple[int, int],
) -> None:
    req = dataclasses.replace(
        request(tmp_path),
        processed_shape_hw=(5, 5),
        detections=(ProcessedDetectionPrompt(7, source_bbox, processed_bbox),),
    )
    expected = np.zeros((5, 5), dtype=bool)
    expected[allowed_pixel] = True
    result = load_or_compute_frame_masks(req, lambda: {7: np.ones((5, 5), dtype=bool)})
    np.testing.assert_array_equal(result.masks_by_object_id[7], expected)

    escaped = expected.copy()
    escaped[outside_pixel] = True
    payload_path = req.cache_root / "entries" / "7" / "masks.npz"
    np.savez(
        payload_path,
        packed_masks=np.packbits(escaped.reshape(1, -1), axis=1, bitorder="little"),
    )
    with pytest.raises(FrameMaskCacheError, match="outside"):
        load_complete_frame_masks(req)


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


@pytest.mark.parametrize(
    "source_bbox",
    [
        (-0.1, 0.0, 8.0, 8.0),
        (0.0, -0.1, 8.0, 8.0),
    ],
)
def test_source_bbox_negative_coordinate_is_rejected(
    tmp_path: Path,
    source_bbox: tuple[float, float, float, float],
) -> None:
    req = request(
        tmp_path,
        (ProcessedDetectionPrompt(7, source_bbox, (0.0, 0.0, 4.0, 4.0)),),
    )
    with pytest.raises(FrameMaskCacheError, match="source_bbox_xyxy.*bounds"):
        load_complete_frame_masks(req)
    assert not req.cache_root.exists()


@pytest.mark.parametrize(
    "source_bbox",
    [
        (0.0, 0.0, 8.1, 8.0),
        (0.0, 0.0, 8.0, 8.1),
    ],
)
def test_source_bbox_right_or_bottom_overflow_is_rejected(
    tmp_path: Path,
    source_bbox: tuple[float, float, float, float],
) -> None:
    req = request(
        tmp_path,
        (ProcessedDetectionPrompt(7, source_bbox, (0.0, 0.0, 4.0, 4.0)),),
    )
    with pytest.raises(FrameMaskCacheError, match="source_bbox_xyxy.*bounds"):
        load_complete_frame_masks(req)
    assert not req.cache_root.exists()


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


def test_manifest_missing_required_key_is_rejected(tmp_path: Path) -> None:
    req = request(tmp_path)
    load_or_compute_frame_masks(req, lambda: {7: np.eye(4, dtype=bool)})
    manifest_path = req.cache_root / "entries" / "7" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["payload"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FrameMaskCacheError, match="manifest"):
        load_complete_frame_masks(req)


def _corrupt_packed_masks_member_crc(payload_path: Path) -> None:
    """Flip a stored member byte without updating the ZIP central-directory CRC."""
    with zipfile.ZipFile(payload_path) as archive:
        member = archive.getinfo("packed_masks.npy")
    raw = bytearray(payload_path.read_bytes())
    name_length, extra_length = struct.unpack_from(
        "<HH", raw, member.header_offset + 26
    )
    data_offset = member.header_offset + 30 + name_length + extra_length
    raw[data_offset + member.compress_size - 1] ^= 1
    payload_path.write_bytes(raw)


def test_crc_corrupt_payload_fails_closed_with_typed_error(tmp_path: Path) -> None:
    req = request(tmp_path)
    load_or_compute_frame_masks(req, lambda: {7: np.eye(4, dtype=bool)})
    _corrupt_packed_masks_member_crc(req.cache_root / "entries" / "7" / "masks.npz")
    with pytest.raises(FrameMaskCacheError, match="payload"):
        load_complete_frame_masks(req)


def test_crc_corrupt_payload_is_quarantined_and_recomputed(tmp_path: Path) -> None:
    req = request(tmp_path)
    load_or_compute_frame_masks(req, lambda: {7: np.eye(4, dtype=bool)})
    _corrupt_packed_masks_member_crc(req.cache_root / "entries" / "7" / "masks.npz")
    calls: list[int] = []

    def compute() -> dict[int, np.ndarray]:
        calls.append(1)
        return {7: np.ones((4, 4), dtype=bool)}

    result = load_or_compute_frame_masks(req, compute)
    assert result.cache_event == "miss"
    assert calls == [1]
    assert len(list((req.cache_root / "corrupt").iterdir())) == 1
    np.testing.assert_array_equal(
        load_complete_frame_masks(req).masks_by_object_id[7],
        np.ones((4, 4), dtype=bool),
    )


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
