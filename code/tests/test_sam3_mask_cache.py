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

import utils.sam3_mask_cache as sam3_mask_cache
import utils.sam3_utils as sam3_utils
from utils.sam3_mask_cache import (
    DetectionPrompt,
    FrameMaskCacheRequest,
    canonical_frame_mask_key,
    load_or_compute_frame_masks,
)
from utils.sam3_utils import clip_mask_to_bbox


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
    if mutation == "image_id":
        return replace(request, image_id=43)
    if mutation == "image_size":
        request.image_path.write_bytes(request.image_path.read_bytes() + b"trailing-cache-key-bytes")
        return request
    if mutation == "code_fingerprint":
        return replace(request, code_fingerprint={"files": {"utils/sam3_utils.py": "d" * 64}})
    if mutation == "runtime_fingerprint":
        return replace(request, runtime_fingerprint={"python": "3.12", "device": "cuda"})
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


@pytest.mark.parametrize(
    "mutation",
    [
        "image",
        "image_id",
        "image_size",
        "bbox",
        "order",
        "checkpoint",
        "contract",
        "code_fingerprint",
        "runtime_fingerprint",
    ],
)
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


@pytest.mark.parametrize(
    "bbox_xyxy",
    [
        (2.0, 2.0, 6.0, 6.0),
        (2.2, 1.7, 7.6, 6.4),
        (-4.0, 2.0, -1.0, 4.0),
        (9.0, 2.0, 12.0, 4.0),
        (2.0, -4.0, 4.0, -1.0),
        (2.0, 9.0, 4.0, 12.0),
        (0.0, 0.0, 8.0, 8.0),
    ],
)
def test_cache_masks_match_sam3_canonical_bbox_clip_on_cold_and_hit(
    tmp_path: Path, bbox_xyxy: tuple[float, float, float, float]
) -> None:
    """Catches producer or hit validation drifting from SAM3 boundary pixels."""
    request = _request(tmp_path, detections=((7, bbox_xyxy),))
    source = (np.arange(64).reshape(8, 8) % 3) != 0
    expected = clip_mask_to_bbox(source, bbox_xyxy)

    cold = load_or_compute_frame_masks(request, lambda: [source.copy()])
    hit = load_or_compute_frame_masks(request, _producer_that_fails_if_called)

    np.testing.assert_array_equal(cold.masks[0], expected)
    np.testing.assert_array_equal(hit.masks[0], expected)
    assert cold.masks[0].tobytes() == expected.tobytes()
    assert hit.masks[0].tobytes() == expected.tobytes()
    assert cold.events == ("miss", "written")
    assert hit.events == ("hit",)


def test_reversed_bbox_is_rejected_before_cache_operation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reversed"):
        _request(tmp_path, detections=((7, (5.0, 2.0, 2.0, 4.0)),))
    assert not (tmp_path / "sam3_mask_cache").exists()


def test_non_finite_bbox_rejects_before_cache_operation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        _prompt(7, (1.0, 2.0, float("nan"), 8.0))


def test_negative_zero_has_same_canonical_key_as_positive_zero(tmp_path: Path) -> None:
    negative = _request(tmp_path, detections=((7, (-0.0, 0.0, 8.0, 8.0)),))
    positive = _request(tmp_path, detections=((7, (0.0, 0.0, 8.0, 8.0)),))
    assert canonical_frame_mask_key(negative) == canonical_frame_mask_key(positive)


def test_direct_prompt_constructor_normalizes_hex_and_negative_zero(tmp_path: Path) -> None:
    direct = DetectionPrompt(
        object_id=7,
        bbox_xyxy_f64be_hex=(
            " 8000000000000000 ",
            " 3FF8000000000000 ",
            "4010000000000000",
            "4010000000000000",
        ),
    )
    canonical = _prompt(7, (0.0, 1.5, 4.0, 4.0))
    direct_request = replace(_request(tmp_path), detections=(direct,))
    canonical_request = replace(_request(tmp_path), detections=(canonical,))
    assert direct.bbox_xyxy_f64be_hex == canonical.bbox_xyxy_f64be_hex
    assert canonical_frame_mask_key(direct_request) == canonical_frame_mask_key(canonical_request)


def test_checkpoint_digest_is_lowercase_syntax_normalized(tmp_path: Path) -> None:
    request = _request(tmp_path)
    upper = replace(request, checkpoint_sha256=("A" * 64))
    assert upper.checkpoint_sha256 == "a" * 64
    assert canonical_frame_mask_key(upper) == canonical_frame_mask_key(request)


def test_two_processes_publish_one_complete_bundle(tmp_path: Path) -> None:
    results, request = _concurrently_load_same_request(tmp_path)
    assert all(np.asarray(mask, dtype=bool).sum() == 42 for _, mask in results)
    assert sorted(events for events, _ in results) == [("hit",), ("miss", "written")]
    assert len(list((request.cache_root / "entries").iterdir())) == 1


def test_complete_empty_mask_is_cached_without_bbox_fallback(tmp_path: Path) -> None:
    result = load_or_compute_frame_masks(_request(tmp_path), lambda: [np.zeros((8, 8), dtype=bool)])
    assert result.masks[0].sum() == 0
    assert result.events == ("miss", "written")


def test_image_mutation_during_producer_rejects_and_does_not_publish(tmp_path: Path) -> None:
    request = _request(tmp_path)
    original_key = canonical_frame_mask_key(request)

    def mutate_image_then_return_masks() -> list[np.ndarray]:
        Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8)).save(request.image_path)
        return [np.ones((8, 8), dtype=bool)]

    with pytest.raises(ValueError, match="source image changed"):
        load_or_compute_frame_masks(request, mutate_image_then_return_masks)
    assert not (request.cache_root / "entries" / original_key).exists()


def test_first_write_failure_reports_miss_then_cache_write_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(sam3_mask_cache.np, "savez_compressed", fail_write)
    result = load_or_compute_frame_masks(request, lambda: [np.ones((8, 8), dtype=bool)])
    assert result.events == ("miss", "cache_write_failed")


def test_entries_directory_initialization_failure_returns_fresh_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    fresh_mask = np.zeros((8, 8), dtype=bool)
    fresh_mask[2:8, 1:8] = True
    original_mkdir = sam3_mask_cache.Path.mkdir
    entries = request.cache_root / "entries"

    def fail_entries_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        if path == entries:
            raise OSError("injected entries initialization failure")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(sam3_mask_cache.Path, "mkdir", fail_entries_mkdir)
    result = load_or_compute_frame_masks(request, lambda: [fresh_mask])

    np.testing.assert_array_equal(result.masks[0], fresh_mask)
    assert result.payload_sha256 is None
    assert result.events == ("miss", "cache_write_failed")


def test_temporary_bundle_initialization_failure_returns_fresh_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    fresh_mask = np.zeros((8, 8), dtype=bool)
    fresh_mask[2:8, 1:8] = True

    def fail_mkdtemp(*_args: object, **_kwargs: object) -> str:
        raise OSError("injected temporary initialization failure")

    monkeypatch.setattr(sam3_mask_cache.tempfile, "mkdtemp", fail_mkdtemp)
    result = load_or_compute_frame_masks(request, lambda: [fresh_mask])

    np.testing.assert_array_equal(result.masks[0], fresh_mask)
    assert result.payload_sha256 is None
    assert result.events == ("miss", "cache_write_failed")


def test_invalid_rebuild_write_failure_reports_invalid_then_cache_write_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    first = load_or_compute_frame_masks(request, lambda: [np.ones((8, 8), dtype=bool)])
    (request.cache_root / "entries" / first.key / "masks.npz").write_bytes(b"truncated")

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(sam3_mask_cache.np, "savez_compressed", fail_write)
    result = load_or_compute_frame_masks(request, lambda: [np.ones((8, 8), dtype=bool)])
    assert result.events == ("invalid", "cache_write_failed")
    assert any((request.cache_root / "corrupt").iterdir())


def test_zero_detection_bundle_writes_bool_payload_and_hits_without_producer(tmp_path: Path) -> None:
    request = _request(tmp_path, detections=())
    calls: list[int] = []

    def producer() -> list[np.ndarray]:
        calls.append(1)
        return []

    first = load_or_compute_frame_masks(request, producer)
    second = load_or_compute_frame_masks(request, _producer_that_fails_if_called)
    with np.load(request.cache_root / "entries" / first.key / "masks.npz", allow_pickle=False) as payload:
        assert payload["masks"].dtype == np.bool_
        assert payload["masks"].shape == (0, 8, 8)
    assert calls == [1]
    assert first.masks == ()
    assert second.masks == ()
    assert second.events == ("hit",)


def _checkpoint(tmp_path: Path, contents: bytes) -> Path:
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(contents)
    return checkpoint


def test_model_cache_reloads_when_checkpoint_bytes_are_replaced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches a path-only model-cache key after a checkpoint replacement."""
    checkpoint = _checkpoint(tmp_path, b"first checkpoint contents")
    loads: list[tuple[str, str]] = []

    def build(checkpoint_path: str, device: str) -> tuple[object, object]:
        loads.append((checkpoint_path, device))
        return object(), object()

    monkeypatch.setattr(sam3_utils, "_SAM3_PREDICT_INST_CACHE", {})
    monkeypatch.setattr(sam3_utils, "_build_sam3_model_and_processor", build, raising=False)
    first_digest = hashlib.sha256(b"first checkpoint contents").hexdigest()
    sam3_utils._get_sam3_model_and_processor(
        str(checkpoint), "cuda:0", expected_checkpoint_sha256=first_digest
    )
    checkpoint.write_bytes(b"replacement checkpoint contents")
    second_digest = hashlib.sha256(b"replacement checkpoint contents").hexdigest()
    sam3_utils._get_sam3_model_and_processor(
        str(checkpoint), "cuda:0", expected_checkpoint_sha256=second_digest
    )

    assert loads == [(str(checkpoint), "cuda:0"), (str(checkpoint), "cuda:0")]
    assert len(sam3_utils._SAM3_PREDICT_INST_CACHE) == 2


def test_checkpoint_mutation_during_model_load_is_rejected_without_cache_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches publishing a model whose checkpoint changed during its load."""
    checkpoint = _checkpoint(tmp_path, b"first checkpoint contents")
    expected_digest = hashlib.sha256(b"first checkpoint contents").hexdigest()

    def mutate_checkpoint_while_building(*_args: object) -> tuple[object, object]:
        checkpoint.write_bytes(b"mutated during model load")
        return object(), object()

    monkeypatch.setattr(sam3_utils, "_SAM3_PREDICT_INST_CACHE", {})
    monkeypatch.setattr(sam3_utils, "_build_sam3_model_and_processor", mutate_checkpoint_while_building, raising=False)

    with pytest.raises(RuntimeError, match="changed while loading"):
        sam3_utils._get_sam3_model_and_processor(
            str(checkpoint), "cuda:0", expected_checkpoint_sha256=expected_digest
        )

    assert sam3_utils._SAM3_PREDICT_INST_CACHE == {}


def test_model_cache_keeps_explicit_cuda_ordinals_in_independent_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches collapsing explicit CUDA devices into one process-cache entry."""
    checkpoint = _checkpoint(tmp_path, b"checkpoint contents")
    expected_digest = hashlib.sha256(b"checkpoint contents").hexdigest()
    builder_devices: list[str] = []

    def build(_checkpoint_path: str, device: str) -> tuple[object, object]:
        builder_devices.append(device)
        return object(), object()

    monkeypatch.setattr(sam3_utils, "_SAM3_PREDICT_INST_CACHE", {})
    monkeypatch.setattr(sam3_utils, "_build_sam3_model_and_processor", build)

    sam3_utils._get_sam3_model_and_processor(
        str(checkpoint), "cuda:0", expected_checkpoint_sha256=expected_digest
    )
    sam3_utils._get_sam3_model_and_processor(
        str(checkpoint), "cuda:1", expected_checkpoint_sha256=expected_digest
    )

    assert builder_devices == ["cuda:0", "cuda:1"]
    assert {key[1] for key in sam3_utils._SAM3_PREDICT_INST_CACHE} == {"cuda:0", "cuda:1"}


def test_model_cache_resolves_bare_cuda_to_current_ordinal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches passing an ambiguous bare CUDA device to the model builder."""
    checkpoint = _checkpoint(tmp_path, b"checkpoint contents")
    expected_digest = hashlib.sha256(b"checkpoint contents").hexdigest()
    builder_devices: list[str] = []

    def build(_checkpoint_path: str, device: str) -> tuple[object, object]:
        builder_devices.append(device)
        return object(), object()

    monkeypatch.setattr(sam3_utils, "_SAM3_PREDICT_INST_CACHE", {})
    monkeypatch.setattr(sam3_utils.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(sam3_utils, "_build_sam3_model_and_processor", build)

    sam3_utils._get_sam3_model_and_processor(
        str(checkpoint), "cuda", expected_checkpoint_sha256=expected_digest
    )

    assert builder_devices == ["cuda:2"]
    assert {key[1] for key in sam3_utils._SAM3_PREDICT_INST_CACHE} == {"cuda:2"}


def test_model_cache_reloads_when_inference_contract_fingerprint_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches reusing a model after the fixed predict_inst contract changes."""
    checkpoint = _checkpoint(tmp_path, b"checkpoint contents")
    expected_digest = hashlib.sha256(b"checkpoint contents").hexdigest()
    builder_calls: list[int] = []

    def build(*_args: object) -> tuple[object, object]:
        builder_calls.append(1)
        return object(), object()

    monkeypatch.setattr(sam3_utils, "_SAM3_PREDICT_INST_CACHE", {})
    monkeypatch.setattr(sam3_utils, "_build_sam3_model_and_processor", build)
    monkeypatch.setattr(sam3_utils, "_PREDICT_INST_CONTRACT_FINGERPRINT", "contract-a")
    sam3_utils._get_sam3_model_and_processor(
        str(checkpoint), "cuda:0", expected_checkpoint_sha256=expected_digest
    )
    monkeypatch.setattr(sam3_utils, "_PREDICT_INST_CONTRACT_FINGERPRINT", "contract-b")
    sam3_utils._get_sam3_model_and_processor(
        str(checkpoint), "cuda:0", expected_checkpoint_sha256=expected_digest
    )

    assert builder_calls == [1, 1]
    assert {key[2] for key in sam3_utils._SAM3_PREDICT_INST_CACHE} == {"contract-a", "contract-b"}


def test_final_digest_change_before_cache_publish_rejects_without_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches publishing between post-build and final checkpoint verification."""
    checkpoint = _checkpoint(tmp_path, b"checkpoint contents")
    digest_before = "d" * 64
    digest_after = "e" * 64
    digests = iter((digest_before, digest_before, digest_after))

    monkeypatch.setattr(sam3_utils, "_SAM3_PREDICT_INST_CACHE", {})
    monkeypatch.setattr(sam3_utils, "checkpoint_sha256", lambda _path: next(digests))
    monkeypatch.setattr(sam3_utils, "_build_sam3_model_and_processor", lambda *_args: (object(), object()))

    with pytest.raises(RuntimeError, match="changed while loading"):
        sam3_utils._get_sam3_model_and_processor(
            str(checkpoint), "cuda:0", expected_checkpoint_sha256=digest_before
        )

    assert sam3_utils._SAM3_PREDICT_INST_CACHE == {}
