"""Immutable, audited per-source-frame SAM3 mask cache primitives.

This module intentionally knows nothing about global IDs, DA3, or measurement
totals.  It stores only the complete ordered source-frame mask bundle supplied
by its caller.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np

from utils.sam3_utils import clip_mask_to_bbox


_SCHEMA_VERSION = "sam3_frame_mask_cache_v2_canonical_bbox_clip"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _canonical_binary64_hex(encoded: str) -> tuple[str, float]:
    try:
        raw = bytes.fromhex(encoded)
        value = struct.unpack(">d", raw)[0]
    except (TypeError, ValueError, struct.error) as exc:
        raise ValueError("invalid binary64 bbox encoding") from exc
    if len(raw) != 8 or not math.isfinite(value):
        raise ValueError("bbox coordinates must be finite binary64 values")
    if value == 0.0:
        value = 0.0
    return struct.pack(">d", value).hex(), value


@dataclass(frozen=True)
class DetectionPrompt:
    """One ordered source-frame prompt with canonical binary64 bbox values."""

    object_id: int
    bbox_xyxy_f64be_hex: tuple[str, str, str, str]

    def __post_init__(self) -> None:
        if len(self.bbox_xyxy_f64be_hex) != 4:
            raise ValueError("bbox_xyxy_f64be_hex must contain exactly four values")
        normalized: list[str] = []
        values: list[float] = []
        for encoded in self.bbox_xyxy_f64be_hex:
            canonical, value = _canonical_binary64_hex(encoded)
            normalized.append(canonical)
            values.append(value)
        if values[2] < values[0] or values[3] < values[1]:
            raise ValueError("bbox_xyxy must not be reversed")
        object.__setattr__(self, "object_id", int(self.object_id))
        object.__setattr__(self, "bbox_xyxy_f64be_hex", tuple(normalized))

    @classmethod
    def from_bbox(
        cls, object_id: int, bbox_xyxy: Sequence[float]
    ) -> "DetectionPrompt":
        if len(bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain exactly four coordinates")
        encoded: list[str] = []
        for coordinate in bbox_xyxy:
            value = float(coordinate)
            if not math.isfinite(value):
                raise ValueError("bbox coordinates must be finite binary64 values")
            if value == 0.0:
                value = 0.0
            encoded.append(struct.pack(">d", value).hex())
        return cls(int(object_id), tuple(encoded))  # type: ignore[arg-type]

    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        values: list[float] = []
        for encoded in self.bbox_xyxy_f64be_hex:
            _, value = _canonical_binary64_hex(encoded)
            values.append(value)
        return tuple(values)  # type: ignore[return-value]


@dataclass(frozen=True)
class FrameMaskCacheRequest:
    cache_root: Path
    image_id: int
    image_path: Path
    detections: Sequence[DetectionPrompt]
    checkpoint_path: Path
    checkpoint_sha256: str
    code_fingerprint: Mapping[str, object]
    runtime_fingerprint: Mapping[str, object]
    inference_contract: Mapping[str, object]
    output_shape_hw: tuple[int, int]

    def __post_init__(self) -> None:
        digest = str(self.checkpoint_sha256).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "checkpoint_sha256", digest)
        object.__setattr__(self, "detections", tuple(self.detections))


@dataclass(frozen=True)
class FrameMaskCacheResult:
    masks: tuple[np.ndarray, ...]
    key: str
    events: tuple[str, ...]
    payload_sha256: str | None
    checkpoint_sha256: str
    code_fingerprint: Mapping[str, object]
    invalid_reason: str | None


def _source_image_identity(image_path: Path) -> dict[str, object]:
    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read source image for cache key: {image_path}") from exc
    return {"sha256": _sha256_bytes(image_bytes), "size_bytes": len(image_bytes)}


def _request_key_payload(
    request: FrameMaskCacheRequest, source_image_identity: Mapping[str, object] | None = None
) -> dict[str, object]:
    image_identity = dict(source_image_identity or _source_image_identity(request.image_path))
    height, width = request.output_shape_hw
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        raise ValueError("output_shape_hw must contain positive integer height and width")
    detections = []
    for prompt in request.detections:
        # Validate raw canonical form before any cache filesystem operation.
        prompt.bbox_xyxy()
        detections.append(
            {"object_id": int(prompt.object_id), "bbox_xyxy_f64be_hex": list(prompt.bbox_xyxy_f64be_hex)}
        )
    return {
        "schema": _SCHEMA_VERSION,
        "image": {"image_id": int(request.image_id), **image_identity},
        "detections": detections,
        "checkpoint_sha256": request.checkpoint_sha256,
        "code_fingerprint": dict(request.code_fingerprint),
        "runtime_fingerprint": dict(request.runtime_fingerprint),
        "inference_contract": dict(request.inference_contract),
        "output_contract": {"shape_hw": [height, width], "dtype": "bool"},
    }


def canonical_frame_mask_key(request: FrameMaskCacheRequest) -> str:
    """Return the content-addressed key for one complete source-frame bundle."""
    return _sha256_bytes(_canonical_json_bytes(_request_key_payload(request)))


def _paths(request: FrameMaskCacheRequest, key: str) -> tuple[Path, Path, Path, Path]:
    root = request.cache_root
    return root / "locks", root / "entries", root / "corrupt", root / "entries" / key


@contextlib.contextmanager
def _key_lock(cache_root: Path, key: str, *, exclusive: bool) -> Iterator[None]:
    locks = cache_root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock_path = locks / f"{key}.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_produced_masks(
    request: FrameMaskCacheRequest, masks: Sequence[np.ndarray]
) -> tuple[np.ndarray, ...]:
    if len(masks) != len(request.detections):
        raise ValueError("mask producer returned a different number of masks than detections")
    validated: list[np.ndarray] = []
    for prompt, mask in zip(request.detections, masks):
        array = np.asarray(mask)
        if array.dtype != np.bool_:
            raise ValueError("mask producer must return boolean source masks")
        if array.shape != request.output_shape_hw:
            raise ValueError("mask producer returned a mask with the wrong source shape")
        validated.append(clip_mask_to_bbox(array, prompt.bbox_xyxy()))
    return tuple(validated)


def _mask_metadata(mask: np.ndarray) -> dict[str, object]:
    return {"sha256": _sha256_bytes(mask.tobytes(order="C")), "true_pixel_count": int(mask.sum())}


def _manifest(
    key: str,
    key_payload: Mapping[str, object],
    masks: tuple[np.ndarray, ...],
    payload_sha256: str,
) -> dict[str, object]:
    return {
        "complete": True,
        "key": key,
        "key_payload": dict(key_payload),
        "payload_sha256": payload_sha256,
        "masks": [_mask_metadata(mask) for mask in masks],
    }


def _load_valid_bundle(
    request: FrameMaskCacheRequest, key: str, key_payload: Mapping[str, object]
) -> tuple[FrameMaskCacheResult | None, str | None]:
    _, _, _, entry = _paths(request, key)
    if not entry.exists():
        return None, None
    manifest_path = entry / "manifest.json"
    payload_path = entry / "masks.npz"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("complete") is not True:
            raise ValueError("manifest is incomplete")
        if manifest.get("key") != key or manifest.get("key_payload") != key_payload:
            raise ValueError("manifest request provenance does not match")
        payload_sha256 = manifest.get("payload_sha256")
        if not isinstance(payload_sha256, str) or _sha256_file(payload_path) != payload_sha256:
            raise ValueError("payload SHA-256 does not match manifest")
        with np.load(payload_path, allow_pickle=False) as payload:
            if payload.files != ["masks"]:
                raise ValueError("payload must contain exactly one masks array")
            array = payload["masks"]
        expected_shape = (len(request.detections), *request.output_shape_hw)
        if array.dtype != np.bool_ or array.shape != expected_shape:
            raise ValueError("payload mask dtype or shape does not match contract")
        metadata = manifest.get("masks")
        if not isinstance(metadata, list) or len(metadata) != len(request.detections):
            raise ValueError("manifest mask metadata does not match detections")
        masks = []
        for prompt, mask, expected in zip(request.detections, array, metadata):
            if not isinstance(expected, dict) or expected != _mask_metadata(mask):
                raise ValueError("per-mask digest or pixel count does not match")
            if not np.array_equal(mask, clip_mask_to_bbox(mask, prompt.bbox_xyxy())):
                raise ValueError("cached mask contains true pixels outside its clipped bbox")
            masks.append(mask.copy())
        return (
            FrameMaskCacheResult(
                masks=tuple(masks),
                key=key,
                events=("hit",),
                payload_sha256=payload_sha256,
                checkpoint_sha256=request.checkpoint_sha256,
                code_fingerprint=dict(request.code_fingerprint),
                invalid_reason=None,
            ),
            None,
        )
    except (OSError, ValueError, json.JSONDecodeError, KeyError, zipfile.BadZipFile) as exc:
        return None, str(exc)


def _quarantine_invalid_entry(request: FrameMaskCacheRequest, key: str) -> None:
    _, entries, corrupt, entry = _paths(request, key)
    if not entry.exists():
        return
    corrupt.mkdir(parents=True, exist_ok=True)
    destination = corrupt / f"{key}.{uuid.uuid4().hex}"
    os.replace(entry, destination)
    _fsync_directory(entries)
    _fsync_directory(corrupt)


def _publish_new_bundle(
    request: FrameMaskCacheRequest,
    key: str,
    key_payload: Mapping[str, object],
    masks: tuple[np.ndarray, ...],
    *,
    invalid_reason: str | None,
) -> FrameMaskCacheResult:
    _, entries, _, final_entry = _paths(request, key)
    temporary: Path | None = None
    try:
        entries.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=entries))
        payload_path = temporary / "masks.npz"
        array = (
            np.stack(masks, axis=0).astype(bool, copy=False)
            if masks
            else np.zeros((0, *request.output_shape_hw), dtype=bool)
        )
        np.savez_compressed(payload_path, masks=array)
        _fsync_file(payload_path)
        payload_sha256 = _sha256_file(payload_path)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(_canonical_json_bytes(_manifest(key, key_payload, masks, payload_sha256)))
        _fsync_file(manifest_path)
        _fsync_directory(temporary)
        if final_entry.exists():
            raise FileExistsError(f"cache entry unexpectedly appeared: {final_entry}")
        os.rename(temporary, final_entry)
        _fsync_directory(entries)
        return FrameMaskCacheResult(
            masks=masks,
            key=key,
            events=("invalid", "written") if invalid_reason else ("miss", "written"),
            payload_sha256=payload_sha256,
            checkpoint_sha256=request.checkpoint_sha256,
            code_fingerprint=dict(request.code_fingerprint),
            invalid_reason=invalid_reason,
        )
    except OSError as exc:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        return FrameMaskCacheResult(
            masks=masks,
            key=key,
            events=("invalid", "cache_write_failed") if invalid_reason else ("miss", "cache_write_failed"),
            payload_sha256=None,
            checkpoint_sha256=request.checkpoint_sha256,
            code_fingerprint=dict(request.code_fingerprint),
            invalid_reason=str(exc),
        )


def load_or_compute_frame_masks(
    request: FrameMaskCacheRequest,
    compute_masks: Callable[[], Sequence[np.ndarray]],
) -> FrameMaskCacheResult:
    """Load one immutable bundle or compute and atomically publish it.

    ``checkpoint_sha256`` is caller-supplied opaque input.  This utility does
    not verify the checkpoint file or load-time equality; Task 3 owns that
    TOCTOU contract.
    """
    source_identity = _source_image_identity(request.image_path)
    key_payload = _request_key_payload(request, source_identity)
    key = _sha256_bytes(_canonical_json_bytes(key_payload))
    with _key_lock(request.cache_root, key, exclusive=False):
        cached, _ = _load_valid_bundle(request, key, key_payload)
        if cached is not None:
            return cached
    with _key_lock(request.cache_root, key, exclusive=True):
        cached, invalid_reason = _load_valid_bundle(request, key, key_payload)
        if cached is not None:
            return cached
        if invalid_reason is not None:
            _quarantine_invalid_entry(request, key)
        if _source_image_identity(request.image_path) != source_identity:
            raise ValueError("source image changed before mask production")
        masks = _validate_produced_masks(request, tuple(compute_masks()))
        if _source_image_identity(request.image_path) != source_identity:
            raise ValueError("source image changed during mask production")
        return _publish_new_bundle(request, key, key_payload, masks, invalid_reason=invalid_reason)
