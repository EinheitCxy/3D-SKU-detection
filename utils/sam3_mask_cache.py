"""Processed-space, complete-frame SAM3 self-exemplar mask cache."""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Callable, Iterator, Literal, Mapping

import numpy as np


SCHEMA = "sam3_self_exemplar_processed_mask_cache_v1"
_SAFE_INTEGER_LIMIT = (1 << 53) - 1
_INFERENCE_CONTRACT = {
    "api": "self_exemplar",
    "threshold": 0.5,
    "image_size": 1008,
    "max_batch_size": 32,
    "max_dets_per_query": 1,
    "clip_to_bbox": True,
}


class FrameMaskCacheError(RuntimeError):
    """A frame cache entry is missing, malformed, or mismatched."""


@dataclass(frozen=True)
class ProcessedDetectionPrompt:
    object_id: int
    source_bbox_xyxy: tuple[float, float, float, float]
    processed_bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameMaskCacheRequest:
    cache_root: Path
    image_id: int
    image_path: Path
    source_size_wh: tuple[int, int]
    processed_shape_hw: tuple[int, int]
    source_to_processed_affine: np.ndarray
    detections: tuple[ProcessedDetectionPrompt, ...]
    inference_contract: Mapping[str, object]


@dataclass(frozen=True)
class FrameMaskCacheResult:
    masks_by_object_id: Mapping[int, np.ndarray]
    cache_event: Literal["hit", "miss"]
    schema: str


@dataclass(frozen=True)
class _ValidatedRequest:
    manifest_prefix: dict[str, object]
    object_ids: tuple[int, ...]
    processed_bboxes: tuple[tuple[float, float, float, float], ...]
    processed_shape_hw: tuple[int, int]


def _safe_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise FrameMaskCacheError(f"{name} must be a safe integer")
    result = int(value)
    if not -_SAFE_INTEGER_LIMIT <= result <= _SAFE_INTEGER_LIMIT:
        raise FrameMaskCacheError(f"{name} must be a safe integer")
    return result


def _positive_pair(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise FrameMaskCacheError(f"{name} must contain two positive integers")
    first = _safe_integer(value[0], name)
    second = _safe_integer(value[1], name)
    if first <= 0 or second <= 0:
        raise FrameMaskCacheError(f"{name} must contain two positive integers")
    return first, second


def _bbox(
    value: object, bounds_wh: tuple[int, int], name: str
) -> tuple[float, float, float, float]:
    if not isinstance(value, tuple) or len(value) != 4:
        raise FrameMaskCacheError(f"{name} must contain four finite coordinates")
    try:
        x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    except (TypeError, ValueError) as exc:
        raise FrameMaskCacheError(
            f"{name} must contain four finite coordinates"
        ) from exc
    if not all(math.isfinite(coordinate) for coordinate in (x1, y1, x2, y2)):
        raise FrameMaskCacheError(f"{name} coordinates must be finite")
    width, height = bounds_wh
    if x1 > x2 or y1 > y2:
        raise FrameMaskCacheError(f"{name} coordinates must be ordered")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise FrameMaskCacheError(
            f"{name} must be clipped to its coordinate-space bounds"
        )
    return x1, y1, x2, y2


def _validated_request(request: FrameMaskCacheRequest) -> _ValidatedRequest:
    cache_root = Path(request.cache_root)
    if cache_root.name != "v2":
        raise FrameMaskCacheError("cache_root must name the v2 cache root")
    image_id = _safe_integer(request.image_id, "image_id")
    source_size_wh = _positive_pair(request.source_size_wh, "source_size_wh")
    processed_height, processed_width = _positive_pair(
        request.processed_shape_hw, "processed_shape_hw"
    )
    affine = np.asarray(request.source_to_processed_affine)
    if affine.shape != (2, 3):
        raise FrameMaskCacheError("source_to_processed_affine must have shape (2, 3)")
    try:
        affine_values = affine.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise FrameMaskCacheError("source_to_processed_affine must be finite") from exc
    if not np.all(np.isfinite(affine_values)):
        raise FrameMaskCacheError("source_to_processed_affine must be finite")
    try:
        inference = dict(request.inference_contract)
    except (TypeError, ValueError) as exc:
        raise FrameMaskCacheError("inference contract is invalid") from exc
    if set(inference) != set(_INFERENCE_CONTRACT):
        raise FrameMaskCacheError(
            "inference contract keys do not match the v2 contract"
        )
    if (
        inference["api"] != "self_exemplar"
        or type(inference["threshold"]) is not float
        or inference["threshold"] != 0.5
        or type(inference["image_size"]) is not int
        or inference["image_size"] != 1008
        or type(inference["max_batch_size"]) is not int
        or inference["max_batch_size"] != 32
        or type(inference["max_dets_per_query"]) is not int
        or inference["max_dets_per_query"] != 1
        or type(inference["clip_to_bbox"]) is not bool
        or inference["clip_to_bbox"] is not True
    ):
        raise FrameMaskCacheError(
            "inference contract values do not match the v2 contract"
        )
    if not isinstance(request.detections, tuple):
        raise FrameMaskCacheError("detections must be an ordered tuple")
    object_ids: list[int] = []
    processed_bboxes: list[tuple[float, float, float, float]] = []
    manifest_detections: list[dict[str, object]] = []
    for mask_index, prompt in enumerate(request.detections):
        if not isinstance(prompt, ProcessedDetectionPrompt):
            raise FrameMaskCacheError(
                "detections must contain ProcessedDetectionPrompt values"
            )
        object_id = _safe_integer(prompt.object_id, "object_id")
        source_bbox = _bbox(prompt.source_bbox_xyxy, source_size_wh, "source_bbox_xyxy")
        processed_bbox = _bbox(
            prompt.processed_bbox_xyxy,
            (processed_width, processed_height),
            "processed_bbox_xyxy",
        )
        object_ids.append(object_id)
        processed_bboxes.append(processed_bbox)
        manifest_detections.append(
            {
                "object_id": object_id,
                "source_bbox_xyxy": list(source_bbox),
                "processed_bbox_xyxy": list(processed_bbox),
                "mask_index": mask_index,
            }
        )
    if len(set(object_ids)) != len(object_ids):
        raise FrameMaskCacheError("detection object IDs must be unique")
    return _ValidatedRequest(
        manifest_prefix={
            "schema": SCHEMA,
            "image_id": image_id,
            "source_size_wh": list(source_size_wh),
            "processed_shape_hw": [processed_height, processed_width],
            "source_to_processed_affine": affine_values.tolist(),
            "detections": manifest_detections,
            "inference": _INFERENCE_CONTRACT.copy(),
        },
        object_ids=tuple(object_ids),
        processed_bboxes=tuple(processed_bboxes),
        processed_shape_hw=(processed_height, processed_width),
    )


def _entry_path(request: FrameMaskCacheRequest) -> Path:
    return Path(request.cache_root) / "entries" / str(int(request.image_id))


def _lock_path(request: FrameMaskCacheRequest) -> Path:
    return Path(request.cache_root) / "locks" / f"{int(request.image_id)}.lock"


@contextlib.contextmanager
def _frame_lock(request: FrameMaskCacheRequest, *, exclusive: bool) -> Iterator[None]:
    lock_path = _lock_path(request)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _canonical_clip(
    mask: np.ndarray, bbox_xyxy: tuple[float, float, float, float]
) -> np.ndarray:
    """Apply the producer's pixel-boundary clip rule in processed space."""
    height, width = mask.shape
    x1, y1, x2, y2 = bbox_xyxy
    xi1 = int(max(0, min(width - 1, round(x1))))
    yi1 = int(max(0, min(height - 1, round(y1))))
    xi2 = int(max(xi1 + 1, min(width, round(x2))))
    yi2 = int(max(yi1 + 1, min(height, round(y2))))
    clipped = mask.copy()
    clipped[:yi1, :] = False
    clipped[yi2:, :] = False
    clipped[:, :xi1] = False
    clipped[:, xi2:] = False
    return clipped


def _pack_masks(masks: np.ndarray) -> np.ndarray:
    mask_count, height, width = masks.shape
    return np.packbits(
        masks.reshape(mask_count, height * width), axis=1, bitorder="little"
    )


def _unpack_masks(
    packed: np.ndarray, mask_count: int, shape_hw: tuple[int, int]
) -> np.ndarray:
    flat_size = shape_hw[0] * shape_hw[1]
    unpacked = np.unpackbits(packed, axis=1, count=flat_size, bitorder="little")
    return unpacked.reshape(mask_count, *shape_hw).astype(bool, copy=False)


def _manifest(validated: _ValidatedRequest) -> dict[str, object]:
    mask_count = len(validated.object_ids)
    flat_mask_size = validated.processed_shape_hw[0] * validated.processed_shape_hw[1]
    packed_width = (flat_mask_size + 7) // 8
    return {
        **validated.manifest_prefix,
        "payload": {
            "path": "masks.npz",
            "array": "packed_masks",
            "dtype": "uint8",
            "bitorder": "little",
            "mask_count": mask_count,
            "flat_mask_size": flat_mask_size,
            "packed_width": packed_width,
        },
        "complete": True,
    }


def _load_entry(
    request: FrameMaskCacheRequest, validated: _ValidatedRequest
) -> FrameMaskCacheResult:
    entry = _entry_path(request)
    if not entry.exists():
        raise FrameMaskCacheError("cache entry is missing")
    if not entry.is_dir():
        raise FrameMaskCacheError("cache entry is not a directory")
    manifest_path = entry / "manifest.json"
    payload_path = entry / "masks.npz"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameMaskCacheError("cache manifest is malformed") from exc
    expected_manifest = _manifest(validated)
    if manifest != expected_manifest:
        raise FrameMaskCacheError(
            "cache manifest does not match the requested v2 contract"
        )
    try:
        with np.load(payload_path, allow_pickle=False) as payload:
            if payload.files != ["packed_masks"]:
                raise FrameMaskCacheError(
                    "cache payload keys do not match the v2 contract"
                )
            packed = payload["packed_masks"].copy()
    except FrameMaskCacheError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise FrameMaskCacheError("cache payload is malformed") from exc
    mask_count = len(validated.object_ids)
    flat_size = validated.processed_shape_hw[0] * validated.processed_shape_hw[1]
    packed_width = (flat_size + 7) // 8
    if packed.dtype != np.uint8 or packed.shape != (mask_count, packed_width):
        raise FrameMaskCacheError(
            "cache payload dtype or shape does not match the v2 contract"
        )
    used_bits = flat_size % 8
    tail_mask = ((1 << (8 - used_bits)) - 1) << used_bits if used_bits else 0
    if tail_mask and np.any(packed[:, -1] & np.uint8(tail_mask)):
        raise FrameMaskCacheError("cache payload has nonzero tail bits")
    masks = _unpack_masks(packed, mask_count, validated.processed_shape_hw)
    masks_by_object_id: dict[int, np.ndarray] = {}
    for object_id, bbox_xyxy, mask in zip(
        validated.object_ids, validated.processed_bboxes, masks
    ):
        if not np.array_equal(mask, _canonical_clip(mask, bbox_xyxy)):
            raise FrameMaskCacheError(
                "cache mask contains true pixels outside its processed bbox"
            )
        masks_by_object_id[object_id] = mask.copy()
    return FrameMaskCacheResult(masks_by_object_id, "hit", SCHEMA)


def load_complete_frame_masks(request: FrameMaskCacheRequest) -> FrameMaskCacheResult:
    """Read a complete, matching v2 frame entry without invoking inference."""
    validated = _validated_request(request)
    with _frame_lock(request, exclusive=False):
        return _load_entry(request, validated)


def _validate_computed_masks(
    computed: Mapping[int, np.ndarray], validated: _ValidatedRequest
) -> np.ndarray:
    if not isinstance(computed, Mapping):
        raise FrameMaskCacheError(
            "mask producer must return a mapping keyed by object IDs"
        )
    computed_ids = {
        _safe_integer(object_id, "computed object_id") for object_id in computed
    }
    if computed_ids != set(validated.object_ids) or len(computed) != len(
        validated.object_ids
    ):
        raise FrameMaskCacheError(
            "mask producer object IDs must exactly match requested object IDs"
        )
    masks: list[np.ndarray] = []
    for object_id, bbox_xyxy in zip(validated.object_ids, validated.processed_bboxes):
        mask = np.asarray(computed[object_id])
        if mask.dtype != np.bool_:
            raise FrameMaskCacheError(
                "mask producer must return boolean processed masks"
            )
        if mask.shape != validated.processed_shape_hw:
            raise FrameMaskCacheError(
                "mask producer returned a mask with the wrong processed shape"
            )
        masks.append(_canonical_clip(mask, bbox_xyxy))
    if masks:
        return np.stack(masks, axis=0)
    height, width = validated.processed_shape_hw
    return np.zeros((0, height, width), dtype=bool)


def _quarantine_entry(request: FrameMaskCacheRequest) -> None:
    entry = _entry_path(request)
    if not entry.exists():
        return
    corrupt = Path(request.cache_root) / "corrupt"
    corrupt.mkdir(parents=True, exist_ok=True)
    destination = corrupt / f"{int(request.image_id)}.{time.time_ns()}"
    os.replace(entry, destination)


def _write_entry(
    request: FrameMaskCacheRequest, validated: _ValidatedRequest, masks: np.ndarray
) -> None:
    entries = Path(request.cache_root) / "entries"
    entries.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{int(request.image_id)}.", dir=entries))
    try:
        np.savez(temporary / "masks.npz", packed_masks=_pack_masks(masks))
        (temporary / "manifest.json").write_text(
            json.dumps(_manifest(validated), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        final_entry = _entry_path(request)
        if final_entry.exists():
            raise FrameMaskCacheError("cache entry appeared while publishing")
        os.rename(temporary, final_entry)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_or_compute_frame_masks(
    request: FrameMaskCacheRequest,
    compute_masks: Callable[[], Mapping[int, np.ndarray]],
) -> FrameMaskCacheResult:
    """Load a complete v2 frame or compute and atomically publish it once."""
    validated = _validated_request(request)
    with _frame_lock(request, exclusive=False):
        try:
            return _load_entry(request, validated)
        except FrameMaskCacheError:
            pass
    with _frame_lock(request, exclusive=True):
        try:
            return _load_entry(request, validated)
        except FrameMaskCacheError:
            if _entry_path(request).exists():
                _quarantine_entry(request)
        masks = _validate_computed_masks(compute_masks(), validated)
        _write_entry(request, validated, masks)
        return FrameMaskCacheResult(
            {
                object_id: mask.copy()
                for object_id, mask in zip(validated.object_ids, masks)
            },
            "miss",
            SCHEMA,
        )
