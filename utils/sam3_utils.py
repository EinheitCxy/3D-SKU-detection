"""
SAM3 utilities (optional dependency).

This module encapsulates all SAM3-related code so the rest of the pipeline can
stay clean and continue to work when SAM3 is disabled/unavailable.

Design goals:
  - No HF downloads (load_from_HF=False), so it works in restricted environments.
  - Minimal integration points: bbox -> mask; mask -> sampled points; mask -> sampled 3D points.

API Modes:
  - predict_inst API (default): Fast single-image inference, supports text + geometry prompts.
    Does NOT support visual exemplars.
  - batch inference API: Supports visual exemplars (positive/negative reference boxes).
    Use `sam3_masks_from_bboxes_batch_api()` for visual exemplar functionality.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
import threading
from contextlib import contextmanager, nullcontext
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .config import SKUMatchingConfig
from .sam3_mask_cache import (
    FrameMaskCacheRequest,
    ProcessedDetectionPrompt,
    load_or_compute_frame_masks,
    map_source_bbox_to_processed,
)
from .transforms import ImageTransformBase

logger = logging.getLogger(__name__)

# The predict_inst contract is fixed for this process-cache boundary.  Keep its
# fingerprint separate from the mutable checkpoint path so loaded weights cannot
# be reused for different checkpoint bytes or inference behavior.
_PREDICT_INST_CONTRACT_FINGERPRINT = hashlib.sha256(
    json.dumps(
        {
            "api": "predict_inst",
            "builder": {"enable_inst_interactivity": True, "load_from_HF": False},
            "empty_mask_retry": "bbox_center_positive_point",
            "mask_postprocess": "best_iou_then_clip_to_bbox",
            "predict": {
                "multimask_output": True,
                "normalize_coords": True,
                "return_logits": False,
            },
            "processor": {"confidence_threshold": 0.0},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

# Cache for predict_inst API (model + processor), keyed by immutable checkpoint
# bytes, normalized device, and the fixed predict_inst inference contract.
_SAM3_PREDICT_INST_CACHE: Dict[Tuple[str, str, str], Tuple[object, object]] = {}

# Cache for batch inference API (model + transform + postprocessor)
_SAM3_BATCH_API_CACHE: Dict[Tuple[Any, ...], Tuple[object, object, object]] = {}

# Global counter for batch inference query IDs
_BATCH_QUERY_COUNTER = 0
_CACHE_BOUNDARY_RNG_LOCK = threading.RLock()

# Lock guarding _BATCH_QUERY_COUNTER for thread-safe allocation under
# parallel_refs ThreadPoolExecutor (query_id must be unique across threads).
_BATCH_QUERY_LOCK = threading.Lock()


def _sam3_autocast_context(model: Any, device: str) -> Any:
    """Return a safe autocast context matching SAM3 model dtype.

    Older/mixed checkpoints can report BF16 input/FP32-weight mismatches when a
    fixed BF16 autocast is used. We infer model dtype from its first tensor
    parameter and align the nested autocast context:
      - half precision weights -> keep corresponding autocast on
      - fp32 weights -> force autocast off to avoid inherited outer BF16 contexts
    """
    if not device.startswith("cuda"):
        return torch.autocast(device_type=device, enabled=False)

    params = [
        p
        for p in getattr(model, "parameters", lambda: [])()
        if isinstance(p, torch.Tensor)
    ]
    if not params:
        return nullcontext()

    dtype = params[0].dtype
    if dtype == torch.float16:
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    if dtype == torch.bfloat16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

    return torch.autocast(device_type="cuda", enabled=False)


def _coerce_sam3_batch_dtype(batch: Any, model: Any) -> Any:
    """Cast floating tensors in batch to the model's parameter dtype when needed."""
    params = [
        p
        for p in getattr(model, "parameters", lambda: [])()
        if isinstance(p, torch.Tensor)
    ]
    if not params:
        return batch
    target_dtype = params[0].dtype
    if target_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return batch

    if isinstance(batch, torch.Tensor):
        if batch.is_floating_point() and batch.dtype != target_dtype:
            return batch.to(dtype=target_dtype)
        return batch
    if isinstance(batch, tuple):
        return tuple(_coerce_sam3_batch_dtype(item, model) for item in batch)
    if isinstance(batch, list):
        return [_coerce_sam3_batch_dtype(item, model) for item in batch]
    if isinstance(batch, dict):
        return {k: _coerce_sam3_batch_dtype(v, model) for k, v in batch.items()}

    return batch


def _infer_transform_output_size(
    transform: Optional[ImageTransformBase],
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    if transform is None:
        return None, None

    target_w = getattr(transform, "target_width", None)
    target_h = getattr(transform, "target_height", None)
    if (
        isinstance(target_w, int)
        and isinstance(target_h, int)
        and target_w > 0
        and target_h > 0
    ):
        return "pi3", (target_w, target_h)

    padded_w = getattr(transform, "padded_width", None)
    padded_h = getattr(transform, "padded_height", None)
    if (
        isinstance(padded_w, int)
        and isinstance(padded_h, int)
        and padded_w > 0
        and padded_h > 0
    ):
        return "vggt", (padded_w, padded_h)

    final_w = getattr(transform, "final_width", None)
    final_h = getattr(transform, "final_height", None)
    if (
        isinstance(final_w, int)
        and isinstance(final_h, int)
        and final_w > 0
        and final_h > 0
    ):
        return "vggt", (final_w, final_h)

    return None, None


def normalize_device(device: str) -> str:
    d = str(device).strip()
    if d == "cuda":
        return f"cuda:{torch.cuda.current_device()}"
    if d.startswith("cuda:"):
        return d
    if d.startswith("cpu"):
        return "cpu"
    return d


def clip_mask_to_bbox(mask: np.ndarray, bbox_xyxy: Sequence[float]) -> np.ndarray:
    """将 mask 裁剪到 bbox 范围内，bbox 外部设为 False

    Args:
        mask: 布尔 mask，形状 (H, W)
        bbox_xyxy: 边界框 [x1, y1, x2, y2]，像素坐标

    Returns:
        裁剪后的 mask，形状 (H, W)
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    h, w = mask.shape[:2]  # mask 的 shape 是 (height, width)

    # 将 bbox 坐标限制在图像范围内
    # 注意：x 对应 width，y 对应 height
    xi1 = int(max(0, min(w - 1, round(x1))))
    yi1 = int(max(0, min(h - 1, round(y1))))
    xi2 = int(max(xi1 + 1, min(w, round(x2))))
    yi2 = int(max(yi1 + 1, min(h, round(y2))))

    # 如果 bbox 无效，返回全 False mask
    if xi2 <= xi1 or yi2 <= yi1:
        return np.zeros((h, w), dtype=bool)

    # 创建输出 mask（只保留 bbox 内部）
    out = mask.astype(bool).copy()

    # 将 bbox 外部区域设为 False
    out[:yi1, :] = False  # 上方
    out[yi2:, :] = False  # 下方
    out[:, :xi1] = False  # 左侧
    out[:, xi2:] = False  # 右侧

    return out


def _ensure_sam3_in_path() -> Path:
    """Ensure SAM3 repo is in sys.path, return the repo path."""
    sam3_repo = Path(__file__).resolve().parents[1] / "sam3"
    if not (sam3_repo / "sam3" / "__init__.py").exists():
        raise ImportError(f"SAM3 repo not found at {sam3_repo}")
    sam3_repo_str = str(sam3_repo)
    if sam3_repo_str not in sys.path:
        sys.path.insert(0, sam3_repo_str)
    return sam3_repo


def checkpoint_sha256(path: Path) -> str:
    """Return the SHA-256 digest of checkpoint bytes without loading them."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_sam3_model_and_processor(
    checkpoint_path: str, device: str
) -> Tuple[object, object]:
    """Build one predict_inst model/processor pair from local SAM3 source."""
    _ensure_sam3_in_path()

    from sam3.model_builder import build_sam3_image_model  # type: ignore
    from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore

    setup_device = "cuda" if device.startswith("cuda") else device
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        device=setup_device,
        # Use the dedicated box-prompt mask head (SAM1/2-style) for bbox->mask.
        enable_inst_interactivity=True,
    )
    if setup_device == "cuda":
        model = model.to(device)
    processor = Sam3Processor(model, device=device, confidence_threshold=0.0)
    return model, processor


def _get_sam3_model_and_processor(
    checkpoint_path: str, device: str, *, expected_checkpoint_sha256: str
) -> Tuple[object, object]:
    """Load SAM3 model+processor from local `sam3/` directory (no HF download).

    This is for the predict_inst API (fast single-image inference).
    Does NOT support visual exemplars.
    """
    checkpoint = Path(checkpoint_path)
    before = checkpoint_sha256(checkpoint)
    if before != expected_checkpoint_sha256:
        raise RuntimeError("SAM3 checkpoint digest changed before loading")

    canonical_device = normalize_device(device)
    cache_key = (before, canonical_device, _PREDICT_INST_CONTRACT_FINGERPRINT)
    cached = _SAM3_PREDICT_INST_CACHE.get(cache_key)
    if cached is None:
        candidate = _build_sam3_model_and_processor(checkpoint_path, canonical_device)
        if checkpoint_sha256(checkpoint) != before:
            raise RuntimeError("SAM3 checkpoint changed while loading")
        # Check again immediately before cache publication so a checkpoint
        # mutation after the post-build check cannot leave a stale entry.
        if checkpoint_sha256(checkpoint) != before:
            raise RuntimeError("SAM3 checkpoint changed while loading")
        _SAM3_PREDICT_INST_CACHE[cache_key] = candidate
        return candidate
    return cached


def _build_sam3_batch_transform(
    *,
    fallback_size: int,
    skip_resize: bool = False,
) -> Any:
    """Build SAM3 batch API transforms."""
    _ensure_sam3_in_path()
    from sam3.train.transforms.basic_for_api import (  # type: ignore
        ComposeAPI,
        NormalizeAPI,
        RandomResizeAPI,
        ToTensorAPI,
    )

    if skip_resize:
        return ComposeAPI(
            transforms=[
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    fallback = int(fallback_size) if fallback_size and int(fallback_size) > 0 else 1008

    return ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=fallback,
                max_size=fallback,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _get_sam3_batch_components(
    checkpoint_path: str,
    device: str,
    detection_threshold: float = 0.5,
    max_dets_per_img: int = -1,
    *,
    fallback_size: int = 1008,
    skip_resize: bool = False,
) -> Tuple[Any, Any, Any]:
    """Load SAM3 model + transform + postprocessor for batch inference API.

    This API supports visual exemplars (positive/negative reference boxes).

    Args:
        checkpoint_path: Path to SAM3 checkpoint
        device: Device to run on ('cuda' or 'cpu')
        detection_threshold: Confidence threshold for detections (default 0.5)
        max_dets_per_img: Maximum detections per image (-1 for unlimited, 1 for self-exemplar mode)

    Returns:
        Tuple of (model, transform, postprocessor)
    """
    # Postprocessor behavior depends on detection_threshold and max_dets_per_img, so include both in cache key.
    fallback = int(fallback_size) if fallback_size and int(fallback_size) > 0 else 1008
    cache_key = (
        str(checkpoint_path),
        str(device),
        round(float(detection_threshold), 6),
        int(max_dets_per_img),
        int(bool(skip_resize)),
        fallback,
    )
    if cache_key in _SAM3_BATCH_API_CACHE:
        return _SAM3_BATCH_API_CACHE[cache_key]

    sam3_repo = _ensure_sam3_in_path()

    from sam3 import build_sam3_image_model  # type: ignore
    from sam3.eval.postprocessors import PostProcessImage  # type: ignore

    # Build model with bpe_path for text processing
    bpe_path = sam3_repo / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.exists():
        logger.warning(f"BPE vocab not found at {bpe_path}, trying without bpe_path")
        bpe_path = None

    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        bpe_path=str(bpe_path) if bpe_path else None,
    )
    model = model.to(device)
    model.eval()

    # Transform aligned to backend geometry when possible.
    transform = _build_sam3_batch_transform(
        fallback_size=fallback_size, skip_resize=skip_resize
    )

    # Postprocessor for extracting masks
    postprocessor = PostProcessImage(
        max_dets_per_img=max_dets_per_img,  # Configurable: -1 for unlimited, 1 for self-exemplar
        iou_type="segm",  # We want masks
        use_original_sizes_box=True,  # Resize boxes to original image size
        use_original_sizes_mask=True,  # Resize masks to original image size
        convert_mask_to_rle=False,  # Keep binary masks for easy processing
        detection_threshold=detection_threshold,
        to_cpu=False,  # Keep on GPU for faster processing
    )

    _SAM3_BATCH_API_CACHE[cache_key] = (model, transform, postprocessor)
    return model, transform, postprocessor


# ============================================================================
# Batch Inference API Helper Functions
# ============================================================================


def _create_empty_datapoint() -> Any:
    """Create an empty Datapoint for batch inference.

    A Datapoint is a container for a single image and its associated queries.
    """
    _ensure_sam3_in_path()
    from sam3.train.data.sam3_image_dataset import Datapoint  # type: ignore

    return Datapoint(find_queries=[], images=[])


def _set_image(datapoint: Any, pil_image: Any) -> None:
    """Set the image to be processed in the datapoint.

    Args:
        datapoint: The Datapoint object
        pil_image: PIL.Image object
    """
    _ensure_sam3_in_path()
    from sam3.train.data.sam3_image_dataset import Image as SAMImage  # type: ignore

    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def _resize_mask_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a binary mask to (height, width) using nearest-neighbor.

    Avoids requiring OpenCV at runtime.
    """
    if mask.shape == (height, width):
        return mask.astype(bool, copy=False)
    from PIL import Image

    im = Image.fromarray(mask.astype(np.uint8) * 255)
    im2 = im.resize((int(width), int(height)), resample=Image.NEAREST)
    return (np.asarray(im2) > 0).astype(bool)


def _prepare_sam3_input_image(
    pil_image: Any, transform: Optional[ImageTransformBase]
) -> Tuple[Any, bool]:
    """Align SAM3 input image to backend final space."""
    if transform is None:
        return pil_image, False

    from PIL import Image

    img = pil_image.convert("RGB") if hasattr(pil_image, "convert") else pil_image
    backend, out_size = _infer_transform_output_size(transform)

    if backend == "pi3" and out_size is not None:
        resized = img.resize(out_size, Image.Resampling.LANCZOS)
        return resized, True

    # VGGT: resize -> crop -> pad (match transform fields).
    proc_w = getattr(transform, "proc_width", None)
    proc_h = getattr(transform, "proc_height", None)
    if (
        backend == "vggt"
        and isinstance(proc_w, int)
        and isinstance(proc_h, int)
        and proc_w > 0
        and proc_h > 0
    ):
        resized = img.resize((proc_w, proc_h), Image.Resampling.BICUBIC)

        crop_applied = bool(getattr(transform, "crop_applied", False))
        crop_start_y = int(getattr(transform, "crop_start_y", 0))
        final_w = int(getattr(transform, "final_width", proc_w))
        final_h = int(getattr(transform, "final_height", proc_h))

        if crop_applied:
            y0 = max(0, min(proc_h, crop_start_y))
            y1 = max(0, min(proc_h, y0 + final_h))
            cropped = resized.crop((0, y0, final_w, y1))
        else:
            cropped = resized
            if cropped.size != (final_w, final_h):
                cropped = cropped.crop((0, 0, final_w, final_h))

        padded_w, padded_h = out_size if out_size is not None else (final_w, final_h)
        pad_left = int(getattr(transform, "batch_pad_left", 0))
        pad_top = int(getattr(transform, "batch_pad_top", 0))

        if padded_w != final_w or padded_h != final_h or pad_left or pad_top:
            canvas = Image.new("RGB", (padded_w, padded_h), (255, 255, 255))
            canvas.paste(cropped, (pad_left, pad_top))
            return canvas, True

        return cropped, True

    return img, False


def map_mask_to_final_space(
    mask: np.ndarray, transform: ImageTransformBase
) -> np.ndarray:
    """把原图坐标系的mask映射到 transform 定义的 final 空间。

    - Pi3: 原图 -> resize到 (target_height, target_width)
    - VGGT: 原图 -> resize到(proc_h, proc_w) -> (可选)按crop_start_y裁剪到final_h -> 按batch_pad_top/left pad到 padded_h/w

    全程 nearest，保证mask二值属性。
    """
    mask_bool = mask.astype(bool, copy=False)

    # ---- Pi3：只做resize ----
    target_w = getattr(transform, "target_width", None)
    target_h = getattr(transform, "target_height", None)
    if (
        isinstance(target_w, int)
        and isinstance(target_h, int)
        and target_w > 0
        and target_h > 0
    ):
        return _resize_mask_nearest(mask_bool, width=target_w, height=target_h)

    # ---- VGGT：resize -> crop -> pad ----
    proc_w = getattr(transform, "proc_width", None)
    proc_h = getattr(transform, "proc_height", None)
    if (
        isinstance(proc_w, int)
        and isinstance(proc_h, int)
        and proc_w > 0
        and proc_h > 0
    ):
        resized = _resize_mask_nearest(mask_bool, width=proc_w, height=proc_h)

        crop_applied = bool(getattr(transform, "crop_applied", False))
        crop_start_y = int(getattr(transform, "crop_start_y", 0))
        final_h = int(getattr(transform, "final_height", resized.shape[0]))
        final_w = int(getattr(transform, "final_width", resized.shape[1]))

        if crop_applied:
            y0 = max(0, min(resized.shape[0], crop_start_y))
            y1 = max(0, min(resized.shape[0], y0 + final_h))
            cropped = resized[y0:y1, :final_w]
        else:
            cropped = resized[:final_h, :final_w]

        pad_left = int(getattr(transform, "batch_pad_left", 0))
        pad_top = int(getattr(transform, "batch_pad_top", 0))
        padded_w = int(
            getattr(transform, "padded_width", cropped.shape[1] + max(0, pad_left))
        )
        padded_h = int(
            getattr(transform, "padded_height", cropped.shape[0] + max(0, pad_top))
        )
        padded_w = max(1, padded_w)
        padded_h = max(1, padded_h)

        out = np.zeros((padded_h, padded_w), dtype=bool)
        x0 = max(0, pad_left)
        y0 = max(0, pad_top)
        x1 = min(padded_w, x0 + cropped.shape[1])
        y1 = min(padded_h, y0 + cropped.shape[0])
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = cropped[: (y1 - y0), : (x1 - x0)]
        return out

    logger.warning(
        "map_mask_to_final_space: unsupported transform; returning original mask."
    )
    return mask_bool


def _clamp_bbox_xyxy_to_image(
    bbox_xyxy: Sequence[Union[int, float]], width: int, height: int
) -> Tuple[int, int, int, int]:
    """Clamp bbox to image bounds and return integer pixel XYXY with x2>x1, y2>y1."""
    x1f, y1f, x2f, y2f = [float(v) for v in bbox_xyxy]
    x1 = int(max(0, min(width - 1, np.floor(x1f))))
    y1 = int(max(0, min(height - 1, np.floor(y1f))))
    x2 = int(max(x1 + 1, min(width, np.ceil(x2f))))
    y2 = int(max(y1 + 1, min(height, np.ceil(y2f))))
    return x1, y1, x2, y2


def _add_visual_prompt(
    datapoint: Any,
    boxes: List[List[float]],
    labels: List[bool],
    text_prompt: str = "visual",
) -> int:
    """Add a visual prompt (exemplar) query to the datapoint.

    This is the core function for visual exemplar support. The model will find
    objects that resemble the positive exemplars while avoiding the negative ones.

    Args:
        datapoint: The Datapoint object (must have image already set)
        boxes: List of bounding boxes in XYXY format (top-left and bottom-right corners)
        labels: List of boolean labels for each box (True=positive, False=negative)
        text_prompt: Optional text hint to guide segmentation (default "visual" for pure visual matching)

    Returns:
        Query ID for retrieving results from postprocessor

    Raises:
        AssertionError: If image not set, no boxes provided, or box/label count mismatch

    Note:
        - The model expects prompts to be consistent. If text reads "bottle" but the
          provided box points to a dog, results will be undefined.
        - If no positive boxes are provided and text_prompt is "visual", results are undefined.
    """
    global _BATCH_QUERY_COUNTER
    _ensure_sam3_in_path()
    from sam3.train.data.sam3_image_dataset import (  # type: ignore
        FindQueryLoaded,
        InferenceMetadata,
    )

    if len(datapoint.images) != 1:
        raise ValueError("Please set exactly one image first")
    if len(boxes) == 0:
        raise ValueError("Please provide at least one box")
    if len(boxes) != len(labels):
        raise ValueError(
            f"Expecting one label per box. Found {len(boxes)} boxes but {len(labels)} labels"
        )
    for b in boxes:
        if len(b) != 4:
            raise ValueError(f"Boxes must have 4 coordinates, found {len(b)}")

    labels_tensor = torch.tensor(labels, dtype=torch.bool).view(-1)
    if not labels_tensor.any().item() and text_prompt == "visual":
        logger.warning(
            "No positive box provided and text_prompt='visual'. "
            "The prompt is ambiguous and results may be undefined."
        )

    h_img, w_img = datapoint.images[0].size  # (height, width)
    with _BATCH_QUERY_LOCK:
        query_id = _BATCH_QUERY_COUNTER
        _BATCH_QUERY_COUNTER += 1

    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_prompt,
            image_id=0,
            object_ids_output=[],  # Unused for inference
            is_exhaustive=True,  # Unused for inference
            query_processing_order=0,
            input_bbox=torch.tensor(boxes, dtype=torch.float).view(-1, 4),
            input_bbox_label=labels_tensor,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=(h_img, w_img),
                object_id=0,
                frame_index=0,
            ),
        )
    )

    return query_id


def _add_text_prompt(datapoint: Any, text_query: str) -> int:
    """Add a text-only query to the datapoint.

    Args:
        datapoint: The Datapoint object (must have image already set)
        text_query: Text description of objects to find (e.g., "bottle", "red car")

    Returns:
        Query ID for retrieving results from postprocessor
    """
    global _BATCH_QUERY_COUNTER
    _ensure_sam3_in_path()
    from sam3.train.data.sam3_image_dataset import (  # type: ignore
        FindQueryLoaded,
        InferenceMetadata,
    )

    if len(datapoint.images) != 1:
        raise ValueError("Please set exactly one image first")

    h_img, w_img = datapoint.images[0].size  # (height, width)
    with _BATCH_QUERY_LOCK:
        query_id = _BATCH_QUERY_COUNTER
        _BATCH_QUERY_COUNTER += 1

    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=(h_img, w_img),
                object_id=0,
                frame_index=0,
            ),
        )
    )

    return query_id


def sam3_masks_from_bboxes_predict_inst(
    image_path: str,
    bboxes_xyxy: List[List[float]],
    checkpoint_path: str,
    device: str,
    text_prompt: Optional[str] = None,
    positive_exemplar: Optional[List[float]] = None,
) -> List[np.ndarray]:
    """Return one HxW boolean mask per bbox (same order).

    Uses SAM3 bbox prompt (XYXY pixels) via `predict_inst` and clips the final mask to bbox.
    Optionally uses text prompt for improved segmentation guidance.

    **Note on Visual Exemplars**:
    - Visual exemplar support requires SAM3's batch inference API (not available via predict_inst).
    - The positive_exemplar parameter is currently NOT supported and will be ignored with a warning.
    - For visual exemplar functionality, use SAM3's batch inference API directly.

    Args:
        image_path: Path to input image
        bboxes_xyxy: List of bounding boxes in [x1, y1, x2, y2] format
        checkpoint_path: Path to SAM3 checkpoint
        device: Device to run on ('cuda' or 'cpu')
        text_prompt: Optional text description to guide segmentation (e.g., 'bottle', 'red bottle')
        positive_exemplar: NOT SUPPORTED - Visual exemplars require batch inference API.
                          This parameter is kept for API compatibility but will be ignored.

    Returns:
        List of boolean masks, one per bbox

    Examples:
        >>> # Case 1: Text prompt guidance
        >>> masks = sam3_masks_from_bboxes_predict_inst(
        ...     image_path="image.jpg",
        ...     bboxes_xyxy=[[100, 200, 150, 250], [300, 400, 350, 450]],
        ...     text_prompt="bottle",
        ...     ...
        ... )
        >>> # Returns 2 masks, guided by text prompt "bottle"

        >>> # Case 2: Geometry only (no text prompt)
        >>> masks = sam3_masks_from_bboxes_predict_inst(
        ...     image_path="image.jpg",
        ...     bboxes_xyxy=[[100, 200, 150, 250], [300, 400, 350, 450]],
        ...     ...
        ... )
        >>> # Returns 2 masks, guided only by geometry (bboxes)
    """
    from PIL import Image

    if len(bboxes_xyxy) == 0:
        return []

    expected_checkpoint_sha256 = checkpoint_sha256(Path(checkpoint_path))
    model, processor = _get_sam3_model_and_processor(
        checkpoint_path=checkpoint_path,
        device=device,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    with _sam3_autocast_context(model, device):
        state = processor.set_image(image)

        # Warn if visual exemplar is provided (not supported by predict_inst API)
        if positive_exemplar:
            logger.warning(
                "positive_exemplar parameter provided but is NOT supported by SAM3's predict_inst API. "
                "Visual exemplars require batch inference API (see sam3_image_batched_inference.ipynb). "
                "Ignoring exemplar and falling back to text+geometry prompts only."
            )

        # Set text prompt if provided
        if text_prompt:
            logger.info(f"Using text prompt: '{text_prompt}'")
            state = processor.set_text_prompt(text_prompt, state)
        else:
            logger.info(f"Using geometry prompts only for {len(bboxes_xyxy)} bbox(es)")

    # Use geometry prompts (bboxes) - this is the only mode supported by predict_inst
    boxes = np.asarray(bboxes_xyxy, dtype=np.float32)

    # Prefer multimask; pick the best by IoU. Box-only prompting can be ambiguous.
    def _ensure_bchw(m: np.ndarray) -> np.ndarray:
        # (C,H,W) -> (1,C,H,W), (B,C,H,W) -> as-is
        if m.ndim == 3:
            return m[None, ...]
        if m.ndim == 4:
            return m
        raise RuntimeError(f"Unexpected SAM3 mask output shape: {m.shape}")

    # Run SAM3 inference with geometry (+ optional text) prompts
    with _sam3_autocast_context(model, device):
        masks_np, ious_np, _ = model.predict_inst(  # type: ignore[attr-defined]
            state,
            box=boxes,
            multimask_output=True,
            normalize_coords=True,
            return_logits=False,
        )

    masks_bchw = _ensure_bchw(masks_np)
    ious_b = np.asarray(ious_np)
    if ious_b.ndim == 1:
        ious_b = ious_b[None, ...]

    # Verify output shape matches target bboxes
    if masks_bchw.shape[0] != len(bboxes_xyxy):
        raise RuntimeError(
            f"SAM3 returned {masks_bchw.shape[0]} masks for {len(bboxes_xyxy)} boxes"
        )

    # Pick best mask per bbox (C dimension).
    best_masks: List[np.ndarray] = []
    for bbox_idx, bbox in enumerate(bboxes_xyxy):
        cur_masks_chw = masks_bchw[bbox_idx]
        if cur_masks_chw.shape[0] == 0:
            best_masks.append(np.zeros((h, w), dtype=bool))
            continue
        cur_ious = ious_b[bbox_idx] if bbox_idx < ious_b.shape[0] else None
        if cur_ious is not None and cur_ious.shape[0] == cur_masks_chw.shape[0]:
            best_c = int(np.argmax(cur_ious))
        else:
            best_c = 0
        best_masks.append(cur_masks_chw[best_c].astype(bool))

    # Fallback: if a bbox yields an empty mask, add a positive center point and retry for that bbox.
    out: List[np.ndarray] = []
    for bbox, mask in zip(bboxes_xyxy, best_masks):
        raw_true = int(mask.sum())
        mask = clip_mask_to_bbox(mask, bbox)
        clipped_true = int(mask.sum())
        if raw_true > 0 and clipped_true == 0:
            logger.warning(
                "SAM3 produced a non-empty mask but it vanished after bbox clipping; likely coord mismatch. "
                "raw_true=%d bbox=%s image_wh=%sx%s",
                raw_true,
                bbox,
                w,
                h,
            )
        if mask.any():
            out.append(mask)
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        try:
            with _sam3_autocast_context(model, device):
                masks2, ious2, _ = model.predict_inst(  # type: ignore[attr-defined]
                    state,
                    box=np.asarray([bbox], dtype=np.float32),
                    point_coords=np.asarray([[cx, cy]], dtype=np.float32),
                    point_labels=np.asarray([1], dtype=np.int32),
                    multimask_output=True,
                    normalize_coords=True,
                    return_logits=False,
                )
            masks2_bchw = _ensure_bchw(masks2)
            ious2_b = np.asarray(ious2)
            if ious2_b.ndim == 1:
                ious2_b = ious2_b[None, ...]
            cur_masks_chw = masks2_bchw[0]
            cur_ious = ious2_b[0] if ious2_b.shape[0] > 0 else None
            best_c = int(np.argmax(cur_ious)) if cur_ious is not None else 0
            mask2 = cur_masks_chw[best_c].astype(bool)
            raw2_true = int(mask2.sum())
            mask2c = clip_mask_to_bbox(mask2, bbox)
            clipped2_true = int(mask2c.sum())
            if raw2_true > 0 and clipped2_true == 0:
                logger.warning(
                    "SAM3 center-point fallback produced non-empty mask but it vanished after bbox clipping; "
                    "raw_true=%d bbox=%s image_wh=%sx%s",
                    raw2_true,
                    bbox,
                    w,
                    h,
                )
            out.append(mask2c)
        except Exception as e:
            logger.error(
                "SAM3 mask fallback (center-point) failed for bbox=%s: %s", bbox, e
            )
            raise RuntimeError(f"SAM3 failed to generate mask for bbox {bbox}") from e

    return out


# ============================================================================
# Batch Inference API - Full Visual Exemplar Support
# ============================================================================


def sam3_masks_from_bboxes_batch_api(
    image_path: str,
    bboxes_xyxy: List[List[float]],
    checkpoint_path: str,
    device: str,
    text_prompt: Optional[str] = None,
    positive_exemplars: Optional[List[List[float]]] = None,
    negative_exemplars: Optional[List[List[float]]] = None,
    detection_threshold: float = 0.3,
) -> List[np.ndarray]:
    """Return HxW boolean masks using SAM3's batch inference API with visual exemplar support.

    This function uses SAM3's full batch inference API which supports:
    - Text prompts
    - Visual exemplars (positive and negative reference boxes)

    The model finds objects that resemble the positive exemplars while avoiding
    objects that look like negative exemplars.

    Args:
        image_path: Path to input image
        bboxes_xyxy: List of target bounding boxes in [x1, y1, x2, y2] format.
                     These are the boxes to generate masks for.
        checkpoint_path: Path to SAM3 checkpoint
        device: Device to run on ('cuda' or 'cpu')
        text_prompt: Optional text description to guide segmentation (e.g., 'bottle').
                    If None and no exemplars provided, defaults to "visual".
        positive_exemplars: List of positive exemplar bboxes in XYXY format.
                           The model will find objects similar to these.
        negative_exemplars: List of negative exemplar bboxes in XYXY format.
                           The model will avoid objects similar to these.
        detection_threshold: Confidence threshold for detections (default 0.3)

    Returns:
        List of boolean masks. May return more or fewer masks than input bboxes
        depending on what the model detects with the given prompts.

    Examples:
        >>> # Case 1: Visual exemplar only
        >>> masks = sam3_masks_from_bboxes_batch_api(
        ...     image_path="image.jpg",
        ...     bboxes_xyxy=[],  # Empty - let model find all similar objects
        ...     positive_exemplars=[[100, 200, 150, 250]],  # One reference bottle
        ...     checkpoint_path="sam3.pt",
        ...     device="cuda",
        ... )
        >>> # Returns masks for all objects similar to the positive exemplar

        >>> # Case 2: Visual exemplar + text prompt
        >>> masks = sam3_masks_from_bboxes_batch_api(
        ...     image_path="image.jpg",
        ...     bboxes_xyxy=[],
        ...     positive_exemplars=[[100, 200, 150, 250]],
        ...     text_prompt="bottle",  # Reinforce that we want bottles
        ...     checkpoint_path="sam3.pt",
        ...     device="cuda",
        ... )

        >>> # Case 3: Negative exemplar to exclude certain objects
        >>> masks = sam3_masks_from_bboxes_batch_api(
        ...     image_path="image.jpg",
        ...     bboxes_xyxy=[],
        ...     positive_exemplars=[[100, 200, 150, 250]],  # Bottles we want
        ...     negative_exemplars=[[300, 400, 350, 450]],  # Box to exclude
        ...     text_prompt="bottle",
        ...     checkpoint_path="sam3.pt",
        ...     device="cuda",
        ... )

    Note:
        - This function uses SAM3's batch inference API which is more powerful
          but slower than predict_inst.
        - If no positive_exemplars are provided and text_prompt is None,
          the function falls back to using target bboxes as positive exemplars.
    """
    from PIL import Image

    # Load batch inference components
    model, transform, postprocessor = _get_sam3_batch_components(
        checkpoint_path=checkpoint_path,
        device=device,
        detection_threshold=detection_threshold,
    )

    # Import batch inference utilities
    _ensure_sam3_in_path()
    from sam3.train.data.collator import collate_fn_api as collate  # type: ignore
    from sam3.model.utils.misc import copy_data_to_device  # type: ignore

    # Load and prepare image
    pil_image = Image.open(image_path).convert("RGB")
    w, h = pil_image.size

    # Create datapoint and set image
    datapoint = _create_empty_datapoint()
    _set_image(datapoint, pil_image)

    # Build exemplar boxes and labels
    exemplar_boxes: List[List[float]] = []
    exemplar_labels: List[bool] = []

    # Add positive exemplars
    if positive_exemplars:
        for box in positive_exemplars:
            exemplar_boxes.append(box)
            exemplar_labels.append(True)

    # Add negative exemplars
    if negative_exemplars:
        for box in negative_exemplars:
            exemplar_boxes.append(box)
            exemplar_labels.append(False)

    # If no exemplars provided, use target bboxes as positive exemplars
    if not exemplar_boxes and bboxes_xyxy:
        logger.info("No exemplars provided, using target bboxes as positive exemplars")
        for box in bboxes_xyxy:
            exemplar_boxes.append(box)
            exemplar_labels.append(True)

    # Determine text prompt
    effective_text_prompt = text_prompt if text_prompt else "visual"

    # Add query
    if exemplar_boxes:
        logger.info(
            f"Using batch API with {sum(exemplar_labels)} positive and "
            f"{len(exemplar_labels) - sum(exemplar_labels)} negative exemplars, "
            f"text_prompt='{effective_text_prompt}'"
        )
        query_id = _add_visual_prompt(
            datapoint,
            boxes=exemplar_boxes,
            labels=exemplar_labels,
            text_prompt=effective_text_prompt,
        )
    else:
        # No boxes at all - use text-only query
        if text_prompt:
            logger.info(f"Using batch API with text-only prompt: '{text_prompt}'")
            query_id = _add_text_prompt(datapoint, text_prompt)
        else:
            logger.warning(
                "No exemplars and no text prompt provided, results may be empty"
            )
            return []

    # Apply transforms
    datapoint = transform(datapoint)

    # Collate and move to device
    batch = collate([datapoint], dict_key="sam3")["sam3"]
    batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)
    batch = _coerce_sam3_batch_dtype(batch, model)

    # Run inference
    with torch.inference_mode():
        # Enable optimizations for CUDA
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        autocast_ctx = _sam3_autocast_context(model, device)
        with autocast_ctx:
            output = model(batch)

    # Postprocess results
    processed_results = postprocessor.process_results(output, batch.find_metadatas)

    # Extract masks from results
    if query_id not in processed_results:
        logger.warning(
            f"Query ID {query_id} not found in results, returning empty masks"
        )
        return []

    result = processed_results[query_id]
    masks_tensor = result.get("masks")

    if masks_tensor is None or len(masks_tensor) == 0:
        logger.warning("No masks returned from batch inference")
        return []

    # Convert to numpy boolean masks
    # masks_tensor shape: [N, 1, H, W]
    if isinstance(masks_tensor, torch.Tensor):
        masks_np = masks_tensor.squeeze(1).cpu().numpy().astype(bool)
    else:
        # Handle list of tensors
        masks_np = np.stack(
            [m.squeeze().cpu().numpy().astype(bool) for m in masks_tensor], axis=0
        )

    # Convert to list of masks
    out_masks: List[np.ndarray] = []
    for i in range(masks_np.shape[0]):
        mask = masks_np[i]
        # Clip mask to image size (should already be correct but verify)
        if mask.shape != (h, w):
            logger.warning(f"Mask shape {mask.shape} doesn't match image size {(h, w)}")
            mask = _resize_mask_nearest(mask, width=w, height=h)
        out_masks.append(mask)

    logger.info(f"Batch inference returned {len(out_masks)} masks")
    return out_masks


def sam3_masks_from_single_exemplar(
    image_path: str,
    target_bboxes_xyxy: List[List[float]],
    exemplar_bbox_xyxy: List[float],
    checkpoint_path: str,
    device: str,
    text_prompt: Optional[str] = None,
    detection_threshold: float = 0.3,
) -> List[np.ndarray]:
    """Convenience function: Find objects similar to a single exemplar.

    This is a simplified wrapper around sam3_masks_from_bboxes_batch_api for the
    common case of having one positive exemplar.

    Args:
        image_path: Path to input image
        target_bboxes_xyxy: Target bounding boxes (for reference, may be empty)
        exemplar_bbox_xyxy: Single positive exemplar bbox in XYXY format
        checkpoint_path: Path to SAM3 checkpoint
        device: Device to run on ('cuda' or 'cpu')
        text_prompt: Optional text description (e.g., 'bottle')
        detection_threshold: Confidence threshold for detections

    Returns:
        List of boolean masks for detected objects similar to the exemplar

    Example:
        >>> # Use first detected object as exemplar to find all similar objects
        >>> masks = sam3_masks_from_single_exemplar(
        ...     image_path="image.jpg",
        ...     target_bboxes_xyxy=[],
        ...     exemplar_bbox_xyxy=[100, 200, 150, 250],  # Reference object
        ...     text_prompt="bottle",
        ...     checkpoint_path="sam3.pt",
        ...     device="cuda",
        ... )
    """
    return sam3_masks_from_bboxes_batch_api(
        image_path=image_path,
        bboxes_xyxy=target_bboxes_xyxy,
        checkpoint_path=checkpoint_path,
        device=device,
        text_prompt=text_prompt,
        positive_exemplars=[exemplar_bbox_xyxy],
        negative_exemplars=None,
        detection_threshold=detection_threshold,
    )


def sample_points_from_mask(
    mask: np.ndarray,
    max_points: int,
    enable_gaussian: bool = False,
    gaussian_sigma: float = 0.3,
    gaussian_truncate: float = 3.0,
    bbox_xyxy: Optional[List[float]] = None,
) -> np.ndarray:
    """Sample up to `max_points` 2D pixels from a binary mask (returns Nx2 xy).

    支持两种采样模式：
    1. 均匀采样（默认）：mask内所有点等概率采样
    2. 高斯加权采样：以mask质心或bbox中心为高斯中心，中心密集向外递减

    Args:
        mask: 二值mask (H, W)
        max_points: 最大采样点数
        enable_gaussian: 是否启用高斯加权采样
        gaussian_sigma: 高斯标准差（相对于mask半径）
        gaussian_truncate: 高斯截断倍数
        bbox_xyxy: 可选的bbox坐标 [x1, y1, x2, y2]，用于计算高斯中心

    Returns:
        采样点数组 (N, 2)，格式为 [x, y]
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    k = int(min(max_points, len(xs)))
    if k <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    if not enable_gaussian:
        # 均匀采样
        sel = np.random.choice(len(xs), size=k, replace=False)
        pts = np.stack([xs[sel], ys[sel]], axis=-1).astype(np.float32)
        return pts

    # === 高斯加权采样 ===
    # 计算高斯中心：优先使用bbox中心，否则使用mask质心
    if bbox_xyxy is not None:
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
        rx = (bbox_xyxy[2] - bbox_xyxy[0]) / 2
        ry = (bbox_xyxy[3] - bbox_xyxy[1]) / 2
    else:
        # 使用mask质心和边界
        cx = np.mean(xs)
        cy = np.mean(ys)
        rx = max((np.max(xs) - np.min(xs)) / 2, 1.0)
        ry = max((np.max(ys) - np.min(ys)) / 2, 1.0)

    # 计算每个mask点到中心的归一化距离
    dx_norm = (xs - cx) / (rx + 1e-6)
    dy_norm = (ys - cy) / (ry + 1e-6)
    distances = np.sqrt(dx_norm**2 + dy_norm**2)

    # 计算高斯权重
    weights = np.exp(-(distances**2) / (2 * gaussian_sigma**2))

    # 截断距离过大的点
    truncate_distance = gaussian_sigma * gaussian_truncate
    weights[distances > truncate_distance] = 0

    # 只在非零权重内采样，避免频繁回退
    nonzero_idx = np.flatnonzero(weights > 0)
    if nonzero_idx.size == 0:
        sel = np.random.choice(len(xs), size=k, replace=False)
        pts = np.stack([xs[sel], ys[sel]], axis=-1).astype(np.float32)
        return pts

    xs_nz = xs[nonzero_idx]
    ys_nz = ys[nonzero_idx]
    w_nz = weights[nonzero_idx]
    weight_sum = w_nz.sum()
    if weight_sum < 1e-10:
        sel = np.random.choice(len(xs_nz), size=min(k, len(xs_nz)), replace=False)
        pts = np.stack([xs_nz[sel], ys_nz[sel]], axis=-1).astype(np.float32)
        return pts

    probabilities = w_nz / weight_sum

    replace = len(xs_nz) < k
    sel = np.random.choice(len(xs_nz), size=k, replace=replace, p=probabilities)
    pts = np.stack([xs_nz[sel], ys_nz[sel]], axis=-1).astype(np.float32)
    return pts


def map_points_to_final_space(
    transform: ImageTransformBase, points_xy: np.ndarray
) -> np.ndarray:
    fn = getattr(transform, "map_points_to_final", None)
    if callable(fn):
        return fn(points_xy).astype(np.float32)
    mapped = []
    for x, y in points_xy:
        xf, yf = transform.map_xy_to_final(float(x), float(y))
        mapped.append((xf, yf))
    return np.asarray(mapped, dtype=np.float32)


def sample_3d_points_from_mask(
    scene_data: Dict,
    img_idx: int,
    mask: np.ndarray,
    transform: ImageTransformBase,
    config: SKUMatchingConfig,
    mask_space: Literal["original", "final"] = "original",
    bbox_xyxy: Optional[List[float]] = None,
) -> Optional[torch.Tensor]:
    """Sample 3D points by sampling 2D mask pixels then indexing scene_data.

    支持高斯加权采样：当 config.enable_gaussian_sampling=True 时，
    在 mask 内部应用高斯权重，使采样点集中在物体中心区域。

    Args:
        scene_data: VGGT场景数据
        img_idx: 图像索引
        mask: 二值mask
        transform: 图像变换
        config: 配置参数
        mask_space: mask坐标空间 ("original" 或 "final")
        bbox_xyxy: 可选的bbox坐标，用于高斯采样的中心计算
    """
    device = scene_data["depth"].device
    _, H, W, _ = scene_data["depth"].shape

    # 使用高斯加权采样（如果启用）
    candidates_2d = sample_points_from_mask(
        mask,
        max_points=int(config.max_3d_points_per_bbox) * 10,
        enable_gaussian=config.enable_gaussian_sampling,
        gaussian_sigma=config.gaussian_sigma,
        gaussian_truncate=config.gaussian_truncate,
        bbox_xyxy=bbox_xyxy,
    )
    if candidates_2d.shape[0] == 0:
        return None

    if mask_space == "final":
        pts_final = candidates_2d
    else:
        pts_final = map_points_to_final_space(transform, candidates_2d)
    xs = torch.from_numpy(np.round(pts_final[:, 0]).astype(np.int64)).to(device=device)
    ys = torch.from_numpy(np.round(pts_final[:, 1]).astype(np.int64)).to(device=device)
    xs = xs.clamp(0, W - 1)
    ys = ys.clamp(0, H - 1)

    depth = scene_data["depth"][img_idx, ys, xs, 0]
    depth_conf = scene_data["depth_conf"][img_idx, ys, xs]
    wp = scene_data["world_points"][img_idx, ys, xs]
    wp_conf = scene_data["world_points_conf"][img_idx, ys, xs]

    valid = (
        (depth_conf > config.depth_confidence_threshold)
        & (wp_conf > config.point_3d_confidence_threshold)
        & (depth > config.min_depth)
        & (depth < config.max_depth)
        & torch.isfinite(depth)
        & torch.isfinite(wp).all(dim=-1)
    )
    if valid.sum().item() < 10:
        return None

    wp = wp[valid]
    if wp.shape[0] > int(config.max_3d_points_per_bbox):
        idx = torch.randperm(wp.shape[0], device=wp.device)[
            : int(config.max_3d_points_per_bbox)
        ]
        wp = wp[idx]
    return wp


def sam3_masks_self_exemplar(
    image_path: Union[str, Path, Any],
    bboxes_xyxy: List[List[float]],
    checkpoint_path: str,
    device: str,
    detection_threshold: float = 0.3,
    max_batch_size: int = 5,  # 新增：每批最多处理的bbox数量
    *,
    fallback_size: int = 1008,
    skip_resize: bool = False,
) -> List[np.ndarray]:
    """为每个bbox生成self-exemplar分割mask（支持分批处理）。

    每个bbox作为自己的positive exemplar，使用batch inference API获取精确的self-segmentation。
    这比普通的bbox prompt分割更精确，因为使用了visual exemplar来指导分割。

    工作原理：
    - 对于每个bbox，将其作为positive_exemplar
    - 如果bbox数量超过max_batch_size，会自动分批处理以避免GPU OOM
    - 使用batch API进行inference
    - 返回的mask中选择与该bbox IoU最高的那个作为self-segmentation结果

    Args:
        image_path: 输入图像路径
        bboxes_xyxy: 待分割的bboxes列表，每个bbox格式为[x1, y1, x2, y2]
        checkpoint_path: SAM3 checkpoint路径
        device: 设备 ('cuda' or 'cpu')
        detection_threshold: 检测阈值 (default 0.3)
        max_batch_size: 每批最多处理的bbox数量，避免GPU OOM (default 30)

    Returns:
        List of boolean masks，每个mask对应一个输入bbox的self-segmentation结果

    Example:
        >>> # 为检测到的所有物体生成精确的self-segmentation masks
        >>> masks = sam3_masks_self_exemplar(
        ...     image_path="image.jpg",
        ...     bboxes_xyxy=[[100, 200, 150, 250], [300, 400, 350, 450]],
        ...     checkpoint_path="sam3.pt",
        ...     device="cuda",
        ...     max_batch_size=30,
        ... )
        >>> # masks[0] 是第一个bbox的self-segmentation结果
        >>> # masks[1] 是第二个bbox的self-segmentation结果
    """
    from PIL import Image

    if len(bboxes_xyxy) == 0:
        return []

    # 清空CUDA缓存，为SAM3腾出显存空间
    # 这对于已经加载了其他大模型（如VGGT）的情况特别重要
    if device.startswith("cuda") and torch.cuda.is_available():
        # NOTE(Cycle6): 移除 PI3_SCENE_CACHE.clear() —— scene_data 是只读 npz cache，
        # 跨 ref 复用可省去重复 scene_data_build（11次->1次）。
        # OOM 监控由 smoke 验证；若显存累积则恢复 clear。

        # 强制垃圾回收和清空缓存
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # 等待所有CUDA操作完成

        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        mem_free = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated()
        ) / 1024**3
        logger.info(
            f"GPU memory before SAM3: allocated={mem_allocated:.2f}GB, reserved={mem_reserved:.2f}GB, free={mem_free:.2f}GB"
        )

    # Load batch inference components
    model, transform, postprocessor = _get_sam3_batch_components(
        checkpoint_path=checkpoint_path,
        device=device,
        detection_threshold=detection_threshold,
        max_dets_per_img=1,  # Self-exemplar模式：每个query只保留置信度最高的1个mask
        fallback_size=fallback_size,
        skip_resize=skip_resize,
    )

    # Import batch inference utilities
    _ensure_sam3_in_path()
    from sam3.train.data.collator import collate_fn_api as collate  # type: ignore
    from sam3.model.utils.misc import copy_data_to_device  # type: ignore

    # Load and prepare image (只加载一次)
    if isinstance(image_path, (str, Path)):
        pil_image = Image.open(str(image_path)).convert("RGB")
    elif isinstance(image_path, Image.Image):
        pil_image = image_path.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_path)}")
    w, h = pil_image.size

    # 分批处理：将bboxes分成多个小批次
    num_bboxes = len(bboxes_xyxy)
    num_batches = (num_bboxes + max_batch_size - 1) // max_batch_size  # 向上取整

    if num_batches > 1:
        logger.info(
            f"Processing {num_bboxes} bboxes in {num_batches} batches (max_batch_size={max_batch_size})"
        )
    else:
        logger.info(f"Processing {num_bboxes} bboxes in single batch")

    # 存储所有批次的结果
    all_result_masks: List[np.ndarray] = []

    # 逐批处理
    for batch_idx in range(num_batches):
        start_idx = batch_idx * max_batch_size
        end_idx = min((batch_idx + 1) * max_batch_size, num_bboxes)
        batch_bboxes = bboxes_xyxy[start_idx:end_idx]

        logger.info(
            f"Batch {batch_idx + 1}/{num_batches}: processing bboxes [{start_idx}:{end_idx}] ({len(batch_bboxes)} bboxes)"
        )

        # 为当前批次创建datapoint
        datapoint = _create_empty_datapoint()
        _set_image(datapoint, pil_image)

        # 为当前批次的bbox添加query
        query_id_to_local_idx: Dict[int, int] = {}
        for local_idx, bbox in enumerate(batch_bboxes):
            query_id = _add_visual_prompt(
                datapoint,
                boxes=[bbox],
                labels=[True],  # positive exemplar
                text_prompt="visual",  # pure visual matching
            )
            query_id_to_local_idx[query_id] = local_idx

        # Apply transforms
        datapoint = transform(datapoint)

        # Collate and move to device
        batch = collate([datapoint], dict_key="sam3")["sam3"]
        batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)
        batch = _coerce_sam3_batch_dtype(batch, model)

        # Run inference
        with torch.inference_mode():
            if device.startswith("cuda"):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            autocast_ctx = _sam3_autocast_context(model, device)
            with autocast_ctx:
                output = model(batch)

        # Postprocess results
        processed_results = postprocessor.process_results(output, batch.find_metadatas)

        # 提取当前批次的结果
        batch_result_masks: List[np.ndarray] = [
            np.zeros((h, w), dtype=bool) for _ in batch_bboxes
        ]

        for query_id, local_idx in query_id_to_local_idx.items():
            bbox = batch_bboxes[local_idx]

            if query_id not in processed_results:
                logger.warning(
                    f"Query ID {query_id} not found for batch {batch_idx}, bbox {local_idx}"
                )
                continue

            result = processed_results[query_id]
            masks_tensor = result.get("masks")

            if masks_tensor is None or len(masks_tensor) == 0:
                logger.warning(
                    f"No masks returned for batch {batch_idx}, bbox {local_idx}"
                )
                continue

            # Convert masks to numpy (max_dets_per_img=1 确保只返回1个最佳mask)
            if isinstance(masks_tensor, torch.Tensor):
                masks_np = masks_tensor.squeeze(1).cpu().numpy().astype(bool)
            else:
                masks_np = np.stack(
                    [m.squeeze().cpu().numpy().astype(bool) for m in masks_tensor],
                    axis=0,
                )

            # 取第一个mask (postprocessor已经按score排序，只保留Top-1)
            best_mask = (
                masks_np[0] if len(masks_np) > 0 else np.zeros((h, w), dtype=bool)
            )

            # Ensure mask is correct size
            if best_mask.shape != (h, w):
                best_mask = _resize_mask_nearest(best_mask, width=w, height=h)

            # Defensive: clip to bbox to avoid leaking pixels from nearby objects
            x1, y1, x2, y2 = _clamp_bbox_xyxy_to_image(bbox, width=w, height=h)
            best_mask = clip_mask_to_bbox(best_mask, (x1, y1, x2, y2))
            batch_result_masks[local_idx] = best_mask

        # 将当前批次的结果添加到总结果中
        all_result_masks.extend(batch_result_masks)

        # 批次间清理GPU缓存（仅清理临时tensors，不清理模型）
        if (
            device.startswith("cuda")
            and torch.cuda.is_available()
            and batch_idx < num_batches - 1
        ):
            import gc

            # 显式删除所有批次临时变量（从大到小的顺序）
            # 1. processed_results 包含GPU tensor，必须先删除
            del processed_results
            # 2. batch 是输入的GPU tensor
            del batch
            # 3. output 是模型输出的GPU tensor
            del output
            # 4. datapoint 包含transform后的数据
            del datapoint
            # 5. batch_result_masks 包含numpy数组（虽然在CPU但也占内存）
            del batch_result_masks
            # 6. 其他辅助变量
            del query_id_to_local_idx, batch_bboxes

            # 强制垃圾回收
            gc.collect()
            # empty_cache只清理未被引用的tensors，模型权重仍保留在GPU
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # 等待所有CUDA操作完成

            # 记录清理后的显存状态
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_free = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            ) / 1024**3
            logger.debug(
                f"Cleared GPU cache after batch {batch_idx + 1}: allocated={mem_allocated:.2f}GB, free={mem_free:.2f}GB"
            )

    logger.info(
        f"Self-exemplar segmentation complete: {len(all_result_masks)} masks for {num_bboxes} bboxes ({num_batches} batches)"
    )

    # SAM3推理完成后，立即清理GPU显存
    if device.startswith("cuda") and torch.cuda.is_available():
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared after SAM3 inference")

    return all_result_masks


@contextmanager
def _preserve_cache_boundary_rng(device: str) -> Iterator[None]:
    """Keep cache lookup and miss inference invisible to downstream sampling RNG."""
    with _CACHE_BOUNDARY_RNG_LOCK:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_cpu_state = torch.random.get_rng_state()
        cuda_state = None
        cuda_device = None
        if device.startswith("cuda") and torch.cuda.is_available():
            cuda_device = torch.device(device)
            cuda_state = torch.cuda.get_rng_state(cuda_device)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_cpu_state)
            if cuda_state is not None and cuda_device is not None:
                torch.cuda.set_rng_state(cuda_state, cuda_device)


def _processed_frame_request(
    *,
    cache_root: Path,
    image_path: Path,
    image_id: int,
    frame_detections: Sequence[dict[str, object]],
    transform: ImageTransformBase,
) -> FrameMaskCacheRequest:
    """Build the exact DA3 processed-space request in source detection order."""
    if isinstance(image_id, bool) or not isinstance(image_id, Integral):
        raise ValueError("image_id must be an integer")
    try:
        source_width = int(getattr(transform, "orig_width"))
        source_height = int(getattr(transform, "orig_height"))
        processed_width = int(getattr(transform, "target_width"))
        processed_height = int(getattr(transform, "target_height"))
        scale_x = float(getattr(transform, "scale_x"))
        scale_y = float(getattr(transform, "scale_y"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("SAM3 cache requires a DA3 resize transform") from exc
    if min(source_width, source_height, processed_width, processed_height) <= 0:
        raise ValueError("SAM3 cache transform dimensions must be positive")

    from PIL import Image

    with Image.open(image_path) as source_image:
        if source_image.size != (source_width, source_height):
            raise ValueError("SAM3 cache transform source size does not match image")

    affine = getattr(transform, "source_to_processed_affine", None)
    if affine is None:
        affine = np.asarray(
            [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0]], dtype=np.float64
        )
    else:
        affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        raise ValueError("SAM3 cache transform affine must be finite with shape (2, 3)")

    prompts: list[ProcessedDetectionPrompt] = []
    for object_id, detection in enumerate(frame_detections):
        position = detection.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 4:
            raise ValueError(f"frame detection {object_id} has no valid position")
        source_bbox = tuple(float(value) for value in position)
        processed_bbox = map_source_bbox_to_processed(source_bbox, affine)
        prompts.append(
            ProcessedDetectionPrompt(
                object_id=object_id,
                source_bbox_xyxy=source_bbox,
                processed_bbox_xyxy=processed_bbox,
            )
        )

    return FrameMaskCacheRequest(
        cache_root=cache_root,
        image_id=int(image_id),
        image_path=image_path,
        source_size_wh=(source_width, source_height),
        processed_shape_hw=(processed_height, processed_width),
        source_to_processed_affine=affine,
        detections=tuple(prompts),
        inference_contract={
            "api": "self_exemplar",
            "threshold": 0.5,
            "image_size": 1008,
            "max_batch_size": 32,
            "max_dets_per_query": 1,
            "clip_to_bbox": True,
        },
    )


def get_self_exemplar_masks_for_reference(
    config: SKUMatchingConfig,
    *,
    image_path: Path,
    image_id: int,
    frame_detections: Sequence[dict[str, object]],
    matching_object_ids: Sequence[int],
    transform: ImageTransformBase,
) -> dict[int, np.ndarray]:
    """Load or publish a complete processed-space SAM3 frame for matching."""
    if not config.enable_sam3_mask_sampling:
        return {}
    if not config.sam3_checkpoint_path:
        raise ValueError("SAM3 enabled but sam3_checkpoint_path is empty.")
    cache_root = Path(config.sam3_mask_cache_root)
    if cache_root.name != "v2":
        raise ValueError("sam3_mask_cache_root must name the v2 cache root")

    request = _processed_frame_request(
        cache_root=cache_root,
        image_path=Path(image_path),
        image_id=image_id,
        frame_detections=frame_detections,
        transform=transform,
    )
    all_ids = [prompt.object_id for prompt in request.detections]
    matching_ids = [int(object_id) for object_id in matching_object_ids]
    if len(set(matching_ids)) != len(matching_ids) or any(
        object_id not in set(all_ids) for object_id in matching_ids
    ):
        raise ValueError("matching_object_ids must be unique frame object IDs")
    matching_id_set = set(matching_ids)
    ordered_ids = matching_ids + [
        object_id for object_id in all_ids if object_id not in matching_id_set
    ]
    prompts_by_id = {prompt.object_id: prompt for prompt in request.detections}
    device = normalize_device(config.device)

    def compute_masks() -> dict[int, np.ndarray]:
        if not ordered_ids:
            return {}

        from PIL import Image

        with Image.open(request.image_path) as source_image:
            processed_image, aligned = _prepare_sam3_input_image(
                source_image.convert("RGB"), transform
            )
        if not aligned or processed_image.size != (
            request.processed_shape_hw[1],
            request.processed_shape_hw[0],
        ):
            raise ValueError("SAM3 cache requires DA3 processed-space image alignment")
        ordered_bboxes = [
            list(prompts_by_id[object_id].processed_bbox_xyxy)
            for object_id in ordered_ids
        ]
        masks = sam3_masks_self_exemplar(
            image_path=processed_image,
            bboxes_xyxy=ordered_bboxes,
            checkpoint_path=str(config.sam3_checkpoint_path),
            device=device,
            detection_threshold=0.5,
            fallback_size=1008,
            skip_resize=False,
            max_batch_size=32,
        )
        if len(masks) != len(ordered_ids):
            raise RuntimeError("SAM3 self-exemplar returned an incomplete frame")
        return {
            object_id: np.asarray(mask, dtype=bool)
            for object_id, mask in zip(ordered_ids, masks)
        }

    with _preserve_cache_boundary_rng(device):
        result = load_or_compute_frame_masks(request, compute_masks)
    return {
        int(object_id): np.asarray(mask, dtype=bool)
        for object_id, mask in result.masks_by_object_id.items()
    }
