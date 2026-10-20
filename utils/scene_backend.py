"""Backend capabilities for the metric frame grids used by Viewer and footprint.

Model inference stays in its own adapter. Consumers use the same world-point,
camera and source-pixel contract while retaining the actual backend identity.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SceneBackend:
    name: str
    label: str
    preprocess_method: str


SCENE_BACKENDS = {
    "da3": SceneBackend("da3", "DA3", "upper_bound_resize"),
    "promptda": SceneBackend("promptda", "PromptDA", "promptda_resize"),
}


def get_scene_backend(backend: str) -> SceneBackend:
    """Require an explicitly supported metric-grid adapter; never relabel it."""
    if not isinstance(backend, str) or backend.lower() not in SCENE_BACKENDS:
        raise ValueError(f"Unsupported metric scene backend: {backend}")
    return SCENE_BACKENDS[backend.lower()]


def validate_scene_cache_metadata(cache, backend: str, *, error=ValueError) -> None:
    """Validate backend identity and units without loading dense grid arrays.

    Geometry consumers validate the array shapes they actually use separately.
    PromptDA's validity field must not be interpreted as a model confidence head.
    """
    spec = get_scene_backend(backend)
    required = {"cache_schema_version", "is_metric", "preprocess_method"}
    if spec.name == "promptda":
        required |= {"confidence_kind", "preprocess_resolution", "source_model"}
    missing = required - set(cache)
    if missing:
        raise error(f"{spec.label} scene cache missing metadata: {sorted(missing)}")
    for field, expected in (("cache_schema_version", 3), ("is_metric", 1)):
        value = cache[field]
        if value.shape != () or value.dtype.kind not in "iu" or value.item() != expected:
            raise error(f"{spec.label} scene cache {field} must be {expected}")
    method = cache["preprocess_method"]
    methods = ("promptda_resize", "promptda_native_pad14") if spec.name == "promptda" else (spec.preprocess_method,)
    if method.shape != () or method.dtype.kind != "U" or method.item() not in methods:
        raise error(f"{spec.label} scene cache preprocess_method must be one of {methods}")
    if spec.name == "promptda":
        kind, resolution, model = (cache[key] for key in (
            "confidence_kind", "preprocess_resolution", "source_model",
        ))
        if kind.shape != () or kind.dtype.kind != "U" or kind.item() != "validity":
            raise error("PromptDA confidence_kind must be validity; no learned confidence is available")
        if resolution.shape != () or resolution.dtype.kind not in "iu" or resolution.item() <= 0:
            raise error("PromptDA scene cache preprocess_resolution must be a positive integer")
        if method.item() == "promptda_resize" and resolution.item() not in {504, 1008}:
            raise error("PromptDA resize scene cache preprocess_resolution must be 504 or 1008")
        if method.item() == "promptda_native_pad14":
            if "source_image_sizes" not in cache:
                raise error("PromptDA native scene cache requires source_image_sizes")
            sizes = cache["source_image_sizes"]
            if sizes.ndim != 2 or sizes.shape[1] != 2 or len(sizes) == 0 or sizes.dtype.kind not in "iu" or np.any(sizes <= 0):
                raise error("PromptDA native source_image_sizes must be positive per-frame (width,height)")
            if int(sizes.max()) != resolution.item():
                raise error("PromptDA native preprocess_resolution must equal the source image longest edge")
            if "network_image_sizes" in cache:
                network = cache["network_image_sizes"]
                if network.shape != sizes.shape or network.dtype.kind not in "iu" or not np.array_equal(network, (sizes + 13) // 14 * 14):
                    raise error("PromptDA native network_image_sizes must describe right/bottom patch14 padding")
        if model.shape != () or model.dtype.kind != "U" or not str(model.item()).startswith("promptda/"):
            raise error("PromptDA scene cache source_model must identify promptda")


def promptda_processed_size(
    original_width: int, original_height: int, max_size: int = 504,
) -> tuple[int, int]:
    """Mirror the selected official RGB loader: AREA resize, floor to patch14.

    Small images are kept at their original resolution. This differs from DA3's
    resize/round/crop preprocessing and must not be inferred from its settings.
    """
    if min(original_width, original_height) <= 0 or max_size < 14:
        raise ValueError("PromptDA image dimensions must be positive; max_size >= 14")
    limit = max_size // 14 * 14
    if max(original_width, original_height) <= limit:
        return original_width, original_height
    scale = limit / max(original_width, original_height)
    width = int(original_width * scale // 14 * 14)
    height = int(original_height * scale // 14 * 14)
    if min(width, height) <= 0:
        raise ValueError("PromptDA resize produces an empty patch grid")
    return width, height


def resize_pixel_center_affine(
    source_width: int, source_height: int, target_width: int, target_height: int,
) -> np.ndarray:
    """OpenCV resize coordinates, including the half-pixel translation."""
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Resize dimensions must be positive")
    sx, sy = target_width / source_width, target_height / source_height
    return np.asarray([[sx, 0., (sx - 1.) / 2.],
                       [0., sy, (sy - 1.) / 2.]], dtype=np.float64)


def prepare_promptda_rgb(rgb: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Reproduce official float32 AREA resizing and the candidate's uint8 packing."""
    import cv2

    pixels = np.asarray(rgb, dtype=np.float32) / 255.
    if (pixels.shape[1], pixels.shape[0]) != target_size:
        pixels = cv2.resize(pixels, target_size, interpolation=cv2.INTER_AREA)
    return np.rint(pixels * 255).clip(0, 255).astype(np.uint8)
