"""Isolated PromptDA request experiments; importing performs no work.

No RNG reset here: the caller must reset Python/NumPy/Torch/CUDA to 42 before
each request, as candidates.predict_candidate does after model construction.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np


def _scanner():
    scanner = str(Path(__file__).resolve().parents[2] / "scripts/3d/scanner")
    if scanner not in sys.path:
        sys.path.insert(0, scanner)
    import scanner_io

    return scanner_io


def _prepare_rgb(paths, source_size, width, height, reduced):
    """CPU-only producer: no Torch import, tensors, CUDA, or Torch CPU ops."""
    import cv2
    from PIL import Image

    pad_width, pad_height = -width % 14, -height % 14
    if reduced:
        packed = np.empty((len(paths), 3, height + pad_height, width + pad_width), np.float32)
        colors = np.empty((len(paths), height, width, 3), np.uint8)
    else:
        packed, colors = [], []
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            if source.size != source_size:
                raise ValueError("PromptDA RGB size differs from Scanner image_size")
            rgb = np.asarray(source.convert("RGB"))
        if reduced:
            pixels = rgb.astype(np.float32)
            pixels /= 255
        else:
            pixels = rgb.astype(np.float32) / 255
        if source_size != (width, height):
            pixels = cv2.resize(pixels, (width, height), interpolation=cv2.INTER_AREA)
        if reduced:
            saved = pixels * 255
            np.rint(saved, out=saved)
            np.clip(saved, 0, 255, out=saved)
            colors[index] = saved.astype(np.uint8)
            padded = (
                np.pad(pixels, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")
                if pad_width or pad_height else pixels
            )
            packed[index] = padded.transpose(2, 0, 1)
        else:
            colors.append(np.rint(pixels * 255).clip(0, 255).astype(np.uint8))
            packed.append(np.pad(pixels, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge"))
    return packed, colors


def predict_prefetch(
    model, images, sensor, device, *, batch_size=2, precision="fp16",
    max_size=1008, reduced=False, prefetch=True, delayed_d2h=False,
):
    """Pure prefetch by default; reduced/delayed_d2h are independent switches.

    At most one producer and one future (next batch), plus the consumed batch.
    Executor creation, decode, shutdown and all CPU outputs are inside this call.
    prefetch=False,reduced=True isolates copy/allocation changes without overlap.
    """
    import torch
    import torch.nn.functional as functional

    if batch_size < 1:
        raise ValueError("PromptDA batch_size must be positive")
    if precision not in ("fp32", "fp16"):
        raise ValueError("PromptDA precision must be fp32 or fp16")
    if precision == "fp16" and torch.device(device).type != "cuda":
        raise ValueError("PromptDA FP16 autocast requires CUDA")
    prompt = np.where(
        np.isfinite(sensor["lidar_depth"]) & (sensor["lidar_depth"] > 0),
        sensor["lidar_depth"], 0,
    ).astype(np.float32)
    if np.any(np.ptp(prompt, axis=(1, 2)) <= 0):
        raise ValueError("Official PromptDA normalization needs nonconstant depth prompts")
    source_size = tuple(map(int, sensor["image_size"]))
    width, height = _scanner().promptda_grid_size(source_size, max_size)
    pad_width, pad_height = -width % 14, -height % 14
    depths, colors, held_depths = [], [], []
    starts = list(range(0, len(images), batch_size))
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="promptda-rgb") if prefetch else None
    future = None

    def prepare(start):
        # Validate the original strict zip before work (including short tails).
        paths = images[start:start + batch_size]
        list(zip(paths, prompt[start:start + batch_size], strict=True))
        return _prepare_rgb(paths, source_size, width, height, reduced)

    try:
        if starts and executor is not None:
            future = executor.submit(prepare, starts[0])
        for batch_index, start in enumerate(starts):
            if executor is not None:
                packed, batch_colors = future.result()
                future = None
                if batch_index + 1 < len(starts):
                    future = executor.submit(prepare, starts[batch_index + 1])
            else:
                packed, batch_colors = prepare(start)
            colors.extend(batch_colors)
            if reduced:
                image = torch.from_numpy(packed)
            else:
                tensors = [torch.from_numpy(padded).permute(2, 0, 1) for padded in packed]
                image = torch.stack(tensors)
            prompts = []
            for native_depth in prompt[start:start + batch_size]:
                depth_input = torch.from_numpy(native_depth[None, None])
                if pad_width or pad_height:
                    depth_input = functional.interpolate(depth_input, size=(height, width), mode="nearest-exact")
                    depth_input = functional.pad(depth_input, (0, pad_width, 0, pad_height), mode="replicate")
                prompts.append(depth_input)
            depth_input = torch.cat(prompts)
            autocast = torch.autocast("cuda", dtype=torch.float16) if precision == "fp16" else nullcontext()
            with torch.no_grad(), autocast:
                depth = model.predict(image.to(device), depth_input.to(device))
                if tuple(depth.shape) != (len(prompts), 1, height + pad_height, width + pad_width):
                    raise ValueError("PromptDA output must match the padded RGB grid")
                if delayed_d2h:
                    held_depths.append(depth[:, 0, :height, :width].float())
                else:
                    depths.extend(depth[:, 0, :height, :width].float().cpu().numpy().copy())
            # Drop the consumed RGB batch before waiting for the next future.
            del packed, image, batch_colors, depth_input, prompts, depth
            if not reduced:
                del tensors
        for depth in held_depths:
            depths.extend(depth.cpu().numpy().copy())
    finally:
        if future is not None:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return {
        "depth": np.stack(depths), "processed_images": np.stack(colors),
        "intrinsics": _scanner().resize_intrinsics(sensor["intrinsics"], source_size, (width, height)),
        "extrinsics": sensor["w2c"][:, :3, :].copy(),
        "frame_ids": sensor["frame_ids"].copy(), "is_metric": np.asarray(1),
        "preprocess_method": np.asarray("promptda_native_pad14" if max_size == 0 else "promptda_resize"),
        "preprocess_resolution": np.asarray(max(source_size) if max_size == 0 else max_size, dtype=np.int64),
        "network_image_sizes": np.tile(np.asarray([width + pad_width, height + pad_height], dtype=np.int64), (len(images), 1)),
    }


def cpu_input_check(images, sensor, *, batch_size=2, max_size=1008):
    """No GPU/model load; compare native batch inputs and every returned field.

    This checker retains native CPU inputs and executes three CPU requests; do
    not include it in GPU timing processes (Torch CPU packing can start pools).
    """
    import torch
    _scanner()
    from candidates import predict_promptda

    captured = []

    class Capture:
        def __init__(self, reference=None):
            self.reference, self.index = reference, 0

        def predict(self, image, prompt):
            actual = (image.numpy(), prompt.numpy())
            if self.reference is None:
                captured.append(tuple(value.copy() for value in actual))
            else:
                expected = self.reference[self.index]
                for name, a, b in zip(("image", "prompt"), actual, expected, strict=True):
                    if a.dtype != b.dtype or not np.array_equal(a, b):
                        raise AssertionError(f"batch {self.index} {name} differs")
            self.index += 1
            return torch.from_numpy(np.ones((len(image), 1, *image.shape[-2:]), np.float32))

    native = predict_promptda(Capture(), images, sensor, "cpu", batch_size=batch_size, precision="fp32", max_size=max_size)
    checked = []
    for name, flags in (
        ("prefetch", {}),
        ("reduced_serial", {"reduced": True, "prefetch": False}),
        ("reduced_prefetch", {"reduced": True}),
    ):
        model = Capture(captured)
        result = predict_prefetch(model, images, sensor, "cpu", batch_size=batch_size, precision="fp32", max_size=max_size, **flags)
        assert model.index == len(captured), "batch count differs"
        assert result.keys() == native.keys(), "metadata keys differ"
        for key in result:
            if result[key].dtype != native[key].dtype or not np.array_equal(result[key], native[key]):
                raise AssertionError(f"{name}: result {key} differs")
        checked.append(name)
    return {"status": "PASS", "frames": len(images), "batch_sizes": [len(pair[0]) for pair in captured], "variants": checked}
