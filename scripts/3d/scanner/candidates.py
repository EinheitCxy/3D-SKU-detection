"""The two selected Scanner reconstruction policies; no matching imports."""

from __future__ import annotations

import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from scanner_io import promptda_grid_size, resize_intrinsics

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTDA_MAX_SIZE = 1008
DEFAULT_PROMPTDA_PRECISION = "fp16"
DEFAULT_PROMPTDA_BATCH_SIZE = 2


def normalize_prompt_depth(prompt_depth):
    """Official q=0/q=1 normalization without sorting finite depth prompts."""
    import torch

    minimum, maximum = torch.aminmax(prompt_depth.flatten(1), dim=1, keepdim=True)
    minimum, maximum = minimum[:, :, None, None], maximum[:, :, None, None]
    return (prompt_depth - minimum) / (maximum - minimum), minimum, maximum


def load_candidate(
    name, checkpoint, device, promptda_repo=None, *, promptda_matmul_precision="highest"
):
    import torch

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Selected candidate inference requires CUDA")
    if name == "da3":
        sys.path.insert(0, str(REPO_ROOT / "src"))
        sys.path.insert(0, str(REPO_ROOT / "Depth-Anything-3/src"))
        from da3_model_loading import load_da3_model

        model = load_da3_model(str(checkpoint)).eval()
    elif name == "promptda":
        if not checkpoint.is_file():
            raise ValueError("PromptDA requires a local model.ckpt file")
        if (
            promptda_repo is None
            or not (Path(promptda_repo) / "promptda/promptda.py").is_file()
        ):
            raise FileNotFoundError(
                "Provide the official PromptDA checkout with --promptda-repo"
            )
        sys.path.insert(0, str(Path(promptda_repo).resolve()))
        from promptda.promptda import PromptDA

        class NativePromptDA(PromptDA):
            def normalize(self, prompt_depth):
                return normalize_prompt_depth(prompt_depth)

        torch.set_float32_matmul_precision(promptda_matmul_precision)
        model = NativePromptDA.from_pretrained(
            str(checkpoint), model_kwargs={"encoder": "vitl"}
        ).eval()
        if next(model.parameters()).dtype != torch.float32:
            raise ValueError(
                "The selected PromptDA full policy requires FP32 parameters"
            )
    else:
        raise ValueError(f"Unknown candidate: {name}")
    return model.to(device)


def predict_da3(model, images, sensor) -> dict:
    prediction = model.inference(
        images,
        intrinsics=sensor["intrinsics"].copy(),
        extrinsics=sensor["w2c"].copy(),
        align_to_input_ext_scale=False,
        process_res=504,
        process_res_method="upper_bound_resize",
        use_ray_pose=False,
        ref_view_strategy="saddle_balanced",
    )
    if prediction.is_metric != 1:
        raise ValueError(
            "DA3 policy requires native metric depth; no automatic scale fitting"
        )
    # False keeps depth but still returns similarity-aligned predicted poses.
    # Explicitly restore trusted Scanner w2c; keep native depth and processed K.
    prediction.extrinsics = sensor["w2c"][:, :3, :].copy()
    return {
        "depth": prediction.depth,
        "conf": prediction.conf,
        "intrinsics": prediction.intrinsics,
        "extrinsics": prediction.extrinsics,
        "processed_images": prediction.processed_images,
        "frame_ids": sensor["frame_ids"].copy(),
        "is_metric": np.asarray(prediction.is_metric),
        "scale_factor": np.asarray(prediction.scale_factor),
    }


def predict_promptda(
    model,
    images,
    sensor,
    device,
    *,
    batch_size=DEFAULT_PROMPTDA_BATCH_SIZE,
    precision=DEFAULT_PROMPTDA_PRECISION,
    max_size=DEFAULT_PROMPTDA_MAX_SIZE,
) -> dict:
    import cv2
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    if batch_size < 1:
        raise ValueError("PromptDA batch_size must be positive")
    if precision not in ("fp32", "fp16"):
        raise ValueError("PromptDA precision must be fp32 or fp16")
    if precision == "fp16" and torch.device(device).type != "cuda":
        raise ValueError("PromptDA FP16 autocast requires CUDA")

    prompt = np.where(
        np.isfinite(sensor["lidar_depth"]) & (sensor["lidar_depth"] > 0),
        sensor["lidar_depth"],
        0,
    ).astype(np.float32)
    if np.any(np.ptp(prompt, axis=(1, 2)) <= 0):
        raise ValueError(
            "Official PromptDA normalization needs nonconstant depth prompts"
        )
    source_size = tuple(map(int, sensor["image_size"]))
    width, height = promptda_grid_size(source_size, max_size)
    pad_width, pad_height = -width % 14, -height % 14
    depths, colors = [], []
    for start in range(0, len(images), batch_size):
        tensors, prompts = [], []
        for image_path, native_depth in zip(
            images[start : start + batch_size],
            prompt[start : start + batch_size],
            strict=True,
        ):
            with Image.open(image_path) as source:
                if source.size != source_size:
                    raise ValueError(
                        "PromptDA RGB size differs from Scanner image_size"
                    )
                rgb = np.asarray(source.convert("RGB"))
            pixels = rgb.astype(np.float32) / 255
            if source_size != (width, height):
                pixels = cv2.resize(
                    pixels, (width, height), interpolation=cv2.INTER_AREA
                )
            colors.append(np.rint(pixels * 255).clip(0, 255).astype(np.uint8))
            padded = np.pad(
                pixels, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge"
            )
            tensors.append(torch.from_numpy(padded).permute(2, 0, 1))
            depth_input = torch.from_numpy(native_depth[None, None])
            if pad_width or pad_height:
                # Keep the source RGB/LiDAR field of view aligned after padding.
                # Nearest-exact uses pixel centres and preserves raw metric values.
                depth_input = functional.interpolate(
                    depth_input, size=(height, width), mode="nearest-exact"
                )
                depth_input = functional.pad(
                    depth_input, (0, pad_width, 0, pad_height), mode="replicate"
                )
            prompts.append(depth_input)
        image = torch.stack(tensors)
        depth_input = torch.cat(prompts)
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if precision == "fp16"
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            depth = model.predict(image.to(device), depth_input.to(device))
            if tuple(depth.shape) != (
                len(tensors),
                1,
                height + pad_height,
                width + pad_width,
            ):
                raise ValueError("PromptDA output must match the padded RGB grid")
            depths.extend(depth[:, 0, :height, :width].float().cpu().numpy().copy())
    return {
        "depth": np.stack(depths),
        "processed_images": np.stack(colors),
        "intrinsics": resize_intrinsics(
            sensor["intrinsics"], source_size, (width, height)
        ),
        "extrinsics": sensor["w2c"][:, :3, :].copy(),
        "frame_ids": sensor["frame_ids"].copy(),
        "is_metric": np.asarray(1),
        "preprocess_method": np.asarray(
            "promptda_native_pad14" if max_size == 0 else "promptda_resize"
        ),
        "preprocess_resolution": np.asarray(
            max(source_size) if max_size == 0 else max_size, dtype=np.int64
        ),
        "network_image_sizes": np.tile(
            np.asarray([width + pad_width, height + pad_height], dtype=np.int64),
            (len(images), 1),
        ),
    }


def predict_candidate(
    name,
    model,
    images,
    sensor,
    device,
    *,
    promptda_batch_size=DEFAULT_PROMPTDA_BATCH_SIZE,
    promptda_precision=DEFAULT_PROMPTDA_PRECISION,
    promptda_max_size=DEFAULT_PROMPTDA_MAX_SIZE,
) -> dict:
    import torch

    # Reset after model construction, which itself consumes random numbers.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if name == "da3":
        result = predict_da3(model, images, sensor)
    elif name == "promptda":
        result = predict_promptda(
            model,
            images,
            sensor,
            device,
            batch_size=promptda_batch_size,
            precision=promptda_precision,
            max_size=promptda_max_size,
        )
    else:
        raise ValueError(f"Unknown candidate: {name}")
    if len(result["depth"]) != len(images) or not np.all(
        np.isfinite(result["depth"]) & (result["depth"] > 0)
    ):
        raise ValueError("Predicted depths must be finite, positive and frame aligned")
    return result
