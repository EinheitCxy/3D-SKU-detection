"""The two selected Scanner reconstruction policies; no matching imports."""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import numpy as np

from scanner_io import resize_intrinsics

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_candidate(name, checkpoint, device, promptda_repo=None):
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

        # Import the official RGB loader's dependencies before request timing.
        importlib.import_module("promptda.utils.io_wrapper")
        model = PromptDA.from_pretrained(
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


def predict_promptda(model, images, sensor, device) -> dict:
    import torch
    from promptda.utils.io_wrapper import load_image

    prompt = np.where(
        np.isfinite(sensor["lidar_depth"]) & (sensor["lidar_depth"] > 0),
        sensor["lidar_depth"],
        0,
    ).astype(np.float32)
    if np.any(np.ptp(prompt, axis=(1, 2)) <= 0):
        raise ValueError(
            "Official PromptDA normalization needs nonconstant depth prompts"
        )
    depths, colors = [], []
    for image_path, native_depth in zip(images, prompt, strict=True):
        image = load_image(str(image_path), max_size=504, multiple_of=14)
        depth_input = torch.from_numpy(native_depth[None, None])
        with torch.no_grad():
            depth = (
                model.predict(image.to(device), depth_input.to(device))
                .cpu()
                .numpy()[0, 0]
            )
        rgb = image[0].permute(1, 2, 0).numpy()
        depths.append(depth)
        colors.append(np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8))
    depth = np.stack(depths)
    height, width = depth.shape[1:]
    return {
        "depth": depth,
        "processed_images": np.stack(colors),
        "intrinsics": resize_intrinsics(
            sensor["intrinsics"], sensor["image_size"], (width, height)
        ),
        "extrinsics": sensor["w2c"][:, :3, :].copy(),
        "frame_ids": sensor["frame_ids"].copy(),
        "is_metric": np.asarray(1),
    }


def predict_candidate(name, model, images, sensor, device) -> dict:
    import torch

    # Reset after model construction, which itself consumes random numbers.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if name == "da3":
        result = predict_da3(model, images, sensor)
    elif name == "promptda":
        result = predict_promptda(model, images, sensor, device)
    else:
        raise ValueError(f"Unknown candidate: {name}")
    if len(result["depth"]) != len(images) or not np.all(
        np.isfinite(result["depth"]) & (result["depth"] > 0)
    ):
        raise ValueError("Predicted depths must be finite, positive and frame aligned")
    return result
