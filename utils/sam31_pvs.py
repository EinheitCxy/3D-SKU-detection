"""Native SAM3.1 box-prompt segmentation shared by matching and experiments."""

from pathlib import Path
from contextlib import nullcontext
from importlib import import_module
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from .sam3_utils import clip_mask_to_bbox
from .sam31_initialization import audit_skipped_weights, skip_large_weight_initialization


def load_model(checkpoint: Path, device: str = "cuda", *, skip_random_init: bool = True):
    """Load the native tracker with audited selective initialization by default.

    Explicit False retains native random fills for controlled comparisons.
    """
    if torch.device(device).type != "cuda":
        raise ValueError("SAM3.1 PVS requires a CUDA device")
    from sam3.model_builder import build_sam3_multiplex_video_model

    # Position-encoding caches allocate on the current CUDA device during build.
    if skip_random_init:
        # ViT binds timm's trunc_normal_ by name and reapplies it after resets.
        initialization = skip_large_weight_initialization(
            random_initializer_modules=(import_module("sam3.model.vitdet"),),
        )
    else:
        initialization = nullcontext({})
    started = time.perf_counter()
    with initialization as skipped, torch.cuda.device(device):
        model = build_sam3_multiplex_video_model(
            checkpoint_path=None,
            load_from_HF=False,
            device="cpu",
            use_fa3=False,
            use_rope_real=False,
            compile=False,
        )
    construct_s = time.perf_counter() - started
    started = time.perf_counter()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if "model" in state:
        state = state["model"]
    # The released combined checkpoint shares its detector's vision backbone
    # with the tracker; the standalone native tracker owns that backbone.
    weights = {
        key.removeprefix("tracker.model."): value
        for key, value in state.items()
        if key.startswith("tracker.model.")
    }
    weights.update(
        {
            key.removeprefix("detector."): value
            for key, value in state.items()
            if key.startswith("detector.backbone.vision_backbone.")
        }
    )
    checkpoint_s = time.perf_counter() - started
    started = time.perf_counter()
    report = audit_skipped_weights(model, weights, skipped)
    report.update(skip_random_init=skip_random_init, construct_s=construct_s,
                  checkpoint_s=checkpoint_s, audit_s=time.perf_counter() - started)
    started = time.perf_counter()
    model.load_state_dict(weights, strict=True)
    report["strict_load_s"] = time.perf_counter() - started
    # Do not retain extra Parameter references in the loading report.
    skipped.clear()
    started = time.perf_counter()
    model = model.eval().to(device)
    report["eval_to_device_host_s"] = time.perf_counter() - started
    model._pvs_loading_report = report
    return model, len(weights)


@torch.inference_mode()
def predict(
    model, image: Image.Image, boxes: np.ndarray, batch_size: int, device: str = "cuda"
):
    """Return one clipped mask per box in the input image coordinate space."""
    if batch_size < 1:
        raise ValueError("PVS batch_size must be positive")
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.shape != (len(boxes), 4):
        raise ValueError("PVS boxes must have shape (N, 4)")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("PVS boxes must be finite XYXY with positive area")
    if not len(boxes):
        return np.zeros((0, image.height, image.width), dtype=bool), [], [], [], {
            "encoder_s": 0.0, "inference_and_postprocess_s": 0.0
        }
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((1008, 1008)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    image_tensor = transform(image).unsqueeze(0).to(device)
    box_tensor = torch.as_tensor(boxes, device=device)
    box_tensor = box_tensor * box_tensor.new_tensor(
        [1008 / image.width, 1008 / image.height] * 2
    )
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # Native forward_image projects the two high-resolution feature levels
        # with interactive conv_s0/conv_s1 once for all box batches.
        output = model.forward_image(image_tensor, need_interactive_out=True)
        features = [entry.tensors for entry in output["interactive"]["backbone_fpn"]]
        memory = model.interactivity_no_mem_embed.reshape(1, 256, 1, 1)
        image_embedding = features[-1] + memory
        torch.cuda.synchronize(device)
        encoder_s = time.perf_counter() - start
        masks, scores, presence, raw_areas = [], [], [], []
        for offset in range(0, len(boxes), batch_size):
            batch_boxes = box_tensor[offset : offset + batch_size]
            # Labels 2/3 are the native SAM top-left/bottom-right box corners,
            # exactly as used by SAM3InteractiveImagePredictor._predict.
            point_inputs = {
                "point_coords": batch_boxes.reshape(-1, 2, 2),
                "point_labels": torch.tensor([[2, 3]], device=device, dtype=torch.int32)
                .expand(len(batch_boxes), -1),
            }
            result = model._forward_sam_heads(
                image_embedding,
                point_inputs=point_inputs,
                interactive_high_res_features=features[:2],
                multimask_output=False,
                multiplex_state=None,
                objects_to_interact=list(range(len(batch_boxes))),
            )
            logits = F.interpolate(
                result["high_res_masks"], size=(image.height, image.width),
                mode="bilinear", align_corners=False,
            )
            if not torch.isfinite(logits).all() or not torch.isfinite(result["ious"]).all():
                raise RuntimeError("PVS returned nonfinite masks or quality scores")
            batch_masks = (logits[:, 0] > 0).cpu().numpy()
            if len(batch_masks) != len(batch_boxes):
                raise RuntimeError("PVS did not return one mask per box")
            for box, mask in zip(boxes[offset : offset + batch_size], batch_masks):
                raw_areas.append(int(mask.sum()))
                masks.append(clip_mask_to_bbox(mask, box))
            scores.extend(result["ious"][:, 0].float().cpu().tolist())
            presence.extend(result["object_score_logits"].flatten().float().cpu().tolist())
        torch.cuda.synchronize(device)
    return np.stack(masks), scores, presence, raw_areas, {
        "encoder_s": encoder_s,
        "inference_and_postprocess_s": time.perf_counter() - start,
    }
