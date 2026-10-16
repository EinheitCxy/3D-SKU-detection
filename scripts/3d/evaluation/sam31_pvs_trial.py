"""Try SAM3.1 box-prompt instance masks without publishing matching caches."""

from __future__ import annotations

import argparse
import colorsys
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sam31"))

import numpy as np
from PIL import Image, ImageDraw
import torch

from utils.detection_objects import flatten_detection_objects
from utils.sam31_pvs import load_model, predict


def load_frame(dataset: Path, frame: str, max_side: int):
    candidates = [
        path for path in (dataset / "images").iterdir()
        if path.stem == frame and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one image for frame {frame}: {candidates}")
    image_path = candidates[0]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    original_wh = image.size
    scale = min(1.0, max_side / max(original_wh))
    size = tuple(max(1, round(value * scale)) for value in original_wh)
    image = image.resize(size, Image.Resampling.BILINEAR)
    payload = json.loads((dataset / "detections_results" / f"{frame}.json").read_text())
    objects = flatten_detection_objects(payload)
    boxes = np.asarray([obj["position"] for obj in objects], dtype=np.float32)
    if boxes.shape != (len(objects), 4) or not len(objects):
        raise ValueError(f"Frame {frame} needs nonempty XYXY boxes")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError(f"Invalid XYXY boxes in frame {frame}")
    boxes *= np.asarray([size[0] / original_wh[0], size[1] / original_wh[1]] * 2)
    return image, boxes, image_path, original_wh


def save_frame(output: Path, frame: str, image, boxes, masks, scores, presence, raw_areas):
    canvas = np.array(image).astype(np.float32)
    rows = []
    for object_id, (box, mask, score, present, raw_area) in enumerate(
        zip(boxes, masks, scores, presence, raw_areas)
    ):
        color = np.asarray(colorsys.hsv_to_rgb((object_id * 0.618034) % 1, 0.8, 1)) * 255
        canvas[mask] = canvas[mask] * 0.55 + color * 0.45
        rows.append({
            "object_id": object_id, "bbox_xyxy": box.tolist(),
            "mask_pixels": int(mask.sum()), "raw_mask_pixels": raw_area,
            "predicted_iou": score, "object_presence_logit": present,
        })
    overlay = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    for object_id, box in enumerate(boxes):
        draw.rectangle(tuple(box), outline="yellow", width=1)
        draw.text((float(box[0]), float(box[1])), str(object_id), fill="red", stroke_width=1,
                  stroke_fill="white")
    image.save(output / f"{frame}_input.jpg", quality=92)
    overlay.save(output / f"{frame}_pvs.jpg", quality=94)
    np.savez_compressed(output / f"{frame}_masks.npz", masks=masks, boxes_xyxy=boxes)
    (output / f"{frame}_objects.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "imdata/floor_display2")
    parser.add_argument("--frames", nargs="+", default=["0"])
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "sam31/checkpoints/sam3.1_multiplex.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-image-side", type=int, default=1008)
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=9.0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_image_side < 1 or args.gpu_memory_limit_gib <= 0:
        parser.error("batch size, image side and memory limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("PVS trial requires an explicitly selected CUDA GPU")
    if len(set(args.frames)) != len(args.frames):
        parser.error("frames must be unique")
    torch.set_num_threads(4)
    limit = args.gpu_memory_limit_gib * 1024**3
    free, total = torch.cuda.mem_get_info()
    if free < limit + 1024**3 or limit > total:
        raise RuntimeError("Insufficient free GPU memory for trial limit plus 1 GiB headroom")
    torch.cuda.set_per_process_memory_fraction(limit / total)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model, loaded_tensors = load_model(args.checkpoint)
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - start
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "loaded_tensors": loaded_tensors,
        "strict_load": True, "dataset": str(args.dataset.resolve()),
        "gpu": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "precision": "FP32 weights / BF16 autocast",
        "encoder_size": [1008, 1008], "max_image_side": args.max_image_side,
        "batch_size": args.batch_size, "multimask_output": False,
        "bbox_clip": True, "empty_mask_retry": False,
        "model_load_s": model_load_s, "warmup": False, "frames": [],
        "native_interactive_mask_selection": {
            "dynamic_multimask_via_stability": model.interactive_sam_mask_decoder.dynamic_multimask_via_stability,
            "stability_delta": model.interactive_sam_mask_decoder.dynamic_multimask_stability_delta,
            "stability_threshold": model.interactive_sam_mask_decoder.dynamic_multimask_stability_thresh,
        },
        "timing_scope": "GPU-synchronized image encoder + box decoding + mask CPU transfer and bbox clipping; excludes input preprocessing and output files",
        "scope": "Independent PVS trial; no matching/footprint cache publication or accuracy ground truth",
    }
    for frame in args.frames:
        image, boxes, source, original_wh = load_frame(args.dataset, frame, args.max_image_side)
        masks, scores, presence, raw_areas, timing = predict(model, image, boxes, args.batch_size)
        save_frame(args.output, frame, image, boxes, masks, scores, presence, raw_areas)
        entry = {
            "frame": frame, "source": str(source.resolve()),
            "original_wh": original_wh, "output_wh": image.size, "boxes": len(boxes),
            "nonempty_masks": int(masks.any(axis=(1, 2)).sum()), **timing,
        }
        report["frames"].append(entry)
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
