"""Fresh-process native/candidate PVS component comparison, no prepare thread.

GPU execution belongs to the coordinator. Inputs are a real DA3 processed RGB
frame and its source detections mapped through the cache affine. This measures
PVS loading/predict, not source-image preparation or whole-pipeline speed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[3]


def single(args):
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(ROOT))
    from utils.data_utils import load_detections
    from utils.sam3_mask_cache import map_source_bbox_to_processed
    from utils.sam3_utils import set_sam3_code_root, _ensure_sam3_in_path
    from utils.sam31_pvs import load_model, predict

    started = time.perf_counter()
    result = {"mode": args.mode, "status": "running", "pid": os.getpid(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "interpretation": __doc__}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("The real PVS probe requires a coordinator-approved CUDA device")
        with np.load(args.da3_cache, allow_pickle=False) as cache:
            image_id = int(cache["image_ids"][args.reference_index])
            rgb = cache["images"][args.reference_index]
            affine = cache["source_to_processed_affine"][args.reference_index]
        frame_detections = dict(load_detections(str(args.detections_dir), return_index_map=True))[image_id]["objects"]
        boxes = np.asarray([
            map_source_bbox_to_processed(detection["position"], affine, tuple(rgb.shape[:2]))
            for detection in frame_detections
        ], dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            raise ValueError("Select a real frame containing detections")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("DA3 images must be uint8 processed-space RGB")
        result["input"] = {"da3_cache": str(args.da3_cache.resolve()),
                           "detections_dir": str(args.detections_dir.resolve()),
                           "reference_index": args.reference_index, "image_id": image_id,
                           "rgb_shape": list(rgb.shape), "boxes": len(boxes),
                           "box_order": "source detection order; identical in both arms",
                           "checkpoint": str(args.checkpoint.resolve()),
                           "code_root": str(args.code_root.resolve())}
        set_sam3_code_root(args.code_root)
        _ensure_sam3_in_path()
        torch.manual_seed(args.seed)
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        begin = time.perf_counter()
        model, weight_keys = load_model(args.checkpoint, args.device, skip_random_init=args.mode == "candidate")
        torch.cuda.synchronize(args.device)
        result["load_completed_s"] = time.perf_counter() - begin
        result["loading"] = model._pvs_loading_report
        result["checkpoint_weight_keys"] = weight_keys
        result["load_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
        result["load_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
        retained_cache = []
        for name, module in model.named_modules():
            for key, value in getattr(module, "cache", {}).items():
                if isinstance(value, torch.Tensor):
                    retained_cache.append({"module": name, "key": str(key), "shape": list(value.shape),
                                           "dtype": str(value.dtype), "device": str(value.device)})
        result["ordinary_tensor_cache_structure"] = retained_cache
        begin = time.perf_counter()
        masks, scores, presence, areas, metrics = predict(model, Image.fromarray(rgb), boxes, 4, args.device)
        result["predict_completed_s"] = time.perf_counter() - begin
        result["predict_native_metrics"] = metrics
        result["task_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
        result["task_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
        result["nonempty_masks"] = int(masks.any(axis=(1, 2)).sum())
        np.savez(args.out / f"{args.mode}-output.npz", masks=masks, scores=np.asarray(scores),
                 presence=np.asarray(presence), raw_areas=np.asarray(areas), boxes=boxes)
        result["status"] = "success"
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(elapsed_s=time.perf_counter() - started,
                      rss_hwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (args.out / f"{args.mode}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"mode": args.mode, "status": result["status"],
                          "result": str(args.out / f"{args.mode}.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "candidate", "pair"), default="pair")
    parser.add_argument("--da3-cache", type=Path, required=True)
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "sam31/checkpoints/sam3.1_multiplex.pt")
    parser.add_argument("--code-root", type=Path, default=ROOT / "sam31")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode != "pair":
        args.out.mkdir(parents=True, exist_ok=True)
        if (args.out / f"{args.mode}.json").exists():
            raise FileExistsError("Use a new probe output directory")
        single(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "interpretation": "Sequential fresh-process PVS component pair; not pipeline speedup or driver memory peak."}
    try:
        for mode in ("baseline", "candidate"):
            command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode,
                       "--da3-cache", str(args.da3_cache), "--detections-dir", str(args.detections_dir),
                       "--reference-index", str(args.reference_index), "--checkpoint", str(args.checkpoint),
                       "--code-root", str(args.code_root), "--device", args.device,
                       "--seed", str(args.seed), "--out", str(args.out)]
            with (args.out / f"{mode}.log").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        import numpy as np

        native = json.loads((args.out / "baseline.json").read_text())
        candidate = json.loads((args.out / "candidate.json").read_text())
        with np.load(args.out / "baseline-output.npz", allow_pickle=False) as a, np.load(args.out / "candidate-output.npz", allow_pickle=False) as b:
            equality = {key: bool(np.array_equal(a[key], b[key])) for key in a.files}
        result.update(status="success", output_equal=equality,
                      cache_structure_equal=native["ordinary_tensor_cache_structure"] == candidate["ordinary_tensor_cache_structure"],
                      baseline_load_completed_s=native["load_completed_s"],
                      candidate_load_completed_s=candidate["load_completed_s"],
                      candidate_coverage=candidate["loading"])
        if not all(equality.values()) or not result["cache_structure_equal"]:
            raise ValueError("PVS native/candidate outputs or cache structure differ")
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.out / "pair.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"status": result["status"], "result": str(args.out / "pair.json")}))


if __name__ == "__main__":
    main()
