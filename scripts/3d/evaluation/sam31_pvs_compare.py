"""Measure PVS or self-exemplar on a shared, explicit image/box sample list."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

import numpy as np
import torch

import sam31_pvs_trial as trial
from utils import sam3_utils


class ProcessMemorySampler:
    """Observe this PID's driver memory, independently of other GPU users."""

    def __init__(self):
        self.phase = "load"
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        start = time.perf_counter()
        while not self.stop_event.is_set():
            phase = self.phase
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                     "--format=csv,noheader,nounits"], text=True, timeout=5,
                )
                for line in output.splitlines():
                    pid, memory = [item.strip() for item in line.split(",")]
                    if int(pid) == os.getpid():
                        self.samples.append({
                            "elapsed_s": time.perf_counter() - start,
                            "phase": phase, "used_mib": int(memory),
                        })
            except Exception as error:
                self.error = repr(error)
                return
            self.stop_event.wait(0.2)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=6)


def gpu_stats():
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def write_report(path, report, sampler):
    report["driver_memory_sampler"] = {
        "pid": os.getpid(), "interval_after_query_s": 0.2,
        "error": sampler.error, "samples": list(sampler.samples),
        "observed_peak_mib": max((s["used_mib"] for s in sampler.samples), default=None),
    }
    path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["pvs", "self-exemplar"], required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-image-side", type=int, default=1008)
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=9)
    args = parser.parse_args()
    if min(args.batch_size, args.repeats, args.max_image_side, args.gpu_memory_limit_gib) <= 0:
        parser.error("batch, repeat, image size and memory limit must be positive")
    samples = json.loads(args.samples.read_text())["samples"]
    if not samples:
        parser.error("sample list must not be empty")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    free, total = torch.cuda.mem_get_info()
    limit = args.gpu_memory_limit_gib * 1024**3
    if free < limit + 1024**3 or limit > total:
        raise RuntimeError("Insufficient GPU memory for trial cap plus 1 GiB headroom")
    torch.cuda.set_per_process_memory_fraction(limit / total)
    args.output.mkdir(parents=True, exist_ok=False)
    sampler = ProcessMemorySampler()
    sampler.start()
    checkpoint = trial.ROOT / "sam31/checkpoints/sam3.1_multiplex.pt"
    report = {
        "status": "running", "arm": args.arm, "samples": samples,
        "batch_size": args.batch_size, "repeats": args.repeats,
        "max_image_side": args.max_image_side, "encoder_wh": [1008, 1008],
        "checkpoint": str(checkpoint.resolve()),
        "gpu": torch.cuda.get_device_name(), "gpu_total_gib": total / 1024**3,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "precision": "FP32 weights / BF16 autocast",
        "tf32": True, "cpu_threads": 4, "mask_cache": False,
        "warmup_count": 1, "warmup_sample": samples[0],
        "timing_scope": "Resident image API: preloaded PIL image + boxes to CPU bool masks; includes API preprocessing/H2D/encoder/decoder/D2H/bbox clipping; excludes image reading, model loading, validation and output files",
        "production_batch_size": 32, "baseline_threshold": 0.5,
        "boundary": "Standalone SAM stage at longest side 1008, not DA3 processed space or full matching service",
        "frames": [],
    }
    path = args.output / "results.json"
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if args.arm == "pvs":
            model, count = trial.load_model(checkpoint)
            report["loaded_tensors"] = count
            report["native_interactive_stability_selection"] = {
                "enabled": model.interactive_sam_mask_decoder.dynamic_multimask_via_stability,
                "delta": model.interactive_sam_mask_decoder.dynamic_multimask_stability_delta,
                "threshold": model.interactive_sam_mask_decoder.dynamic_multimask_stability_thresh,
            }
        else:
            sam3_utils.set_sam3_code_root(trial.ROOT / "sam31")
            model, _, _ = sam3_utils._get_sam3_batch_components(
                str(checkpoint), "cuda", detection_threshold=0.5,
                max_dets_per_img=1, fallback_size=1008, skip_resize=False,
            )
        torch.cuda.synchronize()
        report["model_load_s"] = time.perf_counter() - start
        report["load_gpu_memory"] = gpu_stats()

        def infer(image, boxes):
            if args.arm == "pvs":
                return trial.predict(model, image, boxes, args.batch_size)[0]
            return sam3_utils.sam3_masks_self_exemplar(
                image_path=image, bboxes_xyxy=boxes.tolist(),
                checkpoint_path=str(checkpoint), device="cuda",
                detection_threshold=0.5, max_batch_size=args.batch_size,
                fallback_size=1008, skip_resize=False,
            )

        sampler.phase = "warmup"
        first = samples[0]
        image, boxes, _, _ = trial.load_frame(
            trial.ROOT / first["dataset"], first["frame"], args.max_image_side,
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        warmup_masks = infer(image, boxes)
        torch.cuda.synchronize()
        report["warmup_s"] = time.perf_counter() - start
        report["warmup_gpu_memory"] = gpu_stats()
        del warmup_masks
        write_report(path, report, sampler)

        for sample in samples:
            dataset = trial.ROOT / sample["dataset"]
            frame = sample["frame"]
            image, boxes, source, original_wh = trial.load_frame(dataset, frame, args.max_image_side)
            if len(boxes) != sample["boxes"]:
                raise RuntimeError(f"Detection count changed for {sample}")
            row = {"dataset": sample["dataset"], "frame": frame,
                   "source": str(source), "boxes": len(boxes),
                   "original_wh": original_wh, "output_wh": image.size, "runs": []}
            reference_masks = None
            for repeat in range(args.repeats):
                sampler.phase = "measure"
                torch.cuda.synchronize()
                resident = gpu_stats()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                result = infer(image, boxes)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                memory = gpu_stats()
                sampler.phase = "validation"
                masks = np.asarray(result)
                if masks.dtype != np.bool_ or masks.shape != (len(boxes), image.height, image.width):
                    raise RuntimeError(f"Unexpected masks: {masks.dtype}, {masks.shape}")
                areas = masks.sum(axis=(1, 2))
                empty_ids = np.flatnonzero(areas == 0).tolist()
                changed = 0 if reference_masks is None else int(np.count_nonzero(masks != reference_masks))
                if reference_masks is None:
                    reference_masks = masks
                row["runs"].append({
                    "repeat": repeat, "seconds": elapsed, "returned_masks": len(masks),
                    "nonempty_masks": int((areas > 0).sum()), "empty_object_ids": empty_ids,
                    "mask_pixels": areas.tolist(), "changed_pixels_vs_first": changed,
                    "resident_before": resident, **memory,
                })
                del result
            row["median_s"] = statistics.median(run["seconds"] for run in row["runs"])
            row["repeat_masks_identical"] = all(run["changed_pixels_vs_first"] == 0 for run in row["runs"])
            sampler.phase = "write"
            target = args.output / dataset.name / frame
            target.mkdir(parents=True)
            np.savez_compressed(target / "masks.npz", masks=masks, boxes_xyxy=boxes)
            report["frames"].append(row)
            write_report(path, report, sampler)
            print(json.dumps({"arm": args.arm, "completed": len(report["frames"]),
                              "dataset": dataset.name, "frame": frame, "boxes": len(boxes),
                              "nonempty": [run["nonempty_masks"] for run in row["runs"]],
                              "median_s": row["median_s"],
                              "peak_allocated_gib": max(run["peak_allocated_gib"] for run in row["runs"])}), flush=True)
            del masks, reference_masks
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        sampler.stop()
        write_report(path, report, sampler)


if __name__ == "__main__":
    main()
