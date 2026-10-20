"""Interleaved resident-model requests, original arithmetic/output scope."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import statistics
import sys
import time
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/3d/scanner"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import psutil
    import torch
    from candidates import load_candidate, normalize_prompt_depth, predict_promptda
    from scanner_io import load_prepared
    from profile_candidates import RSSSampler
    from request_probe import predict_prefetch

    torch.set_num_threads(8)
    sampler = RSSSampler(psutil.Process())
    sampler.start()
    images, sensor = load_prepared(args.base)
    report = {"status": "running", "frames": len(images), "requests": [], "comparisons": [],
              "scope": "resident model + original PNG paths through all CPU depth/RGB/metadata; includes RNG reset, validity checks, decode, worker lifecycle and copies; CUDA sync before/after",
              "order_note": "One warmup per variant, then3 interleaved rounds with reversed/rotated order; no cache clearing",
              "comparison_scope": "All result keys/dtypes/shapes/array values checked outside request timing; depth drift reported rather than concealed",
              "environment": {"torch": torch.__version__, "numpy": np.__version__, "cpu_threads": 8,
                              "device": torch.cuda.get_device_name(0), "matmul_precision": "highest"}}
    variants = {
        "native": ("native", {}),
        "prefetch": ("probe", {}),
        "reduced_serial": ("probe", {"reduced": True, "prefetch": False}),
        "reduced_prefetch": ("probe", {"reduced": True}),
        "delayed_d2h": ("probe", {"prefetch": False, "delayed_d2h": True}),
        "combined_delayed": ("probe", {"reduced": True, "delayed_d2h": True}),
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def predict(name, model):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        kind, flags = variants[name]
        kwargs = dict(batch_size=2, precision="fp16", max_size=1008)
        kwargs.update(flags)
        result = (predict_promptda if kind == "native" else predict_prefetch)(model, images, sensor, "cuda:0", **kwargs)
        if len(result["depth"]) != len(images) or not np.all(np.isfinite(result["depth"]) & (result["depth"] > 0)):
            raise ValueError("Predicted depths must be finite, positive and frame aligned")
        return result

    def measured(name, model, phase):
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        sampler.set_phase(phase)
        started = time.perf_counter()
        result = predict(name, model)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        sampler.sample()
        row = {"variant": name, "phase": phase, "wall_s": elapsed,
               "start_monotonic": started, "end_monotonic": started + elapsed,
               "allocated_baseline_bytes": baseline_allocated, "reserved_baseline_bytes": baseline_reserved,
               "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
               "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
               "host_rss": sampler.phase_stats(phase)}
        report["requests"].append(row)
        print(f"{phase} {name}: {elapsed:.6f}s", flush=True)
        return result

    def compare(result, reference, label):
        if result.keys() != reference.keys():
            raise AssertionError("Output keys differ")
        equal = []
        mismatched = []
        for key in reference:
            left, right = reference[key], result[key]
            if left.dtype != right.dtype or left.shape != right.shape:
                raise AssertionError(f"Output shape/dtype differs: {key}")
            (equal if np.array_equal(left, right) else mismatched).append(key)
        delta = np.abs(result["depth"] - reference["depth"])
        row = {"label": label, "variant": report["requests"][-1]["variant"],
               "exact_fields": equal, "mismatched_fields": mismatched,
               "depth_differing_pixels": int(np.count_nonzero(delta)),
               "depth_max_abs_m": float(delta.max(initial=0)), "depth_mean_abs_m": float(delta.mean()),
               "all_fields_exact": not mismatched}
        report["comparisons"].append(row)
        if any(key != "depth" for key in mismatched):
            raise AssertionError(f"Non-depth result fields differ: {mismatched}")

    try:
        sampler.set_phase("model_load_native")
        started = time.perf_counter()
        model = load_candidate("promptda", args.checkpoint, "cuda:0", args.repo)
        torch.cuda.synchronize()
        report["native_model_load_wall_s"] = time.perf_counter() - started
        reference = measured("native", model, "cold_reference")
        for name in list(variants)[1:]:
            result = measured(name, model, "warmup_" + name)
            compare(result, reference, "warmup_" + name)
            del result
            save()
        names = list(variants)
        orders = [names, list(reversed(names)), names[3:] + names[:3]]
        report["timed_orders"] = orders
        for repeat, order in enumerate(orders):
            for name in order:
                result = measured(name, model, f"timed_{repeat}_{name}")
                compare(result, reference, f"timed_{repeat}_{name}")
                del result
                save()
        report["timing_summary"] = {}
        for name in variants:
            rows = [row for row in report["requests"] if row["variant"] == name and row["phase"].startswith("timed_")]
            values = [row["wall_s"] for row in rows]
            report["timing_summary"][name] = {"n": len(values), "mean_s": statistics.mean(values),
                "median_s": statistics.median(values), "min_s": min(values), "max_s": max(values),
                "sample_sd_s": statistics.stdev(values),
                "allocated_peak_bytes": max(row["allocated_peak_bytes"] for row in rows),
                "reserved_peak_bytes": max(row["reserved_peak_bytes"] for row in rows)}
        # The loader candidate is checked only after CPU equality gate was run.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        from load_probe import load_promptda_model
        sampler.set_phase("model_load_skip")
        started = time.perf_counter()
        model, load_metadata = load_promptda_model(args.repo, args.checkpoint, skip=True)
        # Production Scanner uses this aminmax normalization override.
        model.normalize = types.MethodType(lambda self, prompt: normalize_prompt_depth(prompt), model)
        model = model.to("cuda:0").eval()
        torch.cuda.synchronize()
        report["skip_model_load_wall_s"] = time.perf_counter() - started
        report["skip_audit"] = load_metadata
        result = measured("native", model, "skip_loader_request")
        compare(result, reference, "skip_loader_request")
        del result, model, reference
        report["equivalence"] = {
            name: all(row["all_fields_exact"] for row in report["comparisons"]
                      if row["variant"] == name and row["label"] != "skip_loader_request")
            for name in variants
        }
        report["equivalence"]["skip_loader"] = next(
            row["all_fields_exact"] for row in report["comparisons"]
            if row["label"] == "skip_loader_request")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        sampler.stop()
        report["rss"] = sampler.summary()
        save()


if __name__ == "__main__":
    main()
