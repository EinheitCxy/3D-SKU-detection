"""Matched native/experimental cold delivery, fixed real-input domain.

Optimized sampling is explicitly restricted to inputs with a sufficient finite
domain certificate; it is not a production replacement for unknown domains.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import types

SCRIPT_STARTED = time.perf_counter()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/3d/scanner"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "optimized"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from candidates import load_candidate, normalize_prompt_depth, predict_candidate
    from scanner_io import export_prediction, load_prepared
    from load_probe import load_promptda_model
    from request_probe import predict_prefetch
    from sampling_probe import certify_finite_domain, select_before_backprojection, export_candidate_once
    import random

    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "result.json"
    if report_path.exists() or (args.output / "predictions.npz").exists():
        raise FileExistsError(args.output)
    report = {"status": "running", "mode": args.mode, "stages": {},
              "scope": "Matched fresh-process prepared PNGs to complete dense stored NPZ and colored POINTS GLB, including original format readback",
              "optimized_scope": "selective initialization + pure bounded prefetch + same-index selected projection on the mathematically certified real input only"}
    try:
        begin = time.perf_counter()
        images, sensor = load_prepared(args.base)
        report["frames"] = len(images)
        report["stages"]["prepared_load_s"] = time.perf_counter() - begin
        begin = time.perf_counter()
        if args.mode == "native":
            model = load_candidate("promptda", args.checkpoint, "cuda:0", args.repo)
        else:
            model, audit = load_promptda_model(args.repo, args.checkpoint, skip=True)
            report["initialization_audit"] = audit
            model.normalize = types.MethodType(lambda self, prompt: normalize_prompt_depth(prompt), model)
            model = model.to("cuda:0").eval()
        torch.cuda.synchronize()
        report["stages"]["model_load_s"] = time.perf_counter() - begin
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        if args.mode == "native":
            prediction = predict_candidate("promptda", model, images, sensor, "cuda:0",
                promptda_batch_size=2, promptda_precision="fp16", promptda_max_size=1008)
        else:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            prediction = predict_prefetch(model, images, sensor, "cuda:0", batch_size=2, precision="fp16", max_size=1008)
            if len(prediction["depth"]) != len(images) or not np.all(np.isfinite(prediction["depth"]) & (prediction["depth"] > 0)):
                raise ValueError("Predicted depths must be finite, positive and frame aligned")
        torch.cuda.synchronize()
        report["stages"]["predict_s"] = time.perf_counter() - begin
        report["request_allocated_peak_bytes"] = torch.cuda.max_memory_allocated()
        report["request_reserved_peak_bytes"] = torch.cuda.max_memory_reserved()
        begin = time.perf_counter()
        np.savez(args.output / "predictions.npz", **prediction)
        report["stages"]["npz_save_s"] = time.perf_counter() - begin
        begin = time.perf_counter()
        if args.mode == "native":
            exported = export_prediction(prediction, sensor, args.output / "cloud.glb")
        else:
            certificate_started = time.perf_counter()
            certificate = certify_finite_domain(prediction, sensor)
            report["finite_certificate_s"] = time.perf_counter() - certificate_started
            report["finite_certificate"] = certificate
            if certificate["status"] != "true":
                raise ValueError("Experimental selected-projection domain was not certified; no alternative path used")
            projection_started = time.perf_counter()
            selected = select_before_backprojection(prediction, sensor)
            report["selected_projection_s"] = time.perf_counter() - projection_started
            exported = export_candidate_once(selected, sensor, args.output / "cloud.glb")
            report["selection"] = selected[3]
        report["stages"]["glb_export_s"] = time.perf_counter() - begin
        report["export"] = exported
        report["total_script_to_glb_s"] = time.perf_counter() - SCRIPT_STARTED
        report["processed_shape"] = list(prediction["depth"].shape)
        report["npz_bytes"] = (args.output / "predictions.npz").stat().st_size
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("mode", "status", "stages", "total_script_to_glb_s")}))


if __name__ == "__main__":
    main()
