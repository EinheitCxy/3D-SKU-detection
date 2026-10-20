#!/usr/bin/env python3
"""Run one selected candidate on prepared Scanner frames and export a GLB."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from candidates import (
    DEFAULT_PROMPTDA_BATCH_SIZE,
    DEFAULT_PROMPTDA_MAX_SIZE,
    DEFAULT_PROMPTDA_PRECISION,
    load_candidate,
    predict_candidate,
)
from scanner_io import export_prediction, load_prepared, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, required=True, help="Prepared frames + camera_inputs.npz"
    )
    parser.add_argument("--candidate", choices=("da3", "promptda"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--promptda-repo", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--promptda-batch-size", type=int, default=DEFAULT_PROMPTDA_BATCH_SIZE
    )
    parser.add_argument(
        "--promptda-max-size",
        type=int,
        choices=(0, 504, 1008),
        default=DEFAULT_PROMPTDA_MAX_SIZE,
        help="RGB longest side; 0 preserves native resolution",
    )
    parser.add_argument(
        "--promptda-precision",
        choices=("fp32", "fp16"),
        default=DEFAULT_PROMPTDA_PRECISION,
    )
    parser.add_argument(
        "--promptda-matmul-precision", choices=("highest", "high"), default="highest"
    )
    parser.add_argument(
        "--compress-prediction",
        action="store_true",
        help="Compress PromptDA NPZ to reduce disk usage; DA3 retains compression",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import torch

    torch.set_num_threads(8)
    started = time.perf_counter()
    images, sensor = load_prepared(args.base)
    model = load_candidate(
        args.candidate,
        args.checkpoint,
        args.device,
        args.promptda_repo,
        promptda_matmul_precision=args.promptda_matmul_precision,
    )
    torch.cuda.synchronize(args.device)
    loaded = time.perf_counter()
    prediction = predict_candidate(
        args.candidate,
        model,
        images,
        sensor,
        args.device,
        promptda_batch_size=args.promptda_batch_size,
        promptda_precision=args.promptda_precision,
        promptda_max_size=args.promptda_max_size,
    )
    torch.cuda.synchronize(args.device)
    predicted = time.perf_counter()
    compress = args.candidate == "da3" or args.compress_prediction
    save = np.savez_compressed if compress else np.savez
    save(args.output / "predictions.npz", **prediction)
    saved = time.perf_counter()
    cloud = export_prediction(prediction, sensor, args.output / "cloud.glb")
    finished = time.perf_counter()
    save_json(
        args.output / "run.json",
        {
            "candidate": args.candidate,
            "promptda_max_size": args.promptda_max_size
            if args.candidate == "promptda"
            else None,
            "npz_compression": "deflate" if compress else "stored",
            "frames": len(images),
            "device": torch.cuda.get_device_name(args.device),
            "processed_shape": list(prediction["depth"].shape),
            "promptda_batch_size": args.promptda_batch_size
            if args.candidate == "promptda"
            else None,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "promptda_precision": args.promptda_precision
            if args.candidate == "promptda"
            else None,
            "input_and_model_load_s": loaded - started,
            "request_s": predicted - loaded,
            "npz_save_s": saved - predicted,
            "cloud_export_s": finished - saved,
            "total_after_imports_s": finished - started,
            "export": cloud,
        },
    )
    print(f"{args.candidate}: {len(images)} frames -> {args.output / 'cloud.glb'}")


if __name__ == "__main__":
    main()
