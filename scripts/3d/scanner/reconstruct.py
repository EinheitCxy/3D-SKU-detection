#!/usr/bin/env python3
"""Run one selected candidate on prepared Scanner frames and export a GLB."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from candidates import load_candidate, predict_candidate
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import torch

    torch.set_num_threads(8)
    started = time.perf_counter()
    images, sensor = load_prepared(args.base)
    model = load_candidate(
        args.candidate, args.checkpoint, args.device, args.promptda_repo
    )
    torch.cuda.synchronize(args.device)
    loaded = time.perf_counter()
    prediction = predict_candidate(args.candidate, model, images, sensor, args.device)
    torch.cuda.synchronize(args.device)
    predicted = time.perf_counter()
    np.savez_compressed(args.output / "predictions.npz", **prediction)
    saved = time.perf_counter()
    cloud = export_prediction(prediction, sensor, args.output / "cloud.glb")
    finished = time.perf_counter()
    save_json(
        args.output / "run.json",
        {
            "candidate": args.candidate,
            "frames": len(images),
            "device": torch.cuda.get_device_name(args.device),
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
