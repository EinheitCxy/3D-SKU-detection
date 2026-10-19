#!/usr/bin/env python3
"""Prepare synchronized 1 Hz Stray Scanner RGB, per-frame K/E and metric depth."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def read_selected(input_dir: Path):
    with (input_dir / "odometry.csv").open(newline="") as handle:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle, skipinitialspace=True)
        ]
    if not rows or [int(row["frame"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Expected contiguous Scanner frame IDs beginning at zero")
    timestamps = [float(row["timestamp"]) for row in rows]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Odometry timestamps must increase")
    times = [timestamp - timestamps[0] for timestamp in timestamps]
    selected = []
    for target in range(int(times[-1]) + 1):
        index = min(range(len(rows)), key=lambda i: abs(times[i] - target))
        selected.append((target, times[index], rows[index]))
    if len({int(row["frame"]) for _, _, row in selected}) != len(selected):
        raise ValueError("Sampling selected a frame more than once")
    return rows, selected


def ffmpeg_command(input_dir: Path, output_dir: Path, selected):
    select_filter = "select=" + "+".join(
        f"eq(n\\,{int(row['frame'])})" for _, _, row in selected
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-ignore_editlist",
        "1",
        "-noautorotate",
        "-i",
        str(input_dir / "rgb.mp4"),
        "-map",
        "0:v:0",
        "-vf",
        select_filter,
        "-fps_mode",
        "vfr",
        "-start_number",
        "0",
        str(output_dir / "frames" / ".selected_%06d.png"),
    ]


def extract_frames(input_dir, output_dir, selected):
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    if any(frames_dir.iterdir()):
        raise FileExistsError(
            f"Extraction needs an empty frames directory: {frames_dir}"
        )
    subprocess.run(ffmpeg_command(input_dir, output_dir, selected), check=True)
    extracted = sorted(frames_dir.glob(".selected_*.png"))
    if len(extracted) != len(selected):
        raise ValueError(
            f"Expected {len(selected)} extracted frames, got {len(extracted)}"
        )
    for index, (_, _, row) in enumerate(selected):
        (frames_dir / f".selected_{index:06d}.png").rename(
            frames_dir / f"{int(row['frame']):06d}.png"
        )


def camera_from_row(row):
    import numpy as np

    quaternion = np.array([float(row[key]) for key in ("qx", "qy", "qz", "qw")])
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid Scanner quaternion")
    x, y, z, w = quaternion / norm
    c2w = np.eye(4, dtype=np.float32)
    # Scanner already encodes OpenCV camera -> world. No second Rx(pi).
    c2w[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    c2w[:3, 3] = [float(row[key]) for key in ("x", "y", "z")]
    intrinsic = np.array(
        [
            [float(row["fx"]), 0, float(row["cx"])],
            [0, float(row["fy"]), float(row["cy"])],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return intrinsic, c2w


def prepare_metadata(input_dir, output_dir, rows, selected):
    import cv2
    import numpy as np

    from scanner_io import save_json

    path = output_dir / "camera_inputs.npz"
    if path.exists():
        raise FileExistsError(path)
    intrinsics, c2ws, depths, confidences = [], [], [], []
    image_size, depth_shape = None, None
    for _, _, row in selected:
        filename = f"{int(row['frame']):06d}.png"
        rgb = cv2.imread(str(output_dir / "frames" / filename), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(input_dir / "depth" / filename), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(
            str(input_dir / "confidence" / filename), cv2.IMREAD_UNCHANGED
        )
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Invalid unrotated RGB: {filename}")
        if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
            raise ValueError(f"Expected native uint16 millimeter depth: {filename}")
        if (
            confidence is None
            or confidence.shape != depth.shape
            or confidence.dtype != np.uint8
        ):
            raise ValueError(f"Invalid LiDAR confidence: {filename}")
        if not np.isin(confidence, [0, 1, 2]).all():
            raise ValueError(f"Unexpected LiDAR confidence values: {filename}")
        size = (rgb.shape[1], rgb.shape[0])
        if image_size is None:
            image_size, depth_shape = size, depth.shape
        if size != image_size or depth.shape != depth_shape:
            raise ValueError("All sampled frames must share RGB/depth dimensions")
        k, c2w = camera_from_row(row)
        intrinsics.append(k)
        c2ws.append(c2w)
        depths.append(depth.astype(np.float32) / 1000)
        confidences.append(confidence)
    c2w = np.stack(c2ws)
    arrays = {
        "frame_ids": np.array(
            [int(row["frame"]) for _, _, row in selected], dtype=np.int64
        ),
        "target_seconds": np.array(
            [target for target, _, _ in selected], dtype=np.float64
        ),
        "timestamps_relative": np.array(
            [time for _, time, _ in selected], dtype=np.float64
        ),
        "intrinsics": np.stack(intrinsics),
        "c2w": c2w,
        "w2c": np.linalg.inv(c2w).astype(np.float32),
        "image_size": np.array(image_size, dtype=np.int64),
        "lidar_depth": np.stack(depths),
        "lidar_confidence": np.stack(confidences),
    }
    np.savez(path, **arrays)
    save_json(
        output_dir / "frames.json",
        {
            "source_frame_count": len(rows),
            "frame_ids": arrays["frame_ids"].tolist(),
            "sampling": "Nearest odometry timestamp to each integer second since first frame",
            "rgb_decode": "Original frame ordinal, ignore_editlist=1, noautorotate",
            "coordinates": "CSV qxyzw is OpenCV camera to world; w2c=inverse(c2w)",
            "depth_units": "meters; native uint16 PNG millimeters / 1000; zero invalid",
            "source_format": "https://github.com/strayrobots/scanner/blob/main/docs/format.md",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument(
        "--extract-only",
        action="store_true",
        help="ffmpeg only; no Python ML dependencies",
    )
    phase.add_argument(
        "--metadata-only",
        action="store_true",
        help="Use already extracted raw-orientation PNGs",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, selected = read_selected(args.input_dir)
    if not args.metadata_only:
        extract_frames(args.input_dir, args.output_dir, selected)
    if not args.extract_only:
        prepare_metadata(args.input_dir, args.output_dir, rows, selected)


if __name__ == "__main__":
    main()
