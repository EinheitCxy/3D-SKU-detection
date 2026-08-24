#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def extract_with_ffmpeg(video_path: Path, output_dir: Path) -> None:
    pattern = output_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        "fps=1",
        "-q:v",
        "2",
        "-start_number",
        "0",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)


def extract_with_cv2(video_path: Path, output_dir: Path) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video FPS.")

    total_seconds = int(frame_count / fps)
    for sec in range(total_seconds + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(output_dir / f"frame_{sec:06d}.jpg"), frame)

    cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract one JPG frame per second from a video.")
    parser.add_argument("-video", help="Path to the input video")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not video_path.is_file():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg"):
        extract_with_ffmpeg(video_path, output_dir)
        return 0

    try:
        extract_with_cv2(video_path, output_dir)
        return 0
    except ModuleNotFoundError:
        print("Need either ffmpeg or opencv-python. Install one of them first.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
