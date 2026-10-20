#!/usr/bin/env python3
"""Convert a completed SKU Scanner RGB-D bundle to existing reconstruction inputs."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scanner_io import save_json

BUNDLE_CONTRACT = {
    "format": "sku-rgbd",
    "version": 1,
    "sample_interval_seconds": 1,
    "sampling_policy": "first_complete_normal_tracking_frame_per_elapsed_second",
    "timestamp_clock": "ARFrame.timestamp_seconds_since_boot",
    "rgb_encoding": "png_srgb8",
    "rgb_orientation": "captured_image_unrotated_unmirrored",
    "depth_encoding": "uint16_png_millimeters_zero_invalid",
    "depth_alignment": "same_normalized_extent_as_rgb",
    "confidence_encoding": "uint8_png_levels_0_1_2",
    "camera_convention": "opencv_x_right_y_down_z_forward",
    "world_convention": "arkit_right_handed_y_up_meters",
    "pose_convention": "camera_to_world",
    "matrix_layout": "row_major",
}


def _integer(value, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _matrix(value, shape, field):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid finite {field} matrix: expected {shape}")
    return result


def read_bundle(input_dir: Path) -> tuple[dict, dict]:
    """Validate the explicit v1 boundary; no video decoding or second sampling."""
    manifest = json.loads((input_dir / "scan_bundle.json").read_text())
    for key, expected in BUNDLE_CONTRACT.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Unsupported bundle {key}: expected {expected!r}")
    if manifest.get("status") != "ready" or manifest.get("errors") != []:
        raise ValueError("Bundle must be ready with no capture errors")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Bundle must contain complete frames")
    source_count = _integer(manifest.get("source_frame_count"), "source_frame_count")
    intrinsics, poses, depths, confidences = [], [], [], []
    ids, timestamps, relative, targets = [], [], [], []
    image_size, depth_size = None, None
    for record in frames:
        frame_id = _integer(record["frame_id"], "frame_id")
        target = _integer(record["target_seconds"], "target_seconds")
        folder = input_dir / "frames" / f"{frame_id:06d}"
        if json.loads((folder / "frame.json").read_text()) != record:
            raise ValueError(f"Frame {frame_id}: metadata differs from bundle manifest")
        if record.get("tracking_state") != "normal":
            raise ValueError(f"Frame {frame_id}: tracking must be normal")
        rgb = cv2.imread(str(folder / "rgb.png"), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(folder / "depth.png"), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(folder / "confidence.png"), cv2.IMREAD_UNCHANGED)
        if (
            rgb is None
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] not in (3, 4)
            or (rgb.shape[2] == 4 and not np.all(rgb[..., 3] == 255))
        ):
            raise ValueError(f"Frame {frame_id}: expected opaque 8-bit RGB PNG")
        if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError(f"Frame {frame_id}: expected uint16 millimeter depth PNG")
        if (
            confidence is None
            or confidence.dtype != np.uint8
            or confidence.shape != depth.shape
            or not np.isin(confidence, [0, 1, 2]).all()
        ):
            raise ValueError(f"Frame {frame_id}: invalid native confidence PNG")
        rgb_wh = [rgb.shape[1], rgb.shape[0]]
        depth_wh = [depth.shape[1], depth.shape[0]]
        if rgb_wh != record["rgb_size"] or depth_wh != record["depth_size"]:
            raise ValueError(f"Frame {frame_id}: image dimensions differ from metadata")
        if rgb_wh[0] * depth_wh[1] != rgb_wh[1] * depth_wh[0]:
            raise ValueError(f"Frame {frame_id}: RGB/depth aspect ratios differ")
        if image_size is None:
            image_size, depth_size = rgb_wh, depth_wh
        if rgb_wh != image_size or depth_wh != depth_size:
            raise ValueError(
                "All frames must have identical RGB and native depth grids"
            )
        k = _matrix(record["intrinsics"], (3, 3), "intrinsics")
        c2w = _matrix(record["camera_to_world"], (4, 4), "camera_to_world")
        if min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError(f"Frame {frame_id}: invalid pinhole intrinsics")
        rotation = c2w[:3, :3]
        if (
            not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0)
        ):
            raise ValueError(
                f"Frame {frame_id}: expected rigid row-major camera_to_world"
            )
        ids.append(frame_id)
        timestamps.append(float(record["timestamp_seconds"]))
        relative.append(float(record["relative_timestamp_seconds"]))
        targets.append(target)
        intrinsics.append(k)
        poses.append(c2w)
        depths.append(depth.astype(np.float32) / 1000)
        confidences.append(confidence)
    times = np.asarray(timestamps, dtype=np.float64)
    rel = np.asarray(relative, dtype=np.float64)
    if (
        ids[-1] >= source_count
        or np.any(np.diff(ids) <= 0)
        or not np.isfinite(times).all()
        or not np.isfinite(rel).all()
        or np.any(times < 0)
        or np.any(rel < 0)
        or np.any(np.diff(times) <= 0)
        or np.any(np.diff(rel) <= 0)
        or np.any(np.diff(targets) <= 0)
        or not np.array_equal(np.floor(rel).astype(np.int64), targets)
        or not np.allclose(times - rel, times[0] - rel[0], rtol=0, atol=1e-5)
    ):
        raise ValueError("Frame IDs, timestamps and elapsed-second targets disagree")
    c2w = np.stack(poses)
    return manifest, {
        "frame_ids": np.asarray(ids, dtype=np.int64),
        "target_seconds": np.asarray(targets, dtype=np.float64),
        "timestamps_relative": rel,
        "timestamps_absolute": times,
        "intrinsics": np.stack(intrinsics),
        "c2w": c2w,
        "w2c": np.linalg.inv(c2w).astype(np.float32),
        "image_size": np.asarray(image_size, dtype=np.int64),
        "lidar_depth": np.stack(depths),
        "lidar_confidence": np.stack(confidences),
    }


def prepare_bundle(input_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Use a new prepared-input directory: {output_dir}")
    manifest, arrays = read_bundle(input_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after every frame and metadata file has been prepared.
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}-"
    ) as temporary:
        staged = Path(temporary) / "prepared"
        (staged / "frames").mkdir(parents=True)
        for frame_id in arrays["frame_ids"]:
            name = f"{int(frame_id):06d}"
            shutil.copyfile(
                input_dir / "frames" / name / "rgb.png",
                staged / "frames" / f"{name}.png",
            )
        np.savez(staged / "camera_inputs.npz", **arrays)
        summary = {
            "source_format": "sku-rgbd/v1",
            "bundle_id": manifest["id"],
            "source_frame_count": manifest["source_frame_count"],
            "frame_ids": arrays["frame_ids"].tolist(),
            "target_seconds": arrays["target_seconds"].tolist(),
            "sampling": manifest["sampling_policy"],
            "rejected_candidate_frames": manifest["rejected_candidate_frames"],
            "rgb_decode": "Direct native PNG; no rotation, scaling or resampling",
            "coordinates": "Bundle already contains OpenCV c2w; w2c=inverse(c2w)",
            "depth_units": "meters; native uint16 PNG millimeters / 1000; zero invalid",
        }
        save_json(staged / "frames.json", summary)
        if output_dir.exists():
            raise FileExistsError(output_dir)
        staged.rename(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Extracted .rgbd directory")
    parser.add_argument("output_dir", type=Path, help="New prepared-input directory")
    args = parser.parse_args()
    result = prepare_bundle(args.input_dir, args.output_dir)
    print(
        f"Prepared {len(result['frame_ids'])} synchronized frames in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
