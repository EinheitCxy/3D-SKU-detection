"""CPU checks for the phone bundle -> existing reconstruction input boundary."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/3d/scanner"
sys.path.insert(0, str(SCRIPT_DIR))
import prepare_bundle  # noqa: E402
import scanner_io  # noqa: E402


class ScannerBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "scan.rgbd"
        self.bundle.mkdir()
        # Independent wire-format example, rather than importing the reader's schema.
        self.manifest = {
            "format": "sku-rgbd",
            "version": 1,
            "id": "2C9F61BE-9401-4685-BAB8-21A2C22BFB53",
            "status": "ready",
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
            "source_frame_count": 200,
            "rejected_candidate_frames": {"missing_depth_or_confidence": 3},
            "errors": [],
            "frames": [],
        }
        self.rgb = np.zeros((6, 8, 4), np.uint8)
        self.rgb[..., :3] = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        self.rgb[..., 3] = 255  # CIContext's opaque RGBA output.
        self.depth = np.array(
            [[0, 500, 1000, 1500], [2000, 2500, 3000, 3500], [4000, 4500, 5000, 5500]],
            dtype=np.uint16,
        )
        self.confidence = np.tile(np.array([[0], [1], [2]], np.uint8), (1, 4))
        self.k = np.array([[8, 0, 3], [0, 9, 2], [0, 0, 1]], np.float32)
        # Rz(90deg) @ Rx(pi): OpenCV camera -> unchanged ARKit world.
        self.pose = np.array(
            [[0, 1, 0, 2], [1, 0, 0, 3], [0, 0, -1, 4], [0, 0, 0, 1]], np.float32
        )
        for frame_id, elapsed in [(0, 0.0), (61, 1.02), (182, 3.04)]:
            folder = self.bundle / "frames" / f"{frame_id:06d}"
            folder.mkdir(parents=True)
            record = {
                "frame_id": frame_id,
                "timestamp_seconds": 100 + elapsed,
                "relative_timestamp_seconds": elapsed,
                "target_seconds": int(elapsed),
                "rgb_size": [8, 6],
                "depth_size": [4, 3],
                "intrinsics": self.k.tolist(),
                "camera_to_world": self.pose.tolist(),
                "tracking_state": "normal",
            }
            self.manifest["frames"].append(record)
            for name, image in (
                ("rgb", self.rgb),
                ("depth", self.depth),
                ("confidence", self.confidence),
            ):
                self.assertTrue(cv2.imwrite(str(folder / f"{name}.png"), image))
        self.save_metadata()

    def save_metadata(self):
        (self.bundle / "scan_bundle.json").write_text(json.dumps(self.manifest))
        for record in self.manifest["frames"]:
            path = self.bundle / "frames" / f"{record['frame_id']:06d}" / "frame.json"
            path.write_text(json.dumps(record))

    def test_direct_frames_preserve_pixels_units_cameras_and_gaps(self):
        output = self.root / "prepared"
        prepare_bundle.prepare_bundle(self.bundle, output)
        images, sensor = scanner_io.load_prepared(output)
        np.testing.assert_array_equal(sensor["frame_ids"], [0, 61, 182])
        np.testing.assert_array_equal(sensor["target_seconds"], [0, 1, 3])
        np.testing.assert_array_equal(sensor["image_size"], [8, 6])
        np.testing.assert_array_equal(
            sensor["intrinsics"], np.repeat(self.k[None], 3, 0)
        )
        np.testing.assert_array_equal(sensor["c2w"], np.repeat(self.pose[None], 3, 0))
        np.testing.assert_array_equal(
            sensor["lidar_depth"][0], self.depth / np.float32(1000)
        )
        np.testing.assert_array_equal(sensor["lidar_confidence"][0], self.confidence)
        for image in images:
            np.testing.assert_array_equal(
                cv2.imread(image, cv2.IMREAD_UNCHANGED), self.rgb
            )
        # A two-meter point on the optical axis is (2,3,2) in this ARKit world.
        camera_point = np.array([0, 0, 2, 1], np.float32)
        world = sensor["c2w"][0] @ camera_point
        np.testing.assert_array_equal(world, [2, 3, 2, 1])
        np.testing.assert_array_equal(sensor["w2c"][0] @ world, camera_point)

    def test_incomplete_or_ambiguous_bundles_never_publish(self):
        changes = [
            ("status", "recording"),
            ("errors", ["disk full"]),
            ("version", 2),
            ("rgb_orientation", "portrait_rotated"),
            ("pose_convention", "world_to_camera"),
            ("depth_encoding", "float32_meters"),
        ]
        for key, invalid in changes:
            with self.subTest(field=key):
                original = self.manifest[key]
                self.manifest[key] = invalid
                self.save_metadata()
                output = self.root / "invalid"
                with self.assertRaises(ValueError):
                    prepare_bundle.prepare_bundle(self.bundle, output)
                self.assertFalse(output.exists())
                self.manifest[key] = original
        self.save_metadata()

    def test_missing_member_and_metadata_mismatch_are_errors(self):
        folder = self.bundle / "frames/000061"
        (folder / "confidence.png").unlink()
        with self.assertRaisesRegex(ValueError, "confidence"):
            prepare_bundle.read_bundle(self.bundle)
        cv2.imwrite(str(folder / "confidence.png"), self.confidence)
        (folder / "frame.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            prepare_bundle.read_bundle(self.bundle)

    def test_sensor_values_and_shape_are_not_silently_repaired(self):
        path = self.bundle / "frames/000061/confidence.png"
        for invalid in (np.full((3, 4), 255, np.uint8), np.zeros((2, 2), np.uint8)):
            with self.subTest(shape=invalid.shape):
                cv2.imwrite(str(path), invalid)
                with self.assertRaisesRegex(ValueError, "confidence"):
                    prepare_bundle.read_bundle(self.bundle)
        cv2.imwrite(str(path), self.confidence)
        depth_path = self.bundle / "frames/000061/depth.png"
        cv2.imwrite(str(depth_path), self.depth.astype(np.uint8))
        with self.assertRaisesRegex(ValueError, "uint16"):
            prepare_bundle.read_bundle(self.bundle)

    def test_transposed_or_nonrigid_pose_is_rejected(self):
        for invalid in (self.pose.T, self.pose * 2):
            with self.subTest(pose=invalid.tolist()):
                self.manifest["frames"][1]["camera_to_world"] = invalid.tolist()
                self.save_metadata()
                with self.assertRaisesRegex(ValueError, "row-major"):
                    prepare_bundle.read_bundle(self.bundle)

    def test_timestamp_or_second_mismatch_is_rejected(self):
        record = self.manifest["frames"][1]
        original = dict(record)
        for key, value in (
            ("target_seconds", 0),
            ("target_seconds", 2),
            ("timestamp_seconds", 99),
            ("relative_timestamp_seconds", 1.7),
        ):
            with self.subTest(field=key, value=value):
                record.update(original)
                record[key] = value
                self.save_metadata()
                with self.assertRaisesRegex(ValueError, "timestamps"):
                    prepare_bundle.read_bundle(self.bundle)

    def test_existing_output_is_preserved(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "camera_inputs.npz"
        marker.write_bytes(b"existing user input")
        with self.assertRaises(FileExistsError):
            prepare_bundle.prepare_bundle(self.bundle, output)
        self.assertEqual(marker.read_bytes(), b"existing user input")


if __name__ == "__main__":
    unittest.main()
