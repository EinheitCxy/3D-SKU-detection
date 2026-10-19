"""CPU checks for Scanner alignment, metric-depth policy and point geometry."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/3d/scanner"
sys.path.insert(0, str(SCRIPT_DIR))
import candidates  # noqa: E402 - import the standalone script directory above
import prepare_scanner  # noqa: E402
import scanner_io  # noqa: E402


class ScannerCandidatesTests(unittest.TestCase):
    def test_sampling_uses_timestamps_and_keeps_frame_ordinals(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            times = [10, 10.51, 11.02, 11.49, 12.02]
            text = "timestamp, frame, distortion_center_x, distortion_center_y\n"
            text += "".join(
                f"{time}, {index:06d}, , \n" for index, time in enumerate(times)
            )
            (base / "odometry.csv").write_text(text)
            rows, selected = prepare_scanner.read_selected(base)
            self.assertEqual([int(row["frame"]) for _, _, row in selected], [0, 2, 4])
            self.assertEqual(len(rows), 5)
            command = prepare_scanner.ffmpeg_command(base, base / "prepared", selected)
            self.assertEqual(command[command.index("-ignore_editlist") + 1], "1")
            self.assertIn("-noautorotate", command)
            self.assertIn("select=eq(n\\,0)+eq(n\\,2)+eq(n\\,4)", command)

    def test_camera_is_already_cv_c2w(self):
        row = dict(
            qx="0",
            qy="0",
            qz="0",
            qw="1",
            x="2",
            y="3",
            z="4",
            fx="8",
            fy="9",
            cx="3",
            cy="2",
        )
        k, pose = prepare_scanner.camera_from_row(row)
        np.testing.assert_array_equal(pose[:3, :3], np.eye(3))
        np.testing.assert_array_equal(pose[:3, 3], [2, 3, 4])
        np.testing.assert_array_equal(k, [[8, 0, 3], [0, 9, 2], [0, 0, 1]])

    def test_metric_policy_preserves_depth_and_restores_supplied_E(self):
        source_k = np.array([[[8, 0, 3], [0, 8, 2], [0, 0, 1]]], dtype=np.float32)
        scanner_e = np.eye(4, dtype=np.float32)[None]
        scanner_e[0, 0, 3] = 0.25
        depth = np.array([[[2.0, 2.5]]], dtype=np.float32)
        processed_k = source_k * 0.5
        conf = np.array([[[2.0, 3.0]]], dtype=np.float32)
        calls = []

        def inference(images, **kwargs):
            calls.append(kwargs)
            # Returned poses deliberately disagree with Scanner E.
            return SimpleNamespace(
                depth=depth,
                conf=conf,
                intrinsics=processed_k,
                extrinsics=np.zeros((1, 3, 4)),
                is_metric=1,
                scale_factor=2.72,
                processed_images=np.zeros((1, 1, 2, 3), np.uint8),
            )

        sensor = dict(intrinsics=source_k, w2c=scanner_e, frame_ids=np.array([0]))
        result = candidates.predict_da3(
            SimpleNamespace(inference=inference), ["frame.png"], sensor
        )
        self.assertFalse(calls[0]["align_to_input_ext_scale"])
        self.assertIs(result["depth"], depth)
        self.assertIs(result["intrinsics"], processed_k)
        self.assertIs(result["conf"], conf)
        np.testing.assert_array_equal(result["extrinsics"], scanner_e[:, :3])
        self.assertFalse(np.shares_memory(result["extrinsics"], scanner_e))
        self.assertFalse(np.shares_memory(calls[0]["extrinsics"], scanner_e))

    def test_nonmetric_da3_is_rejected_without_scale_guess(self):
        model = SimpleNamespace(
            inference=lambda *args, **kwargs: SimpleNamespace(is_metric=0)
        )
        sensor = dict(intrinsics=np.eye(3)[None], w2c=np.eye(4)[None])
        with self.assertRaisesRegex(ValueError, "native metric depth"):
            candidates.predict_da3(model, ["frame.png"], sensor)

    def test_promptda_keeps_full_native_prompt_at_all_confidences(self):
        import torch

        prompts = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
        calls = []

        def predict(image, prompt):
            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(prompt.dtype, torch.float32)
            calls.append(prompt.numpy().copy())
            return torch.full((1, 1, 28, 28), 2.5)

        loader = SimpleNamespace(
            load_image=lambda *args, **kwargs: torch.full((1, 3, 28, 28), 0.5)
        )
        sensor = dict(
            lidar_depth=prompts,
            lidar_confidence=np.zeros_like(prompts),
            intrinsics=np.repeat(np.eye(3)[None], 2, axis=0),
            image_size=[28, 28],
            w2c=np.repeat(np.eye(4)[None], 2, axis=0),
            frame_ids=np.array([0, 1]),
        )
        with patch.dict(sys.modules, {"promptda.utils.io_wrapper": loader}):
            result = candidates.predict_promptda(
                SimpleNamespace(predict=predict), ["a", "b"], sensor, "cpu"
            )
        np.testing.assert_array_equal(np.concatenate(calls)[:, 0], prompts)
        np.testing.assert_array_equal(result["depth"], np.full((2, 28, 28), 2.5))

    def test_resize_K_preserves_half_pixel_centers(self):
        source = np.array([[[4, 0, 3], [0, 6, 2], [0, 0, 1]]], dtype=np.float32)
        resized = scanner_io.resize_intrinsics(source, (8, 6), (4, 3))
        np.testing.assert_allclose(resized[0], [[2, 0, 1.25], [0, 3, 0.75], [0, 0, 1]])

    def test_known_geometry_survives_GLB_boundary(self):
        depth = np.array([[[2.0]]], dtype=np.float32)
        k = np.eye(3, dtype=np.float32)[None]
        c2w = np.eye(4, dtype=np.float32)[None]
        c2w[0, 0, 3] = 1
        colors = np.array([[[[10, 20, 30]]]], dtype=np.uint8)
        points, rgb = scanner_io.backproject(depth, k, c2w, colors, depth > 0)
        np.testing.assert_array_equal(points, [[1, 0, 2]])
        np.testing.assert_array_equal(rgb, [[10, 20, 30]])
        prediction = dict(
            depth=depth, intrinsics=k, processed_images=colors, frame_ids=np.array([0])
        )
        sensor = dict(w2c=np.linalg.inv(c2w), frame_ids=np.array([0]))
        with tempfile.TemporaryDirectory() as directory:
            record = scanner_io.export_prediction(
                prediction, sensor, Path(directory) / "cloud.glb"
            )
        self.assertEqual(record["glb_readback"], "PASS")
        self.assertEqual(record["points"], 1)
        np.testing.assert_allclose(record["bounds_m"], [[0, 0, -2], [0, 0, -2]])


if __name__ == "__main__":
    unittest.main()
