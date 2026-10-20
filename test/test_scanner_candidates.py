"""CPU checks for Scanner alignment, metric-depth policy and point geometry."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/3d/scanner"
sys.path.insert(0, str(SCRIPT_DIR))
import candidates  # noqa: E402 - import the standalone script directory above
import prepare_scanner  # noqa: E402
import scanner_io  # noqa: E402


class ScannerCandidatesTests(unittest.TestCase):
    def test_promptda_rejects_invalid_execution_settings(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            candidates.predict_promptda(None, [], {}, "cpu", batch_size=0)
        with self.assertRaisesRegex(ValueError, "precision"):
            candidates.predict_promptda(None, [], {}, "cpu", precision="unknown")
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            candidates.predict_promptda(None, [], {}, "cpu", precision="fp16")

    def test_promptda_extrema_normalization_matches_official_quantiles(self):
        import torch

        prompt = torch.from_numpy(
            np.random.default_rng(5).uniform(0, 8, (3, 1, 29, 37)).astype(np.float32)
        )
        prompt[0, 0, 0, 0] = 0
        low = torch.quantile(prompt.flatten(1), 0.0, dim=1, keepdim=True)[
            :, :, None, None
        ]
        high = torch.quantile(prompt.flatten(1), 1.0, dim=1, keepdim=True)[
            :, :, None, None
        ]
        normalized, minimum, maximum = candidates.normalize_prompt_depth(prompt)
        torch.testing.assert_close(minimum, low, rtol=0, atol=0)
        torch.testing.assert_close(maximum, high, rtol=0, atol=0)
        torch.testing.assert_close(
            normalized, (prompt - low) / (high - low), rtol=0, atol=0
        )

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
        from PIL import Image

        prompts = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
        calls = []

        def predict(image, prompt):
            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(prompt.dtype, torch.float32)
            calls.append(prompt.numpy().copy())
            return torch.full((1, 1, 28, 28), 2.5)

        sensor = dict(
            lidar_depth=prompts,
            lidar_confidence=np.zeros_like(prompts),
            intrinsics=np.repeat(np.eye(3)[None], 2, axis=0),
            image_size=[28, 28],
            w2c=np.repeat(np.eye(4)[None], 2, axis=0),
            frame_ids=np.array([0, 1]),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = [str(Path(directory) / f"{i}.png") for i in range(2)]
            for path in paths:
                Image.fromarray(np.full((28, 28, 3), 127, np.uint8)).save(path)
            result = candidates.predict_promptda(
                SimpleNamespace(predict=predict),
                paths,
                sensor,
                "cpu",
                batch_size=1,
                precision="fp32",
            )
        np.testing.assert_array_equal(np.concatenate(calls)[:, 0], prompts)
        np.testing.assert_array_equal(result["depth"], np.full((2, 28, 28), 2.5))

    def test_promptda_native_padding_keeps_pixels_cameras_and_batch_order(self):
        import torch
        from PIL import Image

        rgb = np.random.default_rng(4).integers(0, 256, (3, 29, 37, 3), dtype=np.uint8)
        prompt = np.arange(36, dtype=np.float32).reshape(3, 3, 4)
        k = np.repeat(
            np.array([[[18, 0, 17], [0, 19, 13], [0, 0, 1]]], np.float32), 3, axis=0
        )
        sensor = dict(
            lidar_depth=prompt,
            intrinsics=k,
            image_size=[37, 29],
            w2c=np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0),
            frame_ids=np.array([0, 59, 119]),
        )
        seen = []

        def predict(image, depth_input):
            self.assertEqual(tuple(image.shape[1:]), (3, 42, 42))
            self.assertEqual(tuple(depth_input.shape[1:]), (1, 42, 42))
            for pixels, supplied in zip(image, depth_input, strict=True):
                index = len(seen)
                restored = torch.round(pixels * 255).byte().permute(1, 2, 0).numpy()
                np.testing.assert_array_equal(restored[:29, :37], rgb[index])
                np.testing.assert_array_equal(
                    restored[:29, 37:], np.repeat(rgb[index, :, -1:, :], 5, axis=1)
                )
                np.testing.assert_array_equal(
                    restored[29:, :37], np.repeat(rgb[index, -1:, :, :], 13, axis=0)
                )
                np.testing.assert_array_equal(
                    np.unique(supplied.numpy()), np.unique(prompt[index])
                )
                seen.append(index)
            # Values are unique across rows/columns, including the padded border.
            grid = torch.arange(42 * 42, dtype=torch.float32).reshape(1, 1, 42, 42) + 1
            return grid.repeat(len(image), 1, 1, 1)

        with tempfile.TemporaryDirectory() as directory:
            paths = [str(Path(directory) / f"{i}.png") for i in range(3)]
            for path, pixels in zip(paths, rgb, strict=True):
                Image.fromarray(pixels).save(path)
            result = candidates.predict_promptda(
                SimpleNamespace(predict=predict),
                paths,
                sensor,
                "cpu",
                batch_size=2,
                precision="fp32",
                max_size=0,
            )
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(result["depth"].shape, (3, 29, 37))
        self.assertEqual(result["depth"][0, -1, -1], 28 * 42 + 37)
        np.testing.assert_array_equal(result["processed_images"], rgb)
        np.testing.assert_array_equal(result["intrinsics"], k)
        np.testing.assert_array_equal(result["network_image_sizes"], [[42, 42]] * 3)
        self.assertEqual(result["preprocess_method"].item(), "promptda_native_pad14")

    def test_promptda_1008_uses_resized_float_rgb_native_lidar_and_scaled_K(self):
        import cv2
        import torch
        from PIL import Image

        y, x = np.indices((1440, 1920), dtype=np.uint16)
        rgb = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
        prompt = np.array([[[0, 1], [2, 3]]], dtype=np.float32)
        k = np.array([[[1800, 0, 959.5], [0, 1800, 719.5], [0, 0, 1]]], np.float32)
        sensor = dict(
            lidar_depth=prompt,
            image_size=[1920, 1440],
            intrinsics=k,
            w2c=np.eye(4, dtype=np.float32)[None],
            frame_ids=np.array([59]),
        )
        expected = cv2.resize(
            rgb.astype(np.float32) / 255.0, (1008, 756), interpolation=cv2.INTER_AREA
        )

        def predict(image, supplied):
            self.assertEqual(tuple(image.shape), (1, 3, 756, 1008))
            actual = image[0].permute(1, 2, 0).numpy()
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(np.any(actual != np.rint(actual * 255) / 255))
            np.testing.assert_array_equal(supplied.numpy()[:, 0], prompt)
            return torch.full((1, 1, 756, 1008), 2.5)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000059.png"
            Image.fromarray(rgb).save(path)
            result = candidates.predict_promptda(
                SimpleNamespace(predict=predict),
                [str(path)],
                sensor,
                "cpu",
                precision="fp32",
            )
        self.assertEqual(result["depth"].shape, (1, 756, 1008))
        np.testing.assert_allclose(
            result["intrinsics"], [[[945, 0, 503.5], [0, 945, 377.5], [0, 0, 1]]]
        )
        np.testing.assert_array_equal(result["extrinsics"], sensor["w2c"][:, :3])
        np.testing.assert_array_equal(result["frame_ids"], [59])
        np.testing.assert_array_equal(
            result["processed_images"][0],
            np.rint(expected * 255).clip(0, 255).astype(np.uint8),
        )
        np.testing.assert_array_equal(result["network_image_sizes"], [[1008, 756]])
        self.assertEqual(result["preprocess_method"].item(), "promptda_resize")
        self.assertEqual(result["preprocess_resolution"].item(), 1008)

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
