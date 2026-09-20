"""Pi3X 导出数据复用及失败传播，无需加载模型。"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.pi3_3d_reconstructor import (
    PI33DReconstructor,
    _estimate_intrinsics_from_local_points,
    _save_predictions_npz,
)
from src.pi3x_3d_reconstructor import Pi3X3DReconstructor


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
))])
def test_export_reuses_cpu_storage_and_preserves_cache_and_glb(monkeypatch, tmp_path, device):
    recon = object.__new__(Pi3X3DReconstructor)
    images = torch.linspace(0, 1, 2 * 3 * 4 * 5, device=device).reshape(2, 3, 4, 5)
    local = torch.arange(120, dtype=torch.float32, device=device).reshape(1, 2, 4, 5, 3)
    conf = torch.ones((1, 2, 4, 5, 1), device=device)
    pred = {
        "points": local + 1,
        "local_points": local,
        "conf": conf,
        "depth": local[..., 2:3],
        "depth_conf": conf[..., 0],
        "world_points_conf": conf[..., 0],
        "images": images[None].permute(0, 1, 3, 4, 2),
        "camera_poses": torch.eye(4, device=device).repeat(1, 2, 1, 1),
        "extrinsic": torch.eye(4, device=device).repeat(1, 2, 1, 1),
        "intrinsic": torch.eye(3, device=device).repeat(1, 2, 1, 1),
    }
    _save_predictions_npz(pred, images, tmp_path / "before.npz", image_ids=[7, 2], source_model="pi3x")
    original_cpu = torch.Tensor.cpu
    transfers = []

    def track_cpu(tensor, *args, **kwargs):
        if tensor.device.type == "cuda":
            transfers.append(tensor.numel())
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", track_cpu)
    cpu_pred, cpu_images = recon.prepare_export_data(pred, images)
    for key, tensor in cpu_pred.items():
        assert tensor.device.type == "cpu"
        assert tensor.shape == pred[key].shape
        assert tensor.dtype == pred[key].dtype
        torch.testing.assert_close(tensor, pred[key].to("cpu"), rtol=0, atol=0)
    assert cpu_pred["depth"].untyped_storage().data_ptr() == cpu_pred["local_points"].untyped_storage().data_ptr()
    assert cpu_pred["depth_conf"].untyped_storage().data_ptr() == cpu_pred["conf"].untyped_storage().data_ptr()
    assert cpu_pred["world_points_conf"] is cpu_pred["depth_conf"]
    assert cpu_pred["images"].untyped_storage().data_ptr() == cpu_images.untyped_storage().data_ptr()
    if device == "cuda":
        assert len(transfers) == 7  # six independent prediction tensors plus input images
    transfers.clear()
    _save_predictions_npz(cpu_pred, cpu_images, tmp_path / "after.npz", image_ids=[7, 2], source_model="pi3x")
    with np.load(tmp_path / "before.npz", allow_pickle=True) as before, np.load(tmp_path / "after.npz", allow_pickle=True) as after:
        assert before.files == after.files
        for key in before.files:
            assert before[key].dtype == after[key].dtype
            np.testing.assert_array_equal(before[key], after[key])

    def capture_glb(pred_np, **kwargs):
        for key, value in pred.items():
            np.testing.assert_array_equal(pred_np[key], value.to("cpu").numpy()[0])
        assert kwargs["conf_thres"] == 42
        return SimpleNamespace(export=lambda **kwargs: None)

    monkeypatch.setattr("src.pi3_glb_export.predictions_to_glb", capture_glb)
    recon.export_glb(cpu_pred, tmp_path / "result.glb", conf_thres=42)
    assert not transfers


def test_intrinsic_fit_fails_explicitly_without_default():
    invalid_points = torch.zeros((1, 1, 4, 5, 3))
    with pytest.raises(ValueError, match="无有效点"):
        _estimate_intrinsics_from_local_points(invalid_points)


@pytest.mark.parametrize("reconstructor", [PI33DReconstructor, Pi3X3DReconstructor])
def test_transform_failure_prevents_cache_save(monkeypatch, tmp_path, reconstructor):
    recon = object.__new__(reconstructor)

    def fail(*args, **kwargs):
        raise OSError("transform write failed")

    monkeypatch.setattr("utils.transforms.build_transforms", fail)
    with pytest.raises(OSError, match="transform write failed"):
        recon.save_predictions_cache({}, torch.empty(0), tmp_path, image_names=["1.jpg"], input_dir=str(tmp_path))
    assert not (tmp_path / "predictions.npz").exists()


def test_intrinsic_fit_recovers_pinhole_camera():
    y, x = torch.meshgrid(torch.arange(10), torch.arange(12), indexing="ij")
    z = torch.ones_like(x)
    points = torch.stack(((x + 0.5 - 6) / 12, (y + 0.5 - 5) / 12, z), dim=-1)[None, None]
    actual = _estimate_intrinsics_from_local_points(points)
    expected = torch.tensor([[[[12., 0., 6.], [0., 12., 5.], [0., 0., 1.]]]])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("failure", ["depth_edge", "camera_pose", "intrinsics"])
def test_pi3_postprocessing_failure_propagates(monkeypatch, failure):
    from pi3.utils import geometry
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    recon = object.__new__(PI33DReconstructor)
    recon.device = "cpu"
    y, x = torch.meshgrid(torch.arange(10), torch.arange(12), indexing="ij")
    local = torch.stack(((x + 0.5 - 6) / 12, (y + 0.5 - 5) / 12, torch.ones_like(x)), dim=-1)[None, None]
    pred = {"local_points": local, "conf": torch.ones(1, 1, 10, 12, 1),
            "camera_poses": torch.eye(4)[None, None]}
    recon.model = lambda _: pred
    if failure == "depth_edge":
        def broken_edge(*args, **kwargs):
            raise RuntimeError("depth edge failed")
        monkeypatch.setattr(geometry, "depth_edge", broken_edge)
        expected = RuntimeError
    elif failure == "camera_pose":
        pred["camera_poses"].zero_()
        expected = torch.linalg.LinAlgError
    else:
        pred["local_points"].zero_()
        expected = ValueError
    with pytest.raises(expected):
        recon.run_inference(torch.zeros(1, 3, 10, 12))
