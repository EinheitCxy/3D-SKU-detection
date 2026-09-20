"""Reconstruction data release, model reuse, and failure propagation."""

import weakref

import pytest

from src.reconstructor_base import ReconstructorBase


class _Payload:
    pass


class _Reconstructor(ReconstructorBase):
    def __init__(self, fail=None):
        self.model = None
        self.loads = 0
        self.fail = fail
        self.events = []

    def load_model(self):
        self.model = _Payload()
        self.loads += 1

    def load_images(self, input_dir):
        result = _Payload()
        self.original_image = weakref.ref(result)
        return result

    def run_inference(self, images):
        result = _Payload()
        self.original_prediction = weakref.ref(result)
        return result

    def prepare_export_data(self, predictions, images):
        self.events.append("prepare")
        return _Payload(), _Payload()

    def save_predictions_cache(self, predictions, images, out_dir, **kwargs):
        assert self.original_image() is None
        assert self.original_prediction() is None
        self.events.append("cache")
        if self.fail == "cache":
            raise ValueError("cache failed")

    def export_glb(self, predictions, output_path, **kwargs):
        self.events.append("export")
        if self.fail == "export":
            raise ValueError("export failed")

    def finish_reconstruction(self):
        self.events.append("finish")


@pytest.mark.parametrize("fail", [None, "cache", "export"])
def test_lifecycle(tmp_path, monkeypatch, fail):
    import src.reconstructor_base as base

    monkeypatch.setattr(base.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        base.torch.cuda, "empty_cache", lambda: pytest.fail("CPU cleanup invoked CUDA")
    )
    recon = _Reconstructor(fail)
    if fail:
        with pytest.raises(ValueError, match=f"{fail} failed"):
            with recon:
                recon.reconstruct_from_directory(
                    input_dir=str(tmp_path), output_path=str(tmp_path / "a.glb")
                )
        assert recon.model is None
        assert recon.events[-1] == "finish"
        if fail == "cache":
            assert "export" not in recon.events
    else:
        (tmp_path / "1.jpg").touch()
        (tmp_path / "scene.jpg").touch()
        with recon:
            recon.reconstruct_from_directory(
                input_dir=str(tmp_path), output_path=str(tmp_path / "a.glb")
            )
            model = recon.model
            recon.reconstruct_from_directory(
                input_dir=str(tmp_path), output_path=str(tmp_path / "b.glb")
            )
            assert recon.loads == 1
            assert recon.model is model
        assert recon.model is None
        assert recon.events == ["prepare", "cache", "export", "finish"] * 2
