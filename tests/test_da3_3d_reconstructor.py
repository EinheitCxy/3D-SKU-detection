import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

from src.da3_3d_reconstructor import DA33DReconstructor, REPO_ROOT
from src.da3_runner import _source_to_processed_affines


def test_da3_default_interpreter_is_root_environment():
    """DA3 subprocesses must use the unified repository host environment."""
    expected = REPO_ROOT / ".venv" / "bin" / "python"

    assert DA33DReconstructor.DEFAULT_DA3_VENV_PYTHON == expected


def test_da3_reconstructor_uses_explicit_existing_python(monkeypatch, tmp_path):
    existing_python = tmp_path / "existing-da3" / "bin" / "python"
    existing_python.parent.mkdir(parents=True)
    existing_python.touch()
    monkeypatch.setenv("DA3_VENV_PYTHON", str(existing_python))

    reconstructor = DA33DReconstructor(device="cpu")

    assert reconstructor.da3_venv_python == existing_python


def test_da3_runner_persists_resize_and_batch_crop_affines():
    affines = _source_to_processed_affines(
        [(960, 540), (1000, 600)], output_height=280, output_width=504, process_res=504
    )

    assert np.allclose(
        affines[0], [[0.525, 0.0, -0.2375], [0.0, 280 / 540, (280 / 540 - 1.0) / 2.0]]
    )
    assert np.allclose(
        affines[1],
        [[0.504, 0.0, -0.248], [0.0, 308 / 600, (308 / 600 - 1.0) / 2.0 - 14.0]],
    )


def _write_runner_cache(
    path, *, is_metric=1, include_matcher_fields=True, dtype_overrides=None
):
    """Write the runner-shaped cache without NumPy appending a .npz suffix."""
    payload = {
        "cache_schema_version": np.asarray(3, dtype=np.int32),
        "is_metric": np.asarray(is_metric, dtype=np.int32),
        "scale_factor": np.asarray(1.0, dtype=np.float32),
        "source_model": np.asarray("depth-anything/test", dtype="<U32"),
        "source_image_sha256": np.asarray(["a" * 64], dtype="<U64"),
        "affine_convention": np.asarray("pixel_center_v1", dtype="<U32"),
        "preprocess_resolution": np.asarray(504, dtype=np.int32),
        "preprocess_method": np.asarray("upper_bound_resize", dtype="<U32"),
        "frame_alignment_sorted_indices": np.asarray([0], dtype=np.intp),
        "frame_alignment_map_keys": np.asarray([0], dtype=np.int32),
        "frame_alignment_map_values": np.asarray([0], dtype=np.int32),
    }
    if include_matcher_fields:
        payload.update(
            depth=np.zeros((1, 2, 2, 1), dtype=np.float32),
            depth_conf=np.ones((1, 2, 2), dtype=np.float32),
            world_points=np.zeros((1, 2, 2, 3), dtype=np.float32),
            world_points_conf=np.ones((1, 2, 2), dtype=np.float32),
            extrinsic=np.zeros((1, 3, 4), dtype=np.float32),
            intrinsic=np.eye(3, dtype=np.float32)[None],
            images=np.zeros((1, 2, 2, 3), dtype=np.uint8),
            image_ids=np.asarray([0], dtype=np.int32),
            source_image_sizes=np.asarray([[2, 2]], dtype=np.int32),
            source_to_processed_affine=np.asarray(
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32
            ),
        )
    for key, dtype in (dtype_overrides or {}).items():
        payload[key] = payload[key].astype(dtype)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def test_da3_reconstructor_validates_and_atomically_publishes_runner_partial(
    monkeypatch, tmp_path
):
    existing_python = tmp_path / "existing-da3" / "bin" / "python"
    existing_python.parent.mkdir(parents=True)
    existing_python.touch()
    monkeypatch.setenv("DA3_VENV_PYTHON", str(existing_python))
    reconstructor = DA33DReconstructor(device="cpu")
    cache_dir = tmp_path / "da3_cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "predictions.npz"
    cache_path.write_bytes(b"previous-complete-cache")
    partial_path = cache_dir / "predictions.npz.partial"
    _write_runner_cache(partial_path)

    reconstructor.save_predictions_cache(
        {"_npz_path": partial_path}, images=None, out_dir=cache_dir
    )

    assert not partial_path.exists()
    with np.load(cache_path, allow_pickle=False) as cache:
        assert int(cache["cache_schema_version"]) == 3
        assert int(cache["is_metric"]) == 1
        assert cache["world_points"].shape == (1, 2, 2, 3)


def test_da3_reconstructor_rejects_invalid_partial_without_replacing_cache(
    monkeypatch, tmp_path
):
    existing_python = tmp_path / "existing-da3" / "bin" / "python"
    existing_python.parent.mkdir(parents=True)
    existing_python.touch()
    monkeypatch.setenv("DA3_VENV_PYTHON", str(existing_python))
    reconstructor = DA33DReconstructor(device="cpu")
    cache_dir = tmp_path / "da3_cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "predictions.npz"
    cache_path.write_bytes(b"previous-complete-cache")
    partial_path = cache_dir / "predictions.npz.partial"
    _write_runner_cache(partial_path, is_metric=0)

    with pytest.raises(ValueError, match="is_metric"):
        reconstructor.save_predictions_cache(
            {"_npz_path": partial_path}, images=None, out_dir=cache_dir
        )

    assert cache_path.read_bytes() == b"previous-complete-cache"
    assert not partial_path.exists()


def test_da3_reconstructor_rejects_wrong_dtype_without_replacing_cache(
    monkeypatch, tmp_path
):
    existing_python = tmp_path / "existing-da3" / "bin" / "python"
    existing_python.parent.mkdir(parents=True)
    existing_python.touch()
    monkeypatch.setenv("DA3_VENV_PYTHON", str(existing_python))
    reconstructor = DA33DReconstructor(device="cpu")
    cache_dir = tmp_path / "da3_cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "predictions.npz"
    cache_path.write_bytes(b"previous-complete-cache")
    partial_path = cache_dir / "predictions.npz.partial"
    _write_runner_cache(partial_path, dtype_overrides={"world_points": np.float64})

    with pytest.raises(ValueError, match="world_points dtype"):
        reconstructor.save_predictions_cache(
            {"_npz_path": partial_path}, images=None, out_dir=cache_dir
        )

    assert cache_path.read_bytes() == b"previous-complete-cache"
    assert not partial_path.exists()


def test_da3_reconstructor_uses_output_parent_for_runner_partial_and_publish(
    monkeypatch, tmp_path
):
    existing_python = tmp_path / "existing-da3" / "bin" / "python"
    existing_python.parent.mkdir(parents=True)
    existing_python.touch()
    monkeypatch.setenv("DA3_VENV_PYTHON", str(existing_python))
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "0.jpg").touch()
    output_path = (
        tmp_path / "Output" / "floor_display2" / "da3_cache" / "reconstruction_da3.glb"
    )

    def fake_runner(cmd, **_):
        partial_path = Path(cmd[cmd.index("--output_npz") + 1])
        _write_runner_cache(partial_path)
        return SimpleNamespace(returncode=0, stdout="runner complete", stderr="")

    monkeypatch.setattr("subprocess.run", fake_runner)
    reconstructor = DA33DReconstructor(device="cpu")

    reconstructor.reconstruct_from_directory(
        input_dir=str(images_dir), output_path=str(output_path)
    )

    assert (output_path.parent / "predictions.npz").is_file()
    assert not (output_path.parent / "predictions.npz.partial").exists()
    assert not (output_path.parent / "da3_cache").exists()


def test_da3_reconstruction_requires_cache_publication(monkeypatch, tmp_path):
    existing_python = tmp_path / "existing-da3" / "bin" / "python"
    existing_python.parent.mkdir(parents=True)
    existing_python.touch()
    monkeypatch.setenv("DA3_VENV_PYTHON", str(existing_python))
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "0.jpg").touch()
    reconstructor = DA33DReconstructor(device="cpu")
    monkeypatch.setattr(reconstructor, "run_inference", lambda _: {})
    monkeypatch.setattr(reconstructor, "export_glb", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="requires save_predictions=True"):
        reconstructor.reconstruct_from_directory(
            input_dir=str(images_dir),
            output_path=str(tmp_path / "reconstruction_da3.glb"),
            save_predictions=False,
        )


def test_da3_reconstruction_fails_when_cache_publication_fails(monkeypatch, tmp_path):
    existing_python = tmp_path / "existing-da3" / "bin" / "python"
    existing_python.parent.mkdir(parents=True)
    existing_python.touch()
    monkeypatch.setenv("DA3_VENV_PYTHON", str(existing_python))
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "0.jpg").touch()
    output_path = (
        tmp_path / "Output" / "floor_display2" / "da3_cache" / "reconstruction_da3.glb"
    )
    cache_path = output_path.parent / "predictions.npz"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"previous-complete-cache")

    def fake_runner(cmd, **_):
        _write_runner_cache(Path(cmd[cmd.index("--output_npz") + 1]), is_metric=0)
        return SimpleNamespace(returncode=0, stdout="runner complete", stderr="")

    monkeypatch.setattr("subprocess.run", fake_runner)
    reconstructor = DA33DReconstructor(device="cpu")

    with pytest.raises(ValueError, match="is_metric"):
        reconstructor.reconstruct_from_directory(
            input_dir=str(images_dir), output_path=str(output_path)
        )

    assert cache_path.read_bytes() == b"previous-complete-cache"
    assert not (output_path.parent / "predictions.npz.partial").exists()
