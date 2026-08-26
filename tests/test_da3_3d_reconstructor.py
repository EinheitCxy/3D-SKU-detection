import numpy as np

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


def test_da3_reconstructor_preserves_schema_v3_provenance(monkeypatch, tmp_path):
    existing_python = tmp_path / "existing-da3" / "bin" / "python"
    existing_python.parent.mkdir(parents=True)
    existing_python.touch()
    monkeypatch.setenv("DA3_VENV_PYTHON", str(existing_python))
    reconstructor = DA33DReconstructor(device="cpu")
    provenance = {
        "cache_schema_version": np.asarray(3, dtype=np.int32),
        "is_metric": np.asarray(1, dtype=np.int32),
        "scale_factor": np.asarray(1.0, dtype=np.float32),
        "source_image_sha256": np.asarray(["a" * 64], dtype="<U64"),
        "affine_convention": np.asarray("pixel_center_v1", dtype="<U32"),
        "preprocess_resolution": np.asarray(504, dtype=np.int32),
        "preprocess_method": np.asarray("upper_bound_resize", dtype="<U32"),
    }
    predictions = {
        "world_points": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "world_points_conf": np.ones((1, 2, 2), dtype=np.float32),
        **provenance,
    }

    reconstructor.save_predictions_cache(
        predictions, images=None, out_dir=tmp_path / "da3_cache", image_names=["0.jpg"]
    )

    with np.load(
        tmp_path / "da3_cache" / "predictions.npz", allow_pickle=False
    ) as cache:
        for key, expected in provenance.items():
            assert cache[key].dtype == expected.dtype
            assert np.array_equal(cache[key], expected)
