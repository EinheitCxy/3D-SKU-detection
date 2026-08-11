import numpy as np

from modules.da3_3d_reconstructor import DA33DReconstructor
from modules.da3_runner import _source_to_processed_affines


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
        affines[0], [[0.525, 0.0, 0.0], [0.0, 280 / 540, 0.0]]
    )
    assert np.allclose(
        affines[1], [[0.504, 0.0, 0.0], [0.0, 308 / 600, -14.0]]
    )
