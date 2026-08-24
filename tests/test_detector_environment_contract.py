"""The detector environment remains compatible with the DA3 NumPy ABI."""

from __future__ import annotations

import tomllib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_detector_pins_numpy_1_and_opencv_4() -> None:
    config = tomllib.loads(
        (REPOSITORY_ROOT / "modules/sku_detector/pyproject.toml").read_text()
    )
    dependencies = set(config["project"]["dependencies"])

    assert "numpy==1.26.4" in dependencies
    assert "opencv-python>=4.8,<5" in dependencies
