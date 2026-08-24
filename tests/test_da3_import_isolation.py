"""DA3 must remain importable after the retired VGGT source is absent."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]


def test_da3_matching_core_import_does_not_require_vggt_source() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from utils import SKUMatchingSystem; print(SKUMatchingSystem.__name__)",
        ],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "SKUMatchingSystem"
