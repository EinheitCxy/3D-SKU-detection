"""Regression coverage for the maintained video-to-dedup shell entrypoint."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "modules" / "video_to_dedup" / "run.sh"


def test_preflight_reports_missing_core_env_without_legacy_code_env(
    tmp_path: Path,
) -> None:
    """The migrated preflight must consume CORE_ENV before any workflow writes."""
    video = tmp_path / "input.mp4"
    video.touch()
    environment = os.environ | {
        "DETECTOR_ENV": "/usr",
        "CORE_ENV": "/missing-core-env",
    }
    environment.pop("CODE_ENV", None)

    result = subprocess.run(
        ["bash", str(SCRIPT), str(video)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "核心环境不存在: /missing-core-env" in result.stderr
    assert "unbound variable" not in result.stderr
