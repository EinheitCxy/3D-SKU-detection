"""DA3 must remain importable after the retired VGGT source is absent."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_ENV_BUILDER = CODE_ROOT / "scripts" / "3d" / "ops" / "build_unified_env.sh"


def test_root_project_declares_unified_da3_dependency_contract() -> None:
    """The root environment pins the shared core/DA3 runtime contract."""
    with (CODE_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    dependencies = set(project["project"]["dependencies"])

    assert project["project"]["requires-python"] == ">=3.11,<3.12"
    assert {
        "numpy==1.26.4",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "xformers==0.0.31",
    } <= dependencies
    assert {
        "addict",
        "omegaconf",
        "e3nn",
        "evo",
        "fastapi",
        "uvicorn",
        "bson",
    } <= dependencies


def test_unified_env_builder_rejects_existing_output(tmp_path: Path) -> None:
    """A pre-existing candidate is left untouched instead of being overwritten."""
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    marker = candidate / "keep"
    marker.write_text("preserve", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(UNIFIED_ENV_BUILDER), str(candidate)],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "already exists" in result.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    "protected_environment",
    [CODE_ROOT / ".venv", CODE_ROOT / "Depth-Anything-3" / ".venv"],
    ids=["root-venv", "da3-venv"],
)
def test_unified_env_builder_rejects_protected_descendant_before_uv(
    tmp_path: Path, protected_environment: Path
) -> None:
    """Protected-environment descendants cannot become candidate locations."""
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    uv_called = tmp_path / "uv-called"
    stub_uv = stub_bin / "uv"
    stub_uv.write_text(
        f"#!/usr/bin/env bash\ntouch {uv_called}\n",
        encoding="utf-8",
    )
    stub_uv.chmod(0o755)
    environment = os.environ | {"PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}"}

    result = subprocess.run(
        ["bash", str(UNIFIED_ENV_BUILDER), str(protected_environment / "candidate")],
        cwd=CODE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "protected environment" in result.stderr
    assert not uv_called.exists()


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
