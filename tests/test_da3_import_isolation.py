"""DA3 must remain importable after the retired VGGT source is absent."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = CODE_ROOT / "sam3"
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


def test_root_project_excludes_incompatible_unified_runtime_packages() -> None:
    """The Python 3.11 candidate avoids obsolete or unsupported runtime wheels."""
    with (CODE_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)
    with (CODE_ROOT / "uv.lock").open("rb") as lock_file:
        lock = tomllib.load(lock_file)

    dependencies = set(project["project"]["dependencies"])
    locked_package_names = {package["name"] for package in lock["package"]}

    assert "ipywidgets>=8.0.4" in dependencies
    assert not any(
        dependency.startswith(("decord", "pygltflib", "dataclasses"))
        for dependency in dependencies
    )
    assert {"decord", "pygltflib", "dataclasses"}.isdisjoint(locked_package_names)


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


def test_sam3_image_model_builder_import_does_not_require_optional_video_decoder(
    tmp_path: Path,
) -> None:
    """Image-only SAM3 imports must not load the optional decord video reader."""
    (tmp_path / "decord.py").write_text(
        "raise ModuleNotFoundError('decord intentionally unavailable')\n",
        encoding="utf-8",
    )
    pythonpath_entries = [str(tmp_path), str(SAM3_ROOT), str(CODE_ROOT)]
    if existing_pythonpath := os.environ.get("PYTHONPATH"):
        pythonpath_entries.append(existing_pythonpath)
    environment = os.environ | {"PYTHONPATH": os.pathsep.join(pythonpath_entries)}

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sam3.model_builder import build_sam3_image_model; "
            "print(build_sam3_image_model.__name__)",
        ],
        cwd=CODE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "build_sam3_image_model"


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
