"""MapAnything packaging and local-model contract tests."""

import builtins

import pytest


def test_mapanything_requires_explicit_model_path_before_upstream_import(monkeypatch):
    """A missing local snapshot fails before importing the optional submodule."""
    from src.mapanything_3d_reconstructor import MapAnything3DReconstructor

    original_import = builtins.__import__

    def forbid_mapanything_import(name, *args, **kwargs):
        if name == "mapanything.models":
            raise AssertionError("upstream MapAnything import must not run")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_mapanything_import)
    recon = MapAnything3DReconstructor(device="cpu")

    with pytest.raises(ValueError, match="model_path"):
        recon.load_model()


def test_mapanything_rejects_missing_local_snapshot_before_upstream_import(
    monkeypatch, tmp_path
):
    """A nonexistent explicit snapshot fails before importing model code."""
    import src.mapanything_3d_reconstructor as mapanything_backend

    package_dir = tmp_path / "map-anything" / "mapanything"
    package_dir.mkdir(parents=True)
    monkeypatch.setattr(
        mapanything_backend, "MAPANYTHING_ROOT", package_dir.parent
    )

    original_import = builtins.__import__

    def forbid_mapanything_import(name, *args, **kwargs):
        if name == "mapanything.models":
            raise AssertionError("upstream MapAnything import must not run")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_mapanything_import)
    recon = mapanything_backend.MapAnything3DReconstructor(
        device="cpu", model_path=str(tmp_path / "missing-snapshot")
    )

    with pytest.raises(FileNotFoundError, match="model_path"):
        recon.load_model()
