"""Publish the scale sidecar whenever the declared Surfel contract requires it."""
import io
import json
import zipfile

import pytest

from processor import _VIEWER_FILES, pack_viewer_bundle


@pytest.mark.parametrize("version", [2, 3])
def test_pack_preserves_declared_surfel_assets(tmp_path, version):
    for name in (*_VIEWER_FILES, "surfel-texture-0.jpg"):
        (tmp_path / name).write_bytes(b"asset")
    (tmp_path / "surfel.json").write_text(json.dumps({
        "version": version, "frames": [{"texture": "surfel-texture-0.jpg"}],
    }))
    if version == 3:
        (tmp_path / "surfel-scale.f16.bin").write_bytes(b"\x00\x38\x00\x40")
    with zipfile.ZipFile(io.BytesIO(pack_viewer_bundle(tmp_path))) as archive:
        assert ("surfel-scale.f16.bin" in archive.namelist()) == (version == 3)
        if version == 3:
            assert archive.read("surfel-scale.f16.bin") == b"\x00\x38\x00\x40"
    if version == 3:
        (tmp_path / "surfel-scale.f16.bin").unlink()
        with pytest.raises(FileNotFoundError, match="surfel-scale"):
            pack_viewer_bundle(tmp_path)
