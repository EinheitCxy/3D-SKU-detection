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
    (tmp_path / "thumbs").mkdir()
    (tmp_path / "thumbs" / "item.jpg").write_bytes(b"thumbnail")
    with zipfile.ZipFile(io.BytesIO(pack_viewer_bundle(tmp_path))) as archive:
        for member in archive.infolist():
            assert member.compress_type == (
                zipfile.ZIP_DEFLATED if member.filename.endswith(".bin") else zipfile.ZIP_STORED
            )
            assert archive.read(member) == (tmp_path / member.filename).read_bytes()
        assert ("surfel-scale.f16.bin" in archive.namelist()) == (version == 3)
        if version == 3:
            assert archive.read("surfel-scale.f16.bin") == b"\x00\x38\x00\x40"
    if version == 3:
        (tmp_path / "surfel-scale.f16.bin").unlink()
        with pytest.raises(FileNotFoundError, match="surfel-scale"):
            pack_viewer_bundle(tmp_path)
