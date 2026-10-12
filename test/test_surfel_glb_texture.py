from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.surfel_glb_texture import compact_png, rgba_png


def test_indexed_texture_preserves_alpha_and_simple_colors():
    source = np.zeros((16, 16, 4), dtype=np.uint8)
    source[:8] = [240, 30, 20, 255]
    source[8:, :8] = [10, 100, 220, 255]
    source[8:, 8:] = [240, 30, 20, 0]
    raw = BytesIO()
    Image.fromarray(source).save(raw, format="PNG")
    result = Image.open(BytesIO(compact_png(raw.getvalue())))
    assert result.mode == "P"
    np.testing.assert_array_equal(np.asarray(result.convert("RGBA")), source)


def test_rejects_nonbinary_alpha():
    raw = BytesIO()
    Image.new("RGBA", (1, 1), (1, 2, 3, 127)).save(raw, format="PNG")
    with pytest.raises(ValueError, match="binary alpha"):
        compact_png(raw.getvalue())


def test_64_color_palette_has_at_most_64_opaque_entries():
    source = np.zeros((16, 16, 4), dtype=np.uint8)
    source[..., :3] = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16, 1)
    source[..., 3] = 255
    source[0, :, 3] = 0
    raw = BytesIO()
    Image.fromarray(source).save(raw, format="PNG")

    with Image.open(BytesIO(compact_png(raw.getvalue(), colors=64))) as result:
        assert result.mode == "P"
        rgba = np.asarray(result.convert("RGBA"))
        opaque_colors = np.unique(rgba[rgba[..., 3] == 255, :3], axis=0)
    assert len(opaque_colors) <= 64


def test_rgba_png_preserves_transparent_rgb_and_binary_alpha():
    source = np.array(
        [[[1, 2, 3, 0], [4, 5, 6, 255]], [[7, 8, 9, 0], [10, 11, 12, 255]]],
        dtype=np.uint8,
    )
    raw = BytesIO()
    Image.fromarray(source).save(raw, format="PNG")

    with Image.open(BytesIO(rgba_png(raw.getvalue()))) as result:
        assert result.mode == "RGBA"
        decoded = np.asarray(result)
    np.testing.assert_array_equal(decoded, source)
    assert set(np.unique(decoded[..., 3])) == {0, 255}
