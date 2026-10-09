from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.surfel_glb_texture import compact_png


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
