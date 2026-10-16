"""Compact baked binary-alpha atlases using standard indexed PNG."""
from io import BytesIO
import operator

import numpy as np
from PIL import Image


def _rgba_with_binary_alpha(raw: bytes) -> Image.Image:
    with Image.open(BytesIO(raw)) as source:
        rgba = source.convert("RGBA")
    return _binary_alpha(rgba)


def _binary_alpha(rgba: Image.Image) -> Image.Image:
    alpha = np.asarray(rgba)[:, :, 3]
    if not np.isin(alpha, [0, 255]).all():
        raise ValueError("Surfel atlas must have binary alpha")
    return rgba


def compact_png(raw: bytes, colors: int = 128) -> bytes:
    """Quantize RGB for both alpha states while retaining filtered edge colors."""
    return _compact_rgba(_rgba_with_binary_alpha(raw), colors)


def encode_atlas(rgba: np.ndarray, colors: int | None = 128) -> bytes:
    """Encode a baked RGBA atlas directly, without an intermediate PNG round trip."""
    image = _binary_alpha(Image.fromarray(rgba))
    return _rgba_png(image) if colors is None else _compact_rgba(image, colors)


def _compact_rgba(rgba: Image.Image, colors: int) -> bytes:
    try:
        colors = operator.index(colors)
    except TypeError as error:
        raise ValueError("colors must be an integer within [2, 128]") from error
    if isinstance(colors, bool) or not 2 <= colors <= 128:
        raise ValueError("colors must be an integer within [2, 128]")
    alpha = np.asarray(rgba)[:, :, 3]
    indexed = rgba.convert("RGB").quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    indices = np.array(indexed)
    # Transparent texels retain their RGB: blackening them produces dark
    # outlines when a generic loader bilinearly samples the masked material.
    palette_colors = indexed.getpalette()[: 3 * colors]
    palette_colors += [0] * (3 * colors - len(palette_colors))
    indices[alpha == 0] += colors
    result = Image.fromarray(indices)
    result.putpalette(palette_colors + palette_colors)
    output = BytesIO()
    result.save(
        output,
        format="PNG",
        transparency=bytes([255] * colors + [0] * colors),
        optimize=True,
    )
    return output.getvalue()


def rgba_png(raw: bytes) -> bytes:
    """Re-encode a binary-alpha atlas losslessly as RGBA PNG."""
    return _rgba_png(_rgba_with_binary_alpha(raw))


def _rgba_png(rgba: Image.Image) -> bytes:
    output = BytesIO()
    rgba.save(output, format="PNG", optimize=True)
    return output.getvalue()
