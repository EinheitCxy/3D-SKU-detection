"""Compact baked binary-alpha atlases using standard indexed PNG."""
from io import BytesIO

import numpy as np
from PIL import Image


def compact_png(raw: bytes) -> bytes:
    """Use 128 RGB colors with both alpha states, retaining filtered edge colors."""
    with Image.open(BytesIO(raw)) as source:
        rgba = source.convert("RGBA")
    alpha = np.asarray(rgba)[:, :, 3]
    if not np.isin(alpha, [0, 255]).all():
        raise ValueError("Surfel atlas must have binary alpha")
    colors = rgba.convert("RGB").quantize(
        colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    indices = np.array(colors)
    # Transparent texels retain their RGB: blackening them produces dark
    # outlines when a generic loader bilinearly samples the masked material.
    indices[alpha == 0] += 128
    result = Image.fromarray(indices)
    palette = colors.getpalette()[:384]
    palette += [0] * (384 - len(palette))
    result.putpalette(palette + palette)
    output = BytesIO()
    result.save(output, format="PNG", transparency=bytes([255] * 128 + [0] * 128), optimize=True)
    return output.getvalue()
