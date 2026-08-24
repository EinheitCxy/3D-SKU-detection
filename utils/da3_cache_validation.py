"""Canonical DA3 cache scalar/affine validation shared by exporter and
footprint stage.

Both ``src/web_viewer_export.py`` and ``src/da3_footprint_stage.py``
validate schema-v3 scalar provenance fields and the affine linear part of a
DA3 cache. This module holds the single implementation; callers pass their
own error type so each module keeps raising its own fail-closed exception
(both subclass ``ValueError``).
"""

from __future__ import annotations

import numpy as np


class Da3CacheValidationError(ValueError):
    """Default error for a DA3 cache field violating the schema-v3 contract."""


def unicode_scalar(
    value: np.ndarray,
    field: str,
    *,
    error: type[Exception] = Da3CacheValidationError,
) -> str:
    """Return the string of a scalar unicode field (must be nonempty)."""
    if value.shape != () or value.dtype.kind != "U" or not value.item():
        raise error(f"DA3 cache {field} must be a nonempty unicode scalar")
    return str(value.item())


def integer_scalar(
    value: np.ndarray,
    field: str,
    *,
    error: type[Exception] = Da3CacheValidationError,
) -> int:
    """Return the int of a scalar integer field."""
    if value.shape != () or value.dtype.kind not in "iu":
        raise error(f"DA3 cache {field} must be an integer scalar")
    return int(value.item())


def validate_affine_linear_parts(
    affine: np.ndarray,
    *,
    error: type[Exception] = Da3CacheValidationError,
) -> None:
    """Validate the linear part of every frame's source->processed affine."""
    linear = affine[:, :, :2]
    off_diagonal = np.stack([linear[:, 0, 1], linear[:, 1, 0]], axis=0)
    if not np.allclose(off_diagonal, 0.0, rtol=0.0, atol=1e-8):
        raise error("DA3 cache affine linear part must be axis-aligned")
    if np.any(linear[:, 0, 0] <= 0.0) or np.any(linear[:, 1, 1] <= 0.0):
        raise error("DA3 cache affine linear scales must be positive")
    if np.any(np.linalg.det(linear) <= 0.0):
        raise error("DA3 cache affine determinant must be positive")
    if any(np.linalg.matrix_rank(matrix) != 2 for matrix in linear):
        raise error("DA3 cache affine linear part must have rank two")
