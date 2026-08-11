"""Pure helpers for calibrated ground-stack bounding-box measurements."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence


class BBoxAreaError(ValueError):
    """Raised when a bbox or its physical calibration is invalid."""


@dataclass(frozen=True)
class PlanarCalibration:
    """Pixel-to-centimetre scale derived from one known-size anchor bbox."""

    source_bbox: tuple[float, float, float, float]
    width_cm: float
    height_cm: float


@dataclass(frozen=True)
class SelectedInstance:
    """One valid, largest-area observation selected for a physical object."""

    global_id: str
    image_id: int
    object_id: int
    bbox: tuple[float, float, float, float]
    source_area_px2: float


@dataclass(frozen=True)
class RejectedInstance:
    """A global ID that has no observation suitable for measurement."""

    global_id: str
    reason: str


def validate_bbox(bbox: Sequence[float] | Any) -> tuple[float, float, float, float]:
    """Return a finite, non-empty `(x1, y1, x2, y2)` bbox or raise."""
    if isinstance(bbox, (str, bytes)):
        raise BBoxAreaError("bbox must contain four numeric coordinates")
    try:
        if len(bbox) != 4:
            raise BBoxAreaError("bbox must contain four numeric coordinates")
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise BBoxAreaError("bbox must contain four numeric coordinates") from exc

    if not all(isfinite(value) for value in (x1, y1, x2, y2)):
        raise BBoxAreaError("bbox coordinates must be finite")
    if x2 <= x1 or y2 <= y1:
        raise BBoxAreaError("bbox must have positive width and height")
    return x1, y1, x2, y2


def _validate_dimension(value: float | Any, name: str) -> float:
    try:
        dimension = float(value)
    except (TypeError, ValueError) as exc:
        raise BBoxAreaError(f"{name} must be a positive finite value") from exc
    if not isfinite(dimension) or dimension <= 0:
        raise BBoxAreaError(f"{name} must be a positive finite value")
    return dimension


def calibrate_from_anchor(
    bbox: Sequence[float] | Any, width_cm: float, height_cm: float
) -> PlanarCalibration:
    """Build a planar calibration from one detected bbox of known dimensions."""
    return PlanarCalibration(
        source_bbox=validate_bbox(bbox),
        width_cm=_validate_dimension(width_cm, "width_cm"),
        height_cm=_validate_dimension(height_cm, "height_cm"),
    )


def calibrated_bbox_area_cm2(
    bbox: Sequence[float] | Any, calibration: PlanarCalibration
) -> float:
    """Return the planar bbox-equivalent physical area in square centimetres."""
    x1, y1, x2, y2 = validate_bbox(bbox)
    ax1, ay1, ax2, ay2 = calibration.source_bbox
    scale_x = calibration.width_cm / (ax2 - ax1)
    scale_y = calibration.height_cm / (ay2 - ay1)
    area_cm2 = (x2 - x1) * (y2 - y1) * scale_x * scale_y
    if not isfinite(area_cm2) or area_cm2 <= 0:
        raise BBoxAreaError("calibrated bbox area must be positive and finite")
    return area_cm2


def _global_id_key(global_id: str) -> tuple[int, int | str]:
    try:
        return 0, int(global_id)
    except ValueError:
        return 1, global_id


def _validated_observation(
    global_id: str, observation: Mapping[str, Any]
) -> SelectedInstance:
    try:
        image_id = int(observation["image_id"])
        object_id = int(observation["object_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BBoxAreaError("observation must contain integer image_id and object_id") from exc

    bbox = validate_bbox(observation.get("bbox"))
    x1, y1, x2, y2 = bbox
    return SelectedInstance(
        global_id=str(global_id),
        image_id=image_id,
        object_id=object_id,
        bbox=bbox,
        source_area_px2=(x2 - x1) * (y2 - y1),
    )


def select_best_instances(
    global_mapping: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[list[SelectedInstance], list[RejectedInstance]]:
    """Select the largest valid bbox observation for each global ID.

    A repeated observation across frames is evidence for one physical instance,
    not an additional contribution to the final area sum.
    """
    selected: list[SelectedInstance] = []
    rejected: list[RejectedInstance] = []

    for global_id in sorted((str(key) for key in global_mapping), key=_global_id_key):
        observations = global_mapping[global_id]
        candidates: list[SelectedInstance] = []
        errors: list[str] = []
        for observation in observations:
            try:
                candidates.append(_validated_observation(global_id, observation))
            except BBoxAreaError as exc:
                errors.append(str(exc))

        if not candidates:
            reason = errors[0] if errors else "global ID has no observations"
            rejected.append(RejectedInstance(global_id=global_id, reason=reason))
            continue

        candidates.sort(
            key=lambda item: (
                -item.source_area_px2,
                item.image_id,
                item.object_id,
            )
        )
        selected.append(candidates[0])

    return selected, rejected
