"""Strict personalcare classification validation and global aggregation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

_METADATA = {
    "status": "master_data_pending",
    "manufacturer": None,
    "brand": None,
    "category": None,
    "object_kind": None,
}
_RESOLVED_KEYS = frozenset(
    {
        "schema_version",
        "source",
        "project_id",
        "status",
        "sku_id",
        "sku_name",
        "confidence",
        "metadata",
    }
)
_UNAVAILABLE_KEYS = frozenset(
    {"schema_version", "source", "project_id", "status", "reason"}
)
OTHER_SKU = ("56642", "其他品类")


def candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    """Keep the canonical catch-all category behind every specific SKU."""
    identity = (candidate["sku_id"], candidate["sku_name"])
    return (
        identity == OTHER_SKU,
        -candidate["confidence_sum"],
        -candidate["support_count"],
        -candidate["max_confidence"],
        *identity,
    )


def aggregate_classifications(
    classifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate all observations of one physical object deterministically."""
    groups: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = {}
    for classification in classifications:
        validated = validate_classification(classification)
        if validated["status"] == "unavailable":
            continue
        key = (validated["sku_id"], validated["sku_name"])
        groups.setdefault(key, []).append((validated["confidence"], validated))

    candidates: list[dict[str, Any]] = []
    metadata_by_candidate: dict[tuple[str, str], dict[str, Any]] = {}
    for (sku_id, sku_name), observations in groups.items():
        confidences = sorted(confidence for confidence, _ in observations)
        candidates.append(
            {
                "sku_id": sku_id,
                "sku_name": sku_name,
                "confidence_sum": math.fsum(confidences),
                "support_count": len(confidences),
                "max_confidence": max(confidences),
            }
        )
        metadata_by_candidate[(sku_id, sku_name)] = deepcopy(
            observations[0][1]["metadata"]
        )

    candidates.sort(key=candidate_sort_key)
    if not candidates:
        return {
            "status": "unavailable",
            "primary_sku_id": None,
            "candidates": [],
            "metadata": deepcopy(_METADATA),
        }

    primary = candidates[0]
    return {
        "status": "resolved" if len(candidates) == 1 else "conflict",
        "primary_sku_id": primary["sku_id"],
        "candidates": candidates,
        "metadata": metadata_by_candidate[(primary["sku_id"], primary["sku_name"])],
    }


def validate_classification(classification: Mapping[str, Any]) -> dict[str, Any]:
    """Return an independent normalized V1 record or reject malformed input."""
    if not isinstance(classification, Mapping):
        raise ValueError("classification must be an object")
    status = classification.get("status")
    expected_keys = _RESOLVED_KEYS if status == "resolved" else _UNAVAILABLE_KEYS
    if (
        status not in {"resolved", "unavailable"}
        or set(classification) != expected_keys
    ):
        raise ValueError("classification schema is invalid")
    if classification["schema_version"] != "1.0.0":
        raise ValueError("classification schema_version is invalid")
    if classification["source"] != "personalcare":
        raise ValueError("classification source is invalid")
    if (
        isinstance(classification["project_id"], bool)
        or not isinstance(classification["project_id"], int)
        or classification["project_id"] != 51
    ):
        raise ValueError("classification project_id is invalid")
    if status == "unavailable":
        if (
            not isinstance(classification["reason"], str)
            or not classification["reason"]
        ):
            raise ValueError("classification unavailable reason is invalid")
        return deepcopy(dict(classification))

    sku_id = classification["sku_id"]
    sku_name = classification["sku_name"]
    confidence = classification["confidence"]
    if not isinstance(sku_id, str) or not sku_id.strip():
        raise ValueError("classification sku_id is invalid")
    if not isinstance(sku_name, str) or not sku_name.strip():
        raise ValueError("classification sku_name is invalid")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError("classification confidence is invalid")
    if classification["metadata"] != _METADATA:
        raise ValueError("classification metadata is invalid")

    normalized = deepcopy(dict(classification))
    normalized["confidence"] = float(confidence)
    return normalized
