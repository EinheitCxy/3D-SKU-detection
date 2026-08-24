"""Canonical object enumeration for supported detection JSON payloads."""

from __future__ import annotations

from typing import Any


def flatten_detection_objects(payload: Any) -> list[dict[str, Any]]:
    """Return objects in the stable order used by matching and deduplication."""
    if isinstance(payload, dict) and "skus" in payload:
        skus = payload["skus"]
        if not isinstance(skus, list):
            raise ValueError("'skus' must be a list")
        objects: list[dict[str, Any]] = []
        for sku in skus:
            if not isinstance(sku, dict):
                raise ValueError("each sku must be an object")
            sku_objects = sku.get("objects", [])
            if not isinstance(sku_objects, list):
                raise ValueError("sku 'objects' must be a list")
            objects.extend(sku_objects)
        return objects
    if isinstance(payload, list):
        if not payload:
            return []
        if not isinstance(payload[0], dict):
            raise ValueError("first list entry must be an object")
        objects = payload[0].get("objects", [])
    elif isinstance(payload, dict):
        objects = payload.get("objects", [])
    else:
        raise ValueError("unsupported detection JSON structure")
    if not isinstance(objects, list):
        raise ValueError("'objects' must be a list")
    return objects
