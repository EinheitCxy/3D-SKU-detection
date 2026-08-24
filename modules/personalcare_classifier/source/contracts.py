from __future__ import annotations

import math


def split_sku_label(label: str) -> tuple[str, str]:
    sku_id, separator, sku_name = label.partition("^")
    if separator == "" or sku_id.strip() == "" or sku_name.strip() == "":
        raise ValueError("personalcare label must be 'sku_id^sku_name'")
    return sku_id.strip(), sku_name.strip()


def lookup_sku_metadata(sku_id: str, sku_name: str) -> dict[str, object]:
    if not sku_id or not sku_name:
        raise ValueError("sku_id and sku_name must be non-empty")
    return {
        "status": "master_data_pending",
        "manufacturer": None,
        "brand": None,
        "category": None,
        "object_kind": None,
    }


def resolved_classification(
    project_id: int, label: str, confidence: float
) -> dict[str, object]:
    if isinstance(project_id, bool) or not isinstance(project_id, int):
        raise ValueError("project_id must be an integer")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and within [0, 1]")
    sku_id, sku_name = split_sku_label(label)
    return {
        "schema_version": "1.0.0",
        "source": "personalcare",
        "project_id": project_id,
        "status": "resolved",
        "sku_id": sku_id,
        "sku_name": sku_name,
        "confidence": float(confidence),
        "metadata": lookup_sku_metadata(sku_id, sku_name),
    }
