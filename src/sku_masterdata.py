"""Convert SKU master-data workbooks and select compact Viewer metadata."""

from __future__ import annotations

import csv
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from python_calamine import load_workbook

_CSV_FIELDS = ("sku_id", "manufacturer", "brand", "category", "is_posm")
_WORKBOOK_FIELDS = ("sku_id", "group_name", "brand_name", "category_name", "type_id")
_SKU_ID = re.compile(r"^(0|[1-9][0-9]*)$")


class SkuMasterDataError(ValueError):
    """Raised when SKU master data cannot form a complete Viewer index."""


def convert_sku_maindata_xlsx(input_path: Path, output_path: Path) -> int:
    """Write a narrow CSV from the source workbook with `type_id == 27` as POSM."""
    source = Path(input_path)
    if not source.is_file():
        raise SkuMasterDataError(f"source workbook not found: {source}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        sheet = load_workbook(source).get_sheet_by_name("Sheet1")
    except Exception as error:
        raise SkuMasterDataError(f"cannot read source workbook: {source}") from error
    rows = sheet.iter_rows()
    try:
        header_row = next(rows)
    except StopIteration as error:
        raise SkuMasterDataError("source workbook has no header row") from error
    header = {
        str(value).strip(): index
        for index, value in enumerate(header_row)
        if isinstance(value, str) and value.strip()
    }
    missing = sorted(set(_WORKBOOK_FIELDS) - set(header))
    if missing:
        raise SkuMasterDataError("source workbook missing columns: " + ", ".join(missing))
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            seen_sku_ids: set[str] = set()
            count = 0
            for row in rows:
                sku_id = _normalize_sku_id(_row_value(row, header["sku_id"]))
                if sku_id in seen_sku_ids:
                    raise SkuMasterDataError(
                        f"source workbook has duplicate sku_id: {sku_id}"
                    )
                seen_sku_ids.add(sku_id)
                writer.writerow(
                    {
                        "sku_id": sku_id,
                        "manufacturer": _normalize_text(
                            _row_value(row, header["group_name"])
                        )
                        or "",
                        "brand": _normalize_text(_row_value(row, header["brand_name"]))
                        or "",
                        "category": _normalize_text(
                            _row_value(row, header["category_name"])
                        )
                        or "",
                        "is_posm": "1"
                        if _normalize_sku_id(_row_value(row, header["type_id"]))
                        == "27"
                        else "0",
                    }
                )
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return count
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_sku_masterdata_csv(
    source_path: Path, requested_sku_ids: set[str]
) -> dict[str, dict[str, str | bool | None]]:
    """Select exactly the SKU records a Viewer bundle references."""
    source = Path(source_path)
    if not source.is_file():
        raise SkuMasterDataError(f"SKU master-data CSV not found: {source}")
    requested = {_normalize_sku_id(sku_id) for sku_id in requested_sku_ids}
    selected: dict[str, dict[str, str | bool | None]] = {}
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _CSV_FIELDS:
            raise SkuMasterDataError(
                "SKU master-data CSV columns must be: " + ", ".join(_CSV_FIELDS)
            )
        for row in reader:
            sku_id = _normalize_sku_id(row.get("sku_id"))
            if sku_id not in requested:
                continue
            if sku_id in selected:
                raise SkuMasterDataError(
                    f"SKU master-data CSV has duplicate sku_id: {sku_id}"
                )
            selected[sku_id] = {
                "manufacturer": _normalize_text(row.get("manufacturer")),
                "brand": _normalize_text(row.get("brand")),
                "category": _normalize_text(row.get("category")),
                "is_posm": _parse_posm(row.get("is_posm")),
            }
    missing = sorted(requested - set(selected), key=int)
    if missing:
        raise SkuMasterDataError("SKU master-data CSV missing SKU IDs: " + ", ".join(missing))
    return selected


def _row_value(row: list[Any], index: int) -> Any:
    return row[index] if index < len(row) else None


def _normalize_sku_id(value: Any) -> str:
    if isinstance(value, bool):
        raise SkuMasterDataError("SKU ID must be a non-negative decimal integer")
    if isinstance(value, int):
        normalized = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise SkuMasterDataError("SKU ID must be a non-negative decimal integer")
        normalized = str(int(value))
    elif isinstance(value, str):
        normalized = value.strip()
    else:
        raise SkuMasterDataError("SKU ID must be a non-negative decimal integer")
    if _SKU_ID.fullmatch(normalized) is None:
        raise SkuMasterDataError("SKU ID must be a non-negative decimal integer")
    return normalized


def _normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return None if not normalized or normalized.upper() == "NULL" else normalized


def _parse_posm(value: str | None) -> bool:
    if value == "1":
        return True
    if value == "0":
        return False
    raise SkuMasterDataError("SKU master-data is_posm must be 0 or 1")


def _temporary_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)
