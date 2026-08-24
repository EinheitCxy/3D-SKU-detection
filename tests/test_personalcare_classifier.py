import pytest

from modules.personalcare_classifier.source.contracts import (
    lookup_sku_metadata,
    resolved_classification,
    split_sku_label,
)


def test_split_sku_label_preserves_name_suffix() -> None:
    assert split_sku_label("430085^产品^限定版") == ("430085", "产品^限定版")


def test_mapping_placeholder_is_explicit_and_empty() -> None:
    assert lookup_sku_metadata("430085", "产品A") == {
        "status": "master_data_pending",
        "manufacturer": None,
        "brand": None,
        "category": None,
        "object_kind": None,
    }


def test_resolved_classification_has_stable_schema() -> None:
    assert resolved_classification(51, "430085^产品A", 0.75) == {
        "schema_version": "1.0.0",
        "source": "personalcare",
        "project_id": 51,
        "status": "resolved",
        "sku_id": "430085",
        "sku_name": "产品A",
        "confidence": 0.75,
        "metadata": lookup_sku_metadata("430085", "产品A"),
    }


@pytest.mark.parametrize("label", ["^产品A", "430085", "430085^"])
def test_split_sku_label_rejects_malformed_values(label: str) -> None:
    with pytest.raises(ValueError, match="sku_id\\^sku_name"):
        split_sku_label(label)


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.01, 1.01])
def test_resolved_classification_rejects_invalid_confidence(
    confidence: float,
) -> None:
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        resolved_classification(51, "430085^产品A", confidence)
