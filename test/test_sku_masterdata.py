import csv
import json

import pytest

from src.sku_masterdata import SkuMasterDataError, load_sku_masterdata_csv
from src.web_viewer_export import publish_sku_masterdata_for_current_bundle


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sku_id", "manufacturer", "brand", "category", "is_posm"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_loads_requested_skus_and_uses_type_27_as_posm(tmp_path):
    source = tmp_path / "sku_masterdata.csv"
    _write_csv(
        source,
        [
            {
                "sku_id": "100",
                "manufacturer": "百事",
                "brand": "冲劲",
                "category": "饮料酒水",
                "is_posm": "0",
            },
            {
                "sku_id": "200",
                "manufacturer": "可口可乐",
                "brand": "Coca Cola",
                "category": "饮料酒水",
                "is_posm": "1",
            },
        ],
    )

    assert load_sku_masterdata_csv(source, {"200", "100"}) == {
        "100": {
            "manufacturer": "百事",
            "brand": "冲劲",
            "category": "饮料酒水",
            "is_posm": False,
        },
        "200": {
            "manufacturer": "可口可乐",
            "brand": "Coca Cola",
            "category": "饮料酒水",
            "is_posm": True,
        },
    }


def test_rejects_a_bundle_sku_missing_from_masterdata(tmp_path):
    source = tmp_path / "sku_masterdata.csv"
    _write_csv(
        source,
        [{"sku_id": "100", "manufacturer": "百事", "brand": "冲劲", "category": "饮料酒水", "is_posm": "0"}],
    )

    with pytest.raises(SkuMasterDataError, match="missing SKU IDs: 200"):
        load_sku_masterdata_csv(source, {"100", "200"})


def test_publishes_masterdata_in_a_new_immutable_viewer_run(tmp_path):
    output = tmp_path / "viewer"
    existing = output / "runs" / "previous"
    existing.mkdir(parents=True)
    (existing / "objects.json").write_text(
        '{"11":{"ordered_skus":[{"sku_id":"100","sku_name":"产品"}]}}',
        encoding="utf-8",
    )
    (existing / "positions.f32.bin").write_bytes(b"point-data")
    (output / "CURRENT").write_text('{"run_id":"previous"}', encoding="utf-8")
    source = tmp_path / "sku_masterdata.csv"
    _write_csv(
        source,
        [{"sku_id": "100", "manufacturer": "百事", "brand": "冲劲", "category": "饮料酒水", "is_posm": "0"}],
    )

    generation = publish_sku_masterdata_for_current_bundle(output, source)

    assert generation.name != "previous"
    assert not (existing / "sku_masterdata.json").exists()
    assert (generation / "positions.f32.bin").read_bytes() == b"point-data"
    assert json.loads((generation / "sku_masterdata.json").read_text()) == {
        "100": {"manufacturer": "百事", "brand": "冲劲", "category": "饮料酒水", "is_posm": False}
    }
    assert json.loads((output / "CURRENT").read_text()) == {"run_id": generation.name}
