"""Add compact SKU metadata to the current immutable Viewer bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.web_viewer_export import publish_sku_masterdata_for_current_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--viewer-output", type=Path, default=Path("modules/viewer_web/public/data")
    )
    parser.add_argument(
        "--sku-masterdata-csv", type=Path, default=Path("runtime/sku_masterdata.csv")
    )
    args = parser.parse_args()
    generation = publish_sku_masterdata_for_current_bundle(
        args.viewer_output, args.sku_masterdata_csv
    )
    print(f"published SKU master data in {generation}")


if __name__ == "__main__":
    main()
