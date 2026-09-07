"""Create the compact CSV index consumed by the static SKU Viewer exporter."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.sku_masterdata import convert_sku_maindata_xlsx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("sku-maindata.xlsx"))
    parser.add_argument(
        "--output", type=Path, default=Path("runtime/sku_masterdata.csv")
    )
    args = parser.parse_args()
    count = convert_sku_maindata_xlsx(args.input, args.output)
    print(f"wrote {count} SKU records to {args.output}")


if __name__ == "__main__":
    main()
