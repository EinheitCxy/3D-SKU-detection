"""Submit a classifier-result dataset to the local BSON mapping service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import bson
import requests


_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--classifier-result", required=True, type=Path)
    parser.add_argument("--taskID", required=True)
    parser.add_argument(
        "--output-dir", type=Path, help="default: <dataset>/docker_mapping_response"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8011/api")
    return parser.parse_args(argv)


def _numeric_files(directory: Path, suffixes: frozenset[str]) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"directory does not exist: {directory}")
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes and path.stem.isdigit()
    ]
    if not paths:
        raise ValueError(f"no numeric files in: {directory}")
    return sorted(paths, key=lambda path: int(path.stem))


def _load_request(
    dataset: Path, classifier_result: Path, taskid: str
) -> dict[str, object]:
    image_paths = _numeric_files(dataset / "images", _IMAGE_SUFFIXES)
    result_paths = _numeric_files(classifier_result, frozenset({".json"}))
    if [path.stem for path in image_paths] != [path.stem for path in result_paths]:
        raise ValueError("numeric image and classifier-result frame IDs must match")
    return {
        "taskID": taskid,
        "images": [path.read_bytes() for path in image_paths],
        "skus": [path.read_text(encoding="utf-8") for path in result_paths],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = _load_request(args.dataset, args.classifier_result, args.taskID)
    response = requests.post(
        args.url,
        data=bson.dumps(payload),
        headers={"Content-Type": "application/bson"},
        timeout=None,
    )
    response.raise_for_status()
    decoded = bson.loads(response.content)
    if not isinstance(decoded, dict) or set(decoded) != {"global_skus"}:
        raise ValueError("mapping response must contain only global_skus")
    global_skus = decoded["global_skus"]
    if not isinstance(global_skus, list) or not all(
        isinstance(item, str) for item in global_skus
    ):
        raise ValueError("global_skus must be a string list")

    output_dir = args.output_dir or args.dataset / "docker_mapping_response"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "global_skus.json").write_text(
        json.dumps(global_skus, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
