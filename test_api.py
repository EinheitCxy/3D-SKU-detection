"""Submit a classifier-result dataset to the local BSON mapping service."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Sequence

import bson
import requests


_FIXED_VIEWER_FILES = (
    "manifest.json",
    "positions.f32.bin",
    "colors.u8.bin",
    "normals.i8.bin",
    "objects.json",
)
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--classifier-result", required=True, type=Path)
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


def _load_request(dataset: Path, classifier_result: Path) -> dict[str, object]:
    image_paths = _numeric_files(dataset / "images", _IMAGE_SUFFIXES)
    result_paths = _numeric_files(classifier_result, frozenset({".json"}))
    if [path.stem for path in image_paths] != [path.stem for path in result_paths]:
        raise ValueError("numeric image and classifier-result frame IDs must match")
    return {
        "images": [path.read_bytes() for path in image_paths],
        "skus": [path.read_text(encoding="utf-8") for path in result_paths],
    }


def _verify_viewer_bundle(bundle: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        members = set(archive.namelist())
        expected = set(_FIXED_VIEWER_FILES)
        if not expected.issubset(members):
            raise ValueError("viewer bundle is missing fixed members")
        if any(
            name not in expected
            and not (name.startswith("thumbs/") and name.endswith(".jpg"))
            for name in members
        ):
            raise ValueError("viewer bundle contains an unexpected member")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = _load_request(args.dataset, args.classifier_result)
    response = requests.post(
        args.url,
        data=bson.dumps(payload),
        headers={"Content-Type": "application/bson"},
        timeout=None,
    )
    response.raise_for_status()
    decoded = bson.loads(response.content)
    if not isinstance(decoded, dict) or set(decoded) != {"global_skus", "viewer_bundle"}:
        raise ValueError("mapping response must contain only global_skus and viewer_bundle")
    global_skus = decoded["global_skus"]
    viewer_bundle = decoded["viewer_bundle"]
    if not isinstance(global_skus, list) or not all(
        isinstance(item, str) for item in global_skus
    ):
        raise ValueError("global_skus must be a string list")
    if not isinstance(viewer_bundle, bytes):
        raise ValueError("viewer_bundle must be bytes")
    _verify_viewer_bundle(viewer_bundle)

    output_dir = args.output_dir or args.dataset / "docker_mapping_response"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "global_skus.json").write_text(
        json.dumps(global_skus, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "viewer_bundle.zip").write_bytes(viewer_bundle)


if __name__ == "__main__":
    main()
