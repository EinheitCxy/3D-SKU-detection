from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from docker import processor
from docker.processor import (
    build_success_response,
    pack_viewer_bundle,
    prepare_request,
)


def _image_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((8, 10, 3), 127, np.uint8))
    assert ok
    return encoded.tobytes()


def _frame(
    *,
    label: str = "430085^产品A",
    cls: int = 0,
    confidence: float = 0.87,
    det_cls: int = 0,
    det_confidence: float = 0.93,
):
    return json.dumps(
        {
            "classes": {"det": ["bottle"], "cls": [label]},
            "objects": [
                {
                    "position": [1, 2, 7, 8],
                    "classes": {"det": det_cls, "cls": cls},
                    "confidences": {
                        "det": det_confidence,
                        "cls": confidence,
                    },
                }
            ],
        },
        ensure_ascii=False,
    )


def test_prepare_request_converts_classifier_frames_in_object_order(
    tmp_path: Path,
) -> None:
    prepared = prepare_request(
        {
            "images": [_image_bytes(), _image_bytes()],
            "skus": [_frame(), _frame()],
        },
        tmp_path,
    )

    assert not hasattr(prepared, "project_id")
    assert sorted(
        path.name for path in (prepared.dataset_dir / "images").iterdir()
    ) == [
        "0.jpg",
        "1.jpg",
    ]
    frame = json.loads(
        (prepared.dataset_dir / "detections_results" / "0.json").read_text()
    )
    assert list(frame) == ["skus"]
    assert frame["skus"][0]["classes"] == {
        "det": ["bottle"],
        "cls": ["430085^产品A"],
    }
    obj = frame["skus"][0]["objects"][0]
    assert obj["classes"] == {"det": 0, "cls": 0}
    assert obj["confidences"] == {"det": 0.93, "cls": 0.87}
    assert obj["classification"]["sku_id"] == "430085"
    assert obj["classification"]["confidence"] == 0.87


def test_prepare_request_preserves_two_object_order_and_detector_metadata(
    tmp_path: Path,
) -> None:
    frame = {
        "classes": {
            "det": ["bottle", "box"],
            "cls": ["430085^产品A", "428987^产品B"],
        },
        "objects": [
            {
                "position": [8, 9, 18, 19],
                "classes": {"det": 1, "cls": 0},
                "confidences": {"det": 0.71, "cls": 0.87},
                "detector_note": "first",
            },
            {
                "position": [1, 2, 5, 6],
                "classes": {"det": 0, "cls": 1},
                "confidences": {"det": 0.62, "cls": 0.42},
                "detector_note": "second",
            },
        ],
    }
    prepared = prepare_request(
        {
            "images": [_image_bytes()],
            "skus": [json.dumps(frame, ensure_ascii=False)],
        },
        tmp_path,
    )

    actual = json.loads(
        (prepared.dataset_dir / "detections_results" / "0.json").read_text()
    )["skus"][0]
    assert actual["classes"] == frame["classes"]
    assert [obj["position"] for obj in actual["objects"]] == [
        [8, 9, 18, 19],
        [1, 2, 5, 6],
    ]
    assert [obj["classes"] for obj in actual["objects"]] == [
        {"det": 1, "cls": 0},
        {"det": 0, "cls": 1},
    ]
    assert [obj["confidences"] for obj in actual["objects"]] == [
        {"det": 0.71, "cls": 0.87},
        {"det": 0.62, "cls": 0.42},
    ]
    assert [obj["detector_note"] for obj in actual["objects"]] == [
        "first",
        "second",
    ]
    assert [obj["classification"]["sku_id"] for obj in actual["objects"]] == [
        "430085",
        "428987",
    ]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda inputs: inputs.update(images=[]), "images"),
        (lambda inputs: inputs.update(skus=[]), "count"),
        (lambda inputs: inputs.update(images=[b"not an image"]), "image"),
        (lambda inputs: inputs.update(skus=[_frame(cls=1)]), "index"),
        (lambda inputs: inputs.update(skus=[_frame(label="malformed")]), "label"),
        (
            lambda inputs: inputs.update(skus=[_frame(confidence=float("nan"))]),
            "finite",
        ),
    ],
)
def test_prepare_request_rejects_malformed_input(
    mutator, message: str, tmp_path: Path
) -> None:
    inputs = {"images": [_image_bytes()], "skus": [_frame()]}
    mutator(inputs)
    with pytest.raises(ValueError, match=message):
        prepare_request(inputs, tmp_path)


def test_prepare_request_rejects_wrapper_features_and_extra_keys(
    tmp_path: Path,
) -> None:
    frame = json.loads(_frame())
    for payload in (
        {"skus": [{"skus": [frame]}]},
        {"skus": [{**frame, "features": []}]},
    ):
        inputs = {
            "images": [_image_bytes()],
            "skus": [json.dumps(payload["skus"][0])],
        }
        with pytest.raises(ValueError):
            prepare_request(inputs, tmp_path)


class _UnreadableTopLevelMetadata:
    def __iter__(self):
        raise AssertionError("top-level metadata must not be iterated")

    def __deepcopy__(self, _memo):
        raise AssertionError("top-level metadata must not be copied")

    def __str__(self) -> str:
        raise AssertionError("top-level metadata must not be parsed")


def test_prepare_request_ignores_top_level_features_and_metadata(tmp_path: Path) -> None:
    ignored = _UnreadableTopLevelMetadata()

    prepared = prepare_request(
        {
            "images": [_image_bytes()],
            "skus": [_frame()],
            "features": ignored,
            "project_id": ignored,
            "upstream_trace": ignored,
        },
        tmp_path,
    )

    assert {path.name for path in prepared.dataset_dir.iterdir()} == {
        "images",
        "detections_results",
    }
    written = json.loads(
        (prepared.dataset_dir / "detections_results" / "0.json").read_text()
    )
    assert set(written) == {"skus"}
    assert all(
        key not in json.dumps(written) for key in ("features", "upstream_trace")
    )


@pytest.mark.parametrize(
    "inputs",
    [
        {"skus": [_frame()]},
        {"images": [_image_bytes()]},
    ],
)
def test_prepare_request_requires_images_and_skus(
    inputs: dict[str, object], tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        prepare_request(inputs, tmp_path)


def test_prepare_request_uses_fixed_personalcare_domain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    domains: list[int] = []
    real_builder = processor.build_resolved_classification

    def record_domain(domain: int, label: str, confidence: float) -> dict[str, object]:
        domains.append(domain)
        return real_builder(domain, label, confidence)

    monkeypatch.setattr(processor, "build_resolved_classification", record_domain)

    processor.prepare_request(
        {
            "images": [_image_bytes()],
            "skus": [_frame()],
            "project_id": "ignored-request-value",
        },
        tmp_path,
    )

    assert domains
    assert set(domains) == {51}


def test_prepare_request_rejects_missing_or_malformed_detector_fields(
    tmp_path: Path,
) -> None:
    frame = json.loads(_frame())
    cases = [
        {**frame, "classes": {"cls": frame["classes"]["cls"]}},
        {
            **frame,
            "classes": {"det": ["bottle", 1], "cls": frame["classes"]["cls"]},
        },
        {
            **frame,
            "objects": [
                {
                    **frame["objects"][0],
                    "classes": {"det": 1, "cls": 0},
                }
            ],
        },
        {
            **frame,
            "objects": [
                {
                    **frame["objects"][0],
                    "confidences": {"det": 2.0, "cls": 0.87},
                }
            ],
        },
    ]
    for case in cases:
        with pytest.raises(ValueError):
            prepare_request(
                {
                    "images": [_image_bytes()],
                    "skus": [json.dumps(case)],
                },
                tmp_path,
            )


def test_pack_viewer_bundle_contains_flat_fixed_generation_files(
    tmp_path: Path,
) -> None:
    generation = tmp_path / "viewer" / "runs" / "run-1"
    (generation / "thumbs").mkdir(parents=True)
    for name in (
        "manifest.json",
        "positions.f32.bin",
        "colors.u8.bin",
        "normals.i8.bin",
        "objects.json",
    ):
        (generation / name).write_bytes(name.encode())
    (generation / "thumbs" / "0.jpg").write_bytes(b"jpg")
    (generation / "thumbs" / "ignored.png").write_bytes(b"png")
    (generation / "ignored.txt").write_text("no", encoding="utf-8")

    bundle = pack_viewer_bundle(generation)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert all(
            info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()
        )
        assert archive.namelist() == [
            "manifest.json",
            "positions.f32.bin",
            "colors.u8.bin",
            "normals.i8.bin",
            "objects.json",
            "thumbs/0.jpg",
        ]


def test_build_success_response_has_exact_keys(tmp_path: Path) -> None:
    global_skus = tmp_path / "global_skus.json"
    global_skus.write_text('["{\\"objects\\":[]}"]', encoding="utf-8")
    generation = tmp_path / "viewer" / "runs" / "r"
    (generation / "thumbs").mkdir(parents=True)
    for name in (
        "manifest.json",
        "positions.f32.bin",
        "colors.u8.bin",
        "normals.i8.bin",
        "objects.json",
    ):
        (generation / name).write_bytes(b"x")
    (generation / "thumbs" / "0.jpg").write_bytes(b"x")

    response = build_success_response(global_skus, generation)
    assert set(response) == {"global_skus", "viewer_bundle"}
    assert response["global_skus"] == ['{"objects":[]}']


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[1]", "global_skus"),
        ("[\"1\"]", "global_skus"),
        ("[\"NaN\"]", "non-finite JSON"),
    ],
)
def test_build_success_response_rejects_non_object_global_sku_entries(
    payload: str, message: str, tmp_path: Path
) -> None:
    global_skus = tmp_path / "global_skus.json"
    global_skus.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        build_success_response(global_skus, tmp_path / "viewer")


def test_build_success_response_propagates_missing_global_skus_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        build_success_response(tmp_path / "missing.json", tmp_path / "viewer")


def test_build_success_response_propagates_invalid_global_skus_json(
    tmp_path: Path,
) -> None:
    global_skus = tmp_path / "global_skus.json"
    global_skus.write_text("not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        build_success_response(global_skus, tmp_path / "viewer")


def test_build_success_response_propagates_invalid_embedded_global_sku_json(
    tmp_path: Path,
) -> None:
    global_skus = tmp_path / "global_skus.json"
    global_skus.write_text('["not json"]', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        build_success_response(global_skus, tmp_path / "viewer")
