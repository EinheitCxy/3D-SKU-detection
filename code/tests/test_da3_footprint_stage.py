import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modules import da3_footprint_stage as stage
from modules.da3_runner import (
    _preprocess_geometry,
    _source_to_processed_affines,
    _validate_model_id,
)


def test_ground_stack_area_cli_calls_da3_footprint_stage(monkeypatch, tmp_path):
    import main

    calls: list[tuple[str, Path]] = []

    def run_da3_footprint(dataset: str, save_root: Path) -> dict[str, object]:
        calls.append((dataset, save_root))
        return {
            "success": True,
            "status": "accepted",
            "report_path": str(tmp_path / "measurement_report.json"),
        }

    monkeypatch.setattr(stage, "run_da3_footprint", run_da3_footprint)
    dataset = tmp_path / "dataset"
    save_root = tmp_path / "Output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--mode",
            "ground-stack-area",
            "--dataset",
            str(dataset),
            "--save_root",
            str(save_root),
        ],
    )

    main.main()

    assert calls == [(str(dataset), save_root.resolve())]


@pytest.mark.parametrize(
    "removed_option,value",
    [
        ("--area-mode", "calibrated_bbox"),
        ("--area-anchor-frame", "0"),
        ("--area-anchor-object", "0"),
        ("--area-anchor-width-cm", "40"),
        ("--area-anchor-height-cm", "30"),
    ],
)
def test_ground_stack_area_cli_rejects_removed_bbox_anchor_options(
    monkeypatch, removed_option, value
):
    import main

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--mode", "ground-stack-area", removed_option, value],
    )

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 2


def test_ground_stack_area_cli_preserves_rejected_exit_and_report_log(
    monkeypatch, tmp_path, capsys
):
    import main

    report_path = tmp_path / "measurement_report.json"
    monkeypatch.setattr(
        stage,
        "run_da3_footprint",
        lambda *_: {
            "success": False,
            "status": "rejected",
            "report_path": str(report_path),
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--mode", "ground-stack-area", "--save_root", str(tmp_path / "Output")],
    )

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 2
    assert str(report_path) in capsys.readouterr().out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image(path: Path, value: int = 200) -> None:
    Image.fromarray(np.full((126, 126, 3), value, dtype=np.uint8)).save(path)


def _put_carton(
    world_points: np.ndarray,
    frame: int,
    bbox: list[int],
    x_range: tuple[float, float],
    height: float,
) -> None:
    x1, y1, x2, y2 = bbox
    y_values = np.linspace(0.0, 1.0, y2 - y1, endpoint=True)
    x_values = np.linspace(*x_range, x2 - x1, endpoint=True)
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    world_points[frame, y1:y2, x1:x2] = np.stack(
        [x_grid, y_grid, np.full_like(x_grid, height)], axis=-1
    )


def make_metric_fixture(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    dataset = tmp_path / "metric_dataset"
    images = dataset / "images"
    detections = dataset / "detections_results"
    images.mkdir(parents=True)
    detections.mkdir()
    save_root = tmp_path / "Output"
    output = save_root / dataset.name
    (output / "da3_cache").mkdir(parents=True)
    (output / "dedup_detections").mkdir()

    boxes = {
        0: [[16, 24, 80, 104]],
        1: [[18, 22, 82, 102]],
        2: [[62, 24, 126, 104]],
    }
    input_paths: list[Path] = []
    for frame, frame_boxes in boxes.items():
        image_path = images / f"{frame}.png"
        _write_image(image_path)
        detection_path = detections / f"{frame}.json"
        detection_path.write_text(json.dumps({"objects": [{"position": box} for box in frame_boxes]}))
        input_paths.extend([image_path, detection_path])

    world_points = np.zeros((3, 126, 126, 3), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(-1.0, 2.0, 126), np.linspace(-1.0, 2.0, 126), indexing="xy")
    world_points[..., 0] = xs
    world_points[..., 1] = ys
    _put_carton(world_points, 0, boxes[0][0], (0.0, 1.0), 0.02)
    _put_carton(world_points, 1, boxes[1][0], (0.0, 1.0), 0.03)
    _put_carton(world_points, 2, boxes[2][0], (0.5, 1.5), 0.80)
    source_paths = [images / f"{frame}.png" for frame in range(3)]
    np.savez_compressed(
        output / "da3_cache" / "predictions.npz",
        world_points=world_points,
        world_points_conf=np.ones((3, 126, 126), dtype=np.float32),
        image_ids=np.asarray([0, 1, 2], dtype=np.int32),
        source_image_sizes=np.asarray([(126, 126)] * 3, dtype=np.int32),
        source_to_processed_affine=np.asarray(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * 3, dtype=np.float32
        ),
        cache_schema_version=np.asarray(2, dtype=np.int32),
        source_model=np.asarray("depth-anything/DA3NESTED-GIANT-LARGE", dtype="<U64"),
        source_image_sha256=np.asarray([_sha256(path) for path in source_paths], dtype="<U64"),
        affine_convention=np.asarray("pixel_center_v1", dtype="<U32"),
        preprocess_resolution=np.asarray(126, dtype=np.int32),
        preprocess_method=np.asarray("upper_bound_resize", dtype="<U32"),
    )
    mapping = {
        "1": [
            {"image_id": 0, "object_id": 0, "bbox": boxes[0][0]},
            {"image_id": 1, "object_id": 0, "bbox": boxes[1][0]},
        ],
        "2": [{"image_id": 2, "object_id": 0, "bbox": boxes[2][0]}],
    }
    mapping_path = output / "dedup_detections" / "global_mapping.json"
    mapping_path.write_text(json.dumps(mapping))
    input_paths.extend([output / "da3_cache" / "predictions.npz", mapping_path])
    return dataset, save_root, input_paths


def exact_bbox_masks(image_path: str, bboxes: list[list[float]], *_: object) -> list[np.ndarray]:
    with Image.open(image_path) as image:
        width, height = image.size
    masks: list[np.ndarray] = []
    for x1, y1, x2, y2 in bboxes:
        mask = np.zeros((height, width), dtype=bool)
        mask[int(y1):int(y2), int(x1):int(x2)] = True
        masks.append(mask)
    return masks


def masks_with_one_empty(image_path: str, bboxes: list[list[float]], *args: object) -> list[np.ndarray]:
    masks = exact_bbox_masks(image_path, bboxes, *args)
    if Path(image_path).stem == "2":
        masks[0][:] = False
    return masks


def _run_and_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)
    result = stage.run_da3_footprint(str(dataset), save_root)
    return json.loads(Path(result["report_path"]).read_text())


def test_stage_fuses_global_id_views_and_uses_polygon_union(monkeypatch, tmp_path):
    dataset, save_root, input_paths = make_metric_fixture(tmp_path)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)
    before = {path: path.read_bytes() for path in input_paths}

    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    assert report["metric"] == "da3_ground_footprint_union"
    assert report["status"] == "accepted"
    assert report["value_m2"] == pytest.approx(1.5, abs=0.03)
    assert report["per_global_id"]["1"]["observations_used"] == 2
    assert report["per_global_id"]["2"]["height_median_m"] > 0.7
    assert all(path.read_bytes() == content for path, content in before.items())
    output_names = {path.name for path in Path(result["report_path"]).parent.iterdir()}
    assert output_names == {"measurement_report.json", "footprints.geojson", "top_down_footprint.png"}


def test_stage_rejects_total_when_one_global_id_has_no_mask(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", masks_with_one_empty)

    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    assert result["success"] is False
    assert report["status"] == "rejected"
    assert report["value_m2"] is None
    assert "empty" in report["per_global_id"]["2"]["observations"][0]["rejection"]
    geojson = json.loads((Path(result["report_path"]).parent / "footprints.geojson").read_text())
    assert geojson["status"] == "rejected"
    assert geojson["measurement_complete"] is False
    assert geojson["features"] == []


@pytest.mark.parametrize("mutation", ["stale_hash", "legacy_schema", "incomplete_mapping"])
def test_stage_rejects_cache_or_mapping_contract(monkeypatch, tmp_path, mutation):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    output = save_root / dataset.name
    cache_path = output / "da3_cache" / "predictions.npz"
    if mutation == "stale_hash":
        _write_image(dataset / "images" / "0.png", value=201)
    elif mutation == "legacy_schema":
        with np.load(cache_path, allow_pickle=False) as cache:
            fields = {key: cache[key] for key in cache.files if key != "cache_schema_version"}
        np.savez_compressed(cache_path, **fields)
    else:
        (output / "dedup_detections" / "global_mapping.json").write_text(
            json.dumps({"1": [{"image_id": 0, "object_id": 0, "bbox": [16, 24, 80, 104]}]})
        )
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)

    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    assert result["success"] is False
    assert report["status"] == "rejected"
    assert report["value_m2"] is None
    assert report["rejection_reason"]


def test_runner_affine_uses_pixel_centres():
    affine = _source_to_processed_affines(
        [(100, 50)], output_height=42, output_width=84, process_res=84
    )[0]

    assert affine[0, 0] == pytest.approx(0.84)
    assert affine[0, 2] == pytest.approx((0.84 - 1.0) / 2.0)
    assert affine[1, 1] == pytest.approx(0.84)
    assert affine[1, 2] == pytest.approx((0.84 - 1.0) / 2.0)


def test_runner_preprocess_geometry_records_input_processor_dimensions():
    geometry = _preprocess_geometry(100, 50, process_res=84, output_height=42, output_width=84)

    assert geometry["processed_width"] == 84
    assert geometry["processed_height"] == 42
    assert np.allclose(geometry["affine"], [[0.84, 0.0, -0.08], [0.0, 0.84, -0.08]])


def test_stage_rejects_current_numeric_image_and_detection_absent_from_cache(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    _write_image(dataset / "images" / "3.png")
    (dataset / "detections_results" / "3.json").write_text(json.dumps({"objects": []}))
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)

    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    assert result["success"] is False
    assert report["value_m2"] is None
    assert "numeric source image ids" in report["rejection_reason"]


def test_stage_converts_sam_runtime_error_to_rejected_report(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(tmp_path)

    def sam_failure(*_: object) -> list[np.ndarray]:
        raise RuntimeError("SAM GPU failure")

    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", sam_failure)
    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    assert result["success"] is False
    assert report["status"] == "rejected"
    assert "SAM3 failed" in report["rejection_reason"]


def test_runner_rejects_unsafe_model_id_before_inference():
    with pytest.raises(ValueError, match="safe model id"):
        _validate_model_id("model id with spaces")


@pytest.mark.parametrize(
    "mutation",
    ["singular_affine", "reflection_affine", "translated_affine", "missing_provenance", "bad_method"],
)
def test_stage_rejects_unverified_affine_or_preprocess_provenance(monkeypatch, tmp_path, mutation):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    cache_path = save_root / dataset.name / "da3_cache" / "predictions.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        fields = {key: cache[key] for key in cache.files}
    affine = fields["source_to_processed_affine"].copy()
    if mutation == "singular_affine":
        affine[0, 0, 0] = 0.0
    elif mutation == "reflection_affine":
        affine[0, 0, 0] = -1.0
    elif mutation == "translated_affine":
        affine[0, 0, 2] += 10.0
    elif mutation == "missing_provenance":
        fields.pop("preprocess_resolution")
    else:
        fields["preprocess_method"] = np.asarray("unknown_method", dtype="<U32")
    if mutation in {"singular_affine", "reflection_affine", "translated_affine"}:
        fields["source_to_processed_affine"] = affine
    np.savez_compressed(cache_path, **fields)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)

    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    assert result["success"] is False
    assert report["status"] == "rejected"
    assert report["value_m2"] is None
    assert "preprocess" in report["rejection_reason"] or "affine" in report["rejection_reason"]


@pytest.mark.parametrize("mutation", ["affine_unicode", "sizes_unicode"])
def test_stage_rejects_non_numeric_affine_or_non_integer_source_sizes(monkeypatch, tmp_path, mutation):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    cache_path = save_root / dataset.name / "da3_cache" / "predictions.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        fields = {key: cache[key] for key in cache.files}
    if mutation == "affine_unicode":
        fields["source_to_processed_affine"] = np.full((3, 2, 3), "1", dtype="<U2")
    else:
        fields["source_image_sizes"] = np.full((3, 2), "126", dtype="<U4")
    np.savez_compressed(cache_path, **fields)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)

    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    assert result["success"] is False
    assert report["status"] == "rejected"
    assert report["value_m2"] is None
    assert "dtype" in report["rejection_reason"]
