import hashlib
import json
import multiprocessing
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
from utils import footprint_evidence as evidence_module


@pytest.fixture(autouse=True)
def _verified_sam3_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"task-4-verified-sam3-checkpoint")
    monkeypatch.setattr(stage, "_SAM3_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(stage, "_SAM3_DEVICE", "cpu")


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


def _formal_projection(report: dict[str, object]) -> dict[str, object]:
    return {
        "metric": report["metric"],
        "unit": report["unit"],
        "status": report["status"],
        "value_m2": report["value_m2"],
        "plane": report["plane"],
        "per_global_id": report["per_global_id"],
        "union": report["union"],
        "rejection_reason": report.get("rejection_reason"),
    }


def _published_projection(result: dict[str, object]) -> tuple[dict[str, object], str, np.ndarray]:
    generation = Path(str(result["report_path"])).parent
    report = json.loads((generation / "measurement_report.json").read_text())
    geojson = json.loads((generation / "footprints.geojson").read_text())
    canonical_geojson = json.dumps(geojson, sort_keys=True, separators=(",", ":"))
    with Image.open(generation / "top_down_footprint.png") as image:
        pixels = np.asarray(image.convert("RGBA")).copy()
    return _formal_projection(report), canonical_geojson, pixels


def _add_camera_fields(cache_path: Path, *, bad_contract: bool = False) -> None:
    with np.load(cache_path, allow_pickle=False) as loaded:
        fields = {key: loaded[key] for key in loaded.files}
    world_points = fields["world_points"]
    frame_count, height, width, _ = world_points.shape
    depth = np.maximum(world_points[..., 2:3], 0.01).astype(np.float64)
    intrinsic = np.repeat(np.eye(3, dtype=np.float64)[None], frame_count, axis=0)
    extrinsic = np.repeat(
        np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)[None],
        frame_count,
        axis=0,
    )
    if bad_contract:
        intrinsic[0, 0, 0] = 0.0
    fields.update(
        {
            "depth": depth.reshape(frame_count, height, width, 1),
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
        }
    )
    np.savez_compressed(cache_path, **fields)


def _run_evidence_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    masks=exact_bbox_masks,
    camera: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    cache_path = save_root / dataset.name / "da3_cache" / "predictions.npz"
    if camera is not None:
        _add_camera_fields(cache_path, bad_contract=camera == "bad")
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", masks)
    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())
    return result, report


def _rejected_publication_report() -> dict[str, object]:
    return {"status": "rejected", "per_global_id": {}}


def _assert_complete_current(output_root: Path) -> tuple[bytes, dict[str, str]]:
    current_bytes = (output_root / "CURRENT").read_bytes()
    resolved = stage._artifact_paths_from_current(output_root)
    generation = Path(resolved["measurement_report"]).parent
    assert {path.name for path in generation.iterdir()} == {
        "measurement_report.json",
        "footprints.geojson",
        "top_down_footprint.png",
        "manifest.json",
    }
    return current_bytes, resolved


def _publish_marker_process(
    output_root: str,
    marker: str,
    queue: object,
    *,
    block_after_replace: bool,
    replaced: object,
    release: object,
    generation_ready: object,
    finished: object,
) -> None:
    """Run the real publisher while process A is paused after CURRENT replace."""
    original_write_generation = stage._write_fsynced_generation

    def observed_write_generation(*args: object, **kwargs: object) -> Path:
        generation = original_write_generation(*args, **kwargs)
        generation_ready.set()
        return generation

    stage._write_fsynced_generation = observed_write_generation
    if block_after_replace:
        original_replace = stage._atomic_replace_current

        def blocked_replace(*args: object, **kwargs: object) -> None:
            original_replace(*args, **kwargs)
            replaced.set()
            if not release.wait(timeout=20):
                raise RuntimeError("publication interleave release timed out")

        stage._atomic_replace_current = blocked_replace
    try:
        paths = stage._publish_generation(
            Path(output_root),
            {"status": "rejected", "per_global_id": {}, "marker": marker},
            {},
            None,
        )
        report = json.loads(Path(paths["measurement_report"]).read_text())
        queue.put(("ok", marker, report["marker"], paths))
    except BaseException as error:
        queue.put(("error", marker, repr(error), {}))
    finally:
        finished.set()


def _publish_two_processes_with_forced_replace_resolve_interleaving(
    output_root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    first_replaced = context.Event()
    release_first = context.Event()
    first_generation_ready = context.Event()
    second_generation_ready = context.Event()
    first_finished = context.Event()
    second_finished = context.Event()
    first = context.Process(
        target=_publish_marker_process,
        kwargs={
            "output_root": str(output_root),
            "marker": "first",
            "queue": queue,
            "block_after_replace": True,
            "replaced": first_replaced,
            "release": release_first,
            "generation_ready": first_generation_ready,
            "finished": first_finished,
        },
    )
    second = context.Process(
        target=_publish_marker_process,
        kwargs={
            "output_root": str(output_root),
            "marker": "second",
            "queue": queue,
            "block_after_replace": False,
            "replaced": context.Event(),
            "release": context.Event(),
            "generation_ready": second_generation_ready,
            "finished": second_finished,
        },
    )
    first.start()
    assert first_generation_ready.wait(timeout=20)
    assert first_replaced.wait(timeout=20)
    second.start()
    assert second_generation_ready.wait(timeout=20)
    second_finished.wait(timeout=1)
    release_first.set()
    for process in (first, second):
        process.join(timeout=20)
        assert process.exitcode == 0
    results = [queue.get(timeout=2), queue.get(timeout=2)]
    assert all(item[0] == "ok" for item in results), results
    by_marker = {item[1]: {"returned_marker": item[2], "paths": item[3]} for item in results}
    current_paths = stage._artifact_paths_from_current(output_root)
    current_report = json.loads(Path(current_paths["measurement_report"]).read_text())
    return by_marker["first"], by_marker["second"], current_report


def _install_publication_boundary_probe(
    monkeypatch: pytest.MonkeyPatch, output_root: Path, failure_point: str
) -> None:
    error = OSError(f"injected {failure_point}")
    if failure_point in {
        "output_root_fsync_after_runs_mkdir",
        "runs_root_fsync_after_generation_rename",
    }:
        target = (
            output_root
            if failure_point == "output_root_fsync_after_runs_mkdir"
            else output_root / "runs"
        )
        original_fsync_directory = stage._fsync_directory

        def fail_target_directory_fsync(path: Path) -> None:
            if path == target:
                raise error
            original_fsync_directory(path)

        monkeypatch.setattr(stage, "_fsync_directory", fail_target_directory_fsync)
        return

    target_name = (
        "measurement_report.json"
        if failure_point == "measurement_report_write"
        else "manifest.json"
    )
    original_write_text = Path.write_text

    def fail_target_write(path: Path, *args: object, **kwargs: object) -> int:
        if path.name == target_name:
            raise error
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_target_write)


def _raise_if_sam3_called(*_args: object, **_kwargs: object) -> list[np.ndarray]:
    raise AssertionError("a valid persistent mask-cache hit must not invoke SAM3")


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
    generation = Path(result["report_path"]).parent
    output_root = generation.parent.parent
    output_names = {path.name for path in generation.iterdir()}
    assert output_names == {
        "measurement_report.json",
        "footprints.geojson",
        "top_down_footprint.png",
        "manifest.json",
    }
    assert (output_root / "CURRENT").is_file()
    assert (output_root / "runs").is_dir()
    assert report["artifacts"] == {
        "measurement_report": "measurement_report.json",
        "footprints_geojson": "footprints.geojson",
        "top_down_footprint_png": "top_down_footprint.png",
    }


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


def test_no_formal_plane_keeps_unavailable_shadow_schema_and_rejected_artifacts(
    monkeypatch, tmp_path
):
    """Catches bypassing the shadow builder when formal plane selection rejects."""
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    cache_path = save_root / dataset.name / "da3_cache" / "predictions.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        fields = {key: cache[key] for key in cache.files}
    world_points = np.zeros_like(fields["world_points"])
    _put_carton(world_points, 0, [16, 24, 80, 104], (0.0, 1.0), 0.02)
    _put_carton(world_points, 1, [18, 22, 82, 102], (0.0, 1.0), 0.03)
    _put_carton(world_points, 2, [62, 24, 126, 104], (0.5, 1.5), 0.80)
    fields["world_points"] = world_points
    np.savez_compressed(cache_path, **fields)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)

    first_result = stage.run_da3_footprint(str(dataset), save_root)
    first_report = json.loads(Path(first_result["report_path"]).read_text())
    first_projection = _published_projection(first_result)
    second_result = stage.run_da3_footprint(str(dataset), save_root)
    second_projection = _published_projection(second_result)

    assert first_result["success"] is False
    assert first_report["status"] == "rejected"
    assert first_report["value_m2"] is None
    assert first_report["plane"] == {"candidates": [], "selected": None}
    assert first_report["evidence"]["status"] == "unavailable_no_formal_geometry"
    assert (
        first_report["evidence"]["mask_robustness"]["status"]
        == "unavailable_no_formal_geometry"
    )
    assert first_projection[0] == second_projection[0]
    assert first_projection[1] == second_projection[1]
    assert np.array_equal(first_projection[2], second_projection[2])
    assert json.loads(first_projection[1]) == {
        "type": "FeatureCollection",
        "coordinate_space": "local_support_plane_meters",
        "status": "rejected",
        "measurement_complete": False,
        "features": [],
    }


@pytest.mark.parametrize("formal_status", ["accepted", "rejected"])
def test_evidence_failures_do_not_change_frozen_formal_result(
    monkeypatch, tmp_path, formal_status
):
    baseline_root = tmp_path / f"baseline-{formal_status}"
    observed_root = tmp_path / f"observed-{formal_status}"
    masks = exact_bbox_masks if formal_status == "accepted" else masks_with_one_empty
    builder_calls: list[str] = []

    def baseline_builder(**_kwargs: object) -> dict[str, object]:
        builder_calls.append("baseline")
        return {"mode": "shadow", "status": "baseline_evidence"}

    monkeypatch.setattr(stage, "build_shadow_evidence", baseline_builder, raising=False)
    baseline_result, _ = _run_evidence_fixture(
        baseline_root, monkeypatch, masks=masks
    )
    baseline_projection = _published_projection(baseline_result)

    def failed_builder(**_kwargs: object) -> dict[str, object]:
        builder_calls.append("failed")
        raise RuntimeError("boom")

    monkeypatch.setattr(stage, "build_shadow_evidence", failed_builder, raising=False)
    observed_result, observed = _run_evidence_fixture(
        observed_root, monkeypatch, masks=masks
    )
    observed_projection = _published_projection(observed_result)

    assert observed_projection[:2] == baseline_projection[:2]
    assert np.array_equal(observed_projection[2], baseline_projection[2])
    assert observed["evidence"]["status"] == "failed_evidence"
    assert observed["evidence"]["reason"] == "boom"
    assert observed["evidence"]["mask_robustness"] == {
        "status": "failed_evidence",
        "reason": "boom",
    }
    assert builder_calls == ["baseline", "failed"]


def test_stage_reports_valid_shadow_evidence_without_changing_formal_area(
    monkeypatch, tmp_path
):
    without_camera_result, without_camera = _run_evidence_fixture(
        tmp_path / "without-camera", monkeypatch
    )
    with_camera_result, with_camera = _run_evidence_fixture(
        tmp_path / "with-camera", monkeypatch, camera="valid"
    )

    assert _formal_projection(with_camera) == _formal_projection(without_camera)
    assert with_camera["evidence"]["status"] == "available"
    assert without_camera["evidence"]["status"] == (
        "unavailable_missing_camera_fields"
    )
    observation = with_camera["evidence"]["per_global_id"]["1"]["observations"][0]
    assert observation["source_mask_pixel_count"] == 64 * 80
    assert observation["processed_mask_pixel_count"] == 64 * 80
    with_projection = _published_projection(with_camera_result)
    without_projection = _published_projection(without_camera_result)
    assert with_projection[:2] == without_projection[:2]
    assert np.array_equal(with_projection[2], without_projection[2])


def test_stage_reports_failed_camera_contract_without_changing_formal_area(
    monkeypatch, tmp_path
):
    baseline_result, baseline = _run_evidence_fixture(
        tmp_path / "camera-baseline", monkeypatch
    )
    observed_result, observed = _run_evidence_fixture(
        tmp_path / "bad-camera", monkeypatch, camera="bad"
    )

    assert observed["evidence"]["status"] == "failed_camera_contract"
    assert _formal_projection(observed) == _formal_projection(baseline)
    observed_projection = _published_projection(observed_result)
    baseline_projection = _published_projection(baseline_result)
    assert observed_projection[:2] == baseline_projection[:2]
    assert np.array_equal(observed_projection[2], baseline_projection[2])


def test_mask_robustness_failure_does_not_change_formal_projection_or_artifacts(
    monkeypatch, tmp_path
):
    baseline_result, baseline = _run_evidence_fixture(
        tmp_path / "robustness-baseline", monkeypatch, camera="valid"
    )
    baseline_projection = _published_projection(baseline_result)

    def fail_source_erosion(*_args: object, **_kwargs: object) -> np.ndarray:
        raise RuntimeError("injected source morphology failure")

    monkeypatch.setattr(evidence_module.cv2, "erode", fail_source_erosion)
    observed_result, observed = _run_evidence_fixture(
        tmp_path / "robustness-failure", monkeypatch, camera="valid"
    )
    observed_projection = _published_projection(observed_result)

    assert observed["status"] == baseline["status"] == "accepted"
    assert observed["evidence"]["status"] == "available"
    robustness = observed["evidence"]["mask_robustness"]
    assert robustness["status"] == "available"
    assert robustness["variants"]["eroded"]["status"] == "rejected"
    assert robustness["variants"]["eroded"]["value_m2"] is None
    assert "injected source morphology failure" in robustness["variants"]["eroded"]["reason"]
    assert "polygons" not in robustness["variants"]["eroded"]
    assert observed_projection[:2] == baseline_projection[:2]
    assert np.array_equal(observed_projection[2], baseline_projection[2])


def test_duplicate_observation_does_not_increase_distinct_image_count(
    monkeypatch, tmp_path
):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    detection_path = dataset / "detections_results" / "0.json"
    detections = json.loads(detection_path.read_text())
    duplicate_bbox = list(detections["objects"][0]["position"])
    detections["objects"].append({"position": duplicate_bbox})
    detection_path.write_text(json.dumps(detections))
    mapping_path = save_root / dataset.name / "dedup_detections" / "global_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    mapping["1"].append(
        {"image_id": 0, "object_id": 1, "bbox": duplicate_bbox}
    )
    mapping_path.write_text(json.dumps(mapping))
    _add_camera_fields(save_root / dataset.name / "da3_cache" / "predictions.npz")
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)

    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    per_id = report["evidence"]["per_global_id"]["1"]
    assert report["status"] == "accepted"
    assert len(per_id["observations"]) == 3
    assert per_id["distinct_image_id_count"] == 2
    assert all(
        pair["source_image_id"] != pair["target_image_id"]
        for pair in per_id["pairs"]
    )


def test_wrong_mask_is_the_highest_leave_one_out_influence(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    mapping_path = save_root / dataset.name / "dedup_detections" / "global_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    mapping["1"].extend(mapping.pop("2"))
    mapping_path.write_text(json.dumps(mapping))
    _add_camera_fields(save_root / dataset.name / "da3_cache" / "predictions.npz")

    def masks_with_wrong_third_view(
        image_path: str, bboxes: list[list[float]], *args: object
    ) -> list[np.ndarray]:
        masks = exact_bbox_masks(image_path, bboxes, *args)
        if Path(image_path).stem == "2":
            x1, y1, x2, y2 = (int(value) for value in bboxes[0])
            masks[0][:] = False
            masks[0][y1:y2, (x1 + x2) // 2:x2] = True
        return masks

    monkeypatch.setattr(
        stage, "sam3_masks_from_bboxes_predict_inst", masks_with_wrong_third_view
    )
    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    leave_one_out = report["evidence"]["per_global_id"]["1"][
        "leave_one_observation_out"
    ]
    available = [item for item in leave_one_out if item["status"] == "available"]
    highest_influence = min(available, key=lambda item: item["polygon_iou"])
    assert report["status"] == "accepted"
    assert highest_influence["image_id"] == 2
    assert highest_influence["polygon_iou"] < 0.8


def test_evidence_input_mutation_cannot_change_published_formal_artifacts(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        stage,
        "build_shadow_evidence",
        lambda **_kwargs: {"mode": "shadow", "status": "baseline_evidence"},
        raising=False,
    )
    baseline_result, _ = _run_evidence_fixture(
        tmp_path / "mutation-baseline", monkeypatch
    )
    baseline_projection = _published_projection(baseline_result)

    def mutating_builder(**kwargs: object) -> dict[str, object]:
        snapshot = kwargs["formal_snapshot"]
        snapshot.polygons.clear()
        assert snapshot.plane is not None
        snapshot.plane.point[:] = 10_000.0
        for observation in kwargs["observations"]:
            observation.processed_mask[:] = False
            observation.valid_mask[:] = False
        return {"mode": "shadow", "status": "mutated_inputs"}

    monkeypatch.setattr(stage, "build_shadow_evidence", mutating_builder, raising=False)
    observed_result, observed = _run_evidence_fixture(
        tmp_path / "mutation-observed", monkeypatch
    )
    observed_projection = _published_projection(observed_result)

    assert observed["evidence"]["status"] == "mutated_inputs"
    assert observed_projection[:2] == baseline_projection[:2]
    assert np.array_equal(observed_projection[2], baseline_projection[2])


def test_stage_second_run_uses_cached_masks_and_keeps_area(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)

    first_result = stage.run_da3_footprint(str(dataset), save_root)
    first = json.loads(Path(first_result["report_path"]).read_text())
    first_projection = _published_projection(first_result)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", _raise_if_sam3_called)
    second_result = stage.run_da3_footprint(str(dataset), save_root)
    second = json.loads(Path(second_result["report_path"]).read_text())
    second_projection = _published_projection(second_result)

    assert first_projection[:2] == second_projection[:2]
    assert np.array_equal(first_projection[2], second_projection[2])
    assert [frame["cache_events"] for frame in first["sam3_mask_cache"]["frames"]] == [
        ["miss", "written"],
        ["miss", "written"],
        ["miss", "written"],
    ]
    assert [frame["cache_events"] for frame in second["sam3_mask_cache"]["frames"]] == [
        ["hit"],
        ["hit"],
        ["hit"],
    ]
    assert {
        frame["checkpoint_sha256"] for frame in second["sam3_mask_cache"]["frames"]
    } == {_sha256(Path(stage._SAM3_CHECKPOINT))}


def test_hit_only_stage_hashes_checkpoint_only_at_entry_and_exit(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)
    stage.run_da3_footprint(str(dataset), save_root)
    expected_digest = _sha256(Path(stage._SAM3_CHECKPOINT))
    checksum_calls: list[Path] = []

    def counted_checkpoint_sha256(path: Path) -> str:
        checksum_calls.append(path)
        return expected_digest

    monkeypatch.setattr(stage, "checkpoint_sha256", counted_checkpoint_sha256)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", _raise_if_sam3_called)
    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    assert checksum_calls == [Path(stage._SAM3_CHECKPOINT), Path(stage._SAM3_CHECKPOINT)]
    assert result["success"] is True
    assert [frame["cache_events"] for frame in report["sam3_mask_cache"]["frames"]] == [
        ["hit"],
        ["hit"],
        ["hit"],
    ]


def test_exit_checkpoint_mismatch_rejects_after_recording_all_hit_events(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)
    stage.run_da3_footprint(str(dataset), save_root)
    expected_digest = _sha256(Path(stage._SAM3_CHECKPOINT))
    digests = iter((expected_digest, "f" * 64))

    monkeypatch.setattr(stage, "checkpoint_sha256", lambda _path: next(digests))
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", _raise_if_sam3_called)
    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    assert result["success"] is False
    assert report["status"] == "rejected"
    assert report["value_m2"] is None
    assert [frame["cache_events"] for frame in report["sam3_mask_cache"]["frames"]] == [
        ["hit"],
        ["hit"],
        ["hit"],
    ]
    assert "checkpoint changed" in report["rejection_reason"]


def test_cached_empty_mask_rejects_full_total(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", masks_with_one_empty)
    stage.run_da3_footprint(str(dataset), save_root)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", _raise_if_sam3_called)

    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    assert result["success"] is False
    assert report["status"] == "rejected"
    assert report["value_m2"] is None
    assert "empty" in report["per_global_id"]["2"]["observations"][0]["rejection"]
    assert all(frame["cache_events"] == ["hit"] for frame in report["sam3_mask_cache"]["frames"])


def test_cache_write_failure_uses_complete_fresh_masks_and_records_event(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(tmp_path)

    def fresh_masks_with_failed_write(request, compute_masks):
        del compute_masks
        masks = exact_bbox_masks(
            str(request.image_path),
            [list(prompt.bbox_xyxy()) for prompt in request.detections],
        )
        return stage.FrameMaskCacheResult(
            masks=tuple(masks),
            key="1" * 64,
            events=("miss", "cache_write_failed"),
            payload_sha256=None,
            checkpoint_sha256=request.checkpoint_sha256,
            code_fingerprint=request.code_fingerprint,
            invalid_reason="injected cache write failure",
        )

    monkeypatch.setattr(stage, "load_or_compute_frame_masks", fresh_masks_with_failed_write)
    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    assert result["success"] is True
    assert report["status"] == "accepted"
    assert report["value_m2"] == pytest.approx(1.5, abs=0.03)
    assert [frame["cache_events"] for frame in report["sam3_mask_cache"]["frames"]] == [
        ["miss", "cache_write_failed"],
        ["miss", "cache_write_failed"],
        ["miss", "cache_write_failed"],
    ]


@pytest.mark.parametrize(
    ("masks", "expected_status"),
    [(exact_bbox_masks, "accepted"), (masks_with_one_empty, "rejected")],
    ids=["accepted", "rejected"],
)
def test_stage_timing_and_cache_events_are_additive_and_json_safe(
    monkeypatch, tmp_path, masks, expected_status
):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", masks)

    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    expected_stages = {
        "validation_and_io",
        "sam3_source_masks",
        "select_support_plane",
        "per_id_obb_union",
        "shadow_evidence",
        "artifact_creation",
    }
    stages = report["performance"]["stages_seconds"]
    assert report["status"] == expected_status
    assert set(stages) == expected_stages
    assert all(value is None or value >= 0.0 for value in stages.values())
    assert all(stages[name] is not None for name in expected_stages)
    assert report["performance"]["total_seconds_pre_publication"] >= sum(
        value for value in stages.values() if value is not None
    )
    assert all(
        frame["cache_events"] == ["miss", "written"]
        for frame in report["sam3_mask_cache"]["frames"]
    )
    assert all("events" not in frame for frame in report["sam3_mask_cache"]["frames"])
    json.dumps(report, allow_nan=False)


def test_rejection_before_sam3_leaves_unentered_performance_stages_null(
    monkeypatch, tmp_path
):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    (save_root / dataset.name / "dedup_detections" / "global_mapping.json").write_text(
        json.dumps({"1": []})
    )
    monkeypatch.setattr(
        stage,
        "sam3_masks_from_bboxes_predict_inst",
        lambda *_args: (_ for _ in ()).throw(AssertionError("SAM3 must not run")),
    )

    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())
    stages = report["performance"]["stages_seconds"]

    assert report["status"] == "rejected"
    assert stages["validation_and_io"] >= 0.0
    assert stages["sam3_source_masks"] is None
    assert stages["select_support_plane"] is None
    assert stages["per_id_obb_union"] is None
    assert stages["shadow_evidence"] >= 0.0
    assert stages["artifact_creation"] >= 0.0
    json.dumps(report, allow_nan=False)


def test_cached_masks_are_not_requested_until_complete_inputs_are_validated(
    monkeypatch, tmp_path
):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    mapping_path = save_root / dataset.name / "dedup_detections" / "global_mapping.json"
    mapping_path.write_text(
        json.dumps({"1": [{"image_id": 0, "object_id": 0, "bbox": [16, 24, 80, 104]}]})
    )

    def fail_cache_request(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("mask cache must not run before complete mapping validation")

    monkeypatch.setattr(stage, "load_or_compute_frame_masks", fail_cache_request)
    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())

    assert result["success"] is False
    assert "mapping" in report["rejection_reason"]


def test_current_points_to_complete_single_artifact_generation(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)

    result = stage.run_da3_footprint(str(dataset), save_root)
    generation = Path(result["report_path"]).parent
    output_root = generation.parent.parent
    current = json.loads((output_root / "CURRENT").read_text())
    manifest = json.loads((generation / "manifest.json").read_text())

    assert current == {"complete": True, "run_id": generation.name}
    assert generation == output_root / "runs" / current["run_id"]
    assert {path.name for path in generation.iterdir()} == {
        "measurement_report.json",
        "footprints.geojson",
        "top_down_footprint.png",
        "manifest.json",
    }
    assert manifest["complete"] is True
    assert manifest["run_id"] == current["run_id"]
    assert manifest["sha256"] == {
        name: _sha256(generation / name)
        for name in (
            "measurement_report.json",
            "footprints.geojson",
            "top_down_footprint.png",
        )
    }


def test_two_publishers_return_own_generation_and_reader_sees_complete_current(
    tmp_path,
):
    """Catches resolving a competing publisher's CURRENT after releasing identity."""
    output_root = tmp_path / "ground_stack_footprint"
    output_root.mkdir()

    first, second, current = _publish_two_processes_with_forced_replace_resolve_interleaving(
        output_root
    )

    assert first["returned_marker"] == "first"
    assert second["returned_marker"] == "second"
    assert current["marker"] in {"first", "second"}
    for result in (first, second):
        paths = result["paths"]
        generation = Path(paths["measurement_report"]).parent
        assert {path.name for path in generation.iterdir()} == {
            "measurement_report.json",
            "footprints.geojson",
            "top_down_footprint.png",
            "manifest.json",
        }


def test_unlocked_current_resolver_rejects_another_expected_run(tmp_path):
    output_root = tmp_path / "ground_stack_footprint"
    output_root.mkdir()
    stage._publish_generation(output_root, _rejected_publication_report(), {}, None)

    with pytest.raises(OSError, match="expected run"):
        stage._artifact_paths_from_current_unlocked(
            output_root, expected_run_id="0" * 32
        )


def test_failed_first_generation_write_never_creates_current(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)

    def fail_generation_write(*_args: object, **_kwargs: object) -> Path:
        raise OSError("injected generation write failure")

    monkeypatch.setattr(stage, "_write_fsynced_generation", fail_generation_write)
    with pytest.raises(OSError, match="injected generation write failure"):
        stage.run_da3_footprint(str(dataset), save_root)

    output_root = save_root / dataset.name / "ground_stack_footprint"
    assert not (output_root / "CURRENT").exists()


@pytest.mark.parametrize(
    "failure_point",
    [
        "generation_write",
        "artifact_fsync",
        "generation_fsync",
        "generation_rename",
        "current_temp",
        "current_replace",
    ],
)
def test_pre_current_replace_failure_preserves_previous_complete_generation(
    monkeypatch, tmp_path, failure_point
):
    output_root = tmp_path / "ground_stack_footprint"
    output_root.mkdir()
    old_paths = stage._publish_generation(
        output_root, _rejected_publication_report(), {}, None
    )
    old_current, old_resolved = _assert_complete_current(output_root)
    assert old_resolved == old_paths

    if failure_point == "generation_write":
        monkeypatch.setattr(
            stage,
            "_write_artifacts",
            lambda *_args: (_ for _ in ()).throw(OSError("injected generation write failure")),
        )
    elif failure_point == "artifact_fsync":
        monkeypatch.setattr(
            stage,
            "_fsync_file",
            lambda *_args: (_ for _ in ()).throw(OSError("injected artifact fsync failure")),
        )
    elif failure_point == "generation_fsync":
        original_fsync_directory = stage._fsync_directory

        def fail_generation_fsync(path: Path) -> None:
            if path.parent == output_root / "runs" and path.name.startswith("."):
                raise OSError("injected generation fsync failure")
            original_fsync_directory(path)

        monkeypatch.setattr(stage, "_fsync_directory", fail_generation_fsync)
    elif failure_point == "generation_rename":
        monkeypatch.setattr(
            stage.os,
            "rename",
            lambda *_args: (_ for _ in ()).throw(OSError("injected generation rename failure")),
        )
    elif failure_point == "current_temp":
        monkeypatch.setattr(
            stage.tempfile,
            "mkstemp",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected CURRENT temp failure")),
        )
    else:
        monkeypatch.setattr(
            stage.os,
            "replace",
            lambda *_args: (_ for _ in ()).throw(OSError("injected CURRENT replace failure")),
        )

    with pytest.raises(OSError, match="injected"):
        stage._publish_generation(output_root, _rejected_publication_report(), {}, None)

    assert (output_root / "CURRENT").read_bytes() == old_current
    assert stage._artifact_paths_from_current(output_root) == old_paths


@pytest.mark.parametrize("has_previous_current", [False, True], ids=["first", "replacement"])
@pytest.mark.parametrize(
    "failure_point",
    [
        "output_root_fsync_after_runs_mkdir",
        "measurement_report_write",
        "manifest_write",
        "runs_root_fsync_after_generation_rename",
    ],
)
def test_pre_replace_boundary_failure_preserves_old_or_no_current(
    monkeypatch, tmp_path, failure_point, has_previous_current
):
    output_root = tmp_path / "ground_stack_footprint"
    output_root.mkdir()
    old_current: bytes | None = None
    old_paths: dict[str, str] | None = None
    if has_previous_current:
        old_paths = stage._publish_generation(
            output_root, _rejected_publication_report(), {}, None
        )
        old_current, resolved = _assert_complete_current(output_root)
        assert resolved == old_paths

    _install_publication_boundary_probe(monkeypatch, output_root, failure_point)
    with pytest.raises(OSError, match=failure_point):
        stage._publish_generation(output_root, _rejected_publication_report(), {}, None)

    if has_previous_current:
        assert old_current is not None
        assert old_paths is not None
        assert (output_root / "CURRENT").read_bytes() == old_current
        assert stage._artifact_paths_from_current(output_root) == old_paths
    else:
        assert not (output_root / "CURRENT").exists()


def test_post_current_replace_directory_fsync_warns_and_returns_new_generation(
    monkeypatch, tmp_path, caplog
):
    output_root = tmp_path / "ground_stack_footprint"
    output_root.mkdir()
    old_paths = stage._publish_generation(
        output_root, _rejected_publication_report(), {}, None
    )
    output_root_fsyncs = 0
    original_fsync_directory = stage._fsync_directory

    def fail_post_replace_fsync(path: Path) -> None:
        nonlocal output_root_fsyncs
        if path == output_root:
            output_root_fsyncs += 1
            if output_root_fsyncs == 2:
                raise OSError("injected post-replace directory fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(stage, "_fsync_directory", fail_post_replace_fsync)
    caplog.set_level("WARNING")
    new_paths = stage._publish_generation(
        output_root, _rejected_publication_report(), {}, None
    )

    assert new_paths != old_paths
    assert stage._artifact_paths_from_current(output_root) == new_paths
    assert "CURRENT was replaced" in caplog.text
    assert "durability" in caplog.text


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
    assert report["value_m2"] is None
    assert "SAM3 failed" in report["rejection_reason"]
    geojson = json.loads((Path(result["report_path"]).parent / "footprints.geojson").read_text())
    assert geojson["measurement_complete"] is False
    assert geojson["features"] == []


def test_nonempty_mask_without_valid_da3_points_rejects_total(monkeypatch, tmp_path):
    dataset, save_root, _ = make_metric_fixture(tmp_path)
    cache_path = save_root / dataset.name / "da3_cache" / "predictions.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        fields = {key: cache[key] for key in cache.files}
    world_points = fields["world_points"].copy()
    world_points[2, 24:104, 62:126] = 0.0
    fields["world_points"] = world_points
    np.savez_compressed(cache_path, **fields)
    monkeypatch.setattr(stage, "sam3_masks_from_bboxes_predict_inst", exact_bbox_masks)

    result = stage.run_da3_footprint(str(dataset), save_root)
    report = json.loads(Path(result["report_path"]).read_text())
    observation = report["per_global_id"]["2"]["observations"][0]
    geojson = json.loads((Path(result["report_path"]).parent / "footprints.geojson").read_text())

    assert result["success"] is False
    assert report["status"] == "rejected"
    assert report["value_m2"] is None
    assert observation["valid_point_count"] == 0
    assert "fewer than 32 valid DA3 points" in observation["rejection"]
    assert geojson["measurement_complete"] is False
    assert geojson["features"] == []


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
