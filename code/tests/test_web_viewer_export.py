import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.web_viewer_export import export_web_viewer_bundle


_BUNDLE_FILES = {
    "manifest.json",
    "positions.f32.bin",
    "colors.u8.bin",
    "confidences.f32.bin",
    "frame_ids.i32.bin",
    "objects.json",
    "footprints.json",
}


def _cache_provenance(*, source_model: str = "depth-anything/DA3NESTED-GIANT-LARGE"):
    return {
        "schema_version": 2,
        "source_model": source_model,
        "affine_convention": "pixel_center_v1",
        "preprocess_resolution": 2,
        "preprocess_method": "upper_bound_resize",
        "frame_count": 1,
        "processed_size": [2, 2],
        "image_ids": [7],
        "source_image_sha256": ["0" * 64],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cache(
    path: Path,
    *,
    points: np.ndarray,
    confidence: np.ndarray,
    image_ids: np.ndarray | None = None,
    images: np.ndarray | None = None,
) -> None:
    frame_count, height, width, _ = points.shape
    images = (
        np.asarray(
            [[[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]]],
            dtype=np.uint8,
        )
        if images is None
        else images
    )
    assert images.shape == (frame_count, height, width, 3)
    np.savez_compressed(
        path,
        cache_schema_version=np.asarray(2, dtype=np.int32),
        source_model=np.asarray("depth-anything/DA3NESTED-GIANT-LARGE", dtype="<U64"),
        image_ids=np.asarray([7], dtype=np.int32) if image_ids is None else image_ids,
        world_points=points,
        world_points_conf=confidence,
        images=images,
        source_image_sizes=np.asarray([[2, 2]], dtype=np.int32),
        source_to_processed_affine=np.asarray(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32
        ),
        source_image_sha256=np.asarray(["0" * 64], dtype="<U64"),
        affine_convention=np.asarray("pixel_center_v1", dtype="<U32"),
        preprocess_resolution=np.asarray(2, dtype=np.int32),
        preprocess_method=np.asarray("upper_bound_resize", dtype="<U32"),
    )


def _write_mapping(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "11": [
                    {
                        "image_id": 7,
                        "object_id": 3,
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                        "removed": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_footprint_generation(
    root: Path,
    *,
    status: str,
    cache_provenance: dict | None = None,
) -> None:
    run_id = "a" * 32
    generation = root / "runs" / run_id
    generation.mkdir(parents=True, exist_ok=True)
    accepted = status == "accepted"
    report = {
        "metric": "da3_ground_footprint_union",
        "unit": "m2",
        "status": status,
        "value_m2": 1.0 if accepted else None,
        "rejection_reason": None if accepted else "formal input rejected",
        "cache": _cache_provenance() if cache_provenance is None else cache_provenance,
        "plane": {
            "selected": (
                {
                    "point": [1.0, 2.0, 3.0],
                    "u_axis": [1.0, 0.0, 0.0],
                    "v_axis": [0.0, 1.0, 0.0],
                    "normal": [0.0, 0.0, 1.0],
                }
                if accepted
                else None
            )
        },
    }
    features = []
    if accepted:
        square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [square]},
                "properties": {"global_id": "11", "area_m2": 0.5},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [square]},
                "properties": {"global_id": "union", "area_m2": 1.0},
            },
        ]
    (generation / "measurement_report.json").write_text(
        json.dumps(report, allow_nan=False), encoding="utf-8"
    )
    (generation / "footprints.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "coordinate_space": "local_support_plane_meters",
                "status": status,
                "measurement_complete": accepted,
                "features": features,
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (generation / "top_down_footprint.png").write_bytes(b"not-a-real-png")
    artifact_names = (
        "measurement_report.json",
        "footprints.geojson",
        "top_down_footprint.png",
    )
    (generation / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "run_id": run_id,
                "sha256": {name: _sha256(generation / name) for name in artifact_names},
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (root / "CURRENT").write_text(
        json.dumps({"complete": True, "run_id": run_id}), encoding="utf-8"
    )


@pytest.fixture
def exporter_inputs(tmp_path: Path) -> dict[str, Path]:
    cache_path = tmp_path / "predictions.npz"
    _write_cache(
        cache_path,
        points=np.asarray(
            [
                [
                    [[1.000, 1.0, 1.0], [1.001, 1.0, 1.0]],
                    [[2.000, 2.0, 2.0], [0.000, 0.0, 0.0]],
                ]
            ],
            dtype=np.float32,
        ),
        confidence=np.asarray([[[1.0, 2.0], [0.5, 2.0]]], dtype=np.float32),
    )
    mapping_path = tmp_path / "global_mapping.json"
    _write_mapping(mapping_path)
    footprint_root = tmp_path / "ground_stack_footprint"
    _write_footprint_generation(footprint_root, status="accepted")
    return {
        "da3_cache_path": cache_path,
        "global_mapping_path": mapping_path,
        "footprint_root": footprint_root,
        "output_dir": tmp_path / "bundle",
    }


def test_accepted_generation_exports_fixed_bundle_and_exact_binary_lengths(exporter_inputs):
    result = export_web_viewer_bundle(**exporter_inputs, voxel_size_m=0.01, max_points=10)
    output_dir = exporter_inputs["output_dir"]
    generation = Path(result["manifest_path"]).parent

    assert set(path.name for path in generation.iterdir()) == _BUNDLE_FILES
    assert json.loads((output_dir / "CURRENT").read_text(encoding="utf-8")) == {
        "complete": True,
        "run_id": generation.name,
        "schema_version": "1.0.0",
    }
    assert result == {
        "output_dir": str(output_dir),
        "manifest_path": str(generation / "manifest.json"),
        "point_count": 1,
        "footprint_status": "accepted",
    }
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["point_count"] == 1
    assert manifest["arrays"] == {
        "positions": {"path": "positions.f32.bin", "dtype": "float32", "components": 3, "byte_length": 12},
        "colors": {"path": "colors.u8.bin", "dtype": "uint8", "components": 3, "byte_length": 3},
        "confidences": {"path": "confidences.f32.bin", "dtype": "float32", "components": 1, "byte_length": 4},
        "frame_ids": {"path": "frame_ids.i32.bin", "dtype": "int32", "components": 1, "byte_length": 4},
    }
    assert (generation / "positions.f32.bin").stat().st_size == 12
    assert (generation / "colors.u8.bin").stat().st_size == 3
    assert (generation / "confidences.f32.bin").stat().st_size == 4
    assert (generation / "frame_ids.i32.bin").stat().st_size == 4


def test_rejected_generation_preserves_null_value_and_exports_no_footprint_geometry(exporter_inputs):
    _write_footprint_generation(exporter_inputs["footprint_root"], status="rejected")

    result = export_web_viewer_bundle(**exporter_inputs)

    footprints = json.loads(Path(result["manifest_path"]).with_name("footprints.json").read_text(encoding="utf-8"))
    assert result["footprint_status"] == "rejected"
    assert footprints["status"] == "rejected"
    assert footprints["value_m2"] is None
    assert footprints["per_global_id"] == {}
    assert footprints["union"] is None
    assert footprints["support_plane"] is None


@pytest.mark.parametrize("corruption", ["manifest", "current"])
def test_corrupt_formal_generation_fails_closed(exporter_inputs, corruption):
    root = exporter_inputs["footprint_root"]
    if corruption == "manifest":
        (root / "runs" / ("a" * 32) / "manifest.json").write_text("{}", encoding="utf-8")
    else:
        (root / "CURRENT").write_text("{}", encoding="utf-8")

    with pytest.raises(OSError):
        export_web_viewer_bundle(**exporter_inputs)

    assert not exporter_inputs["output_dir"].exists()


def test_voxel_sampling_keeps_highest_confidence_valid_point(exporter_inputs):
    export_web_viewer_bundle(**exporter_inputs, voxel_size_m=0.01, max_points=10)
    output_dir = Path(
        export_web_viewer_bundle(**exporter_inputs, voxel_size_m=0.01, max_points=10)[
            "manifest_path"
        ]
    ).parent

    positions = np.fromfile(output_dir / "positions.f32.bin", dtype="<f4").reshape(-1, 3)
    colors = np.fromfile(output_dir / "colors.u8.bin", dtype=np.uint8).reshape(-1, 3)
    confidences = np.fromfile(output_dir / "confidences.f32.bin", dtype="<f4")
    frame_ids = np.fromfile(output_dir / "frame_ids.i32.bin", dtype="<i4")
    np.testing.assert_allclose(positions, [[1.001, 1.0, 1.0]], rtol=0.0, atol=1e-7)
    assert colors.tolist() == [[40, 50, 60]]
    np.testing.assert_allclose(confidences, [2.0], rtol=0.0, atol=0.0)
    assert frame_ids.tolist() == [7]


def test_complete_cache_metadata_must_match_formal_report_provenance(exporter_inputs):
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="accepted",
        cache_provenance=_cache_provenance(source_model="different-formal-cache"),
    )

    with pytest.raises(ValueError, match="provenance"):
        export_web_viewer_bundle(**exporter_inputs)


def test_missing_formal_schema_v2_metadata_fails_closed(exporter_inputs):
    with np.load(exporter_inputs["da3_cache_path"], allow_pickle=False) as loaded:
        fields = {name: loaded[name] for name in loaded.files if name != "preprocess_method"}
    np.savez_compressed(exporter_inputs["da3_cache_path"], **fields)

    with pytest.raises(ValueError, match="missing required schema-v2 fields"):
        export_web_viewer_bundle(**exporter_inputs)


@pytest.mark.parametrize("image_id", [np.iinfo(np.int32).min - 1, np.iinfo(np.int32).max + 1])
def test_image_id_outside_int32_range_is_rejected(exporter_inputs, image_id):
    with np.load(exporter_inputs["da3_cache_path"], allow_pickle=False) as loaded:
        points = loaded["world_points"]
        confidence = loaded["world_points_conf"]
        images = loaded["images"]
    _write_cache(
        exporter_inputs["da3_cache_path"],
        points=points,
        confidence=confidence,
        image_ids=np.asarray([image_id], dtype=np.int64),
        images=images,
    )

    with pytest.raises(ValueError, match="int32"):
        export_web_viewer_bundle(**exporter_inputs)


@pytest.mark.parametrize("array_name", ["world_points", "world_points_conf"])
def test_world_grids_must_be_float32(exporter_inputs, array_name):
    with np.load(exporter_inputs["da3_cache_path"], allow_pickle=False) as loaded:
        fields = {name: loaded[name] for name in loaded.files}
    fields[array_name] = fields[array_name].astype(np.float64)
    np.savez_compressed(exporter_inputs["da3_cache_path"], **fields)

    with pytest.raises(ValueError, match="float32"):
        export_web_viewer_bundle(**exporter_inputs)


def test_real_uint8_image_grid_is_accepted_and_non_uint8_is_rejected(exporter_inputs):
    accepted = export_web_viewer_bundle(**exporter_inputs)
    assert accepted["point_count"] == 1

    with np.load(exporter_inputs["da3_cache_path"], allow_pickle=False) as loaded:
        fields = {name: loaded[name] for name in loaded.files}
    fields["images"] = fields["images"].astype(np.float32)
    np.savez_compressed(exporter_inputs["da3_cache_path"], **fields)

    with pytest.raises(ValueError, match="images must be uint8"):
        export_web_viewer_bundle(**exporter_inputs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_model", np.asarray("unsafe model!", dtype="<U32"), "source_model"),
        ("source_image_sha256", np.asarray(["A" * 64], dtype="<U64"), "source_image_sha256"),
        (
            "source_to_processed_affine",
            np.asarray([[[1.0, 0.1, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32),
            "affine linear",
        ),
    ],
)
def test_malformed_formal_cache_metadata_is_rejected(exporter_inputs, field, value, message):
    with np.load(exporter_inputs["da3_cache_path"], allow_pickle=False) as loaded:
        fields = {name: loaded[name] for name in loaded.files}
    fields[field] = value
    np.savez_compressed(exporter_inputs["da3_cache_path"], **fields)

    with pytest.raises(ValueError, match=message):
        export_web_viewer_bundle(**exporter_inputs)


def test_repeated_exports_atomically_switch_current_and_preserve_generations(exporter_inputs):
    output_dir = exporter_inputs["output_dir"]
    output_dir.mkdir()
    (output_dir / "user-note.txt").write_text("preserve me", encoding="utf-8")

    first = export_web_viewer_bundle(**exporter_inputs)
    first_generation = Path(first["manifest_path"]).parent
    second = export_web_viewer_bundle(**exporter_inputs)
    second_generation = Path(second["manifest_path"]).parent

    assert first_generation != second_generation
    assert set(path.name for path in first_generation.iterdir()) == _BUNDLE_FILES
    assert set(path.name for path in second_generation.iterdir()) == _BUNDLE_FILES
    assert json.loads((output_dir / "CURRENT").read_text(encoding="utf-8")) == {
        "complete": True,
        "run_id": second_generation.name,
        "schema_version": "1.0.0",
    }
    assert (output_dir / "user-note.txt").read_text(encoding="utf-8") == "preserve me"
    assert {path.name for path in output_dir.iterdir()} == {"CURRENT", "runs", "user-note.txt"}
