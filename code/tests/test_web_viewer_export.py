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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cache(path: Path, *, points: np.ndarray, confidence: np.ndarray) -> None:
    frame_count, height, width, _ = points.shape
    images = np.asarray(
        [[[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]]],
        dtype=np.uint8,
    )
    assert images.shape == (frame_count, height, width, 3)
    np.savez_compressed(
        path,
        cache_schema_version=np.asarray(2, dtype=np.int32),
        source_model=np.asarray("depth-anything/DA3NESTED-GIANT-LARGE", dtype="<U64"),
        image_ids=np.asarray([7], dtype=np.int32),
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


def _write_footprint_generation(root: Path, *, status: str) -> None:
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

    assert set(path.name for path in output_dir.iterdir()) == _BUNDLE_FILES
    assert result == {
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "manifest.json"),
        "point_count": 1,
        "footprint_status": "accepted",
    }
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["point_count"] == 1
    assert manifest["arrays"] == {
        "positions": {"path": "positions.f32.bin", "dtype": "float32", "components": 3, "byte_length": 12},
        "colors": {"path": "colors.u8.bin", "dtype": "uint8", "components": 3, "byte_length": 3},
        "confidences": {"path": "confidences.f32.bin", "dtype": "float32", "components": 1, "byte_length": 4},
        "frame_ids": {"path": "frame_ids.i32.bin", "dtype": "int32", "components": 1, "byte_length": 4},
    }
    assert (output_dir / "positions.f32.bin").stat().st_size == 12
    assert (output_dir / "colors.u8.bin").stat().st_size == 3
    assert (output_dir / "confidences.f32.bin").stat().st_size == 4
    assert (output_dir / "frame_ids.i32.bin").stat().st_size == 4


def test_rejected_generation_preserves_null_value_and_exports_no_footprint_geometry(exporter_inputs):
    _write_footprint_generation(exporter_inputs["footprint_root"], status="rejected")

    result = export_web_viewer_bundle(**exporter_inputs)

    footprints = json.loads(
        (exporter_inputs["output_dir"] / "footprints.json").read_text(encoding="utf-8")
    )
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
    output_dir = exporter_inputs["output_dir"]

    positions = np.fromfile(output_dir / "positions.f32.bin", dtype="<f4").reshape(-1, 3)
    colors = np.fromfile(output_dir / "colors.u8.bin", dtype=np.uint8).reshape(-1, 3)
    confidences = np.fromfile(output_dir / "confidences.f32.bin", dtype="<f4")
    frame_ids = np.fromfile(output_dir / "frame_ids.i32.bin", dtype="<i4")
    np.testing.assert_allclose(positions, [[1.001, 1.0, 1.0]], rtol=0.0, atol=1e-7)
    assert colors.tolist() == [[40, 50, 60]]
    np.testing.assert_allclose(confidences, [2.0], rtol=0.0, atol=0.0)
    assert frame_ids.tolist() == [7]
