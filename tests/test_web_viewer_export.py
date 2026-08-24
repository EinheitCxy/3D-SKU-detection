import hashlib
import io
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module
from src.web_viewer_export import WebViewerExportError, export_web_viewer_bundle
from utils.pointcloud_filter import PointCloudFilterConfig, filter_scene_points

_BUNDLE_FILES = {
    "manifest.json",
    "positions.f32.bin",
    "colors.u8.bin",
    "normals.i8.bin",
    "objects.json",
    "footprints.json",
}


def _make_source_image_bytes() -> bytes:
    image = Image.new("RGB", (4, 4))
    image.putdata(
        [
            (10, 20, 30),
            (40, 50, 60),
            (70, 80, 90),
            (100, 110, 120),
            (13, 26, 39),
            (52, 65, 78),
            (91, 104, 117),
            (130, 143, 156),
            (17, 34, 51),
            (68, 85, 102),
            (119, 136, 153),
            (170, 187, 204),
            (21, 42, 63),
            (84, 105, 126),
            (147, 168, 189),
            (210, 231, 252),
        ]
    )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


_SOURCE_IMAGE_BYTES = _make_source_image_bytes()
_SOURCE_IMAGE_SHA256 = hashlib.sha256(_SOURCE_IMAGE_BYTES).hexdigest()


def _write_source_image(path: Path, payload: bytes = _SOURCE_IMAGE_BYTES) -> None:
    path.write_bytes(payload)


def _cache_provenance(
    *,
    source_model: str = "depth-anything/DA3NESTED-GIANT-LARGE",
    source_image_sha256: list[str] | None = None,
):
    return {
        "schema_version": 2,
        "source_model": source_model,
        "affine_convention": "pixel_center_v1",
        "preprocess_resolution": 2,
        "preprocess_method": "upper_bound_resize",
        "frame_count": 1,
        "processed_size": [2, 2],
        "image_ids": [7],
        "source_image_sha256": (
            [_SOURCE_IMAGE_SHA256]
            if source_image_sha256 is None
            else source_image_sha256
        ),
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
    source_image_sha256: list[str] | None = None,
    source_image_sizes: list[list[int]] | None = None,
    extrinsic: np.ndarray | None = None,
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
        cache_schema_version=np.asarray(3, dtype=np.int32),
        is_metric=np.asarray(1, dtype=np.int32),
        scale_factor=np.asarray(1.0, dtype=np.float32),
        source_model=np.asarray("depth-anything/DA3NESTED-GIANT-LARGE", dtype="<U64"),
        image_ids=np.asarray([7], dtype=np.int32) if image_ids is None else image_ids,
        world_points=points,
        world_points_conf=confidence,
        images=images,
        extrinsic=(
            np.asarray(
                [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
                dtype=np.float32,
            )
            if extrinsic is None
            else extrinsic
        ),
        source_image_sizes=np.asarray(
            source_image_sizes if source_image_sizes is not None else [[4, 4]],
            dtype=np.int32,
        ),
        source_to_processed_affine=np.asarray(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32
        ),
        source_image_sha256=np.asarray(
            (
                [_SOURCE_IMAGE_SHA256]
                if source_image_sha256 is None
                else source_image_sha256
            ),
            dtype="<U64",
        ),
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


def _bbox_hex(bbox: list[float]) -> list[str]:
    return [struct.pack(">d", float(value)).hex() for value in bbox]


def _write_sam3_mask_entry(
    root: Path,
    image_sha256: str,
    detections: list[tuple[int, list[float]]],
    masks: np.ndarray,
) -> None:
    """写一个 sam3_mask_cache/v1 entry，结构与 utils/sam3_mask_cache.py 产出一致。"""
    key_payload = {
        "schema": "sam3_frame_mask_cache_v2_canonical_bbox_clip",
        "image": {"image_id": 7, "sha256": image_sha256, "size_bytes": 16},
        "detections": [
            {"object_id": object_id, "bbox_xyxy_f64be_hex": _bbox_hex(bbox)}
            for object_id, bbox in detections
        ],
        "checkpoint_sha256": "0" * 64,
        "code_fingerprint": {},
        "runtime_fingerprint": {},
        "inference_contract": {},
        "output_contract": {
            "shape_hw": [int(masks.shape[1]), int(masks.shape[2])],
            "dtype": "bool",
        },
    }
    key = hashlib.sha256(
        json.dumps(
            key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    entry_dir = root / "entries" / key
    entry_dir.mkdir(parents=True, exist_ok=True)
    payload_path = entry_dir / "masks.npz"
    np.savez_compressed(payload_path, masks=masks)
    manifest = {
        "complete": True,
        "key": key,
        "key_payload": key_payload,
        "payload_sha256": _sha256(payload_path),
        "masks": [
            {
                "sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
                "true_pixel_count": int(mask.sum()),
            }
            for mask in masks
        ],
    }
    (entry_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_footprint_generation(
    root: Path,
    *,
    status: str,
    cache_provenance: dict | None = None,
    global_mapping_path: Path,
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
        "global_mapping_sha256": _sha256(global_mapping_path),
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
                "properties": {
                    "coordinate_space": "local_support_plane_meters",
                    "global_id": "11",
                    "area_m2": 0.5,
                    "observations_used": 1,
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [square]},
                "properties": {
                    "coordinate_space": "local_support_plane_meters",
                    "global_id": "union",
                    "area_m2": 1.0,
                },
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


def _refresh_formal_manifest(generation: Path) -> None:
    artifact_names = (
        "measurement_report.json",
        "footprints.geojson",
        "top_down_footprint.png",
    )
    (generation / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "run_id": generation.name,
                "sha256": {name: _sha256(generation / name) for name in artifact_names},
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def exporter_inputs(tmp_path: Path) -> dict[str, Path]:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_source_image(images_dir / "7.JPG")
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
    mask_cache_root = tmp_path / "sam3_mask_cache" / "v1"
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 2:4, 1:3] = True  # bbox [1,2,3,4] 内的商品轮廓
    _write_sam3_mask_entry(
        mask_cache_root, _SOURCE_IMAGE_SHA256, [(3, [1.0, 2.0, 3.0, 4.0])], masks
    )
    mapping_path = tmp_path / "global_mapping.json"
    _write_mapping(mapping_path)
    footprint_root = tmp_path / "ground_stack_footprint"
    _write_footprint_generation(
        footprint_root, status="accepted", global_mapping_path=mapping_path
    )
    return {
        "da3_cache_path": cache_path,
        "global_mapping_path": mapping_path,
        "footprint_root": footprint_root,
        "output_dir": tmp_path / "bundle",
        "source_images_dir": images_dir,
        "sam3_mask_cache_root": mask_cache_root,
    }


def test_export_writes_world_to_view_and_instance_point_ranges(exporter_inputs):
    result = export_web_viewer_bundle(**exporter_inputs)
    generation = Path(result["manifest_path"]).parent
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))

    world_to_view = manifest["world_to_view"]
    assert len(world_to_view) == 16
    assert all(
        isinstance(value, float) and np.isfinite(value) for value in world_to_view
    )
    matrix = np.asarray(world_to_view, dtype=np.float64).reshape(4, 4)
    np.testing.assert_allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-9)
    assert isinstance(manifest["coordinate_convention"], str)
    assert "world_to_view" in manifest["coordinate_convention"]
    objects = json.loads((generation / "objects.json").read_text(encoding="utf-8"))
    for entry in objects.values():
        for instance in entry["instances"]:
            start, end = instance["point_index_range"]
            assert 0 <= start <= end <= manifest["point_count"]


def test_export_manifest_writes_exact_robust_display_bounds(exporter_inputs):
    result = export_web_viewer_bundle(**exporter_inputs)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    expected = np.percentile(
        np.asarray([[1.001, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        [1.0, 99.0],
        axis=0,
    )
    np.testing.assert_allclose(
        manifest["display_bounds"], [*expected[0], *expected[1]], rtol=0.0, atol=0.0
    )


def test_instance_point_range_covers_only_bounded_grid_points(exporter_inputs):
    mapping_path = exporter_inputs["global_mapping_path"]
    mapping_path.write_text(
        json.dumps(
            {
                "11": [
                    {
                        "image_id": 7,
                        "object_id": 3,
                        "bbox": [0.0, 0.0, 2.0, 2.0],
                        "removed": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="accepted",
        global_mapping_path=mapping_path,
    )
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True  # 覆盖 2x2 网格映射到的源像素
    _write_sam3_mask_entry(
        exporter_inputs["sam3_mask_cache_root"],
        _SOURCE_IMAGE_SHA256,
        [(3, [0.0, 0.0, 2.0, 2.0])],
        masks,
    )

    result = export_web_viewer_bundle(**exporter_inputs)
    generation = Path(result["manifest_path"]).parent
    objects = json.loads((generation / "objects.json").read_text(encoding="utf-8"))
    assert objects["11"]["instances"][0]["point_index_range"] == [0, 2]
    positions = np.fromfile(generation / "positions.f32.bin", dtype="<f4").reshape(
        -1, 3
    )
    np.testing.assert_allclose(
        positions, [[1.001, 1.0, 1.0], [2.0, 2.0, 2.0]], rtol=0.0, atol=1e-7
    )


def test_instance_referencing_unknown_image_fails_closed(exporter_inputs):
    mapping_path = exporter_inputs["global_mapping_path"]
    mapping_path.write_text(
        json.dumps(
            {
                "11": [
                    {
                        "image_id": 99,
                        "object_id": 3,
                        "bbox": [0.0, 0.0, 2.0, 2.0],
                        "removed": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="accepted",
        global_mapping_path=mapping_path,
    )

    with pytest.raises(ValueError, match="absent from cache"):
        export_web_viewer_bundle(**exporter_inputs)


@pytest.mark.parametrize("bbox", [[3.0, 2.0, 1.0, 4.0], [1.0, 2.0, float("nan"), 4.0]])
def test_malformed_mapping_object_bbox_fails_before_bundle_publication(
    exporter_inputs, bbox
):
    mapping_path = exporter_inputs["global_mapping_path"]
    mapping_path.write_text(
        json.dumps(
            {"11": [{"image_id": 7, "object_id": 3, "bbox": bbox, "removed": False}]}
        ),
        encoding="utf-8",
    )
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="accepted",
        global_mapping_path=mapping_path,
    )
    with pytest.raises(ValueError, match="object index|bbox"):
        export_web_viewer_bundle(**exporter_inputs)
    assert not exporter_inputs["output_dir"].exists()


@pytest.mark.parametrize("field", ["image_id", "object_id"])
def test_mapping_ids_outside_javascript_safe_integer_fail_before_bundle_publication(
    exporter_inputs, field
):
    mapping_path = exporter_inputs["global_mapping_path"]
    instance = {
        "image_id": 7,
        "object_id": 3,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "removed": False,
    }
    instance[field] = 2**53
    mapping_path.write_text(json.dumps({"11": [instance]}), encoding="utf-8")
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="accepted",
        global_mapping_path=mapping_path,
    )
    with pytest.raises(ValueError, match="identity"):
        export_web_viewer_bundle(**exporter_inputs)
    assert not exporter_inputs["output_dir"].exists()


def test_level_rotation_skips_degenerate_plane_without_warning(monkeypatch):
    from src.web_viewer_export import _fit_level_rotation
    import sys
    import types

    class FakePointCloud:
        def __init__(self):
            self.points = [0, 1, 2]

        def compute_nearest_neighbor_distance(self):
            return [1.0]

        def segment_plane(self, *_args, **_kwargs):
            return [0.0, 0.0, 0.0, 0.0], [0, 1, 2]

        def select_by_index(self, *_args, **_kwargs):
            return self

    monkeypatch.setitem(
        sys.modules,
        "open3d",
        types.SimpleNamespace(
            geometry=types.SimpleNamespace(PointCloud=FakePointCloud),
            utility=types.SimpleNamespace(Vector3dVector=lambda values: values),
        ),
    )
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rotation, leveled = _fit_level_rotation(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.asarray(
                [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
            ),
        )
    assert not caught
    np.testing.assert_allclose(rotation, np.diag([1.0, -1.0, -1.0]))
    assert not leveled


def test_accepted_generation_exports_fixed_bundle_and_exact_binary_lengths(
    exporter_inputs,
):
    result = export_web_viewer_bundle(
        **exporter_inputs, voxel_size_m=0.01, max_points=10
    )
    output_dir = exporter_inputs["output_dir"]
    generation = Path(result["manifest_path"]).parent

    assert set(path.name for path in generation.iterdir()) == _BUNDLE_FILES | {"thumbs"}
    assert json.loads((output_dir / "CURRENT").read_text(encoding="utf-8")) == {
        "complete": True,
        "run_id": generation.name,
        "schema_version": "1.0.0",
    }
    assert result == {
        "output_dir": str(output_dir),
        "manifest_path": str(generation / "manifest.json"),
        "point_count": 2,
        "footprint_status": "accepted",
        "thumbnail_count": 1,
    }
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["point_count"] == 2
    assert manifest["arrays"] == {
        "positions": {
            "path": "positions.f32.bin",
            "dtype": "float32",
            "components": 3,
            "byte_length": 24,
        },
        "colors": {
            "path": "colors.u8.bin",
            "dtype": "uint8",
            "components": 3,
            "byte_length": 6,
        },
        "normals": {
            "path": "normals.i8.bin",
            "dtype": "int8",
            "components": 3,
            "byte_length": 6,
        },
    }
    assert (generation / "positions.f32.bin").stat().st_size == 24
    assert (generation / "colors.u8.bin").stat().st_size == 6
    assert (generation / "normals.i8.bin").stat().st_size == 6


def test_export_writes_quantized_unit_normals(exporter_inputs):
    result = export_web_viewer_bundle(**exporter_inputs)
    generation = Path(result["manifest_path"]).parent
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))

    normals = np.fromfile(generation / "normals.i8.bin", dtype=np.int8).reshape(-1, 3)
    assert normals.shape == (manifest["point_count"], 3)
    magnitudes = np.linalg.norm(normals.astype(np.float64) / 127.0, axis=1)
    assert np.all(magnitudes > 0.9)
    assert np.all(np.abs(magnitudes - 1.0) <= np.sqrt(3.0) / 127.0)
    # Normals must be row-aligned with positions through the voxel/sort path.
    assert normals.dtype == np.int8


def test_normals_are_estimated_only_for_final_representatives(
    exporter_inputs, monkeypatch
):
    import src.web_viewer_export as exporter

    observed_lengths = []

    def estimate(points, _extrinsic):
        observed_lengths.append(len(points))
        return points / np.linalg.norm(points, axis=1, keepdims=True)

    monkeypatch.setattr(exporter, "_estimate_scene_normals", estimate)

    result = exporter.export_web_viewer_bundle(**exporter_inputs)
    assert observed_lengths == [result["point_count"]]
    generation = Path(result["manifest_path"]).parent
    positions = np.fromfile(generation / "positions.f32.bin", dtype="<f4").reshape(
        -1, 3
    )
    normals = (
        np.fromfile(generation / "normals.i8.bin", dtype=np.int8)
        .reshape(-1, 3)
        .astype(np.float64)
        / 127.0
    )
    expected = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    np.testing.assert_allclose(normals, expected, atol=0.5 / 127.0 + 1e-12, rtol=0.0)


def test_normals_follow_final_representatives_through_nonidentity_label_sort(
    exporter_inputs, monkeypatch
):
    import src.web_viewer_export as exporter

    _write_cache(
        exporter_inputs["da3_cache_path"],
        points=np.asarray(
            [[[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [3.0, 1.0, 1.0], [4.0, 1.0, 1.0]]]],
            dtype=np.float32,
        ),
        confidence=np.asarray([[[1.0, 1.0, 1.0, 1.0]]], dtype=np.float32),
        images=np.asarray(
            [[[[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]]]],
            dtype=np.uint8,
        ),
    )
    mapping_path = exporter_inputs["global_mapping_path"]
    mapping_path.write_text(
        json.dumps(
            {
                "11": [
                    {
                        "image_id": 7,
                        "object_id": 3,
                        "bbox": [2.0, 0.0, 3.0, 1.0],
                        "removed": False,
                    }
                ],
                "12": [
                    {
                        "image_id": 7,
                        "object_id": 4,
                        "bbox": [0.0, 0.0, 4.0, 1.0],
                        "removed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    provenance = _cache_provenance()
    provenance["processed_size"] = [4, 1]
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="rejected",
        cache_provenance=provenance,
        global_mapping_path=mapping_path,
    )
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0, 2] = True
    masks[1, 0, 0] = True
    masks[1, 0, 3] = True
    _write_sam3_mask_entry(
        exporter_inputs["sam3_mask_cache_root"],
        _SOURCE_IMAGE_SHA256,
        [(3, [2.0, 0.0, 3.0, 1.0]), (4, [0.0, 0.0, 4.0, 1.0])],
        masks,
    )
    observed = []

    def estimate(points, _extrinsic):
        observed.append(points.copy())
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
        )[(points[:, 0].astype(int) - 1)]

    monkeypatch.setattr(exporter, "_estimate_scene_normals", estimate)
    original_argsort = exporter.np.argsort
    source_labels = []

    def argsort(values, *args, **kwargs):
        source_labels.append(np.asarray(values).copy())
        return original_argsort(values, *args, **kwargs)

    monkeypatch.setattr(exporter.np, "argsort", argsort)

    result = exporter.export_web_viewer_bundle(
        **exporter_inputs, filter_config=PointCloudFilterConfig(enabled=False)
    )
    generation = Path(result["manifest_path"]).parent
    positions = np.fromfile(generation / "positions.f32.bin", dtype="<f4").reshape(
        -1, 3
    )
    normals = np.fromfile(generation / "normals.i8.bin", dtype=np.int8).reshape(-1, 3)
    np.testing.assert_allclose(
        observed[0], [[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1]]
    )
    np.testing.assert_array_equal(source_labels[-1], [1, -1, 0, 1])
    np.testing.assert_array_equal(
        original_argsort(source_labels[-1], kind="stable"), [1, 2, 0, 3]
    )
    np.testing.assert_allclose(positions, [[2, 1, 1], [3, 1, 1], [1, 1, 1], [4, 1, 1]])
    np.testing.assert_array_equal(
        np.fromfile(generation / "colors.u8.bin", dtype=np.uint8).reshape(-1, 3),
        [[40, 50, 60], [70, 80, 90], [10, 20, 30], [100, 110, 120]],
    )
    np.testing.assert_allclose(
        normals.astype(np.float64) / 127.0,
        [[0, 1, 0], [0, 0, 1], [1, 0, 0], [-1, 0, 0]],
        atol=0.5 / 127.0 + 1e-12,
        rtol=0.0,
    )


def test_export_manifest_records_filter_and_input_provenance(exporter_inputs):
    filter_config = PointCloudFilterConfig(enabled=False, min_points=42)
    result = export_web_viewer_bundle(**exporter_inputs, filter_config=filter_config)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    export_source = manifest["source"]["export"]
    entry_dir = next((exporter_inputs["sam3_mask_cache_root"] / "entries").iterdir())
    entry_manifest = json.loads(
        (entry_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert export_source["filter_config"] == filter_config.__dict__
    assert export_source["exporter_source_sha256"] == _sha256(
        Path(__file__).resolve().parents[1] / "src" / "web_viewer_export.py"
    )
    assert export_source["global_mapping_sha256"] == _sha256(
        exporter_inputs["global_mapping_path"]
    )
    assert export_source["sam3_mask_entries"] == [
        {
            "image_id": 7,
            "key": entry_dir.name,
            "payload_sha256": entry_manifest["payload_sha256"],
        }
    ]


def test_rejected_generation_preserves_null_value_and_exports_no_footprint_geometry(
    exporter_inputs,
):
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="rejected",
        global_mapping_path=exporter_inputs["global_mapping_path"],
    )

    result = export_web_viewer_bundle(**exporter_inputs)

    footprints = json.loads(
        Path(result["manifest_path"])
        .with_name("footprints.json")
        .read_text(encoding="utf-8")
    )
    assert result["footprint_status"] == "rejected"
    assert footprints["status"] == "rejected"
    assert footprints["value_m2"] is None
    assert footprints["per_global_id"] == {}
    assert footprints["union"] is None
    assert footprints["support_plane"] is None


@pytest.mark.parametrize("status", ["accepted", "rejected"])
def test_mapping_digest_mismatch_rejects_before_any_bundle_publication(
    exporter_inputs, status
):
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status=status,
        global_mapping_path=exporter_inputs["global_mapping_path"],
    )
    generation = exporter_inputs["footprint_root"] / "runs" / ("a" * 32)
    report_path = generation / "measurement_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["global_mapping_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    _refresh_formal_manifest(generation)

    with pytest.raises(ValueError, match="mapping"):
        export_web_viewer_bundle(**exporter_inputs)

    assert not exporter_inputs["output_dir"].exists()


def test_accepted_object_index_and_footprint_ids_must_match_before_publication(
    exporter_inputs,
):
    generation = exporter_inputs["footprint_root"] / "runs" / ("a" * 32)
    geojson_path = generation / "footprints.geojson"
    geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
    geojson["features"][0]["properties"]["global_id"] = "12"
    geojson_path.write_text(json.dumps(geojson, allow_nan=False), encoding="utf-8")
    _refresh_formal_manifest(generation)

    with pytest.raises(ValueError, match="ID set"):
        export_web_viewer_bundle(**exporter_inputs)

    assert not exporter_inputs["output_dir"].exists()


@pytest.mark.parametrize(
    "corruption",
    ["accepted_value", "per_id_properties", "union_area", "rejected_reason"],
)
def test_footprint_bundle_contract_rejects_malformed_properties_and_value_relations(
    exporter_inputs, corruption
):
    status = "rejected" if corruption == "rejected_reason" else "accepted"
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status=status,
        global_mapping_path=exporter_inputs["global_mapping_path"],
    )
    generation = exporter_inputs["footprint_root"] / "runs" / ("a" * 32)
    report_path = generation / "measurement_report.json"
    geojson_path = generation / "footprints.geojson"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
    if corruption == "accepted_value":
        report["value_m2"] = -1.0
    elif corruption == "per_id_properties":
        geojson["features"][0]["properties"] = {
            "coordinate_space": "local_support_plane_meters",
            "global_id": "11",
            "area_m2": 0.5,
            "observations_used": True,
            "unexpected": "not in browser contract",
        }
    elif corruption == "union_area":
        geojson["features"][1]["properties"]["area_m2"] = 0.5
    else:
        report["rejection_reason"] = " "
    report_path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    geojson_path.write_text(json.dumps(geojson, allow_nan=False), encoding="utf-8")
    _refresh_formal_manifest(generation)

    with pytest.raises(ValueError):
        export_web_viewer_bundle(**exporter_inputs)

    assert not exporter_inputs["output_dir"].exists()


@pytest.mark.parametrize("corruption", ["manifest", "current"])
def test_corrupt_formal_generation_fails_closed(exporter_inputs, corruption):
    root = exporter_inputs["footprint_root"]
    if corruption == "manifest":
        (root / "runs" / ("a" * 32) / "manifest.json").write_text(
            "{}", encoding="utf-8"
        )
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

    positions = np.fromfile(output_dir / "positions.f32.bin", dtype="<f4").reshape(
        -1, 3
    )
    colors = np.fromfile(output_dir / "colors.u8.bin", dtype=np.uint8).reshape(-1, 3)
    np.testing.assert_allclose(
        positions, [[1.001, 1.0, 1.0], [2.0, 2.0, 2.0]], rtol=0.0, atol=1e-7
    )
    assert colors.tolist() == [[40, 50, 60], [70, 80, 90]]


def test_voxel_sampling_keeps_highest_confidence_point_regardless_of_sam3_label(
    exporter_inputs,
):
    """SAM3 labels annotate points but cannot override generic voxel sampling."""
    points = np.asarray(
        [[[[1.000, 1.0, 1.0], [1.001, 1.0, 1.0], [1.002, 1.0, 1.0]]]],
        dtype=np.float32,
    )
    _write_cache(
        exporter_inputs["da3_cache_path"],
        points=points,
        confidence=np.asarray([[[0.1, 0.2, 99.0]]], dtype=np.float32),
        images=np.asarray(
            [[[[10, 20, 30], [40, 50, 60], [70, 80, 90]]]], dtype=np.uint8
        ),
    )
    mapping_path = exporter_inputs["global_mapping_path"]
    mapping_path.write_text(
        json.dumps(
            {
                "11": [
                    {
                        "image_id": 7,
                        "object_id": 3,
                        "bbox": [0.0, 0.0, 1.0, 1.0],
                        "removed": False,
                    },
                    {
                        "image_id": 7,
                        "object_id": 4,
                        "bbox": [1.0, 0.0, 2.0, 1.0],
                        "removed": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    provenance = _cache_provenance()
    provenance["processed_size"] = [3, 1]
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="accepted",
        cache_provenance=provenance,
        global_mapping_path=mapping_path,
    )
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, 0, 1] = True
    _write_sam3_mask_entry(
        exporter_inputs["sam3_mask_cache_root"],
        _SOURCE_IMAGE_SHA256,
        [(3, [0.0, 0.0, 1.0, 1.0]), (4, [1.0, 0.0, 2.0, 1.0])],
        masks,
    )

    result = export_web_viewer_bundle(
        **exporter_inputs, voxel_size_m=0.01, max_points=10
    )
    generation = Path(result["manifest_path"]).parent
    positions = np.fromfile(generation / "positions.f32.bin", dtype="<f4").reshape(
        -1, 3
    )
    np.testing.assert_allclose(positions[:, 0], [1.002], atol=1e-7)


def test_labeled_points_do_not_bypass_max_points_sampling(exporter_inputs):
    mapping_path = exporter_inputs["global_mapping_path"]
    mapping_path.write_text(
        json.dumps(
            {
                "11": [
                    {
                        "image_id": 7,
                        "object_id": 3,
                        "bbox": [0.0, 0.0, 2.0, 2.0],
                        "removed": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="accepted",
        global_mapping_path=mapping_path,
    )
    masks = np.zeros((1, 4, 4), dtype=bool)
    masks[0, 0, 0] = True
    masks[0, 1, 0] = True
    _write_sam3_mask_entry(
        exporter_inputs["sam3_mask_cache_root"],
        _SOURCE_IMAGE_SHA256,
        [(3, [0.0, 0.0, 2.0, 2.0])],
        masks,
    )

    result = export_web_viewer_bundle(
        **exporter_inputs, voxel_size_m=0.01, max_points=1
    )
    assert result["point_count"] == 1


def test_complete_cache_metadata_must_match_formal_report_provenance(exporter_inputs):
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="accepted",
        cache_provenance=_cache_provenance(source_model="different-formal-cache"),
        global_mapping_path=exporter_inputs["global_mapping_path"],
    )

    with pytest.raises(ValueError, match="provenance"):
        export_web_viewer_bundle(**exporter_inputs)


def test_missing_cache_required_field_fails_closed(exporter_inputs):
    with np.load(exporter_inputs["da3_cache_path"], allow_pickle=False) as loaded:
        fields = {
            name: loaded[name] for name in loaded.files if name != "preprocess_method"
        }
    np.savez_compressed(exporter_inputs["da3_cache_path"], **fields)

    with pytest.raises(ValueError, match="missing required schema-v3 fields"):
        export_web_viewer_bundle(**exporter_inputs)


@pytest.mark.parametrize(
    "image_id", [np.iinfo(np.int32).min - 1, np.iinfo(np.int32).max + 1]
)
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
    assert accepted["point_count"] == 2

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
        (
            "source_image_sha256",
            np.asarray(["A" * 64], dtype="<U64"),
            "source_image_sha256",
        ),
        (
            "source_to_processed_affine",
            np.asarray([[[1.0, 0.1, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32),
            "affine linear",
        ),
    ],
)
def test_malformed_formal_cache_metadata_is_rejected(
    exporter_inputs, field, value, message
):
    with np.load(exporter_inputs["da3_cache_path"], allow_pickle=False) as loaded:
        fields = {name: loaded[name] for name in loaded.files}
    fields[field] = value
    np.savez_compressed(exporter_inputs["da3_cache_path"], **fields)

    with pytest.raises(ValueError, match=message):
        export_web_viewer_bundle(**exporter_inputs)


def test_repeated_exports_atomically_switch_current_and_preserve_generations(
    exporter_inputs,
):
    output_dir = exporter_inputs["output_dir"]
    output_dir.mkdir()
    (output_dir / "user-note.txt").write_text("preserve me", encoding="utf-8")

    first = export_web_viewer_bundle(**exporter_inputs)
    first_generation = Path(first["manifest_path"]).parent
    second = export_web_viewer_bundle(**exporter_inputs)
    second_generation = Path(second["manifest_path"]).parent

    assert first_generation != second_generation
    assert set(path.name for path in first_generation.iterdir()) == _BUNDLE_FILES | {
        "thumbs"
    }
    assert set(path.name for path in second_generation.iterdir()) == _BUNDLE_FILES | {
        "thumbs"
    }
    assert json.loads((output_dir / "CURRENT").read_text(encoding="utf-8")) == {
        "complete": True,
        "run_id": second_generation.name,
        "schema_version": "1.0.0",
    }
    assert (output_dir / "user-note.txt").read_text(encoding="utf-8") == "preserve me"
    assert {path.name for path in output_dir.iterdir()} == {
        "CURRENT",
        "runs",
        "user-note.txt",
    }


def test_thumbnails_exported_for_active_and_removed_instances(exporter_inputs):
    mapping_path = exporter_inputs["global_mapping_path"]
    mapping_path.write_text(
        json.dumps(
            {
                "11": [
                    {
                        "image_id": 7,
                        "object_id": 3,
                        "bbox": [0.0, 0.0, 2.0, 2.0],
                        "removed": False,
                    },
                    {
                        "image_id": 7,
                        "object_id": 4,
                        "bbox": [2.0, 2.0, 4.0, 4.0],
                        "removed": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="accepted",
        global_mapping_path=mapping_path,
    )
    masks = np.zeros((2, 4, 4), dtype=bool)
    _write_sam3_mask_entry(
        exporter_inputs["sam3_mask_cache_root"],
        _SOURCE_IMAGE_SHA256,
        [(3, [0.0, 0.0, 2.0, 2.0]), (4, [2.0, 2.0, 4.0, 4.0])],
        masks,
    )

    result = export_web_viewer_bundle(**exporter_inputs)
    generation = Path(result["manifest_path"]).parent
    objects = json.loads((generation / "objects.json").read_text(encoding="utf-8"))
    instances = objects["11"]["instances"]
    assert result["thumbnail_count"] == 2
    assert [instance["thumbnail"] for instance in instances] == [
        "thumbs/11_0.jpg",
        "thumbs/11_1.jpg",
    ]
    assert sorted(path.name for path in (generation / "thumbs").iterdir()) == [
        "11_0.jpg",
        "11_1.jpg",
    ]
    for instance in instances:
        thumb = generation / instance["thumbnail"]
        assert thumb.is_file()
        assert thumb.stat().st_size < 64 * 1024
        with Image.open(thumb) as image:
            assert image.format == "JPEG"
            assert max(image.size) <= 256


def test_missing_source_image_fails_closed(exporter_inputs):
    (exporter_inputs["source_images_dir"] / "7.JPG").unlink()

    with pytest.raises(ValueError, match="not found"):
        export_web_viewer_bundle(**exporter_inputs)

    assert not exporter_inputs["output_dir"].exists()


def test_missing_source_images_directory_fails_closed(exporter_inputs):
    shutil.rmtree(exporter_inputs["source_images_dir"])

    with pytest.raises(ValueError, match="source images directory"):
        export_web_viewer_bundle(**exporter_inputs)

    assert not exporter_inputs["output_dir"].exists()


def test_source_image_digest_mismatch_fails_closed(exporter_inputs):
    _write_source_image(
        exporter_inputs["source_images_dir"] / "7.JPG", payload=b"different-bytes"
    )

    with pytest.raises(ValueError, match="provenance"):
        export_web_viewer_bundle(**exporter_inputs)

    assert not exporter_inputs["output_dir"].exists()


def test_instance_bbox_outside_source_image_fails_closed(exporter_inputs):
    mapping_path = exporter_inputs["global_mapping_path"]
    mapping_path.write_text(
        json.dumps(
            {
                "11": [
                    {
                        "image_id": 7,
                        "object_id": 3,
                        "bbox": [0.0, 0.0, 99.0, 99.0],
                        "removed": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="accepted",
        global_mapping_path=mapping_path,
    )

    with pytest.raises(ValueError, match="exceeds source image bounds"):
        export_web_viewer_bundle(**exporter_inputs)

    assert not exporter_inputs["output_dir"].exists()


def test_same_shape_sam3_mask_tampering_fails_closed(exporter_inputs):
    entry_dir = next((exporter_inputs["sam3_mask_cache_root"] / "entries").iterdir())
    payload_path = entry_dir / "masks.npz"
    with np.load(payload_path, allow_pickle=False) as loaded:
        masks = loaded["masks"].copy()
    masks[0, 0, 0] = ~masks[0, 0, 0]
    np.savez_compressed(payload_path, masks=masks)
    manifest_path = entry_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_sha256"] = _sha256(payload_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(WebViewerExportError, match="per-mask"):
        export_web_viewer_bundle(**exporter_inputs)


def test_sam3_mask_payload_digest_mismatch_fails_closed(exporter_inputs):
    entry_dir = next((exporter_inputs["sam3_mask_cache_root"] / "entries").iterdir())
    manifest_path = entry_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(WebViewerExportError, match="payload SHA-256"):
        export_web_viewer_bundle(**exporter_inputs)


def test_sam3_mask_key_payload_tampering_fails_closed(exporter_inputs):
    entry_dir = next((exporter_inputs["sam3_mask_cache_root"] / "entries").iterdir())
    manifest_path = entry_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["key_payload"]["runtime_fingerprint"] = {"tampered": True}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        WebViewerExportError, match="key does not match canonical payload"
    ):
        export_web_viewer_bundle(**exporter_inputs)


def test_sam3_mask_entry_directory_tampering_fails_closed(exporter_inputs):
    entry_dir = next((exporter_inputs["sam3_mask_cache_root"] / "entries").iterdir())
    entry_dir.rename(entry_dir.with_name("f" * 64))

    with pytest.raises(
        WebViewerExportError, match="key does not match canonical payload"
    ):
        export_web_viewer_bundle(**exporter_inputs)


def test_sam3_mask_manifest_key_tampering_fails_closed(exporter_inputs):
    entry_dir = next((exporter_inputs["sam3_mask_cache_root"] / "entries").iterdir())
    manifest_path = entry_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["key"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        WebViewerExportError, match="key does not match canonical payload"
    ):
        export_web_viewer_bundle(**exporter_inputs)


def test_sam3_mask_pixels_outside_canonical_bbox_fail_closed(exporter_inputs):
    entry_dir = next((exporter_inputs["sam3_mask_cache_root"] / "entries").iterdir())
    payload_path = entry_dir / "masks.npz"
    with np.load(payload_path, allow_pickle=False) as loaded:
        masks = loaded["masks"].copy()
    masks[0] = False
    masks[0, 0, 0] = True
    np.savez_compressed(payload_path, masks=masks)
    manifest_path = entry_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_sha256"] = _sha256(payload_path)
    manifest["masks"] = [
        {
            "sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
            "true_pixel_count": int(mask.sum()),
        }
        for mask in masks
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(WebViewerExportError, match="outside its canonical bbox"):
        export_web_viewer_bundle(**exporter_inputs)


def test_scene_filter_removes_mask_labeled_and_background_outliers_equally(
    exporter_inputs,
):
    """SAM3 labels stay as metadata and never exempt a point from filtering."""
    image = Image.new("RGB", (128, 128), (90, 120, 150))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    (exporter_inputs["source_images_dir"] / "7.JPG").write_bytes(payload)

    height = width = 64
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    points = np.zeros((1, height, width, 3), dtype=np.float32)
    points[0, ..., 0] = xs * 0.01
    points[0, ..., 1] = ys * 0.01
    points[0, ..., 2] = 1.0
    points[0, 60:, 60:, :] += np.asarray([10.0, 10.0, 10.0], dtype=np.float32)
    _write_cache(
        exporter_inputs["da3_cache_path"],
        points=points,
        confidence=np.ones((1, height, width), dtype=np.float32),
        images=np.zeros((1, height, width, 3), dtype=np.uint8),
        source_image_sha256=[digest],
        source_image_sizes=[[128, 128]],
    )
    mapping_path = exporter_inputs["global_mapping_path"]
    mapping_path.write_text(
        json.dumps(
            {
                "11": [
                    {
                        "image_id": 7,
                        "object_id": 3,
                        "bbox": [60.0, 60.0, 64.0, 64.0],
                        "removed": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    provenance = _cache_provenance(source_image_sha256=[digest])
    provenance["processed_size"] = [width, height]
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="accepted",
        cache_provenance=provenance,
        global_mapping_path=mapping_path,
    )
    # mask 覆盖 4 个远端点；它们和相邻背景点都必须通过同一场景过滤。
    masks = np.zeros((1, 128, 128), dtype=bool)
    masks[0, 62:64, 62:64] = True
    _write_sam3_mask_entry(
        exporter_inputs["sam3_mask_cache_root"],
        digest,
        [(3, [60.0, 60.0, 64.0, 64.0])],
        masks,
    )

    flat = points.reshape(-1, 3).astype(np.float64)
    far_mask = flat[:, 0] > 5.0
    assert int(far_mask.sum()) == 16
    kept = filter_scene_points(
        flat, PointCloudFilterConfig(min_points=100, remove_ground=False)
    )
    assert not kept[far_mask].any(), "未保护的远端块会被场景过滤删除"

    result = export_web_viewer_bundle(
        **exporter_inputs,
        voxel_size_m=0.001,
        max_points=100_000,
        filter_config=PointCloudFilterConfig(min_points=100, remove_ground=False),
    )
    generation = Path(result["manifest_path"]).parent
    positions = np.fromfile(generation / "positions.f32.bin", dtype="<f4").reshape(
        -1, 3
    )
    assert int((positions[:, 0] > 5.0).sum()) == 0


def test_scene_filter_never_receives_a_sam3_protection_mask(monkeypatch, exporter_inputs):
    captured: dict[str, object] = {}

    def capture_filter(points, config, protect_mask=None):
        captured["point_count"] = len(points)
        captured["protect_mask"] = protect_mask
        return np.ones(len(points), dtype=bool)

    monkeypatch.setattr("src.web_viewer_export.filter_scene_points", capture_filter)

    export_web_viewer_bundle(**exporter_inputs)

    assert captured["point_count"] > 0
    assert captured["protect_mask"] is None


def test_sky_cut_removes_points_above_subject_top(exporter_inputs):
    """天空线裁剪：摆平后主体(实例)顶 p99.9 + 0.15m 以上的点必须被裁掉。

    天花板/天空薄片与主簇稠密相连，SOR/DBSCAN 剔不掉，只能按高度线裁。
    """
    image = Image.new("RGB", (8, 8), (60, 80, 100))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    (exporter_inputs["source_images_dir"] / "7.JPG").write_bytes(payload)

    n = 8
    ys, xs = np.meshgrid(
        np.arange(n, dtype=np.float32),
        np.arange(n, dtype=np.float32),
        indexing="ij",
    )
    points = np.zeros((1, n, n, 3), dtype=np.float32)
    points[0, ..., 0] = xs + 1.0  # x
    points[0, ..., 1] = 0.0  # 地面 y=0（DA3 world y-down，高度 = -y）
    points[0, ..., 2] = ys + 1.0  # z
    for row, col in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        points[0, row, col, 1] = -0.5  # 主体（货架顶）高度 0.5
    for row, col in [(5, 5), (6, 6)]:
        points[0, row, col, 1] = -0.8  # 天花板薄片高度 0.8（主体顶+0.15 以上）
    _write_cache(
        exporter_inputs["da3_cache_path"],
        points=points,
        confidence=np.ones((1, n, n), dtype=np.float32),
        images=np.zeros((1, n, n, 3), dtype=np.uint8),
        source_image_sha256=[digest],
        source_image_sizes=[[n, n]],
        # 相机放在地面上方（world y-down -> 中心 y 为负），地平面法向定向正确
        extrinsic=np.asarray(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 0.0]]],
            dtype=np.float32,
        ),
    )
    mapping_path = exporter_inputs["global_mapping_path"]
    mapping_path.write_text(
        json.dumps(
            {
                "11": [
                    {
                        "image_id": 7,
                        "object_id": 3,
                        "bbox": [2.0, 2.0, 4.0, 4.0],
                        "removed": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    provenance = _cache_provenance(source_image_sha256=[digest])
    provenance["processed_size"] = [n, n]
    _write_footprint_generation(
        exporter_inputs["footprint_root"],
        status="accepted",
        cache_provenance=provenance,
        global_mapping_path=mapping_path,
    )
    masks = np.zeros((1, n, n), dtype=bool)
    masks[0, 2:4, 2:4] = True
    _write_sam3_mask_entry(
        exporter_inputs["sam3_mask_cache_root"],
        digest,
        [(3, [2.0, 2.0, 4.0, 4.0])],
        masks,
    )

    result = export_web_viewer_bundle(
        **exporter_inputs,
        filter_config=PointCloudFilterConfig(enabled=False),
    )
    generation = Path(result["manifest_path"]).parent
    positions = np.fromfile(generation / "positions.f32.bin", dtype="<f4").reshape(
        -1, 3
    )
    # 天花板薄片（y=-0.8，高度 0.8 > 0.5+0.15）被裁掉
    assert not np.isclose(positions[:, 1], -0.8, atol=1e-6).any()
    # 地面与主体保留
    assert int(np.isclose(positions[:, 1], 0.0, atol=1e-6).sum()) == n * n - 4 - 2
    objects = json.loads((generation / "objects.json").read_text(encoding="utf-8"))
    start, end = objects["11"]["instances"][0]["point_index_range"]
    assert end - start == 4, "subject (mask-covered) points survive the sky cut"
    assert np.allclose(positions[start:end, 1], -0.5, atol=1e-6)


def test_level_rotation_rejects_mid_scene_plane(tmp_path):
    """地板性门：斜穿/悬于主体中部的近水平平面必须被拒，选真正的地面。

    中景平面（内点最多）下方有 40% 的点 -> below_ratio 超过 15% 门 ->
    剔除后继续搜索，最终选中地面（0% 点在其下方）。
    """
    from src.web_viewer_export import _fit_level_rotation

    rng = np.random.default_rng(7)
    # DA3 world y-down：地面 y=0 在最底部，上方 y 为负
    grid = rng.uniform(0.0, 2.0, size=(200, 2))
    floor = np.column_stack([grid[:, 0], np.zeros(200), grid[:, 1]])  # 地面 y=0
    grid_mid = rng.uniform(0.0, 2.0, size=(600, 2))
    mid_plane = np.column_stack(
        [grid_mid[:, 0], np.full(600, -1.0), grid_mid[:, 1]]
    )  # 悬空薄片 y=-1（离地 1m，内点最多）
    body = np.column_stack(
        [
            rng.uniform(0.0, 2.0, size=200),
            rng.uniform(-2.0, -1.5, size=200),  # 主体在薄片上方
            rng.uniform(0.0, 2.0, size=200),
        ]
    )
    valid_points = np.concatenate([floor, mid_plane, body]).astype(np.float64)
    # 相机在地面上方（world y-down -> 中心 y 为负）
    extrinsic = np.asarray(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 0.0]]],
        dtype=np.float32,
    )

    rotation, leveled = _fit_level_rotation(valid_points, extrinsic)

    assert leveled, "mid-scene plane rejected, real floor accepted"
    heights = (valid_points @ rotation.T)[:, 1]
    # 摆平后地面在最底部，薄片离地 ~1m，主体在薄片上方
    floor_height = np.median(heights[:200])
    mid_height = np.median(heights[200:800])
    assert floor_height < mid_height
    assert mid_height - floor_height == pytest.approx(1.0, abs=0.02)
    assert heights[800:].min() > mid_height, "body stays above the floating plane"
    assert heights[:200].ptp() < 0.01, "floor is level after rotation"


def test_viewer_web_cli_routes_exporter_arguments(monkeypatch, tmp_path, capsys):
    dataset = tmp_path / "datasets" / "sample"
    save_root = tmp_path / "save-root"
    output_dir = tmp_path / "viewer-bundle"
    captured = {}

    def fake_export_web_viewer_bundle(**kwargs):
        captured.update(kwargs)
        generation = output_dir / "runs" / ("b" * 32)
        return {
            "output_dir": str(output_dir),
            "manifest_path": str(generation / "manifest.json"),
            "point_count": 7,
            "footprint_status": "accepted",
            "thumbnail_count": 7,
        }

    monkeypatch.setattr(
        "src.web_viewer_export.export_web_viewer_bundle",
        fake_export_web_viewer_bundle,
    )
    monkeypatch.setattr(
        main_module.sys,
        "argv",
        [
            "main.py",
            "--config",
            str(main_module.PROJECT_ROOT / "config.yaml"),
            "--mode",
            "viewer-web",
            "--dataset",
            str(dataset),
            "--save_root",
            str(save_root),
            "--viewer-web-output",
            str(output_dir),
            "--viewer-web-voxel-size",
            "0.125",
            "--viewer-web-max-points",
            "123",
        ],
    )

    main_module.main()

    dataset_output = save_root.resolve() / dataset.name
    assert captured == {
        "da3_cache_path": dataset_output / "da3_cache" / "predictions.npz",
        "global_mapping_path": dataset_output
        / "dedup_detections"
        / "global_mapping.json",
        "footprint_root": dataset_output / "ground_stack_footprint",
        "output_dir": output_dir.resolve(),
        "source_images_dir": dataset / "images",
        "sam3_mask_cache_root": dataset_output / "sam3_mask_cache" / "v1",
        "voxel_size_m": 0.125,
        "max_points": 123,
    }
    output = capsys.readouterr().out
    assert f"Custom viewer-web output: {output_dir.resolve()}" in output
    assert (
        "must be served or mounted at browser URL /data/ before starting the frontend"
        in output
    )
    assert "npm --prefix" not in output


def test_viewer_web_cli_default_output_keeps_directly_runnable_next_step(
    monkeypatch, tmp_path, capsys
):
    dataset = tmp_path / "datasets" / "sample"
    save_root = tmp_path / "save-root"
    default_output = (
        main_module.PROJECT_ROOT / "modules" / "viewer_web" / "public" / "data"
    )
    captured = {}

    def fake_export_web_viewer_bundle(**kwargs):
        captured.update(kwargs)
        generation = default_output / "runs" / ("c" * 32)
        return {
            "output_dir": str(default_output),
            "manifest_path": str(generation / "manifest.json"),
            "point_count": 7,
            "footprint_status": "accepted",
            "thumbnail_count": 7,
        }

    monkeypatch.setattr(
        "src.web_viewer_export.export_web_viewer_bundle",
        fake_export_web_viewer_bundle,
    )
    monkeypatch.setattr(
        main_module.sys,
        "argv",
        [
            "main.py",
            "--config",
            str(main_module.PROJECT_ROOT / "config.yaml"),
            "--mode",
            "viewer-web",
            "--dataset",
            str(dataset),
            "--save_root",
            str(save_root),
        ],
    )

    main_module.main()

    assert captured["output_dir"] == default_output
    assert capsys.readouterr().out.endswith(
        f"Next step: npm --prefix {main_module.PROJECT_ROOT / 'modules' / 'viewer_web'} run dev\n"
    )
