import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import main as main_module
import src.web_viewer_export as exporter
from src.web_viewer_export import WebViewerExportError, export_web_viewer_bundle
from utils.pointcloud_filter import PointCloudFilterConfig
from utils.sam3_mask_cache import (
    FrameMaskCacheRequest,
    ProcessedDetectionPrompt,
    load_or_compute_frame_masks,
)


def _classification() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "source": "personalcare",
        "project_id": 51,
        "status": "resolved",
        "sku_id": "430085",
        "sku_name": "产品A",
        "confidence": 0.75,
        "metadata": {
            "status": "master_data_pending",
            "manufacturer": None,
            "brand": None,
            "category": None,
            "object_kind": None,
        },
    }


def _write_cache(path: Path) -> None:
    points = np.asarray(
        [[[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        cache_schema_version=np.asarray(3, dtype=np.int32),
        is_metric=np.asarray(1, dtype=np.int32),
        scale_factor=np.asarray(1.0, dtype=np.float32),
        source_model=np.asarray("depth-anything/DA3NESTED-GIANT-LARGE", dtype="<U64"),
        image_ids=np.asarray([7], dtype=np.int32),
        world_points=points,
        world_points_conf=np.ones((1, 2, 2), dtype=np.float32),
        images=np.asarray(
            [[[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]]],
            dtype=np.uint8,
        ),
        extrinsic=np.asarray(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            dtype=np.float32,
        ),
        source_image_sizes=np.asarray([[512, 512]], dtype=np.int32),
        source_to_processed_affine=np.asarray(
            [[[2.0 / 512.0, 0.0, 0.0], [0.0, 2.0 / 512.0, 0.0]]],
            dtype=np.float32,
        ),
        source_image_sha256=np.asarray(["a" * 64], dtype="<U64"),
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
                        "bbox": [0.0, 0.0, 512.0, 512.0],
                        "removed": False,
                        "classification": _classification(),
                    },
                    {
                        "image_id": 7,
                        "object_id": 4,
                        "bbox": [128.0, 128.0, 384.0, 384.0],
                        "removed": True,
                        "classification": _classification(),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_mask_cache(root: Path) -> None:
    request = FrameMaskCacheRequest(
        cache_root=root,
        image_id=7,
        image_path=Path("unused.jpg"),
        source_size_wh=(512, 512),
        processed_shape_hw=(2, 2),
        source_to_processed_affine=np.asarray(
            [[2.0 / 512.0, 0.0, 0.0], [0.0, 2.0 / 512.0, 0.0]],
            dtype=np.float64,
        ),
        detections=(
            ProcessedDetectionPrompt(
                object_id=3,
                source_bbox_xyxy=(0.0, 0.0, 512.0, 512.0),
                processed_bbox_xyxy=(0.0, 0.0, 2.0, 2.0),
            ),
            ProcessedDetectionPrompt(
                object_id=4,
                source_bbox_xyxy=(128.0, 128.0, 384.0, 384.0),
                processed_bbox_xyxy=(0.5, 0.5, 1.5, 1.5),
            ),
        ),
        inference_contract={
            "api": "self_exemplar",
            "threshold": 0.5,
            "image_size": 1008,
            "max_batch_size": 32,
            "max_dets_per_query": 1,
            "clip_to_bbox": True,
        },
    )
    load_or_compute_frame_masks(
        request,
        lambda: {
            3: np.ones((2, 2), dtype=bool),
            4: np.zeros((2, 2), dtype=bool),
        },
    )


@pytest.fixture
def exporter_inputs(tmp_path: Path) -> dict[str, Path | str]:
    cache_path, mapping_path = (
        tmp_path / "predictions.npz",
        tmp_path / "global_mapping.json",
    )
    masks_root = tmp_path / "sam3_mask_cache" / "v2"
    _write_cache(cache_path)
    _write_mapping(mapping_path)
    _write_mask_cache(masks_root)
    source_images_dir = tmp_path / "images"
    source_images_dir.mkdir()
    Image.new("RGB", (512, 512), (25, 50, 75)).save(source_images_dir / "7.JPG")
    return {
        "dataset_name": "floor_display6",
        "da3_cache_path": cache_path,
        "global_mapping_path": mapping_path,
        "output_dir": tmp_path / "bundle",
        "source_images_dir": source_images_dir,
        "sam3_mask_cache_root": masks_root,
    }


def _stable_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        exporter,
        "_estimate_scene_normals",
        lambda points, _extrinsic: np.tile([0.0, 0.0, 1.0], (len(points), 1)),
    )


def test_export_publishes_schema3_bundle_with_product_thumbnails(
    exporter_inputs, monkeypatch
) -> None:
    _stable_geometry(monkeypatch)
    result = export_web_viewer_bundle(
        **exporter_inputs,
        voxel_size_m=0.001,
        max_points=10,
        filter_config=PointCloudFilterConfig(enabled=False),
    )
    generation = Path(result["manifest_path"]).parent
    assert {path.name for path in generation.iterdir()} == {
        "manifest.json",
        "positions.f32.bin",
        "colors.u8.bin",
        "normals.i8.bin",
        "objects.json",
        "thumbs",
    }
    assert json.loads((exporter_inputs["output_dir"] / "CURRENT").read_text()) == {
        "run_id": generation.name
    }
    manifest = json.loads((generation / "manifest.json").read_text())
    assert set(manifest) == {
        "schema_version",
        "backend",
        "dataset_name",
        "frame_count",
        "display_bounds",
        "world_to_view",
    }
    assert manifest["schema_version"] == "3.0.0"
    assert manifest["backend"] == "DA3"
    assert "source_model" not in manifest
    assert "provenance" not in manifest
    assert manifest["dataset_name"] == "floor_display6"
    assert manifest["frame_count"] == 1
    assert (
        len(manifest["display_bounds"]) == 6
        and np.isfinite(manifest["display_bounds"]).all()
    )
    assert (
        len(manifest["world_to_view"]) == 16
        and np.isfinite(manifest["world_to_view"]).all()
    )
    positions = np.fromfile(generation / "positions.f32.bin", dtype="<f4").reshape(
        -1, 3
    )
    colors = np.fromfile(generation / "colors.u8.bin", dtype=np.uint8).reshape(-1, 3)
    normals = np.fromfile(generation / "normals.i8.bin", dtype=np.int8).reshape(-1, 3)
    assert positions.shape == colors.shape == normals.shape == (4, 3)
    assert json.loads((generation / "objects.json").read_text()) == {
        "11": {
            "ordered_skus": [{"sku_id": "430085", "sku_name": "产品A"}],
            "point_ranges": [[0, 4]],
            "observations": [
                {
                    "image_id": 7,
                    "object_id": 3,
                    "removed": False,
                    "thumbnail": "thumbs/11_0.jpg",
                },
                {
                    "image_id": 7,
                    "object_id": 4,
                    "removed": True,
                    "thumbnail": "thumbs/11_1.jpg",
                },
            ],
        }
    }
    thumbnails = sorted((generation / "thumbs").iterdir())
    assert len(thumbnails) == 2
    for thumbnail in thumbnails:
        with Image.open(thumbnail) as image:
            assert image.format == "JPEG"
            assert max(image.size) <= 256
    assert result == {
        "output_dir": str(exporter_inputs["output_dir"]),
        "manifest_path": str(generation / "manifest.json"),
        "point_count": 4,
        "thumbnail_count": 2,
    }


def test_export_rejects_empty_dataset_name(exporter_inputs) -> None:
    exporter_inputs["dataset_name"] = " "
    with pytest.raises(WebViewerExportError, match="dataset_name"):
        export_web_viewer_bundle(**exporter_inputs)


def test_export_uses_level_rotation_and_centers_rotated_points(
    exporter_inputs, monkeypatch
) -> None:
    """The manifest must retain fitted leveling, not only the CV axis flip."""
    rotation = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    monkeypatch.setattr(
        exporter,
        "_fit_level_rotation",
        lambda _points, _extrinsic: (rotation, True),
        raising=False,
    )

    result = export_web_viewer_bundle(
        **exporter_inputs,
        voxel_size_m=0.001,
        max_points=10,
        filter_config=PointCloudFilterConfig(enabled=False),
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text())
    world_to_view = np.asarray(manifest["world_to_view"]).reshape(4, 4)
    filtered_points = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    )
    assert np.array_equal(world_to_view[:3, :3], rotation)
    assert np.allclose(
        world_to_view[:3, 3], -np.median(filtered_points @ rotation.T, axis=0)
    )


def test_fit_level_rotation_levels_real_tilted_ground() -> None:
    """RANSAC finds a camera-facing sloped floor instead of only axis flipping."""
    x, z = np.meshgrid(
        np.linspace(0.0, 2.0, 21), np.linspace(0.0, 2.0, 21), indexing="ij"
    )
    slope_x, slope_z = 0.15, 0.10
    ground = np.column_stack(
        [x.ravel(), (slope_x * x + slope_z * z).ravel(), z.ravel()]
    )
    rng = np.random.default_rng(23)
    body_xz = rng.uniform(0.25, 1.75, size=(80, 2))
    body_ground_y = slope_x * body_xz[:, 0] + slope_z * body_xz[:, 1]
    body = np.column_stack(
        [
            body_xz[:, 0],
            body_ground_y - rng.uniform(0.4, 1.0, size=len(body_xz)),
            body_xz[:, 1],
        ]
    )
    valid_points = np.concatenate([ground, body]).astype(np.float64)
    extrinsic = np.asarray(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 0.0]]],
        dtype=np.float64,
    )

    rotation, leveled = exporter._fit_level_rotation(valid_points, extrinsic)

    expected_camera_facing_normal = np.asarray([slope_x, -1.0, slope_z])
    expected_camera_facing_normal /= np.linalg.norm(expected_camera_facing_normal)
    ground_heights = (ground @ rotation.T)[:, 1]
    body_heights = (body @ rotation.T)[:, 1]
    assert leveled
    assert np.ptp(ground_heights) < 1e-6
    assert np.allclose(
        rotation @ expected_camera_facing_normal, [0.0, 1.0, 0.0], atol=1e-6
    )
    assert np.median(body_heights) > np.median(ground_heights) + 0.3
    assert not np.allclose(rotation, np.diag([1.0, -1.0, -1.0]))


def test_viewer_web_cli_routes_minimal_exporter_arguments(
    monkeypatch, tmp_path, capsys
):
    dataset, save_root, output_dir = (
        tmp_path / "datasets" / "sample",
        tmp_path / "save-root",
        tmp_path / "viewer-bundle",
    )
    captured: dict[str, object] = {}

    def fake_export_web_viewer_bundle(**kwargs):
        captured.update(kwargs)
        return {
            "output_dir": str(output_dir),
            "manifest_path": str(output_dir / "runs" / ("b" * 32) / "manifest.json"),
            "point_count": 7,
            "thumbnail_count": 2,
        }

    monkeypatch.setattr(
        "src.web_viewer_export.export_web_viewer_bundle", fake_export_web_viewer_bundle
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
        ],
    )
    main_module.main()
    dataset_output = save_root.resolve() / dataset.name
    assert captured == {
        "dataset_name": dataset.name,
        "da3_cache_path": dataset_output / "da3_cache" / "predictions.npz",
        "global_mapping_path": dataset_output
        / "dedup_detections"
        / "global_mapping.json",
        "output_dir": output_dir.resolve(),
        "source_images_dir": dataset / "images",
        "sam3_mask_cache_root": dataset_output / "sam3_mask_cache" / "v2",
        "voxel_size_m": 0.005,
        "max_points": 1500000,
    }
    assert "Custom viewer-web output" in capsys.readouterr().out
