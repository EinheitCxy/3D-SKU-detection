import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from utils.sam3_mask_cache import PVS_INFERENCE_CONTRACT

import main as main_module
import src.web_viewer_export as exporter
from src.web_viewer_export import WebViewerExportError, export_web_viewer_bundle
from src.viewer_export_tuning import ViewerExportTuning
from utils.config import default_sam3_mask_cache_root
from utils.pointcloud_filter import PointCloudFilterConfig
import utils.scene_orientation as scene_orientation
from utils.scene_orientation import SceneOrientation
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


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize(
    "values", [[], [-1, -1], [-1, 0, 0, 2, 4, 4], [0, 0, 1, 2, 3, 4]]
)
def test_point_ranges_preserve_empty_and_missing_observations(dtype, values):
    labels = np.asarray(values, dtype=dtype)
    keys = [("11", index) for index in range(5)]
    objects = {"11": {"instances": [{"removed": index == 2} for index in range(5)]}}
    exporter._attach_point_index_ranges(objects, labels, keys)

    for label, instance in enumerate(objects["11"]["instances"]):
        assert instance["point_index_range"] == [
            int(np.count_nonzero(labels < label)),
            int(np.count_nonzero(labels <= label)),
        ]
        assert all(type(value) is int for value in instance["point_index_range"])
        assert instance["removed"] == (label == 2)


def test_point_ranges_without_observations_leave_objects_unchanged():
    objects = {"11": {"instances": []}}
    exporter._attach_point_index_ranges(objects, np.asarray([-1], dtype=np.int32), [])
    assert objects == {"11": {"instances": []}}


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
                        "bbox": [128.0, 128.0, 384.0, 256.0],
                        "removed": True,
                        "classification": _classification(),
                    },
                    {
                        "image_id": 7,
                        "object_id": 5,
                        "bbox": [128.0, 0.0, 384.0, 128.0],
                        "removed": False,
                        "classification": _classification(),
                    },
                    {
                        "image_id": 7,
                        "object_id": 6,
                        "bbox": [246.0, 246.0, 266.0, 256.0],
                        "removed": False,
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
                source_bbox_xyxy=(128.0, 128.0, 384.0, 256.0),
                processed_bbox_xyxy=(0.5, 0.5, 1.5, 1.0),
            ),
            ProcessedDetectionPrompt(
                object_id=5,
                source_bbox_xyxy=(128.0, 0.0, 384.0, 128.0),
                processed_bbox_xyxy=(0.5, 0.0, 1.5, 0.5),
            ),
            ProcessedDetectionPrompt(
                object_id=6,
                source_bbox_xyxy=(246.0, 246.0, 266.0, 256.0),
                processed_bbox_xyxy=(0.9609375, 0.9609375, 1.0390625, 1.0),
            ),
        ),
        inference_contract=PVS_INFERENCE_CONTRACT.copy(),
    )
    load_or_compute_frame_masks(
        request,
        lambda: {
            3: np.ones((2, 2), dtype=bool),
            4: np.zeros((2, 2), dtype=bool),
            5: np.zeros((2, 2), dtype=bool),
            6: np.zeros((2, 2), dtype=bool),
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
    sku_masterdata_csv = tmp_path / "sku_masterdata.csv"
    sku_masterdata_csv.write_text(
        "sku_id,manufacturer,brand,category,is_posm\n"
        "430085,百事,冲劲,饮料酒水,0\n",
        encoding="utf-8",
    )
    return {
        "dataset_name": "floor_display6",
        "da3_cache_path": cache_path,
        "global_mapping_path": mapping_path,
        "output_dir": tmp_path / "bundle",
        "source_images_dir": source_images_dir,
        "sam3_mask_cache_root": masks_root,
        "sku_masterdata_csv": sku_masterdata_csv,
    }


def _stable_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        exporter,
        "_estimate_scene_normals",
        lambda points, _extrinsic: np.tile([0.0, 0.0, 1.0], (len(points), 1)),
    )


def test_background_voxels_preserve_winners_ties_and_first_seen_order():
    points = np.asarray([
        [2.1, 0.0, 1.0], [-0.1, 0.0, 1.0], [0.1, 0.0, 1.0],
        [2.8, 0.0, 1.0], [-0.8, 0.0, 1.0], [2.5, 0.0, 1.0],
        [0.2, 0.0, 1.0], [-0.5, 0.0, 1.0],
    ])
    confidence = np.asarray([1, 4, 2, 3, 4, 3, 5, 100], dtype=np.float32)
    # The last point represents a protected product and must not compete.
    background = np.arange(7)
    selected = exporter._select_background_voxels(points, confidence, background, 1.0)
    assert np.array_equal(selected, [3, 1, 6])


def test_background_voxels_empty_and_large_negative_keys():
    points = np.asarray([[-1e12, 0, 1], [-1e12, 0, 1], [1e12, 0, 1]])
    confidence = np.asarray([-1, -0.0, 0.0], dtype=np.float32)
    assert np.array_equal(
        exporter._select_background_voxels(points, confidence, np.arange(3), 0.005),
        [1, 2],
    )
    selected = exporter._select_background_voxels(
        points, confidence, np.empty(0, dtype=np.int64), 0.005
    )
    assert selected.shape == (0,) and selected.dtype == np.int64


def test_tuning_requires_surfel_export(exporter_inputs):
    with pytest.raises(WebViewerExportError, match="tuning requires Surfel export"):
        export_web_viewer_bundle(
            **exporter_inputs, tuning=ViewerExportTuning(quality_filter=True)
        )
    assert not exporter_inputs["output_dir"].exists()


def test_all_viewer_tuning_preserves_nonempty_global_ids(exporter_inputs):
    path = exporter_inputs["da3_cache_path"]
    with np.load(path) as archive:
        cache = {key: archive[key] for key in archive.files}
    # Small native tangents form a valid Surfel plane instead of metre-wide steps.
    cache["world_points"][..., :2] *= 0.01
    cache["intrinsic"] = np.eye(3, dtype=np.float32)[None]
    np.savez_compressed(path, **cache)
    common = dict(exporter_inputs, surfel_texture_edge=256,
                  filter_config=PointCloudFilterConfig(enabled=False))
    baseline = export_web_viewer_bundle(**common)
    common["output_dir"] = exporter_inputs["output_dir"].with_name("tuned")
    tuned = export_web_viewer_bundle(
        **common, tuning=ViewerExportTuning(
            quality_filter=True, product_spill_trim=True,
            background_edge_weights=(2, 1, 1), single_owner=True,
        )
    )
    def counts(result):
        objects = json.loads(Path(result["manifest_path"]).with_name("objects.json").read_text())
        return {gid: sum(end - start for start, end in obj["point_ranges"])
                for gid, obj in objects.items()}
    before, after = counts(baseline), counts(tuned)
    assert before and any(before.values())
    assert all(after.get(gid, 0) > 0 for gid, count in before.items() if count > 0)
    assert tuned["point_count"] <= baseline["point_count"]


def _surfel_cache(exporter_inputs):
    path = exporter_inputs["da3_cache_path"]
    with np.load(path) as archive:
        cache = {key: archive[key] for key in archive.files}
    cache["world_points"][..., :2] *= 0.01
    cache["intrinsic"] = np.eye(3, dtype=np.float32)[None]
    # Invalid confidence and behind-camera depth must keep native slot alignment.
    cache["world_points_conf"][0, 0, 1] = np.nan
    cache["world_points"][0, 1, 0, 2] = -1
    np.savez_compressed(path, **cache)


@pytest.mark.parametrize("filter_enabled", [False, True])
def test_prepared_surfel_matches_serial_and_prepares_once(exporter_inputs, monkeypatch, filter_enabled):
    import src.surfel_export as surfels

    _surfel_cache(exporter_inputs)
    common = dict(exporter_inputs, surfel_texture_edge=256,
                  filter_config=PointCloudFilterConfig(
                      enabled=filter_enabled, min_points=1, sor_nb_neighbors=1,
                      keep_main_clusters=False, remove_ground=False,
                  ),
                  tuning=ViewerExportTuning(single_owner=True))
    serial = export_web_viewer_bundle(**common)
    calls = {"grid": 0, "jpeg": 0}
    grid, save = surfels.grid_tangents, Image.Image.save

    def counted_grid(*args, **kwargs):
        calls["grid"] += 1
        return grid(*args, **kwargs)

    def counted_save(image, fp, format=None, **kwargs):
        if format == "JPEG" and kwargs.get("quality") == 95:
            calls["jpeg"] += 1
        return save(image, fp, format=format, **kwargs)

    monkeypatch.setattr(surfels, "grid_tangents", counted_grid)
    monkeypatch.setattr(Image.Image, "save", counted_save)
    prepared = exporter.prepare_web_viewer_geometry(
        da3_cache_path=common["da3_cache_path"], source_images_dir=common["source_images_dir"],
        surfel_texture_edge=256, tuning=common["tuning"], filter_config=common["filter_config"],
    )
    common["output_dir"] = common["output_dir"].with_name("prepared")
    assert not common["output_dir"].exists()
    result = export_web_viewer_bundle(**common, prepared=prepared)
    assert calls == {"grid": 1, "jpeg": 1}

    def contents(result):
        root = Path(result["manifest_path"]).parent
        return {str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}

    assert contents(result) == contents(serial)
    assert result["point_count"] == 2
    sample = exporter._sample_points(
        prepared.cache, {}, mask_cache_root=common["sam3_mask_cache_root"],
        voxel_size=0.005, filter_config=common["filter_config"],
        adaptive_surfels=True, prepared_geometry=prepared, level_rotation=(np.eye(3), False),
    )
    sources = sample["source_indices"]
    assert np.array_equal(sources, [0, 3])
    assert np.array_equal(sample["positions"], prepared.cache["points"].reshape(-1, 3)[sources])
    assert np.array_equal(sample["surfel_geometry"][0], prepared.geometry[0].reshape(-1, 3)[sources])


@pytest.mark.parametrize("changes", [
    {"backend": "Pi3X"}, {"surfel_texture_edge": None}, {"surfel_texture_edge": 512},
    {"tuning": ViewerExportTuning(quality_filter=True)},
    {"da3_cache_path": Path("other.npz")}, {"source_images_dir": Path("other")},
    {"filter_config": PointCloudFilterConfig(enabled=False)},
])
def test_prepared_surfel_rejects_mismatched_inputs(exporter_inputs, changes):
    _surfel_cache(exporter_inputs)
    prepared = exporter.prepare_web_viewer_geometry(
        da3_cache_path=exporter_inputs["da3_cache_path"],
        source_images_dir=exporter_inputs["source_images_dir"], surfel_texture_edge=256,
    )
    common = dict(exporter_inputs, surfel_texture_edge=256)
    common.update(changes)
    with pytest.raises(WebViewerExportError, match="prepared geometry"):
        export_web_viewer_bundle(**common, prepared=prepared)
    assert not exporter_inputs["output_dir"].exists()


def test_prepared_surfel_rejects_changed_cache(exporter_inputs):
    import os

    _surfel_cache(exporter_inputs)
    prepared = exporter.prepare_web_viewer_geometry(
        da3_cache_path=exporter_inputs["da3_cache_path"],
        source_images_dir=exporter_inputs["source_images_dir"], surfel_texture_edge=256,
    )
    path = exporter_inputs["da3_cache_path"]
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(WebViewerExportError, match="cache changed"):
        export_web_viewer_bundle(**exporter_inputs, surfel_texture_edge=256, prepared=prepared)


def test_prepared_surfel_rejects_changed_valid_domain(exporter_inputs):
    _surfel_cache(exporter_inputs)
    config = PointCloudFilterConfig(enabled=False)
    prepared = exporter.prepare_web_viewer_geometry(
        da3_cache_path=exporter_inputs["da3_cache_path"],
        source_images_dir=exporter_inputs["source_images_dir"], surfel_texture_edge=256,
        filter_config=config,
    )
    assert prepared.valid_indices.tolist() == [0, 3]
    prepared.cache["confidence"].reshape(-1)[0] = np.nan
    with pytest.raises(WebViewerExportError, match="domain/order"):
        exporter._sample_points(
            prepared.cache, {}, mask_cache_root=exporter_inputs["sam3_mask_cache_root"],
            voxel_size=0.005, filter_config=config, adaptive_surfels=True,
            prepared_geometry=prepared,
        )


@pytest.mark.parametrize("early_budget", [0, 500_000])
def test_prepare_filter_can_be_explicitly_deferred(exporter_inputs, monkeypatch, early_budget):
    _surfel_cache(exporter_inputs)

    def unexpected(*args, **kwargs):
        raise AssertionError("NN/SOR preparation must not run")

    monkeypatch.setattr(exporter, "prepare_scene_filter", unexpected)
    prepared = exporter.prepare_web_viewer_geometry(
        da3_cache_path=exporter_inputs["da3_cache_path"],
        source_images_dir=exporter_inputs["source_images_dir"], surfel_texture_edge=256,
        prepare_filter=bool(early_budget),
        tuning=ViewerExportTuning(early_background_candidates=early_budget),
    )
    assert prepared.scene_filter is None
    assert prepared.valid_indices.tolist() == [0, 3]


@pytest.mark.parametrize("value", [-1, True, 1.5, "500000"])
def test_early_background_candidates_requires_nonnegative_integer(value):
    with pytest.raises(ValueError, match="early_background_candidates"):
        ViewerExportTuning(early_background_candidates=value)


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("candidate_budget", [9, 500_000])
def test_early_background_preserves_products_alignment_and_reports_underfill(
    monkeypatch, caplog, adaptive, candidate_budget,
):
    from src.surfel_export import grid_tangents
    from src.surfel_sampling import EdgeAdaptiveSampler

    y, x = np.indices((8, 8))
    plane = np.stack((x * .005, y * .005, np.ones(x.shape)), axis=-1).astype(np.float32)
    points = np.stack((plane, plane + [0, 0, .001])).astype(np.float32)
    confidence = np.ones(points.shape[:-1], dtype=np.float32)
    confidence.ravel()[3] = np.nan
    cache = dict(points=points, confidence=confidence,
                 images=np.arange(points.size, dtype=np.uint8).reshape(points.shape),
                 extrinsic=np.repeat(np.eye(4)[None], 2, axis=0))
    labels = np.full(confidence.size, -1, dtype=np.int32)
    labels[[0, 1, 64, 65]] = 0
    monkeypatch.setattr(exporter, "_instance_labels_v2", lambda _cache, _objects, indices, *_:
                        (labels[indices], [("7", 0)]))
    monkeypatch.setattr(exporter, "MAX_BACKGROUND_POINTS", 5)
    calls, supports = [], []

    def candidate_filter(candidate_points, config, protect_mask, **kwargs):
        assert "prepared" not in kwargs  # Never reuse the full-domain raw filter.
        calls.append((candidate_points.copy(), protect_mask.copy()))
        keep = protect_mask.copy()
        keep[np.flatnonzero(~protect_mask)[:3]] = True
        return keep

    select = EdgeAdaptiveSampler.select

    def record_select(self, indices, budget, *, support_indices=None):
        supports.append(None if support_indices is None else support_indices.copy())
        return select(self, indices, budget, support_indices=support_indices)

    monkeypatch.setattr(exporter, "filter_scene_points", candidate_filter)
    monkeypatch.setattr(EdgeAdaptiveSampler, "select", record_select)
    with caplog.at_level("INFO", logger=exporter.__name__):
        sample = exporter._sample_points(
            cache, {}, mask_cache_root=Path("/unused"), voxel_size=.005,
            filter_config=PointCloudFilterConfig(), adaptive_surfels=adaptive,
            level_rotation=(np.eye(3), False),
            tuning=ViewerExportTuning(early_background_candidates=candidate_budget),
        )
    assert len(calls) == 1
    candidate_points, protected = calls[0]
    assert protected.sum() == 4
    assert len(candidate_points) - protected.sum() <= candidate_budget
    assert np.array_equal(candidate_points[protected], points.reshape(-1, 3)[labels >= 0])
    expected_support = np.flatnonzero((labels < 0) & np.isfinite(confidence.ravel()))
    assert np.array_equal(supports[0], expected_support)
    counts = sample["early_background_counts"]
    assert counts["background_after_geometry_filter"] == 3
    assert counts["background_final"] == 3 < counts["background_budget"]
    assert counts["background_candidates"] <= candidate_budget
    assert "no refill" in caplog.text
    sources = sample["source_indices"]
    assert set(np.flatnonzero(labels >= 0)) <= set(sources)
    assert 3 not in sources
    assert np.array_equal(sample["positions"], points.reshape(-1, 3)[sources])
    assert np.array_equal(sample["colors"], cache["images"].reshape(-1, 3)[sources])
    assert np.array_equal(sample["instance_labels"], labels[sources])
    if adaptive:
        u, v, depth = grid_tangents(points, cache["extrinsic"])
        assert np.array_equal(supports[1], expected_support)
        assert np.array_equal(sample["surfel_geometry"][0], u.reshape(-1, 3)[sources])
        assert np.array_equal(sample["surfel_geometry"][1], v.reshape(-1, 3)[sources])
        assert np.array_equal(sample["surfel_geometry"][2], depth)
        assert sample["surfel_scales"].shape == (len(sources), 2)


def test_early_background_zero_preserves_default_path(monkeypatch):
    cache = dict(points=np.asarray([[[[0., 0., 1.], [1., 0., 1.]]]]),
                 confidence=np.ones((1, 1, 2)), images=np.zeros((1, 1, 2, 3), dtype=np.uint8),
                 extrinsic=np.eye(4)[None])
    calls = []
    monkeypatch.setattr(exporter, "_instance_labels_v2", lambda *_:
                        (np.full(2, -1, dtype=np.int32), []))

    def full_filter(points, config, protect_mask, **kwargs):
        calls.append(points.copy())
        return np.ones(len(points), dtype=bool)

    monkeypatch.setattr(exporter, "filter_scene_points", full_filter)
    common = dict(mask_cache_root=Path("/unused"), voxel_size=.005,
                  filter_config=PointCloudFilterConfig(), level_rotation=(np.eye(3), False))
    default = exporter._sample_points(cache, {}, **common)
    explicit = exporter._sample_points(cache, {}, **common,
                                       tuning=ViewerExportTuning(early_background_candidates=0))
    assert len(calls) == 2 and all(len(points) == 2 for points in calls)
    assert "early_background_counts" not in explicit
    for key in ("positions", "colors", "source_indices", "instance_labels"):
        assert np.array_equal(default[key], explicit[key])


@pytest.mark.parametrize("background_count", [0, 1])
def test_early_background_small_pool_has_no_refill(monkeypatch, background_count):
    points = np.asarray([[[[0., 0., 1.], [.005, 0., 1.], [.01, 0., 1.]]]])
    cache = dict(points=points, confidence=np.ones((1, 1, 3)),
                 images=np.zeros(points.shape, dtype=np.uint8), extrinsic=np.eye(4)[None])
    labels = np.zeros(3, dtype=np.int32)
    labels[:background_count] = -1
    monkeypatch.setattr(exporter, "_instance_labels_v2", lambda *_: (labels, [("7", 0)]))
    sample = exporter._sample_points(
        cache, {}, mask_cache_root=Path("/unused"), voxel_size=.005,
        filter_config=PointCloudFilterConfig(enabled=False), level_rotation=(np.eye(3), False),
        tuning=ViewerExportTuning(early_background_candidates=500_000),
    )
    counts = sample["early_background_counts"]
    assert counts["background_candidates"] == background_count
    assert counts["background_final"] == background_count
    assert len(sample["source_indices"]) == 3


def test_nonadaptive_early_background_rejects_invalid_camera_depth(monkeypatch):
    cache = dict(points=np.asarray([[[[0., 0., -1.]]]]), confidence=np.ones((1, 1, 1)),
                 images=np.zeros((1, 1, 1, 3), dtype=np.uint8), extrinsic=np.eye(4)[None])
    with pytest.raises(WebViewerExportError, match="positive camera depth"):
        exporter._sample_points(
            cache, {}, mask_cache_root=Path("/unused"), voxel_size=.005,
            filter_config=PointCloudFilterConfig(), level_rotation=(np.eye(3), False),
            tuning=ViewerExportTuning(early_background_candidates=5),
        )


def test_single_owner_competes_across_instances_of_one_global_id(monkeypatch):
    y, x = np.indices((4, 4))
    plane = np.stack((x * 0.005, y * 0.005, np.full(x.shape, 1.003)), axis=-1).astype(np.float32)
    points = np.stack((plane, plane + [0, 0, 0.001])).astype(np.float32)
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    labels[1] = 1
    labels[1, :, -1] = 2
    keys = [("7", 0), ("7", 1), ("9", 0)]
    cache = dict(points=points, confidence=np.ones(labels.shape, dtype=np.float32),
                 images=np.arange(points.size, dtype=np.uint8).reshape(points.shape),
                 extrinsic=np.repeat(np.eye(4)[None], 2, axis=0))
    monkeypatch.setattr(exporter, "_instance_labels_v2", lambda _cache, _objects, indices, *_:
                        (labels.ravel()[indices], keys))
    common = dict(mask_cache_root=Path("/unused"), voxel_size=0.005,
                  filter_config=PointCloudFilterConfig(enabled=False), adaptive_surfels=True,
                  level_rotation=(np.eye(3), False))
    baseline = exporter._sample_points(cache, {}, **common)
    tuned = exporter._sample_points(
        cache, {}, **common, tuning=ViewerExportTuning(single_owner=True),
        frame_sharpness=np.ones(2),
    )
    sources = tuned["source_indices"]
    assert len(sources) < len(baseline["source_indices"])
    assert set(tuned["instance_labels"]) == {0, 2}
    assert np.all(sources[tuned["instance_labels"] == 0] < 16)
    assert np.all(sources[tuned["instance_labels"] == 2] >= 16)
    assert np.array_equal(tuned["positions"], points.reshape(-1, 3)[sources])
    assert np.array_equal(tuned["colors"], cache["images"].reshape(-1, 3)[sources])
    assert np.array_equal(tuned["filtered_points"], baseline["filtered_points"])


def test_export_publishes_schema3_bundle_with_product_thumbnails(
    exporter_inputs, monkeypatch
) -> None:
    _stable_geometry(monkeypatch)
    result = export_web_viewer_bundle(
        **exporter_inputs,
        voxel_size_m=0.001,
        filter_config=PointCloudFilterConfig(enabled=False),
    )
    generation = Path(result["manifest_path"]).parent
    assert {path.name for path in generation.iterdir()} == {
        "manifest.json",
        "positions.f32.bin",
        "colors.u8.bin",
        "normals.i8.bin",
        "objects.json",
        "sku_masterdata.json",
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
        "measurement",
    }
    assert manifest["schema_version"] == "3.0.0"
    assert manifest["backend"] == "DA3"
    assert manifest["measurement"]["coordinate_unit"] == "m"
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
                {
                    "image_id": 7,
                    "object_id": 5,
                    "removed": False,
                    "thumbnail": "thumbs/11_2.jpg",
                },
                {
                    "image_id": 7,
                    "object_id": 6,
                    "removed": False,
                    "thumbnail": "thumbs/11_3.jpg",
                },
            ],
        }
    }
    assert json.loads((generation / "sku_masterdata.json").read_text()) == {
        "430085": {
            "manufacturer": "百事",
            "brand": "冲劲",
            "category": "饮料酒水",
            "is_posm": False,
        }
    }
    thumbnails = sorted((generation / "thumbs").iterdir())
    assert len(thumbnails) == 4
    for thumbnail in thumbnails:
        with Image.open(thumbnail) as image:
            assert image.format == "JPEG"
            assert image.mode == "RGB"
            assert image.size == (128, 128)
    with Image.open(generation / "thumbs" / "11_1.jpg") as image:
        pixels = np.asarray(image)
    assert np.allclose(pixels[16, 64], exporter._THUMB_BACKGROUND, atol=6)
    assert np.allclose(pixels[112, 64], exporter._THUMB_BACKGROUND, atol=6)
    assert np.allclose(pixels[64, 64], (25, 50, 75), atol=6)
    with Image.open(generation / "thumbs" / "11_2.jpg") as image:
        pixels = np.asarray(image)
    assert np.allclose(pixels[34, 64], exporter._THUMB_BACKGROUND, atol=6)
    assert np.allclose(pixels[44, 64], (25, 50, 75), atol=6)
    with Image.open(generation / "thumbs" / "11_3.jpg") as image:
        pixels = np.asarray(image)
    assert np.allclose(pixels[16, 64], exporter._THUMB_BACKGROUND, atol=6)
    assert np.allclose(pixels[64, 40], (25, 50, 75), atol=6)
    assert np.allclose(pixels[64, 4], (25, 50, 75), atol=6)
    assert result == {
        "output_dir": str(exporter_inputs["output_dir"]),
        "manifest_path": str(generation / "manifest.json"),
        "point_count": 4,
        "thumbnail_count": 4,
    }


def test_export_rejects_empty_dataset_name(exporter_inputs) -> None:
    exporter_inputs["dataset_name"] = " "
    with pytest.raises(WebViewerExportError, match="dataset_name"):
        export_web_viewer_bundle(**exporter_inputs)


@pytest.mark.parametrize("metric,unit", [(1, "m"), (0, "unknown"), (None, "unknown")])
@pytest.mark.parametrize("status", ["fitted", "not_found"])
def test_export_measurement_uses_cache_scale_and_shared_orientation(
    exporter_inputs, monkeypatch, metric, unit, status
):
    _stable_geometry(monkeypatch)
    cache_path = exporter_inputs["da3_cache_path"]
    with np.load(cache_path, allow_pickle=False) as loaded:
        cache = {key: loaded[key].copy() for key in loaded.files if key != "is_metric"}
    if metric is not None:
        cache["is_metric"] = np.asarray(metric, dtype=np.int32)
    np.savez_compressed(cache_path, **cache)
    monkeypatch.setattr(
        exporter, "get_scene_orientation",
        lambda _path: (SceneOrientation(status, np.eye(3), np.array([0., 1., 0.]), None, {}), "hit"),
    )
    result = export_web_viewer_bundle(
        **exporter_inputs, filter_config=PointCloudFilterConfig(enabled=False)
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["measurement"] == {
        "coordinate_unit": unit,
        "horizontal_plane": "viewer_xz",
        "orientation_status": status,
    }


@pytest.mark.parametrize("leveled", [False, True])
def test_pi3x_measurement_has_unknown_scale_and_sampled_orientation(
    exporter_inputs, monkeypatch, leveled
):
    _stable_geometry(monkeypatch)
    cache = exporter._load_da3_cache(exporter_inputs["da3_cache_path"])
    monkeypatch.setattr(exporter, "_load_pi3x_cache", lambda *_: cache)
    monkeypatch.setattr(exporter, "_fit_level_rotation", lambda *_: (np.eye(3), leveled))
    result = export_web_viewer_bundle(
        **exporter_inputs, backend="Pi3X", filter_config=PointCloudFilterConfig(enabled=False)
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["measurement"] == {
        "coordinate_unit": "unknown",
        "horizontal_plane": "viewer_xz",
        "orientation_status": "fitted" if leveled else "not_found",
    }


def test_export_uses_level_rotation_and_centers_rotated_points(
    exporter_inputs, monkeypatch
) -> None:
    """The manifest must retain fitted leveling, not only the CV axis flip."""
    rotation = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    monkeypatch.setattr(
        scene_orientation,
        "fit_scene_orientation",
        lambda _points, _extrinsic: SceneOrientation(
            "fitted", rotation, rotation.T @ [0.0, 1.0, 0.0], 0.4, {}
        ),
    )
    _, event = scene_orientation.get_scene_orientation(exporter_inputs["da3_cache_path"])
    assert event == "computed"

    def unexpected_refit(*_args):
        pytest.fail("Viewer must reuse the orientation sidecar")

    monkeypatch.setattr(scene_orientation, "fit_scene_orientation", unexpected_refit)

    result = export_web_viewer_bundle(
        **exporter_inputs,
        voxel_size_m=0.001,
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


def test_export_rejects_cache_replaced_between_geometry_and_orientation(
    exporter_inputs, monkeypatch
):
    def replace_cache_before_orientation(cache_path):
        replacement = cache_path.with_name("replacement.npz")
        _write_cache(replacement)
        replacement.replace(cache_path)
        return SceneOrientation(
            "fitted", np.eye(3), np.array([0.0, 1.0, 0.0]), 0.0, {}
        ), "computed"

    monkeypatch.setattr(exporter, "get_scene_orientation", replace_cache_before_orientation)
    with pytest.raises(WebViewerExportError, match="DA3 cache changed while preparing viewer geometry"):
        export_web_viewer_bundle(**exporter_inputs)
    assert not (exporter_inputs["output_dir"] / "CURRENT").exists()


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
        "sam3_mask_cache_root": default_sam3_mask_cache_root(dataset_output),
        "sku_masterdata_csv": main_module.PROJECT_ROOT / "runtime" / "sku_masterdata.csv",
        "voxel_size_m": 0.005,
    }
    assert "Custom viewer-web output" in capsys.readouterr().out
