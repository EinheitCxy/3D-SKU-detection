"""PromptDA's native geometry must survive the shared scene-cache boundary."""

import numpy as np
import pytest
from PIL import Image

from src.promptda_cache import convert_promptda_cache
from utils.scene_backend import (
    get_scene_backend, promptda_processed_size, resize_pixel_center_affine,
    validate_scene_cache_metadata,
)


@pytest.fixture
def promptda_input(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    rgb = np.full((2, 28, 28, 3), 72, dtype=np.uint8)
    for image_id, color in zip((7, 3), rgb):
        Image.fromarray(color).save(images / f"{image_id:06d}.png")
    intrinsic = np.tile(np.array([[20., 0., 13.5], [0., 20., 13.5], [0., 0., 1.]], np.float32), (2, 1, 1))
    extrinsic = np.tile(np.eye(4, dtype=np.float32)[:3], (2, 1, 1))
    extrinsic[1, 0, 3] = -1.
    prediction = dict(depth=np.full((2, 28, 28), 2., np.float32),
                      processed_images=rgb, intrinsics=intrinsic, extrinsics=extrinsic,
                      frame_ids=np.array([7, 3]), is_metric=np.array(1))
    raw = tmp_path / "raw.npz"
    np.savez(raw, **prediction)
    return raw, images, tmp_path / "promptda_cache/predictions.npz", prediction


def test_promptda_cache_preserves_metric_cameras_frame_ids_and_projection(promptda_input):
    raw, images, output, prediction = promptda_input
    result = convert_promptda_cache(raw, images, output)
    assert result["backend"] == "PromptDA"
    with np.load(output, allow_pickle=False) as cache:
        validate_scene_cache_metadata(cache, "PromptDA")
        np.testing.assert_array_equal(cache["image_ids"], [7, 3])
        np.testing.assert_array_equal(cache["images"], prediction["processed_images"])
        np.testing.assert_array_equal(cache["extrinsic"], prediction["extrinsics"])
        np.testing.assert_array_equal(cache["intrinsic"], prediction["intrinsics"])
        np.testing.assert_array_equal(cache["depth"][..., 0], prediction["depth"])
        np.testing.assert_allclose(cache["world_points"][0, 0, 0], [-1.35, -1.35, 2.])
        np.testing.assert_allclose(cache["world_points"][1, 0, 0], [-.35, -1.35, 2.], atol=1e-6)
        np.testing.assert_array_equal(cache["source_to_processed_affine"], np.tile([[1., 0., 0.], [0., 1., 0.]], (2, 1, 1)))
        assert cache["world_points"].shape == (2, 28, 28, 3)
        assert cache["world_points"].dtype == np.float32
        assert cache["confidence_kind"].item() == "validity"
        assert cache["preprocess_method"].item() == "promptda_resize"
        assert cache["preprocess_resolution"].item() == 504
        np.testing.assert_array_equal(cache["world_points_conf"], 1.)
    with pytest.raises(FileExistsError):
        convert_promptda_cache(raw, images, output)


def test_promptda_native_cache_preserves_cropped_pixels_cameras_and_patch_padding(promptda_input):
    raw, images, output, prediction = promptda_input
    rgb = np.random.default_rng(7).integers(0, 256, (2, 31, 45, 3), dtype=np.uint8)
    for image_id, pixels in zip(prediction["frame_ids"], rgb):
        Image.fromarray(pixels).save(images / f"{image_id:06d}.png")
    prediction.update(
        depth=np.full((2, 31, 45), 2., np.float32), processed_images=rgb,
        preprocess_method=np.array("promptda_native_pad14"),
        network_image_sizes=np.tile(np.array([56, 42], dtype=np.int64), (2, 1)),
    )
    np.savez(raw, **prediction)
    convert_promptda_cache(raw, images, output)
    with np.load(output, allow_pickle=False) as cache:
        validate_scene_cache_metadata(cache, "promptda")
        assert cache["preprocess_method"].item() == "promptda_native_pad14"
        assert cache["preprocess_resolution"].item() == 45
        assert cache["depth"].shape == (2, 31, 45, 1)
        np.testing.assert_array_equal(cache["images"], rgb)
        np.testing.assert_array_equal(cache["intrinsic"], prediction["intrinsics"])
        np.testing.assert_array_equal(cache["extrinsic"], prediction["extrinsics"])
        np.testing.assert_array_equal(cache["source_image_sizes"], [[45, 31], [45, 31]])
        np.testing.assert_array_equal(cache["network_image_sizes"], [[56, 42], [56, 42]])
        np.testing.assert_array_equal(cache["source_to_processed_affine"], np.tile([[1., 0., 0.], [0., 1., 0.]], (2, 1, 1)))
        np.testing.assert_allclose(cache["world_points"][0, 0, 0], [-1.35, -1.35, 2.])


def test_promptda_1008_cache_preserves_declared_resize_rgb_and_camera_grid(tmp_path):
    import cv2
    from src.da3_footprint_stage import _validate_cache

    images = tmp_path / "images"
    images.mkdir()
    rgb = np.random.default_rng(8).integers(0, 256, (1440, 1920, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(images / "000007.png")
    resized = cv2.resize(rgb.astype(np.float32) / 255., (1008, 756), interpolation=cv2.INTER_AREA)
    packed = np.rint(resized * 255).clip(0, 255).astype(np.uint8)
    affine = resize_pixel_center_affine(1920, 1440, 1008, 756)
    source_k = np.array([[1400., 0., 959.5], [0., 1400., 719.5], [0., 0., 1.]], dtype=np.float32)
    image_affine = np.eye(3, dtype=np.float64)
    image_affine[:2] = affine
    resized_k = (image_affine @ source_k).astype(np.float32)
    prediction = dict(
        depth=np.full((1, 756, 1008), 2., dtype=np.float32), processed_images=packed[None],
        intrinsics=resized_k[None], extrinsics=np.eye(4, dtype=np.float32)[None, :3],
        frame_ids=np.array([7], dtype=np.int64), is_metric=np.array(1),
        preprocess_method=np.array("promptda_resize"), preprocess_resolution=np.array(1008, dtype=np.int64),
        network_image_sizes=np.array([[1008, 756]], dtype=np.int64),
    )
    raw, output = tmp_path / "raw.npz", tmp_path / "cache/predictions.npz"
    np.savez(raw, **prediction)
    convert_promptda_cache(raw, images, output)
    with np.load(output, allow_pickle=False) as cache:
        validate_scene_cache_metadata(cache, "promptda")
        assert cache["preprocess_resolution"].item() == 1008
        assert cache["preprocess_method"].item() == "promptda_resize"
        assert cache["depth"].shape == (1, 756, 1008, 1)
        np.testing.assert_array_equal(cache["images"][0], packed)
        np.testing.assert_array_equal(cache["intrinsic"][0], resized_k)
        np.testing.assert_array_equal(cache["source_image_sizes"], [[1920, 1440]])
        np.testing.assert_array_equal(cache["network_image_sizes"], [[1008, 756]])
        np.testing.assert_array_equal(cache["source_to_processed_affine"][0], affine)
        expected_point = np.linalg.inv(resized_k) @ np.array([100., 100., 1.]) * 2.
        np.testing.assert_allclose(cache["world_points"][0, 100, 100], expected_point, atol=1e-6)
        footprint_cache = _validate_cache(cache, {7: images / "000007.png"}, backend="promptda")
        assert footprint_cache["preprocess_resolution"] == 1008
        assert footprint_cache["processed_size"] == [1008, 756]


@pytest.mark.parametrize("mutation, message", [
    (lambda data: data.update(is_metric=np.array(0)), "metric"),
    (lambda data: data.update(frame_ids=np.array([7, 7])), "unique"),
    (lambda data: data.update(depth=np.zeros((2, 28, 28), np.float32)), "positive"),
    (lambda data: data.pop("intrinsics"), "missing"),
    (lambda data: data.update(conf=np.ones((2, 28, 28))), "no native conf"),
    (lambda data: data.update(processed_images=np.zeros((2, 14, 14, 3), np.uint8)), "RGB"),
    (lambda data: data.update(preprocess_method=np.array("unknown_policy")), "preprocess_method"),
    (lambda data: data.update(preprocess_method=np.array("promptda_native_pad14"), network_image_sizes=np.tile([42, 28], (2, 1))), "network_image_sizes"),
    (lambda data: data.update(preprocess_method=np.array("promptda_resize"), preprocess_resolution=np.array(896)), "preprocess_resolution"),
    (lambda data: data.update(preprocess_resolution=np.array(1008)), "preprocess_method"),
    (lambda data: data.update(preprocess_method=np.array("promptda_resize"), preprocess_resolution=np.array(1008.)), "preprocess_resolution"),
])
def test_promptda_cache_rejects_unusable_native_predictions(promptda_input, mutation, message):
    raw, images, output, prediction = promptda_input
    mutation(prediction)
    np.savez(raw, **prediction)
    with pytest.raises(ValueError, match=message):
        convert_promptda_cache(raw, images, output)
    assert not output.exists()


def test_promptda_grid_policy_and_half_pixel_resize():
    assert promptda_processed_size(1920, 1440) == (504, 378)
    assert promptda_processed_size(1920, 1440, 1008) == (1008, 756)
    assert promptda_processed_size(1024, 640) == (504, 308)
    assert promptda_processed_size(280, 210) == (280, 210)
    np.testing.assert_allclose(resize_pixel_center_affine(8, 6, 4, 3), [[.5, 0, -.25], [0, .5, -.25]])
    with pytest.raises(ValueError, match="Unsupported"):
        get_scene_backend("unknown")


def test_promptda_cache_binds_the_actual_source_rgb(promptda_input):
    raw, images, output, _prediction = promptda_input
    Image.fromarray(np.full((28, 28, 3), 19, np.uint8)).save(images / "000007.png")
    with pytest.raises(ValueError, match="source RGB"):
        convert_promptda_cache(raw, images, output)
    assert not output.exists()


def test_promptda_matching_uses_cache_frame_alignment_and_validity_threshold(promptda_input, monkeypatch):
    import torch
    from utils import matching_algorithms as matching
    from utils.config import SKUMatchingConfig, default_sam3_mask_cache_root
    from utils.transforms import build_transforms

    raw, images, output, _prediction = promptda_input
    convert_promptda_cache(raw, images, output)
    config = SKUMatchingConfig.for_3d_mapping(backend="promptda", device="cpu")
    config.enable_sam3_mask_sampling = False
    config.output_dir = str(output.parent.parent / "output_3dmapping_promptda/0")
    assert config.depth_confidence_threshold == config.point_3d_confidence_threshold == 0.
    assert default_sam3_mask_cache_root(images, "promptda") != default_sam3_mask_cache_root(images)
    transforms = build_transforms([str(images / f"{i:06d}.png") for i in (3, 7)], model_type="promptda")
    monkeypatch.setattr(matching, "PI3_SCENE_CACHE", {})
    matching.find_correspondences_3d_mapping(
        None, [{"objects": []}, {"objects": []}], torch.empty(2, 3, 28, 28), config,
        transforms_info=transforms,
    )
    scene = next(iter(matching.PI3_SCENE_CACHE.values()))
    assert scene["image_ids"].tolist() == [3, 7]
    np.testing.assert_allclose(scene["world_points"][0, 0, 0].numpy(), [-.35, -1.35, 2.], atol=1e-6)
    assert transforms[0].map_xy_to_final(2., 5.) == (2., 5.)
    from utils.geometry_3d import sample_3d_points_from_non_overlap_regions

    points = sample_3d_points_from_non_overlap_regions(
        scene, 0, [2., 2., 12., 12.], transforms[0], config, [],
    )
    assert points is not None and len(points) >= config.min_3d_sample_points
    assert torch.isfinite(points).all()


@pytest.mark.parametrize("native", [False, True])
def test_promptda_sam_input_matches_official_area_rgb_and_prefetch_identity(tmp_path, native):
    import cv2
    from utils.sam3_utils import _prepare_sam3_input_image, _sam3_input_prefetch_key
    from utils.transforms import build_transforms

    rgb = np.random.default_rng(2).integers(0, 256, (640, 1024, 3), dtype=np.uint8)
    path = tmp_path / "000007.png"
    Image.fromarray(rgb).save(path)
    transform = build_transforms([str(path)], model_type="promptda")[0]
    if native:
        transform.bind_da3_cache_geometry(np.array([[1., 0., 0.], [0., 1., 0.]]), (640, 1024))
    prepared, aligned = _prepare_sam3_input_image(Image.fromarray(rgb), transform)
    expected = cv2.resize(rgb.astype(np.float32) / 255., (504, 308), interpolation=cv2.INTER_AREA)
    assert aligned
    expected_rgb = rgb if native else np.rint(expected * 255).clip(0, 255).astype(np.uint8)
    np.testing.assert_array_equal(np.asarray(prepared), expected_rgb)
    area_key = _sam3_input_prefetch_key(path, transform)
    transform.rgb_interpolation = "lanczos"
    assert _sam3_input_prefetch_key(path, transform) != area_key
