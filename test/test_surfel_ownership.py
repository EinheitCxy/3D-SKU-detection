"""Synthetic coverage for single-frame Surfel ownership."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from src.surfel_ownership import frame_sharpness_weights, select_single_owner


def _select(points, frames, labels=None, normals=None, centers=None, confidence=None):
    points = np.asarray(points, dtype=np.float32)
    frames = np.asarray(frames, dtype=np.int64)
    frame_count = max(2, int(frames.max()) + 1)
    return select_single_owner(
        points=points,
        frames=frames,
        labels=np.zeros(len(points), dtype=np.int64) if labels is None else labels,
        normals=np.tile([0, 0, 1], (len(points), 1)) if normals is None else normals,
        centers=np.zeros((frame_count, 3), dtype=np.float32) if centers is None else centers,
        confidence=np.ones(len(points), dtype=np.float32) if confidence is None else confidence,
        sharpness=np.ones(frame_count, dtype=np.float32),
        spacing=0.01,
        patch_cells=8,
    )


def test_patch_keeps_frame_with_higher_view_cosine():
    points = [[0, 0, 1], [0, 0, 1.004]]
    centers = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    keep = _select(points, [0, 1], centers=centers)
    assert np.array_equal(keep, [True, False])


def test_fine_voxel_holes_are_refilled_from_their_best_observed_frame():
    points = [[0.001, 0, 1], [0.001, 0, 1], [0.011, 0, 1]]
    normals = np.zeros((3, 3), dtype=np.float32)
    keep = _select(points, [0, 1, 1], normals=normals)
    assert np.array_equal(keep, [True, False, True])


def test_label_seen_only_by_lower_scoring_frame_is_retained():
    points = [[0, 0, 1], [0, 0, 1.004], [0, 0, 1.004]]
    labels = np.array([1, 1, 7])
    centers = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    keep = _select(points, [0, 1, 1], labels=labels, centers=centers)
    assert np.array_equal(keep, [True, False, True])


def test_selection_is_deterministic():
    args = dict(
        points=[[0, 0, 1], [0, 0, 1.004], [0.011, 0, 1]],
        frames=[0, 1, 1],
        normals=np.zeros((3, 3), dtype=np.float32),
    )
    assert np.array_equal(_select(**args), _select(**args))


def test_zero_normals_choose_only_an_observed_frame():
    keep = select_single_owner(
        points=np.array([[0, 0, 1], [0.001, 0, 1]], dtype=np.float32),
        frames=np.array([3, 3]),
        labels=np.array([0, 0]),
        normals=np.zeros((2, 3), dtype=np.float32),
        centers=np.zeros((4, 3), dtype=np.float32),
        confidence=np.ones(2, dtype=np.float32),
        sharpness=np.ones(4, dtype=np.float32),
        spacing=0.01,
        patch_cells=8,
    )
    assert keep.all()


def test_sharpness_weights_rank_blurred_and_clear_images(tmp_path: Path):
    checker = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
    clear_path = tmp_path / "clear.png"
    blurred_path = tmp_path / "blurred.png"
    clear = Image.fromarray(checker, mode="L")
    clear.save(clear_path)
    clear.filter(ImageFilter.GaussianBlur(radius=4)).save(blurred_path)

    weights = frame_sharpness_weights([blurred_path, clear_path])
    assert 0.25 <= weights[0] < weights[1] <= 1


def test_sharpness_weights_handle_zero_median(tmp_path: Path):
    flat_paths = [tmp_path / "flat-1.png", tmp_path / "flat-2.png"]
    for path in flat_paths:
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L").save(path)
    textured_path = tmp_path / "textured.png"
    checker = ((np.indices((32, 32)).sum(axis=0) % 2) * 255).astype(np.uint8)
    Image.fromarray(checker, mode="L").save(textured_path)

    weights = frame_sharpness_weights([*flat_paths, textured_path])
    assert np.array_equal(weights, [0.25, 0.25, 1])
