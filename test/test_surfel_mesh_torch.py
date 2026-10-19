"""Tiny CPU numerical gates for the bounded experimental baker; no GPU needed."""
from io import BytesIO

import numpy as np
from PIL import Image
import pytest
import torch

from src import surfel_mesh_torch as candidate
from test import test_surfel_mesh as existing
from src.surfel_mesh import _bake_batch, _bilinear, bake_surfel_pages


def batch(args, image=None):
    frame = args['frames'][0]
    if image is None:
        image = np.asarray(Image.open(BytesIO(args['read_asset'](frame['texture']))).convert('RGB'), dtype=np.float32)
    arrays = [np.asarray(args[key], dtype=np.float32) for key in ('positions', 'u', 'v', 'scales')]
    depth = np.asarray(args['depths'][0], dtype=np.float32)
    expected = _bake_batch(*arrays, depth, frame, image, args['radius'], args['tile_size'])
    for fixed in (False, True):
        baker = candidate.FrameBaker(frame, image, depth, radius=args['radius'], tile_size=args['tile_size'], fixed_shape=fixed)
        actual = baker.bake(*arrays)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)
            assert got.dtype == want.dtype
    return actual, baker


def test_bilinear_clip_f64_and_hand_computed_coordinates():
    image = np.array([[[0, 2, 4], [2, 4, 6]], [[4, 6, 8], [6, 8, 10]]], dtype=np.float32)
    x = np.array([-10, 10, .5, .25, 0], dtype=np.float32)
    y = np.array([0, 1, .5, .75, 0], dtype=np.float32)
    actual = candidate.bilinear(torch.from_numpy(image), torch.from_numpy(x), torch.from_numpy(y))
    expected = np.array([0, 6, 3, 3.5, 0])[:, None] + [0, 2, 4]
    assert actual.dtype == torch.float64
    np.testing.assert_array_equal(actual.numpy(), expected)
    np.testing.assert_array_equal(actual.numpy(), _bilinear(image, x, y))


def test_rgb_round_ties_even_and_binary_alpha():
    args = existing.fixture()
    args['frames'][0]['processed_to_source'] = [[0, 0, .5], [0, 0, .5], [0, 0, 1]]
    image = np.empty((9, 9, 3), dtype=np.float32)
    image[:] = [1, 2, 3]
    image[:, 0] = [0, 1, 2]
    (_, tiles), _ = batch(args, image)
    np.testing.assert_array_equal(tiles[0, 4, 4], [0, 2, 2, 255])
    assert set(np.unique(tiles[..., 3])) == {0, 255}


def test_depth_nearest_half_tie_is_floor_plus_half_in_both_dimensions():
    args = existing.fixture()
    args['radius'] = 4 / 7
    args['depths'][0, :, 5:] = 3
    (_, tiles), _ = batch(args)
    # q=+/-.875 scaled by 4/7 projects to 3.5/4.5. The two central
    # rows lie inside the disk; final x=4.5 must gather col 5, not 4.
    np.testing.assert_array_equal(tiles[0, 4:6, 1:-1, 3], [[255]*7+[0], [255]*7+[0]])


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('coordinate,kept', [(0., True), (8., True), (-1e-6, False),
                                            (np.nextafter(np.float32(8), np.float32(np.inf)), False)])
def test_source_inclusive_2d_boundaries(axis, coordinate, kept):
    args = existing.fixture()
    affine = np.zeros((3, 3), dtype=np.float32)
    affine[2, 2] = 1
    affine[:2, 2] = 4
    affine[axis, 2] = coordinate
    args['frames'][0]['processed_to_source'] = affine.tolist()
    (corners, tiles), _ = batch(args)
    assert len(corners) == len(tiles) == int(kept)


def test_depth_2d_edges_and_interior_hole():
    args = existing.fixture()
    args['depths'][0, 4, 4] = 2
    args['depths'][0, 5:, :] = 3
    args['depths'][0, :, 5:] = 3
    (_, tiles), _ = batch(args)
    assert tiles[0, 2, 2, 3] == 255
    assert tiles[0, 4, 4, 3] == tiles[0, 7, 4, 3] == tiles[0, 4, 7, 3] == 0


@pytest.mark.parametrize('above', [False, True])
def test_depth_threshold_adjacent_float32(above):
    args = existing.fixture()
    boundary = np.float32(1.016)
    args['positions'][0, 2] = boundary if above else np.nextafter(boundary, np.float32(0))
    (corners, _), _ = batch(args)
    assert len(corners) == int(not above)


@pytest.mark.parametrize('shape', [(0, 3, 4, 4), (2, 0, 4, 4), (2, 3, 0, 4), (2, 5, 7, 4)])
@pytest.mark.parametrize('unknown', [False, True])
def test_dilation_old_known_all_unknown_and_empty(shape, unknown):
    rng = np.random.default_rng(52)
    tiles = rng.integers(0, 256, size=shape, dtype=np.uint8)
    tiles[..., 3] = 0 if unknown else tiles[..., 3] * rng.integers(0, 2, size=shape[:-1], dtype=np.uint8)
    expected = existing._roll_dilate_rgb_reference(tiles.copy())
    tensor = torch.from_numpy(tiles.copy())
    assert candidate.dilate_rgb(tensor) is tensor
    np.testing.assert_array_equal(tensor.numpy(), expected)
    np.testing.assert_array_equal(tensor.numpy()[..., 3], tiles[..., 3])
    if unknown:
        np.testing.assert_array_equal(tensor.numpy(), tiles)


def test_dilation_first_direction_wins_without_same_round_propagation():
    tiles = np.zeros((1, 3, 5, 4), dtype=np.uint8)
    tiles[0, 1, 0] = [10, 20, 30, 1]
    tiles[0, 1, 4] = [40, 50, 60, 255]
    tiles[0, 0, 2] = [70, 80, 90, 128]
    # Correct old-known gives up seed center in round 1; an in-round flood
    # from the left would incorrectly reach it first. Center's left/right tie
    # one row down is won by direction left->right on the next round.
    expected = existing._roll_dilate_rgb_reference(tiles.copy())
    actual = candidate.dilate_rgb(torch.from_numpy(tiles.copy())).numpy()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[0, 1, 2], [70, 80, 90, 0])
    tie = np.zeros((1, 3, 3, 4), dtype=np.uint8)
    tie[0, 1, 0] = [10, 20, 30, 1]
    tie[0, 1, 2] = [40, 50, 60, 255]
    tie[0, 0, 1] = [70, 80, 90, 128]
    tie[0, 2, 1] = [100, 110, 120, 255]
    result = candidate.dilate_rgb(torch.from_numpy(tie)).numpy()
    np.testing.assert_array_equal(result[0, 1, 1], [10, 20, 30, 0])


def test_empty_batch_and_frame_uploads_are_reused():
    args = existing.fixture(0)
    (corners, tiles), baker = batch(args)
    assert corners.shape == (0, 4, 3) and corners.dtype == np.float32
    assert tiles.shape == (0, 10, 10, 4) and tiles.dtype == np.uint8
    one = existing.fixture()
    baker.bake(*(one[key] for key in ('positions', 'u', 'v', 'scales')))
    assert baker.counts['rgb_uploads'] == baker.counts['depth_uploads'] == 1
    assert baker.counts['batches'] == 2



def test_fixed_shape_stable_compaction_mixed_invisible_and_degenerate():
    args = existing.fixture(5)
    args['positions'][0, 2] = -1
    args['v'][2] = 0
    args['positions'][3] = [100, 100, 1]
    (corners, tiles), _ = batch(args)
    assert len(corners) == len(tiles) == 2


def test_cpu_default_never_imports_torch_backend(tmp_path):
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c',
        "import sys; from src.surfel_mesh import bake_surfel_pages; "
        "from test.test_surfel_mesh import fixture; list(bake_surfel_pages(**fixture())); "
        "assert 'src.surfel_mesh_torch' not in sys.modules; "
        "assert 'torch' not in sys.modules or not sys.modules['torch'].cuda.is_initialized()"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('options', [dict(bake_device='auto'), dict(bake_device='cuda'),
    dict(bake_device='cuda:0', tile_size=4), dict(bake_device='cuda:0', batch_size=2048),
    dict(bake_device='cuda:0', projection_mode='camera_basis'), dict(bake_fixed_shape=True)])
def test_invalid_device_combination_before_io(options):
    args = existing.fixture()
    args.update(options)
    args['read_asset'] = lambda _: pytest.fail('must reject before I/O')
    with pytest.raises(ValueError):
        list(bake_surfel_pages(**args))


@pytest.mark.parametrize('fixed_shape', [False, True])
def test_shared_paging_with_cpu_device_adapter(monkeypatch, fixed_shape):
    original = candidate.FrameBaker

    def cpu_adapter(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return original(*args, **kwargs)

    monkeypatch.setattr(candidate, 'FrameBaker', cpu_adapter)
    args = existing.fixture(11)
    args['v'][3] = 0
    args['positions'][5, 2] = -1
    # Crossing batches and four-tile pages must preserve compaction and page order.
    expected = list(bake_surfel_pages(**args))
    actual = list(bake_surfel_pages(**args, bake_device='cuda:0', bake_fixed_shape=fixed_shape))
    assert len(actual) == len(expected) == 3
    for got, want in zip(actual, expected):
        for key in ('positions', 'uv', 'indices'):
            np.testing.assert_array_equal(got[key], want[key])
        np.testing.assert_array_equal(existing.decode(got), existing.decode(want))
