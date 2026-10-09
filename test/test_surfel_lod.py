"""Conservative source-pixel thinning keeps original geometry and boundaries."""
import numpy as np
import pytest

from src.surfel_lod import select_surfel_lod


def patch():
    return dict(positions=np.array([[2., 2., 1.], [3., 2., 1.],
                                    [2., 3., 1.], [3., 3., 1.]]),
                u=np.tile([1., 0., 0.], (4, 1)),
                v=np.tile([0., 1., 0.], (4, 1)), scales=np.ones((4, 2)),
                frame_indices=np.zeros(4, dtype=int), object_ids=np.ones(4, dtype=int),
                frames=[dict(extrinsic=np.eye(4)[:3], intrinsic=np.eye(3))],
                depths=np.ones((1, 8, 8)))


def test_complete_flat_cell_selects_original_top_left_without_mutation():
    args = patch()
    before = {k: v.copy() for k, v in args.items() if isinstance(v, np.ndarray)}
    indices, report = select_surfel_lod(**args)
    np.testing.assert_array_equal(indices, [0])
    assert report == dict(input_count=4, selected_count=1, removed_count=3,
                          eligible_blocks=1, reduced_blocks=1)
    for key, value in before.items():
        np.testing.assert_array_equal(args[key], value)


@pytest.mark.parametrize('boundary', ['owner', 'unknown', 'depth', 'normal', 'zero_normal',
                                    'scale', 'thin_ellipse', 'depth_neighbor', 'duplicate'])
def test_boundaries_keep_all(boundary):
    args = patch()
    if boundary == 'owner':
        args['object_ids'][3] = 2
    elif boundary == 'unknown':
        args['object_ids'][:] = -1
    elif boundary == 'depth':
        args['positions'][3] *= 1.02
    elif boundary == 'normal':
        args['u'][3] *= -1
    elif boundary == 'zero_normal':
        args['u'][3] = 0
    elif boundary == 'scale':
        args['scales'][0, 0] = .5
    elif boundary == 'thin_ellipse':
        args['u'] *= .1
    elif boundary == 'depth_neighbor':
        args['depths'][0, 2, 1] = 3
    elif boundary == 'duplicate':
        args['positions'][3] = args['positions'][0]
    indices, report = select_surfel_lod(**args)
    np.testing.assert_array_equal(indices, np.arange(4))
    assert report['reduced_blocks'] == 0


def test_incomplete_and_invalid_projection_remain_for_baker():
    args = patch()
    args['positions'][3] = [0, 0, -1]
    indices, _ = select_surfel_lod(**args)
    np.testing.assert_array_equal(indices, np.arange(4))
    args['positions'][3] = [np.nan, 0, 1]
    indices, _ = select_surfel_lod(**args)
    np.testing.assert_array_equal(indices, np.arange(4))


def test_input_order_does_not_change_selected_pixel_and_frames_do_not_merge():
    args = patch()
    order = [3, 2, 1, 0]
    for key in ('positions', 'u', 'v', 'scales', 'frame_indices', 'object_ids'):
        args[key] = args[key][order]
    indices, _ = select_surfel_lod(**args)
    np.testing.assert_array_equal(indices, [3])
    args['frames'] *= 2
    args['depths'] = np.ones((2, 8, 8))
    args['frame_indices'][0] = 1
    indices, _ = select_surfel_lod(**args)
    np.testing.assert_array_equal(indices, np.arange(4))


def test_empty_input():
    args = patch()
    for key in ('positions', 'u', 'v', 'scales', 'frame_indices', 'object_ids'):
        args[key] = args[key][:0]
    indices, report = select_surfel_lod(**args)
    assert len(indices) == report['selected_count'] == 0
