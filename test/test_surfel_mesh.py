"""Portable Surfel geometry, texture coordinates, and source visibility."""
from io import BytesIO

import numpy as np
from PIL import Image

from src.surfel_mesh import bake_surfel_pages


def fixture(n=1):
    yy, xx = np.mgrid[:9, :9]
    image = np.stack((xx * 20, yy * 20, np.full_like(xx, 100)), axis=-1).astype('uint8')
    stream = BytesIO()
    Image.fromarray(image).save(stream, format='PNG')
    frame = dict(texture='source.png', source_size=[9, 9], texture_size=[9, 9],
                 intrinsic=[[1, 0, 4], [0, 1, 4], [0, 0, 1]],
                 extrinsic=np.eye(4)[:3].tolist(), processed_to_source=np.eye(3).tolist())
    return dict(positions=np.tile([0., 0., 1.], (n, 1)),
                u=np.tile([1., 0., 0.], (n, 1)), v=np.tile([0., 1., 0.], (n, 1)),
                scales=np.ones((n, 2)), frame_indices=np.zeros(n, dtype='uint8'),
                depths=np.ones((1, 9, 9)), frames=[frame], read_asset=lambda _: stream.getvalue(),
                radius=1., tile_size=8, atlas_size=20, batch_size=2)


def decode(page):
    return np.asarray(Image.open(BytesIO(page['image_png'])))


def test_flat_patch_orientation_and_texel_centers():
    page, = bake_surfel_pages(**fixture())
    texture = decode(page)
    assert page['surfel_count'] == 1
    np.testing.assert_allclose(page['positions'], [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    np.testing.assert_allclose(page['uv'], [[.1, .1], [.9, .1], [.9, .9], [.1, .9]])
    np.testing.assert_array_equal(page['indices'], [[0, 1, 2], [0, 2, 3]])
    # Local x/y increase right/down; each source texel is 20 RGB units.
    np.testing.assert_array_equal(texture[4, 4], [78, 78, 100, 255])
    np.testing.assert_array_equal(texture[5, 5], [82, 82, 100, 255])
    assert texture[4, 7, 0] > texture[4, 2, 0]
    assert texture[7, 4, 1] > texture[2, 4, 1]
    assert not texture[0, :, 3].any()
    assert not texture[:, 0, 3].any()
    assert texture[0, 4, :3].sum() > 0  # transparent gutter has dilated color
    assert texture[1, 1, 3] == 0  # outside disk


def test_depth_edge_clips_native_boundary():
    args = fixture()
    args['depths'][0, :, 5:] = 3
    page, = bake_surfel_pages(**args)
    texture = decode(page)
    assert texture[4, 6, 3] == 255  # projected x=4.375
    assert texture[4, 7, 3] == 0    # projected x=4.625, past +0.5 cell boundary


def test_source_depth_test_rejects_interior_even_with_continuous_neighbors():
    args = fixture()
    args['depths'][0, 4, 4] = 2
    page, = bake_surfel_pages(**args)
    texture = decode(page)
    assert texture[4, 4, 3] == 0  # native four neighbors remain continuous
    assert texture[4, 7, 3] == 255


def test_scale_applies_after_footprint_clamp_and_degenerate_skipped():
    args = fixture(2)
    args['u'][0] = [10, 0, 0]
    args['scales'][0] = [2, 1]
    args['v'][1] = 0
    page, = bake_surfel_pages(**args)
    assert page['surfel_count'] == 1
    np.testing.assert_allclose(page['positions'][:, 0], [-4, 4, 4, -4])
    np.testing.assert_allclose(page['positions'][:, 1], [-1, -1, 1, 1])


def test_multiple_pages_have_compact_valid_dimensions():
    pages = list(bake_surfel_pages(**fixture(5)))
    assert [p['surfel_count'] for p in pages] == [4, 1]
    assert [decode(p).shape for p in pages] == [(20, 20, 4), (10, 10, 4)]
    assert all(p['positions'].dtype == np.float32 and p['uv'].dtype == np.float32
               and p['indices'].dtype == np.uint32 for p in pages)
    assert pages[0]['indices'].max() == 15
    assert pages[1]['indices'].max() == 3


def test_all_invisible_yields_no_pages():
    args = fixture()
    args['depths'][:] = 0
    assert list(bake_surfel_pages(**args)) == []


def test_resized_source_texture_uses_half_pixel_mapping():
    args = fixture()
    args['frames'][0]['source_size'] = [18, 18]
    args['frames'][0]['processed_to_source'] = [[2, 0, .5], [0, 2, .5], [0, 0, 1]]
    page, = bake_surfel_pages(**args)
    np.testing.assert_array_equal(decode(page)[4, 4], [78, 78, 100, 255])


def test_smaller_texture_layer_preserves_zero_padding_filter():
    args = fixture()
    frame = args['frames'][0]
    frame['source_size'] = [18, 18]
    frame['processed_to_source'] = [[0, 0, 17], [0, 0, 8.5], [0, 0, 1]]
    args['frames'].append(dict(frame, texture_size=[18, 18]))
    page, = bake_surfel_pages(**args)
    # x=8.25 samples 75% final photo texel and 25% array-layer zero padding.
    np.testing.assert_array_equal(decode(page)[4, 4], [120, 60, 75, 255])


def test_zero_and_negative_camera_depth_are_skipped_without_warnings():
    import warnings

    args = fixture(3)
    args['positions'][:, 2] = [0, -1, 1]
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        page, = bake_surfel_pages(**args)
    assert page['surfel_count'] == 1
    np.testing.assert_array_equal(page['positions'][:, 2], np.ones(4))
