"""Portable Surfel geometry, texture coordinates, and source visibility."""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.surfel_mesh import _dilate_rgb, bake_surfel_pages


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


def quad_axes(page, index=0):
    corners = page['positions'].reshape(-1, 4, 3)[index]
    return (corners[1] - corners[0]) / 2, (corners[3] - corners[0]) / 2


def quad_diameter(page, index=0):
    bu, bv = quad_axes(page, index)
    return 2 * np.linalg.svd(np.stack((bu, bv), axis=1), compute_uv=False)[0]


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


def test_scale_applies_after_footprint_clamp_within_final_cap_and_degenerate_skipped():
    args = fixture(2)
    args['u'][0] = [10, 0, 0]
    args['scales'][0] = [1.25, 1]
    args['v'][1] = 0
    page, = bake_surfel_pages(**args)
    assert page['surfel_count'] == 1
    np.testing.assert_allclose(page['positions'][:, 0], [-2.5, 2.5, 2.5, -2.5])
    np.testing.assert_allclose(page['positions'][:, 1], [-1, -1, 1, 1])


def test_final_diameter_cap_limits_radius_and_anisotropic_scale_bake_geometry():
    args = fixture()
    args['scales'][0] = [8, .5]
    args['radius'] = 2
    page, = bake_surfel_pages(**args)
    bu, bv = quad_axes(page)
    np.testing.assert_allclose(bu, [3, 0, 0])
    np.testing.assert_allclose(bv, [0, 1, 0])
    np.testing.assert_allclose(quad_diameter(page), 6)


def test_final_diameter_cap_preserves_skew_plane_short_value_and_major_direction():
    args = fixture()
    args['u'][0] = [1, 0, 0]
    args['v'][0] = [1, 1, 0]
    args['scales'][0] = [8, .5]
    args['radius'] = 2
    page, = bake_surfel_pages(**args)
    bu, bv = quad_axes(page)
    raw_bu, raw_bv = np.array([16, 0, 0]), np.array([1, 1, 0])
    raw_directions, raw_values, _ = np.linalg.svd(
        np.stack((raw_bu, raw_bv), axis=1), full_matrices=False
    )
    baked_directions, baked_values, _ = np.linalg.svd(
        np.stack((bu, bv), axis=1), full_matrices=False
    )
    np.testing.assert_allclose(baked_values, [3, raw_values[1]], rtol=2e-6)
    np.testing.assert_allclose(abs(raw_directions[:, 0] @ baked_directions[:, 0]), 1,
                               rtol=2e-6)
    np.testing.assert_allclose(np.cross(bu, bv) / np.linalg.norm(np.cross(bu, bv)), [0, 0, 1])
    assert quad_diameter(page) <= 6 + 1e-6


def test_final_diameter_cap_limits_both_axes_of_oversize_isotropic_footprint():
    args = fixture()
    args['scales'][0] = [8, 8]
    args['radius'] = 2
    page, = bake_surfel_pages(**args)
    bu, bv = quad_axes(page)
    np.testing.assert_allclose(bu, [3, 0, 0])
    np.testing.assert_allclose(bv, [0, 3, 0])
    np.testing.assert_allclose(quad_diameter(page), 6)


def test_final_diameter_cap_leaves_small_footprint_unchanged():
    args = fixture()
    args['u'][0] = [.75, 0, 0]
    args['v'][0] = [0, .25, 0]
    page, = bake_surfel_pages(**args)
    bu, bv = quad_axes(page)
    np.testing.assert_allclose(bu, [.75, 0, 0])
    np.testing.assert_allclose(bv, [0, .25, 0])


def test_final_diameter_cap_tracks_native_grid_at_different_depths():
    diameters = []
    for distance in (1, 2):
        args = fixture()
        args['positions'][0, 2] = distance
        args['depths'][:] = distance
        args['u'][0] = [10, 0, 0]
        args['scales'][0] = [8, .5]
        args['radius'] = 2
        page, = bake_surfel_pages(**args)
        diameters.append(quad_diameter(page))
    np.testing.assert_allclose(diameters, [6, 12])


def test_multiple_pages_have_compact_valid_dimensions():
    pages = list(bake_surfel_pages(**fixture(5)))
    assert [p['surfel_count'] for p in pages] == [4, 1]
    assert [decode(p).shape for p in pages] == [(20, 20, 4), (10, 10, 4)]
    assert all(p['positions'].dtype == np.float32 and p['uv'].dtype == np.float32
               and p['indices'].dtype == np.uint32 for p in pages)
    assert pages[0]['indices'].max() == 15
    assert pages[1]['indices'].max() == 3


def test_raw_atlas_pages_preserve_geometry_and_transparent_rgb():
    encoded = list(bake_surfel_pages(**fixture(5)))
    raw = list(bake_surfel_pages(**fixture(5), image_format='rgba'))
    assert len(raw) == len(encoded) == 2
    for actual, expected in zip(raw, encoded):
        assert 'image_png' not in actual
        np.testing.assert_array_equal(actual['image_rgba'], decode(expected))
        for key in ('positions', 'uv', 'indices', 'surfel_count'):
            np.testing.assert_array_equal(actual[key], expected[key])


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


def _roll_dilate_rgb_reference(tiles):
    """Extend nearest encountered opaque color without changing coverage alpha."""
    known = tiles[..., 3] != 0
    for _ in range(tiles.shape[1] + tiles.shape[2]):
        if known.all():
            break
        old = known.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            shifted = np.roll(old, (dy, dx), axis=(1, 2))
            if dy == 1:
                shifted[:, 0] = False
            elif dy == -1:
                shifted[:, -1] = False
            if dx == 1:
                shifted[:, :, 0] = False
            elif dx == -1:
                shifted[:, :, -1] = False
            fill = shifted & ~known
            tiles[..., :3][fill] = np.roll(tiles[..., :3], (dy, dx), axis=(1, 2))[fill]
            known |= fill
    return tiles


@pytest.mark.parametrize('shape', [
    (0, 3, 4, 4), (2, 0, 4, 4), (2, 3, 0, 4), (1, 0, 0, 4),
    (1, 1, 1, 4), (2, 1, 7, 4), (2, 7, 1, 4), (3, 5, 7, 4),
])
@pytest.mark.parametrize('coverage', ['known', 'transparent', 'mixed'])
def test_dilate_rgb_matches_original_roll_rgba_in_place(shape, coverage):
    rng = np.random.default_rng(42)
    tiles = rng.integers(0, 256, size=shape, dtype=np.uint8)
    if coverage == 'known':
        tiles[..., 3] = rng.integers(1, 256, size=shape[:-1], dtype=np.uint8)
    elif coverage == 'transparent':
        tiles[..., 3] = 0
    else:
        tiles[..., 3] *= rng.integers(0, 2, size=shape[:-1], dtype=np.uint8)
    alpha = tiles[..., 3].copy()
    expected = _roll_dilate_rgb_reference(tiles.copy())
    actual = _dilate_rgb(tiles)
    assert actual is tiles
    np.testing.assert_array_equal(actual, expected)  # Includes transparent RGB.
    np.testing.assert_array_equal(actual[..., 3], alpha)


def test_dilate_rgb_first_direction_wins_tie():
    tiles = np.zeros((1, 3, 3, 4), dtype=np.uint8)
    tiles[0, 1, 0] = [10, 20, 30, 1]
    tiles[0, 1, 2] = [40, 50, 60, 255]
    tiles[0, 0, 1] = [70, 80, 90, 128]
    tiles[0, 2, 1] = [100, 110, 120, 255]
    alpha = tiles[..., 3].copy()
    expected = _roll_dilate_rgb_reference(tiles.copy())
    actual = _dilate_rgb(tiles)
    assert actual is tiles
    np.testing.assert_array_equal(actual[0, 1, 1], [10, 20, 30, 0])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[..., 3], alpha)
