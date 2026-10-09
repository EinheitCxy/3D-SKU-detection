"""Bake source-space Surfel appearance into portable, masked texture pages.

The source visibility tests are evaluated at tile texel centers. View-dependent
front-depth accumulation in the interactive renderer is intentionally absent.
"""
from io import BytesIO
import operator

import numpy as np
from PIL import Image


def _bound_footprint(u, v, native):
    a, b = u / native[:, :1], v / native[:, 1:]
    aa, ab, bb = (a * a).sum(1), (a * b).sum(1), (b * b).sum(1)
    gap = np.hypot(aa - bb, 2 * ab)
    major = np.sqrt(np.maximum(0, (aa + bb + gap) * 0.5))
    minor = np.sqrt(np.maximum(0, (aa + bb - gap) * 0.5))
    major_scale = np.minimum(1, np.minimum(2, 4 * minor) / np.maximum(major, 1e-8))
    minor_scale = np.minimum(1, 2 / np.maximum(minor, 1e-8))
    factor = np.divide(major_scale - minor_scale, gap, out=np.zeros_like(gap), where=gap > 1e-6)
    c00 = minor_scale + factor * (aa - minor * minor)
    c11 = minor_scale + factor * (bb - minor * minor)
    c01 = factor * ab
    c00 = np.where(gap > 1e-6, c00, major_scale)
    c11 = np.where(gap > 1e-6, c11, major_scale)
    return ((a * c00[:, None] + b * c01[:, None]) * native[:, :1],
            (a * c01[:, None] + b * c11[:, None]) * native[:, 1:])


def _bilinear(image, x, y):
    h, w = image.shape[:2]
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return ((image[y0, x0] * (1 - fx) + image[y0, x1] * fx) * (1 - fy)
            + (image[y1, x0] * (1 - fx) + image[y1, x1] * fx) * fy)


def _dilate_rgb(tiles):
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


def _bake_batch(p, u, v, scales, depth, frame, image, radius, tile_size):
    e = np.asarray(frame['extrinsic'], dtype=np.float32)
    k = np.asarray(frame['intrinsic'], dtype=np.float32)
    inv_k = np.linalg.inv(k)
    affine = np.asarray(frame['processed_to_source'], dtype=np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        center = p @ e[:3, :3].T + e[:3, 3]
        projected = center @ k.T
    visible = (np.isfinite(center).all(axis=1) & (center[:, 2] > 0)
               & np.isfinite(projected).all(axis=1) & (projected[:, 2] > 0))
    p, u, v, scales = (a[visible] for a in (p, u, v, scales))
    center, projected = center[visible], projected[visible]
    if not len(p):
        return (np.empty((0, 4, 3), dtype=np.float32),
                np.empty((0, tile_size + 2, tile_size + 2, 4), dtype=np.uint8))
    native = np.maximum(1e-6, center[:, 2:3] * np.linalg.norm(inv_k[:, :2], axis=0))
    bu, bv = _bound_footprint(u, v, native)
    pixel = np.floor(projected[:, :2] / projected[:, 2:3] + 0.5).astype(np.intp)
    depth_step = np.stack((u @ e[2, :3], v @ e[2, :3]), axis=1)
    h, w = depth.shape
    bounds = np.tile([-1e6, 1e6, -1e6, 1e6], (len(p), 1)).astype(np.float32)
    for column, (dx, dy, axis, sign) in enumerate(((-1, 0, 0, -1), (1, 0, 0, 1), (0, -1, 1, -1), (0, 1, 1, 1))):
        x, y = pixel[:, 0] + dx, pixel[:, 1] + dy
        d = depth[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
        continuous = ((x >= 0) & (x < w) & (y >= 0) & (y < h) & (d > 0)
                      & (np.abs(d - center[:, 2] - sign * depth_step[:, axis]) <= 0.001 + 0.5 * native[:, axis]))
        bounds[:, column] = np.where(continuous, bounds[:, column], pixel[:, axis] + sign * 0.5)
    bu = bu * (radius * scales[:, :1])
    bv = bv * (radius * scales[:, 1:])
    q = (np.arange(tile_size, dtype=np.float32) + 0.5) * (2 / tile_size) - 1
    x, y = np.meshgrid(q, q)
    world = p[:, None, None] + x[None, ..., None] * bu[:, None, None] + y[None, ..., None] * bv[:, None, None]
    camera = world @ e[:3, :3].T + e[:3, 3]
    projection = camera @ k.T
    processed = np.divide(projection[..., :2], projection[..., 2:3], out=np.zeros_like(projection[..., :2]), where=projection[..., 2:3] != 0)
    px, py = processed[..., 0], processed[..., 1]
    valid = ((x * x + y * y <= 1)[None] & (camera[..., 2] > 0)
             & (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
             & (px >= bounds[:, None, None, 0]) & (px <= bounds[:, None, None, 1])
             & (py >= bounds[:, None, None, 2]) & (py <= bounds[:, None, None, 3]))
    ix, iy = np.floor(px + 0.5).astype(np.intp), np.floor(py + 0.5).astype(np.intp)
    sampled = depth[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)]
    valid &= (sampled > 0) & (np.abs(sampled - camera[..., 2]) <= 0.015 * sampled + 0.001)
    source = processed @ affine[:2, :2].T + affine[:2, 2]
    sw, sh = frame['source_size']
    valid &= ((source[..., 0] >= 0) & (source[..., 0] <= sw - 1)
              & (source[..., 1] >= 0) & (source[..., 1] <= sh - 1))
    valid &= (np.linalg.norm(np.cross(bu, bv), axis=1) > 0)[:, None, None]
    keep = valid.any(axis=(1, 2))
    source, valid = source[keep], valid[keep]
    tw, th = frame['texture_size']
    rgb = _bilinear(image, (source[..., 0] + 0.5) / sw * tw - 0.5,
                    (source[..., 1] + 0.5) / sh * th - 0.5)
    tiles = np.zeros((keep.sum(), tile_size + 2, tile_size + 2, 4), dtype=np.uint8)
    tiles[:, 1:-1, 1:-1, :3] = np.rint(rgb).astype(np.uint8)
    tiles[:, 1:-1, 1:-1, 3] = valid * 255
    corners = np.stack((p - bu - bv, p + bu - bv, p + bu + bv, p - bu + bv), axis=1)[keep]
    return corners, _dilate_rgb(tiles)


def _page(corners, tiles, count, columns, tile_size):
    columns = min(columns, count)
    rows = (count + columns - 1) // columns
    stride = tile_size + 2
    padded = np.zeros((rows * columns, stride, stride, 4), dtype=np.uint8)
    padded[:count] = tiles[:count]
    image = padded.reshape(rows, columns, stride, stride, 4).transpose(0, 2, 1, 3, 4).reshape(rows * stride, columns * stride, 4)
    stream = BytesIO()
    Image.fromarray(image).save(stream, format='PNG')
    slot = np.arange(count)
    origin = np.stack((slot % columns, slot // columns), axis=1) * stride + 1
    # UV bounds lie at inner tile edges: texel centers map to disk sample centers.
    uv = (origin[:, None] + np.array([[0, 0], [tile_size, 0], [tile_size, tile_size], [0, tile_size]])[None]) / [columns * stride, rows * stride]
    indices = np.arange(count, dtype=np.uint32)[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return dict(positions=corners[:count].reshape(-1, 3).copy(), uv=uv.astype(np.float32).reshape(-1, 2),
                indices=indices.reshape(-1, 3), image_png=stream.getvalue(), surfel_count=count)


def bake_surfel_pages(*, positions, u, v, scales, frame_indices, depths, frames,
                      read_asset, radius=2.0, tile_size=8, atlas_size=2048, batch_size=1024):
    """Yield independent bounded atlas pages in source-frame order."""
    tile_size, atlas_size, batch_size = map(operator.index, (tile_size, atlas_size, batch_size))
    if tile_size < 2 or atlas_size < tile_size + 2 or atlas_size > 2048 or batch_size < 1:
        raise ValueError('Require tile_size >= 2, tile_size + 2 <= atlas_size <= 2048, batch_size >= 1')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be finite and positive')
    positions, u, v, scales = (np.asarray(a, dtype=np.float32) for a in (positions, u, v, scales))
    frame_indices = np.asarray(frame_indices)
    n = len(positions)
    if positions.shape != (n, 3) or u.shape != (n, 3) or v.shape != (n, 3) or scales.shape != (n, 2) or frame_indices.shape != (n,):
        raise ValueError('Surfel arrays have incompatible shapes')
    if not all(np.isfinite(a).all() for a in (positions, u, v, scales)):
        raise ValueError('Surfel geometry must be finite')
    if not np.issubdtype(frame_indices.dtype, np.integer) or np.any(frame_indices < 0) or np.any(frame_indices >= len(frames)):
        raise ValueError('Invalid Surfel frame indices')
    columns = atlas_size // (tile_size + 2)
    capacity = columns * columns
    page_corners = np.empty((capacity, 4, 3), dtype=np.float32)
    page_tiles = np.empty((capacity, tile_size + 2, tile_size + 2, 4), dtype=np.uint8)
    filled = 0
    for frame_id in np.unique(frame_indices):
        frame = frames[int(frame_id)]
        with Image.open(BytesIO(read_asset(frame['texture']))) as texture:
            image = np.asarray(texture.convert('RGB'), dtype=np.float32)
        if tuple(frame['texture_size']) != (image.shape[1], image.shape[0]):
            raise ValueError('Source texture dimensions differ from metadata')
        # The WebGL array pads smaller photographs with zero RGB to the
        # largest layer dimensions; retain that bilinear edge behavior.
        width = max(f['texture_size'][0] for f in frames)
        height = max(f['texture_size'][1] for f in frames)
        if image.shape[:2] != (height, width):
            padded = np.zeros((height, width, 3), dtype=np.float32)
            padded[:image.shape[0], :image.shape[1]] = image
            image = padded
        frame_depth = np.asarray(depths[int(frame_id)], dtype=np.float32)
        slots = np.flatnonzero(frame_indices == frame_id)
        for start in range(0, len(slots), batch_size):
            batch = slots[start:start + batch_size]
            corners, tiles = _bake_batch(positions[batch], u[batch], v[batch], scales[batch],
                                         frame_depth, frame, image, radius, tile_size)
            offset = 0
            while offset < len(corners):
                take = min(capacity - filled, len(corners) - offset)
                page_corners[filled:filled + take] = corners[offset:offset + take]
                page_tiles[filled:filled + take] = tiles[offset:offset + take]
                filled += take
                offset += take
                if filled == capacity:
                    yield _page(page_corners, page_tiles, filled, columns, tile_size)
                    filled = 0
    if filled:
        yield _page(page_corners, page_tiles, filled, columns, tile_size)
