"""Bounded experiment: native NumPy source_world projection, Torch masks/RGB.

No production hooks, GPU auto-selection, fallback, compiler or new dependency.
The caller explicitly chooses cpu or its isolated logical CUDA device.
"""
from __future__ import annotations

from io import BytesIO
import operator
import time

import numpy as np
from PIL import Image
import torch

from src.surfel_mesh import _bound_footprint, _frame_constants, _limit_final_diameter, _page

MAX_BATCH_SIZE = 1024


def bilinear(image, x, y):
    """Preserve NumPy float32-coordinate minus intp -> float64 promotion."""
    h, w = image.shape[:2]
    x, y = x.clamp(0, w-1), y.clamp(0, h-1)
    x0, y0 = x.floor().to(torch.int64), y.floor().to(torch.int64)
    x1, y1 = (x0+1).clamp(max=w-1), (y0+1).clamp(max=h-1)
    fx = (x.to(torch.float64)-x0)[..., None]
    fy = (y.to(torch.float64)-y0)[..., None]
    return ((image[y0, x0]*(1-fx)+image[y0, x1]*fx)*(1-fy)
            + (image[y1, x0]*(1-fx)+image[y1, x1]*fx)*fy)


def dilate_rgb(tiles):
    """Same old-known, first-direction wins semantics; keep binary alpha.

    Fixed H+W rounds avoid .all().item() GPU barriers. Once filled, all later
    masks are empty, so the result matches the original batch-wide early stop.
    Directional RGB fills use fixed-shape where, avoiding CUDA nonzero/dynamic
    allocation for each direction. Per-Surfel keep compaction is still dynamic.
    """
    known = tiles[..., 3] != 0
    directions = ((slice(None), slice(None, -1), slice(None), slice(1, None)),
                  (slice(None), slice(1, None), slice(None), slice(None, -1)),
                  (slice(None, -1), slice(None), slice(1, None), slice(None)),
                  (slice(1, None), slice(None), slice(None, -1), slice(None)))
    for _ in range(tiles.shape[1]+tiles.shape[2]):
        old = known.clone()
        for sy, sx, dy, dx in directions:
            source = (slice(None), sy, sx)
            destination = (slice(None), dy, dx)
            fill = old[source] & ~known[destination]
            destination_rgb = tiles[destination][..., :3]
            destination_rgb.copy_(torch.where(fill[..., None], tiles[source][..., :3], destination_rgb))
            known[destination] |= fill
    return tiles


class FrameBaker:
    """One decoded canonical frame; exactly one RGB/depth upload per instance."""

    def __init__(self, frame, image, depth, *, device='cpu', radius=2., tile_size=8,
                 batch_limit=MAX_BATCH_SIZE):
        if not 1 <= operator.index(batch_limit) <= MAX_BATCH_SIZE:
            raise ValueError('batch_limit must be within [1, 1024]')
        if operator.index(tile_size) != 8 or not np.isfinite(radius) or radius <= 0:
            raise ValueError('Experiment requires tile_size=8 and positive finite radius')
        self.frame, self.radius, self.tile_size = frame, radius, tile_size
        self.batch_limit = batch_limit
        self.device = torch.device(device)
        self.depth = np.asarray(depth, dtype=np.float32)
        self.constants = _frame_constants(frame, tile_size)
        e, k, inv_k, affine, x, y = self.constants
        self.disk = torch.from_numpy(x*x+y*y <= 1).to(self.device)
        self.image_tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).to(self.device)
        self.depth_tensor = torch.from_numpy(np.ascontiguousarray(self.depth)).to(self.device)
        self.counts = dict(rgb_uploads=1, depth_uploads=1, batches=0, input_points=0,
                           baked_points=0, frame_setup_includes_transfers=True)

    @torch.inference_mode()
    def bake(self, p, u, v, scales):
        if len(p) > self.batch_limit:
            raise ValueError(f'Batch exceeds hard limit {self.batch_limit}')
        self.counts['batches'] += 1
        self.counts['input_points'] += len(p)
        p, u, v, scales = (np.asarray(a, dtype=np.float32) for a in (p, u, v, scales))
        e, k, inv_k, affine, x, y = self.constants
        with np.errstate(over='ignore', invalid='ignore'):
            center = p @ e[:3, :3].T + e[:3, 3]
            projected = center @ k.T
        visible = (np.isfinite(center).all(axis=1) & (center[:, 2] > 0)
                   & np.isfinite(projected).all(axis=1) & (projected[:, 2] > 0))
        p, u, v, scales = (a[visible] for a in (p, u, v, scales))
        center, projected = center[visible], projected[visible]
        if not len(p):
            return np.empty((0, 4, 3), dtype=np.float32), np.empty((0, self.tile_size+2, self.tile_size+2, 4), dtype=np.uint8)
        native = np.maximum(1e-6, center[:, 2:3]*np.linalg.norm(inv_k[:, :2], axis=0))
        bu, bv = _bound_footprint(u, v, native)
        pixel = np.floor(projected[:, :2]/projected[:, 2:3]+.5).astype(np.intp)
        depth_step = np.stack((u @ e[2, :3], v @ e[2, :3]), axis=1)
        h, w = self.depth.shape
        bounds = np.tile([-1e6, 1e6, -1e6, 1e6], (len(p), 1)).astype(np.float32)
        for column, (dx, dy, axis, sign) in enumerate(((-1, 0, 0, -1), (1, 0, 0, 1), (0, -1, 1, -1), (0, 1, 1, 1))):
            xx, yy = pixel[:, 0]+dx, pixel[:, 1]+dy
            d = self.depth[np.clip(yy, 0, h-1), np.clip(xx, 0, w-1)]
            continuous = ((xx >= 0) & (xx < w) & (yy >= 0) & (yy < h) & (d > 0)
                          & (np.abs(d-center[:, 2]-sign*depth_step[:, axis]) <= .001+.5*native[:, axis]))
            bounds[:, column] = np.where(continuous, bounds[:, column], pixel[:, axis]+sign*.5)
        bu, bv = bu*(self.radius*scales[:, :1]), bv*(self.radius*scales[:, 1:])
        bu, bv = _limit_final_diameter(bu, bv, native)
        # Identical source_world expression/order; no Torch projection matmul.
        world = p[:, None, None]+x[None, ..., None]*bu[:, None, None]+y[None, ..., None]*bv[:, None, None]
        camera = world @ e[:3, :3].T+e[:3, 3]
        projection = camera @ k.T
        processed = np.divide(projection[..., :2], projection[..., 2:3], out=np.zeros_like(projection[..., :2]), where=projection[..., 2:3] != 0)
        source = processed @ affine[:2, :2].T+affine[:2, 2]
        area = np.linalg.norm(np.cross(bu, bv), axis=1) > 0
        corners = np.stack((p-bu-bv, p+bu-bv, p+bu+bv, p-bu+bv), axis=1)

        def upload(value):
            return torch.from_numpy(np.ascontiguousarray(value)).to(self.device)

        xy, camera_z = upload(processed), upload(camera[..., 2])
        bounds_t, source_t, area_t = upload(bounds), upload(source), upload(area)
        px, py = xy[..., 0], xy[..., 1]
        valid = (self.disk[None] & (camera_z > 0)
                 & (px >= 0) & (px <= w-1) & (py >= 0) & (py <= h-1)
                 & (px >= bounds_t[:, None, None, 0]) & (px <= bounds_t[:, None, None, 1])
                 & (py >= bounds_t[:, None, None, 2]) & (py <= bounds_t[:, None, None, 3]))
        ix, iy = (px+.5).floor().to(torch.int64), (py+.5).floor().to(torch.int64)
        sampled = self.depth_tensor[iy.clamp(0, h-1), ix.clamp(0, w-1)]
        valid &= (sampled > 0) & ((sampled-camera_z).abs() <= .015*sampled+.001)
        sw, sh = self.frame['source_size']
        valid &= ((source_t[..., 0] >= 0) & (source_t[..., 0] <= sw-1)
                  & (source_t[..., 1] >= 0) & (source_t[..., 1] <= sh-1))
        valid &= area_t[:, None, None]
        keep = valid.any(dim=2).any(dim=1)
        valid = valid[keep]
        # Keep original NumPy f32 half-pixel mapping before the f64 interpolation.
        tw, th = self.frame['texture_size']
        tx = upload((source[..., 0]+.5)/sw*tw-.5)[keep]
        ty = upload((source[..., 1]+.5)/sh*th-.5)[keep]
        rgb = bilinear(self.image_tensor, tx, ty)
        tiles = torch.zeros((rgb.shape[0], self.tile_size+2, self.tile_size+2, 4), dtype=torch.uint8, device=self.device)
        tiles[:, 1:-1, 1:-1, :3] = rgb.round().to(torch.uint8)
        tiles[:, 1:-1, 1:-1, 3] = valid.to(torch.uint8)*255
        tiles = dilate_rgb(tiles)
        # Downloads complete all preceding frame/batch work before returning.
        keep_cpu, tiles_cpu = keep.cpu().numpy(), tiles.cpu().numpy()
        self.counts['baked_points'] += int(keep_cpu.sum())
        return corners[keep_cpu], tiles_cpu


def bake_surfel_pages(*, positions, u, v, scales, frame_indices, depths, frames,
                      read_asset, radius=2., tile_size=8, atlas_size=2048,
                      batch_size=MAX_BATCH_SIZE, image_format='png',
                      projection_mode='source_world', device='cpu', metrics=None):
    """Experimental page producer compatible with the existing CPU GLB writer."""
    tile_size, atlas_size, batch_size = map(operator.index, (tile_size, atlas_size, batch_size))
    if projection_mode != 'source_world':
        raise ValueError('This experiment only supports source_world')
    if not 1 <= operator.index(batch_size) <= MAX_BATCH_SIZE:
        raise ValueError('batch_size must be within [1, 1024]')
    if image_format not in ('png', 'rgba') or tile_size != 8 or not tile_size+2 <= atlas_size <= 2048:
        raise ValueError('Invalid image_format or atlas_size')
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
    columns = atlas_size//(tile_size+2)
    capacity = columns*columns
    page_corners = np.empty((capacity, 4, 3), dtype=np.float32)
    page_tiles = np.empty((capacity, tile_size+2, tile_size+2, 4), dtype=np.uint8)
    filled = 0
    for frame_id in np.unique(frame_indices):
        tick = time.perf_counter()
        frame = frames[int(frame_id)]
        with Image.open(BytesIO(read_asset(frame['texture']))) as image:
            rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
        if tuple(frame['texture_size']) != (rgb.shape[1], rgb.shape[0]):
            raise ValueError('Source texture dimensions differ from metadata')
        width = max(f['texture_size'][0] for f in frames)
        height = max(f['texture_size'][1] for f in frames)
        padded = np.zeros((height, width, 3), dtype=np.float32)
        padded[:rgb.shape[0], :rgb.shape[1]] = rgb
        baker = FrameBaker(frame, padded, np.asarray(depths[int(frame_id)], dtype=np.float32),
                           device=device, radius=radius, tile_size=tile_size, batch_limit=batch_size)
        row = dict(frame_id=int(frame_id), setup_submission_wall_s=time.perf_counter()-tick)
        if metrics is not None:
            metrics.setdefault('frames', []).append(row)
        slots = np.flatnonzero(frame_indices == frame_id)
        for start in range(0, len(slots), batch_size):
            batch = slots[start:start+batch_size]
            corners, tiles = baker.bake(positions[batch], u[batch], v[batch], scales[batch])
            offset = 0
            while offset < len(corners):
                take = min(capacity-filled, len(corners)-offset)
                page_corners[filled:filled+take] = corners[offset:offset+take]
                page_tiles[filled:filled+take] = tiles[offset:offset+take]
                filled += take
                offset += take
                if filled == capacity:
                    yield _page(page_corners, page_tiles, filled, columns, tile_size, image_format)
                    filled = 0
        row.update(baker.counts)
        del baker
    if filled:
        yield _page(page_corners, page_tiles, filled, columns, tile_size, image_format)
