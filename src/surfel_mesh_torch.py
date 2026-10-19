"""Explicit bounded Torch sampler; CPU projection and paging are shared.

No automatic device selection or fallback. This module is lazily imported only
for explicit CUDA baking; its CPU device is a numerical testing interface.
"""
from __future__ import annotations

import operator

import numpy as np
import torch

from src.surfel_mesh import _frame_constants, _prepare_batch

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
                 batch_limit=MAX_BATCH_SIZE, fixed_shape=False):
        if not 1 <= operator.index(batch_limit) <= MAX_BATCH_SIZE:
            raise ValueError('batch_limit must be within [1, 1024]')
        if operator.index(tile_size) != 8 or not np.isfinite(radius) or radius <= 0:
            raise ValueError('Torch bake requires tile_size=8 and positive finite radius')
        self.frame, self.radius, self.tile_size = frame, radius, tile_size
        self.batch_limit = batch_limit
        self.fixed_shape = fixed_shape
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
        prepared = _prepare_batch(p, u, v, scales, self.depth, self.frame,
                                  self.radius, self.tile_size, constants=self.constants)
        if prepared is None:
            return np.empty((0, 4, 3), dtype=np.float32), np.empty((0, self.tile_size+2, self.tile_size+2, 4), dtype=np.uint8)
        processed, camera_z, bounds, source, area, corners = (
            prepared[key] for key in ('processed', 'camera_z', 'bounds', 'source', 'area', 'corners'))
        h, w = self.depth.shape

        def upload(value):
            return torch.from_numpy(np.ascontiguousarray(value)).to(self.device)

        xy, camera_z = upload(processed), upload(camera_z)
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
        if not self.fixed_shape:
            valid = valid[keep]
        # Keep original NumPy f32 half-pixel mapping before the f64 interpolation.
        tw, th = self.frame['texture_size']
        tx = upload((source[..., 0]+.5)/sw*tw-.5)
        ty = upload((source[..., 1]+.5)/sh*th-.5)
        if self.fixed_shape:
            # Invalid, discarded surfels must never generate unsafe gather addresses.
            tx = torch.where(keep[:, None, None], tx, 0.)
            ty = torch.where(keep[:, None, None], ty, 0.)
        else:
            tx, ty = tx[keep], ty[keep]
        rgb = bilinear(self.image_tensor, tx, ty)
        tiles = torch.zeros((rgb.shape[0], self.tile_size+2, self.tile_size+2, 4), dtype=torch.uint8, device=self.device)
        tiles[:, 1:-1, 1:-1, :3] = rgb.round().to(torch.uint8)
        tiles[:, 1:-1, 1:-1, 3] = valid.to(torch.uint8)*255
        tiles = dilate_rgb(tiles)
        # Downloads complete all preceding frame/batch work before returning.
        keep_cpu, tiles_cpu = keep.cpu().numpy(), tiles.cpu().numpy()
        self.counts['baked_points'] += int(keep_cpu.sum())
        return corners[keep_cpu], tiles_cpu[keep_cpu] if self.fixed_shape else tiles_cpu
