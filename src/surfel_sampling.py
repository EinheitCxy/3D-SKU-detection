"""Fixed-budget depth-edge sampling on original camera-grid observations.

Geometry support is distinct from the candidate pool: voxel thinning must not
turn an otherwise continuous surface into a grid of artificial depth holes.
"""
from __future__ import annotations

import operator

import numpy as np
from scipy.ndimage import binary_dilation

TILE_SIZE = 16
MIN_SCALE = 0.5
MAX_SCALE = 8.0


def _depth_breaks(depth, valid, spacing, axis):
    depth = np.moveaxis(depth, axis, -1)
    valid = np.moveaxis(valid, axis, -1)
    pair_valid = valid[..., :-1] & valid[..., 1:]
    difference = np.abs(np.subtract(
        depth[..., 1:], depth[..., :-1],
        out=np.zeros_like(depth[..., 1:]), where=pair_valid,
    ))
    neighbor_slope = np.full_like(difference, np.inf)
    neighbor_slope[..., 1:] = np.where(pair_valid[..., :-1], difference[..., :-1], np.inf)
    neighbor_slope[..., :-1] = np.minimum(
        neighbor_slope[..., :-1],
        np.where(pair_valid[..., 1:], difference[..., 1:], np.inf),
    )
    neighbor_slope[~np.isfinite(neighbor_slope)] = 0
    tolerance = np.maximum(2 * spacing, 0.005 * np.minimum(depth[..., :-1], depth[..., 1:]))
    return np.moveaxis(~pair_valid | (difference > tolerance + 3 * neighbor_slope), -1, axis)


def _axis_caps(blocked, shape, axis):
    blocked = np.moveaxis(blocked, axis, -1)
    moved_shape = (*shape[:axis], *shape[axis + 1:], shape[axis])
    coordinate = np.arange(moved_shape[-1], dtype=np.float32)
    left = np.zeros(moved_shape, dtype=np.float32)
    left[..., 1:] = np.where(blocked, coordinate[1:] - 0.5, 0)
    left = np.maximum.accumulate(left, axis=-1)
    right = np.full(moved_shape, moved_shape[-1] - 1, dtype=np.float32)
    right[..., :-1] = np.where(blocked, coordinate[:-1] + 0.5, moved_shape[-1] - 1)
    right = np.minimum.accumulate(right[..., ::-1], axis=-1)[..., ::-1]
    cap = np.minimum(coordinate - left, right - coordinate) * (0.95 / 1.05)
    return np.moveaxis(np.clip(cap, MIN_SCALE, MAX_SCALE), -1, axis)


def _apportion(capacities, masses, budget):
    occupied = capacities > 0
    quotas = occupied.astype(np.int64) if budget >= occupied.sum() else np.zeros_like(capacities)
    remaining = budget - int(quotas.sum())
    while remaining:
        active = np.flatnonzero(quotas < capacities)
        share = remaining * masses[active] / masses[active].sum()
        available = capacities[active] - quotas[active]
        saturated = share >= available
        if saturated.any():
            full = active[saturated]
            remaining -= int((capacities[full] - quotas[full]).sum())
            quotas[full] = capacities[full]
            continue
        addition = np.floor(share).astype(np.int64)
        quotas[active] += addition
        remaining -= int(addition.sum())
        if remaining:
            order = np.argsort(-(share - addition), kind="stable")
            quotas[active[order[:remaining]]] += 1
        break
    return quotas


def _probabilities(weights, quota):
    probability = np.zeros(len(weights), dtype=np.float64)
    active = np.ones(len(weights), dtype=bool)
    remaining = quota
    while remaining:
        rate = remaining / weights[active].sum()
        saturated = active & (rate * weights >= 1)
        if not saturated.any():
            probability[active] = rate * weights[active]
            break
        probability[saturated] = 1
        active[saturated] = False
        remaining -= int(saturated.sum())
    return probability


class EdgeAdaptiveSampler:
    """Compute dense features once, then sample independent budget groups.

    ``regions`` contains global-ID group numbers, with a separate background
    group. Region boundaries also cap discs to reduce expansion across a
    neighboring object's source mask; the minimum scale is still 0.5.
    """

    def __init__(self, depth, valid, spacing, regions=None):
        depth = np.asarray(depth)
        valid = np.asarray(valid, dtype=bool)
        if depth.ndim != 3 or valid.shape != depth.shape or not all(depth.shape):
            raise ValueError("depth and valid must share a nonempty camera grid")
        if not np.isfinite(spacing) or spacing < 0:
            raise ValueError("native spacing must be finite and nonnegative")
        if np.any(valid & (~np.isfinite(depth) | (depth <= 0))):
            raise ValueError("valid observations require finite positive camera depth")
        if regions is not None and np.shape(regions) != depth.shape:
            raise ValueError("region labels must align with the camera grid")
        self.shape = depth.shape
        self.valid = valid.ravel()
        self.weights = np.empty(depth.shape, dtype=np.uint8)
        self.caps = np.empty((*depth.shape, 2), dtype=np.float32)
        for frame in range(len(depth)):
            bx = _depth_breaks(depth[frame], valid[frame], spacing, 1)
            by = _depth_breaks(depth[frame], valid[frame], spacing, 0)
            if regions is not None:
                bx |= regions[frame, :, 1:] != regions[frame, :, :-1]
                by |= regions[frame, 1:, :] != regions[frame, :-1, :]
            edge = np.zeros(depth.shape[1:], dtype=bool)
            edge[:, :-1] |= bx
            edge[:, 1:] |= bx
            edge[:-1] |= by
            edge[1:] |= by
            edge[[0, -1], :] = True
            edge[:, [0, -1]] = True
            edge &= valid[frame]
            halo = binary_dilation(edge, iterations=2) & valid[frame]
            self.weights[frame] = np.where(valid[frame], np.where(edge, 4, np.where(halo, 2, 1)), 0)
            self.caps[frame, ..., 0] = _axis_caps(bx, depth.shape[1:], 1)
            self.caps[frame, ..., 1] = _axis_caps(by, depth.shape[1:], 0)
        self.weights = self.weights.ravel()
        self.caps = self.caps.reshape(-1, 2)
        self.tiles_y = (depth.shape[1] + TILE_SIZE - 1) // TILE_SIZE
        self.tiles_x = (depth.shape[2] + TILE_SIZE - 1) // TILE_SIZE
        self.tile_count = len(depth) * self.tiles_y * self.tiles_x

    def _tile_coordinates(self, indices):
        frame, remainder = np.divmod(indices, self.shape[1] * self.shape[2])
        y, x = np.divmod(remainder, self.shape[2])
        tiles = (frame * self.tiles_y + y // TILE_SIZE) * self.tiles_x + x // TILE_SIZE
        morton = np.zeros(len(indices), dtype=np.int64)
        for bit in range(4):
            morton |= ((x >> bit) & 1) << (2 * bit)
            morton |= ((y >> bit) & 1) << (2 * bit + 1)
        return tiles, morton

    def select(self, indices, budget, *, support_indices=None):
        """Return positions within ``indices`` and their aligned U/V scales.

        Candidates must be unique original-grid indices. ``support_indices``
        is the same group's dense eligible support before voxel thinning; its
        tile density accounts for that earlier reduction as well as sampling.
        """
        if isinstance(budget, (bool, np.bool_)):
            raise ValueError("budget must be an integer within the candidate count")
        budget = operator.index(budget)
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not 0 <= budget <= len(indices):
            raise ValueError("budget must be within the candidate count")
        if budget == 0:
            return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.float32)
        if np.any(indices < 0) or np.any(indices >= self.valid.size) or not self.valid[indices].all():
            raise ValueError("candidate indices must refer to valid original observations")
        tiles, morton = self._tile_coordinates(indices)
        weights = self.weights[indices]
        capacities = np.bincount(tiles, minlength=self.tile_count)
        masses = np.bincount(tiles, weights=weights, minlength=self.tile_count)
        if support_indices is None:
            support_mass = masses
        else:
            support_indices = np.asarray(support_indices, dtype=np.int64)
            support_tiles, _ = self._tile_coordinates(support_indices)
            support_mass = np.bincount(support_tiles, weights=self.weights[support_indices], minlength=self.tile_count)
        quotas = _apportion(capacities, masses, budget)
        order = np.argsort(tiles * TILE_SIZE**2 + morton, kind="stable")
        offsets = np.concatenate(([0], np.cumsum(capacities)))
        chosen, scales = [], []
        for tile in np.flatnonzero(quotas):
            group = order[offsets[tile]:offsets[tile + 1]]
            quota = int(quotas[tile])
            probability = _probabilities(weights[group], quota)
            positions = np.searchsorted(np.cumsum(probability), np.arange(quota) + 0.5, side="right")
            picked = group[positions]
            # Account for voxel thinning without treating missing candidates as
            # geometric discontinuities. Caps still come from the dense grid.
            density = probability[positions] * masses[tile] / support_mass[tile]
            nominal = np.clip(1 / np.sqrt(density), MIN_SCALE, MAX_SCALE)
            chosen.append(picked)
            scales.append(np.minimum(nominal[:, None], self.caps[indices[picked]]))
        chosen = np.concatenate(chosen)
        scales = np.concatenate(scales).astype(np.float32)
        order = np.argsort(chosen, kind="stable")
        return chosen[order], scales[order]
