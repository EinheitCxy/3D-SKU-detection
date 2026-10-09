"""Fixed-budget edge-adaptive sampling for the isolated coverage experiment.

Only original valid pixels are selected. Small image tiles retain coverage while
bounded boundary weights redistribute their quotas and local sampling density.
The coordinator retains the original source frame and strict texture gate.
"""

from __future__ import annotations

import operator
import time

import numpy as np
from scipy.ndimage import binary_dilation


TILE_SIZE = 16
EDGE_WEIGHT = 4
HALO_WEIGHT = 2
HALO_RADIUS = 2
MIN_SCALE = 0.5
MAX_SCALE = 8.0
DISC_RADIUS = 1.05
BOUNDARY_MARGIN = 0.95


def _depth_breaks(depth, valid, spacing, axis):
    """Find broken links; a consistent local depth slope is not a depth jump."""
    depth = np.moveaxis(depth, axis, -1)
    valid = np.moveaxis(valid, axis, -1)
    pair_valid = valid[..., :-1] & valid[..., 1:]
    difference = np.abs(
        np.subtract(
            depth[..., 1:], depth[..., :-1],
            out=np.zeros_like(depth[..., 1:]), where=pair_valid,
        )
    )
    neighbor_slope = np.full_like(difference, np.inf)
    neighbor_slope[..., 1:] = np.where(
        pair_valid[..., :-1], difference[..., :-1], np.inf
    )
    neighbor_slope[..., :-1] = np.minimum(
        neighbor_slope[..., :-1],
        np.where(pair_valid[..., 1:], difference[..., 1:], np.inf),
    )
    neighbor_slope[~np.isfinite(neighbor_slope)] = 0
    tolerance = np.maximum(
        2 * spacing, 0.005 * np.minimum(depth[..., :-1], depth[..., 1:])
    )
    jump = pair_valid & (difference > tolerance + 3 * neighbor_slope)
    return np.moveaxis(~pair_valid | jump, -1, axis), int(jump.sum())


def _frame_features(depth, valid, spacing):
    blocked_x, jump_x = _depth_breaks(depth, valid, spacing, 1)
    blocked_y, jump_y = _depth_breaks(depth, valid, spacing, 0)
    edge = np.zeros_like(valid)
    edge[:, :-1] |= blocked_x
    edge[:, 1:] |= blocked_x
    edge[:-1] |= blocked_y
    edge[1:] |= blocked_y
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    edge &= valid
    halo = binary_dilation(edge, iterations=HALO_RADIUS) & valid
    weights = np.where(edge, EDGE_WEIGHT, np.where(halo, HALO_WEIGHT, 1))
    weights = np.where(valid, weights, 0).astype(np.uint8)
    return weights, blocked_x, blocked_y, jump_x + jump_y


def _tile_sums(values):
    frames, height, width = values.shape
    pad_h, pad_w = (-height) % TILE_SIZE, (-width) % TILE_SIZE
    padded = np.pad(values, ((0, 0), (0, pad_h), (0, pad_w)))
    return padded.reshape(
        frames, (height + pad_h) // TILE_SIZE, TILE_SIZE,
        (width + pad_w) // TILE_SIZE, TILE_SIZE,
    ).sum(axis=(2, 4), dtype=np.int64)


def _apportion(capacities, masses, budget):
    """Capped largest-remainder quotas, with one per occupied tile if feasible."""
    capacities = capacities.ravel()
    masses = masses.ravel().astype(np.float64)
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


def _inclusion_probabilities(weights, quota):
    """Solve sum(min(1, rate * weight)) == quota, including dense budgets."""
    weights = weights.astype(np.float64)
    if quota == len(weights):
        return np.ones_like(weights)
    probability = np.zeros_like(weights)
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


def _axis_caps(blocked, shape, axis):
    """Axiswise distance to broken pixel interfaces and the image boundary.

    This restrains projected extent; it is not a replacement for the texture
    visibility test, especially at corners and for nonplanar tangent patches.
    """
    blocked = np.moveaxis(blocked, axis, -1)
    moved_shape = (*shape[:axis], *shape[axis + 1:], shape[axis])
    width = moved_shape[-1]
    coordinate = np.arange(width, dtype=np.float32)
    left = np.zeros(moved_shape, dtype=np.float32)
    left[..., 1:] = np.where(blocked, coordinate[1:] - 0.5, 0)
    left = np.maximum.accumulate(left, axis=-1)
    right = np.full(moved_shape, width - 1, dtype=np.float32)
    right[..., :-1] = np.where(blocked, coordinate[:-1] + 0.5, width - 1)
    right = np.minimum.accumulate(right[..., ::-1], axis=-1)[..., ::-1]
    raw = np.minimum(coordinate - left, right - coordinate) * (BOUNDARY_MARGIN / DISC_RADIUS)
    return np.moveaxis(np.clip(raw, MIN_SCALE, MAX_SCALE), -1, axis), np.moveaxis(raw < MIN_SCALE, -1, axis)


def _quantiles(values):
    if not len(values):
        return None
    return dict(zip(
        ("min", "p05", "p50", "p95", "max"),
        np.quantile(values, [0, 0.05, 0.5, 0.95, 1]).tolist(),
    ))


def select(data, budget):
    """Return exact unique valid flat indices, U/V scales, and JSON-safe stats.

    ``depth``, ``valid`` and ``spacing`` are the only dense fields read. Indices
    preserve the original C-order frame/xy/source mapping for all other fields.
    Selection is deterministic; neither input arrays nor observation depths are
    modified. The accepted budget range is 1 through the valid pixel count.
    """
    started = time.monotonic()
    if isinstance(budget, (bool, np.bool_)):
        raise ValueError("budget must be a positive integer")
    try:
        budget = operator.index(budget)
    except TypeError as error:
        raise ValueError("budget must be a positive integer") from error
    valid = np.asarray(data["valid"], dtype=bool)
    depth = np.asarray(data["depth"], dtype=np.float64)
    spacing = float(data["spacing"])
    if valid.ndim != 3 or depth.shape != valid.shape or not all(valid.shape):
        raise ValueError("depth and valid must share nonempty (frame, height, width) grids")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    if np.any(valid & (~np.isfinite(depth) | (depth <= 0))):
        raise ValueError("valid pixels must have finite positive original depth")
    valid_count = int(valid.sum())
    if not 1 <= budget <= valid_count:
        raise ValueError("budget must be between one and the valid pixel count")

    frames, height, width = valid.shape
    weights = np.empty(valid.shape, dtype=np.uint8)
    broken_x, broken_y, depth_jumps = [], [], []
    for frame in range(frames):
        weights[frame], bx, by, jumps = _frame_features(depth[frame], valid[frame], spacing)
        broken_x.append(bx)
        broken_y.append(by)
        depth_jumps.append(jumps)
    capacities, masses = _tile_sums(valid), _tile_sums(weights)
    quotas = _apportion(capacities, masses, budget).reshape(capacities.shape)

    # Interleaved pixel bits keep successive strata spatially local. This is
    # precomputed once for the fixed tile, rather than sorting all scene pixels.
    tile_y, tile_x = np.indices((TILE_SIZE, TILE_SIZE))
    morton = np.zeros_like(tile_x)
    for bit in range(4):
        morton |= ((tile_x >> bit) & 1) << (2 * bit)
        morton |= ((tile_y >> bit) & 1) << (2 * bit + 1)
    order = np.argsort(morton.ravel(), kind="stable")
    tile_y, tile_x = tile_y.ravel()[order], tile_x.ravel()[order]

    all_indices, all_scales = [], []
    floor_count = 0
    for frame in range(frames):
        cap_x, floor_x = _axis_caps(broken_x[frame], (height, width), 1)
        cap_y, floor_y = _axis_caps(broken_y[frame], (height, width), 0)
        for row, column in np.argwhere(quotas[frame] > 0):
            quota = int(quotas[frame, row, column])
            yy, xx = row * TILE_SIZE + tile_y, column * TILE_SIZE + tile_x
            inside = (yy < height) & (xx < width)
            yy, xx = yy[inside], xx[inside]
            keep = valid[frame, yy, xx]
            yy, xx = yy[keep], xx[keep]
            probability = _inclusion_probabilities(weights[frame, yy, xx], quota)
            # Probabilities are at most one, so each integer-spaced threshold
            # crosses a different pixel, even if the requested budget is dense.
            positions = np.searchsorted(
                np.cumsum(probability), np.arange(quota) + 0.5, side="right"
            )
            yy, xx = yy[positions], xx[positions]
            nominal = np.clip(1 / np.sqrt(probability[positions]), MIN_SCALE, MAX_SCALE)
            scales = np.column_stack((np.minimum(nominal, cap_x[yy, xx]),
                                      np.minimum(nominal, cap_y[yy, xx])))
            all_indices.append(frame * height * width + yy * width + xx)
            all_scales.append(scales.astype(np.float32))
            floor_count += int((floor_x[yy, xx] | floor_y[yy, xx]).sum())

    indices = np.concatenate(all_indices).astype(np.int64)
    scales = np.concatenate(all_scales)
    order = np.argsort(indices, kind="stable")
    indices, scales = indices[order], scales[order]
    if len(indices) != budget or np.any(indices[1:] == indices[:-1]):
        raise RuntimeError("adaptive allocation failed its exact unique budget")
    selected_weights = weights.ravel()[indices]
    edge = selected_weights == EDGE_WEIGHT
    interior = selected_weights == 1
    edge_pixels = int(np.count_nonzero(weights == EDGE_WEIGHT))
    halo_pixels = int(np.count_nonzero(weights == HALO_WEIGHT))
    interior_pixels = valid_count - edge_pixels - halo_pixels
    stats = {
        "method": "edge-adaptive-tile-systematic",
        "input_valid_pixels": valid_count,
        "point_budget": budget,
        "selected_points": len(indices),
        "tile_size_pixels": TILE_SIZE,
        "nonempty_tiles": int(np.count_nonzero(capacities)),
        "covered_tiles": int(np.count_nonzero(quotas)),
        "minimum_nonempty_tile_quota": int(quotas[capacities > 0].min()),
        "per_frame_quota": quotas.sum(axis=(1, 2)).tolist(),
        "per_frame_depth_jump_links": depth_jumps,
        "edge_valid_pixels": edge_pixels,
        "halo_only_valid_pixels": halo_pixels,
        "interior_valid_pixels": interior_pixels,
        "edge_selected_points": int(edge.sum()),
        "halo_only_selected_points": int(np.count_nonzero(selected_weights == HALO_WEIGHT)),
        "interior_selected_points": int(interior.sum()),
        "edge_selected_fraction": float(edge.mean()),
        "edge_sampling_rate": float(edge.sum() / edge_pixels) if edge_pixels else 0.0,
        "interior_sampling_rate": float(interior.sum() / interior_pixels) if interior_pixels else 0.0,
        "scale_u": _quantiles(scales[:, 0]),
        "scale_v": _quantiles(scales[:, 1]),
        "edge_scale_u": _quantiles(scales[edge, 0]),
        "edge_scale_v": _quantiles(scales[edge, 1]),
        "interior_scale_u": _quantiles(scales[interior, 0]),
        "interior_scale_v": _quantiles(scales[interior, 1]),
        "boundary_scale_floor_points": floor_count,
        "boundary_scale_floor_fraction": floor_count / budget,
        "parameters": {
            "weights": {"interior": 1, "halo": HALO_WEIGHT, "edge": EDGE_WEIGHT},
            "halo_manhattan_radius_pixels": HALO_RADIUS,
            "depth_jump_absolute_spacing": 2,
            "depth_jump_relative": 0.005,
            "depth_jump_neighbor_slope_multiplier": 3,
            "scale_bounds": [MIN_SCALE, MAX_SCALE],
            "disc_radius": DISC_RADIUS,
            "boundary_margin": BOUNDARY_MARGIN,
        },
        "coverage_proxy": "occupied 16x16 image tiles; not measured 3D or raycast coverage",
        "texture_gate": "unchanged original-source four-sample gate, applied by coordinator",
        "selection_seconds": time.monotonic() - started,
    }
    return {"indices": indices, "scales": scales, "stats": stats}
