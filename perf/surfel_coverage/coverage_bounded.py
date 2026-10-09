"""One conservative repair: protect baseline tiles and exchange few observations."""
import time

import numpy as np

from perf.surfel_coverage.coverage import _normal_bins, _source_quality


def _features(data, indices, with_quality=False):
    _, h, w = data['valid'].shape
    frames = indices // (h*w)
    keys = np.empty((len(indices), 5), dtype=np.int64)
    quality = np.zeros(len(indices), dtype=np.float32)
    ring = np.zeros(len(indices), dtype=np.uint8)
    for frame in np.unique(frames):
        loc = np.flatnonzero(frames == frame)
        y, x = np.divmod(indices[loc] % (h*w), w)
        points = data['points_grid'][frame, y, x].astype(np.float64)
        u, v = (data[k][frame, y, x].astype(np.float64) for k in ('u_grid', 'v_grid'))
        normals = np.cross(u, v)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        pose = data['E'][frame]
        center = -pose[:3, :3].T @ pose[:3, 3]
        normals[np.einsum('ij,ij->i', normals, center-points) < 0] *= -1
        code, bin_normals = _normal_bins(normals)
        keys[loc, :3] = np.floor(points / (4*float(data['spacing']))).astype(np.int64)
        keys[loc, 3] = code
        keys[loc, 4] = np.floor(np.einsum('ij,ij->i', points, bin_normals) / float(data['spacing'])).astype(np.int64)
        if with_quality:
            quality[loc], ring[loc], _ = _source_quality(points, u, v, normals, y, x, frame, data, float(data['spacing']))
    return keys, quality, ring


def _removable(data, baseline):
    frames, h, w = data['valid'].shape
    frame, remainder = np.divmod(baseline, h*w)
    y, x = np.divmod(remainder, w)
    base_mask = np.zeros(frames*h*w, dtype=bool)
    base_mask[baseline] = True
    safe = (x >= 4) & (x < w-4) & (y >= 4) & (y < h-4)
    z = data['depth'].ravel()[baseline]
    u = data['u_grid'].reshape(-1, 3)[baseline]
    v = data['v_grid'].reshape(-1, 3)[baseline]
    camera_z_axis = data['E'][frame, 2, :3]
    zu = np.einsum('ij,ij->i', u, camera_z_axis)
    zv = np.einsum('ij,ij->i', v, camera_z_axis)
    for dy, dx in ((-4, 0), (4, 0), (0, -4), (0, 4)):
        ny, nx = y+dy, x+dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        neighbor = frame*(h*w) + ny.clip(0, h-1)*w + nx.clip(0, w-1)
        predicted = z + dx*zu + dy*zv
        measured = data['depth'].ravel()[neighbor]
        safe &= inside & base_mask[neighbor] & (np.abs(predicted-measured) <= np.maximum(2*float(data['spacing']), .005*measured))
    # Do not free anchors whose own unchanged disk already straddles support.
    _, _, ring = _features(data, baseline, with_quality=True)
    safe &= ring == 8
    tx, ty = (w+15)//16, (h+15)//16
    tile = (frame*ty+y//16)*tx+x//16
    counts = np.bincount(tile, minlength=frames*tx*ty)
    quota = counts//16
    eligible = np.flatnonzero(safe)
    center_distance = (x[eligible] % 16-8)**2 + (y[eligible] % 16-8)**2
    order = np.lexsort((baseline[eligible], center_distance, tile[eligible]))
    eligible = eligible[order]
    groups = tile[eligible]
    starts = np.r_[0, np.flatnonzero(np.diff(groups))+1] if len(groups) else np.empty(0, dtype=int)
    rank = np.arange(len(groups))-np.repeat(starts, np.diff(np.r_[starts, len(groups)]))
    return baseline[eligible[rank < quota[groups]]], tile, counts


def _spatial_rounds(points, quality, count, spacing):
    """Represent macro cells first; ties across cells never use global quality."""
    macro = np.floor(points/(16*spacing)).astype(np.int64)
    order = np.lexsort((-quality, macro[:, 2], macro[:, 1], macro[:, 0]))
    ordered = macro[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(ordered, axis=0), axis=1))+1]
    counts = np.diff(np.r_[starts, len(order)])
    selected = []
    remaining, layer = count, 0
    while remaining:
        active = np.flatnonzero(counts > layer)
        if remaining < len(active):
            # Systematic positions span the complete sorted cell list.
            active = active[np.floor((np.arange(remaining)+.5)*len(active)/remaining).astype(int)]
        selected.extend(order[starts[active]+layer].tolist())
        remaining -= len(active)
        layer += 1
    return np.asarray(selected, dtype=np.int64), len(starts)


def _disc_area(data, indices):
    u, v = (data[k].reshape(-1, 3)[indices].astype(np.float64) for k in ('u_grid', 'v_grid'))
    return float((4*np.sin(np.pi/4)*(1.05*4)**2)*np.linalg.norm(np.cross(u, v), axis=1).sum())


def select(data, budget):
    started = time.monotonic()
    baseline = np.asarray(data['baseline_indices'], dtype=np.int64)
    if budget != len(baseline) or len(np.unique(baseline)) != budget:
        raise ValueError('Bounded repair requires the unchanged full baseline budget')
    if not data['valid'].ravel()[baseline].all():
        raise ValueError('Baseline must contain valid original observations')
    _, h, w = data['valid'].shape
    removable, tiles, tile_counts = _removable(data, baseline)
    y, x = np.indices((h, w))
    pool = np.flatnonzero(data['valid'] & (y % 2 == 0) & (x % 2 == 0))
    pool = np.setdiff1d(pool, baseline, assume_unique=True)
    # Novelty is relative to the entire original baseline, not holes just created.
    base_keys, _, _ = _features(data, baseline)
    pool_keys, quality, _ = _features(data, pool, with_quality=True)
    _, inverse = np.unique(np.concatenate((base_keys, pool_keys)), axis=0, return_inverse=True)
    represented = np.unique(inverse[:budget])
    novel = ~np.isin(inverse[budget:], represented)
    pool, quality, codes = pool[novel], quality[novel], inverse[budget:][novel]
    order = np.lexsort((-quality, codes))
    first = np.r_[True, np.diff(codes[order]) != 0] if len(order) else np.empty(0, dtype=bool)
    pool, quality = pool[order[first]], quality[order[first]]
    exchange = min(len(removable), len(pool))
    if exchange:
        taken = np.floor((np.arange(exchange)+.5)*len(removable)/exchange).astype(int)
        removed = np.sort(removable[taken])
        picks, macro_count = _spatial_rounds(data['points_grid'].reshape(-1, 3)[pool], quality, exchange, float(data['spacing']))
        added = np.sort(pool[picks])
    else:
        removed = added = np.empty(0, dtype=np.int64)
        macro_count = 0
    indices = np.sort(np.r_[np.setdiff1d(baseline, removed, assume_unique=True), added])
    if len(indices) != budget or len(np.unique(indices)) != budget or len(removed)*16 > budget:
        raise RuntimeError('Bounded exchange violated the exact budget or retention floor')
    removed_positions = np.searchsorted(baseline, removed)
    removed_by_tile = np.bincount(tiles[removed_positions], minlength=len(tile_counts))
    if np.any(removed_by_tile > tile_counts//16):
        raise RuntimeError('Bounded exchange violated its per-tile retention floor')
    per_frame_removed = np.bincount(removed//(h*w), minlength=len(data['depth']))
    per_frame_added = np.bincount(added//(h*w), minlength=len(data['depth']))
    stats = dict(method='baseline-protected-spatial-exchange', selected_observations=budget,
        retained_baseline_observations=budget-exchange, exchanged_observations=exchange,
        exchanged_fraction=exchange/budget, minimum_retained_fraction=15/16,
        candidate_pool_stride=2, novel_candidate_cells=len(pool), candidate_macro_cells=macro_count,
        removed_original_indices=removed.tolist(), added_original_indices=added.tolist(),
        removed_disc_triangle_area_m2=_disc_area(data, removed), added_disc_triangle_area_m2=_disc_area(data, added),
        per_frame_removed=per_frame_removed.tolist(), per_frame_added=per_frame_added.tolist(),
        scale=4, selection_seconds=time.monotonic()-started,
        limitations=['Coverage is protected by source-tile anchors, not a proof of unchanged rendered visibility.',
                    'Novelty is a spatial/normal/layer proxy; candidates are original valid stride-2 observations.',
                    'Quality ranks representatives only inside cells; cell allocation is spatial systematic round-robin.',
                    'World triangle area can change despite fixed scale and point count.'])
    return dict(indices=indices, scales=np.full((budget, 2), 4., dtype=np.float32), stats=stats)
