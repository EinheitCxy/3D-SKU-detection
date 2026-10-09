"""Conservative, fixed-grid source-pixel selection for portable Surfels."""

import numpy as np

from src.surfel_mesh import _bound_footprint


def _covered(groups, positions, u, v, scales, center, pixel, native, e, depth):
    """Check complete blocks in bounded batches; the first slot is top-left."""
    p = positions[groups]
    normals = np.cross(u[groups], v[groups])
    lengths = np.linalg.norm(normals, axis=-1)
    normals = np.divide(normals, lengths[..., None], out=np.zeros_like(normals),
                        where=lengths[..., None] > 0)
    cosine = np.einsum('bij,bkj->bik', normals, normals)
    z = center[groups, 2]
    valid = ((lengths > 0).all(axis=1) & (cosine >= .95).all(axis=(1, 2))
             & (np.ptp(z, axis=1) <= .01 * z.mean(axis=1))
             & (scales[groups] >= 1).all(axis=(1, 2)))
    # Reject cells whose source-depth discontinuities would clip their disks.
    h, w = depth.shape
    step = np.stack((u[groups] @ e[2, :3], v[groups] @ e[2, :3]), axis=-1)
    for dx, dy, axis, sign in ((0, 0, 0, 0), (-1, 0, 0, -1),
                                (1, 0, 0, 1), (0, -1, 1, -1), (0, 1, 1, 1)):
        x, y = pixel[groups, 0] + dx, pixel[groups, 1] + dy
        sampled = depth[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
        continuous = ((x >= 0) & (x < w) & (y >= 0) & (y < h)
                      & np.isfinite(sampled) & (sampled > 0)
                      & (np.abs(sampled - z - sign * step[..., axis])
                         <= .001 + .5 * native[groups, axis]))
        valid &= continuous.all(axis=1)
    representative = groups[:, 0]
    a, b = _bound_footprint(u[representative], v[representative], native[representative])
    a *= 2 * scales[representative, :1]
    b *= 2 * scales[representative, 1:]
    delta = p - p[:, :1]
    aa, ab, bb = (a * a).sum(1), (a * b).sum(1), (b * b).sum(1)
    det = aa * bb - ab * ab
    da, db = (delta * a[:, None]).sum(2), (delta * b[:, None]).sum(2)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        s = (da * bb[:, None] - db * ab[:, None]) / det[:, None]
        t = (db * aa[:, None] - da * ab[:, None]) / det[:, None]
        residual = delta - s[..., None] * a[:, None] - t[..., None] * b[:, None]
        valid &= ((det > 0) & (s * s + t * t <= .9 ** 2).all(axis=1)
                  & (np.linalg.norm(residual, axis=2) <= .005 * z).all(axis=1))
    return valid


def select_surfel_lod(*, positions, u, v, scales, frame_indices, frames,
                      object_ids, depths):
    """Return original indices and counts for conservative 2x2 thinning.

    Unknown owners, incomplete cells, duplicate pixels and discontinuities stay
    intact. Representatives are always the top-left source pixel, retaining
    their original position and footprint. This does not guarantee equal render
    coverage: it checks the four original centers against the bounded disk.
    """
    positions, u, v, scales = (np.asarray(a, dtype=np.float32)
                               for a in (positions, u, v, scales))
    frame_indices, object_ids = np.asarray(frame_indices), np.asarray(object_ids)
    n = len(positions)
    if (positions.shape != (n, 3) or u.shape != (n, 3) or v.shape != (n, 3)
            or scales.shape != (n, 2) or frame_indices.shape != (n,)
            or object_ids.shape != (n,)):
        raise ValueError('Surfel arrays have incompatible shapes')
    if (not np.issubdtype(frame_indices.dtype, np.integer)
            or not np.issubdtype(object_ids.dtype, np.integer)
            or np.any(frame_indices < 0) or np.any(frame_indices >= len(frames))):
        raise ValueError('Invalid Surfel frame indices or object IDs')
    keep = np.ones(n, dtype=bool)
    eligible_count = reduced_count = 0
    for frame_id in np.unique(frame_indices):
        slots = np.flatnonzero(frame_indices == frame_id)
        frame = frames[int(frame_id)]
        e = np.asarray(frame['extrinsic'], dtype=np.float32)
        k = np.asarray(frame['intrinsic'], dtype=np.float32)
        depth = np.asarray(depths[int(frame_id)], dtype=np.float32)
        h, w = depth.shape
        p, fu, fv, fs = (a[slots] for a in (positions, u, v, scales))
        with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
            center = p @ e[:, :3].T + e[:, 3]
            projected = center @ k.T
            xy = projected[:, :2] / projected[:, 2:3]
        valid = (np.isfinite(center).all(axis=1) & (center[:, 2] > 0)
                 & np.isfinite(xy).all(axis=1) & (projected[:, 2] > 0)
                 & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1)
                 & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)
                 & (object_ids[slots] >= 0))
        pixel = np.zeros((len(slots), 2), dtype=np.intp)
        pixel[valid] = np.floor(xy[valid] + .5).astype(np.intp)
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            continue
        keys = np.column_stack((object_ids[slots[candidates]], pixel[candidates] // 2))
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        order = np.argsort(inverse, kind='stable')
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        groups = candidates[order[starts[counts == 4, None] + np.arange(4)]]
        corner = pixel[groups, 0] % 2 + 2 * (pixel[groups, 1] % 2)
        permutation = np.argsort(corner, axis=1, kind='stable')
        groups = np.take_along_axis(groups, permutation, axis=1)
        complete = (np.take_along_axis(corner, permutation, axis=1) == np.arange(4)).all(axis=1)
        groups = groups[complete]
        eligible_count += len(groups)
        native = np.maximum(1e-6, center[:, 2:3] * np.linalg.norm(np.linalg.inv(k)[:, :2], axis=0))
        for start in range(0, len(groups), 8192):
            batch = groups[start:start + 8192]
            accepted = _covered(batch, p, fu, fv, fs, center, pixel, native, e, depth)
            keep[slots[batch[accepted, 1:].ravel()]] = False
            reduced_count += int(accepted.sum())
    indices = np.flatnonzero(keep)
    return indices, dict(input_count=n, selected_count=len(indices),
                         removed_count=n - len(indices), eligible_blocks=eligible_count,
                         reduced_blocks=reduced_count)
