"""Allocate a fixed observation budget using a geometric coverage *proxy*.

No points are fused, moved, filtered by a new threshold, or assigned new sources.
The returned indices address the original C-order (frame, y, x) grids. Occupied
cells are not visibility coverage; that must be measured by the coordinator.
"""

from __future__ import annotations

import operator
import time

import numpy as np


DISPLAY_SCALE = 4.0
CELL_SPACING = 4.0
LAYER_SPACING = 1.0
NORMAL_SLOPE_BINS = 4


def _normal_bins(normals):
    """Signed cube-map bins, plus the unit normal at each bin's center.

    The six signed major axes and two four-bin slopes distinguish opposite
    faces and sides. A separate thin plane-offset bin distinguishes parallel
    layers sharing a spatial cell. Neither operation averages geometry.
    """
    major = np.argmax(np.abs(normals), axis=1)
    row = np.arange(len(normals))
    sign = np.where(normals[row, major] < 0, -1.0, 1.0)
    first, second = (major + 1) % 3, (major + 2) % 3
    denominator = np.abs(normals[row, major])
    slopes = np.column_stack(
        (normals[row, first] / denominator, normals[row, second] / denominator)
    )
    bins = np.floor((slopes + 1) * (NORMAL_SLOPE_BINS / 2)).astype(np.int8)
    np.clip(bins, 0, NORMAL_SLOPE_BINS - 1, out=bins)
    code = ((2 * major + (sign < 0)) * NORMAL_SLOPE_BINS**2
            + bins[:, 0] * NORMAL_SLOPE_BINS + bins[:, 1]).astype(np.uint8)
    centers = np.zeros_like(normals)
    centers[row, major] = sign
    centers[row, first] = (bins[:, 0] + 0.5) * 2 / NORMAL_SLOPE_BINS - 1
    centers[row, second] = (bins[:, 1] + 0.5) * 2 / NORMAL_SLOPE_BINS - 1
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return code, centers


def _source_quality(points, u, v, normals, yy, xx, frame, data, spacing):
    """Rank observations, retaining all existing valid candidates.

    Eight source-view ring samples approximate support of the unchanged 4x
    disc. Four immediate neighbors measure local plane/normal coherence, so
    isolated novel points do not receive an automatic quality advantage.
    This score does not replace or relax the exporter's visibility gate.
    """
    pose, intrinsic = data["E"][frame], data["K"][frame]
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    cu, cv = u @ pose[:3, :3].T, v @ pose[:3, :3].T
    mask, measured = data["valid"][frame], data["depth"][frame]
    height, width = mask.shape
    ring_support = np.zeros(len(points), dtype=np.uint8)
    for angle in np.arange(8) * (2 * np.pi / 8):
        ring = camera + 1.05 * DISPLAY_SCALE * (
            np.cos(angle) * cu + np.sin(angle) * cv
        )
        projected = ring @ intrinsic.T
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = projected[:, :2] / projected[:, 2:3]
        finite = np.isfinite(xy).all(axis=1) & np.isfinite(ring[:, 2])
        inside = (finite & (ring[:, 2] > 0) & (xy[:, 0] >= 0)
                  & (xy[:, 0] <= width - 1) & (xy[:, 1] >= 0)
                  & (xy[:, 1] <= height - 1))
        safe = np.where(finite[:, None], xy, 0)
        px = np.rint(safe[:, 0]).clip(0, width - 1).astype(np.int64)
        py = np.rint(safe[:, 1]).clip(0, height - 1).astype(np.int64)
        depth = measured[py, px]
        tolerance = np.maximum(2 * spacing, 0.005 * ring[:, 2])
        ring_support += (inside & mask[py, px] & np.isfinite(depth) & (depth > 0)
                         & (np.abs(depth - ring[:, 2]) <= tolerance))

    local_support = np.zeros(len(points), dtype=np.uint8)
    normal_cosine = np.cos(np.deg2rad(35))
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = yy + dy, xx + dx
        inside = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
        ny, nx = ny.clip(0, height - 1), nx.clip(0, width - 1)
        neighbor = data["points_grid"][frame, ny, nx]
        neighbor_normal = np.cross(data["u_grid"][frame, ny, nx],
                                   data["v_grid"][frame, ny, nx])
        length = np.linalg.norm(neighbor_normal, axis=1)
        neighbor_normal /= np.maximum(length[:, None], 1e-20)
        alignment = np.abs(np.einsum("ij,ij->i", normals, neighbor_normal))
        plane_distance = np.abs(np.einsum("ij,ij->i", neighbor - points, normals))
        local_support += (inside & mask[ny, nx] & (alignment >= normal_cosine)
                          & (plane_distance <= spacing))

    center = -pose[:3, :3].T @ pose[:3, 3]
    direction = center - points
    cosine = np.abs(np.einsum("ij,ij->i", normals, direction))
    cosine /= np.maximum(np.linalg.norm(direction, axis=1), 1e-20)
    confidence = data["confidence_grid"][frame, yy, xx]
    if not np.isfinite(confidence).all() or np.any(confidence < 0):
        raise ValueError("Existing valid confidence must be finite and nonnegative")
    # Confidence has a bounded 10% contribution, not a global top-K role.
    quality = (0.50 * ring_support / 8 + 0.25 * local_support / 4
               + 0.15 * np.clip(cosine, 0, 1) + 0.10 * confidence / (1 + confidence))
    longest = np.maximum(np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1))
    # Keep footprints unchanged, but demote unusually wide disks as representatives.
    quality *= np.sqrt(np.minimum(1.0, 2 * spacing / longest))
    return quality.astype(np.float32), ring_support, local_support


def select(data, budget):
    """Return exactly ``budget`` valid original observations with scale (4, 4).

    Within each spatial/normal/layer cell, visit distinct half-size subcells
    before repeated observations, choosing higher-quality representatives first.
    Across cells use quality-weighted round-robin slots. Thus repeated views
    cannot buy priority merely by adding more observations to a surface cell.
    """
    started = time.monotonic()
    if isinstance(budget, (bool, np.bool_)):
        raise ValueError("budget must be a positive integer")
    budget = operator.index(budget)
    valid = np.asarray(data["valid"])
    if valid.ndim != 3 or valid.dtype != np.bool_:
        raise ValueError("valid must be a boolean (frame, y, x) grid")
    frames, height, width = valid.shape
    for name in ("points_grid", "u_grid", "v_grid"):
        if np.shape(data[name]) != (*valid.shape, 3):
            raise ValueError(f"{name} must align with the original valid grid")
    for name in ("depth", "confidence_grid"):
        if np.shape(data[name]) != valid.shape:
            raise ValueError(f"{name} must align with the original valid grid")
    if np.shape(data["E"]) != (frames, 4, 4) or np.shape(data["K"]) != (frames, 3, 3):
        raise ValueError("Expected fixed per-frame E[4,4] and K[3,3]")
    if not (np.isfinite(data["E"]).all() and np.isfinite(data["K"]).all()):
        raise ValueError("Camera matrices must be finite")
    spacing = float(data["spacing"])
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    indices = np.flatnonzero(valid.reshape(-1))
    size = len(indices)
    if not 1 <= budget <= size:
        raise ValueError("budget must be between one and the number of valid observations")

    # Only compact keys/scores persist across frames; geometry stays in its grids.
    keys = np.empty((size, 4), dtype=np.int64)  # xyz cell and normal-plane layer
    normal_code = np.empty(size, dtype=np.uint8)
    subcell = np.empty(size, dtype=np.uint8)
    quality = np.empty(size, dtype=np.float32)
    ring_support = np.empty(size, dtype=np.uint8)
    local_support = np.empty(size, dtype=np.uint8)
    offset = 0
    for frame in range(frames):
        yy, xx = np.nonzero(valid[frame])
        count = len(yy)
        if not count:
            continue
        section = slice(offset, offset + count)
        points = np.asarray(data["points_grid"][frame, yy, xx], dtype=np.float64)
        u = np.asarray(data["u_grid"][frame, yy, xx], dtype=np.float64)
        v = np.asarray(data["v_grid"][frame, yy, xx], dtype=np.float64)
        normals = np.cross(u, v)
        lengths = np.linalg.norm(normals, axis=1)
        if (not np.isfinite(points).all() or not np.isfinite(lengths).all()
                or np.any(lengths <= 0)):
            raise ValueError("Existing valid observations require finite nondegenerate geometry")
        normals /= lengths[:, None]
        pose = data["E"][frame]
        camera_center = -pose[:3, :3].T @ pose[:3, 3]
        normals[np.einsum("ij,ij->i", normals, camera_center - points) < 0] *= -1
        code, bin_normals = _normal_bins(normals)
        scaled = points / (CELL_SPACING * spacing)
        if np.max(np.abs(scaled)) >= np.iinfo(np.int64).max / CELL_SPACING:
            raise ValueError("World coordinates exceed the coverage-key integer range")
        keys[section, :3] = np.floor(scaled).astype(np.int64)
        keys[section, 3] = np.floor(
            np.einsum("ij,ij->i", points, bin_normals) / (LAYER_SPACING * spacing)
        ).astype(np.int64)
        half = np.floor(2 * scaled).astype(np.int64) % 2
        subcell[section] = half[:, 0] + 2 * half[:, 1] + 4 * half[:, 2]
        normal_code[section] = code
        quality[section], ring_support[section], local_support[section] = _source_quality(
            points, u, v, normals, yy, xx, frame, data, spacing
        )
        offset += count
    feature_seconds = time.monotonic() - started

    # Stable ties retain the original C-order observation. No geometry averaging.
    order = np.lexsort((-quality, subcell, keys[:, 3], normal_code,
                        keys[:, 2], keys[:, 1], keys[:, 0]))
    different = np.ones(size, dtype=bool)
    different[1:] = normal_code[order[1:]] != normal_code[order[:-1]]
    for axis in range(4):
        different[1:] |= keys[order[1:], axis] != keys[order[:-1], axis]
    starts = np.flatnonzero(different)
    counts = np.diff(np.r_[starts, size])
    cell_ids = np.repeat(np.arange(len(starts), dtype=np.int32), counts)
    fine_different = different.copy()
    fine_different[1:] |= subcell[order[1:]] != subcell[order[:-1]]
    fine_starts = np.flatnonzero(fine_different)
    fine_rank = np.arange(size) - np.repeat(fine_starts, np.diff(np.r_[fine_starts, size]))
    # Each subcell gets a representative before a second repeated observation.
    within = np.lexsort((-quality[order], fine_rank, cell_ids))
    order = order[within]
    del within, fine_rank, fine_different, fine_starts, different, keys, normal_code, subcell
    rank = np.arange(size) - np.repeat(starts, counts)
    cell_weight = 0.35 + 0.65 * quality[order[starts]]
    priority = (rank + 1) / np.repeat(cell_weight, counts)
    if budget == size:
        chosen = np.arange(size)
    else:
        threshold = np.partition(priority, budget - 1)[budget - 1]
        chosen = np.flatnonzero(priority < threshold)
        tied = np.flatnonzero(priority == threshold)
        missing = budget - len(chosen)
        # Spread an exact tie over the sorted spatial cells rather than one end.
        take = np.floor((np.arange(missing) + 0.5) * len(tied) / missing).astype(np.int64)
        chosen = np.r_[chosen, tied[take]]
    selected_local = order[chosen]
    selected = np.sort(indices[selected_local])
    selected_cells = np.unique(cell_ids[chosen])
    baseline_cells = None
    if "baseline_indices" in data:
        baseline = np.asarray(data["baseline_indices"], dtype=np.int64)
        positions = np.searchsorted(indices, baseline)
        if (np.any(positions >= size)
                or not np.array_equal(indices[positions], baseline)):
            raise ValueError("baseline_indices must refer to valid original observations")
        cell_of_observation = np.empty(size, dtype=np.int32)
        cell_of_observation[order] = cell_ids
        baseline_cells = int(len(np.unique(cell_of_observation[positions])))
    ranks, rank_counts = np.unique(rank[chosen] + 1, return_counts=True)
    stats = {
        "method": "quality_weighted_round_robin_coverage_proxy",
        "selection_only": True,
        "input_valid_observations": size,
        "budget": budget,
        "selected_observations": len(selected),
        "per_frame_selected": np.bincount(selected // (height * width), minlength=frames).tolist(),
        "scale": {"min": DISPLAY_SCALE, "max": DISPLAY_SCALE, "unique_uv": [[4.0, 4.0]]},
        "coverage_proxy": {
            "meaning": "occupied spatial/normal/layer cells; not projected visibility coverage",
            "cell_width_m": CELL_SPACING * spacing,
            "subcell_width_m": CELL_SPACING * spacing / 2,
            "normal_bins": 6 * NORMAL_SLOPE_BINS**2,
            "normal_plane_layer_width_m": LAYER_SPACING * spacing,
            "available_cells": len(starts),
            "selected_cells": len(selected_cells),
            "baseline_cells": baseline_cells,
            "selected_slot_histogram": dict(zip(map(str, ranks.tolist()), rank_counts.tolist())),
        },
        "quality": {
            "weights": {"source_ring_support": 0.50, "local_plane_support": 0.25,
                        "source_front_cosine": 0.15, "bounded_confidence": 0.10},
            "allocation_weight": "0.35 + 0.65 * best_cell_representative_quality",
            "large_tangent_penalty": "sqrt(min(1, 2*spacing / max(norm(u), norm(v))))",
            "selected_quantiles_0_25_50_75_100": np.quantile(quality[selected_local], [0, .25, .5, .75, 1]).tolist(),
            "all_ring_supported_fraction_input": float(np.mean(ring_support == 8)),
            "all_ring_supported_fraction_selected": float(np.mean(ring_support[selected_local] == 8)),
            "all_local_neighbors_supported_fraction_selected": float(np.mean(local_support[selected_local] == 4)),
            "source_ring_check": "8 unchanged disc vertices; existing mask and max(2*spacing,0.005*Z) depth tolerance",
            "local_check": "4 image neighbors, abs normal cosine >= cos(35deg), plane distance <= spacing",
        },
        "feature_seconds": feature_seconds,
        "selection_seconds": time.monotonic() - started,
        "limitations": [
            "No actual visibility, occlusion, instance identity, or held-out surface coverage is inferred.",
            "Bin boundaries split similar observations; sub-spacing parallel layers may share a cell.",
            "Coherent reconstruction outliers can still receive novel-cell budget.",
            "Source support only ranks samples; scale-4 disks and the export visibility gate are unchanged.",
        ],
    }
    return {"indices": selected.astype(np.int64, copy=False),
            "scales": np.full((budget, 2), DISPLAY_SCALE, dtype=np.float32),
            "stats": stats}
