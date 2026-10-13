"""Bounded metric raster outlines in the target Viewer's horizontal XZ plane.

Cells are half-open squares [i*r, (i+1)*r) anchored at the metric origin.
Morphology distances are between cell centres, using a Euclidean disk. Foreground
uses four-neighbour connectivity and its complement uses eight neighbours:
corner-only contact never creates a self-touching accepted polygon. Returned
boundaries follow cell edges exactly, without smoothing or topology changes.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np
from scipy import ndimage
import shapely
from shapely.geometry import Polygon


@dataclass(frozen=True, kw_only=True)
class OutlineOptions:
    gap_radius_m: float
    max_hole_area_m2: float
    grid_resolution_m: float = 0.005
    max_grid_cells: int = 4_000_000

    def __post_init__(self):
        for name in ('gap_radius_m', 'max_hole_area_m2', 'grid_resolution_m'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0
                    or (name == 'grid_resolution_m' and value == 0)):
                raise ValueError(f'{name} must be finite and '
                                 f'{"positive" if name == "grid_resolution_m" else "nonnegative"}')
        if (isinstance(self.max_grid_cells, bool)
                or not isinstance(self.max_grid_cells, Integral)
                or self.max_grid_cells <= 0):
            raise ValueError('max_grid_cells must be a positive integer')


@dataclass(frozen=True)
class OutlineResult:
    polygons: tuple[Polygon, ...]
    diagnostics: dict
    issues: tuple[str, ...]


def _cell_geometry(mask, origin_x, origin_z, resolution):
    """Union row runs in bounded batches, never one geometry per raw point."""
    chunks = []
    runs = []
    for row in range(mask.shape[0]):
        edges = np.flatnonzero(np.diff(np.r_[False, mask[row], False]))
        for start, end in edges.reshape(-1, 2):
            runs.append((origin_x + int(start) * resolution,
                         origin_z + row * resolution,
                         origin_x + int(end) * resolution,
                         origin_z + (row + 1) * resolution))
        if len(runs) >= 1024:
            chunks.append(shapely.union_all(shapely.box(*np.asarray(runs).T)))
            runs.clear()
    if runs:
        chunks.append(shapely.union_all(shapely.box(*np.asarray(runs).T)))
    return shapely.union_all(chunks)


def extract_stack_outlines(points_xz: np.ndarray, *, options: OutlineOptions,
                           frame_ids: np.ndarray | None = None) -> OutlineResult:
    """Extract simple, hole-free outlines, retaining partial valid components.

    Invalid input/options and excessive grid extents raise ValueError before
    raster allocation. Empty input, degenerate original support, and large holes
    yield explicit issues. A reported subtotal is only represented geometry;
    diagnostics['complete'] says whether any geometry remains unresolved.
    """
    if not isinstance(options, OutlineOptions):
        raise ValueError('options must be OutlineOptions')
    points = np.asarray(points_xz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('points_xz must be a finite array of shape (N, 2)')
    frames = None
    if frame_ids is not None:
        frames = np.asarray(frame_ids)
        if (frames.ndim != 1 or len(frames) != len(points)
                or frames.dtype.kind not in 'iu'):
            raise ValueError('frame_ids must be a matching one-dimensional integer array')
    r = float(options.grid_resolution_m)
    diagnostics = {
        'complete': False, 'input_point_count': len(points),
        'grid_resolution_m': r, 'gap_radius_m': float(options.gap_radius_m),
        'max_hole_area_m2': float(options.max_hole_area_m2),
        'max_grid_cells': int(options.max_grid_cells),
        'grid_shape': [0, 0], 'grid_cells': 0, 'grid_origin_xz_m': None,
        'raw_occupied_cells': 0, 'raw_frame_count': 0, 'frame_cell_counts': {},
        'closed_occupied_cells': 0, 'filled_hole_cells': 0,
        'removed_cells': 0, 'unresolved_cells': 0, 'represented_cells': 0,
        'represented_area_m2': 0.0, 'components': [],
        'foreground_connectivity': 4, 'background_connectivity': 8,
        'cell_convention': 'floor anchored at metric origin; boundaries at cell edges',
        'morphology_convention': 'Euclidean disk between cell centres',
    }
    if not len(points):
        return OutlineResult((), diagnostics, ('empty original support',))
    with np.errstate(over='ignore', invalid='ignore'):
        scaled = np.floor(points / r)
        radius_cells = float(options.gap_radius_m) / r
    if not np.isfinite(scaled).all() or not math.isfinite(radius_cells):
        raise ValueError('metric extent or morphology padding exceeds max_grid_cells')
    low, high = scaled.min(axis=0), scaled.max(axis=0)
    pad = math.ceil(radius_cells) + 2
    # Subtract Python integers so opposing finite extremes cannot overflow
    # a float before the allocation bound is checked.
    width = int(high[0]) - int(low[0]) + 1 + 2 * pad
    height = int(high[1]) - int(low[1]) + 1 + 2 * pad
    if width * height > options.max_grid_cells:
        raise ValueError(f'grid {height}x{width} including morphology padding '
                         f'exceeds max_grid_cells={options.max_grid_cells}')
    # IEEE doubles cannot resolve unit cell edges beyond this index range.
    if np.max(np.abs(scaled)) > 2**52:
        raise ValueError('metric coordinates cannot represent grid cell edges precisely')
    indices = (scaled - low).astype(np.int64) + pad
    raw = np.zeros((height, width), dtype=bool)
    raw[indices[:, 1], indices[:, 0]] = True
    diagnostics.update(grid_shape=[height, width], grid_cells=height * width,
                       grid_origin_xz_m=((low - pad) * r).tolist(),
                       raw_occupied_cells=int(raw.sum()))
    if frames is not None:
        frame_values = np.unique(frames)
        diagnostics['raw_frame_count'] = len(frame_values)
        # Count occupied cells per frame, preserving overlap across frames.
        triples = np.unique(np.column_stack((frames, indices)), axis=0)
        values, counts = np.unique(triples[:, 0], return_counts=True)
        diagnostics['frame_cell_counts'] = {
            str(int(value)): int(count) for value, count in zip(values, counts)}
    if radius_cells:
        dilated = ndimage.distance_transform_edt(~raw) <= radius_cells
        closed = ndimage.distance_transform_edt(dilated) > radius_cells
    else:
        closed = raw.copy()
    diagnostics['closed_occupied_cells'] = int(closed.sum())
    diagnostics['removed_cells'] = int((raw & ~closed).sum())
    four = ndimage.generate_binary_structure(2, 1)
    eight = ndimage.generate_binary_structure(2, 2)
    labels, count = ndimage.label(closed, structure=four)
    slices = ndimage.find_objects(labels)
    # Associate ORIGINAL observations with closed components once; repeated
    # observations cannot create extra rank or width.
    point_labels = labels[indices[:, 1], indices[:, 0]]
    order = np.argsort(point_labels, kind='stable')
    sorted_labels = point_labels[order]
    polygons, issues = [], []
    for component_id, bounds in enumerate(slices, start=1):
        row_slice, col_slice = bounds
        mask = labels[bounds] == component_id
        cell_count = int(mask.sum())
        begin, end = np.searchsorted(sorted_labels, [component_id, component_id + 1])
        support = points[order[begin:end]]
        record = {'id': component_id, 'closed_cells': cell_count,
                  'original_point_count': len(support), 'status': 'unresolved',
                  'hole_areas_m2': [], 'filled_hole_cells': 0}
        diagnostics['components'].append(record)
        centered = support - support[0] if len(support) else np.empty((0, 2))
        if len(support) < 3 or np.linalg.matrix_rank(centered) < 2:
            reason = f'component {component_id}: original support lacks two-dimensional area'
            record['reason'] = reason
            issues.append(reason)
            diagnostics['unresolved_cells'] += cell_count
            continue
        filled = ndimage.binary_fill_holes(mask, structure=eight)
        holes, hole_count = ndimage.label(filled & ~mask, structure=eight)
        hole_sizes = np.bincount(holes.ravel(), minlength=hole_count + 1)[1:]
        hole_areas = hole_sizes.astype(float) * r * r
        record['hole_areas_m2'] = hole_areas.tolist()
        if np.any(hole_areas > options.max_hole_area_m2 + 1e-12 * r * r):
            reason = f'component {component_id}: hole exceeds max_hole_area_m2'
            record['reason'] = reason
            issues.append(reason)
            diagnostics['unresolved_cells'] += cell_count
            continue
        origin = (low + [col_slice.start - pad, row_slice.start - pad]) * r
        geometry = _cell_geometry(filled, float(origin[0]), float(origin[1]), r)
        candidates = (geometry,) if isinstance(geometry, Polygon) else tuple(geometry.geoms)
        if any(not isinstance(p, Polygon) or not p.is_valid or p.interiors
               or not p.exterior.is_simple or p.area <= 0 for p in candidates):
            reason = f'component {component_id}: cell boundary is not a simple hole-free polygon'
            record['reason'] = reason
            issues.append(reason)
            diagnostics['unresolved_cells'] += cell_count
            continue
        filled_count = int(hole_sizes.sum())
        record.update(status='represented', filled_hole_cells=filled_count,
                      represented_cells=cell_count + filled_count,
                      represented_area_m2=float(geometry.area))
        diagnostics['filled_hole_cells'] += filled_count
        diagnostics['represented_cells'] += cell_count + filled_count
        polygons.extend(shapely.normalize(p) for p in candidates)
    if diagnostics['removed_cells']:
        issues.append('morphology removed original occupied cells')
    polygons.sort(key=lambda p: (*p.bounds, p.area, p.wkb))
    diagnostics['represented_area_m2'] = float(sum(p.area for p in polygons))
    diagnostics['complete'] = not issues
    return OutlineResult(tuple(polygons), diagnostics, tuple(issues))
