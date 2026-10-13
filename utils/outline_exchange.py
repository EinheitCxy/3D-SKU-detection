"""Local Viewer outline interchange and sampled geometric agreement."""
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def validate_scene(scene):
    if not isinstance(scene, dict):
        raise ValueError("scene must be an object")
    for key in ('dataset_name', 'backend'):
        if not isinstance(scene.get(key), str) or not scene[key]:
            raise ValueError(f'scene {key} must be a nonempty string')
    count = scene.get('frame_count')
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError('scene frame_count must be a positive integer')
    transform = np.asarray(scene.get('world_to_view'), dtype=float)
    bounds = np.asarray(scene.get('display_bounds'), dtype=float)
    if transform.shape != (16,) or not np.isfinite(transform).all():
        raise ValueError('world_to_view must contain 16 finite row-major numbers')
    transform = transform.reshape(4, 4)
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError('world_to_view must be a rigid proper transform')
    if bounds.shape != (6,) or not np.isfinite(bounds).all() or np.any(bounds[:3] > bounds[3:]):
        raise ValueError('display_bounds must contain ordered finite min/max bounds')
    return transform


def world_to_view(points, scene):
    transform = validate_scene(scene)
    return np.asarray(points, dtype=float) @ transform[:3, :3].T + transform[:3, 3]


def lift_outline(vertices_xz, scene, viewer_y=0.0):
    transform = validate_scene(scene)
    xz = np.asarray(vertices_xz, dtype=float)
    view = np.column_stack((xz[:, 0], np.full(len(xz), viewer_y), xz[:, 1]))
    return (view - transform[:3, 3]) @ transform[:3, :3]


def matching_scenes(first, second):
    validate_scene(first)
    validate_scene(second)
    for key in ('dataset_name', 'backend', 'frame_count'):
        if first[key] != second[key]:
            raise ValueError(f'scene mismatch: {key}; re-export the target bundle')
    for key in ('world_to_view', 'display_bounds'):
        if not np.allclose(first[key], second[key], atol=1e-6, rtol=0):
            raise ValueError(f'scene mismatch: {key}; re-export the target bundle')


def validate_outline_document(document, expected_kind=None):
    if not isinstance(document, dict):
        raise ValueError("outline document must be an object")
    if (document.get('format') != 'viewer-horizontal-outlines'
            or document.get('version') != 1 or document.get('coordinate_unit') != 'm'):
        raise ValueError('unsupported outline format/version/unit')
    kind = document.get('kind')
    if kind not in ('manual', 'candidate') or (expected_kind and kind != expected_kind):
        raise ValueError(f'outline kind must be {expected_kind or "manual or candidate"}')
    scene = document.get('scene', {})
    validate_scene(scene)
    regions = document.get('regions')
    if not isinstance(regions, list):
        raise ValueError('regions must be a list')
    ids, polygons = set(), []
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("region must be an object")
        identifier = region.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError('region IDs must be unique nonempty strings')
        ids.add(identifier)
        vertices = np.asarray(region.get('vertices_world'), dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3 or not np.isfinite(vertices).all():
            raise ValueError('region requires at least three finite world vertices')
        if np.array_equal(vertices[0], vertices[-1]):
            raise ValueError('region vertices must be an open list')
        view = world_to_view(vertices, scene)
        if np.ptp(view[:, 1]) > 1e-6:
            raise ValueError('region must be coplanar with Viewer XZ')
        polygon = Polygon(view[:, [0, 2]])
        if not polygon.is_valid or not polygon.exterior.is_simple or polygon.area <= 0:
            raise ValueError('region must be a simple positive-area polygon')
        area = region.get('area_m2')
        if isinstance(area, bool) or not isinstance(area, (int, float)) or not np.isfinite(area) or not np.isclose(area, polygon.area, atol=1e-6, rtol=1e-6):
            raise ValueError('region area_m2 differs from recomputed geometry')
        if any(polygon.intersection(other).area > 1e-10 for other in polygons):
            raise ValueError('regions overlap with positive area')
        polygons.append(polygon)
    total = document.get('total_area_m2')
    if isinstance(total, bool) or not isinstance(total, (int, float)) or not np.isfinite(total) or not np.isclose(total, sum(p.area for p in polygons), atol=1e-6, rtol=1e-6):
        raise ValueError('total_area_m2 differs from represented geometry')
    if kind == 'manual':
        if document.get('status') != 'reference' or document.get('complete') is not True or document.get('metric') != 'manual_horizontal_outline':
            raise ValueError('manual document must be a complete reference')
    else:
        complete = document.get('complete')
        if (not isinstance(complete, bool) or document.get('value_m2') is not None
                or document.get('metric') != 'da3_manual_equivalent_stack_outline_candidate'
                or document.get('status') != ('candidate' if complete else 'needs_review')
                or (complete and not polygons)):
            raise ValueError('invalid candidate status/metric/value contract')
    return tuple(polygons)


def _boundary_samples(polygons, spacing):
    samples = []
    for polygon in polygons:
        for ring in (polygon.exterior, *polygon.interiors):
            count = max(1, int(np.ceil(ring.length / spacing)))
            samples.extend(ring.interpolate(index * ring.length / count) for index in range(count))
    return samples


def compare_outlines(candidate, reference, spacing_m=0.01):
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError('boundary spacing must be finite and positive')
    first = validate_outline_document(candidate, 'candidate')
    second = validate_outline_document(reference, 'manual')
    matching_scenes(candidate['scene'], reference['scene'])
    if not first:
        raise ValueError('candidate has no valid regions to compare')
    if not second:
        raise ValueError('reference has no valid regions to compare')
    a, b = unary_union(first), unary_union(second)
    distances = [np.percentile([p.distance(target.boundary) for p in _boundary_samples(source, spacing_m)], 95)
                 for source, target in (((a,) if isinstance(a, Polygon) else tuple(a.geoms), b),
                                        ((b,) if isinstance(b, Polygon) else tuple(b.geoms), a))]
    return {'candidate_region_count': len(first), 'reference_region_count': len(second),
            'candidate_complete': candidate['complete'], 'candidate_area_m2': a.area,
            'reference_area_m2': b.area, 'relative_area_difference': (a.area - b.area) / b.area,
            'union_iou': a.intersection(b).area / a.union(b).area,
            'candidate_to_reference_boundary_p95_m': float(distances[0]),
            'reference_to_candidate_boundary_p95_m': float(distances[1]),
            'boundary_sampling_spacing_m': spacing_m,
            'metric_scope': 'sampled polygon agreement; not physical accuracy'}
