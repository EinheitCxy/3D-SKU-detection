"""Cache-only manual-equivalent stack outline candidate; never publishes CURRENT."""
import argparse
import json
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from src import da3_footprint_stage as preparation
from utils.ground_stack_footprint import (elevated_world_points, select_support_plane_with_direction,
                                        refine_support_normal, estimate_frame_support_offsets)
from utils.outline_exchange import validate_scene, world_to_view, lift_outline, validate_outline_document
from utils.stack_outline import OutlineOptions, extract_stack_outlines


def run_stack_outline(dataset_path, save_root, *, sam3_mask_cache_root, viewer_manifest,
                      output, gap_radius_m, max_hole_area_m2, grid_resolution_m=0.005,
                      overwrite=False):
    options = OutlineOptions(gap_radius_m=gap_radius_m, max_hole_area_m2=max_hole_area_m2,
                             grid_resolution_m=grid_resolution_m)
    dataset, save_root, output = Path(dataset_path), Path(save_root), Path(output)
    png = output.with_suffix('.png')
    if not overwrite and (output.exists() or png.exists()):
        raise FileExistsError('candidate output or diagnostic PNG exists; use --overwrite explicitly')
    manifest = json.loads(Path(viewer_manifest).read_text())
    if not isinstance(manifest, dict):
        raise ValueError('Viewer manifest must be an object')
    scene = {key: manifest.get(key) for key in ('dataset_name', 'backend', 'frame_count', 'world_to_view', 'display_bounds')}
    transform = validate_scene(scene)
    if manifest.get('measurement') != {'coordinate_unit': 'm', 'horizontal_plane': 'viewer_xz', 'orientation_status': 'fitted'}:
        raise ValueError('Viewer manifest must have metric fitted Viewer XZ measurement; re-export bundle')
    report = {'per_global_id': {}, 'mask_frames': []}
    cache, images, detections, mapping, initial_stat = preparation.prepare_da3_inputs(dataset, save_root, report)
    if scene['dataset_name'] != dataset.name or scene['backend'] != 'DA3' or scene['frame_count'] != len(cache['image_ids']):
        raise ValueError('Viewer scene dataset/backend/frame_count mismatch; re-export bundle')
    cache_path = save_root / dataset.name / 'da3_cache' / 'predictions.npz'
    orientation, event = preparation.get_scene_orientation(cache_path)
    if orientation.status != 'fitted' or not np.allclose(transform[:3, :3], orientation.rotation, atol=1e-6, rtol=0):
        raise ValueError('Viewer rotation differs from shared DA3 orientation; re-export bundle')
    observations, background, frames, _, normals = preparation._masked_observations(
        cache, images, detections, mapping, report['per_global_id'], Path(sam3_mask_cache_root),
        report['mask_frames'], collect_evidence_masks=False)
    current_stat = cache_path.stat()
    if any(getattr(initial_stat, key) != getattr(current_stat, key)
           for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
        raise ValueError('DA3 cache changed while preparing outline geometry')
    all_observations = [item for items in observations.values() for item in items]
    all_points = [item.points for item in all_observations if len(item.points)]
    height_diagnostics = {}
    if all_points:
        plane, height_diagnostics = select_support_plane_with_direction(
            background, frames, all_points, up_direction=orientation.normal_world,
            background_normals=normals)
        plane, refinement = refine_support_normal(plane, background, normals, frames)
        offsets, offset_diagnostics = estimate_frame_support_offsets(
            background, normals, frames, plane, image_ids=[int(x) for x in cache['image_ids']])
        height_diagnostics.update(normal_refinement=refinement, frame_support_offsets=offset_diagnostics)
        parts = [elevated_world_points([item], plane, offsets) for item in all_observations]
        points = np.concatenate(parts)
        point_frames = np.concatenate([np.full(len(part), item.image_id, dtype=np.int64)
                                       for item, part in zip(all_observations, parts)])
    else:
        points, point_frames = np.empty((0, 3)), np.empty(0, dtype=np.int64)
    elevated_counts = [{'image_id': int(item.image_id), 'valid_point_count': len(item.points),
                        'elevated_point_count': len(part)}
                       for item, part in zip(all_observations, parts)] if all_points else []
    projected = world_to_view(points, scene)[:, [0, 2]]
    result = extract_stack_outlines(projected, options=options, frame_ids=point_frames)
    complete = bool(result.polygons) and result.diagnostics['complete']
    document = {'format': 'viewer-horizontal-outlines', 'version': 1, 'coordinate_unit': 'm',
                'kind': 'candidate', 'scene': scene, 'status': 'candidate' if complete else 'needs_review',
                'complete': complete, 'value_m2': None,
                'metric': 'da3_manual_equivalent_stack_outline_candidate',
                'parameters': {'gap_radius_m': gap_radius_m, 'max_hole_area_m2': max_hole_area_m2,
                               'grid_resolution_m': grid_resolution_m, 'elevation_threshold_m': 0.015,
                               'max_grid_cells': options.max_grid_cells},
                'regions': [{'id': str(index), 'vertices_world': lift_outline(np.asarray(p.exterior.coords)[:-1], scene).tolist(),
                             'area_m2': p.area} for index, p in enumerate(result.polygons, 1)],
                'total_area_m2': float(sum(p.area for p in result.polygons)),
                'diagnostics': {'geometry': result.diagnostics, 'issues': list(result.issues),
                                'per_global_id': report['per_global_id'], 'mask_frames': report['mask_frames'],
                                'height_selection': height_diagnostics, 'orientation_cache_event': event,
                                'mapped_observation_count': len(all_observations),
                                'observations': elevated_counts,
                                'valid_mapped_point_count': sum(len(item.points) for item in all_observations),
                                'elevated_point_count': len(points)}}
    validate_outline_document(document, 'candidate')
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    for polygon in result.polygons:
        x, z = polygon.exterior.xy
        axis.fill(x, z, alpha=0.35)
    axis.set(xlabel='Viewer X (m)', ylabel='Viewer Z (m)', title=f'{document["status"]}: represented subtotal {document["total_area_m2"]:.4f} m²')
    axis.set_aspect('equal')
    try:
        figure.savefig(png, dpi=140)
    finally:
        plt.close(figure)
    with output.open('w' if overwrite else 'x') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'save-root', 'sam3-mask-cache-root', 'viewer-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('gap-radius-m', 'max-hole-area-m2'):
        parser.add_argument('--' + name, type=float, required=True)
    parser.add_argument('--grid-resolution-m', type=float, default=0.005)
    parser.add_argument('--overwrite', action='store_true')
    args = vars(parser.parse_args())
    args['dataset_path'] = args.pop('dataset')
    try:
        run_stack_outline(**args)
    except (ValueError, OSError) as error:
        parser.exit(2, f'outline candidate failed: {error}\n')


if __name__ == '__main__':
    main()
