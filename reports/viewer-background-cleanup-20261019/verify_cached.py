"""Verify current Viewer filtering from cached models; no inference or GLB bake."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import web_viewer_export as viewer
from utils import pointcloud_filter as filtering


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def product_payloads(generation):
    count = (generation / 'positions.f32.bin').stat().st_size // 12
    product = np.zeros(count, dtype=bool)
    for entry in json.loads((generation / 'objects.json').read_text()).values():
        for start, end in entry['point_ranges']:
            product[start:end] = True
    files = {'positions.f32.bin': '<f4', 'colors.u8.bin': 'u1',
             'normals.i8.bin': 'i1', 'surfel-u.f16.bin': '<f2',
             'surfel-v.f16.bin': '<f2', 'surfel-scale.f16.bin': '<f2',
             'surfel-frame.u8.bin': 'u1'}
    payloads = {name: np.fromfile(generation / name, dtype=dtype).reshape(count, -1)[product].tobytes()
                for name, dtype in files.items()}
    return int(product.sum()), payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'scope': 'Cached Viewer export, one run per input; '
              'no inference, ZIP, GLB, GPU or human visual acceptance.', 'cases': []}
    write_json(out / 'result.json', result)
    config = filtering.PointCloudFilterConfig()
    native_filter = viewer.filter_scene_points
    native_cluster = filtering._cluster_labels
    native_plane = filtering._ground_filter_mask
    try:
        for index, case in enumerate(args.case):
            if shutil.disk_usage(out).free < 4 * 1024**3:
                raise RuntimeError('Less than 4 GiB disk free before export')
            case = case.resolve()
            cache = case / 'outputs/dataset/da3_cache/predictions.npz'
            source = case / 'viewer/runs' / json.loads((case / 'viewer/CURRENT').read_text())['run_id']
            row = {'case': str(case), 'status': 'running', 'config': asdict(config)}
            result['cases'].append(row)
            write_json(out / 'result.json', result)

            def observe_filter(points, active_config, **kwargs):
                started = time.perf_counter()
                keep = native_filter(points, active_config, **kwargs)
                protected = np.asarray(kwargs['protect_mask'], dtype=bool)
                base = kwargs['prepared'].base_mask
                row.update(filter_s=time.perf_counter() - started, input_points=len(points),
                           filter_points=int(keep.sum()), protected_points=int(protected.sum()),
                           removed_protected_points=int(np.count_nonzero(base & protected & ~keep)))
                if row['removed_protected_points']:
                    raise AssertionError('Valid protected product points were removed')
                np.save(out / f'case{index}-keep.npy', keep, allow_pickle=False)
                return keep

            def observe_cluster(cloud, median, active_config):
                started = time.perf_counter()
                labels = native_cluster(cloud, median, active_config)
                row['dbscan_s'] = time.perf_counter() - started
                row['proxy_edge_m'] = median * active_config.cluster_proxy_voxel_scale
                return labels

            def observe_plane(cloud, median, active_config, **kwargs):
                started = time.perf_counter()
                keep, info = native_plane(cloud, median, active_config, **kwargs)
                info['wall_s'] = time.perf_counter() - started
                row['background_plane'] = info
                if keep is not None and np.any(~keep & kwargs['protect_mask']):
                    raise AssertionError('Plane deletion proposed protected product points')
                orientation = json.loads(cache.with_name('scene_orientation.json').read_text())
                if 'plane' in info and orientation['normal_world'] is not None:
                    cosine = abs(float(np.dot(info['plane'][:3], orientation['normal_world'])))
                    info['angle_to_orientation_deg'] = float(np.degrees(np.arccos(np.clip(cosine, 0, 1))))
                return keep, info

            o3d.utility.random.seed(42)
            started = time.perf_counter()
            prepared = viewer.prepare_web_viewer_geometry(
                da3_cache_path=cache, source_images_dir=case / 'dataset/images',
                surfel_texture_edge=1920, filter_config=config,
            )
            with patch.object(viewer, 'filter_scene_points', side_effect=observe_filter), \
                 patch.object(filtering, '_cluster_labels', side_effect=observe_cluster), \
                 patch.object(filtering, '_ground_filter_mask', side_effect=observe_plane):
                exported = viewer.export_web_viewer_bundle(
                    dataset_name=case.name, da3_cache_path=cache,
                    global_mapping_path=case / 'outputs/dataset/dedup_detections/global_mapping.json',
                    output_dir=out / f'case{index}/viewer', source_images_dir=case / 'dataset/images',
                    sam3_mask_cache_root=case / 'outputs/dataset/sam3_mask_cache_sam31_pvs/v2',
                    surfel_texture_edge=1920, voxel_size_m=.005, filter_config=config,
                    prepared=prepared,
                )
            row['viewer_s_including_serial_preparation'] = time.perf_counter() - started
            row['viewer_points'] = exported['point_count']
            generation = Path(exported['manifest_path']).parent
            row['generation'] = str(generation)
            count, payloads = product_payloads(generation)
            reference_count, reference_payloads = product_payloads(source)
            row['viewer_product_points'] = count
            row['product_payload_equal'] = {
                name: payload == reference_payloads[name] for name, payload in payloads.items()
            }
            if count != reference_count or not all(row['product_payload_equal'].values()):
                raise AssertionError('Final product payloads differ from this cache reference')
            row['status'] = 'complete'
            write_json(out / 'result.json', result)
            print(json.dumps(row), flush=True)
        result['status'] = 'complete'
        write_json(out / 'result.json', result)
    except BaseException:
        import traceback
        result.update(status='failed', error=traceback.format_exc())
        write_json(out / 'result.json', result)
        raise


if __name__ == '__main__':
    main()
