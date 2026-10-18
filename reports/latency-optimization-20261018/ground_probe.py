"""CPU-only old-control vs sampled-ground on one shared post-SOR/DBSCAN domain.

Run only when the coordinator schedules this heavy component comparison. It
reads the coordinator-captured domain; it never reruns NN/SOR/DBSCAN or exports a Viewer.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import logging
import os
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASELINE = ROOT / 'runtime/latency-optimization-20261018/baseline-code/utils/pointcloud_filter.py'


def load_baseline(path):
    spec = importlib.util.spec_from_file_location('ground_probe_baseline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NativeTimerCloud:
    """Observe native control calls without modifying or copying its cloud."""
    def __init__(self, cloud, observations):
        self.cloud, self.observations = cloud, observations

    @property
    def points(self):
        return self.cloud.points

    def segment_plane(self, *args, **kwargs):
        started = time.perf_counter()
        plane, inliers = self.cloud.segment_plane(*args, **kwargs)
        self.observations.update(proposal_s=time.perf_counter() - started,
                                 native_inliers=len(inliers), plane=list(plane))
        return plane, inliers

    def select_by_index(self, *args, **kwargs):
        return NativeTimerCloud(self.cloud.select_by_index(*args, **kwargs), self.observations)


def run(captured_domain, out, cap, seed, baseline_path):
    import numpy as np
    import open3d as o3d
    import utils.pointcloud_filter as candidate

    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    receipt = dict(status='failed', captured_domain=str(captured_domain), baseline_source=str(baseline_path),
                   candidate_source='utils/pointcloud_filter.py', candidate_cap=cap,
                   open3d_seed_per_arm=seed, results={}, quality={},
                   scope='One CPU component comparison. Shared coordinator-captured post-SOR/DBSCAN domain; no Viewer/delivery/model replay, no saved point clouds. Seeds controlled here only; production Open3D RNG is not explicitly seeded.',
                   threads={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    try:
        baseline = load_baseline(baseline_path)
        prepared_at = time.perf_counter()
        metadata = json.loads((captured_domain / 'ground_meta.json').read_text())
        domain = np.load(captured_domain / 'ground_points.npy', mmap_mode='r', allow_pickle=False)
        protected = np.load(captured_domain / 'ground_protected.npy', mmap_mode='r', allow_pickle=False)
        if (domain.ndim != 2 or domain.shape[1] != 3 or domain.dtype != np.float64
                or protected.shape != (len(domain),) or protected.dtype != np.bool_):
            raise ValueError('Captured ground domain must be float64[N,3] and aligned bool[N] protection')
        median_nn = float(metadata['median_nn'])
        config_values = dict(metadata['config'])
        if not config_values.get('enabled', True) or not config_values.get('remove_ground', True):
            raise ValueError('Captured configuration does not enable ground filtering')
        if len(domain) < max(3, config_values.get('min_points', 1000)) or median_nn <= 0:
            raise ValueError('Captured domain is below the original geometric ground execution gate')
        cloud = o3d.geometry.PointCloud()
        # Open3D requires a writable buffer; domain stays read-only mmap.
        # This copy and cloud construction belong to captured_domain_load_s,
        # outside both arm timers.
        cloud.points = o3d.utility.Vector3dVector(np.array(domain, dtype=np.float64, copy=True))
        receipt['captured_domain_load_s'] = time.perf_counter() - prepared_at
        receipt['domain'] = dict(current_points=len(domain),
            protected_points=int(np.count_nonzero(protected)),
            background_points=int(np.count_nonzero(~protected)), median_nn=median_nn,
            ground_threshold_m=float(np.clip(config_values.get('ground_dist_scale', 3) * median_nn, .02, .15)),
            boundary='Coordinator baseline capture immediately before ground; no upstream stages rerun',
            points_path=str(captured_domain / 'ground_points.npy'),
            protected_path=str(captured_domain / 'ground_protected.npy'),
            metadata_path=str(captured_domain / 'ground_meta.json'))
        masks = {}
        for name in ('baseline', 'candidate'):
            module = baseline if name == 'baseline' else candidate
            # NN/SOR are supplied via PreparedSceneFilter; DBSCAN is disabled
            # because the capture already contains their exact retained domain.
            arm_config = dict(config_values, keep_main_clusters=False)
            arm_config.pop('ground_proposal_max_points', None)
            if name == 'candidate':
                arm_config['ground_proposal_max_points'] = cap
            config = module.PointCloudFilterConfig(**arm_config)
            info = {}
            observed_cloud = NativeTimerCloud(cloud, info)
            prepared = module.PreparedSceneFilter(domain, config, np.ones(len(domain), dtype=bool), median_nn, None, observed_cloud)
            original_helper = getattr(candidate, '_ground_filter_mask', None)
            if name == 'candidate':
                def helper(*args, **kwargs):
                    mask, values = original_helper(*args, **kwargs)
                    # The outer function augments this scalar dict with apply/guard.
                    info['ground'] = values
                    return mask, values
                candidate._ground_filter_mask = helper
            o3d.utility.random.seed(seed)
            try:
                arm_start = time.perf_counter()
                keep = module.filter_scene_points(domain, config, protected, prepared=prepared)
                elapsed = time.perf_counter() - arm_start
            finally:
                if name == 'candidate':
                    candidate._ground_filter_mask = original_helper
            masks[name] = keep
            receipt['results'][name] = dict(wall_s=elapsed, points_kept=int(keep.sum()),
                product_kept=int(np.count_nonzero(keep & protected)),
                background_kept=int(np.count_nonzero(keep & ~protected)),
                removed=int(np.count_nonzero(~keep)), diagnostics=info, config=asdict(config))
        old, new = masks['baseline'], masks['candidate']
        intersection, union = int(np.count_nonzero(old & new)), int(np.count_nonzero(old | new))
        added, removed = new & ~old, old & ~new
        receipt['quality'] = dict(keep_equal=bool(np.array_equal(old, new)), changed=int(np.count_nonzero(old != new)),
            candidate_adds=int(np.count_nonzero(added)), candidate_drops=int(np.count_nonzero(removed)),
            changed_product=int(np.count_nonzero((old != new) & protected)),
            changed_background=int(np.count_nonzero((old != new) & ~protected)),
            keep_jaccard=intersection / union if union else 1.0,
            both_preserve_all_products=bool(np.all(old[protected]) and np.all(new[protected])),
            acceptance='Measured masks/counts only; background changes require review and visual quality acceptance. No automatic promotion.')
        receipt['status'] = 'complete'
        return receipt
    except BaseException:
        receipt['exception'] = traceback.format_exc()
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        receipt.update(total_wall_s=time.perf_counter() - start, rss_hwm_kib=usage.ru_maxrss,
            cpu_user_s=usage.ru_utime - usage_start.ru_utime, cpu_system_s=usage.ru_stime - usage_start.ru_stime)
        (out / 'result.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captured-domain', type=Path, required=True,
                        help='Case directory containing ground_points.npy, ground_protected.npy, ground_meta.json')
    parser.add_argument('--out', required=True, type=Path, help='New output directory')
    parser.add_argument('--cap', type=int, default=50_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--baseline-source', type=Path, default=BASELINE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True)
    result = run(args.captured_domain.resolve(), args.out.resolve(), args.cap, args.seed, args.baseline_source.resolve())
    print(json.dumps(dict(status=result['status'], domain=result['domain'], results=result['results'], quality=result['quality']), indent=2), flush=True)
