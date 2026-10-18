"""Independent CPU diagnostic of current-default Viewer and optional dedup.

Never invoke beside the baseline benchmark. Outputs are retained, including failures.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager
import functools
import inspect
import json
import logging
import os
from pathlib import Path
import resource
import sys
import textwrap
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
METRICS = defaultdict(lambda: dict(calls=0, inclusive_s=0.0, self_s=0.0, max_s=0.0))
COUNTS = []
SOURCES = []
LOCAL = threading.local()
LOCK = threading.Lock()
NATIVE = {
    'remove_statistical_outlier': 'SOR_native',
    'cluster_dbscan': 'DBSCAN_native',
    'segment_plane': 'RANSAC_native',
    'compute_nearest_neighbor_distance': 'NN_native',
    'select_by_index': 'pcd_select',
}


@contextmanager
def span(name):
    stack = getattr(LOCAL, 'stack', None)
    if stack is None:
        stack = LOCAL.stack = []
    frame = [time.perf_counter(), 0.0]
    stack.append(frame)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - frame[0]
        stack.pop()
        if stack:
            stack[-1][1] += elapsed
        with LOCK:
            item = METRICS[name]
            item['calls'] += 1
            item['inclusive_s'] += elapsed
            item['self_s'] += elapsed - frame[1]
            item['max_s'] = max(item['max_s'], elapsed)


def observe(event, values):
    """Materialize scalar counts only; never retain locals or arrays."""
    import numpy as np

    row = {'event': event, 'thread': threading.current_thread().name}
    for key in ('points', 'valid_points', 'valid_indices', 'valid_idx', 'filtered_points',
                'sub_keep', 'stage_keep', 'background', 'protected', 'background_keep',
                'background_candidates', 'product_keep', 'indices', 'candidates',
                'subsample', 'inliers', 'rest', 'keep', 'order', 'selected'):
        value = values.get(key)
        if isinstance(value, np.ndarray):
            row[key + '_count'] = len(value)
            if value.dtype == np.bool_:
                row[key + '_true'] = int(np.count_nonzero(value))
    for key in ('base_mask', 'valid', 'keep_filter', 'sor_mask', 'product', 'cluster_mask', 'ground_mask'):
        value = values.get(key)
        if isinstance(value, np.ndarray):
            row[key + '_count'] = int(value.size)
            row[key + '_true'] = int(np.count_nonzero(value))
    labels = values.get('labels')
    if isinstance(labels, np.ndarray):
        row['labels_count'] = int(labels.size)
        # Cluster labels also use -1; the event/function identifies their domain.
        row['labels_nonnegative'] = int(np.count_nonzero(labels >= 0))
        row['labels_negative'] = int(np.count_nonzero(labels < 0))
        keep_filter = values.get('keep_filter')
        if isinstance(keep_filter, np.ndarray) and labels.shape == keep_filter.shape:
            row['kept_product'] = int(np.count_nonzero(keep_filter & (labels >= 0)))
            row['kept_background'] = int(np.count_nonzero(keep_filter & (labels < 0)))
    for key in ('median_nn', 'spacing', 'dist', 'distance_threshold', 'min_inliers', 'tilt_deg',
                'below_ratio', 'on_pos', 'on_neg', 'stage', 'n', 'budget', 'background_budget'):
        value = values.get(key)
        if isinstance(value, (str, int, float, np.integer, np.floating)):
            row[key] = value.item() if isinstance(value, np.generic) else value
    prepared = values.get('prepared')
    if prepared is not None and hasattr(prepared, 'sor_mask'):
        row['prepared_median_nn'] = float(prepared.median_nn)
        row['prepared_base'] = int(np.count_nonzero(prepared.base_mask))
        row['prepared_raw_sor'] = None if prepared.sor_mask is None else int(np.count_nonzero(prepared.sor_mask))
    pcd = values.get('pcd')
    if pcd is not None:
        row['pcd_count'] = len(pcd.points)
    remaining = values.get('remaining')
    if remaining is not None and hasattr(remaining, 'points'):
        row['remaining_count'] = len(remaining.points)
    sub_keep, protected, valid_idx = (values.get(key) for key in ('sub_keep', 'protected', 'valid_idx'))
    if (all(isinstance(item, np.ndarray) for item in (sub_keep, protected, valid_idx))
            and sub_keep.shape == valid_idx.shape):
        row['protected_current'] = int(np.count_nonzero(protected[valid_idx]))
        row['protected_rescued'] = int(np.count_nonzero(~sub_keep & protected[valid_idx]))
        config = values.get('config')
        if config is not None:
            row['guard_skipped'] = bool(np.count_nonzero(sub_keep | protected[valid_idx]) < config.min_remaining_ratio * len(valid_idx))
    config = values.get('config')
    if config is not None and hasattr(config, 'cluster_proxy_voxel_scale'):
        row['cluster_proxy_voxel_scale'] = float(config.cluster_proxy_voxel_scale)
    cache = values.get('cache')
    if isinstance(cache, dict) and 'points' in cache:
        row['native_shape'] = list(cache['points'].shape)
        row['native_points'] = int(np.prod(cache['points'].shape[:-1]))
    for key in ('early_counts',):
        if isinstance(values.get(key), dict):
            row[key] = {k: int(v) for k, v in values[key].items()}
    with LOCK:
        COUNTS.append(row)


def measured_call(name, fn, *args, **kwargs):
    owner = getattr(fn, '__self__', None)
    if owner is not None and hasattr(owner, 'points'):
        observe(name + '.input', {'pcd': owner})
    with span(name):
        result = fn(*args, **kwargs)
    # Scalar observations deliberately outside native timer.
    if name.endswith(('SOR_native', 'RANSAC_native')):
        observe(name + '.output', {'inliers': __import__('numpy').asarray(result[1])})
    elif name.endswith('DBSCAN_native'):
        observe(name + '.output', {'labels': __import__('numpy').asarray(result)})
    return result


def observer_statement(event):
    return ast.Expr(ast.Call(ast.Name('_profile_observe', ast.Load()),
                             [ast.Constant(event), ast.Call(ast.Name('locals', ast.Load()), [], [])], []))


def instrument_function(fn, prefix):
    """Add spans/observations, preserving calls, arguments and random state."""
    source, first = inspect.getsourcelines(fn)
    tree = ast.parse(textwrap.dedent(''.join(source)))
    name = fn.__name__
    mapping = {'function': name, 'path': str(Path(inspect.getfile(fn)).relative_to(ROOT)),
               'first_line': first, 'last_line': first + len(source) - 1, 'spans': []}

    class Transform(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr in NATIVE:
                label = prefix + NATIVE[node.func.attr]
                mapping['spans'].append({'name': label, 'line': first + node.lineno - 1})
                return ast.copy_location(ast.Call(ast.Name('_profile_call', ast.Load()),
                    [ast.Constant(label), node.func, *node.args], node.keywords), node)
            return node

        def visit_Return(self, node):
            # Evaluate the original return once, then observe and return it.
            self.generic_visit(node)
            if node.value is None:
                return [ast.copy_location(observer_statement(prefix + name + '.return'), node), node]
            assignment = ast.Assign([ast.Name('_profile_return_value', ast.Store())], node.value)
            observation = observer_statement(prefix + name + '.return')
            returned = ast.Return(ast.Name('_profile_return_value', ast.Load()))
            return [ast.copy_location(item, node) for item in (assignment, observation, returned)]

        def visit_If(self, node):
            original_test = ast.unparse(node.test)
            self.generic_visit(node)
            label = None
            if name == 'filter_scene_points':
                for key, suffix in [('prepared.sor_mask is not None', 'SOR_apply_block'),
                                    ('config.keep_main_clusters', 'DBSCAN_block'),
                                    ('config.remove_ground', 'ground_block')]:
                    if key in original_test:
                        label = prefix + suffix
                        break
            if label is None:
                return node
            mapping['spans'].append({'name': label, 'line': first + node.lineno - 1})
            wrapped = ast.With([ast.withitem(ast.Call(ast.Name('_profile_span', ast.Load()),
                                                       [ast.Constant(label)], []))], [node])
            return ast.copy_location(wrapped, node)

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            event = prefix + node.name
            start = observer_statement(event + '.enter')
            ast.copy_location(start, node)
            at = 1 if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str) else 0
            node.body.insert(at, start)
            body = []
            for statement in node.body:
                body.append(statement)
                # Coarse algorithm boundaries; never per point or pixel.
                if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                    assigned = {item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name)}
                    watched = {'stage_keep', 'valid_idx', 'sor_mask', 'median_nn', 'keep_filter',
                               'filtered_points', 'background_keep', 'product_keep', 'labels', 'inliers', 'remaining'}
                    if assigned & watched:
                        checkpoint = observer_statement(event + '.after_' + '_'.join(sorted(assigned & watched)))
                        body.append(ast.copy_location(checkpoint, statement))
            node.body = body
            return node

    tree = Transform().visit(tree)
    tree.body[0].decorator_list = []
    if name == '_sample_points':
        ranges = [(560, 587, 'setup_geometry'), (588, 590, 'labels'),
                  (591, 612, 'features'), (613, 644, 'geometry_filter'),
                  (645, 686, 'selection_filters'), (687, 693, 'postfilter_voxels'),
                  (694, 725, 'budget'), (726, 755, 'serialize_arrays')]
        grouped, pending, pending_label = [], [], None

        def flush():
            if pending:
                if pending_label is None:
                    grouped.extend(pending)
                else:
                    context = ast.Call(ast.Name('_profile_span', ast.Load()),
                                       [ast.Constant('sampling.' + pending_label)], [])
                    grouped.append(ast.copy_location(ast.With([ast.withitem(context)], list(pending)), pending[0]))
                pending.clear()

        for statement in tree.body[0].body:
            line = first + statement.lineno - 1
            label = next((label for lo, hi, label in ranges if lo <= line <= hi), None)
            if label != pending_label:
                flush()
                pending_label = label
            pending.append(statement)
        flush()
        tree.body[0].body = grouped
        mapping['coarse_spans'] = [{'name': 'sampling.' + label, 'first_line': lo, 'last_line': hi}
                                   for lo, hi, label in ranges]
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, first - 1)
    scope = fn.__globals__
    scope.update(_profile_call=measured_call, _profile_span=span, _profile_observe=observe)
    exec(compile(tree, inspect.getfile(fn), 'exec'), scope)
    SOURCES.append(mapping)
    return scope[name]


def wrap(owner, name, label=None):
    fn = getattr(owner, name)
    label = label or name
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        observe(label + '.input', bound.arguments)
        with span(label):
            result = fn(*args, **kwargs)
        if isinstance(result, dict) and 'positions' in result:
            observe(label + '.output', {'points': result['positions'], 'labels': result['instance_labels']})
        elif name == '_select_background_voxels':
            observe(label + '.output', {'background_keep': result})
        elif name == '_sample_product_points':
            observe(label + '.output', {'product_keep': result})
        elif name == 'select':
            observe(label + '.output', {'selected': result[0]})
        return result
    setattr(owner, name, call)


def install():
    import src.web_viewer_export as viewer
    import src.surfel_export as surfel
    import src.surfel_sampling as sampling
    import src.deduplicate_detections as dedup
    import utils.pointcloud_filter as filters
    import utils.scene_orientation as orientation

    for name in ('_median_nn_distance', 'prepare_scene_filter', '_cluster_labels', 'filter_scene_points'):
        instrument_function(getattr(filters, name), 'filter.')
    instrument_function(orientation.fit_scene_orientation, 'orientation.')
    instrument_function(viewer._sample_points, 'sampling.')
    # Replace imported aliases after compilation, retaining shared live globals.
    viewer.prepare_scene_filter = filters.prepare_scene_filter
    viewer.filter_scene_points = filters.filter_scene_points
    viewer.fit_scene_orientation = orientation.fit_scene_orientation
    for name in ('_median_nn_distance', 'prepare_scene_filter', '_cluster_labels', 'filter_scene_points'):
        wrap(filters, name, 'filter.' + name)
    viewer.prepare_scene_filter = filters.prepare_scene_filter
    viewer.filter_scene_points = filters.filter_scene_points
    wrap(orientation, 'fit_scene_orientation', 'orientation.fit_scene_orientation')
    viewer.fit_scene_orientation = orientation.fit_scene_orientation
    for name in ('prepare_web_viewer_geometry', 'export_web_viewer_bundle', '_load_da3_cache',
                 '_sample_points', '_instance_labels_v2', '_select_background_voxels',
                 '_sample_product_points', '_publish_bundle', '_generate_thumbnails',
                 'get_scene_orientation', '_minimal_objects', '_attach_point_index_ranges',
                 '_compute_world_to_view', '_robust_display_bounds'):
        wrap(viewer, name, 'viewer.' + name)
    for name in ('grid_tangents', 'prepare_surfel_textures', 'prepare_surfels'):
        wrap(surfel, name, 'surfel.' + name)
    wrap(sampling.EdgeAdaptiveSampler, '__init__', 'adaptive.features')
    wrap(sampling.EdgeAdaptiveSampler, 'select', 'adaptive.select')
    for name in ('parse_all_matches', 'filter_best_matches', 'build_global_mapping',
                 'add_global_id_to_jsons', 'save_merged_detections_with_gid', '_publish_global_pair'):
        wrap(dedup, name, 'dedup.' + name)
    return viewer


def diagnose(base, out, include_dedup):
    viewer = install()
    dataset = base / 'dataset'
    output = base / 'outputs/dataset'
    cache_dir = out / 'cache'
    cache_dir.mkdir()
    cache = cache_dir / 'predictions.npz'
    os.link(output / 'da3_cache/predictions.npz', cache)
    dedup_result = None
    if include_dedup:
        import main
        isolated = out / 'outputs/dataset'
        isolated.mkdir(parents=True)
        (isolated / 'output_3dmapping_da3').symlink_to(output / 'output_3dmapping_da3', target_is_directory=True)
        app = main.SKUDetectionMain()
        app.save_root = out / 'outputs'
        app.match_backend = 'da3'
        with span('dedup.total'):
            dedup_result = app.run_dedup_sequence(str(dataset), algorithm='3d', backend='da3',
                                                detection_dir=dataset / 'detections_results')
    with span('viewer.prepare_total'):
        prepared = viewer.prepare_web_viewer_geometry(da3_cache_path=cache,
            source_images_dir=dataset / 'images', surfel_texture_edge=1920)
    observe('prepared.summary', {'prepared': prepared.scene_filter, 'cache': prepared.cache,
                                'valid_indices': prepared.valid_indices})
    with span('viewer.finalize_total'):
        result = viewer.export_web_viewer_bundle(dataset_name='dataset', da3_cache_path=cache,
            global_mapping_path=output / 'dedup_detections/global_mapping.json', output_dir=out / 'viewer',
            source_images_dir=dataset / 'images', sam3_mask_cache_root=output / 'sam3_mask_cache_sam31_pvs/v2',
            surfel_texture_edge=1920, prepared=prepared)
    return {'viewer': result, 'dedup': dedup_result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('runtime/profiling-20261018'))
    parser.add_argument('--out', type=Path, required=True, help='New independent directory; must not exist')
    parser.add_argument('--dedup', action='store_true', help='Also rerun dedup into isolated outputs')
    args = parser.parse_args()
    base, out = args.base.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True)
    started = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF)
    result, error, status = None, None, 'failed'
    try:
        with span('diagnostic.total'):
            result = diagnose(base, out, args.dedup)
        status = 'complete'
    except BaseException:
        error = traceback.format_exc()
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        receipt = {'status': status, 'base': str(base), 'out': str(out), 'result': result, 'exception': error,
            'elapsed_s': time.perf_counter() - started, 'metrics': dict(METRICS), 'counts': COUNTS,
            'source_map': SOURCES, 'rss_hwm_kib': usage.ru_maxrss,
            'cpu_user_s': usage.ru_utime - before.ru_utime, 'cpu_system_s': usage.ru_stime - before.ru_stime,
            'threads': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')},
            'scope': 'Independent serial CPU diagnostic; instrumented timings do not replace baseline or measure matching overlap. No RNG seeding or production algorithm changes.'}
        (out / 'result.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'status': status, 'result_path': str(out / 'result.json')}), flush=True)


if __name__ == '__main__':
    main()
