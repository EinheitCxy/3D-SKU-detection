"""Scalar metrics for equal-video averaging; no model imports or file access.

Read each unique process detail once, never its summarize_core projections.
Inclusive/self/context/CUDA observations are distinct metrics, not additive
phases. Missing observations remain None, never an invented zero.
"""
from __future__ import annotations

from collections import defaultdict
import math


WALL_FIELDS = ('inclusive_s', 'self_s', 'calls', 'max_s', 'thread_cpu_ns')
SNAPSHOT_FIELDS = ('allocated_bytes', 'reserved_bytes', 'peak_allocated_bytes',
                   'peak_reserved_bytes', 'gpu_parameter_bytes', 'gpu_buffer_bytes')


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pointer(value):
    return str(value).replace('~', '~0').replace('/', '~1')


def _role(detail):
    argv = detail.get('process', {}).get('argv', [])
    for role in ('da3', 'core', 'delivery'):
        if any(str(value).endswith('profile_' + role + '.py') for value in argv):
            return role
    scope = detail.get('profile_scope', '')
    if scope.startswith('DA3 child'):
        return 'da3'
    if scope.startswith('One core pipeline'):
        return 'core'
    return None


def extract(summary: dict) -> dict:
    """Return flat key -> {value,unit,section,source,semantics,...}.

    Accept either a whole build_summary case (``core_detail`` member), or the
    direct summarize_core result. Source strings identify raw detail pointers;
    no aggregate process projections or independent diagnostic inputs are used.
    """
    evidence = summary.get('core_detail', summary)
    process_rows = evidence.get('process_details', [])
    output = {}
    seen_pid, seen_role = set(), set()

    def put(section, role, label, field, value, unit, source, semantics):
        key = '/'.join((section, role, label, field))
        if key in output:
            raise ValueError('Duplicate independent metric: ' + key)
        output[key] = dict(value=value, unit=unit, section=section, source=source,
                           semantics=semantics, process_role=role, label=label, field=field)

    for ordinal, process_row in enumerate(process_rows):
        detail = process_row['detail']
        pid = detail.get('pid', process_row.get('pid'))
        if pid is not None:
            if pid in seen_pid:
                raise ValueError(f'Duplicate raw process detail PID {pid}')
            seen_pid.add(pid)
        role = _role(detail) or ('other_' + str(ordinal))
        if role in seen_role:
            raise ValueError('More than one raw process detail for role ' + role)
        seen_role.add(role)
        filename = process_row.get('source_file', 'detail-' + str(pid) + '.json')
        for table in ('wall', 'wall_contexts'):
            for label, stats in detail.get(table, {}).items():
                for field in WALL_FIELDS:
                    raw = stats.get(field)
                    value = raw if _number(raw) else None
                    result_field = 'thread_cpu_s' if field == 'thread_cpu_ns' else field
                    if field == 'thread_cpu_ns' and value is not None:
                        value /= 1e9
                    put(table, role, label, result_field, value,
                        'calls' if field == 'calls' else 's',
                        filename + '#/' + table + '/' + _pointer(label) + '/' + field,
                        'Process-local measured interval. Inclusive contains child spans; self subtracts instrumented direct children only. Context rows project the same calls by ancestry and must not be added to wall rows. Thread CPU excludes worker threads; instrumentation overhead included.')
        for label, stats in detail.get('cuda_events', {}).items():
            for field in ('event_s', 'calls', 'max_s'):
                value = stats.get(field)
                put('cuda_events', role, label, field, value if _number(value) else None,
                    'calls' if field == 'calls' else 's',
                    filename + '#/cuda_events/' + _pointer(label) + '/' + field,
                    'CUDA stream elapsed interval; parents and children overlap. Native synchronization boundaries retained; not exclusive kernel time and not additive to host wall.')
        grouped = defaultdict(list)
        for index, snapshot in enumerate(detail.get('memory_snapshots', [])):
            grouped[snapshot['label']].append((index, snapshot))
        for label, snapshots in grouped.items():
            sources = ';'.join(filename + '#/memory_snapshots/' + str(index) for index, _ in snapshots)
            put('allocator_snapshot', role, label, 'samples', len(snapshots), 'samples', sources,
                'Number of snapshots at this named boundary, not a phase peak sampling rate.')
            fields = list(SNAPSHOT_FIELDS) + ['unused_reserved_bytes', 'non_parameter_buffer_allocated_bytes']
            for field in fields:
                values = []
                for _, row in snapshots:
                    if field == 'unused_reserved_bytes':
                        operands = (row.get('reserved_bytes'), row.get('allocated_bytes'))
                        value = operands[0] - operands[1] if all(_number(item) for item in operands) else None
                    elif field == 'non_parameter_buffer_allocated_bytes':
                        operands = (row.get('allocated_bytes'), row.get('gpu_parameter_bytes'), row.get('gpu_buffer_bytes'))
                        value = operands[0] - operands[1] - operands[2] if all(_number(item) for item in operands) else None
                    else:
                        value = row.get(field)
                    values.append(value if _number(value) else None)
                # A partly missing label group is missing, rather than averaging
                # a biased subset and presenting it as all occurrences.
                average = sum(values) / len(values) if all(value is not None for value in values) else None
                meaning = ('Lifetime allocator high-water mark read at each snapshot, averaged over occurrences; not a phase maximum.'
                           if field.startswith('peak_') else
                           'Current boundary values averaged over occurrences of this label; not a phase peak. Uninitialized CUDA or absent model information remains missing.')
                if field == 'unused_reserved_bytes':
                    meaning += ' Reserved minus allocated is unused caching-allocator reservation; it excludes driver/context allocations.'
                if field == 'non_parameter_buffer_allocated_bytes':
                    meaning += ' Allocated minus model parameter/buffer bytes includes scene tensors, activations and other allocations; it is not a measured model KV cache.'
                put('allocator_snapshot', role, label, 'sample_mean_' + field, average, 'bytes', sources, meaning)
        _pointflow(detail.get('viewer_counts', []), role, filename, put)
    return output


def _pointflow(rows, role, filename, put):
    """Use named observations, never mix orientation/cluster labels with products."""
    if not rows:
        return

    def take(event, field, stage=None, first=False):
        candidates = [(index, row) for index, row in enumerate(rows)
                      if row.get('event') == event and (stage is None or row.get('stage') == stage)]
        candidates = [(index, row) for index, row in candidates if field in row]
        if not candidates:
            return None, filename + '#/viewer_counts[missing:' + event + '/' + field + ']'
        index, row = candidates[0] if first else candidates[-1]
        value = row[field]
        return value if _number(value) or isinstance(value, bool) else None, filename + '#/viewer_counts/' + str(index) + '/' + field

    def emit(label, event, field, stage=None, first=False, semantics='Observed point-domain count at this named stage; repeated observer aliases are not summed.'):
        value, source = take(event, field, stage, first)
        put('pointflow', role, label, 'value', value, 'boolean' if field == 'guard_skipped' else 'points', source, semantics)

    emit('dense_native', 'sampling._sample_points.enter', 'native_points')
    emit('valid_geometry', 'viewer._instance_labels_v2.input', 'valid_indices_count')
    emit('product_before_filters', 'sampling._sample_points.after_labels', 'labels_nonnegative', first=True)
    emit('background_before_filters', 'sampling._sample_points.after_labels', 'labels_negative', first=True)
    emit('SOR_prepared_raw_keep', 'filter.prepare_scene_filter.return', 'sor_mask_true')
    for short, stage in (('SOR', 'statistical outlier removal'), ('DBSCAN', 'cluster retention'), ('ground', 'ground-plane removal')):
        emit(short + '_input', 'filter.apply.enter', 'valid_idx_count', stage)
        emit(short + '_raw_keep', 'filter.apply.enter', 'sub_keep_true', stage)
        emit(short + '_protected_rescued', 'filter.apply.enter', 'protected_rescued', stage,
             semantics='Protected points rejected by the raw rule; actual retained output also depends on the recorded guard.')
        emit(short + '_guard_skipped', 'filter.apply.enter', 'guard_skipped', stage)
        emit(short + '_after_apply', 'filter.apply.after_valid_idx', 'valid_idx_count', stage)
    emit('ground_RANSAC_input', 'filter.RANSAC_native.input', 'pcd_count')
    emit('ground_RANSAC_inliers', 'filter.RANSAC_native.output', 'inliers_count')
    plane_input, input_source = take('filter.RANSAC_native.input', 'pcd_count')
    plane_inliers, inlier_source = take('filter.RANSAC_native.output', 'inliers_count')
    ratio = plane_inliers / plane_input if _number(plane_input) and plane_input > 0 and _number(plane_inliers) else None
    put('geometry_observation', role, 'ground_RANSAC_inlier_ratio', 'value', ratio, 'ratio',
        input_source+';'+inlier_source,
        'Observed native ground inliers divided by its actual input domain; no assumed application or threshold decision.')
    emit('after_all_filters', 'sampling._sample_points.after_filtered_points', 'filtered_points_count')
    emit('product_after_filters', 'sampling._sample_points.after_filtered_points', 'kept_product')
    emit('background_after_filters', 'sampling._sample_points.after_filtered_points', 'kept_background')
    emit('background_voxel_candidates', 'sampling._sample_points.after_background_keep', 'background_keep_count')
    emit('background_budget', 'sampling._sample_points.after_product_keep', 'background_budget')
    emit('product_selected', 'viewer._sample_points.output', 'labels_nonnegative')
    emit('background_selected', 'viewer._sample_points.output', 'labels_negative')
    emit('selected_total', 'viewer._sample_points.output', 'points_count')
    # A capped product call can invoke sample_group once per global-ID quota.
    # Retain all observed quotas; the final quota is not the configured cap.
    active = False
    product_groups = []
    for index, row in enumerate(rows):
        if row.get('event') == 'viewer._sample_product_points.input':
            active = True
        elif row.get('event') == 'viewer._sample_product_points.output':
            active = False
        elif active and row.get('event') == 'sampling.sample_group.enter':
            value = row.get('budget')
            product_groups.append((index, value if _number(value) else None))
    product_source = (';'.join(filename + '#/viewer_counts/' + str(index) + '/budget'
                              for index, _ in product_groups)
                      or filename + '#/viewer_counts[missing:product_sampling_groups]')
    values = [value for _, value in product_groups]
    group_sum = sum(values) if values and all(value is not None for value in values) else None
    put('pointflow', role, 'product_group_budget_sum', 'value', group_sum, 'points', product_source,
        'Sum of all recorded sample_group quotas inside product sampling calls; allocated group-budget total, not necessarily the configured product cap. Missing if no groups or any quota is missing.')
    put('pointflow', role, 'product_last_group_budget', 'value', values[-1] if values else None,
        'points', product_source,
        'Last recorded product-group quota only; not a total allocation or configured product cap.')
    put('pointflow', role, 'product_group_count', 'value', len(values) if values else None,
        'groups', product_source,
        'Count of observed product sample_group calls. No observations remain missing rather than zero.')
