"""Pure JSON extraction for one multivideo case; no model/GPU imports or writes.

Keep each process and each instrumented interval separate. Inclusive parent,
child, CUDA stream, and overlapping thread durations must never be added as
an end-to-end total. Whole-run resources/phase means belong to the coordinator.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


DOMAINS = ('core', 'da3', 'pvs', 'viewer')


def _domain(name: str) -> str:
    if (name.startswith('da3.') or name.startswith('da3_') or 'depth_anything_3/' in name
            or name.startswith('src/da3_runner.py:')):
        return 'da3'
    if name.startswith('pvs.') or name.startswith('pvs_') or 'utils/sam31_pvs.py:' in name:
        return 'pvs'
    if (name.startswith(('viewer.', 'filter.', 'orientation.', 'surfel.', 'adaptive.', 'sampling.'))
            or name in {'prepare_web_viewer_geometry', 'export_web_viewer_bundle', '_load_da3_cache',
                        '_sample_points', '_instance_labels_v2', '_select_background_voxels',
                        '_publish_bundle', '_generate_thumbnails', 'get_scene_orientation',
                        '_attach_point_index_ranges', 'prepare_scene_filter'}):
        return 'viewer'
    return 'core'


def _leaf(name: str) -> str:
    return name.rsplit(' > ', 1)[-1]


def _optional_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def _inside(child: dict, parent: dict) -> bool:
    return (child.get('pid') == parent.get('pid')
            and parent['start'] <= child['start'] <= child['end'] <= parent['end'])


def summarize(case: Path) -> dict:
    """Extract one case's completed/partial timing evidence without mutation.

    Missing optional inputs remain absent/null and are listed in missing_files;
    malformed existing JSON raises its original parsing error. Process schemas
    are identified, never upgraded from independent legacy diagnostic fixtures.
    """
    case = Path(case).resolve()
    if not case.is_dir():
        raise NotADirectoryError(case)
    detail_paths = sorted(case.glob('detail-*.json'))
    event_paths = sorted(case.glob('events-*.jsonl'))
    details = [(path, json.loads(path.read_text())) for path in detail_paths]
    events = []
    for path in event_paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                event = json.loads(line)
                events.append(dict(event, source_file=path.name, source_line=number))
    events.sort(key=lambda row: (row['start'], row['end'], row.get('pid', -1)))
    pipeline = _optional_json(case / 'pipeline-summary.json')
    stages = _optional_json(case / 'matching-stages.json')
    components = {name: [] for name in DOMAINS}
    snapshots, points, mappings, process_details = [], [], [], []
    for path, detail in details:
        pid = detail.get('pid')
        schema = 'in_run_detail_v1' if ('process' in detail and 'memory_snapshots' in detail) else 'unrecognized_or_legacy_detail'
        # Preserve unknown fields (including delivery extension) for auditability;
        # the extraction below does not reinterpret their inclusive timings.
        process_details.append(dict(pid=pid, source_file=path.name, schema=schema,
                                    status=detail.get('status'), detail=detail))
        by_domain = {name: dict(pid=pid, source_file=path.name, wall={}, wall_contexts={}, cuda_events={})
                     for name in DOMAINS}
        for name, value in detail.get('wall', {}).items():
            by_domain[_domain(name)]['wall'][name] = value
        for name, value in detail.get('wall_contexts', {}).items():
            by_domain[_domain(_leaf(name))]['wall_contexts'][name] = value
        for name, value in detail.get('cuda_events', {}).items():
            by_domain[_domain(_leaf(name))]['cuda_events'][name] = value
        for domain, row in by_domain.items():
            if any(row[field] for field in ('wall', 'wall_contexts', 'cuda_events')):
                components[domain].append(row)
        for index, snapshot in enumerate(detail.get('memory_snapshots', [])):
            snapshots.append(dict(pid=pid, source_file=path.name, source_index=index, snapshot=snapshot))
        for index, count in enumerate(detail.get('viewer_counts', [])):
            points.append(dict(pid=pid, source_file=path.name, source_index=index, observation=count))
        mappings.append(dict(pid=pid, source_file=path.name,
                             ast_source_mapping=detail.get('source_mapping', {}),
                             viewer_sources=detail.get('viewer_sources', [])))
    snapshots.sort(key=lambda row: row['snapshot'].get('t', 0))
    grouped = defaultdict(list)
    for event in events:
        grouped[event['name']].append(event)
    reference_events = grouped['reference_matching']
    predict_events = grouped['pvs_predict']
    load_events = grouped['pvs_model_load']
    matching_complete = bool(isinstance(pipeline, dict) and pipeline.get('matching') is True)
    # Profile_core uses serial all-reference execution and next pairing. Event
    # records do not carry original frame IDs, so preserve that limitation.
    complete_frame_count = len(reference_events) if matching_complete and reference_events else None
    references = []
    for slot, reference in enumerate(reference_events):
        predicts = [row for row in predict_events if _inside(row, reference)]
        loads = [row for row in load_events if _inside(row, reference)]
        reference = dict(reference)
        references.append(dict(reference_slot=slot,
                               target_slot=(slot + 1) % complete_frame_count if complete_frame_count else None,
                               source_image_id=None, reference_event=reference,
                               pvs_predict_events=predicts, pvs_model_load_events=loads,
                               association='same PID and event time containment; serial ordinal slot, original image ID not recorded'))
    pvs_native_rows = []
    for event in predict_events:
        metrics = event.get('metrics', {})
        pvs_native_rows.append(dict(pid=event.get('pid'), start=event['start'], end=event['end'],
                                    wall_s=event['seconds'], boxes=event.get('box_count'),
                                    encoder_s=metrics.get('encoder_s'),
                                    inference_and_postprocess_including_encoder_s=metrics.get('inference_and_postprocess_s'),
                                    error=event.get('error'),
                                    source_file=event['source_file'], source_line=event['source_line']))
    # These sums are repeated invocations of one exact interval; do not sum
    # encoder and native prediction, which share the same start boundary.
    pvs_native = dict(predict_calls=len(pvs_native_rows), rows=pvs_native_rows,
                      boxes=sum(row['boxes'] for row in pvs_native_rows) if all(isinstance(row['boxes'], int) for row in pvs_native_rows) else None,
                      encoder_s=sum(row['encoder_s'] for row in pvs_native_rows) if all(isinstance(row['encoder_s'], (int, float)) for row in pvs_native_rows) else None,
                      inference_and_postprocess_including_encoder_s=sum(row['inference_and_postprocess_including_encoder_s'] for row in pvs_native_rows)
                      if all(isinstance(row['inference_and_postprocess_including_encoder_s'], (int, float)) for row in pvs_native_rows) else None)
    expected = ('pipeline-summary.json', 'matching-stages.json')
    return dict(schema_version='multivideo_core_summary_v1', case=str(case),
                pipeline_summary=pipeline, matching_stages=stages,
                matching=dict(complete=matching_complete, completed_reference_events=len(reference_events),
                              frame_count=complete_frame_count, pairing_contract='next; profile_core default',
                              original_frame_ids_recorded=False),
                component_details=components, references=references, pvs_native=pvs_native,
                memory_snapshots=snapshots, point_counts=points, source_mappings=mappings,
                process_details=process_details,
                event_rows={domain: [event for event in events if _domain(event['name']) == domain] for domain in DOMAINS},
                provenance=dict(detail_files=[path.name for path in detail_paths], event_files=[path.name for path in event_paths],
                                optional_files=[name for name in expected if (case / name).is_file()]),
                missing_files=[name for name in expected if not (case / name).is_file()]
                              + ([] if detail_paths else ['detail-*.json']) + ([] if event_paths else ['events-*.jsonl']),
                interpretation='All components belong to this case only. Process-local inclusive/self statistics, full contexts/source maps, CUDA events and point observations retained separately; no parent/child inclusive sum, no independent diagnostic splice, no total-phase/resource/mean estimate. Instrumentation overhead remains unquantified.')
