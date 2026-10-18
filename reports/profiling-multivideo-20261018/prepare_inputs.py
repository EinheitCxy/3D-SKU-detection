"""Restore every detection frame using an explicit sampling record; no model runs."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CANONICAL_SOURCE = ROOT / 'runtime/worktrees/mapping-docker-rebuild/processor.py'
HELPERS = frozenset({'_canonical_frame', '_canonical_object', '_validate_labels',
                     '_validate_confidence', '_reject_nonfinite'})


def load_adapter():
    from utils.classification_aggregation import build_resolved_classification

    tree = ast.parse(CANONICAL_SOURCE.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
    if {node.name for node in functions} != HELPERS:
        raise ValueError('Canonical processor helper set differs from the five required helpers')
    tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *functions]
    ast.fix_missing_locations(tree)
    scope = dict(json=json, math=math, deepcopy=deepcopy,
                 build_resolved_classification=build_resolved_classification,
                 _FRAME_KEYS=frozenset({'classes', 'objects'}),
                 _CLASS_KEYS=frozenset({'det', 'cls'}))
    exec(compile(tree, str(CANONICAL_SOURCE), 'exec'), scope)
    return scope['_canonical_frame']


def read_frames(video):
    path = ROOT / f'fd-videos/{video}.json'
    if not path.is_file():
        raise ValueError(f'Video {video} has no detection JSON: {path}; excluded from same-protocol profiling')
    frames = json.loads(path.read_text())
    if not isinstance(frames, list) or not frames:
        raise ValueError('Detection JSON must be a nonempty frame list')
    counts = []
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != {'classes', 'objects'} or not isinstance(frame['objects'], list):
            raise ValueError('Detection frame must have classes and objects')
        counts.append(len(frame['objects']))
    return frames, counts


def sampling_plan(video, count, fps, total_frames, record_path):
    if record_path is None:
        if video != 3:
            raise ValueError(f'No verified sampling record registered for video {video}; supply --sampling-record with original frame indices and evidence')
        indices = [round(i * 1.5 * fps) for i in range(count)]
        evidence = {'type': 'historical_restoration_recipe', 'interval_s': 1.5,
            'source': 'runtime/video3-full-profile-20261016/run_profile.py:prepare_input',
            'qualification': 'Reproduces the established video3 profiling input recipe; original detector timestamps are absent'}
    else:
        record = json.loads(record_path.read_text())
        if record.get('video_id') != video:
            raise ValueError('Sampling record video_id must equal --video')
        indices = record.get('video_frame_indices')
        evidence = record.get('sampling_basis')
        if not isinstance(evidence, dict) or not isinstance(evidence.get('source'), str) or not evidence['source'].strip():
            raise ValueError('Sampling record requires sampling_basis.source describing evidence or an explicit user decision')
        evidence = dict(evidence, record_path=str(record_path.resolve()))
    if (not isinstance(indices, list) or len(indices) != count
            or any(type(index) is not int for index in indices)
            or any(a >= b for a, b in zip(indices, indices[1:]))
            or indices[0] < 0 or indices[-1] >= total_frames):
        raise ValueError('Sampling indices must be strictly increasing in-range integers, one per JSON frame')
    return indices, evidence


def prepare(video, out, record_path):
    import cv2

    started = time.perf_counter()
    receipt = {'status': 'failed', 'video_id': video, 'dataset_name': 'dataset',
               'preprocessing_excluded_from_core_pipeline': True, 'timings_s': {}}
    out.mkdir(parents=True, exist_ok=False)
    cap = None
    try:
        phase = time.perf_counter()
        frames, counts = read_frames(video)
        video_path = ROOT / f'fd-videos/{video}.mp4'
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {video_path}')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or total_frames < 1:
            raise ValueError('Video FPS/frame count metadata is invalid')
        indices, evidence = sampling_plan(video, len(frames), fps, total_frames, record_path)
        receipt.update(frames=len(frames), objects=sum(counts), objects_per_frame=counts,
            video_path=str(video_path), detection_json=str(ROOT / f'fd-videos/{video}.json'),
            fps=fps, video_total_frames=total_frames, video_duration_s=total_frames / fps,
            video_frame_indices=indices, frame_times_s=[index / fps for index in indices],
            sampling_basis=evidence, jpeg_quality=95, frames_truncated=False,
            classification='Existing canonical five helpers and build_resolved_classification(domain=51); no classifier inference',
            canonical_source=str(CANONICAL_SOURCE), canonical_helpers=sorted(HELPERS))
        receipt['timings_s']['json_video_metadata_and_sampling_plan'] = time.perf_counter() - phase
        phase = time.perf_counter()
        adapter = load_adapter()
        receipt['timings_s']['canonical_adapter_load'] = time.perf_counter() - phase
        images = out / 'dataset/images'
        detections = out / 'dataset/detections_results'
        images.mkdir(parents=True)
        detections.mkdir()
        selected = dict(zip(indices, range(len(frames))))
        decoding = encoding = canonical = writing = 0.0
        shapes = set()
        restored = 0
        for index in range(indices[-1] + 1):
            phase = time.perf_counter()
            grabbed = cap.grab()
            decoding += time.perf_counter() - phase
            if not grabbed:
                raise RuntimeError(f'Video decode ended at frame {index}')
            if index not in selected:
                continue
            frame_id = selected[index]
            phase = time.perf_counter()
            ok, image = cap.retrieve()
            decoding += time.perf_counter() - phase
            if not ok or image is None:
                raise RuntimeError(f'Cannot retrieve video frame {index}')
            shapes.add(tuple(image.shape))
            phase = time.perf_counter()
            written = cv2.imwrite(str(images / f'{frame_id}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            encoding += time.perf_counter() - phase
            if not written:
                raise RuntimeError(f'JPEG write failed at detection frame {frame_id}')
            phase = time.perf_counter()
            adapted = adapter(json.dumps(frames[frame_id], ensure_ascii=False, allow_nan=False), frame_id)
            canonical += time.perf_counter() - phase
            phase = time.perf_counter()
            (detections / f'{frame_id}.json').write_text(json.dumps(adapted, ensure_ascii=False, allow_nan=False))
            writing += time.perf_counter() - phase
            restored += 1
        receipt.update(status='complete', restored_frames=restored, image_shapes=[list(shape) for shape in sorted(shapes)])
        receipt['timings_s'].update(video_grab_retrieve=decoding, jpeg_encode_and_write=encoding,
                                   canonical_frame_adaptation=canonical, canonical_json_encode_and_write=writing)
        return receipt
    except BaseException:
        receipt['exception'] = traceback.format_exc()
        raise
    finally:
        if cap is not None:
            cap.release()
        receipt['preprocessing_wall_s'] = time.perf_counter() - started
        (out / 'input.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=int, choices=range(1, 9))
    parser.add_argument('--out', required=True, type=Path, help='New independent case directory')
    parser.add_argument('--sampling-record', type=Path,
                        help='JSON: video_id, video_frame_indices, sampling_basis.source; required absent established recipe')
    args = parser.parse_args()
    result = prepare(args.video, args.out.resolve(), args.sampling_record)
    print(json.dumps({'status': result['status'], 'video_id': args.video, 'frames': result['frames'],
                      'objects': result['objects'], 'preprocessing_wall_s': result['preprocessing_wall_s']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
