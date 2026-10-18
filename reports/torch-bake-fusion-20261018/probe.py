"""Coordinator-invoked bounded Torch bake experiment; CPU smoke by explicit flag."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
GPU2 = 'GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'sample', 'glb'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0',
                        help='CUDA requires CUDA_VISIBLE_DEVICES set to the physical GPU2 UUID')
    parser.add_argument('--generation', type=Path)
    parser.add_argument('--reference-glb', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='New independent output directory')
    parser.add_argument('--frames', type=int, default=6)
    parser.add_argument('--points-per-frame', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--encoder-workers', choices=(1, 2, 4), type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 1024 or args.frames < 1 or args.points_per_frame < 1:
        parser.error('Require positive sample bounds and batch-size within [1,1024]')
    if args.mode != 'smoke' and args.generation is None:
        parser.error('--generation is required for real samples/exports')
    if args.device != 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') != GPU2:
        parser.error('CUDA experiment requires explicit GPU2 UUID isolation')
    if args.mode == 'smoke' and args.device != 'cpu':
        parser.error('smoke mode requires --device cpu')
    if args.reference_glb is not None and not args.reference_glb.is_file():
        parser.error('--reference-glb does not exist')
    args.out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    result = dict(status='failed', mode=args.mode, device=args.device, command=sys.argv,
                  generation=str(args.generation) if args.generation else None,
                  reference_glb=str(args.reference_glb) if args.reference_glb else None,
                  batch_size=args.batch_size,
                  timing_scope='Imports separately measured. Sample frame decoding/input preparation is outside both timers. Native timer includes one frame constant setup and same batch partition; candidate includes FrameBaker first CUDA context, frame uploads, CPU source_world projection, Torch masks/RGB/dilation and downloads. Comparisons outside both timers. Full export timer includes input loading/LOD/encoding/writing too. No pre-timer CUDA synchronization.',
                  counts_scope='rgb_uploads/depth_uploads count one .to(device) per frame (CPU mode has no H2D)',
                  projection_mode='source_world', candidate_frames=[], comparisons=[])
    try:
        sys.path.insert(0, str(ROOT))
        tick = time.perf_counter()
        import numpy as np
        from PIL import Image
        import torch
        import torch_bake
        from src import surfel_mesh as native
        from src.surfel_glb import _assets, _load, export_scene_glb
        from src.surfel_lod import select_surfel_lod
        sys.path.insert(0, str(ROOT / 'reports/latency-optimization-20261018'))
        from bake_probe import synthetic_cases, compare_arrays
        from bake_export_compare import compare_glb
        result.update(imports_wall_s=time.perf_counter()-tick,
                      numpy=np.__version__, torch=torch.__version__,
                      cuda_initialized_before_candidate=torch.cuda.is_initialized())

        if args.mode == 'smoke':
            for label, values in synthetic_cases():
                p, u, v, scales, depth, frame, image, radius, tile_size = values
                tick = time.perf_counter()
                baker = torch_bake.FrameBaker(frame, image, depth, device='cpu',
                                             radius=radius, tile_size=tile_size,
                                             batch_limit=args.batch_size)
                candidate = baker.bake(p, u, v, scales)
                elapsed = time.perf_counter()-tick
                comparison = compare_arrays(native._bake_batch(*values), candidate)
                result['comparisons'].append(dict(label=label, **comparison))
                result['candidate_frames'].append(dict(label=label, wall_s=elapsed, **baker.counts))
            result['correctness_passed'] = all(x['geometry_order_equal'] and x['complete_rgba_equal'] for x in result['comparisons'])
        elif args.mode == 'sample':
            with _assets(args.generation) as read:
                _, inputs = _load(read, None)
                owners = np.full(len(inputs['positions']), -1, dtype=np.int32)
                for owner, entry in enumerate(json.loads(read('objects.json')).values()):
                    for start, end in entry['point_ranges']:
                        owners[start:end] = owner
                chosen, result['lod'] = select_surfel_lod(
                    **{key: inputs[key] for key in ('positions', 'u', 'v', 'scales', 'frame_indices', 'frames', 'depths')}, object_ids=owners)
                for key in ('positions', 'u', 'v', 'scales', 'frame_indices'):
                    inputs[key] = inputs[key][chosen]
                frames = np.unique(inputs['frame_indices'])
                frames = frames[np.linspace(0, len(frames)-1, min(args.frames, len(frames)), dtype=int)]
                for frame_id in frames:
                    tick = time.perf_counter()
                    frame = inputs['frames'][int(frame_id)]
                    with Image.open(BytesIO(read(frame['texture']))) as image:
                        rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
                    height = max(f['texture_size'][1] for f in inputs['frames'])
                    width = max(f['texture_size'][0] for f in inputs['frames'])
                    padded = np.zeros((height, width, 3), dtype=np.float32)
                    padded[:rgb.shape[0], :rgb.shape[1]] = rgb
                    depth = np.asarray(inputs['depths'][int(frame_id)], dtype=np.float32)
                    slots = np.flatnonzero(inputs['frame_indices'] == frame_id)
                    slots = slots[np.linspace(0, len(slots)-1, min(args.points_per_frame, len(slots)), dtype=int)]
                    values = tuple(np.asarray(inputs[key][slots], dtype=np.float32) for key in ('positions', 'u', 'v', 'scales'))
                    preparation_wall = time.perf_counter()-tick
                    tick = time.perf_counter()
                    constants = native._frame_constants(frame, 8)
                    baseline = []
                    for start in range(0, len(slots), args.batch_size):
                        baseline.append(native._bake_batch(
                            *(value[start:start+args.batch_size] for value in values), depth, frame,
                            padded, 2., 8, constants=constants, projection_mode='source_world'))
                    baseline_wall = time.perf_counter()-tick
                    context_before = torch.cuda.is_initialized()
                    tick = time.perf_counter()
                    baker = torch_bake.FrameBaker(frame, padded, depth, device=args.device, batch_limit=args.batch_size)
                    candidate = []
                    for start in range(0, len(slots), args.batch_size):
                        candidate.append(baker.bake(*(value[start:start+args.batch_size] for value in values)))
                    result['candidate_frames'].append(dict(
                        frame_id=int(frame_id), baseline_wall_s=baseline_wall,
                        candidate_wall_s=time.perf_counter()-tick,
                        decode_input_preparation_wall_s=preparation_wall,
                        first_candidate_frame=len(result['candidate_frames']) == 0,
                        cuda_initialized_before_frame=context_before, **baker.counts))
                    # Compare exactly the same batch partition outside both timers.
                    for index, start in enumerate(range(0, len(slots), args.batch_size)):
                        result['comparisons'].append(dict(frame_id=int(frame_id), batch_start=start,
                                                          **compare_arrays(baseline[index], candidate[index])))
                    del baker, baseline, candidate
            result['candidate_frame_wall_sum_s'] = sum(row['candidate_wall_s'] for row in result['candidate_frames'])
            result['baseline_frame_wall_sum_s'] = sum(row['baseline_wall_s'] for row in result['candidate_frames'])
            result['decode_input_preparation_wall_sum_s'] = sum(row['decode_input_preparation_wall_s'] for row in result['candidate_frames'])
            result['correctness_passed'] = all(x['geometry_order_equal'] and x['complete_rgba_equal'] for x in result['comparisons'])
        else:
            original = native.bake_surfel_pages
            metrics = {}
            def experimental(**kwargs):
                # Only page generation changes; CPU LOD/compact/PNG/FIFO stay native.
                kwargs['batch_size'] = args.batch_size
                return torch_bake.bake_surfel_pages(**kwargs, device=args.device, metrics=metrics)
            tick = time.perf_counter()
            native.bake_surfel_pages = experimental
            try:
                result['production_report'] = export_scene_glb(
                    args.generation, args.out / 'candidate.glb', projection_mode='source_world',
                    encoder_workers=args.encoder_workers)
            finally:
                native.bake_surfel_pages = original
                result['candidate_export_wall_s'] = time.perf_counter()-tick
                result['candidate_frames'] = metrics.get('frames', [])
            result['process_to_export_ready_s'] = time.perf_counter()-began
            if args.reference_glb is not None:
                comparison = compare_glb(args.reference_glb, args.out / 'candidate.glb')
                result['comparisons'].append(comparison)
                result['correctness_passed'] = (comparison['document_associations_equal']
                                                and comparison['accessor_bytes_equal']
                                                and comparison['decoded_rgba_equal'])
            else:
                result['correctness_passed'] = None
                result['missing'] = ['reference_glb; full output equality not checked']
        if args.mode in ('smoke', 'sample') and (
                not result['comparisons']
                or sum(row['old_surfel_count'] for row in result['comparisons']) == 0
                or sum(row['new_surfel_count'] for row in result['comparisons']) == 0):
            result['correctness_passed'] = False
            result['insufficient_sample_evidence'] = 'Empty comparisons or no baked surfels in one arm'
        if result.get('correctness_passed') is False:
            result['status'] = 'correctness_failed'
            raise AssertionError('Torch bake output differs from native source_world; see exact diff counts')
        result['status'] = ('unvalidated_export_only'
                            if args.mode == 'glb' and args.reference_glb is None else 'complete')
    except BaseException:
        result['exception'] = traceback.format_exc()
        raise
    finally:
        result['whole_process_wall_s'] = time.perf_counter()-began
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result['resources'] = dict(cpu_user_s=usage.ru_utime, cpu_system_s=usage.ru_stime,
                                   lifetime_maxrss_kib=usage.ru_maxrss)
        torch_module = sys.modules.get('torch')
        if torch_module is not None and torch_module.cuda.is_initialized():
            result['resources'].update(cuda_allocated_peak_bytes=torch_module.cuda.max_memory_allocated(),
                                       cuda_reserved_peak_bytes=torch_module.cuda.max_memory_reserved())
        (args.out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(status=result['status'], correctness_passed=result.get('correctness_passed'), output=str(args.out))), flush=True)


if __name__ == '__main__':
    main()
