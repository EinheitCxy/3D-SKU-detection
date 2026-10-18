"""One isolated DA3 allocator candidate; parent never imports Torch/CUDA."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
CASES = {3: (ROOT / 'runtime/latency-optimization-20261018/candidate-video3', 29),
         6: (ROOT / 'runtime/profiling-multivideo-20261018/video6', 48)}
MODEL = ROOT / 'runtime/profiling-20261018/da3-model/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7'
PYTHON = ROOT / 'runtime/viewer-latency-20261017/test-env/bin/python'
UUID = 'GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7'
CAP, STOP, BUDGET = 12 * 2**30, 13_500_000_000, 14_000_000_000
CONF = 'backend:native,expandable_segments:True'
KEYS = ('depth', 'depth_conf', 'world_points', 'world_points_conf', 'extrinsic', 'intrinsic',
        'images', 'image_ids', 'source_image_sizes', 'source_to_processed_affine', 'is_metric',
        'scale_factor', 'frame_alignment_sorted_indices', 'frame_alignment_map_keys', 'frame_alignment_map_values')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker(out, video):
    case, frames = CASES[video]
    images = case / 'dataset/images'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'worker-start.json').exists():
        raise FileExistsError('Refusing to overwrite existing worker-start.json')
    with (out / 'worker-start.partial').open('x') as stream:
        json.dump(dict(pid=os.getpid(), ppid=os.getppid(), pgid=os.getpgrp(), wall_time=time.time(), video_id=video), stream)
    (out/'worker-start.partial').rename(out/'worker-start.json')
    os.environ.update(CUDA_VISIBLE_DEVICES=UUID, PYTORCH_CUDA_ALLOC_CONF=CONF,
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    started = time.perf_counter()
    result, torch = dict(status='failed', allocator_conf=CONF, snapshots=[], video_id=video,
                         frame_count=frames, input_dir=str(images), model=str(MODEL),
                         reference_npz=str(case/'outputs/dataset/da3_cache/predictions.npz'),
                         scope=f'n=1 fixed video{video}/{frames} frames/504/seed42'), None
    def snap(label):
        if torch is not None and torch.cuda.is_initialized():
            segments = torch.cuda.memory_snapshot()
            result['snapshots'].append(dict(label=label, wall_s=time.perf_counter()-started,
                allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                max_allocated=torch.cuda.max_memory_allocated(), max_reserved=torch.cuda.max_memory_reserved(),
                expandable_segments=sum(bool(s.get('is_expandable', False)) for s in segments)))
    try:
        sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'Depth-Anything-3/src')]
        import torch as imported_torch
        torch = imported_torch
        props = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(CAP / props.total_memory, device=0)
        backend = torch.cuda.get_allocator_backend()
        result.update(backend=backend, cap_bytes=CAP, fraction=CAP/props.total_memory,
                      device_name=props.name, device_total_bytes=props.total_memory, physical_uuid=UUID,
                      torch_version=torch.__version__, cuda_version=torch.version.cuda)
        if backend != 'native':
            raise RuntimeError('Allocator backend is not native')
        snap('allocator_configured_before_model')
        from src import da3_runner as runner
        original = runner.load_model
        def load(*args, **kwargs):
            model = original(*args, **kwargs)
            snap('model_cpu_loaded_before_to')
            original_to = model.to
            def move(*a, **k):
                tick = time.perf_counter()
                try:
                    value = original_to(*a, **k)
                finally:
                    del model.to  # Release closure/bound method: preserve normal model lifetime.
                result['model_to_submission_wall_s'] = time.perf_counter()-tick
                snap('after_model_to')
                return value
            model.to = move
            return model
        runner.load_model = load
        tick = time.perf_counter()
        runner.main(['--input_dir', str(images), '--output_npz', str(out/'predictions.npz'),
                     '--model_path', str(MODEL), '--device', 'cuda', '--process_res', '504', '--seed', '42'])
        result.update(status='complete', runner_main_wall_s=time.perf_counter()-tick)
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        try:
            snap('process_end')
        except BaseException:
            result['snapshot_error'] = traceback.format_exc()
        result['lifecycle_wall_s'] = time.perf_counter()-started
        save(out/'worker.json', result)
    return 0 if result['status'] == 'complete' else 1


def parent(out, video):
    case, frames = CASES[video]
    images_dir = case / 'dataset/images'
    reference = case / 'outputs/dataset/da3_cache/predictions.npz'
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = dict(status='failed', budget_bytes=BUDGET, stop_line_bytes=STOP, allocator_cap_bytes=CAP,
                  video_id=video, input_dir=str(images_dir), model=str(MODEL), reference_npz=str(reference), frame_count=frames,
                  sampling_ms=50, invalid_monitor_rows=0, sampled_task_peak_bytes=0, other_resident_peak_bytes={},
                  scope=f'n=1 fixed video{video}/{frames} frames/504/seed42; sampled driver bound, not a hard guarantee')
    child = monitor = thread = log = None
    try:
        import psutil
        for path in (images_dir, MODEL, reference, PYTHON, ROOT/'src/da3_runner.py'):
            if not path.exists():
                raise FileNotFoundError(str(path))
        result['declared_frame_counts'] = {}
        for name in ('receipt.json', 'input.json'):
            declared = json.loads((case/name).read_text())
            result['declared_frame_counts'][name] = declared['frames']
            if declared['video_id'] != video or declared['frames'] != frames:
                raise ValueError(f'{case/name}: video/frame declaration differs from fixed case')
        images = [p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp') and p.is_file()]
        if len(images) != frames or not list(MODEL.glob('*.safetensors')):
            raise ValueError(f'Require the fixed {frames} image dataset and local safetensors snapshot')
        free = subprocess.check_output(['nvidia-smi', '-i', UUID, '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'], text=True, timeout=5)
        result['preflight_gpu_memory_mib'] = free.strip()
        if float(free.split(',')[0]) * 2**20 < 15*2**30:
            raise RuntimeError('Require >=15GiB free before launch')
        rows = queue.Queue()
        monitor = subprocess.Popen(['nvidia-smi', '-i', UUID, '--query-compute-apps=timestamp,pid,used_memory',
                                    '--format=csv,noheader,nounits', '--loop-ms=50'], stdout=subprocess.PIPE,
                                   stderr=(out/'monitor.stderr.log').open('w'), text=True, start_new_session=True)
        def read():
            with (out/'gpu-processes.csv').open('w') as stream:
                for line in monitor.stdout:
                    stream.write(line); stream.flush(); rows.put(line)
        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID, PYTORCH_CUDA_ALLOC_CONF=CONF,
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONHASHSEED='42',
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   UV_CACHE_DIR='/tmp/3d-profile-20261018-uv', PYTHONDONTWRITEBYTECODE='1')
        command = ['uv', 'run', '--no-project', '--python', str(PYTHON), 'python', str(Path(__file__).resolve()), '--mode', 'worker', '--video', str(video), '--out', str(out)]
        result['command'] = command
        log = (out/'worker.log').open('w')
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result.update(supervisor_child_pid=child.pid, worker_pgid=child.pid)
        known, stamp, group, samples, last_time = {child.pid}, None, {}, 0, None
        def flush():
            nonlocal samples, last_time
            if stamp is None:
                return
            samples += 1
            now = datetime.strptime(stamp, '%Y/%m/%d %H:%M:%S.%f').timestamp()
            if last_time is not None:
                result['max_observed_timestamp_gap_s'] = max(now-last_time, result.get('max_observed_timestamp_gap_s', 0))
            last_time = now
            total = sum(v for p, v in group.items() if p in known)
            result['sampled_task_peak_bytes'] = max(total, result['sampled_task_peak_bytes'])
            for pid, value in group.items():
                if pid not in known:
                    key = str(pid)
                    result['other_resident_peak_bytes'][key] = max(value, result['other_resident_peak_bytes'].get(key, 0))
            if total > STOP:
                result['status'] = 'budget_exceeded' if total > BUDGET else 'stop_line_exceeded'
                raise RuntimeError('Sampled own process group exceeded stop line')
        launched = time.monotonic()
        while child.poll() is None or not rows.empty():
            try:
                known.update(p.pid for p in psutil.Process(child.pid).children(recursive=True))
            except psutil.NoSuchProcess:
                pass
            if (out/'worker-start.json').exists():
                known.add(json.loads((out/'worker-start.json').read_text())['pid'])
            if time.monotonic()-launched > 120:
                result['status'] = 'timeout'
                raise TimeoutError('Worker deadline 120s')
            if monitor.poll() is not None:
                raise RuntimeError('GPU monitor exited before worker completion')
            try:
                row = next(csv.reader([rows.get(timeout=.02)]))
                timestamp, pid, mib = row[0].strip(), int(row[1]), float(row[2])
            except queue.Empty:
                continue
            except (ValueError, IndexError):
                result['invalid_monitor_rows'] = result.get('invalid_monitor_rows', 0)+1
                continue
            if timestamp != stamp:
                flush(); stamp, group = timestamp, {}
            group[pid] = int(mib * 2**20)
        flush()
        result.update(worker_exitcode=child.wait(), observed_task_pids=sorted(known), timestamp_groups=samples)
        log.close()
        result['worker'] = json.loads((out/'worker.json').read_text())
        unsupported = bool(re.search(r'expandable_segments[^\n]*(?:not supported|unsupported)', (out/'worker.log').read_text(), re.I))
        result['expandable_unsupported_warning'] = unsupported
        if child.returncode or result['worker']['status'] != 'complete' or unsupported:
            raise RuntimeError('Worker/configuration failed; see worker.log/worker.json')
        if result['invalid_monitor_rows']:
            result['status'] = 'invalid_monitor_evidence'
            raise RuntimeError('Invalid GPU monitor rows; budget evidence rejected, raw log retained')
        if not result['sampled_task_peak_bytes'] or not any(s['expandable_segments'] for s in result['worker']['snapshots']):
            raise RuntimeError('Missing positive task GPU samples or expandable segment evidence')
        import numpy as np
        result['numeric_outputs'] = {}
        with np.load(reference, allow_pickle=False) as before, np.load(out/'predictions.npz', allow_pickle=False) as after:
            for key in KEYS:
                a, b = before[key], after[key]
                result['numeric_outputs'][key] = dict(equal=a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True), shape=list(b.shape), dtype=str(b.dtype))
        result['status'] = 'complete' if all(v['equal'] for v in result['numeric_outputs'].values()) else 'numeric_mismatch'
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        if child is not None and result['status'] != 'complete':
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=2)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)  # Include surviving descendants after uv exits.
            except ProcessLookupError:
                pass
            child.wait()
        if child is not None:
            result['worker_exitcode'] = child.returncode
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=2)
            except subprocess.TimeoutExpired:
                monitor.kill(); monitor.wait()
            if thread is not None:
                thread.join(timeout=2)
        if log is not None:
            log.close()
        result['parent_wall_s'] = time.perf_counter()-started
        save(out/'result.json', result)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('parent', 'worker'), default='parent')
    parser.add_argument('--video', type=int, choices=(3, 6), default=3)
    args = parser.parse_args()
    sys.exit((parent if args.mode == 'parent' else worker)(args.out.resolve(), args.video))
