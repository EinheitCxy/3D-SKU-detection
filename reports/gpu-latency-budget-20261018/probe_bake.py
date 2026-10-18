"""Run one bounded Torch bake probe on GPU2 with external driver-memory sampling."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback

from probe_da3 import BUDGET, CONF, PYTHON, ROOT, STOP, UUID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('sample', 'glb'), required=True)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--reference-glb', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'glb' and args.reference_glb is None:
        parser.error('Full export requires a native reference GLB')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status='failed', mode=args.mode, generation=str(args.generation.resolve()),
                  budget_bytes=BUDGET, stop_line_bytes=STOP, sampling_ms=50,
                  sampled_task_peak_bytes=0, invalid_monitor_rows=0,
                  timestamp_groups=0, other_resident_peak_bytes={},
                  scope='Owned GPU PIDs summed at the same timestamp; sampled bound, not an instantaneous hard guarantee')
    began = time.monotonic()
    def terminate(_signum, _frame):
        raise InterruptedError('Supervisor received SIGTERM')

    signal.signal(signal.SIGTERM, terminate)
    child = monitor = reader = None
    known, stamp, group, previous = set(), None, {}, None
    rows = queue.Queue()
    reader_errors = queue.Queue()
    reader_state = {'last_arrival': None}
    last_arrival = None

    def flush():
        nonlocal previous
        if stamp is None:
            return
        result['timestamp_groups'] += 1
        current = datetime.strptime(stamp, '%Y/%m/%d %H:%M:%S.%f').timestamp()
        if previous is not None:
            result['max_observed_timestamp_gap_s'] = max(
                current - previous, result.get('max_observed_timestamp_gap_s', 0))
        previous = current
        total = sum(value for pid, value in group.items() if pid in known)
        result['sampled_task_peak_bytes'] = max(result['sampled_task_peak_bytes'], total)
        for pid, value in group.items():
            if pid not in known:
                key = str(pid)
                result['other_resident_peak_bytes'][key] = max(
                    value, result['other_resident_peak_bytes'].get(key, 0))
        if total > STOP:
            result['status'] = 'budget_exceeded' if total > BUDGET else 'stop_line_exceeded'
            raise RuntimeError('Owned GPU processes exceeded the stop line')

    try:
        import psutil
        free = subprocess.check_output(
            ['nvidia-smi', '-i', UUID, '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            text=True, timeout=5)
        result['preflight_free_mib'] = float(free.strip())
        if result['preflight_free_mib'] < 2048:
            raise RuntimeError('Require 2GiB physical headroom for bounded bake')
        command = ['uv', 'run', '--no-project', '--python', str(PYTHON), 'python',
                   str(ROOT/'reports/torch-bake-fusion-20261018/probe.py'),
                   '--mode', args.mode, '--device', 'cuda:0',
                   '--generation', str(args.generation.resolve()), '--out', str(out/'probe'),
                   '--batch-size', '1024']
        if args.mode == 'sample':
            command += ['--frames', '48', '--points-per-frame', '1024']
        else:
            command += ['--reference-glb', str(args.reference_glb.resolve()), '--encoder-workers', '2']
        result['command'] = command
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID, PYTORCH_CUDA_ALLOC_CONF=CONF,
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   UV_CACHE_DIR='/tmp/3d-profile-20261018-uv', PYTHONDONTWRITEBYTECODE='1')
        with (out/'monitor.stderr.log').open('w') as monitor_error, (out/'worker.log').open('w') as log:
            monitor = subprocess.Popen(
                ['nvidia-smi', '-i', UUID, '--query-compute-apps=timestamp,pid,used_memory',
                 '--format=csv,noheader,nounits', '--loop-ms=50'], stdout=subprocess.PIPE,
                stderr=monitor_error, text=True, start_new_session=True)

            def read():
                try:
                    with (out/'gpu-processes.csv').open('w') as stream:
                        for line in monitor.stdout:
                            stream.write(line)
                            stream.flush()
                            arrived = time.monotonic()
                            reader_state['last_arrival'] = arrived
                            rows.put((arrived, line))
                except BaseException:
                    reader_errors.put(traceback.format_exc())

            reader = threading.Thread(target=read, daemon=True)
            reader.start()
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            known.add(child.pid)
            result['worker_pgid'] = child.pid
            launched = time.monotonic()
            while child.poll() is None or not rows.empty():
                try:
                    known.update(p.pid for p in psutil.Process(child.pid).children(recursive=True))
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() - began > 300:
                    result['status'] = 'timeout'
                    raise TimeoutError('Bake probe deadline 300s')
                if monitor.poll() is not None:
                    raise RuntimeError('GPU monitor exited before completion')
                if not reader_errors.empty():
                    raise RuntimeError('GPU sample reader failed: '+reader_errors.get())
                # Queued fresh rows may not have been consumed after a scheduler pause.
                if rows.empty() and time.monotonic() - (reader_state['last_arrival'] or launched) > 1:
                    result['status'] = 'invalid_monitor_evidence'
                    raise RuntimeError('No fresh GPU samples for more than 1s')
                try:
                    arrived, line = rows.get(timeout=.02)
                    row = next(csv.reader([line]))
                    timestamp, pid, mib = row[0].strip(), int(row[1]), float(row[2])
                except queue.Empty:
                    continue
                except (ValueError, IndexError):
                    result['invalid_monitor_rows'] += 1
                    continue
                last_arrival = arrived
                result['max_sample_consume_delay_s'] = max(
                    time.monotonic()-arrived, result.get('max_sample_consume_delay_s', 0))
                result.setdefault('first_valid_sample_elapsed_s', arrived-launched)
                result['last_valid_sample_elapsed_s'] = arrived-launched
                if timestamp != stamp:
                    flush()
                    stamp, group = timestamp, {}
                group[pid] = int(mib * 2**20)
            flush()
            result['worker_lifetime_s'] = time.monotonic()-launched
            result['last_sample_age_at_exit_s'] = time.monotonic()-(last_arrival or launched)
            result['worker_exitcode'] = child.wait()
        result['probe'] = json.loads((out/'probe/result.json').read_text())
        if child.returncode or result['probe']['status'] != 'complete':
            raise RuntimeError('Bake worker failed; see worker.log and probe/result.json')
        if (result['invalid_monitor_rows'] or not result['sampled_task_peak_bytes']
                or result['first_valid_sample_elapsed_s'] > 1
                or result['last_sample_age_at_exit_s'] > 1
                or result['max_observed_timestamp_gap_s'] > 1):
            result['status'] = 'invalid_monitor_evidence'
            raise RuntimeError('Missing or invalid task GPU samples')
        result['status'] = 'complete'
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        if child is not None and result['status'] != 'complete':
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=2)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=2)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
            if reader is not None:
                reader.join(timeout=2)
        result['observed_task_pids'] = sorted(known)
        result['supervisor_wall_s'] = time.monotonic() - began
        (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({key: result.get(key) for key in ('status', 'sampled_task_peak_bytes', 'worker_exitcode')}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
