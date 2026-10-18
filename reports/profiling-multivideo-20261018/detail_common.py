"""Shared in-run instrumentation. AST/observations/events add unquantified overhead.

Host wall and CUDA stream event intervals are separate and nested. This is an
instrumented pipeline, not a measured low-overhead production baseline.
"""
from __future__ import annotations
import atexit
from collections import defaultdict
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['PROFILE_OUT']).resolve()
OUT.mkdir(parents=True, exist_ok=True)
LOCK = threading.RLock()
LOCAL = threading.local()
START = time.perf_counter()
WALL = defaultdict(lambda: dict(inclusive_s=0., self_s=0., calls=0, max_s=0., thread_cpu_ns=0))
CONTEXTS = defaultdict(lambda: dict(inclusive_s=0., self_s=0., calls=0, max_s=0., thread_cpu_ns=0))
MEMORY = []
DATA = {'instrumentation': 'In-run AST statements, scalar observations, host spans and CUDA events; overhead unquantified. No extra synchronize or allocator peak resets.',
        'wall_semantics': 'Thread-local nested host wall; self subtracts measured direct children only. Thread CPU excludes intra-op workers. CUDA events are separate nested stream intervals, not exclusive kernel times.'}


def current_context():
    return ' > '.join(row[2] for row in getattr(LOCAL, 'stack', []))


@contextmanager
def span(name):
    stack = getattr(LOCAL, 'stack', None)
    if stack is None:
        stack = LOCAL.stack = []
    context = ' > '.join([row[2] for row in stack] + [name])
    row = [time.perf_counter(), 0., name, time.thread_time_ns()]
    stack.append(row)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - row[0]
        cpu = time.thread_time_ns() - row[3]
        stack.pop()
        if stack:
            stack[-1][1] += elapsed
        with LOCK:
            for value in (WALL[name], CONTEXTS[context]):
                value['inclusive_s'] += elapsed
                value['self_s'] += max(0., elapsed-row[1])
                value['calls'] += 1
                value['max_s'] = max(value['max_s'], elapsed)
                value['thread_cpu_ns'] += cpu


def record(name, start, end, **extra):
    row = dict(name=name, start=start, end=end, seconds=end-start,
               pid=os.getpid(), thread=threading.current_thread().name, **extra)
    with LOCK, (OUT / f'events-{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')


def wrap(owner, attr, name=None, fields=None):
    original = getattr(owner, attr)
    label = name or attr
    @functools.wraps(original)
    def measured(*args, **kwargs):
        begin = time.perf_counter()
        try:
            with span(label):
                result = original(*args, **kwargs)
        except BaseException as error:
            record(label, begin, time.perf_counter(), error=repr(error))
            raise
        end = time.perf_counter()
        record(label, begin, end, **(fields(result) if fields else {}))
        return result
    setattr(owner, attr, measured)


def memory_snapshot(label, model=None):
    """Inspect allocator without initializing CUDA/synchronizing; retain scalars only."""
    torch = sys.modules.get('torch')
    row = dict(label=label, t=time.perf_counter(), thread=threading.current_thread().name,
               cuda_initialized=bool(torch is not None and torch.cuda.is_initialized()))
    if row['cuda_initialized']:
        row.update(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if model is not None:
        row['gpu_parameter_bytes'] = sum(p.numel()*p.element_size() for p in model.parameters() if p.device.type=='cuda')
        row['gpu_buffer_bytes'] = sum(p.numel()*p.element_size() for p in model.buffers() if p.device.type=='cuda')
    with LOCK:
        MEMORY.append(row)
    return row


def save_outputs():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    process = dict(pid=os.getpid(), argv=sys.argv, elapsed_s=time.perf_counter()-START,
                   rss_hwm_kib=usage.ru_maxrss, cpu_user_s=usage.ru_utime, cpu_system_s=usage.ru_stime,
                   peak_scope='process lifetime; no allocator resets')
    torch = sys.modules.get('torch')
    if torch is not None:
        process.update(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads())
        if torch.cuda.is_initialized():
            process.update(cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                           cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                           cuda_allocated_final_bytes=torch.cuda.memory_allocated())
    with LOCK:
        detail = dict(DATA, pid=os.getpid(), elapsed_s=process['elapsed_s'], wall=dict(WALL),
                      wall_contexts=dict(CONTEXTS), memory_snapshots=list(MEMORY), process=process)
        (OUT / f'detail-{os.getpid()}.json').write_text(json.dumps(detail, indent=2))
        (OUT / f'process-{os.getpid()}.json').write_text(json.dumps(process, indent=2))


atexit.register(save_outputs)
