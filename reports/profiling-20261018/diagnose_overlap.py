"""One bounded replay of CPU Viewer preparation overlapping native PVS loading.

No predict, cache publication, or extra CUDA synchronization. The coordinator
must enforce timeout externally; SIGTERM writes a partial result when Python
can dispatch its handler. A prior snapshot survives even a forced SIGKILL.
Thread CPU/schedstat/stack observations do not establish GIL causality.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import functools
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
LOCAL = threading.local()
LOCK = threading.RLock()
STATS = defaultdict(lambda: dict(wall_s=0.0, self_wall_s=0.0, thread_cpu_ns=0,
                                 self_thread_cpu_ns=0, calls=0, max_wall_s=0.0))
MAPPING = {}


@contextmanager
def span(name):
    stack = getattr(LOCAL, 'stack', None)
    if stack is None:
        stack = LOCAL.stack = []
    context = ' > '.join([entry['name'] for entry in stack] + [name])
    entry = dict(name=name, start=time.perf_counter(), cpu=time.thread_time_ns(), children_wall=0.0, children_cpu=0)
    stack.append(entry)
    try:
        yield
    finally:
        wall = time.perf_counter() - entry['start']
        cpu = time.thread_time_ns() - entry['cpu']
        stack.pop()
        if stack:
            stack[-1]['children_wall'] += wall
            stack[-1]['children_cpu'] += cpu
        with LOCK:
            row = STATS[context]
            row['wall_s'] += wall
            row['self_wall_s'] += max(0.0, wall - entry['children_wall'])
            row['thread_cpu_ns'] += cpu
            row['self_thread_cpu_ns'] += max(0, cpu - entry['children_cpu'])
            row['calls'] += 1
            row['max_wall_s'] = max(row['max_wall_s'], wall)


def wrap(owner, attr, name):
    original = getattr(owner, attr)

    @functools.wraps(original)
    def measured(*args, **kwargs):
        with span(name):
            return original(*args, **kwargs)

    setattr(owner, attr, measured)


def instrument_load(path):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != 'load_model':
            continue
        body = []
        for statement in node.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                body.append(statement)
                continue
            name = f'pvs.load_model:L{statement.lineno}'
            MAPPING[name] = dict(path=str(path.relative_to(ROOT)), line=statement.lineno,
                                 end_line=statement.end_lineno,
                                 statement=ast.get_source_segment(source, statement).splitlines()[0][:220])
            wrapped = ast.With(items=[ast.withitem(context_expr=ast.Call(
                func=ast.Name(id='_overlap_span', ctx=ast.Load()), args=[ast.Constant(name)], keywords=[]))], body=[statement])
            body.append(ast.copy_location(wrapped, statement))
        node.body = body
    ast.fix_missing_locations(tree)
    name = 'utils.diagnostic_overlap_pvs'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module._overlap_span = span
    sys.modules[name] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


def sample_threads(path, stop):
    """Keep scheduler counters and Python frames only; never read frame locals."""
    with path.open('a', buffering=1) as stream:
        while not stop.is_set():
            frames = sys._current_frames()
            python_threads = {thread.ident: thread for thread in threading.enumerate()}
            stacks = {}
            for ident, frame in frames.items():
                thread = python_threads.get(ident)
                if thread is None or thread.native_id is None:
                    continue
                entries = []
                while frame is not None:
                    entries.append(dict(function=frame.f_code.co_name, path=frame.f_code.co_filename,
                                        line=frame.f_lineno))
                    frame = frame.f_back
                stacks[thread.native_id] = dict(name=thread.name, frames=entries)
            del frames, frame
            rows = []
            for task in Path('/proc/self/task').iterdir():
                try:
                    values = [int(value) for value in (task / 'schedstat').read_text().split()]
                    row = dict(tid=int(task.name), cpu_runtime_ns=values[0], runqueue_wait_ns=values[1], timeslices=values[2])
                    if int(task.name) in stacks:
                        row['python'] = stacks[int(task.name)]
                    rows.append(row)
                except (FileNotFoundError, ProcessLookupError):
                    continue
            stream.write(json.dumps(dict(t=time.perf_counter(), threads=rows)) + '\n')
            stop.wait(0.2)


def instrument_sor(filters):
    # Patch just the Python caller, preserving the original lazy Open3D import.
    path = ROOT / 'utils/pointcloud_filter.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'prepare_scene_filter')

    class Transformer(ast.NodeTransformer):
        def visit_Assign(self, node):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'remove_statistical_outlier':
                wrapped = ast.With(items=[ast.withitem(context_expr=ast.Call(
                    func=ast.Name(id='_overlap_span', ctx=ast.Load()), args=[ast.Constant('SOR')], keywords=[]))], body=[node])
                return ast.copy_location(wrapped, node)
            return node

    module = ast.Module(body=[Transformer().visit(function)], type_ignores=[])
    ast.fix_missing_locations(module)
    filters._overlap_span = span
    exec(compile(module, str(path), 'exec'), filters.__dict__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'runtime/profiling-20261018')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.base, args.out = args.base.resolve(), args.out.resolve()
    if args.out.exists():
        raise FileExistsError(f'Use a new output directory: {args.out}')
    args.out.mkdir(parents=True)
    start = time.perf_counter()
    detail = dict(status='starting', pid=os.getpid(), argv=sys.argv,
                  semantics='Independent overlap diagnosis, no inference/publication; wall/thread CPU differences and stacks do not prove GIL contention. Per-thread CPU excludes intra-op worker CPU. Sample period 200ms nominal; Python/GIL scheduling can delay sampling.',
                  environment={key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                  allocator_scope='process lifetime, no peak resets', sampling_path=str(args.out / 'thread-samples.jsonl'))
    torch = None
    stop = threading.Event()
    sampler = None
    executor = None

    def save():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        with LOCK:
            timers = {name: dict(value) for name, value in STATS.items()}
        row = dict(detail, elapsed_s=time.perf_counter()-start, timers=timers, source_mapping=MAPPING,
                   rss_hwm_kib=usage.ru_maxrss, process_cpu_user_s=usage.ru_utime, process_cpu_system_s=usage.ru_stime)
        if torch is not None:
            row.update(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                       cuda_initialized=torch.cuda.is_initialized())
            if torch.cuda.is_initialized():
                row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                           final_allocated_bytes=torch.cuda.memory_allocated())
        temporary = args.out / 'result.json.partial'
        temporary.write_text(json.dumps(row, indent=2))
        os.replace(temporary, args.out / 'result.json')

    def terminate(signum, frame):
        detail.update(status='terminated', signal=signum, interrupted_stack=' > '.join(
            entry['name'] for entry in getattr(LOCAL, 'stack', [])),
            limitation='SIGTERM snapshot may contain active spans only in stack samples; completed timer aggregates exclude unfinished spans.')
        save()
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    save()
    try:
        sys.path[:0] = [str(ROOT), str(ROOT / 'sam31'), str(ROOT / 'Depth-Anything-3/src')]
        with span('imports'):
            import torch as torch_module
            torch = torch_module
            import src.web_viewer_export as viewer
            import utils.pointcloud_filter as filters
            pvs = instrument_load(ROOT / 'utils/sam31_pvs.py')
        # Match main profile: environment controls intra-op settings; do not call
        # torch.set_num_threads here, which would change the replay conditions.
        detail.update(status='running', torch_threads=torch.get_num_threads(),
                      torch_interop_threads=torch.get_num_interop_threads())
        instrument_sor(filters)
        viewer.prepare_scene_filter = filters.prepare_scene_filter
        wrap(viewer, 'prepare_scene_filter', 'prepare_scene_filter')
        wrap(filters, '_median_nn_distance', 'NN')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for native PVS load')
        sampler = threading.Thread(target=sample_threads, args=(args.out / 'thread-samples.jsonl', stop),
                                   name='overlap-sampler', daemon=True)
        sampler.start()
        detail['replay_limits'] = ('Fresh process and already resident OS file pages; no prior DA3 subprocess/matching scene H2D, '
                                   'reference data loading, or source prefetch replay. These differences can affect initialization.')
        save()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='surfel-prepare')

        def prepare():
            with span('prepare_web_viewer_geometry'):
                return viewer.prepare_web_viewer_geometry(
                    da3_cache_path=args.base / 'outputs/dataset/da3_cache/predictions.npz',
                    source_images_dir=args.base / 'dataset/images', surfel_texture_edge=1920)

        future = executor.submit(prepare)
        with span('pvs_model_load'):
            model, count = pvs.load_model(ROOT / 'sam31/checkpoints/sam3.1_multiplex.pt', device='cuda:0')
        detail['checkpoint_tensors'] = count
        prepared = future.result()
        detail.update(status='success', prepare_complete=True, pvs_load_complete=True,
                      prepared_shape=list(prepared.cache['points'].shape))
        del prepared, model
    except BaseException as error:
        detail.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        save()
        stop.set()
        if sampler is not None:
            sampler.join(timeout=1)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        save()
        print(json.dumps(dict(status=detail['status'], result=str(args.out / 'result.json'))), flush=True)


if __name__ == '__main__':
    main()
