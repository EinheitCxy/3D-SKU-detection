"""Low-overhead wall spans for one local profiling run; no production edits."""
import atexit
import functools
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PROFILE_OUT", ROOT / "runtime/profiling-20261018"))
OUT.mkdir(parents=True, exist_ok=True)
EVENTS = OUT / f"events-{os.getpid()}.jsonl"
LOCK = threading.Lock()


def record(name, start, end, **extra):
    row = dict(name=name, start=start, end=end, seconds=end-start,
               pid=os.getpid(), thread=threading.current_thread().name, **extra)
    with LOCK, EVENTS.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def wrap(owner, attr, name=None, fields=None):
    original = getattr(owner, attr)

    @functools.wraps(original)
    def measured(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            record(name or attr, start, time.perf_counter(), error=repr(error))
            raise
        end = time.perf_counter()
        record(name or attr, start, end, **(fields(result) if fields else {}))
        return result

    setattr(owner, attr, measured)


def save_process():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    data = dict(pid=os.getpid(), argv=sys.argv, rss_hwm_kib=usage.ru_maxrss,
                cpu_user_s=usage.ru_utime, cpu_system_s=usage.ru_stime,
                peak_scope="process lifetime; no allocator peak resets")
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        data.update(cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                    cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                    cuda_allocated_final_bytes=torch.cuda.memory_allocated(),
                    cuda_reserved_final_bytes=torch.cuda.memory_reserved(),
                    cuda_logical_device=torch.cuda.current_device(),
                    cuda_name=torch.cuda.get_device_name())
    (OUT / f"process-{os.getpid()}.json").write_text(json.dumps(data, indent=2))


atexit.register(save_process)
