"""DA3 child hook, used through DA3_VENV_PYTHON by the measured core run."""
import importlib.util
import sys
import time
from profile_common import OUT, ROOT, record, wrap

start = time.perf_counter()
script, *arguments = sys.argv[1:]
spec = importlib.util.spec_from_file_location("profiled_da3_runner", script)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from depth_anything_3.api import DepthAnything3
record("da3_imports", start, time.perf_counter())
original = DepthAnything3.from_pretrained


def load(cls, *args, **kwargs):
    begin = time.perf_counter()
    result = original(*args, **kwargs)
    record("da3_load_cpu", begin, time.perf_counter())
    return result


DepthAnything3.from_pretrained = classmethod(load)
wrap(DepthAnything3, "inference", "da3_inference_api")
wrap(runner, "_depth_to_world_points", "da3_backprojection")
wrap(runner.np, "savez", "da3_npz_write")
start = time.perf_counter()
runner.main(arguments)
record("da3_runner_main", start, time.perf_counter())
