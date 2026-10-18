"""Time the normal or optimized DA3 runner; no new inference synchronizations."""
import importlib.util
import json
import sys
import time

from detail_common import ROOT, OUT, DATA, record, span, wrap, memory_snapshot, save_outputs

sys.path[:0] = [str(ROOT), str(ROOT / "Depth-Anything-3/src")]
start = time.perf_counter()
import torch
from depth_anything_3.api import DepthAnything3
import safetensors.torch

script, *arguments = sys.argv[1:]
spec = importlib.util.spec_from_file_location("video3_da3_runner", script)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
record("da3_imports_and_instrumentation", start, time.perf_counter())
for owner, name, label in [
    (DepthAnything3, "__init__", "da3.network_constructor"),
    (DepthAnything3, "load_state_dict", "da3.state_assignment"),
    (DepthAnything3, "to", "da3.model_to_device"),
    (safetensors.torch, "load_file", "da3.safetensors_load_file"),
    (safetensors.torch, "load_model", "da3.safetensors_load_model"),
    (runner, "_depth_to_world_points", "da3.CPU_backprojection"),
    (runner.np, "savez", "da3.NPZ_write"),
]:
    wrap(owner, name, label)
for name in ("inference", "_preprocess_inputs", "_prepare_model_inputs", "_run_model_forward",
             "_convert_to_prediction", "_add_processed_images"):
    wrap(DepthAnything3, name, "da3." + name)
pretrained = DepthAnything3.from_pretrained.__func__
def load(cls, *args, **kwargs):
    with span("da3.load_cpu"):
        return pretrained(cls, *args, **kwargs)
DepthAnything3.from_pretrained = classmethod(load)
DATA.update(profile_scope="DA3 child of paired video3 run; function-level spans; fixed seed42.",
            code_root=str(ROOT), status="running")
started = time.perf_counter()
try:
    runner.main([*arguments, "--seed", "42"])
    record("da3_runner_main", started, time.perf_counter())
    DATA["status"] = "success"
finally:
    memory_snapshot("da3.process_end")
    save_outputs()
