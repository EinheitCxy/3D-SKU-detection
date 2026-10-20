"""Separate diagnostic run: source spans, CUDA events, CPU and kernel profiles.

No production files are edited. CUDA events are resolved after the request,
without adding a synchronize at every submodule boundary. Instrumented results
are not substituted for the independent low-overhead baseline.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from contextlib import contextmanager
import cProfile
import functools
import inspect
import io
import json
from pathlib import Path
import pstats
import re
import resource
import sys
import time

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/3d/scanner"))
sys.path.insert(1, str(ROOT))
from profile_candidates import parse_args


class Detail:
    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.events, self.restores, self.handles = [], [], []
        self.phase = "setup"
        self.stack = []

    @contextmanager
    def span(self, name, gpu=False, **fields):
        gpu = gpu and self.torch.cuda.is_initialized()
        start_event = end_event = None
        if gpu:
            start_event = self.torch.cuda.Event(enable_timing=True)
            end_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        row = {"name": name, "phase": self.phase, "start": start,
               "parent": self.stack[-1] if self.stack else None, **fields}
        self.stack.append(name)
        try:
            yield
        finally:
            row["end"] = time.perf_counter()
            row["wall_s"] = row["end"] - start
            if gpu:
                end_event.record()
                row["_gpu"] = (start_event, end_event)
            self.events.append(row)
            self.stack.pop()

    def call(self, name, function, *args, gpu=False, **kwargs):
        with self.span(name, gpu=gpu):
            return function(*args, **kwargs)

    def wrap(self, owner, attr, name, gpu=False):
        original = getattr(owner, attr)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            with self.span(name, gpu=gpu):
                return original(*args, **kwargs)

        setattr(owner, attr, measured)
        self.restores.append((owner, attr, original))

    @staticmethod
    def shapes(value, limit=4):
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, (tuple, list)):
            return [Detail.shapes(x, limit) for x in value[:limit]]
        if isinstance(value, dict):
            return {str(k): Detail.shapes(v, limit) for k, v in list(value.items())[:limit]}
        return type(value).__name__

    def module(self, name, module):
        contexts = []

        def before(_module, inputs):
            context = self.span("module." + name, gpu=True, inputs=self.shapes(inputs))
            context.__enter__()
            contexts.append(context)

        def after(_module, inputs, output):
            contexts.pop().__exit__(None, None, None)

        self.handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))

    def resolve(self):
        self.torch.cuda.synchronize(self.device)
        for row in self.events:
            if "_gpu" in row:
                begin, end = row.pop("_gpu")
                row["cuda_stream_ms"] = begin.elapsed_time(end)

    def restore(self):
        for handle in self.handles:
            handle.remove()
        for owner, attr, original in reversed(self.restores):
            setattr(owner, attr, original)
        self.handles.clear()
        self.restores.clear()

    def summary(self):
        groups = defaultdict(list)
        for row in self.events:
            groups[(row["phase"], row["name"])].append(row)
        result = []
        for (phase, name), rows in groups.items():
            times = [row["wall_s"] for row in rows]
            item = {"phase": phase, "name": name, "count": len(rows),
                    "wall_sum_s": sum(times), "wall_min_s": min(times),
                    "wall_max_s": max(times)}
            cuda = [row["cuda_stream_ms"] for row in rows if "cuda_stream_ms" in row]
            if cuda:
                item["cuda_stream_sum_ms"] = sum(cuda)
            result.append(item)
        return sorted(result, key=lambda row: (row["phase"], -row["wall_sum_s"]))


def instrument_lines(module, function_name, detail):
    """Wrap executable native statements, retaining source locations and flow."""
    function = getattr(module, function_name)
    text = Path(inspect.getsourcefile(function)).read_text()
    tree = ast.parse(text)
    target = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    lines = text.splitlines()

    class Instrument(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if function_name == "predict_promptda" and isinstance(node.func, ast.Attribute):
                if node.func.attr == "to" and isinstance(node.func.value, ast.Name) and node.func.value.id in ("image", "depth_input"):
                    label = "transfer.h2d." + node.func.value.id
                elif node.func.attr == "cpu":
                    label = "transfer.d2h.depth"
                else:
                    return node
                return ast.copy_location(ast.Call(
                    func=ast.Attribute(value=ast.Name(id="_profile", ctx=ast.Load()), attr="call", ctx=ast.Load()),
                    args=[ast.Constant(label), node.func, *node.args],
                    keywords=[*node.keywords, ast.keyword(arg="gpu", value=ast.Constant(True))]), node)
            return node

        def generic_visit(self, node):
            node = super().generic_visit(node)
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.Expr, ast.Return, ast.With)):
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    return node
                label = f"line.{module.__name__}.{function_name}:{node.lineno} {lines[node.lineno - 1].strip()[:110]}"
                wrapped = ast.With(items=[ast.withitem(context_expr=ast.Call(
                    func=ast.Attribute(value=ast.Name(id="_profile", ctx=ast.Load()), attr="span", ctx=ast.Load()),
                    args=[ast.Constant(label)], keywords=[]))], body=[node])
                return ast.copy_location(wrapped, node)
            return node

    module.__dict__["_profile"] = detail
    target = Instrument().visit(target)
    ast.fix_missing_locations(target)
    exec(compile(ast.Module(body=[target], type_ignores=[]), inspect.getsourcefile(function), "exec"), module.__dict__)
    detail.restores.append((module, function_name, function))


def profile_rows(profile, count=100):
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _callers) in stats.stats.items():
        rows.append({"file": filename, "line": line, "function": name,
                     "primitive_calls": primitive, "calls": calls,
                     "self_s": own, "cumulative_s": cumulative})
    return sorted(rows, key=lambda row: row["cumulative_s"], reverse=True)[:count]


def main():
    args = parse_args()
    import numpy as np
    import torch
    import candidates
    import scanner_io
    import safetensors.torch

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "diagnostic.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    torch.set_num_threads(8)
    device = torch.device(args.device)
    detail = Detail(torch, device)
    report = {"candidate": args.candidate, "status": "running", "events": [],
              "timing_note": "Independent diagnostic run. Nested inclusive wall and CUDA-stream intervals must not be added across parents/children. Event timings include stream idle/launch gaps, not only kernel active time. No tensors retained by hooks."}
    try:
        with detail.span("setup.prepared_input_load"):
            images, sensor = scanner_io.load_prepared(args.base)
        report["frame_count"] = len(images)
        report["source_image_size"] = sensor["image_size"].tolist()
        detail.wrap(torch, "load", "load.torch_load_cpu_checkpoint")
        detail.wrap(torch.nn.Module, "load_state_dict", "load.state_assignment")
        detail.wrap(torch.nn.Module, "to", "load.module_to")
        detail.wrap(safetensors.torch, "load_model", "load.safetensors_load_model")
        detail.wrap(safetensors.torch, "load_file", "load.safetensors_load_file")
        if args.candidate == "da3":
            sys.path.insert(0, str(ROOT / "src"))
            sys.path.insert(0, str(ROOT / "Depth-Anything-3/src"))
            from depth_anything_3.api import DepthAnything3
            import da3_model_loading
            detail.wrap(DepthAnything3, "__init__", "load.cpu_network_construct")
            detail.wrap(da3_model_loading, "audit_checkpoint", "load.checkpoint_coverage_audit")
        else:
            sys.path.insert(0, str(args.promptda_repo))
            from promptda.promptda import PromptDA
            from promptda.model.dpt import DPTHead
            detail.wrap(torch.hub, "load", "load.dinov2_network_construct")
            detail.wrap(DPTHead, "__init__", "load.dpt_head_construct")
            detail.wrap(PromptDA, "__init__", "load.cpu_network_construct_and_checkpoint")
            detail.wrap(PromptDA, "load_checkpoint", "load.promptda_checkpoint")
        detail.phase = "model_load"
        cpu_load = cProfile.Profile()
        begin = time.perf_counter()
        cpu_load.enable()
        model = candidates.load_candidate(args.candidate, args.checkpoint, args.device, args.promptda_repo,
                                          promptda_matmul_precision=args.promptda_matmul_precision)
        torch.cuda.synchronize(device)
        cpu_load.disable()
        report["instrumented_model_load_wall_s"] = time.perf_counter() - begin
        report["cpu_model_load_top"] = profile_rows(cpu_load)
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        report["parameter_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
        report["module_tree"] = [{"name": name, "type": type(module).__name__}
                                  for name, module in model.named_modules() if name.count(".") < 3]
        detail.restore()
        prediction_options = dict(promptda_batch_size=args.promptda_batch_size,
                                  promptda_precision=args.promptda_precision,
                                  promptda_max_size=args.promptda_max_size)
        detail.phase = "warmup"
        begin = time.perf_counter()
        warmup = candidates.predict_candidate(args.candidate, model, images, sensor, args.device, **prediction_options)
        torch.cuda.synchronize(device)
        report["unhooked_warmup_wall_s"] = time.perf_counter() - begin
        del warmup
        if args.candidate == "da3":
            from depth_anything_3.model.da3 import NestedDepthAnything3Net
            for method in ("_preprocess_inputs", "_prepare_model_inputs", "_normalize_extrinsics", "_run_model_forward",
                           "_convert_to_prediction", "_align_to_input_extrinsics_intrinsics", "_add_processed_images"):
                detail.wrap(type(model), method, "request.da3." + method,
                            gpu=method in ("_prepare_model_inputs", "_normalize_extrinsics", "_run_model_forward", "_convert_to_prediction"))
            for method in ("_apply_metric_scaling", "_apply_depth_alignment", "_handle_sky_regions"):
                detail.wrap(NestedDepthAnything3Net, method, "request.da3." + method, gpu=True)
        else:
            instrument_lines(candidates, "predict_promptda", detail)
            detail.wrap(model, "normalize", "request.promptda.lidar_minmax_normalize", gpu=True)
            detail.wrap(model, "denormalize", "request.promptda.metric_denormalize", gpu=True)
            detail.wrap(model, "predict", "request.promptda.predict", gpu=True)
            detail.wrap(model.pretrained, "get_intermediate_layers", "request.promptda.rgb_encoder", gpu=True)
        for name, module in model.named_modules():
            if args.candidate == "da3":
                selected = name in {"model", "model.da3", "model.da3_metric", "model.da3.backbone", "model.da3_metric.backbone",
                                    "model.da3.head", "model.da3_metric.head", "model.da3.cam_enc", "model.da3.cam_dec"}
            else:
                selected = (name == "depth_head" or name.startswith("depth_head.") and name.count(".") <= 2)
            selected |= bool(re.search(r"\.blocks\.\d+(?:\.(?:attn|mlp))?$", name))
            if selected:
                detail.module(name, module)
        detail.phase = "instrumented_request"
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        cpu_request = cProfile.Profile()
        begin = time.perf_counter()
        cpu_request.enable()
        prediction = candidates.predict_candidate(args.candidate, model, images, sensor, args.device, **prediction_options)
        torch.cuda.synchronize(device)
        cpu_request.disable()
        report["instrumented_request_wall_s"] = time.perf_counter() - begin
        report["cpu_request_top"] = profile_rows(cpu_request)
        report["processed_shape"] = list(prediction["depth"].shape)
        report["gpu_request_allocated_peak_bytes"] = torch.cuda.max_memory_allocated(device)
        report["gpu_request_reserved_peak_bytes"] = torch.cuda.max_memory_reserved(device)
        detail.resolve()
        detail.restore()
        detail.phase = "instrumented_points_export"
        instrument_lines(scanner_io, "export_prediction", detail)
        instrument_lines(scanner_io, "backproject", detail)
        detail.wrap(scanner_io, "prediction_mask", "points.validity_and_confidence_mask")
        cpu_export = cProfile.Profile()
        begin = time.perf_counter()
        cpu_export.enable()
        report["export"] = scanner_io.export_prediction(prediction, sensor, args.output / "diagnostic_points.glb")
        cpu_export.disable()
        report["instrumented_points_export_wall_s"] = time.perf_counter() - begin
        report["cpu_export_top"] = profile_rows(cpu_export)
        detail.restore()
        del prediction
        detail.phase = "operator_profile"
        begin = time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=False, with_stack=False) as operators:
            prediction = candidates.predict_candidate(args.candidate, model, images, sensor, args.device, **prediction_options)
            torch.cuda.synchronize(device)
        report["operator_profile_wall_s"] = time.perf_counter() - begin
        rows = []
        for event in operators.key_averages(group_by_input_shape=True):
            rows.append({"name": event.key, "calls": event.count,
                         "self_cpu_us": event.self_cpu_time_total, "inclusive_cpu_us": event.cpu_time_total,
                         "self_device_us": event.self_device_time_total, "inclusive_device_us": event.device_time_total,
                         "input_shapes": event.input_shapes})
        report["operator_rows"] = sorted(rows, key=lambda row: row["self_device_us"], reverse=True)
        del prediction
        report["status"] = "complete"
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["status"] = "failed"
        raise
    finally:
        detail.resolve()
        detail.restore()
        report["events"] = detail.events
        report["summary"] = detail.summary()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report["host_process_lifetime_peak_rss_kib"] = usage.ru_maxrss
        report["cpu_user_s"] = usage.ru_utime
        report["cpu_system_s"] = usage.ru_stime
        report["script_wall_before_report_s"] = time.perf_counter() - START
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("candidate", "status", "processed_shape", "instrumented_request_wall_s", "instrumented_points_export_wall_s")}))


if __name__ == "__main__":
    main()
