"""Independent GPU component diagnosis; run only after the main profile completes.

All host timers are inclusive wall spans. CUDA events are separate stream spans.
This program never deletes its output NPZ or writes into the main run's cache.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager
import functools
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
LOCAL = threading.local()
STATS = defaultdict(lambda: dict(inclusive_s=0.0, self_s=0.0, calls=0, max_s=0.0))
CONTEXT_STATS = defaultdict(lambda: dict(inclusive_s=0.0, self_s=0.0, calls=0, max_s=0.0))
SOURCES = {}
LOCK = threading.Lock()


@contextmanager
def span(name):
    stack = getattr(LOCAL, "stack", None)
    if stack is None:
        stack = LOCAL.stack = []
    context = " > ".join([item[2] for item in stack] + [name])
    entry = [time.perf_counter(), 0.0, name]
    stack.append(entry)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - entry[0]
        stack.pop()
        if stack:
            stack[-1][1] += elapsed
        with LOCK:
            for row in (STATS[name], CONTEXT_STATS[context]):
                row["inclusive_s"] += elapsed
                row["self_s"] += max(0.0, elapsed - entry[1])
                row["calls"] += 1
                row["max_s"] = max(row["max_s"], elapsed)


def wrap(owner, attr, name):
    original = getattr(owner, attr)

    @functools.wraps(original)
    def measured(*args, **kwargs):
        with span(name):
            return original(*args, **kwargs)

    setattr(owner, attr, measured)


def instrument_tree(path, selectors):
    """Instrument only selected function top-level statements, preserving decorators."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))

    class Transformer(ast.NodeTransformer):
        def __init__(self):
            self.classes = []

        def visit_ClassDef(self, node):
            self.classes.append(node.name)
            self.generic_visit(node)
            self.classes.pop()
            return node

        def visit_FunctionDef(self, node):
            qualified = ".".join([*self.classes, node.name])
            if qualified not in selectors:
                return node
            body = []
            for statement in node.body:
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                    body.append(statement)
                    continue
                label = f"{path.relative_to(ROOT)}:{qualified}:L{statement.lineno}"
                SOURCES[label] = dict(path=str(path.relative_to(ROOT)), function=qualified,
                                      line=statement.lineno, end_line=statement.end_lineno,
                                      statement=ast.get_source_segment(source, statement).splitlines()[0][:220])
                context = ast.Call(func=ast.Name(id="_diagnostic_span", ctx=ast.Load()),
                                   args=[ast.Constant(label)], keywords=[])
                wrapped = ast.With(items=[ast.withitem(context_expr=context)], body=[statement])
                body.append(ast.copy_location(wrapped, statement))
            node.body = body
            return node

    tree = Transformer().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def instrument_module(path, selectors, module_name):
    tree = instrument_tree(path, selectors)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    module._diagnostic_span = span
    sys.modules[module_name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def event_wrapper(torch, owner, attr, name, pending, *, with_context=False):
    original = getattr(owner, attr)

    @functools.wraps(original)
    def measured(*args, **kwargs):
        label = name
        if with_context:
            label = " > ".join([item[2] for item in getattr(LOCAL, "stack", [])] + [name])
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        try:
            with span(name + ".host"):
                return original(*args, **kwargs)
        finally:
            end.record()
            pending.append((label, begin, end))

    setattr(owner, attr, measured)


def consume_events(pending, totals):
    # Caller has returned from native-synchronized predict()/DA3 runner.
    for name, begin, end in pending:
        seconds = begin.elapsed_time(end) / 1000.0
        row = totals.setdefault(name, dict(event_s=0.0, calls=0, max_s=0.0))
        row["event_s"] += seconds
        row["calls"] += 1
        row["max_s"] = max(row["max_s"], seconds)
    pending.clear()


def install_da3_events(torch, instrumented, pending):
    from depth_anything_3.model.dinov2.dinov2 import DinoV2

    for owner, attr, label in [
        (instrumented.NestedDepthAnything3Net, "forward", "da3.nested_forward"),
        (instrumented.DepthAnything3Net, "forward", "da3.branch_forward"),
        (DinoV2, "forward", "da3.backbone_forward"),
        (instrumented.NestedDepthAnything3Net, "_apply_metric_scaling", "da3.metric_scaling"),
        (instrumented.NestedDepthAnything3Net, "_apply_depth_alignment", "da3.depth_alignment"),
        (instrumented.NestedDepthAnything3Net, "_handle_sky_regions", "da3.sky_handling"),
    ]:
        event_wrapper(torch, owner, attr, label, pending, with_context=True)


def diagnose_da3(args, detail):
    with span("da3.imports"):
        # Initialize parent packages before replacing the canonical module;
        # model/__init__.py imports its classes during package initialization.
        canonical = importlib.import_module("depth_anything_3.model.da3")
        model_file = ROOT / "Depth-Anything-3/src/depth_anything_3/model/da3.py"
        instrumented = instrument_module(model_file, {
            "DepthAnything3Net.forward", "DepthAnything3Net._process_depth_head",
            "DepthAnything3Net._process_camera_estimation", "DepthAnything3Net._process_ray_pose_estimation",
            "DepthAnything3Net._process_mono_sky_estimation", "NestedDepthAnything3Net.forward",
            "NestedDepthAnything3Net._apply_metric_scaling", "NestedDepthAnything3Net._apply_depth_alignment",
            "NestedDepthAnything3Net._handle_sky_regions",
        }, "depth_anything_3.model.da3")
        package = importlib.import_module("depth_anything_3.model")
        package.da3 = instrumented
        for name in ("DepthAnything3Net", "NestedDepthAnything3Net"):
            if getattr(package, name, None) is getattr(canonical, name):
                setattr(package, name, getattr(instrumented, name))
        package.__export__ = [instrumented.NestedDepthAnything3Net, instrumented.DepthAnything3Net]
        runner = instrument_module(ROOT / "src/da3_runner.py", {"main"}, "diagnostic_da3_runner")
        from depth_anything_3.api import DepthAnything3
        import safetensors.torch
        import torch
    pending = []
    detail["cuda_events"] = {}
    install_da3_events(torch, instrumented, pending)
    wrap(DepthAnything3, "__init__", "da3.network_constructor")
    wrap(safetensors.torch, "load_file", "da3.safetensors_load_file")
    wrap(safetensors.torch, "load_model", "da3.safetensors_load_model")
    wrap(DepthAnything3, "load_state_dict", "da3.state_assignment")
    wrap(DepthAnything3, "to", "da3.model_to_device")
    for attr, label in [
        ("_preprocess_inputs", "preprocess"), ("_prepare_model_inputs", "input_enqueue"),
        ("_run_model_forward", "forward_native_sync"), ("_convert_to_prediction", "prediction_D2H"),
        ("_add_processed_images", "processed_RGB"),
    ]:
        wrap(DepthAnything3, attr, "da3." + label)
    wrap(runner, "_depth_to_world_points", "da3.CPU_backprojection")
    wrap(runner.np, "savez", "da3.NPZ_write_excluding_argument_evaluation")
    images = args.base / "dataset/images"
    paths = [p for p in images.iterdir() if p.suffix.lower() in runner.IMG_EXTS and p.is_file()]
    if len(paths) != 29:
        raise ValueError(f"Expected 29 source images, found {len(paths)}")
    output = args.out / "diagnostic-predictions.npz"
    with span("da3.runner_main"):
        runner.main(["--input_dir", str(images), "--output_npz", str(output),
                     "--model_path", str(args.model_path), "--device", "cuda:0",
                     "--process_res", "504", "--seed", "42"])
    consume_events(pending, detail["cuda_events"])
    detail.update(frames=29, process_res=504, seed=42, temporary_npz=str(output),
                  temporary_npz_bytes=output.stat().st_size,
                  synchronization="Only original DA3 forward pre/post synchronization; CUDA events consumed after runner.main returns, following native forward sync and Prediction D2H. Input enqueue includes CPU work and enqueue, not completed transfer. Nested event intervals overlap; do not sum parents and children. CUDA event creation/record and AST/wrapper overhead perturb this independent diagnostic.")


def diagnose_pvs(args, detail, torch):
    with span("pvs.imports"):
        import numpy as np
        from PIL import Image
        pvs = instrument_module(ROOT / "utils/sam31_pvs.py", {"load_model", "predict"}, "utils.diagnostic_sam31_pvs")
    with span("pvs.model_load"):
        model, count = pvs.load_model(ROOT / "sam31/checkpoints/sam3.1_multiplex.pt", device="cuda:0")
    pending = []
    event_wrapper(torch, model, "forward_image", "pvs.encoder", pending)
    event_wrapper(torch, model, "_forward_sam_heads", "pvs.decoder", pending)
    manifests = list((args.base / "outputs/dataset/sam3_mask_cache_sam31_pvs/v2/entries").glob("*/manifest.json"))
    manifests.sort(key=lambda p: int(p.parent.name))
    if len(manifests) != 29:
        raise ValueError(f"Expected 29 main-run PVS manifests, found {len(manifests)}")
    rows = []
    detail.update(checkpoint_tensors=count, frames_detail=rows, cuda_events={})
    source_images = {int(p.stem): p for p in (args.base / "dataset/images").iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        frame = int(manifest["image_id"])
        h, w = manifest["processed_shape_hw"]
        boxes = np.asarray([d["processed_bbox_xyxy"] for d in manifest["detections"]], dtype=np.float32).reshape(-1, 4)
        with span("pvs.source_decode_processed_resize"):
            with Image.open(source_images[frame]) as image:
                if list(image.size) != manifest["source_size_wh"]:
                    raise ValueError(f"Source dimensions changed for frame {frame}")
                processed = image.convert("RGB").resize((w, h), Image.Resampling.BICUBIC)
        with span("pvs.predict"):
            masks, scores, presence, areas, metrics = pvs.predict(model, processed, boxes, 4, device="cuda:0")
        consume_events(pending, detail["cuda_events"])
        if len(masks) != len(boxes):
            raise RuntimeError("PVS mask count mismatch")
        rows.append(dict(frame=frame, boxes=len(boxes), mask_shape=list(masks.shape), native_metrics=metrics))
        del masks, scores, presence, areas, processed
    detail.update(frames=len(rows), batch_size=4, seed=42,
                  synchronization="Encoder/decoder CUDA events are read only after native predict final sync. No new synchronization; no tensors retained by instrumentation.",
                  input_contract="Main-run manifests and same decoded source images, BICUBIC resize to DA3 processed shape, canonical manifest prompt order; no matching reordering/RNG restoration/cache publication/prefetch replay.",
                  native_metrics="inference_and_postprocess_s includes encoder_s and final np.stack; do not sum these metrics.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["da3", "pvs"], required=True)
    parser.add_argument("--base", type=Path, default=ROOT / "runtime/profiling-20261018")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    args.base, args.out, args.model_path = args.base.resolve(), args.out.resolve(), args.model_path.resolve()
    if args.out.exists():
        raise FileExistsError(f"Use a new diagnosis output directory: {args.out}")
    args.out.mkdir(parents=True)
    start = time.perf_counter()
    detail = dict(mode=args.mode, status="running", pid=os.getpid(), argv=sys.argv,
                  interpretation="Independent instrumented component diagnosis; never add to or replace main-run timings.",
                  wall_semantics="Thread-local nested wall spans. Self subtracts directly instrumented children only; instrumentation overhead remains. Recursive labels aggregate calls; CUDA event spans are separate and nonadditive with wall.")
    torch = None
    try:
        sys.path[:0] = [str(ROOT), str(ROOT / "Depth-Anything-3/src"), str(ROOT / "sam31")]
        with span("imports"):
            import torch
            import numpy as np
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        torch.set_num_threads(4)
        torch.manual_seed(42)
        np.random.seed(42)
        import random
        random.seed(42)
        detail.update(device=torch.cuda.get_device_name(0), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      threads=torch.get_num_threads(), allocator_scope="process lifetime; no peak resets")
        if args.mode == "da3":
            diagnose_da3(args, detail)
        else:
            diagnose_pvs(args, detail, torch)
        detail["status"] = "success"
    except BaseException as error:
        detail.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        detail.update(elapsed_s=time.perf_counter() - start, wall=dict(STATS), wall_contexts=dict(CONTEXT_STATS), source_mapping=SOURCES,
                      rss_hwm_kib=usage.ru_maxrss, cpu_user_s=usage.ru_utime, cpu_system_s=usage.ru_stime)
        if torch is not None and torch.cuda.is_initialized():
            detail.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          final_allocated_bytes=torch.cuda.memory_allocated())
        result = args.out / (args.mode + "-result.json")
        result.write_text(json.dumps(detail, indent=2))
        print(json.dumps(dict(mode=args.mode, status=detail["status"], result=str(result))), flush=True)


if __name__ == "__main__":
    main()
