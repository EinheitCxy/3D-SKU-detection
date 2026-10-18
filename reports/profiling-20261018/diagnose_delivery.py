"""Independent CPU delivery diagnosis; never add nested inclusive timings."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager
import functools
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
LOCAL = threading.local()
LOCK = threading.Lock()
STATS = defaultdict(lambda: dict(calls=0, inclusive_s=0., self_s=0., max_s=0.))


@contextmanager
def span(name):
    stack = getattr(LOCAL, "stack", None)
    if stack is None:
        stack = LOCAL.stack = []
    frame = [time.perf_counter(), 0.]
    stack.append(frame)
    try:
        yield
    finally:
        duration = time.perf_counter() - frame[0]
        stack.pop()
        if stack:
            stack[-1][1] += duration
        with LOCK:
            item = STATS[name]
            item["calls"] += 1
            item["inclusive_s"] += duration
            item["self_s"] += duration - frame[1]
            item["max_s"] = max(item["max_s"], duration)


def usage():
    current = resource.getrusage(resource.RUSAGE_SELF)
    return dict(user_s=current.ru_utime, system_s=current.ru_stime,
                process_lifetime_maxrss_kib=current.ru_maxrss,
                minor_faults=current.ru_minflt, major_faults=current.ru_majflt,
                block_inputs=current.ru_inblock, block_outputs=current.ru_oublock)


def instrument_batch(function):
    """Wrap contiguous top-level statements; retain original source line numbers."""
    source, first_line = inspect.getsourcelines(function)
    tree = ast.parse("".join(source))
    ast.increment_lineno(tree, first_line - 1)
    definition = tree.body[0]
    # Offsets from function start avoid dependence on its absolute file line.
    boundaries = [(0, "camera_visibility"), (15, "footprint_and_continuity"),
                  (27, "radius_final_diameter"), (30, "texel_projection_mask"),
                  (41, "depth_and_source_mask"), (51, "bilinear_rgb"),
                  (54, "tile_corners_dilation")]
    groups = []
    for statement in definition.body:
        offset = statement.lineno - first_line
        label = next(label for start, label in reversed(boundaries) if offset >= start)
        if not groups or groups[-1][0] != label:
            groups.append((label, []))
        groups[-1][1].append(statement)
    definition.body = [ast.copy_location(ast.With(
        items=[ast.withitem(context_expr=ast.Call(
            func=ast.Name(id="_delivery_span", ctx=ast.Load()),
            args=[ast.Constant("bake.block." + label)], keywords=[]))],
        body=statements), statements[0]) for label, statements in groups]
    scope = dict(function.__globals__, _delivery_span=span)
    ast.fix_missing_locations(tree)
    exec(compile(tree, inspect.getfile(function), "exec"), scope)
    return scope[function.__name__], dict(
        file=inspect.getfile(function), first_line=first_line,
        boundaries=[dict(offset=start, label=label) for start, label in boundaries])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=REPO / "runtime/profiling-20261018")
    parser.add_argument("--out", type=Path, required=True,
                        help="New independent output directory; artifacts are retained")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    before = usage()
    receipt = dict(status="failed", base=str(args.base.resolve()), output=str(out),
                   command=[sys.executable, *sys.argv], python=sys.version,
                   artifacts_retained=True, source_locations={},
                   timing_definition="inclusive includes nested spans; self subtracts direct children on the same thread; do not sum parent and child inclusive_s",
                   resource_definition="RUSAGE_SELF CPU deltas; maxrss is process-lifetime Linux KiB, not a stage-local allocation peak",
                   performance_scope="Independent instrumented CPU delivery; not pipeline latency or GPU time")
    patches = []

    def patch(owner, name, value):
        patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def wrap(owner, name, label):
        original = getattr(owner, name)
        try:
            receipt["source_locations"][label] = dict(
                file=inspect.getfile(original), line=inspect.getsourcelines(original)[1])
        except (TypeError, OSError):
            pass
        @functools.wraps(original)
        def measured(*positional, **keywords):
            with span(label):
                return original(*positional, **keywords)
        patch(owner, name, measured)

    try:
        sys.path.insert(0, str(REPO))
        with span("imports"):
            import zipfile
            from PIL import Image
            from src import surfel_glb as glb, surfel_mesh as mesh, surfel_lod as lod
            from src import surfel_glb_texture as texture, surfel_glb_compact as compact
            from perf.viewer_clarity.pack import pack_bundle
        receipt["environment"] = dict(
            numpy=sys.modules["numpy"].__version__, pillow=sys.modules["PIL"].__version__,
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            variables={key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "TMPDIR")})
        receipt["export_defaults"] = {
            key: parameter.default for key, parameter in
            inspect.signature(glb.export_scene_glb).parameters.items()
            if parameter.default is not inspect.Parameter.empty}
        receipt["baker_defaults"] = {
            key: parameter.default for key, parameter in
            inspect.signature(mesh.bake_surfel_pages).parameters.items()
            if parameter.default is not inspect.Parameter.empty}
        receipt["final_diameter_max_native"] = mesh.FINAL_DIAMETER_MAX_NATIVE
        if receipt["export_defaults"]["encoder_workers"] != 1:
            raise RuntimeError("Production encoder default changed; review diagnostic before use")
        receipt["configuration"] = dict(encoder_workers=1, overrides="encoder_workers only; other values use recorded production defaults")
        for owner, names, prefix in [
            (glb, ["_load"], "glb."),
            (lod, ["select_surfel_lod", "_covered", "_bound_footprint"], "lod."),
            (mesh, ["_bound_footprint", "_limit_final_diameter", "_bilinear", "_dilate_rgb", "_page"], "bake."),
            (texture, ["encode_atlas", "_binary_alpha", "_compact_rgba", "_rgba_png"], "texture."),
            (compact, ["_write_glb"], "compact."),
            (compact._CompactWriter, ["geometry", "image", "finish", "view"], "compact."),
            (Image.Image, ["convert", "quantize", "save"], "PIL.")]:
            for name in names:
                wrap(owner, name, prefix + name)
        batch, mapping = instrument_batch(mesh._bake_batch)
        receipt["batch_instrumentation"] = mapping
        patch(mesh, "_bake_batch", batch)
        wrap(mesh, "_bake_batch", "bake._bake_batch")
        original_write = zipfile.ZipFile.write
        members = []
        receipt["zip_members"] = members
        def zip_write(archive, filename, arcname=None, compress_type=None, compresslevel=None):
            category = "binary" if str(filename).endswith(".bin") else (
                "jpeg" if str(filename).endswith(".jpg") else "json")
            tick = time.perf_counter()
            with span("zip.member." + category):
                value = original_write(archive, filename, arcname, compress_type, compresslevel)
            info = archive.filelist[-1]
            members.append(dict(name=info.filename, category=category,
                                input_bytes=info.file_size, compressed_bytes=info.compress_size,
                                compression=info.compress_type, wall_s=time.perf_counter() - tick))
            return value
        patch(zipfile.ZipFile, "write", zip_write)
        with span("resolve_generation"):
            viewer = args.base.resolve() / "viewer"
            run_id = json.loads((viewer / "CURRENT").read_text())["run_id"]
            generation = viewer / "runs" / run_id
            receipt["generation"] = str(generation)
            receipt["run_id"] = run_id
        with span("delivery_wall"):
            with span("zip.total"):
                receipt["zip_bytes"] = pack_bundle(generation, out / "diagnostic.zip")
            receipt["zip_members"] = members
            with span("glb.total"):
                receipt["production_report"] = glb.export_scene_glb(
                    generation, out / "diagnostic.glb", encoder_workers=1)
        receipt["status"] = "complete"
    except BaseException as error:
        receipt["exception"] = dict(type=type(error).__name__, message=str(error),
                                    traceback=traceback.format_exc())
        raise
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)
        after = usage()
        receipt.update(elapsed_wall_s=time.perf_counter() - started, metrics=dict(STATS),
                       resources_before=before, resources_after=after,
                       cpu_user_s=after["user_s"] - before["user_s"],
                       cpu_system_s=after["system_s"] - before["system_s"])
        receipt["artifacts"] = {name: (out / name).stat().st_size
                                for name in ("diagnostic.zip", "diagnostic.glb")
                                if (out / name).exists()}
        (out / "result.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
