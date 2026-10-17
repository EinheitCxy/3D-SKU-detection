"""Bounded, opt-in atlas comparison. Preparation alone does not authorize execution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import tempfile
import threading
import time
import traceback
from io import BytesIO

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[1]
GENERATION = REPO / "runtime/video3-full-profile-20261016/viewer/runs/c9c3be6802144fcf97a942085b118422"
CPUS = {3, 22, 27, 29}
ORDER = ("B1", "C1", "C2", "B2")
FILE_LIMIT = 256 * 1024**2
MEMORY_LIMIT = 12 * 1024**3
EXPORT = dict(radius=2.0, tile_size=8, lod=True, palette_colors=128)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "official"), required=True)
    parser.add_argument("--candidate-workers", choices=(2, 4), type=int, default=2)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="New, nonexistent directory beneath this script's directory")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(TASK) or run_dir == TASK:
        parser.error("--run-dir must be a new task-local subdirectory")
    run_dir.mkdir(parents=False, exist_ok=False)
    started = time.monotonic()
    cpu_limit, wall_limit = (100, 120) if args.phase == "smoke" else (2000, 900)
    stop = threading.Event()
    active_started = None
    sampling = {"peak_bytes": 0, "samples": 0, "children_seen": False}
    active_record = None
    result = dict(phase=args.phase, status="partial", runs=[], paired_savings_s=None,
                  primary_minimum_paired_saving_s=None, source=str(GENERATION),
                  cpu_ids=sorted(CPUS), export=EXPORT, seed=0, argv=sys.argv,
                  candidate_workers=args.candidate_workers,
                  single_change="encoder_workers and its matching bounded FIFO capacity: baseline 1 versus selected candidate",
                  encoder_queue_pages=dict(baseline=1, candidate=args.candidate_workers),
                  command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                  python=sys.version, limits=dict(cpu_seconds=cpu_limit,
                  wall_seconds=wall_limit, address_space_bytes=MEMORY_LIMIT,
                  rss_bytes=MEMORY_LIMIT, single_file_bytes=FILE_LIMIT,
                  maximum_simultaneous_large_files=2),
                  primary_timer="Unadjusted complete export wall time including identical page copies",
                  resource_scope="CPU RUSAGE_SELF plus reaped RUSAGE_CHILDREN; no subprocesses launched. RSS samples self at 0.1s plus self ru_maxrss; sampling can miss subinterval peaks. RLIMIT_AS caps process address space. GPU hidden; no GPU/model/service calls. Two large files structurally bounded: one temporary BIN and one current GLB, each <=256MiB. Baseline bytes/pages and current pages live in RAM.")

    def save():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        result["resources"] = dict(
            elapsed_wall_seconds=time.monotonic() - started,
            cpu_self_seconds=usage.ru_utime + usage.ru_stime,
            cpu_reaped_children_seconds=children.ru_utime + children.ru_stime,
            cpu_total_seconds=usage.ru_utime + usage.ru_stime + children.ru_utime + children.ru_stime,
            sampled_rss_peak_bytes=sampling["peak_bytes"],
            self_ru_maxrss_bytes=usage.ru_maxrss * 1024,
            rss_sampling_interval_seconds=.1, rss_samples=sampling["samples"],
            children_observed=sampling["children_seen"],
            linked_output_bytes=sum(p.stat().st_size for p in run_dir.iterdir() if p.is_file()),
            gpu_seconds=0,
        )
        text = json.dumps(result, indent=2, allow_nan=False)
        if len(text.encode()) > 1024**2:
            raise RuntimeError("receipt exceeds 1MiB metadata bound")
        pending = run_dir / "partial.pending.json"
        pending.write_text(text + "\n")
        pending.replace(run_dir / "partial.json")

    def interrupted(signum, frame):
        result["interruption_signal"] = signal.Signals(signum).name
        result["status"] = "interrupted"
        if active_started is not None and result["runs"]:
            result["runs"][-1]["raw_export_wall_s_at_signal"] = time.perf_counter() - active_started
        save()  # Persist before an exporter may wait for its running encoders.
        raise RuntimeError(f"execution interrupted by {signal.Signals(signum).name}")

    def sample():
        while not stop.is_set():
            rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            sampling["peak_bytes"] = max(sampling["peak_bytes"], rss)
            sampling["samples"] += 1
            arm = active_record
            if arm is not None:
                arm["sampled_export_rss_peak_bytes"] = max(arm["sampled_export_rss_peak_bytes"], rss)
                arm["export_rss_samples"] += 1
            children = Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
            sampling["children_seen"] |= bool(children)
            if rss > MEMORY_LIMIT or children:
                os.kill(os.getpid(), signal.SIGUSR1)
                return
            stop.wait(.1)

    sampler = None
    try:
        save()
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("This comparison requires the approved Python 3.11 environment")
        for sig in (signal.SIGXCPU, signal.SIGALRM, signal.SIGTERM, signal.SIGINT,
                    signal.SIGUSR1, signal.SIGXFSZ):
            signal.signal(sig, interrupted)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit - 1, cpu_limit))
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_LIMIT, FILE_LIMIT))
        # Leave one second for the receipt and encoder drain before the command's
        # external hard wall deadline (documented timeout wrapper).
        signal.setitimer(signal.ITIMER_REAL, max(.001, wall_limit - 1 - (time.monotonic() - started)))
        if not CPUS.issubset(os.sched_getaffinity(0)):
            raise RuntimeError("Fixed CPU allocation unavailable; no substitute CPUs permitted")
        os.sched_setaffinity(0, CPUS)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[key] = "4"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        temporary = run_dir / "tmp"
        temporary.mkdir()
        os.environ["TMPDIR"] = str(temporary)
        tempfile.tempdir = str(temporary)
        result["environment"] = {key: os.environ[key] for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "CUDA_VISIBLE_DEVICES", "TMPDIR")}
        sampler = threading.Thread(target=sample, name="atlas-rss", daemon=True)
        sampler.start()
        sys.path.insert(0, str(REPO))
        import numpy as np
        from PIL import Image
        from src import surfel_glb, surfel_mesh

        def affinities():
            values = {p.name: sorted(os.sched_getaffinity(int(p.name)))
                      for p in Path("/proc/self/task").iterdir()}
            if any(not set(value).issubset(CPUS) for value in values.values()):
                raise RuntimeError("Thread escaped fixed CPU allocation")
            return values

        result["import_thread_affinities"] = affinities()
        manifest = json.loads((GENERATION / "manifest.json").read_text())
        metadata = json.loads((GENERATION / "surfel.json").read_text())
        if (manifest["frame_count"] != 29 or metadata["point_count"] != 2402589
                or len(metadata["frames"]) != 29 or metadata["grid_size"] != [280, 504]):
            raise RuntimeError("Fixed video3 generation counts changed")
        result["input_counts"] = dict(frames=29, points=2402589)
        original_load = surfel_glb._load
        if args.phase == "smoke":
            def subset(read, global_id):
                matrix, inputs = original_load(read, global_id)
                for key in ("positions", "u", "v", "scales", "frame_indices"):
                    inputs[key] = inputs[key][:2048]
                return matrix, inputs
            surfel_glb._load = subset
            result["smoke_scope"] = "First 2048 input points of fixed generation; full source loads, depth and textures retained. Correctness smoke only; no performance evidence."

        def decode(raw):
            magic, version, length = struct.unpack_from("<4sII", raw)
            if (magic, version, length) != (b"glTF", 2, len(raw)):
                raise ValueError("Invalid GLB header")
            json_length, kind = struct.unpack_from("<I4s", raw, 12)
            if kind != b"JSON":
                raise ValueError("Invalid GLB JSON chunk")
            doc = json.loads(raw[20:20 + json_length])
            binary_length, kind = struct.unpack_from("<I4s", raw, 20 + json_length)
            binary = memoryview(raw)[28 + json_length:]
            if kind != b"BIN\0" or binary_length != len(binary):
                raise ValueError("Invalid GLB BIN chunk")
            return doc, binary

        def payload(doc, binary, index):
            view = doc["bufferViews"][index]
            offset, length = view.get("byteOffset", 0), view["byteLength"]
            if offset < 0 or offset + length > len(binary):
                raise ValueError("Buffer view outside BIN")
            return binary[offset:offset + length]

        def accessor(doc, binary, entry):
            if entry["componentType"] != 5123:
                raise ValueError("Expected compact unsigned-short accessor")
            width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[entry["type"]]
            view = doc["bufferViews"][entry["bufferView"]]
            stride = view.get("byteStride", width * 2)
            raw = payload(doc, binary, entry["bufferView"])
            return np.ndarray((entry["count"], width), dtype="<u2", buffer=raw,
                              offset=entry.get("byteOffset", 0), strides=(stride, 2))

        def compare(current, reference, raw, baseline_raw):
            def same_array(left, right):
                # numpy.testing lazily launches a CPU-feature probe subprocess in
                # this NumPy version; direct equality needs no external process.
                if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                    raise AssertionError("Complete array equality failed")

            if len(current) != len(reference) or not current:
                raise AssertionError("Atlas page count changed or empty")
            if args.phase == "official" and len(current) != 31:
                raise AssertionError("Expected all 31 full-generation atlas pages")
            for left, right in zip(current, reference):
                for key in ("image_rgba", "positions", "uv", "indices"):
                    if left[key].dtype != right[key].dtype or left[key].shape != right[key].shape:
                        raise AssertionError(f"Prequantization {key} dtype/shape changed")
                    same_array(left[key], right[key])
            if raw != baseline_raw:
                raise AssertionError("Complete GLB bytes differ")
            doc, binary = decode(raw)
            other, baseline_binary = decode(baseline_raw)
            if doc != other:
                raise AssertionError("GLB geometry/UV/index/material associations changed")
            for left, right in zip(doc["accessors"], other["accessors"]):
                same_array(accessor(doc, binary, left), accessor(other, baseline_binary, right))
            for left, right in zip(doc["images"], other["images"]):
                with Image.open(BytesIO(payload(doc, binary, left["bufferView"]))) as a:
                    with Image.open(BytesIO(payload(other, baseline_binary, right["bufferView"]))) as b:
                        same_array(np.array(a.convert("RGBA")), np.array(b.convert("RGBA")))
            return dict(all_page_prequant_arrays_equal=True, full_glb_bytes_equal=True,
                        all_accessors_equal=True, all_decoded_png_rgba_equal=True,
                        full_document_associations_equal=True)

        original_baker = surfel_mesh.bake_surfel_pages
        baseline_pages = baseline_raw = None
        for label in ORDER:
            workers = 1 if label.startswith("B") else args.candidate_workers
            pages = []
            output = run_dir / (label + ".glb")
            record = dict(label=label, encoder_workers=workers,
                          encoder_queue_pages=workers,
                          export_complete=False, compare_passed=False)
            result["runs"].append(record)
            save()

            def captured_baker(**kwargs):
                for page in original_baker(**kwargs):
                    # Identical copies inside every arm's raw export timer.
                    pages.append({key: page[key].copy() for key in
                                  ("image_rgba", "positions", "uv", "indices")})
                    yield page

            surfel_mesh.bake_surfel_pages = captured_baker
            np.random.seed(0)
            usage_before = resource.getrusage(resource.RUSAGE_SELF)
            record.update(sampled_export_rss_peak_bytes=0, export_rss_samples=0,
                          same_process_ru_maxrss_bytes_before=usage_before.ru_maxrss * 1024,
                          rss_diagnostic_scope="Self RSS sampled during export including retained B1 pages/bytes; ru_maxrss is same-process cumulative high-water, not arm-local allocation")
            active_record = record
            before = time.perf_counter()
            active_started = before
            try:
                record["production_report"] = surfel_glb.export_scene_glb(
                    GENERATION, output, **EXPORT, encoder_workers=workers)
                record["export_complete"] = True
            finally:
                record["raw_export_wall_s"] = time.perf_counter() - before
                active_started = None
                active_record = None
                usage_after = resource.getrusage(resource.RUSAGE_SELF)
                record.update(export_cpu_user_s=usage_after.ru_utime - usage_before.ru_utime,
                              export_cpu_system_s=usage_after.ru_stime - usage_before.ru_stime,
                              export_cpu_total_s=usage_after.ru_utime + usage_after.ru_stime - usage_before.ru_utime - usage_before.ru_stime,
                              same_process_ru_maxrss_bytes_after=usage_after.ru_maxrss * 1024)
                surfel_mesh.bake_surfel_pages = original_baker
                record["captured_pages"] = len(pages)
                record["captured_array_bytes"] = sum(v.nbytes for p in pages for v in p.values())
                save()
            raw = output.read_bytes()
            record["output_bytes"] = len(raw)
            if label == "B1":
                baseline_pages, baseline_raw = pages, raw
            record["comparison"] = compare(pages, baseline_pages, raw, baseline_raw)
            record["compare_passed"] = True
            record["thread_affinities"] = affinities()
            output.unlink()  # Only this successfully compared arm's own current GLB.
            del pages, raw
            save()
            print(json.dumps({"label": label, "raw_export_wall_s": record["raw_export_wall_s"],
                              "compare_passed": True}), flush=True)
        times = {r["label"]: r["raw_export_wall_s"] for r in result["runs"]}
        savings = [times["B1"] - times["C1"], times["B2"] - times["C2"]]
        result["paired_savings_s"] = savings
        result["primary_minimum_paired_saving_s"] = min(savings)
        result["status"] = ("smoke_pass" if args.phase == "smoke" else
                            "pass" if all(value > 0 for value in savings) else "inconclusive")
        result["performance_accepted"] = args.phase == "official" and all(value > 0 for value in savings)
    except BaseException as error:
        stop.set()
        result["status"] = "failed"
        result["performance_accepted"] = False
        result["failure"] = dict(error=repr(error), traceback=traceback.format_exc())
        if result["runs"]:
            result["runs"][-1]["error"] = repr(error)
        save()
        raise
    finally:
        stop.set()
        signal.setitimer(signal.ITIMER_REAL, 0)
        if sampler is not None:
            sampler.join(1)
        save()


if __name__ == "__main__":
    main()
