"""Opt-in full CPU exports: candidate 1/2/4 encoders and optional snapshot bake.

No GPU/model calls. Each arm exports the same completed Viewer generation.
Timing excludes output comparison; outputs remain for coordinator cleanup.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import resource
import struct
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def compare_glb(left_path, right_path):
    import numpy as np
    from PIL import Image

    def read(path):
        raw = path.read_bytes()
        json_size = struct.unpack_from("<I", raw, 12)[0]
        document = json.loads(raw[20:20+json_size])
        return document, memoryview(raw)[28+json_size:]

    left, left_bin = read(left_path)
    right, right_bin = read(right_path)

    def payload(doc, binary, index):
        view = doc["bufferViews"][index]
        start = view.get("byteOffset", 0)
        return binary[start:start+view["byteLength"]]

    result = dict(left=str(left_path), right=str(right_path),
                  document_associations_equal=left == right,
                  image_count_equal=len(left["images"]) == len(right["images"]),
                  accessor_count_equal=len(left["accessors"]) == len(right["accessors"]),
                  accessor_bytes_equal=True, decoded_rgba_equal=True,
                  compared_texels=0, compared_rgb_channels=0,
                  alpha_texels_changed=0, rgb_texels_changed=0,
                  opaque_rgb_texels_changed=0, rgb_max_abs_change=0,
                  rgb_abs_change_sum=0, both_opaque_texels=0,
                  both_opaque_rgb_channels=0, opaque_rgb_max_abs_change=0,
                  opaque_rgb_abs_change_sum=0,
                  unpaired_images=abs(len(left["images"])-len(right["images"])),
                  image_shape_mismatches=0)
    for a, b in zip(left["accessors"], right["accessors"]):
        if a != b or payload(left, left_bin, a["bufferView"]) != payload(right, right_bin, b["bufferView"]):
            result["accessor_bytes_equal"] = False
    result["accessor_bytes_equal"] &= result["accessor_count_equal"]
    for a, b in zip(left["images"], right["images"]):
        with Image.open(BytesIO(payload(left, left_bin, a["bufferView"]))) as image:
            old = np.array(image.convert("RGBA"))
        with Image.open(BytesIO(payload(right, right_bin, b["bufferView"]))) as image:
            new = np.array(image.convert("RGBA"))
        if old.shape != new.shape:
            result["image_shape_mismatches"] += 1
            result["decoded_rgba_equal"] = False
            continue
        alpha_delta = old[..., 3] != new[..., 3]
        rgb_delta = np.any(old[..., :3] != new[..., :3], axis=-1)
        both_opaque = (old[..., 3] != 0) & (new[..., 3] != 0)
        absolute = np.abs(old[..., :3].astype(np.int16)-new[..., :3])
        opaque_absolute = absolute[both_opaque]
        result["compared_texels"] += int(alpha_delta.size)
        result["compared_rgb_channels"] += int(absolute.size)
        result["alpha_texels_changed"] += int(alpha_delta.sum())
        result["rgb_texels_changed"] += int(rgb_delta.sum())
        result["opaque_rgb_texels_changed"] += int((rgb_delta & both_opaque).sum())
        result["rgb_max_abs_change"] = max(result["rgb_max_abs_change"], int(absolute.max()))
        result["rgb_abs_change_sum"] += int(absolute.sum())
        result["both_opaque_texels"] += int(both_opaque.sum())
        result["both_opaque_rgb_channels"] += int(opaque_absolute.size)
        result["opaque_rgb_max_abs_change"] = max(
            result["opaque_rgb_max_abs_change"], int(opaque_absolute.max()) if opaque_absolute.size else 0)
        result["opaque_rgb_abs_change_sum"] += int(opaque_absolute.sum())
        result["decoded_rgba_equal"] &= not bool(alpha_delta.any() or rgb_delta.any())
    result["decoded_rgba_equal"] &= result["image_count_equal"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New independent directory")
    parser.add_argument("--include-snapshot", action="store_true", help="Also export the snapshot baker with one encoder")
    parser.add_argument("--projection-mode", choices=("source_world", "camera_basis"),
                        default="source_world", help="source_world preserves the baseline projection order")
    parser.add_argument("--reference-glb", type=Path,
                        help="Compare an existing baseline GLB from exactly --generation; no extra snapshot export")
    parser.add_argument("--baseline", type=Path, default=ROOT / "runtime/latency-optimization-20261018/baseline-code/src/surfel_mesh.py")
    args = parser.parse_args()
    if args.reference_glb is not None and not args.reference_glb.is_file():
        parser.error("--reference-glb must be an existing GLB")
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="failed", generation=str(args.generation), runs=[], comparisons=[],
                   projection_mode=args.projection_mode,
                   semantics="One export per arm, same source generation and defaults except encoder_workers. No page copies inside export timers. Single order, not repeated performance acceptance. RSS sampled self every 0.1s during export; ru_maxrss is cumulative process high-water. Outputs retained.")
    if args.reference_glb is not None:
        receipt.update(reference_glb=str(args.reference_glb),
                       reference_scope="Caller must supply a baseline GLB from the exact same generation; independently regenerated geometry can confound this comparison")
    sys.path.insert(0, str(ROOT))
    from src import surfel_mesh as mesh
    from src.surfel_glb import export_scene_glb
    from bake_probe import load_snapshot
    original_baker = mesh.bake_surfel_pages

    def save():
        (args.out / "result.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")

    try:
        arms = [("candidate_1", 1, original_baker), ("candidate_2", 2, original_baker),
                ("candidate_4", 4, original_baker)]
        if args.include_snapshot:
            snapshot = load_snapshot(args.baseline)
            receipt["baseline_source"] = str(args.baseline)
            def snapshot_baker(**kwargs):
                if kwargs.pop("projection_mode") != "source_world":
                    raise ValueError("Snapshot baker only supports its original source_world mode")
                return snapshot.bake_surfel_pages(**kwargs)
            arms.insert(0, ("snapshot_1", 1, snapshot_baker))
        for label, workers, baker in arms:
            mode = "source_world" if label == "snapshot_1" else args.projection_mode
            record = dict(label=label, encoder_workers=workers, projection_mode=mode,
                          status="failed", sampled_rss_peak_bytes=0, samples=0)
            receipt["runs"].append(record)
            mesh.bake_surfel_pages = baker
            stop = threading.Event()

            def sample():
                while not stop.is_set():
                    rss = int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
                    record["sampled_rss_peak_bytes"] = max(record["sampled_rss_peak_bytes"], rss)
                    record["samples"] += 1
                    stop.wait(.1)

            sampler = threading.Thread(target=sample, daemon=True)
            before = resource.getrusage(resource.RUSAGE_SELF)
            record["ru_maxrss_kib_before"] = before.ru_maxrss
            sampler.start()
            tick = time.perf_counter()
            try:
                record["production_report"] = export_scene_glb(
                    args.generation, args.out / (label+".glb"), encoder_workers=workers,
                    projection_mode=mode)
                record["status"] = "complete"
            finally:
                record["export_wall_s"] = time.perf_counter()-tick
                stop.set()
                sampler.join()
                after = resource.getrusage(resource.RUSAGE_SELF)
                record.update(cpu_user_s=after.ru_utime-before.ru_utime,
                              cpu_system_s=after.ru_stime-before.ru_stime,
                              ru_maxrss_kib_after=after.ru_maxrss)
                mesh.bake_surfel_pages = original_baker
                save()
        control = args.out / "candidate_1.glb"
        if args.reference_glb is not None:
            receipt["comparisons"].append(compare_glb(args.reference_glb, control))
            save()
        for label, _, _ in arms:
            if label != "candidate_1":
                receipt["comparisons"].append(compare_glb(control, args.out / (label+".glb")))
                save()
        receipt["status"] = "complete"
    except BaseException:
        receipt["exception"] = traceback.format_exc()
        raise
    finally:
        mesh.bake_surfel_pages = original_baker
        save()


if __name__ == "__main__":
    main()
