"""One fresh video3 pipeline: measured delivery first, output comparison second."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "runtime/profiling-20261018/da3-model/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7"
PYTHON = ROOT / "runtime/viewer-latency-20261017/test-env/bin/python"


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "candidate"), required=True)
    parser.add_argument("--input-case", type=Path, required=True)
    parser.add_argument("--reference-case", type=Path, required=True)
    parser.add_argument("--shared-neighbors", action="store_true",
                        help="Explicit one-tree NN/SOR candidate; independent of PVS/bake mode")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "dataset").symlink_to(args.input_case.resolve() / "dataset", target_is_directory=True)
    shutil.copyfile(ROOT / "config.yaml", out / "config.yaml")
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import open3d as o3d
    import torch
    import main as entry
    import src.web_viewer_export as viewer
    import utils.sam31_pvs as pvs
    from src.da3_3d_reconstructor import DA33DReconstructor

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    o3d.utility.random.seed(42)
    os.environ["DA3_VENV_PYTHON"] = str(PYTHON)
    original_runner = DA33DReconstructor._run_runner

    def seeded_runner(self, command):
        # Parent RNG does not seed the reconstruction subprocess. Match the
        # retained reference's explicit runner --seed42, without changing production.
        return original_runner(self, [*command, "--seed", "42"])

    DA33DReconstructor._run_runner = seeded_runner
    timings = defaultdict(list)
    native_loader = pvs.load_model

    def load(*a, **kw):
        begin = time.perf_counter()
        kw["skip_random_init"] = args.mode == "candidate"
        model, count = native_loader(*a, **kw)
        timings["pvs_load"].append(time.perf_counter()-begin)
        return model, count

    pvs.load_model = load
    original_prepare = viewer.prepare_scene_filter

    def prepare(*a, **kw):
        kw["shared_neighbors"] = args.shared_neighbors
        return original_prepare(*a, **kw)

    viewer.prepare_scene_filter = prepare

    def wrap(owner, name):
        original = getattr(owner, name)

        def measured(*a, **kw):
            begin = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                timings[name].append(time.perf_counter()-begin)

        setattr(owner, name, measured)

    app = entry.SKUDetectionMain()
    app.save_root = out / "outputs"
    app.match_backend = "da3"
    app.classifier_enabled = False
    app.config_path = out / "config.yaml"
    for name in ("run_reconstruction", "run_sku_matching", "run_dedup_sequence"):
        wrap(app, name)
    for name in ("prepare_web_viewer_geometry", "export_web_viewer_bundle", "get_scene_orientation"):
        wrap(viewer, name)
    imported = time.perf_counter()
    summary = app.run_complete_pipeline(
        str(out / "dataset"), "3d", model_path=str(MODEL), evaluate_accuracy=False,
        quiet_outputs=True, surfel_output_dir=out / "viewer", surfel_texture_edge=1920,
        # Use the existing preparation thread; failed process isolation is not
        # part of the published production interface.
    )
    viewer_ready = time.perf_counter()
    generation = out / "viewer/runs" / json.loads((out / "viewer/CURRENT").read_text())["run_id"]
    result = dict(status="viewer_ready", mode=args.mode,
                  shared_neighbors=args.shared_neighbors, summary=summary,
                  measured_scope="Frame-ready DA3/matching/Viewer/ZIP/GLB; excludes detector/classifier/COS/browser",
                  da3_seed=42,
                  input_case=str(args.input_case.resolve()), generation=str(generation),
                  imports_and_setup_s=imported-started, viewer_ready_s=viewer_ready-started)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    from perf.viewer_clarity.pack import pack_bundle
    from src.surfel_glb import export_scene_glb

    begin = time.perf_counter()
    pack_bundle(generation, out / "viewer_bundle.zip")
    zip_ready = time.perf_counter()
    result.update(zip_s=zip_ready-begin, zip_ready_s=zip_ready-started)
    begin = time.perf_counter()
    report = export_scene_glb(
        generation, out / "scene.glb", encoder_workers=4 if args.mode == "candidate" else 2,
        bake_device="cuda:0" if args.mode == "candidate" else "cpu",
        bake_fixed_shape=args.mode == "candidate",
    )
    ready = time.perf_counter()
    result.update(status="ready", total_ready_s=ready-started, glb_s=ready-begin,
                  glb_report=report,
                  timings={k: dict(total_s=sum(v), calls=len(v)) for k, v in timings.items()})
    (out / "result.json").write_text(json.dumps(result, indent=2))
    # All correctness work below is outside the delivery clock.
    reference = args.reference_case.resolve()
    ref_generation = reference / "viewer/runs" / json.loads((reference / "viewer/CURRENT").read_text())["run_id"]
    equality = {}
    for name in ("positions.f32.bin", "colors.u8.bin", "normals.i8.bin", "surfel-u.f16.bin",
                 "surfel-v.f16.bin", "surfel-scale.f16.bin", "surfel-frame.u8.bin", "surfel-depth.f16.bin"):
        equality[name] = (generation / name).read_bytes() == (ref_generation / name).read_bytes()
    for name in ("manifest.json", "objects.json"):
        equality[name] = json.loads((generation / name).read_text()) == json.loads((ref_generation / name).read_text())
    mapping = Path("outputs/dataset/dedup_detections/global_mapping.json")
    equality["global_mapping"] = json.loads((out / mapping).read_text()) == json.loads((reference / mapping).read_text())
    npz = Path("outputs/dataset/da3_cache/predictions.npz")
    with np.load(out / npz, allow_pickle=False) as a, np.load(reference / npz, allow_pickle=False) as b:
        for key in ("depth", "depth_conf", "world_points", "world_points_conf", "extrinsic", "intrinsic", "images",
                    "image_ids", "source_image_sizes", "source_to_processed_affine", "is_metric", "scale_factor",
                    "frame_alignment_sorted_indices", "frame_alignment_map_keys", "frame_alignment_map_values"):
            equality["da3."+key] = bool(np.array_equal(a[key], b[key]))
    sys.path.insert(0, str(ROOT / "reports/latency-optimization-20261018"))
    from bake_export_compare import compare_glb

    result["glb_comparison"] = compare_glb(reference / "scene.glb", out / "scene.glb")
    result.update(output_equal=equality, comparison_s=time.perf_counter()-ready)
    good_glb = all(result["glb_comparison"][k] for k in (
        "document_associations_equal", "accessor_bytes_equal", "decoded_rgba_equal"))
    result["status"] = "complete" if all(equality.values()) and good_glb else "output_difference"
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("status", "mode", "total_ready_s", "viewer_ready_s", "zip_ready_s", "glb_s")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
