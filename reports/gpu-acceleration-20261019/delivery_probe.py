"""Time production GPU GLB delivery, then compare to an existing native GLB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(2, 4), required=True)
    parser.add_argument("--fixed-shape", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT))
    from src.surfel_glb import export_scene_glb

    imported = time.perf_counter()
    output = args.out / "scene.glb"
    report = export_scene_glb(
        args.generation, output, encoder_workers=args.workers,
        bake_device="cuda:0", bake_fixed_shape=args.fixed_shape,
    )
    ready = time.perf_counter()
    result = dict(status="ready", encoder_workers=args.workers, fixed_shape=args.fixed_shape,
                  generation=str(args.generation), reference=str(args.reference),
                  imports_s=imported-start, export_s=ready-imported, script_to_ready_s=ready-start,
                  report=report)
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    # Numerical validation is outside delivery timing, as in the prior experiments.
    sys.path.insert(0, str(ROOT / "reports/latency-optimization-20261018"))
    from bake_export_compare import compare_glb

    result["comparison"] = compare_glb(args.reference, output)
    result["comparison_s"] = time.perf_counter()-ready
    import torch

    result["allocated_peak_bytes"] = torch.cuda.max_memory_allocated()
    result["reserved_peak_bytes"] = torch.cuda.max_memory_reserved()
    result["status"] = "complete" if all(
        result["comparison"].get(key) is True
        for key in ("document_associations_equal", "accessor_count_equal",
                    "accessor_bytes_equal", "image_count_equal", "decoded_rgba_equal")
    ) else "comparison_failed"
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
