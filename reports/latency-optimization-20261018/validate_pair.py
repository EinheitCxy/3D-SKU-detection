"""Validate the actual fixed-input pair, without hashes or a regression sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def validate(baseline: Path, candidate: Path):
    import numpy as np

    sys.path.insert(0, str(ROOT / "reports/profiling-multivideo-20261018"))
    from build_summary import resource_summary
    result = dict(status="running", numeric_outputs={}, viewer_arrays={}, resources={},
                  scope="Actual same-input video3 output/format/resource checks; no hashes, detector/classifier/browser/accuracy benchmark")
    for name, case in (("baseline", baseline), ("candidate", candidate)):
        receipt = json.loads((case / "receipt.json").read_text())
        if receipt["status"] != "complete":
            raise ValueError(name + " is not complete")
        events = [json.loads(line) for path in case.glob("events-*.jsonl")
                  for line in path.read_text().splitlines() if line.strip()]
        result["resources"][name] = resource_summary(case, receipt, events)
        pipeline = json.loads((case / "pipeline-summary.json").read_text())
        if not pipeline or not all(value is True for value in pipeline.values()):
            raise ValueError(name + " pipeline did not complete all stages")

    outputs = "outputs/dataset/da3_cache/predictions.npz"
    keys = ("depth", "depth_conf", "world_points", "world_points_conf", "extrinsic", "intrinsic",
            "images", "image_ids", "source_image_sizes", "source_to_processed_affine",
            "is_metric", "scale_factor", "frame_alignment_sorted_indices",
            "frame_alignment_map_keys", "frame_alignment_map_values")
    with np.load(baseline / outputs, allow_pickle=False) as before, np.load(candidate / outputs, allow_pickle=False) as after:
        for key in keys:
            a, b = before[key], after[key]
            equal = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
            result["numeric_outputs"][key] = dict(equal=equal, shape=list(a.shape), dtype=str(a.dtype))
            if not equal:
                result["numeric_outputs"][key]["changed_values"] = int(np.count_nonzero(a != b)) if a.shape == b.shape else None
    result["da3_outputs_exact"] = all(row["equal"] for row in result["numeric_outputs"].values())
    mapping = "outputs/dataset/dedup_detections/global_mapping.json"
    result["global_mapping_equal"] = json.loads((baseline / mapping).read_text()) == json.loads((candidate / mapping).read_text())

    generations = []
    for case in (baseline, candidate):
        run_id = json.loads((case / "viewer/CURRENT").read_text())["run_id"]
        generations.append(case / "viewer/runs" / run_id)
    left, right = generations
    binaries = ("positions.f32.bin", "colors.u8.bin", "normals.i8.bin", "surfel-u.f16.bin",
                "surfel-v.f16.bin", "surfel-scale.f16.bin", "surfel-frame.u8.bin", "surfel-depth.f16.bin")
    for name in binaries:
        result["viewer_arrays"][name] = dict(equal=(left / name).read_bytes() == (right / name).read_bytes(),
                                                   before_bytes=(left / name).stat().st_size,
                                                   after_bytes=(right / name).stat().st_size)
    result["viewer_arrays_exact"] = all(row["equal"] for row in result["viewer_arrays"].values())
    result["viewer_objects_equal"] = json.loads((left / "objects.json").read_text()) == json.loads((right / "objects.json").read_text())
    result["viewer_manifest_equal"] = json.loads((left / "manifest.json").read_text()) == json.loads((right / "manifest.json").read_text())
    result["glb_payload_equal"] = (baseline / "scene.glb").read_bytes() == (candidate / "scene.glb").read_bytes()
    result["status"] = "pass" if result["da3_outputs_exact"] and result["global_mapping_equal"] else "fail"
    result["interpretation"] = "DA3/matching equality is a hard check. Viewer/GLB equality is reported separately because native randomized plane fitting remains enabled; differences must be investigated rather than silently accepted."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.baseline, args.candidate)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("resources", "numeric_outputs", "viewer_arrays")}, indent=2))
    if result["status"] != "pass":
        raise RuntimeError("Paired model/matching outputs differ")


if __name__ == "__main__":
    main()
