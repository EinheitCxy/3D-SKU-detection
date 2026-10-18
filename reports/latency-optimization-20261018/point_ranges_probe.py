"""One exact endpoint comparison on the final labels captured in the paired run."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def function(path):
    tree = ast.parse(path.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_attach_point_index_ranges")
    module = ast.Module(body=[definition], type_ignores=[])
    scope = {"np": np, "Any": Any}
    exec(compile(module, str(path), "exec"), scope)
    return scope[definition.name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Use CPU only for this endpoint comparison")
    labels = np.load(args.case / "range_labels.npy", allow_pickle=False)
    keys = json.loads((args.case / "range_keys.json").read_text())
    baseline = function(ROOT / "runtime/latency-optimization-20261018/baseline-code/src/web_viewer_export.py")
    candidate = function(ROOT / "src/web_viewer_export.py")
    results = {}
    endpoints = {}
    for name, helper in (("baseline", baseline), ("candidate", candidate)):
        objects = {}
        for gid, index in keys:
            group = objects.setdefault(gid, {"instances": []})["instances"]
            while len(group) <= index:
                group.append({})
        before = time.perf_counter()
        helper(objects, labels, keys)
        results[name] = time.perf_counter() - before
        endpoints[name] = [objects[gid]["instances"][index]["point_index_range"] for gid, index in keys]
    equal = endpoints["baseline"] == endpoints["candidate"]
    report = dict(status="pass" if equal else "fail", case=str(args.case),
                  labels_count=len(labels), labels_dtype=str(labels.dtype), observations=len(keys),
                  timings_s=results, range_endpoints_equal=equal,
                  scope="One same-array CPU component comparison; not an end-to-end speedup")
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not equal:
        raise RuntimeError("Point range endpoints changed")


if __name__ == "__main__":
    import numpy as np
    main()
