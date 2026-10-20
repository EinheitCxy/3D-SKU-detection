"""Direct fresh-run NPZ and GLB semantic output comparison; no hashes."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import trimesh

    report = {"status": "running", "scope": "Direct comparison of the two new cold-delivery NPZ arrays and GLB geometry/color/node transforms; not image-based visual or independent accuracy evaluation"}
    started = time.perf_counter()
    try:
        rows = []
        with np.load(args.native / "predictions.npz", allow_pickle=False) as left, np.load(args.candidate / "predictions.npz", allow_pickle=False) as right:
            assert set(left.files) == set(right.files), "NPZ fields differ"
            for key in left.files:
                a, b = left[key], right[key]
                assert a.dtype == b.dtype and a.shape == b.shape, key
                equal = np.array_equal(a, b)
                row = {"field": key, "dtype": str(a.dtype), "shape": list(a.shape), "array_equal": equal}
                if a.dtype.kind == "f":
                    row["max_abs"] = float(np.max(np.abs(a - b), initial=0))
                rows.append(row)
                assert equal, f"NPZ values differ: {key}"
        report["npz_fields"] = rows
        left = trimesh.load(str(args.native / "cloud.glb"), force="scene", process=False)
        right = trimesh.load(str(args.candidate / "cloud.glb"), force="scene", process=False)
        assert set(left.geometry) == set(right.geometry), "GLB geometry names differ"
        geometries = []
        for key, a in left.geometry.items():
            b = right.geometry[key]
            assert isinstance(a, trimesh.points.PointCloud) and isinstance(b, trimesh.points.PointCloud)
            positions_equal = np.array_equal(a.vertices, b.vertices)
            colors_equal = np.array_equal(a.colors, b.colors)
            assert positions_equal and colors_equal, "GLB geometry payload differs"
            geometries.append({"name": key, "points": len(a.vertices), "positions_equal": positions_equal,
                               "colors_equal": colors_equal, "max_position_abs_m": float(np.max(np.abs(a.vertices-b.vertices), initial=0))})
        assert set(left.graph.nodes_geometry) == set(right.graph.nodes_geometry), "GLB nodes differ"
        for node in left.graph.nodes_geometry:
            a, ga = left.graph[node]
            b, gb = right.graph[node]
            assert ga == gb and np.array_equal(a, b), "GLB node transform differs"
        report["glb_geometries"] = geometries
        report["glb_node_transforms_equal"] = True
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["comparison_wall_s"] = time.perf_counter() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "npz_fields_equal": len(report["npz_fields"]), "glb": report["glb_geometries"]}))


if __name__ == "__main__":
    main()
