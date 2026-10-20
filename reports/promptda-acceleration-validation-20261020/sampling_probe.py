"""CPU-only experiment; selected-point validation is weaker than production.

No model imports. Importable candidate returns WORLD coordinates before display.
source_indices index the original depth tensor flattened in frame/C order.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "sampling_probe_scanner_io", ROOT / "scripts/3d/scanner/scanner_io.py"
)
_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native)


def _mask(prediction, sensor, max_points):
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if not np.array_equal(prediction["frame_ids"], sensor["frame_ids"]):
        raise ValueError("Prediction and Scanner frame IDs differ")
    return _native.prediction_mask(prediction)


def _selection(before, max_points, seed):
    if not before:
        raise ValueError("Cannot export an empty or nonfinite point cloud")
    if before > max_points:
        return np.random.default_rng(seed).choice(before, max_points, replace=False)
    return np.arange(before)


def select_before_backprojection(prediction, sensor, max_points=1_000_000, seed=42):
    """Same selected source domain/order; full-world finite check NOT preserved."""
    return _select(prediction, sensor, max_points, seed, direct_float32=False)


def select_before_backprojection_float32_pixels(
    prediction, sensor, max_points=1_000_000, seed=42
):
    """Separate variant: additionally remove the float64 pixel temporary."""
    return _select(prediction, sensor, max_points, seed, direct_float32=True)


def _select(prediction, sensor, max_points, seed, direct_float32):
    mask, threshold = _mask(prediction, sensor, max_points)
    depth = prediction["depth"]
    # Retain the extra backproject validity predicate, including conf masking.
    valid = mask & np.isfinite(depth) & (depth > 0)
    counts = np.count_nonzero(valid, axis=(1, 2))
    ends = np.cumsum(counts)
    before = int(ends[-1])
    indices = _selection(before, max_points, seed)
    frames = np.searchsorted(ends, indices, side="right")
    starts = ends - counts
    poses = np.linalg.inv(sensor["w2c"])
    points = np.empty((len(indices), 3), dtype=np.float32)
    colors = np.empty((len(indices), *prediction["processed_images"].shape[3:]),
                      dtype=prediction["processed_images"].dtype)
    source_indices = np.empty(len(indices), dtype=np.int64)
    frame_pixels = depth.shape[1] * depth.shape[2]
    for frame, (d, k, pose, rgb, frame_valid) in enumerate(zip(
        depth, prediction["intrinsics"], poses,
        prediction["processed_images"], valid, strict=True
    )):
        slots = np.flatnonzero(frames == frame)
        # Invert K even for an unsampled frame, preserving singular-K failures.
        inverse_k = np.linalg.inv(k)
        local = np.flatnonzero(frame_valid)[indices[slots] - starts[frame]]
        y, x = np.divmod(local, d.shape[1])
        if direct_float32:
            pixels = np.empty((len(x), 3), dtype=np.float32)
            pixels[:, 0], pixels[:, 1], pixels[:, 2] = x, y, 1
        else:
            pixels = np.column_stack((x, y, np.ones(len(x)))).astype(np.float32)
        camera = (pixels @ inverse_k.T) * d[y, x, None]
        world = camera @ pose[:3, :3].T + pose[:3, 3]
        # Original backproject casts after frame concatenation, before sampling.
        points[slots] = world.astype(np.float32)
        colors[slots] = rgb[y, x]
        source_indices[slots] = frame * frame_pixels + local
    if not np.isfinite(points).all():
        raise ValueError("Cannot export an empty or nonfinite point cloud")
    return points, colors, source_indices, {
        "points_before_limit": before, "points": len(points),
        "confidence_threshold": threshold, "seed": seed,
        "direct_float32_pixels": direct_float32,
        "full_world_finite_validation": False,
        "validation_scope": "selected world points only; experimental, not default replacement",
    }


def _reference(prediction, sensor, max_points, seed, source_domain):
    mask, threshold = _mask(prediction, sensor, max_points)
    points, colors = _native.backproject(
        prediction["depth"], prediction["intrinsics"],
        np.linalg.inv(sensor["w2c"]), prediction["processed_images"], mask
    )
    before = len(points)
    if not before or not np.isfinite(points).all():
        raise ValueError("Cannot export an empty or nonfinite point cloud")
    indices = _selection(before, max_points, seed)
    if before > max_points:
        points, colors = points[indices], colors[indices]
    return points, colors, source_domain[indices], {
        "points_before_limit": before, "points": len(points),
        "confidence_threshold": threshold, "full_world_finite_validation": True,
    }


def _display(points, sensor):
    display = np.diag([1, -1, -1, 1]).astype(np.float32) @ sensor["w2c"][0]
    return points @ display[:3, :3].T + display[:3, 3]


def certify_finite_domain(prediction, sensor):
    """Sufficient bound for this input, not a replacement failure contract.

    Only certifies float32 operands and small exactly represented pixel indices.
    Factor 8 for each 3-term dot and 2 per product/add deliberately dominate
    float32 roundoff, including subnormal absolute errors via a tiny cushion.
    A further 1024x margin below FLT_MAX avoids boundary claims. Unknown does
    not imply nonfinite, and does not trigger a fallback.
    """
    depth = prediction["depth"]
    arrays = (depth, prediction["intrinsics"], sensor["w2c"])
    if any(a.dtype != np.float32 for a in arrays):
        return {"status": "unknown", "reason": "certificate limited to float32 input operands"}
    if max(depth.shape[1:]) > 2**24:
        return {"status": "unknown", "reason": "pixel range exceeds exact float32 integers"}
    try:
        mask, _ = _native.prediction_mask(prediction)
        poses = np.linalg.inv(sensor["w2c"])
        inverses = np.linalg.inv(prediction["intrinsics"])
    except (ValueError, np.linalg.LinAlgError) as error:
        return {"status": "unknown", "reason": str(error)}
    if not np.isfinite(poses).all() or not np.isfinite(inverses).all():
        return {"status": "unknown", "reason": "nonfinite inverse; cannot certify"}
    pixel_bound = np.array([depth.shape[2] - 1, depth.shape[1] - 1, 1], dtype=np.float64)
    max_camera, max_world, count = 0., 0., 0
    cushion = float(np.finfo(np.float32).tiny)
    for d, valid, inverse, pose in zip(depth, mask, inverses, poses, strict=True):
        valid = valid & np.isfinite(d) & (d > 0)
        count += int(np.count_nonzero(valid))
        if not np.any(valid):
            continue
        max_depth = float(np.max(d, where=valid, initial=0))
        ray_bound = 8 * (pixel_bound @ np.abs(inverse.astype(np.float64)).T) + cushion
        camera_bound = 2 * ray_bound * max_depth + cushion
        dot_bound = 8 * (camera_bound @ np.abs(pose[:3, :3].astype(np.float64)).T) + cushion
        world_bound = 2 * (dot_bound + np.abs(pose[:3, 3].astype(np.float64))) + cushion
        max_camera = max(max_camera, float(np.max(ray_bound)), float(np.max(camera_bound)))
        max_world = max(max_world, float(np.max(dot_bound)), float(np.max(world_bound)))
    largest = max(max_camera, max_world)
    limit = float(np.finfo(np.float32).max) / 1024
    certified = count > 0 and np.isfinite(largest) and largest < limit
    return {"status": "true" if certified else "unknown", "valid_points": count,
            "max_camera_intermediate_abs_bound": max_camera if np.isfinite(max_camera) else None,
            "max_world_intermediate_abs_bound": max_world if np.isfinite(max_world) else None,
            "conservative_acceptance_limit": limit,
            "reason": "absolute bounds dominate every three-term dot partial sum, depth product, world sum and final float32 cast" if certified else "sufficient bound did not establish finiteness",
            "scope": "world coordinates only; runtime candidate still checks selected points only"}


def _coordinate_diff(a, b):
    bits_equal = a.dtype == b.dtype and np.array_equal(a.view(np.uint8), b.view(np.uint8))
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"bitwise_equal": bits_equal, "numeric_differing_components": int(np.count_nonzero(a != b)),
            "bitwise_differing_components": int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))),
            "max_abs_m": float(delta.max(initial=0)), "dtype_a": str(a.dtype), "dtype_b": str(b.dtype)}


def _compare(reference, candidate, sensor):
    ids_equal = np.array_equal(reference[2], candidate[2])
    colors_equal = np.array_equal(reference[1], candidate[1])
    result = {"source_indices_and_order_equal": ids_equal, "rgb_and_order_equal": colors_equal,
              "world_coordinates": _coordinate_diff(reference[0], candidate[0]),
              "display_coordinates": _coordinate_diff(_display(reference[0], sensor), _display(candidate[0], sensor))}
    if not ids_equal or not colors_equal:
        raise AssertionError(result)
    return result


def unselected_overflow_check():
    """Finite positive unselected depth can overflow a world point: weaker check."""
    depth = np.ones((1, 1, 4), dtype=np.float32)
    selected = int(np.random.default_rng(42).choice(4, 1, replace=False)[0])
    abnormal = next(i for i in range(1, 4) if i != selected)
    depth[0, 0, abnormal] = np.finfo(np.float32).max
    prediction = {"depth": depth, "intrinsics": np.diag([.1, 1, 1]).astype(np.float32)[None],
                  "processed_images": np.zeros((1, 1, 4, 3), dtype=np.uint8), "frame_ids": np.array([1])}
    sensor = {"frame_ids": np.array([1]), "w2c": np.eye(4, dtype=np.float32)[None]}
    outcomes = {}
    with np.errstate(over="ignore", invalid="ignore"):
        for name, call in {
            "reference": lambda: _reference(prediction, sensor, 1, 42, np.arange(4)),
            "candidate": lambda: select_before_backprojection(prediction, sensor, 1, 42),
        }.items():
            try:
                call()
                outcomes[name] = "accepted"
            except ValueError as error:
                outcomes[name] = f"rejected: {error}"
    return {"selected_source": selected, "unselected_abnormal_source": abnormal,
            "depth_finite_positive": True, "outcomes": outcomes,
            "semantic_equivalence": False}


def export_candidate_once(result, sensor, path):
    """Optional single native trimesh writer + one native format-boundary readback."""
    import trimesh

    points, colors = _display(result[0], sensor), result[1]
    cloud = trimesh.points.PointCloud(points.astype(np.float32), colors=colors)
    scene = trimesh.Scene()
    scene.add_geometry(cloud, geom_name=path.stem, node_name=path.stem)
    scene.export(str(path))
    loaded = trimesh.load(str(path), force="scene", process=False)
    geometries = list(loaded.geometry.values())
    if len(geometries) != 1 or not isinstance(geometries[0], trimesh.points.PointCloud):
        raise ValueError("GLB must contain exactly one PointCloud")
    restored = geometries[0]
    if len(restored.vertices) != len(points) or not np.isfinite(restored.vertices).all() or len(restored.colors) != len(points):
        raise ValueError("Invalid GLB point/color readback")
    return {"file": str(path), "points": len(points), "bytes": path.stat().st_size,
            "bounds_m": restored.bounds.tolist(), "glb_readback": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--sensor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Scalar JSON path")
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-float32-pixels", action="store_true")
    parser.add_argument("--export-glb", type=Path, help="Optional: one primary candidate GLB")
    args = parser.parse_args()
    prediction = _native.read_npz(args.prediction)
    sensor = _native.read_npz(args.sensor)
    # Reference-only source-ID instrumentation excluded from all timings.
    mask, _ = _mask(prediction, sensor, args.max_points)
    source_domain = np.flatnonzero(mask & np.isfinite(prediction["depth"]) & (prediction["depth"] > 0))
    reference = lambda: _reference(prediction, sensor, args.max_points, args.seed, source_domain)
    variants = {"select_before": select_before_backprojection}
    if args.include_float32_pixels:
        variants["select_before_float32_pixels"] = select_before_backprojection_float32_pixels
    report = {"timing_scope": "arrays already read; validity mask through world selection and source-ID mapping; excludes display, GLB, comparisons",
              "reference_instrumentation": "full source domain precomputed outside timing; selected source-ID gather included",
              "repetitions_per_variant": 3, "prediction": str(args.prediction), "sensor": str(args.sensor),
              "numpy_version": np.__version__, "unselected_overflow_check": unselected_overflow_check(),
              "finite_domain_certificate_outside_timing": certify_finite_domain(prediction, sensor), "variants": {}}
    export_result = None
    for name, function in variants.items():
        candidate = lambda: function(prediction, sensor, args.max_points, args.seed)
        timings = {"reference": [], "candidate": []}
        checks, pairs = [], []
        for order in (("reference", "candidate"), ("candidate", "reference"), ("reference", "candidate")):
            outputs = {}
            for label in order:
                started = time.perf_counter()
                outputs[label] = {"reference": reference, "candidate": candidate}[label]()
                timings[label].append(time.perf_counter() - started)
            checks.append(_compare(outputs["reference"], outputs["candidate"], sensor))
            pairs.append({"order": list(order), "reference_s": timings["reference"][-1], "candidate_s": timings["candidate"][-1]})
            if name == "select_before":
                export_result = outputs["candidate"]
            del outputs
        medians = {label: statistics.median(values) for label, values in timings.items()}
        report["variants"][name] = {"pairs": pairs, "median_s": medians,
            "median_ratio_reference_over_candidate": medians["reference"] / medians["candidate"],
            "checks": checks, "default_replacement_eligible": False,
            "limitation": "full-world finite validation unresolved; unchanged source/RGB/coordinates on input cannot remove this boundary"}
    if args.export_glb:
        report["optional_glb"] = export_candidate_once(export_result, sensor, args.export_glb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({name: data["median_s"] for name, data in report["variants"].items()}))


if __name__ == "__main__":
    main()
