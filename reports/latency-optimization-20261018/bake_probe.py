"""Opt-in bounded CPU comparison of snapshot and candidate Surfel batches.

This performs real data loading/LOD and sampled baking when invoked. It does
not export a complete GLB, initialize CUDA or relax any visibility threshold.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def load_snapshot(path):
    spec = importlib.util.spec_from_file_location("snapshot_surfel_mesh", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_arrays(left, right):
    """Pair surviving quads by exact geometry, including duplicate occurrences."""
    import numpy as np

    old_corners, old_tiles = left
    new_corners, new_tiles = right
    slots = defaultdict(deque)
    for index, corners in enumerate(old_corners):
        slots[corners.tobytes()].append(index)
    pairs = []
    for index, corners in enumerate(new_corners):
        pending = slots[corners.tobytes()]
        if pending:
            pairs.append((pending.popleft(), index))
    before = np.asarray([old_tiles[a] for a, _ in pairs], dtype=np.uint8)
    after = np.asarray([new_tiles[b] for _, b in pairs], dtype=np.uint8)
    result = dict(old_surfel_count=len(old_corners), new_surfel_count=len(new_corners),
                  paired_geometry_count=len(pairs), unmatched_old=len(old_corners)-len(pairs),
                  unmatched_new=len(new_corners)-len(pairs),
                  geometry_order_equal=np.array_equal(old_corners, new_corners),
                  complete_rgba_equal=np.array_equal(old_tiles, new_tiles),
                  paired_texels=0, paired_rgb_channels=0,
                  paired_alpha_texels_changed=0, paired_rgb_texels_changed=0,
                  paired_opaque_rgb_texels_changed=0, paired_rgb_max_abs_change=0,
                  paired_rgb_abs_change_sum=0, paired_opaque_texels=0,
                  paired_opaque_rgb_channels=0, paired_opaque_rgb_max_abs_change=0,
                  paired_opaque_rgb_abs_change_sum=0)
    if pairs:
        rgb_difference = np.any(before[..., :3] != after[..., :3], axis=-1)
        both_opaque = (before[..., 3] != 0) & (after[..., 3] != 0)
        absolute = np.abs(before[..., :3].astype(np.int16)-after[..., :3])
        opaque_absolute = absolute[both_opaque]
        result.update(
            paired_texels=int(before[..., 3].size), paired_rgb_channels=int(absolute.size),
            paired_alpha_texels_changed=int(np.count_nonzero(before[..., 3] != after[..., 3])),
            paired_rgb_texels_changed=int(np.count_nonzero(rgb_difference)),
            paired_opaque_rgb_texels_changed=int(np.count_nonzero(rgb_difference & both_opaque)),
            paired_rgb_max_abs_change=int(absolute.max()),
            paired_rgb_abs_change_sum=int(absolute.sum()),
            paired_opaque_texels=int(both_opaque.sum()),
            paired_opaque_rgb_channels=int(opaque_absolute.size),
            paired_opaque_rgb_max_abs_change=int(opaque_absolute.max()) if opaque_absolute.size else 0,
            paired_opaque_rgb_abs_change_sum=int(opaque_absolute.sum()))
    return result


def synthetic_cases():
    import numpy as np

    yy, xx = np.mgrid[:9, :9]
    image = np.stack((xx*20, yy*20, np.full_like(xx, 100)), axis=-1).astype(np.float32)
    k = np.asarray([[1, 0, 4], [0, 1, 4], [0, 0, 1]], dtype=np.float32)
    cases = []
    # Probe source/depth equality and adjacent float32 values, not relaxed epsilons.
    for edge in (np.float32(-4), np.float32(4)):
        for x in (np.nextafter(edge, -np.inf, dtype=np.float32), edge,
                  np.nextafter(edge, np.inf, dtype=np.float32)):
            cases.append((f"source_edge_{float(x)!r}", [x, 0, 1], [0, 0, 0], 1.))
    limit = np.float32(1) + (np.float32(.015)*np.float32(1)+np.float32(.001))
    for z in (np.nextafter(limit, -np.inf, dtype=np.float32), limit,
              np.nextafter(limit, np.inf, dtype=np.float32)):
        cases.append((f"depth_threshold_{float(z)!r}", [0, 0, z], [.05, .01, .02], .5))
    for angle in (0., .47, 1.2):
        rotation = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                               [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)
        translation = np.asarray([.125, -.25, .375], dtype=np.float32)
        p = (np.asarray([[.02, -.03, 1.]], dtype=np.float32)-translation) @ rotation
        u = np.asarray([[.2, 0, .002]], dtype=np.float32) @ rotation
        v = np.asarray([[0, .2, -.001]], dtype=np.float32) @ rotation
        frame = dict(extrinsic=np.column_stack((rotation, translation)).tolist(),
                     intrinsic=k.tolist(), processed_to_source=np.eye(3).tolist(),
                     source_size=[9, 9], texture_size=[9, 9])
        yield f"rotation_{angle}", (p, u, v, np.ones((1, 2), dtype=np.float32),
                                   np.ones((9, 9), dtype=np.float32), frame, image, 2., 8)
    for label, point, u_axis, scale in cases:
        frame = dict(extrinsic=np.eye(4)[:3].tolist(), intrinsic=k.tolist(),
                     processed_to_source=np.eye(3).tolist(), source_size=[9, 9], texture_size=[9, 9])
        # Edge cases use a nondegenerate subpixel disk, not a zero-area axis.
        if not any(u_axis):
            u_axis = [.2, 0, 0]
        yield label, (np.asarray([point], dtype=np.float32),
                      np.asarray([u_axis], dtype=np.float32),
                      np.asarray([[0, .2, 0]], dtype=np.float32),
                      np.full((1, 2), scale, dtype=np.float32),
                      np.ones((9, 9), dtype=np.float32), frame, image, 2., 8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output JSON")
    parser.add_argument("--baseline", type=Path, default=ROOT / "runtime/latency-optimization-20261018/baseline-code/src/surfel_mesh.py")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--points-per-frame", type=int, default=256)
    parser.add_argument("--projection-mode", choices=("source_world", "camera_basis", "both"),
                        default="both", help="Compare one mode, or both against the same snapshot batch")
    args = parser.parse_args()
    if args.frames < 1 or args.points_per_frame < 1:
        parser.error("Sample bounds must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        result = dict(status="failed", generation=str(args.generation), baseline=str(args.baseline),
                      projection_modes=(["source_world", "camera_basis"] if args.projection_mode == "both"
                                        else [args.projection_mode]),
                      rows=[], semantics="Exact array comparison; sampled batch timings are not full-export speed evidence; no threshold relaxation")
        try:
            sys.path.insert(0, str(ROOT))
            import numpy as np
            from PIL import Image
            from src import surfel_mesh as candidate
            from src.surfel_glb import _assets, _load
            from src.surfel_lod import select_surfel_lod
            baseline = load_snapshot(args.baseline)

            def run(label, values):
                tick = time.perf_counter()
                before = baseline._bake_batch(*values)
                old_wall = time.perf_counter()-tick
                constants = candidate._frame_constants(values[5], values[8])
                for mode in result["projection_modes"]:
                    tick = time.perf_counter()
                    after = candidate._bake_batch(*values, constants=constants, projection_mode=mode)
                    row = dict(label=label, projection_mode=mode, input_points=len(values[0]),
                               old_batch_wall_s=old_wall, new_batch_wall_s=time.perf_counter()-tick,
                               **compare_arrays(before, after))
                    result["rows"].append(row)

            for label, values in synthetic_cases():
                run(label, values)
            with _assets(args.generation) as read:
                _, inputs = _load(read, None)
                objects = json.loads(read("objects.json"))
                owners = np.full(len(inputs["positions"]), -1, dtype=np.int32)
                for owner, entry in enumerate(objects.values()):
                    for start, end in entry["point_ranges"]:
                        owners[start:end] = owner
                chosen, result["lod"] = select_surfel_lod(
                    **{key: inputs[key] for key in ("positions", "u", "v", "scales", "frame_indices", "frames", "depths")}, object_ids=owners)
                for key in ("positions", "u", "v", "scales", "frame_indices"):
                    inputs[key] = inputs[key][chosen]
                frame_ids = np.unique(inputs["frame_indices"])
                sample_frames = frame_ids[np.linspace(0, len(frame_ids)-1, min(args.frames, len(frame_ids)), dtype=int)]
                for frame_id in sample_frames:
                    frame = inputs["frames"][int(frame_id)]
                    with Image.open(BytesIO(read(frame["texture"]))) as image:
                        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
                    width = max(f["texture_size"][0] for f in inputs["frames"])
                    height = max(f["texture_size"][1] for f in inputs["frames"])
                    padded = np.zeros((height, width, 3), dtype=np.float32)
                    padded[:rgb.shape[0], :rgb.shape[1]] = rgb
                    slots = np.flatnonzero(inputs["frame_indices"] == frame_id)
                    slots = slots[np.linspace(0, len(slots)-1, min(args.points_per_frame, len(slots)), dtype=int)]
                    values = tuple(np.asarray(inputs[key][slots], dtype=np.float32) for key in ("positions", "u", "v", "scales"))
                    run(f"real_frame_{frame_id}", (*values, np.asarray(inputs["depths"][int(frame_id)], dtype=np.float32), frame, padded, 2., 8))
            result["status"] = "complete"
        except BaseException:
            result["exception"] = traceback.format_exc()
            raise
        finally:
            json.dump(result, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
