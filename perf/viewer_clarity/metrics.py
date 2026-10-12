"""Compute the locked package-level M1/M2/M3/M5/M6/M7/M8 metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _ranges(objects: dict, count: int) -> tuple[np.ndarray, dict[str, int]]:
    labels = np.full(count, -1, dtype=np.int32)
    gid_counts = {}
    for gid, entry in objects.items():
        n = 0
        for start, end in entry.get("point_ranges", []):
            start, end = int(start), int(end)
            if start < 0 or end < start or end > count:
                raise ValueError(f"invalid point range for GID {gid}")
            if end > start:
                labels[start:end] = int(gid)
                n += end - start
        gid_counts[str(gid)] = n
    return labels, gid_counts


def bundle_metrics(bundle_dir: Path, baseline: Path | None = None) -> dict:
    root = Path(bundle_dir)
    surfel = json.loads((root / "surfel.json").read_text(encoding="utf-8"))
    objects = json.loads((root / "objects.json").read_text(encoding="utf-8"))
    count = int(surfel["point_count"])
    positions = np.fromfile(root / "positions.f32.bin", dtype="<f4").reshape(count, 3)
    labels, gid_counts = _ranges(objects, count)
    product = labels >= 0
    frames = np.fromfile(root / "surfel-frame.u8.bin", dtype="u1")
    if len(frames) != count:
        raise ValueError("surfel-frame length differs from point_count")

    # M2: a point has a seam if any of its eight nearest other points is from another frame.
    if count > 1:
        k = min(9, count)
        tree = cKDTree(positions)
        distances, neighbors = tree.query(positions, k=k, workers=-1)
        if neighbors.ndim == 1:
            neighbors = neighbors[:, None]
        other_frames = np.zeros((count, min(8, k - 1)), dtype=np.uint8)
        for index in range(count):
            candidates = neighbors[index]
            candidates = candidates[candidates != index][:8]
            if len(candidates):
                other_frames[index, :len(candidates)] = frames[candidates] != frames[index]
        m2 = float(np.any(other_frames, axis=1).mean()) if other_frames.shape[1] else 0.0
        nearest = cKDTree(positions).query(positions, k=2, workers=-1)[0][:, 1]
        radius = 3.0 * float(np.median(nearest))
        pairs = cKDTree(positions).query_pairs(radius, output_type="ndarray")
        if len(pairs):
            rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(count)])
            cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(count)])
            graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(count, count)).tocsr()
        else:
            graph = coo_matrix((count, count), dtype=np.uint8).tocsr()
            graph.setdiag(1)
        component_count, component_labels = connected_components(graph, directed=False)
        component_sizes = np.bincount(component_labels, minlength=component_count)
        m6 = float(component_sizes[component_sizes < 500].sum() / count)
    else:
        m2, m6 = 0.0, float(count > 0)

    # M3: 8 mm world-space voxels occupied by points from at least two source frames.
    voxel = np.floor(positions.astype(np.float64) / 0.008).astype(np.int64)
    occupied = np.unique(np.column_stack([voxel, frames]), axis=0)
    voxel_ids, voxel_counts = np.unique(occupied[:, :3], axis=0, return_counts=True)
    m3 = float(np.count_nonzero(voxel_counts >= 2) / len(voxel_ids)) if len(voxel_ids) else 0.0

    scale_path = root / "surfel-scale.f16.bin"
    if scale_path.exists():
        scales = np.fromfile(scale_path, dtype="<f2").reshape(count, 2)
        m5 = float((np.min(scales[~product], axis=1) == 0.5).mean()) if np.any(~product) else 0.0
    else:
        m5 = 0.0

    # M7: camera-space depth compared within each (GID, source-frame) group.
    m7_count = 0
    if product.any():
        for frame_id, frame in enumerate(surfel["frames"]):
            selected = product & (frames == frame_id)
            if not selected.any():
                continue
            extrinsic = np.asarray(frame["extrinsic"], dtype=np.float64)
            depth = positions[selected].astype(np.float64) @ extrinsic[:3, :3].T
            depth = depth[:, 2] + extrinsic[2, 3]
            selected_ids = np.flatnonzero(selected)
            for gid in np.unique(labels[selected]):
                local = depth[labels[selected] == gid]
                if len(local):
                    median = float(np.median(local))
                    m7_count += int(np.count_nonzero(local > 1.10 * median))
    m7 = float(m7_count / int(product.sum())) if product.any() else 0.0

    m8 = 0
    if baseline is not None:
        baseline_data = json.loads(Path(baseline).read_text(encoding="utf-8"))
        old_counts = baseline_data.get("gid_point_counts", {})
        m8 = sum(1 for gid, n in old_counts.items() if int(n) > 0 and gid_counts.get(str(gid), 0) == 0)
    result = {
        "point_count": count,
        "product_points": int(product.sum()),
        "background_points": int((~product).sum()),
        "gid_point_counts": gid_counts,
        "M1": {"total": count, "product": int(product.sum()), "background": int((~product).sum())},
        "M2": m2,
        "M3": m3,
        "M5": m5,
        "M6": m6,
        "M7": m7,
        "M8": int(m8),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    result = bundle_metrics(args.bundle_dir, args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
