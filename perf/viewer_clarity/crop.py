"""Deterministic center crop of a Viewer generation for the registered evaluation."""

import json
import argparse
from pathlib import Path
import shutil

import numpy as np


_POINT_FILES = {
    "positions.f32.bin": ("<f4", 3), "colors.u8.bin": ("u1", 3),
    "normals.i8.bin": ("i1", 3), "surfel-u.f16.bin": ("<f2", 3),
    "surfel-v.f16.bin": ("<f2", 3), "surfel-scale.f16.bin": ("<f2", 2),
    "surfel-frame.u8.bin": ("u1", 1),
}


def crop_bundle(generation_dir: Path, out_dir: Path, *, center_fraction: float = 0.4) -> dict:
    """Keep product-centered fraction of the world-to-view XY bounds and all depth."""
    source, target = Path(generation_dir), Path(out_dir)
    if not 0 < center_fraction <= 1:
        raise ValueError("center_fraction must be in (0, 1]")
    shutil.copytree(source, target)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    objects = json.loads((source / "objects.json").read_text(encoding="utf-8"))
    surfel = json.loads((source / "surfel.json").read_text(encoding="utf-8"))
    count = int(surfel["point_count"])
    positions = np.fromfile(source / "positions.f32.bin", dtype="<f4").reshape(count, 3)
    product = np.zeros(count, dtype=bool)
    for entry in objects.values():
        for start, end in entry.get("point_ranges", []):
            product[int(start):int(end)] = True
    if not product.any():
        raise ValueError("generation has no product points to center crop")
    transform = np.asarray(manifest["world_to_view"], dtype=np.float64).reshape(4, 4)
    view = positions.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    box = np.percentile(view[product], [1, 99], axis=0)
    center = box.mean(axis=0)
    half = (box[1] - box[0]) * (center_fraction / 2)
    keep = ((view[:, 0] >= center[0] - half[0]) & (view[:, 0] <= center[0] + half[0]) &
            (view[:, 1] >= center[1] - half[1]) & (view[:, 1] <= center[1] + half[1]))
    mapping = np.full(count, -1, dtype=np.int64)
    mapping[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    removed_zero_ids = []
    new_objects = {}
    for gid, entry in objects.items():
        copied = dict(entry)
        ranges = []
        for start, end in entry.get("point_ranges", []):
            selected = np.flatnonzero(keep[int(start):int(end)]) + int(start)
            if len(selected):
                new_ids = mapping[selected]
                # Each source range is contiguous in label-sorted point order.
                splits = np.flatnonzero(np.diff(new_ids) != 1) + 1
                for run in np.split(new_ids, splits):
                    if len(run):
                        ranges.append([int(run[0]), int(run[-1]) + 1])
        copied["point_ranges"] = ranges
        if not ranges:
            removed_zero_ids.append(str(gid))
        else:
            new_objects[str(gid)] = copied
    (target / "objects.json").write_text(json.dumps(new_objects, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for filename, (dtype, width) in _POINT_FILES.items():
        path = source / filename
        if not path.exists():
            if filename == "surfel-scale.f16.bin" and surfel["version"] == 2:
                continue
            raise FileNotFoundError(path)
        data = np.fromfile(path, dtype=dtype).reshape(count, width)[keep]
        data.tofile(target / filename)
    new_count = int(keep.sum())
    surfel["point_count"] = new_count
    (target / "surfel.json").write_text(json.dumps(surfel, separators=(",", ":")), encoding="utf-8")
    bounds = np.percentile(positions[keep].astype(np.float64), [1, 99], axis=0)
    manifest["display_bounds"] = [*bounds[0].tolist(), *bounds[1].tolist()]
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    # Keep only thumbnails referenced by surviving object observations.
    referenced = {obs["thumbnail"] for entry in new_objects.values() for obs in entry.get("observations", []) if obs.get("thumbnail")}
    thumb_dir = target / "thumbs"
    if thumb_dir.exists():
        for path in thumb_dir.iterdir():
            if path.relative_to(target).as_posix() not in referenced:
                path.unlink()
    result = {"point_count": new_count, "product_points": int(product[keep].sum()), "background_points": int((~product & keep).sum()), "removed_zero_gids": removed_zero_ids}
    (target / "crop.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--center-fraction", type=float, default=0.4)
    args = parser.parse_args()
    result = crop_bundle(args.bundle_dir, args.out_dir, center_fraction=args.center_fraction)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
