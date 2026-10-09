"""Independent source-camera ray coverage on exported float32 geometry.

No candidate selection code is imported here. Reference depths are model estimates,
so this measures consistency and input-view coverage, not reconstruction accuracy.
"""
import argparse
import json
import time

import numpy as np
import open3d as o3d

from perf.surfel_coverage.common import EVAL_FRAMES, OUT, load_sparse


def pixel_metrics(rendered_depth, has_hit, textured, reference_depth, reference_valid, spacing):
    tolerance = np.maximum(2 * spacing, .005 * reference_depth)
    matched = reference_valid & has_hit & (np.abs(rendered_depth - reference_depth) <= tolerance)
    front = reference_valid & has_hit & (rendered_depth < reference_depth - tolerance)
    behind = reference_valid & has_hit & (rendered_depth > reference_depth + tolerance)
    counts = dict(reference_valid=int(reference_valid.sum()), matched=int(matched.sum()),
        textured_matched=int((matched & textured).sum()), inconsistent_nearer=int(front.sum()),
        inconsistent_farther=int(behind.sum()), missing=int((reference_valid & ~has_hit).sum()),
        outside_reference_support=int((~reference_valid & has_hit).sum()),
        reference_invalid=int((~reference_valid).sum()))
    return counts


def fractions(counts):
    total = max(counts["reference_valid"], 1)
    return {"depth_supported_coverage": counts["matched"] / total,
            "textured_supported_coverage": counts["textured_matched"] / total,
            "inconsistent_nearer_fraction": counts["inconsistent_nearer"] / total,
            "inconsistent_farther_fraction": counts["inconsistent_farther"] / total,
            "missing_fraction": counts["missing"] / total,
            "outside_support_fraction": counts["outside_reference_support"] / max(counts["reference_invalid"], 1)}


def make_rays(intrinsic, extrinsic, shape):
    h, w = shape
    y, x = np.indices((h, w))
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
    rays = np.empty((h, w, 6), dtype=np.float32)
    rays[..., :3] = -rotation.T @ translation
    # Direction has camera-Z=1: the ray parameter is source-camera Z depth.
    rays[..., 3:] = pixels @ np.linalg.inv(intrinsic).T @ rotation
    return rays


def evaluate(method):
    start = time.monotonic()
    folder = OUT / method
    if (folder / "evaluation.json").exists():
        raise FileExistsError(folder / "evaluation.json")
    data = load_sparse()
    with np.load(folder / "mesh.npz", allow_pickle=False) as archive:
        vertices, faces, sources = (archive[key] for key in ("vertices", "faces", "sources"))
    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
                        o3d.core.Tensor(faces.astype(np.uint32), dtype=o3d.core.Dtype.UInt32))
    _, h, w = data["depth"].shape
    rows = []
    for frame in EVAL_FRAMES:
        rays = make_rays(data["K"][frame], data["E"][frame], (h, w))
        result = scene.cast_rays(o3d.core.Tensor(rays), nthreads=8)
        z = result["t_hit"].numpy()
        primitives = result["primitive_ids"].numpy()
        hit = np.isfinite(z)
        textured = np.zeros_like(hit)
        textured[hit] = sources[primitives[hit]] >= 0
        counts = pixel_metrics(z, hit, textured, data["depth"][frame], data["mask"][frame], float(data["spacing"]))
        rows.append(dict(frame=frame, image_id=int(data["image_ids"][frame]), counts=counts, **fractions(counts)))
        print(f"{method} frame {frame}: {fractions(counts)}", flush=True)
    totals = {key: sum(row["counts"][key] for row in rows) for key in rows[0]["counts"]}
    report = dict(method=method, frames=list(EVAL_FRAMES), per_frame=rows, counts=totals,
        pooled=fractions(totals), seconds=time.monotonic() - start,
        protocol="First-hit double-sided discs at original integer pixel rays, max(2*spacing,0.005*reference_Z) agreement; fixed seven input views.",
        limits="Model depth reference, not ground truth or held-out images. Outside support includes uncertain reference pixels, not necessarily errors. Texture coverage means first-hit face has an assigned source; does not assess text alignment.")
    (folder / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"method": method, "pooled": report["pooled"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("uniform", "fused", "texture", "adaptive", "coverage", "coverage_bounded"))
    evaluate(parser.parse_args().method)
