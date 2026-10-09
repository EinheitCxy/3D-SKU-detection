"""Conservative fixed-camera, same-product surfel fusion experiment.

Run from the repository root with ``uv run python -m perf.roi_fusion.fusion``.
No camera, full-scene depth, mask or input observation is changed in place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree


def coherent_surfels(data):
    """Derive actual geometric normals and orient the complete tangent basis."""
    points = np.asarray(data["points"], dtype=np.float64)
    u = np.asarray(data["u"], dtype=np.float64).copy()
    v = np.asarray(data["v"], dtype=np.float64).copy()
    normals = np.cross(u, v)
    lengths = np.linalg.norm(normals, axis=1)
    if not (np.isfinite(points).all() and np.isfinite(lengths).all()) or np.any(lengths < 1e-14):
        raise ValueError("Nonfinite points or degenerate surfel tangent bases")
    normals /= lengths[:, None]
    poses = np.asarray(data["E"])
    cameras = -np.einsum("fji,fj->fi", poses[:, :3, :3], poses[:, :3, 3])
    facing = cameras[np.asarray(data["frame"], dtype=int)] - points
    flip = np.einsum("ij,ij->i", normals, facing) < 0
    normals[flip] *= -1
    v[flip] *= -1
    return {"points": points, "u": u, "v": v, "normals": normals,
            "frame": np.asarray(data["frame"], dtype=np.int32).copy()}


def _project(points, data, frame):
    pose = np.asarray(data["E"])[frame]
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    homogeneous = camera @ np.asarray(data["K"])[frame].T
    xy = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-12)
    return xy, camera[:, 2]


def _visible(points, normals, data, frame, tolerance):
    xy, z = _project(points, data, frame)
    height, width = np.asarray(data["depth"])[frame].shape
    px = np.rint(xy).astype(np.int64)
    inside = (z > 0) & (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
    x = np.clip(px[:, 0], 0, width - 1)
    y = np.clip(px[:, 1], 0, height - 1)
    depth = np.asarray(data["depth"])[frame, y, x]
    product = np.asarray(data["mask"])[frame, y, x]
    pose = np.asarray(data["E"])[frame]
    camera = -pose[:3, :3].T @ pose[:3, 3]
    direction = camera - points
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    front = np.einsum("ij,ij->i", normals, direction) > 0.05
    return inside & product & np.isfinite(depth) & (depth > 0) & (np.abs(depth - z) <= tolerance) & front


def _geometric_gate(points_a, points_b, normals_a, normals_b, scales):
    delta = points_a - points_b
    alignment = np.einsum("ij,ij->i", normals_a, normals_b)
    normal_distance = np.maximum(np.abs(np.einsum("ij,ij->i", delta, normals_a)),
                                 np.abs(np.einsum("ij,ij->i", delta, normals_b)))
    distance = np.linalg.norm(delta, axis=1)
    return (alignment >= np.cos(np.deg2rad(25))) & (normal_distance <= 1.5 * scales) & (distance <= 2.5 * scales)


def fuse_surfels(data):
    """Fuse reciprocal cross-frame associations, retaining every singleton.

    A cluster has at most one observation from any frame and every pair passes
    the geometric gate. This bounds transitive drift across neighboring faces.
    """
    start = time.monotonic()
    surfels = coherent_surfels(data)
    points, normals, frame = (surfels[key] for key in ("points", "normals", "frame"))
    size = len(points)
    if size == 0:
        raise ValueError("Fusion requires at least one observation")
    spacing = float(np.asarray(data["spacing"]))
    scales = np.sqrt(np.linalg.norm(surfels["u"], axis=1) * np.linalg.norm(surfels["v"], axis=1))
    # Bound grazing/elongated footprints so large tangents cannot bridge faces.
    scales = np.clip(scales, 0.5 * spacing, 2 * spacing)
    frame_ids = np.unique(frame)
    indices = {int(f): np.flatnonzero(frame == f) for f in frame_ids}
    trees = {f: cKDTree(np.asarray(data["xy"])[idx]) for f, idx in indices.items()}
    edges, pair_stats = [], []
    for position, a in enumerate(frame_ids):
        ia = indices[int(a)]
        for b in frame_ids[position + 1:]:
            ib = indices[int(b)]
            projected, _ = _project(points[ia], data, int(b))
            distance, nearest = trees[int(b)].query(projected, distance_upper_bound=1.5)
            valid = np.isfinite(distance)
            left, right = ia[valid], ib[nearest[valid]]
            initial_count = len(left)
            if len(left):
                backward, _ = _project(points[right], data, int(a))
                back_distance, back_nearest = trees[int(a)].query(backward, distance_upper_bound=1.5)
                mutual = np.isfinite(back_distance)
                mutual[mutual] &= ia[back_nearest[mutual]] == left[mutual]
                left, right = left[mutual], right[mutual]
                pair_scale = np.minimum(scales[left], scales[right])
                keep = _geometric_gate(points[left], points[right], normals[left], normals[right], pair_scale)
                # Both observations must be visible in both original cameras.
                for f in (int(a), int(b)):
                    keep &= _visible(points[left], normals[left], data, f, 2 * pair_scale)
                    keep &= _visible(points[right], normals[right], data, f, 2 * pair_scale)
                left, right, pair_scale = left[keep], right[keep], pair_scale[keep]
                score = np.linalg.norm(points[left] - points[right], axis=1) / pair_scale
                edges.extend(zip(score.tolist(), left.tolist(), right.tolist()))
            pair_stats.append({"frames": [int(a), int(b)], "projected_candidates": initial_count,
                               "accepted_reciprocal_pairs": len(left)})

    parent = np.arange(size)
    members = {i: [i] for i in range(size)}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    rejected_cluster = 0
    for _, left, right in sorted(edges):
        a, b = root(left), root(right)
        if a == b:
            continue
        ma, mb = np.asarray(members[a]), np.asarray(members[b])
        if np.intersect1d(frame[ma], frame[mb]).size:
            rejected_cluster += 1
            continue
        # Complete-link normal/distance constraints prohibit chain-like growth.
        aa, bb = np.repeat(ma, len(mb)), np.tile(mb, len(ma))
        if not _geometric_gate(points[aa], points[bb], normals[aa], normals[bb],
                               np.minimum(scales[aa], scales[bb])).all():
            rejected_cluster += 1
            continue
        parent[b] = a
        members[a].extend(members.pop(b))

    confidence = np.asarray(data["confidence"], dtype=float)
    if not np.isfinite(confidence).all() or np.any(confidence < 0):
        raise ValueError("Confidence must be finite and nonnegative")
    groups = sorted(members.values(), key=min)
    output = {key: [] for key in ("points", "u", "v", "normals", "frame", "support", "confidence")}
    membership = np.empty(size, dtype=np.int32)
    max_displacements = []
    for cluster, group in enumerate(groups):
        idx = np.asarray(group)
        membership[idx] = cluster
        if len(idx) == 1:
            i = idx[0]
            for key in ("points", "u", "v", "normals", "frame"):
                output[key].append(surfels[key][i])
            output["confidence"].append(confidence[i])
            output["support"].append(1)
            max_displacements.append(0.0)
            continue
        weights = np.maximum(confidence[idx], 1e-6)
        median = np.median(points[idx], axis=0)
        residual = np.linalg.norm(points[idx] - median, axis=1)
        weights *= np.minimum(1.0, np.median(scales[idx]) / np.maximum(residual, 1e-12))
        weights /= weights.sum()
        center = weights @ points[idx]
        normal = weights @ normals[idx]
        normal /= np.linalg.norm(normal)
        primary = idx[np.argmax(weights)]
        tangent = surfels["u"][primary] - normal * np.dot(surfels["u"][primary], normal)
        tangent /= np.linalg.norm(tangent)
        # Rotate the primary ellipse basis into the fused tangent plane without
        # averaging incompatible image-space axis directions or changing shear.
        old_u = surfels["u"][primary] / np.linalg.norm(surfels["u"][primary])
        old_v = surfels["v"][primary]
        u = tangent * (weights @ np.linalg.norm(surfels["u"][idx], axis=1))
        v_length = weights @ np.linalg.norm(surfels["v"][idx], axis=1)
        shear = np.clip(np.dot(old_u, old_v) / np.linalg.norm(old_v), -0.999, 0.999)
        v = (shear * tangent + np.sqrt(1 - shear ** 2) * np.cross(normal, tangent)) * v_length
        for key, value in (("points", center), ("normals", normal), ("u", u), ("v", v),
                           ("frame", frame[primary]), ("support", len(idx)),
                           ("confidence", weights @ confidence[idx])):
            output[key].append(value)
        max_displacements.append(float(np.linalg.norm(points[idx] - center, axis=1).max()))
    output = {key: np.asarray(values) for key, values in output.items()}
    output["membership"] = membership
    output["member_offsets"] = np.r_[0, np.cumsum([len(g) for g in groups])]
    output["member_indices"] = np.concatenate([np.asarray(g, dtype=np.int32) for g in groups])
    stats = {"method": "reciprocal_projective_surfel_fusion", "input_observations": size,
             "output_surfels": len(groups), "merged_observations": size - len(groups),
             "unsupported_observations_preserved": int(np.sum(output["support"] == 1)),
             "support_histogram": {str(i): int(np.sum(output["support"] == i)) for i in np.unique(output["support"])},
             "pair_associations": pair_stats, "rejected_cluster_edges": rejected_cluster,
             "fused_displacement_max_m": float(max(max_displacements)),
             "thresholds": {"normal_degrees": 25, "reciprocal_pixels": 1.5,
                            "normal_distance_local_spacing": 1.5, "distance_local_spacing": 2.5,
                            "depth_visibility_local_spacing": 2, "front_facing_dot_min": 0.05,
                            "local_spacing_clip_global": [0.5, 2]},
             "cameras_modified": False, "geometry_seconds": time.monotonic() - start}
    return output, stats


def disc_mesh(points, u, v):
    """Common baseline/fusion geometry: eight-sided discs at radius 1.05."""
    angle = np.arange(8) * (2 * np.pi / 8)
    ring = points[:, None] + 1.05 * (np.cos(angle)[None, :, None] * u[:, None] +
                                   np.sin(angle)[None, :, None] * v[:, None])
    vertices = np.concatenate([points[:, None], ring], axis=1).reshape(-1, 3)
    unit_faces = np.array([[0, 1 + i, 1 + (i + 1) % 8] for i in range(8)])
    faces = (unit_faces[None] + (np.arange(len(points)) * 9)[:, None, None]).reshape(-1, 3)
    return vertices, faces


def main():
    from perf.roi_fusion.texture import choose_sources, export_textured_mesh
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runtime/roi-fusion-video3-gid3/input.npz"))
    parser.add_argument("--output", type=Path, default=Path("runtime/roi-fusion-video3-gid3"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    result, stats = fuse_surfels(data)
    np.savez_compressed(args.output / "fused.npz", **result)
    for name, surfels in (("baseline", coherent_surfels(data)), ("fused", result)):
        vertices, faces = disc_mesh(surfels["points"], surfels["u"], surfels["v"])
        if name == "baseline":
            sources, source_stats = choose_sources(
                vertices, faces, data, preferred_sources=np.repeat(data["frame"], 8)
            )
            source_mode = "original-source"
        else:
            sources, source_stats = choose_sources(vertices, faces, data)
            source_mode = "fixed-spatial-patches"
        export_stats = export_textured_mesh(vertices, faces, sources, data, args.output / f"{name}.glb")
        method_stats = {"method": name, "surfels": len(surfels["points"]), "disc_sides": 8,
                        "disc_radius": 1.05, "texture": source_stats, "export": export_stats,
                        "source_mode": source_mode,
                        "renderer": "common experimental triangulated discs; not production splatting"}
        if name == "fused":
            method_stats.update(stats)
        (args.output / f"{name}-stats.json").write_text(json.dumps(method_stats, indent=2) + "\n")
        print(json.dumps({"method": name, "surfels": len(surfels["points"]), "output": str(args.output / f"{name}.glb")}))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
