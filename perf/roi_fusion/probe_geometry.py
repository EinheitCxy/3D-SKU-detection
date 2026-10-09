"""Single-variable TSDF truncation probe on the fixed video3 product ROI.

Run: OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    uv run --no-sync python -m perf.roi_fusion.probe_geometry
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import open3d as o3d
import trimesh

from perf.roi_fusion.evaluate import local_spreads
from perf.roi_fusion.texture import choose_sources, export_textured_mesh


def integrate(data, truncation):
    spacing = float(data["spacing"])
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=spacing, sdf_trunc=truncation * spacing,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        depth_sampling_stride=1,
    )
    for frame in range(len(data["E"])):
        raw = data["depth"][frame]
        valid = data["mask"][frame] & np.isfinite(raw) & (raw > 0)
        depth = np.ascontiguousarray(np.where(valid, raw, 0), dtype=np.float32)
        height, width = depth.shape
        k = data["K"][frame]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8)),
            o3d.geometry.Image(depth), depth_scale=1.,
            # Preserve the baseline input-depth clipping; vary only sdf_trunc.
            depth_trunc=float(depth.max()) + 4 * spacing,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, data["E"][frame])
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):
        raise RuntimeError("ROI TSDF produced no triangles")
    return np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()


def topology(vertices, faces):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    _, counts, areas = mesh.cluster_connected_triangles()
    areas = np.asarray(areas)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                   faces[:, [2, 0]])), axis=1)
    edges, incidence = np.unique(edges, axis=0, return_counts=True)
    boundary = edges[incidence == 1]
    return {
        "vertices": len(vertices), "triangles": len(faces),
        "edge_connected_components": len(counts),
        "largest_component_area_fraction": float(areas.max() / areas.sum()),
        "surface_area_m2": float(areas.sum()),
        "boundary_edges": len(boundary),
        "boundary_length_m": float(np.linalg.norm(
            vertices[boundary[:, 0]] - vertices[boundary[:, 1]], axis=1).sum()),
        "nonmanifold_edges": int((incidence > 2).sum()),
    }


def samples(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rng = np.random.default_rng(42)
    selection = rng.choice(len(faces), size=100000,
                           p=mesh.area_faces / mesh.area_faces.sum())
    triangle = mesh.triangles[selection]
    a = np.sqrt(rng.random((len(selection), 1)))
    b = rng.random((len(selection), 1))
    points = ((1 - a) * triangle[:, 0] + a * (1 - b) * triangle[:, 1]
              + a * b * triangle[:, 2])
    return points, mesh.face_normals[selection]


def silhouettes(input_path, output_root):
    """Compare mesh hits to the existing masks without reintegrating geometry."""
    with np.load(input_path, allow_pickle=False) as opened:
        data = dict(opened)
    result = {
        "input": str(input_path.resolve()), "methods": {},
        "rays": "Exact original processed integer pixel centers K^-1[x,y,1], original w2c poses; unnormalized world directions, t_hit equals camera Z depth.",
        "interpretation": "Agreement with predicted product masks, not ground-truth silhouette or geometry accuracy. Hits count any mesh intersection including backfaces; misses can reflect incomplete observations. Raw counts aggregate all five frames.",
    }
    for name in ("trunc4", "trunc8", "trunc12"):
        with np.load(output_root / name / "tsdf.npz", allow_pickle=False) as geometry:
            vertices, faces = geometry["vertices"], geometry["faces"]
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(
            o3d.core.Tensor(np.ascontiguousarray(vertices, dtype=np.float32)),
            o3d.core.Tensor(np.ascontiguousarray(faces, dtype=np.uint32)))
        records = []
        for frame in range(len(data["E"])):
            mask = data["mask"][frame].astype(bool)
            height, width = mask.shape
            yy, xx = np.indices((height, width))
            camera_directions = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ np.linalg.inv(data["K"][frame]).T
            rotation = data["E"][frame, :3, :3]
            center = -rotation.T @ data["E"][frame, :3, 3]
            directions = camera_directions @ rotation
            origins = np.broadcast_to(center, directions.shape)
            rays = np.ascontiguousarray(np.concatenate((origins, directions), axis=-1), dtype=np.float32)
            depth = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)["t_hit"].numpy()
            hit = np.isfinite(depth) & (depth > 0)
            tp = int((hit & mask).sum())
            fp = int((hit & ~mask).sum())
            mask_pixels = int(mask.sum())
            records.append({
                "frame_local": frame, "image_id": str(data["image_ids"][frame]),
                "mask_pixels": mask_pixels, "hit_inside_mask_pixels": tp,
                "hit_outside_mask_pixels": fp, "missed_mask_pixels": mask_pixels - tp,
                "mask_hit_fraction": tp / mask_pixels,
                "iou": tp / (mask_pixels + fp),
            })
        tp = sum(item["hit_inside_mask_pixels"] for item in records)
        fp = sum(item["hit_outside_mask_pixels"] for item in records)
        mask_pixels = sum(item["mask_pixels"] for item in records)
        result["methods"][name] = {"frames": records, "aggregate": {
            "mask_pixels": mask_pixels, "hit_inside_mask_pixels": tp,
            "hit_outside_mask_pixels": fp, "missed_mask_pixels": mask_pixels - tp,
            "mask_hit_fraction": tp / mask_pixels, "iou": tp / (mask_pixels + fp)}}
    (output_root / "silhouette.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def run(input_path, baseline_root, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    with np.load(input_path, allow_pickle=False) as opened:
        data = dict(opened)
    spacing = float(data["spacing"])
    methods, surfaces = {}, {}
    for truncation in (4, 8, 12):
        started = time.perf_counter()
        name = f"trunc{truncation}"
        root = output_root / name
        root.mkdir(parents=True, exist_ok=True)
        if truncation == 4:
            with np.load(baseline_root / "tsdf.npz", allow_pickle=False) as original:
                vertices, faces = original["vertices"], original["faces"]
            shutil.copy2(baseline_root / "tsdf.npz", root / "tsdf.npz")
            shutil.copy2(baseline_root / "tsdf.glb", root / "tsdf.glb")
            original_stats = json.loads((baseline_root / "tsdf-stats.json").read_text())
            texture_stats, export_stats = original_stats["texture"], original_stats["export"]
        else:
            vertices, faces = integrate(data, truncation)
            sources, texture_stats = choose_sources(vertices, faces, data)
            export_stats = export_textured_mesh(vertices, faces, sources, data, root / "tsdf.glb")
            np.savez_compressed(root / "tsdf.npz", vertices=vertices, faces=faces,
                                face_sources=sources)
        methods[name] = {
            "sdf_trunc_spacing": truncation, "sdf_trunc_m": truncation * spacing,
            "reused_original_geometry_and_texture": truncation == 4,
            "seconds_including_export_and_topology": None,
            "topology": topology(vertices, faces),
            "texture": texture_stats, "export": export_stats,
        }
        methods[name]["seconds_including_export_and_topology"] = time.perf_counter() - started
        surfaces[name] = samples(vertices, faces)
        print(f"{name}: {len(faces)} triangles", flush=True)
    ref = data["frame"] == 0
    normal = np.median(data["normals"][ref], axis=0)
    normal /= np.linalg.norm(normal)
    right = data["E"][0, 0, :3].copy()
    right -= np.dot(right, normal) * normal
    right /= np.linalg.norm(right)
    axes = np.stack([right, np.cross(normal, right), normal], axis=1)
    center = np.median(data["points"][ref], axis=0)
    cell = 4 * spacing
    ref_points = data["points"][ref & (data["normals"] @ normal > np.cos(np.deg2rad(35)))]
    reference_cells = set(map(tuple, np.floor(
        ((ref_points - center) @ axes)[:, :2] / cell).astype(int)))
    measured = {name: local_spreads(p, n, center, axes, cell, reference_cells)
                for name, (p, n) in surfaces.items()}
    common = set.intersection(*(set(spread) for _, spread in measured.values()))
    if not common:
        raise ValueError("No shared front-surface cells")
    for name, (groups, spread) in measured.items():
        values = np.array([spread[k] for k in sorted(common)])
        methods[name]["geometry_proxy"] = {
            "front_reference_coverage": len(groups) / len(reference_cells),
            "normal_spread_median_m": float(np.median(values)),
            "normal_spread_p90_m": float(np.quantile(values, .9)),
        }
    result = {
        "input": str(input_path.resolve()), "baseline_root": str(baseline_root.resolve()),
        "voxel_length_m": spacing, "depth_scale": 1, "depth_sampling_stride": 1,
        "depth_trunc": "per-frame max masked depth + 4 * spacing for all variants",
        "variable": "sdf_trunc only; immutable K/E/input depths/masks/voxel and original texture algorithm",
        "cell_size_m": cell, "reference_cells": len(reference_cells),
        "common_cells": len(common), "surface_samples_per_method": 100000,
        "sampling_seed": 42, "methods": methods,
        "limitations": [
            "Normal-axis within-cell P90-P10 spread is an internal geometry proxy, not accuracy or physical thickness.",
            "Three TSDF variants use common cells; values are not directly comparable to the earlier baseline/fused/TSDF common-cell comparison.",
            "Samples use raw NPZ face order; prior evaluation sampled exported GLB material-reordered faces, so seed 42 does not select identical points across reports.",
            "Deleting or losing geometry can lower spread; reference-cell hits are not surface or image coverage.",
            "Boundary length and connected components refer to raw original-world geometry; no hole filling or component removal.",
            "Untextured fraction is surface area, not grey screen pixel fraction.",
        ],
    }
    (output_root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runtime/roi-fusion-video3-gid3/input.npz"))
    parser.add_argument("--baseline-root", type=Path, default=Path("runtime/roi-fusion-video3-gid3"))
    parser.add_argument("--output-root", type=Path, default=Path("runtime/roi-fusion-video3-gid3-v2/geometry"))
    parser.add_argument("--silhouette-only", action="store_true",
                        help="Raycast the three saved meshes; do not integrate or export")
    args = parser.parse_args()
    if args.silhouette_only:
        silhouettes(args.input, args.output_root)
    else:
        run(args.input, args.baseline_root, args.output_root)
