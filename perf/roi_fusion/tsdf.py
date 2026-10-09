"""ROI-masked TSDF integration with immutable original world-to-camera poses."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from perf.roi_fusion.texture import choose_sources, export_textured_mesh


def run(input_path, output_root):
    started = time.perf_counter()
    with np.load(input_path, allow_pickle=False) as data:
        spacing = float(data["spacing"])
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=spacing, sdf_trunc=4 * spacing,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
            depth_sampling_stride=1,
        )
        integrated_pixels = []
        for frame in range(len(data["E"])):
            raw = data["depth"][frame]
            valid = data["mask"][frame] & np.isfinite(raw) & (raw > 0)
            depth = np.ascontiguousarray(np.where(valid, raw, 0), dtype=np.float32)
            height, width = depth.shape
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, data["K"][frame, 0, 0], data["K"][frame, 1, 1], data["K"][frame, 0, 2], data["K"][frame, 1, 2])
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8)),
                o3d.geometry.Image(depth), depth_scale=1., depth_trunc=float(depth.max()) + 4 * spacing,
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, intrinsic, data["E"][frame])
            integrated_pixels.append(int(valid.sum()))
        mesh = volume.extract_triangle_mesh()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        if not len(faces):
            raise RuntimeError("ROI TSDF integration produced no triangles")
        sources, texture_stats = choose_sources(vertices, faces, data)
        output_root = Path(output_root)
        export_stats = export_textured_mesh(vertices, faces, sources, data, output_root / "tsdf.glb")
        np.savez_compressed(output_root / "tsdf.npz", vertices=vertices, faces=faces, face_sources=sources)
        stats = {"method": "tsdf", "seconds": time.perf_counter() - started,
            "voxel_length_m": spacing, "sdf_trunc_m": 4 * spacing,
            "depth_scale": 1, "depth_sampling_stride": 1, "integration": "ROI mask only; original world-to-camera; no RGB fusion",
            "integrated_pixels": integrated_pixels, "texture": texture_stats, "export": export_stats}
        (output_root / "tsdf-stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("runtime/roi-fusion-video3-gid3/input.npz"))
    parser.add_argument("--output-root", type=Path, default=Path("runtime/roi-fusion-video3-gid3"))
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_root), indent=2))
