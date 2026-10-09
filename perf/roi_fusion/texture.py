"""Fixed spatial-patch projective textures for the isolated ROI comparison."""
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree


def _project(points, frame, data):
    camera = points @ data["E"][frame, :3, :3].T + data["E"][frame, :3, 3]
    homogeneous = camera @ data["K"][frame].T
    xy = homogeneous[:, :2] / np.where(camera[:, 2:3] != 0, camera[:, 2:3], np.nan)
    return xy, camera[:, 2]


def project_uv(points, frame, data, original_size):
    """Trimesh UV (V up); pixel-center coordinates survive Pillow resizing."""
    xy, _ = _project(np.asarray(points), frame, data)
    original = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(data["affine"][frame]).T
    original = original[:, :2] / original[:, 2:3]
    width, height = original_size
    return np.column_stack(((original[:, 0] + .5) / width, 1 - (original[:, 1] + .5) / height))


def _patches(centers, normals, spacing):
    """Connect nearby faces, including distinct surfel discs, with bounded growth."""
    count = len(centers)
    tree = cKDTree(centers)
    _, neighbors = tree.query(centers, k=min(24, count), distance_upper_bound=2.5 * spacing)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    assigned = np.zeros(count, dtype=bool)
    cosine = np.cos(np.deg2rad(25))
    patches = []
    for seed in range(count):
        if assigned[seed]:
            continue
        assigned[seed] = True
        patch = [seed]
        cursor = 0
        while cursor < len(patch):
            face = patch[cursor]
            cursor += 1
            for candidate in neighbors[face]:
                if candidate >= count or assigned[candidate]:
                    continue
                if np.dot(normals[face], normals[candidate]) < cosine:
                    continue
                if np.dot(normals[seed], normals[candidate]) < cosine:
                    continue
                if np.linalg.norm(centers[seed] - centers[candidate]) > 12 * spacing:
                    continue
                assigned[candidate] = True
                patch.append(int(candidate))
        patches.append(np.asarray(patch, dtype=np.int64))
    return patches


def face_view_candidates(vertices, faces, data):
    """Shared strict four-sample visibility and source quality, without selection."""
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=np.int64)
    spacing = float(data["spacing"])
    triangles = vertices[faces]
    centers = triangles.mean(axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(twice_area[:, None], 1e-20)
    samples = np.concatenate((triangles, centers[:, None]), axis=1)
    visibility, scores = [], []
    for frame in range(len(data["E"])):
        xy, z = _project(samples.reshape(-1, 3), frame, data)
        height, width = data["depth"][frame].shape
        finite = np.isfinite(xy).all(axis=1) & np.isfinite(z)
        inside = finite & (z > 0) & (xy[:, 0] >= 0) & (xy[:, 0] <= width - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= height - 1)
        safe_xy = np.where(finite[:, None], xy, 0)
        px = np.rint(safe_xy[:, 0]).astype(np.int64).clip(0, width - 1)
        py = np.rint(safe_xy[:, 1]).astype(np.int64).clip(0, height - 1)
        measured = data["depth"][frame, py, px]
        tolerance = np.maximum(2 * spacing, .005 * z)
        sample_visible = inside & (measured > 0) & np.isfinite(measured) & (np.abs(measured - z) <= tolerance)
        camera_center = -data["E"][frame, :3, :3].T @ data["E"][frame, :3, 3]
        view = camera_center - centers
        distance = np.linalg.norm(view, axis=1)
        cosine = np.sum(normals * view, axis=1) / np.maximum(distance, 1e-20)
        sample_mask = data["mask"][frame, py, px]
        visible = (sample_visible & sample_mask).reshape(-1, 4).all(axis=1) & (cosine >= .15) & (twice_area > 1e-20)
        visibility.append(visible)
        scores.append(np.where(visible, twice_area * cosine ** 2 / np.maximum(distance ** 2, 1e-20), 0))
    visibility, scores = np.asarray(visibility), np.asarray(scores)
    return visibility, scores, twice_area, centers, normals


def choose_sources(vertices, faces, data, *, preferred_sources=None):
    """Choose one frame per patch; uncovered faces remain explicitly untextured.

    Visibility is checked at all three vertices and the centroid against full-scene
    Z depth and product mask, with max(2*spacing, 0.005*Z) tolerance and front cosine >=0.15.
    A patch selects the largest visible projected-area score, retaining that one
    source only on its visible faces. With preferred_sources, retain each original
    source only when visible. Neither mode chooses per-face alternative sources.
    """
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=np.int64)
    spacing = float(data["spacing"])
    if spacing <= 0 or not len(faces):
        raise ValueError("Nonempty mesh and positive spacing are required")
    visibility, scores, twice_area, centers, normals = face_view_candidates(vertices, faces, data)
    if preferred_sources is not None:
        preferred_sources = np.asarray(preferred_sources)
        if preferred_sources.shape != (len(faces),) or not np.issubdtype(preferred_sources.dtype, np.integer):
            raise ValueError("preferred_sources must contain one integer frame index per face")
        if np.any((preferred_sources < 0) | (preferred_sources >= len(data["E"]))):
            raise ValueError("preferred_sources contains an invalid frame index")
    patches = _patches(centers, normals, spacing) if preferred_sources is None else []
    sources = np.full(len(faces), -1, dtype=np.int32)
    patch_source_counts = {}
    if preferred_sources is not None:
        visible_original = visibility[preferred_sources, np.arange(len(faces))]
        sources[visible_original] = preferred_sources[visible_original]
    else:
        for patch in patches:
            summed = scores[:, patch].sum(axis=1)
            if summed.max() <= 0:
                continue
            source = int(np.argmax(summed))
            sources[patch[visibility[source, patch]]] = source
            patch_source_counts[str(source)] = patch_source_counts.get(str(source), 0) + 1
    stats = {
        "selection_mode": "original-source" if preferred_sources is not None else "spatial-patch",
        "patches": len(patches), "patch_source_counts": patch_source_counts,
        "textured_faces": int(np.count_nonzero(sources >= 0)),
        "untextured_faces": int(np.count_nonzero(sources < 0)),
        "untextured_area_fraction": float(twice_area[sources < 0].sum() / max(twice_area.sum(), 1e-20)),
        "thresholds": {"patch_neighbor_radius_spacing": 2.5, "patch_neighbor_limit": 24,
            "patch_seed_radius_spacing": 12, "normal_degrees": 25, "front_cosine": .15,
            "depth_absolute_tolerance_spacing": 2, "depth_relative_tolerance": .005,
            "visibility_samples": "3 vertices + centroid; all samples must lie in product mask",
            "source_score": "sum visible triangle area * front_cosine^2 / camera_distance^2"},
    }
    return sources, stats


def export_textured_mesh(vertices, faces, face_sources, data, output_path):
    """Embed per-source JPEGs and UVs in an original-world, unlit GLB."""
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=np.int64)
    face_sources = np.asarray(face_sources, dtype=np.int32)
    if face_sources.shape != (len(faces),):
        raise ValueError("Each face must have exactly one source")
    scene = trimesh.Scene()
    texture_sizes = {}
    for source in np.unique(face_sources):
        selected = faces[face_sources == source]
        indices, inverse = np.unique(selected.ravel(), return_inverse=True)
        mesh = trimesh.Trimesh(vertices=vertices[indices], faces=inverse.reshape(-1, 3), process=False)
        if source >= 0:
            with Image.open(str(data["image_paths"][source])) as opened:
                original_size = opened.size
                texture = opened.convert("RGB")
            scale = min(1., 1920 / max(texture.size), 1080 / min(texture.size))
            if scale < 1:
                texture = texture.resize(tuple(max(1, round(d * scale)) for d in texture.size), Image.Resampling.LANCZOS)
            texture.format = "JPEG"
            texture_sizes[str(source)] = list(texture.size)
            material = trimesh.visual.material.PBRMaterial(name=f"frame_{source}", baseColorTexture=texture,
                baseColorFactor=[255, 255, 255, 255], metallicFactor=0., roughnessFactor=1., doubleSided=True)
            mesh.visual = trimesh.visual.TextureVisuals(uv=project_uv(mesh.vertices, source, data, original_size), material=material)
        else:
            material = trimesh.visual.material.PBRMaterial(name="untextured", baseColorFactor=[128, 128, 128, 255],
                metallicFactor=0., roughnessFactor=1., doubleSided=True)
            mesh.visual = trimesh.visual.TextureVisuals(material=material)
        scene.add_geometry(mesh, node_name=f"source_{source}", geom_name=f"source_{source}")

    def unlit(tree):
        tree.setdefault("extensionsUsed", []).append("KHR_materials_unlit")
        for material in tree["materials"]:
            material.setdefault("extensions", {})["KHR_materials_unlit"] = {}
            name = material.get("name", "")
            if name.startswith("frame_"):
                material.setdefault("extras", {})["source_frame"] = int(name[6:])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(trimesh.exchange.gltf.export_glb(scene, tree_postprocessor=unlit))
    return {"vertices": len(vertices), "faces": len(faces), "textured_faces": int((face_sources >= 0).sum()),
        "untextured_faces": int((face_sources < 0).sum()), "texture_sizes": texture_sizes,
        "glb_bytes": output_path.stat().st_size, "uv_convention": "inverse original-to-processed affine; original pixel centers; Pillow resized JPEG; V up before glTF export"}
