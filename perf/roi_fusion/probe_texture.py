"""Bounded TSDF texture ablation: strict depth vs mesh visibility, same geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from perf.roi_fusion.texture import _project, face_view_candidates, export_textured_mesh


def mesh_view_candidates(vertices, faces, data, truncation):
    """Four-sample mask + first mesh hit + bounded original-depth disagreement.

    This is an explicit alternative visibility model, not a rescue pass. Raw
    depth is still required within one TSDF truncation distance. Occluders absent
    from both the mesh and product masks cannot be certified by this probe.
    """
    _, _, area2, centers, normals = face_view_candidates(vertices, faces, data)
    triangles = vertices[faces]
    # 1% inset: float32 world-space rays at 0.01% frequently hit adjacent faces
    # within sub-micrometre residuals. Exact vertices still govern mask/depth.
    samples = np.concatenate((triangles * .99 + centers[:, None] * .01,
                              centers[:, None]), axis=1).reshape(-1, 3)
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    visibility, scores = [], []
    spacing = float(data['spacing'])
    for frame, extrinsic in enumerate(data['E']):
        # Mask/depth sampling uses actual vertices, not inset ray endpoints.
        exact = np.concatenate((triangles, centers[:, None]), axis=1).reshape(-1, 3)
        xy, z = _project(exact, frame, data)
        h, w = data['depth'][frame].shape
        finite = np.isfinite(xy).all(1) & np.isfinite(z)
        inside = finite & (z > 0) & (xy[:, 0] >= 0) & (xy[:, 0] <= w-1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h-1)
        pix = np.rint(np.where(finite[:, None], xy, 0)).astype(int)
        px, py = pix[:, 0].clip(0, w-1), pix[:, 1].clip(0, h-1)
        measured = data['depth'][frame, py, px]
        allowed = np.maximum(truncation, .005 * z)
        valid = inside & data['mask'][frame, py, px] & np.isfinite(measured) & (measured > 0) & (np.abs(measured-z) <= allowed)
        origin = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
        direction = samples - origin
        distance = np.linalg.norm(direction, axis=1)
        rays = np.column_stack((np.broadcast_to(origin, samples.shape), direction / np.maximum(distance[:, None], 1e-20)))
        hits = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)), nthreads=4)
        hit = hits['t_hit'].numpy()
        target_faces = np.repeat(np.arange(len(faces)), 4)
        first_surface = (np.isfinite(hit) & (np.abs(hit-distance) <= .1 * spacing)
                         & (hits['primitive_ids'].numpy() == target_faces))
        view = origin - centers
        center_distance = np.linalg.norm(view, axis=1)
        cosine = np.sum(normals * view, axis=1) / np.maximum(center_distance, 1e-20)
        visible = (valid & first_surface).reshape(-1, 4).all(1) & (cosine >= .15) & (area2 > 1e-20)
        visibility.append(visible)
        scores.append(np.where(visible, area2 * cosine**2 / np.maximum(center_distance**2, 1e-20), 0))
    return np.array(visibility), np.array(scores), area2


def label_faces(mesh, visibility, scores, seam_weight=.25, max_sweeps=12):
    """Deterministic Potts-label ICM; valid labels only, exact mesh adjacency.

    Local optimization, not graph-cut/alpha-expansion or a paper reproduction.
    Independent color classes allow parallel updates without increasing energy.
    """
    count = len(mesh.faces)
    active = visibility.any(axis=0)
    unary = 1 - scores / np.maximum(scores.max(axis=0, keepdims=True), 1e-30)
    unary[~visibility] = np.inf
    labels = np.argmax(scores, axis=0).astype(np.int32)
    labels[~active] = -1
    edges = mesh.face_adjacency
    edges = edges[active[edges].all(1)]
    neighbors = [[] for _ in range(count)]
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    colors = np.full(count, -1, dtype=int)
    for face in np.flatnonzero(active):
        used = {colors[n] for n in neighbors[face] if colors[n] >= 0}
        color = 0
        while color in used:
            color += 1
        colors[face] = color
    width = max(map(len, neighbors), default=0)
    adjacency = np.full((count, max(width, 1)), count, dtype=int)
    for i, ns in enumerate(neighbors):
        adjacency[i, :len(ns)] = ns

    def energy():
        return float(unary[labels[active], np.flatnonzero(active)].sum() +
                     seam_weight * np.count_nonzero(labels[edges[:, 0]] != labels[edges[:, 1]]))

    history = [energy()]
    for _ in range(max_sweeps):
        changes = 0
        for color in range(colors.max(initial=-1)+1):
            group = np.flatnonzero(colors == color)
            neighboring = np.append(labels, -2)[adjacency[group]]
            costs = unary[:, group].copy()
            for source in range(len(scores)):
                costs[source] += seam_weight * ((neighboring != source) & (neighboring != -2)).sum(1)
            best = costs.argmin(axis=0)
            improve = costs[best, np.arange(len(group))] < costs[labels[group], np.arange(len(group))] - 1e-12
            labels[group[improve]] = best[improve]
            changes += int(improve.sum())
        history.append(energy())
        if not changes:
            break
    if np.any(np.diff(history) > 1e-8):
        raise RuntimeError('Label optimization increased its energy')
    if not np.all(visibility[labels[active], np.flatnonzero(active)]):
        raise RuntimeError('Selected invisible texture source')
    return labels, {'optimizer': 'mesh-adjacency Potts ICM', 'seam_weight': seam_weight,
                    'max_sweeps': max_sweeps, 'energy_history': history,
                    'converged': changes == 0,
                    'source_seam_edges': int(np.count_nonzero(labels[edges[:, 0]] != labels[edges[:, 1]]))}


def run(input_path, mesh_path, output_root, truncation):
    data = dict(np.load(input_path, allow_pickle=False))
    original = dict(np.load(mesh_path, allow_pickle=False))
    vertices, faces = original['vertices'], original['faces']
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in ('strict', 'mesh'):
        if mode == 'strict':
            visible, scores, area2, _, _ = face_view_candidates(vertices, faces, data)
        else:
            visible, scores, area2 = mesh_view_candidates(vertices, faces, data, truncation)
        labels, stats = label_faces(mesh, visible, scores)
        stats.update({'visibility': mode, 'untextured_area_fraction': float(area2[labels < 0].sum()/area2.sum()),
                      'eligible_faces': int(visible.any(0).sum()), 'textured_faces': int((labels >= 0).sum()),
                      'truncation_m': truncation, 'geometry_unchanged': True})
        if mode == 'mesh':
            stats['ray_contract'] = {'vertex_to_centroid_inset': .01, 'target_primitive_required': True,
                                     'hit_distance_tolerance_spacing': .1,
                                     'mask_depth_samples': 'exact vertices + centroid',
                                     'depth_tolerance': 'max(TSDF truncation, 0.005*camera_Z)'}
        stats['export'] = export_textured_mesh(vertices, faces, labels, data, output_root/f'{mode}.glb')
        np.savez_compressed(output_root/f'{mode}.npz', vertices=vertices, faces=faces, face_sources=labels)
        results[mode] = stats
    (output_root/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('runtime/roi-fusion-video3-gid3/input.npz'))
    p.add_argument('--mesh', type=Path, default=Path('runtime/roi-fusion-video3-gid3/tsdf.npz'))
    p.add_argument('--output', type=Path, default=Path('runtime/roi-fusion-video3-gid3-v2/texture'))
    p.add_argument('--truncation', type=float, default=None)
    a = p.parse_args()
    truncation = a.truncation
    if truncation is None:
        with np.load(a.input) as d:
            truncation = 4 * float(d['spacing'])
    print(json.dumps(run(a.input, a.mesh, a.output, truncation), indent=2))
