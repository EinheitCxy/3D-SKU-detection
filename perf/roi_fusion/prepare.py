"""Prepare one fixed video3 product; no model inference or input mutation."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.surfel_export import grid_tangents


def prepare(gid, output):
    cache_root = ROOT / 'runtime/video3-resolution-review/504/outputs/dataset'
    images = ROOT / 'runtime/video3-resolution-review/dataset/images'
    mapping = json.loads((cache_root / 'dedup_detections/global_mapping.json').read_text())
    observations = mapping[str(gid)]
    z = np.load(cache_root / 'da3_cache/predictions.npz', allow_pickle=False)
    image_ids = z['image_ids']
    ids = [int(o['image_id']) for o in observations]
    if len(ids) != len(set(ids)):
        raise ValueError('Multiple instances in a frame: inspect mapping before fusion')
    indices = [int(np.flatnonzero(image_ids == i)[0]) for i in ids]
    depth = z['depth'][indices, ..., 0]
    points_grid = z['world_points'][indices]
    confidence_grid = z['depth_conf'][indices]
    K = z['intrinsic'][indices]
    E = np.tile(np.eye(4), (len(ids), 1, 1))
    E[:, :3] = z['extrinsic'][indices]
    affine = np.tile(np.eye(3), (len(ids), 1, 1))
    affine[:, :2] = z['source_to_processed_affine'][indices]
    U, V, _ = grid_tangents(points_grid, E[:, :3])
    masks = np.zeros(depth.shape, bool)
    output.mkdir(parents=True, exist_ok=True)
    sheet = Image.new('RGB', (400 * len(ids), 310), 'white')
    for f, obs in enumerate(observations):
        entry = cache_root / f'sam3_mask_cache/v2/entries/{ids[f]}'
        manifest = json.loads((entry / 'manifest.json').read_text())
        detection = next(d for d in manifest['detections'] if d['object_id'] == obs['object_id'])
        packed = np.load(entry / 'masks.npz')['packed_masks'][detection['mask_index']]
        masks[f] = np.unpackbits(packed, bitorder='little')[:depth.shape[1]*depth.shape[2]].reshape(depth.shape[1:])
        im = Image.open(images / f'{ids[f]}.jpg').convert('RGB')
        crop = im.crop(obs['bbox'])
        crop.thumbnail((390, 270))
        sheet.paste(crop, (f*400, 30))
        ImageDraw.Draw(sheet).text((f*400+5, 5), f'frame {ids[f]}, object {obs["object_id"]}', fill='black')
    sheet.save(output / 'observations.jpg', quality=95)
    normal = np.cross(U, V)
    lengths = np.linalg.norm(normal, axis=-1)
    valid = masks & np.isfinite(points_grid).all(-1) & (depth > 0) & (lengths > 1e-12)
    f, y, x = np.nonzero(valid)
    points, u, v = points_grid[valid], U[valid], V[valid]
    normals = normal[valid] / lengths[valid, None]
    centers = np.linalg.inv(E)[:, :3, 3]
    facing = np.einsum('ij,ij->i', normals, centers[f]-points)
    normals[facing < 0] *= -1
    spacing = float(np.median(np.concatenate([np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)])))
    data = dict(points=points, u=u, v=v, normals=normals, confidence=confidence_grid[valid], frame=f.astype(np.int32),
                xy=np.stack([x,y], axis=1), depth=depth, mask=masks, K=K, E=E, affine=affine,
                image_paths=np.array([str(images / f'{i}.jpg') for i in ids]), image_ids=np.array(ids), spacing=spacing)
    np.savez_compressed(output / 'input.npz', **data)
    center = np.median(points, axis=0)
    radius = float(np.quantile(np.linalg.norm(points-center, axis=1), .98))
    # This is an internal disagreement diagnostic, not geometry ground truth.
    per_pair = []
    for a in range(len(ids)):
        pa, na = points[f==a], normals[f==a]
        for b in range(a+1, len(ids)):
            pb, nb = points[f==b], normals[f==b]
            distance, near = cKDTree(pb).query(pa)
            delta = pb[near]-pa
            along = np.abs(np.einsum('ij,ij->i', delta, na))
            lateral = np.sqrt(np.maximum(0, distance**2-along**2))
            good = (lateral < 2*spacing) & (distance < 10*spacing) & (np.einsum('ij,ij->i', na, nb[near]) > .866)
            per_pair.append(dict(frames=[ids[a],ids[b]], count=int(good.sum()),
                                 normal_gap_median_m=float(np.median(along[good])) if good.any() else None,
                                 normal_gap_p90_m=float(np.quantile(along[good],.9)) if good.any() else None))
    meta = dict(global_id=gid, center=center.tolist(), radius=radius, camera_positions=centers.tolist(),
                camera_up=(-E[0,1,:3]).tolist(), up=(-E[0,1,:3]).tolist(), source_image_ids=ids, observations=observations,
                spacing=spacing, point_count=len(points), per_frame_counts=np.bincount(f).tolist(),
                normal_gaps=per_pair, reference_images=[{'url':'observations.jpg','label':'五帧原图商品区域'}],
                production_reference='production-frame0.png',
                baseline_notice='Common experimental textured discs; not pixel-identical to production splatting.')
    (output / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    shutil.copyfile(ROOT/'runtime/video3-resolution-review/res504-frame0.png', output/'production-frame0.png')
    # Shared context is a modest point cloud from source frame 0 outside the ROI.
    gy,gx = np.mgrid[:depth.shape[1],:depth.shape[2]]
    context_mask = (~masks[0]) & (depth[0] > 0) & (gy%3==0) & (gx%3==0)
    im = np.asarray(Image.open(data['image_paths'][0]).convert('RGB'))
    pixels = np.stack([gx[context_mask],gy[context_mask],np.ones(context_mask.sum())],1) @ np.linalg.inv(affine[0]).T
    ix=np.clip(np.rint(pixels[:,0]).astype(int),0,im.shape[1]-1)
    iy=np.clip(np.rint(pixels[:,1]).astype(int),0,im.shape[0]-1)
    import trimesh
    trimesh.Scene(trimesh.points.PointCloud(points_grid[0][context_mask], im[iy,ix])).export(output/'context.glb')
    print(json.dumps({k:meta[k] for k in ['global_id','point_count','spacing','per_frame_counts','normal_gaps']},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--gid',type=int,default=3)
    p.add_argument('--output',type=Path,default=ROOT/'runtime/roi-fusion-video3-gid3')
    a=p.parse_args();prepare(a.gid,a.output)
