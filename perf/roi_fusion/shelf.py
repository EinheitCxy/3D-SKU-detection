"""Full video3 shelf comparison from all 29 cached views, without inference."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from src.surfel_export import grid_tangents
from perf.roi_fusion.fusion import coherent_surfels, disc_mesh, fuse_surfels
from perf.roi_fusion.probe_geometry import integrate
from perf.roi_fusion.texture import choose_sources, export_textured_mesh

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'runtime/roi-fusion-video3-gid3-v2/shelf-view'


def log(message):
    print(message, flush=True)


def prepare(output, stride, confidence_min):
    cache = ROOT / 'runtime/video3-resolution-review/504/outputs/dataset/da3_cache/predictions.npz'
    images = ROOT / 'runtime/video3-resolution-review/dataset/images'
    with np.load(cache, allow_pickle=False) as z:
        points = z['world_points']
        depth = z['depth'][..., 0]
        confidence = z['depth_conf']
        ids = z['image_ids']
        E = np.tile(np.eye(4), (len(ids), 1, 1))
        E[:, :3] = z['extrinsic']
        K = z['intrinsic']
        affine = np.tile(np.eye(3), (len(ids), 1, 1))
        affine[:, :2] = z['source_to_processed_affine']
    u, v, _ = grid_tangents(points, E[:, :3])
    lengths = np.linalg.norm(np.cross(u, v), axis=-1)
    valid = (np.isfinite(points).all(-1) & np.any(points != 0, axis=-1)
             & np.isfinite(confidence) & (confidence >= confidence_min)
             & np.isfinite(depth) & (depth > 0) & (lengths > 1e-12))
    yy, xx = np.indices(depth.shape[1:])
    sampled = valid & (yy % stride == 0) & (xx % stride == 0)
    f, y, x = np.nonzero(sampled)
    # Geometry gates/TSDF retain original pixel spacing. Larger display discs
    # cover the explicit uniform subsampling, and are not extra observations.
    spacing = float(np.median(np.r_[np.linalg.norm(u[sampled], axis=1),
                                    np.linalg.norm(v[sampled], axis=1)]))
    data = dict(points=points[sampled], u=u[sampled], v=v[sampled], display_stride=stride,
                confidence=confidence[sampled], frame=f.astype(np.int32),
                xy=np.stack([x, y], axis=1), depth=depth, mask=valid,
                K=K, E=E, affine=affine, spacing=spacing, image_ids=ids,
                image_paths=np.array([str(images / f'{int(i)}.jpg') for i in ids]))
    np.savez_compressed(output / 'input.npz', **data)
    center = np.median(data['points'], axis=0)
    radius = float(np.quantile(np.linalg.norm(data['points']-center, axis=1), .98))
    cameras = np.linalg.inv(E)[:, :3, 3]
    sheet = Image.new('RGB', (6*240, 5*180), 'white')
    for i, path in enumerate(data['image_paths']):
        with Image.open(str(path)) as original:
            im = original.convert('RGB')
        im.thumbnail((230, 155))
        origin = ((i % 6)*240, (i // 6)*180)
        sheet.paste(im, (origin[0], origin[1]+20))
        ImageDraw.Draw(sheet).text(origin, f'frame {ids[i]}', fill='black')
    sheet.save(output / 'observations.jpg', quality=92)
    meta = dict(title='Video3 · 整个货架的融合与贴图对比', center=center.tolist(),
                radius=radius, up=(-E[0, 1, :3]).tolist(),
                camera_positions=cameras.tolist(), source_image_ids=ids.tolist(),
                method_labels={'baseline': '原始全场景 Surfel',
                               'fused': '全场景 Surfel 融合＋主纹理',
                               'tsdf': '全场景 TSDF 8×＋分块贴图'},
                notice=f'全部 {len(ids)} 帧、全画面范围，共同 confidence ≥ {confidence_min}。Surfel 每 {stride} 像素采样并扩大显示圆盘；TSDF 使用完整有效深度。均固定原相机；灰色为未贴图。此页不沿用生产点云的商品保护和背景过滤。',
                reference_images=[{'url': 'observations.jpg', 'label': '全部输入帧'}],
                stats={}, preparation={'frames': len(ids), 'grid': list(depth.shape[1:]),
                    'valid_depth_pixels': int(valid.sum()), 'surfel_stride': stride,
                    'confidence_min': confidence_min,
                    'sampled_surfels': len(f), 'spacing_m': spacing,
                    'per_frame_surfels': np.bincount(f, minlength=len(ids)).tolist(),
                    'scope': 'whole cached image, including shelf, products and surroundings',
                    'semantic_instance_gating': False})
    (output / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    log(meta['preparation'])


def export_method(output, name, max_triangles):
    started = time.monotonic()
    with np.load(output / 'input.npz', allow_pickle=False) as z:
        data = dict(z)
    if name == 'tsdf':
        log('TSDF: integrating all full-resolution views')
        vertices, faces = integrate(data, 8)
        raw_faces = len(faces)
        np.savez_compressed(output / 'tsdf-full.npz', vertices=vertices, faces=faces)
        if raw_faces > max_triangles:
            import open3d as o3d
            log(f'TSDF: simplifying browser mesh {raw_faces:,} -> {max_triangles:,} faces')
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                           o3d.utility.Vector3iVector(faces))
            mesh = mesh.simplify_quadric_decimation(max_triangles)
            vertices, faces = np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()
        stats = {'voxel_length_m': float(data['spacing']),
                 'sdf_trunc_m': 8*float(data['spacing']), 'input_frames': len(data['E']),
                 'raw_triangles': raw_faces, 'preview_triangles': len(faces),
                 'preview_simplification': 'quadric decimation before texture selection'}
        preferred = None
    else:
        if name == 'fused':
            log('Surfel: reciprocal fusion over every frame pair')
            surfels, stats = fuse_surfels(data)
            np.savez_compressed(output / 'fused.npz', **surfels)
        else:
            surfels, stats = coherent_surfels(data), {}
        stride = int(data['display_stride'])
        vertices, faces = disc_mesh(surfels['points'], surfels['u']*stride, surfels['v']*stride)
        stats['surfels'] = len(surfels['points'])
        preferred = np.repeat(data['frame'], 8) if name == 'baseline' else None
    log(f'{name}: selecting textures for {len(faces):,} triangles')
    sources, texture = choose_sources(vertices, faces, data, preferred_sources=preferred)
    stats['texture'] = texture
    log(f'{name}: exporting GLB')
    stats['export'] = export_textured_mesh(vertices, faces, sources, data, output / f'{name}.glb')
    stats['seconds'] = time.monotonic()-started
    (output / f'{name}-stats.json').write_text(json.dumps(stats, indent=2)+'\n')
    log({'method': name, 'seconds': stats['seconds'], 'triangles': len(faces),
         'untextured_area_fraction': texture['untextured_area_fraction'],
         'glb_bytes': stats['export']['glb_bytes']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--confidence-min', type=float, default=1.1)
    parser.add_argument('--max-triangles', type=int, default=1_000_000)
    parser.add_argument('--stage', choices=['prepare', 'baseline', 'fused', 'tsdf', 'finalize'], required=True)
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError('stride must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare':
        prepare(args.output, args.stride, args.confidence_min)
    elif args.stage == 'finalize':
        meta = json.loads((args.output / 'meta.json').read_text())
        for name in ('baseline', 'fused', 'tsdf'):
            if not (args.output / f'{name}.glb').is_file():
                raise FileNotFoundError(name)
            meta['stats'][name] = json.loads((args.output / f'{name}-stats.json').read_text())
        preview_faces = meta['stats']['tsdf']['preview_triangles']
        meta['notice'] = meta['notice'].split(' TSDF 网页预览')[0]
        meta['notice'] += f' TSDF 网页预览使用 {preview_faces:,} 面网格，再重新贴图；完整网格另存。'
        (args.output / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
        log('All three full-scene assets ready')
    else:
        if args.max_triangles < 1:
            raise ValueError('max-triangles must be positive')
        export_method(args.output, args.stage, args.max_triangles)


if __name__ == '__main__':
    main()
