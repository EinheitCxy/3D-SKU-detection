"""用统一米制轨迹渲染完整 SH2 高斯及派生商品编号贡献图。"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'runtime/surfel-comparison/deps'))
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from gsplat import rasterization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'runtime/surfel-comparison/gs')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    output = ROOT/'modules/viewer_web/public/comparison/gaussian'
    output.mkdir(parents=True, exist_ok=True)
    bundle = ROOT/'modules/viewer_web/public/data-video1-1fps/runs/112743ae4de5481bb07821432f19b58f'
    cameras = json.loads((output.parent/'cameras.json').read_text())
    inference = json.loads((args.input/'metrics.json').read_text())
    alignment = inference['baseline_to_gs_sim3']
    scale = alignment['scale']
    rotation = np.asarray(alignment['rotation'])
    translation = np.asarray(alignment['translation'])
    started = time.perf_counter()
    with np.load(args.input/'gaussians-metric.npz') as archive:
        arrays = {key: archive[key] for key in archive.files}
    read_seconds = time.perf_counter() - started
    started = time.perf_counter()
    points = np.fromfile(bundle/'positions.f32.bin', dtype='<f4').reshape(-1, 3)
    objects = json.loads((bundle/'objects.json').read_text())
    point_ids = np.zeros(len(points), dtype=np.int64)
    for key in sorted(objects, key=int):
        obj = objects[key]
        for lo, hi in obj['point_ranges']:
            point_ids[lo:hi] = int(key)
    tree = cKDTree(points)
    labels = np.zeros(len(arrays['means']), dtype=np.int64)
    for start in range(0, len(labels), 250000):
        end = min(start + 250000, len(labels))
        baseline_means = (arrays['means'][start:end] - translation) @ rotation / scale
        distance, nearest = tree.query(baseline_means, distance_upper_bound=.015, workers=8)
        matched = np.isfinite(distance)
        labels[start:end][matched] = point_ids[nearest[matched]]
    label_seconds = time.perf_counter() - started
    classes = np.unique(np.r_[0, point_ids])
    del tree, points, point_ids
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    values = {key: torch.from_numpy(value).cuda().contiguous() for key, value in arrays.items()}
    values['harmonics'] = values['harmonics'].transpose(1, 2).contiguous()
    gpu_labels = torch.from_numpy(labels).cuda()
    torch.cuda.synchronize()
    upload_seconds = time.perf_counter() - started
    common = dict(means=values['means'], quats=values['rotations'], scales=values['scales'],
                  opacities=values['opacities'].reshape(-1), width=cameras['width'],
                  height=cameras['height'], packed=False, render_mode='RGB')
    selected = list(enumerate(cameras['frames']))[args.start:]
    if args.limit is not None:
        selected = selected[:args.limit]
    receipts = []
    with torch.inference_mode():
        for index, camera in selected:
            pose = np.linalg.inv(np.asarray(camera['extrinsic']))
            pose[:3, :3] = rotation @ pose[:3, :3]
            pose[:3, 3] = scale * rotation @ pose[:3, 3] + translation
            view = torch.tensor(np.linalg.inv(pose), device='cuda', dtype=torch.float32)[None]
            intrinsics = torch.tensor(camera['intrinsic'], device='cuda', dtype=torch.float32)[None]
            torch.cuda.synchronize()
            started = time.perf_counter()
            rgb, alpha, _ = rasterization(**common, colors=values['harmonics'],
                                         viewmats=view, Ks=intrinsics, sh_degree=2,
                                         backgrounds=torch.ones((1, 3), device='cuda'))
            torch.cuda.synchronize()
            rgb_seconds = time.perf_counter() - started
            Image.fromarray((rgb[0].clamp(0, 1).cpu().numpy()*255).round().astype(np.uint8)).save(output/f'{index:03}.png')
            started = time.perf_counter()
            best = torch.full((cameras['height'], cameras['width']), -1., device='cuda')
            best_id = torch.zeros_like(best, dtype=torch.int64)
            for offset in range(0, len(classes), 32):
                chunk = torch.tensor(classes[offset:offset+32], device='cuda')
                features = (gpu_labels[:, None] == chunk[None]).float()
                contributions, _, _ = rasterization(**common, colors=features, viewmats=view,
                                                    Ks=intrinsics, sh_degree=None)
                maximum, winner = contributions[0].max(dim=-1)
                replace = maximum > best
                best_id[replace] = chunk[winner[replace]]
                best = torch.maximum(best, maximum)
                del features, contributions
            opacity = alpha[0, ..., 0]
            best_id[(opacity < .1) | (best < .5 * opacity)] = 0
            torch.cuda.synchronize()
            id_seconds = time.perf_counter() - started
            ids = best_id.cpu().numpy().astype(np.uint16)
            encoded = np.zeros((*ids.shape, 3), dtype=np.uint8)
            encoded[..., 0] = ids & 255
            encoded[..., 1] = ids >> 8
            Image.fromarray(encoded).save(output/f'{index:03}-ids.png')
            receipt = dict(frame=index, rgb_seconds=rgb_seconds, id_seconds=id_seconds,
                           visible_assigned_pixels=int((ids > 0).sum()))
            receipts.append(receipt)
            print(json.dumps(receipt), flush=True)
    report = dict(inference=inference, file_read_seconds=read_seconds, label_seconds=label_seconds,
                  gpu_upload_seconds=upload_seconds, gaussian_count=len(labels),
                  assigned_gaussians=int((labels > 0).sum()), unknown_gaussians=int((labels == 0).sum()),
                  label_method='派生映射，非原生商品标签、非准确率；基线最近表面 1.5 厘米；未知参与竞争；alpha>=0.1 且主导贡献占比>=0.5',
                  id_encoding='RGB PNG: global_id = R + 256*G; 0=未知',
                  timing_note='首个 RGB 调用包含 gsplat 扩展编译或动态加载；后续帧为 CUDA 同步渲染时间，PNG 写盘另计',
                  sh_degree=2, background='white', peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, frames=receipts)
    path = args.input/f'render-metrics-{args.start:03}.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'完成：{path}', flush=True)


if __name__ == '__main__':
    main()
