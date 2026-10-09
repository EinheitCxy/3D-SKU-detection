"""一次固定规则的多视图 surfel 可见性实验，不改变位置、纹理或商品编号。

运行（项目根目录，Python 通过 uv）：
  uv run --no-sync python perf/surfel_compare/consistency.py BUNDLE OUTPUT

仅使用已有导出深度及相机，保持与前端读取的数据相同。向其他帧投影，
2x2 深度邻域全部有效且跨度 <= 最小深度的 3% 时才投票；深度差在
邻域平均深度的 1.5% 内算支持，点在表面后方算遮挡，前方算自由空间冲突。
至少 3 个冲突且冲突/(支持+冲突) >= 75% 才隐藏。不因缺乏支持而删除。
阈值在首次运行前固定，不按本场景结果调参。帧票并不统计独立，输入几何
也不是真值：此实验只检查内部一致性，不声称改善真实形状或计数准确率。
visibility.bin 每个原始点一个 uint8，1 保留，0 隐藏；原始 slot 顺序不变。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def votes(z, patch):
    """返回支持、遮挡、冲突；深度边界和无效邻域不投票。"""
    low, high = patch.min(axis=1), patch.max(axis=1)
    valid = np.isfinite(patch).all(axis=1) & (low > 0)
    valid &= (high - low) <= 0.03 * low
    observed = patch.mean(axis=1)
    delta, tolerance = z - observed, 0.015 * observed
    return (valid & (np.abs(delta) <= tolerance),
            valid & (delta > tolerance), valid & (delta < -tolerance))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    meta = json.loads((args.bundle / 'surfel.json').read_text())
    points = np.fromfile(args.bundle / 'positions.f32.bin', '<f4').reshape(-1, 3)
    source = np.fromfile(args.bundle / 'surfel-frame.u8.bin', 'u1')
    width, height = meta['grid_size']
    depth = np.fromfile(args.bundle / 'surfel-depth.f16.bin', '<f2').reshape(
        len(meta['frames']), height, width).astype(np.float32)
    assert len(points) == len(source) == meta['point_count']
    support, occlusion, conflict = [np.zeros(len(points), np.uint8) for _ in range(3)]
    for frame_id, frame in enumerate(meta['frames']):
        extrinsic = np.asarray(frame['extrinsic'], dtype=np.float32)
        intrinsic = np.asarray(frame['intrinsic'], dtype=np.float32)
        for first in range(0, len(points), 100_000):
            last = min(first + 100_000, len(points))
            camera = points[first:last] @ extrinsic[:, :3].T + extrinsic[:, 3]
            projected = camera @ intrinsic.T
            front = camera[:, 2] > 0
            xy = projected[:, :2] / np.where(front, projected[:, 2], 1)[:, None]
            valid = front & np.isfinite(xy).all(axis=1) & (source[first:last] != frame_id)
            valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width - 1)
            valid &= (xy[:, 1] >= 0) & (xy[:, 1] < height - 1)
            slots = np.flatnonzero(valid)
            x, y = np.floor(xy[valid]).astype(np.int64).T
            patch = np.stack([depth[frame_id, y, x], depth[frame_id, y, x + 1],
                              depth[frame_id, y + 1, x], depth[frame_id, y + 1, x + 1]], axis=1)
            for counts, voted in zip((support, occlusion, conflict), votes(camera[valid, 2], patch)):
                counts[first + slots] += voted.astype(np.uint8)
    informative = support.astype(np.int16) + conflict
    hidden = (conflict >= 3) & (conflict.astype(np.int16) * 4 >= informative * 3)
    visibility = (~hidden).astype(np.uint8)
    objects = json.loads((args.bundle / 'objects.json').read_text())
    per_object = {}
    for global_id, obj in objects.items():
        ranges = obj['point_ranges']
        before = sum(end - begin for begin, end in ranges)
        after = sum(int(visibility[begin:end].sum()) for begin, end in ranges)
        per_object[global_id] = {'before': before, 'after': after, 'hidden': before - after,
                                 'retained_fraction': after / before if before else None}
    def histogram(counts):
        return {str(i): int(n) for i, n in enumerate(np.bincount(counts, minlength=len(meta['frames']))) if n}
    stats = {
        'point_count': len(points), 'frame_count': len(meta['frames']),
        'rules': {'depth_relative_tolerance': 0.015, 'max_neighborhood_relative_spread': 0.03,
                  'minimum_conflict_votes': 3, 'minimum_conflict_fraction': 0.75,
                  'occlusion_counts_as_conflict': False, 'no_support_alone_is_deleted': False},
        'points_with_support': int((support > 0).sum()),
        'points_with_at_least_two_supports': int((support >= 2).sum()),
        'points_with_conflict': int((conflict > 0).sum()),
        'points_with_occlusion': int((occlusion > 0).sum()),
        'isolated_no_support_or_conflict': int((informative == 0).sum()),
        'retained_points': int(visibility.sum()), 'hidden_points': int(hidden.sum()),
        'retained_fraction': float(visibility.mean()),
        'support_histogram': histogram(support), 'conflict_histogram': histogram(conflict),
        'occlusion_histogram': histogram(occlusion), 'global_ids': per_object,
        'elapsed_seconds': time.perf_counter() - start,
        'limitation': 'Correlated estimated depths are not geometric ground truth. Visibility only; no geometry fusion.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    visibility.tofile(args.output / 'visibility.bin')
    (args.output / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
