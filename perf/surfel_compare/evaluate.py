"""固定相机离线对照；运行 uv run --no-sync python perf/surfel_compare/evaluate.py。

PSNR/SSIM 是输入视图重投影图像拟合，不是几何真值或文字识别准确率。
SSIM: 11x11 Gaussian sigma=1.5、population covariance、RGB均值、忽略5像素边界。
点击使用已有pipeline bbox/global_id：bbox各边内缩20%，排除其他原始bbox
覆盖区域，以全部剩余像素统计（确定性，无随机抽样）。标签0表示未分配，
可能为背景/漏覆盖/归属拒绝，不能直接称几何孔洞。DA3高斯归属为从现有
surfel表面派生，非模型原生语义。全部相机同时统计相对baseline标签覆盖
损失作为代理，不把无真值的新视角误报为几何准确度。只读取完整三组渲染。
"""

import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'modules/viewer_web/public/comparison'
RUN = ROOT / 'runtime/surfel-comparison'
MODES = ('baseline', 'consistent', 'gaussian')


def read_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def read_ids(path):
    rgb = read_rgb(path).astype(np.uint16)
    return rgb[..., 0] + 256 * rgb[..., 1]


def ssim_map(a, b):
    a, b = a.astype(np.float64) / 255, b.astype(np.float64) / 255
    filt = lambda x: gaussian_filter(x, sigma=(1.5, 1.5, 0), radius=(5, 5, 0), mode='reflect')
    ma, mb = filt(a), filt(b)
    va, vb, cov = filt(a*a)-ma*ma, filt(b*b)-mb*mb, filt(a*b)-ma*mb
    result = ((2*ma*mb + .01**2)*(2*cov + .03**2)
              / ((ma*ma + mb*mb + .01**2)*(va + vb + .03**2)))
    return result.mean(axis=2)


def image_metrics(image, reference, mask, ssim):
    mask = mask.copy()
    mask[:5] = False; mask[-5:] = False; mask[:, :5] = False; mask[:, -5:] = False
    n = int(mask.sum())
    if n == 0:
        return {'pixels': 0, 'mse': None, 'psnr_db': None, 'ssim': None}
    error = ((image.astype(np.float64)-reference)**2).mean(axis=2)
    mse = float(error[mask].mean())
    return {'pixels': n, 'mse': mse, 'psnr_db': float(10*np.log10(255**2/mse)) if mse else None,
            'ssim': float(ssim[mask].mean())}


def source_masks(source, mapping, affine, width, height):
    boxes = []
    for gid, observations in mapping.items():
        for observation in observations:
            if observation['image_id'] != source:
                continue
            x1, y1, x2, y2 = observation['bbox']
            corners = np.array([[x1, y1, 1], [x2, y2, 1]]) @ affine[source].T
            boxes.append((int(gid), *(corners[:, :2]*2).ravel()))
    yy, xx = np.mgrid[:height, :width]
    xx, yy = xx+.5, yy+.5
    coverage = np.zeros((height, width), np.uint16)
    for gid, x1, y1, x2, y2 in boxes:
        coverage += ((xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2))
    target = np.zeros((height, width), np.uint16)
    for gid, x1, y1, x2, y2 in boxes:
        dx, dy = .2*(x2-x1), .2*(y2-y1)
        keep = (xx >= x1+dx) & (xx < x2-dx) & (yy >= y1+dy) & (yy < y2-dy) & (coverage == 1)
        target[keep] = gid
    return coverage > 0, target


def ratio(num, den):
    return num/den if den else None


def main():
    started = time.perf_counter()
    cameras = json.loads((PUBLIC/'cameras.json').read_text())
    for mode in MODES:
        for i in range(len(cameras['frames'])):
            for suffix in ('.png', '-ids.png'):
                if not (PUBLIC/mode/f'{i:03}{suffix}').is_file():
                    raise FileNotFoundError(PUBLIC/mode/f'{i:03}{suffix}')
    mapping = json.loads((RUN/'global_mapping.json').read_text())
    with np.load(RUN/'baseline-cache.npz') as cache:
        affine = cache['source_to_processed_affine']
    width, height = cameras['width'], cameras['height']
    result = {'definitions': __doc__, 'width': width, 'height': height, 'views': [], 'summary': {}}
    for i, camera in enumerate(cameras['frames']):
        ids = {mode: read_ids(PUBLIC/mode/f'{i:03}-ids.png') for mode in MODES}
        view = {'index': i, 'kind': camera['kind'], 'source': camera['source'], 'methods': {}}
        reference = bbox_mask = target = None
        if camera['kind'] == 'source':
            reference = read_rgb(PUBLIC/'reference'/f'{i:03}.png')
            bbox_mask, target = source_masks(camera['source'], mapping, affine, width, height)
        for mode in MODES:
            label = ids[mode]
            values, counts = np.unique(label[label > 0], return_counts=True)
            baseline_assigned = ids['baseline'] > 0
            rec = {'assigned_pixels': int((label > 0).sum()),
                   'assigned_fraction': float((label > 0).mean()),
                   'baseline_assigned_pixels': int(baseline_assigned.sum()),
                   'baseline_assigned_now_unassigned': int((baseline_assigned & (label == 0)).sum()),
                   'visible_pixels_by_global_id': {str(k): int(v) for k, v in zip(values, counts)}}
            if reference is not None:
                rgb = read_rgb(PUBLIC/mode/f'{i:03}.png')
                assert rgb.shape == reference.shape == (height, width, 3)
                ssim = ssim_map(rgb, reference)
                rec['image_full'] = image_metrics(rgb, reference, np.ones((height, width), bool), ssim)
                rec['image_bbox'] = image_metrics(rgb, reference, bbox_mask, ssim)
                sample = target > 0
                n, assigned = int(sample.sum()), int((sample & (label > 0)).sum())
                matched = int((sample & (label == target)).sum())
                rec['click'] = {'sample_pixels': n, 'assigned_pixels': assigned, 'matched_pixels': matched,
                                'coverage': ratio(assigned, n), 'match_all_samples': ratio(matched, n),
                                'match_assigned_only': ratio(matched, assigned)}
            view['methods'][mode] = rec
        result['views'].append(view)
    for mode in MODES:
        summary = {}
        for kind in ('all', 'source', 'interpolated', 'lateral'):
            records = [v['methods'][mode] for v in result['views'] if kind == 'all' or v['kind'] == kind]
            baseline_total = sum(r['baseline_assigned_pixels'] for r in records)
            lost = sum(r['baseline_assigned_now_unassigned'] for r in records)
            summary[kind] = {'views': len(records), 'assigned_fraction': float(np.mean([r['assigned_fraction'] for r in records])),
                             'baseline_assigned_now_unassigned_fraction': ratio(lost, baseline_total)}
        records = [v['methods'][mode] for v in result['views'] if v['kind'] == 'source']
        for region in ('image_full', 'image_bbox'):
            entries = [r[region] for r in records]
            n = sum(e['pixels'] for e in entries)
            mse = sum(e['mse']*e['pixels'] for e in entries if e['pixels'])/n
            summary[region] = {'pixels': n, 'psnr_db_from_pooled_mse': float(10*np.log10(255**2/mse)) if mse else None,
                               'ssim_pixel_weighted': sum(e['ssim']*e['pixels'] for e in entries if e['pixels'])/n}
        totals = {key: sum(r['click'][key] for r in records) for key in ('sample_pixels', 'assigned_pixels', 'matched_pixels')}
        summary['click'] = {**totals, 'coverage': ratio(totals['assigned_pixels'], totals['sample_pixels']),
                            'match_all_samples': ratio(totals['matched_pixels'], totals['sample_pixels']),
                            'match_assigned_only': ratio(totals['matched_pixels'], totals['assigned_pixels'])}
        result['summary'][mode] = summary
    result['elapsed_seconds'] = time.perf_counter()-started
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    (RUN/'evaluation.json').write_text(payload)
    (PUBLIC/'evaluation.json').write_text(payload)
    print(json.dumps(result['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
