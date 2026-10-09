"""Package the controlled probes for the existing three-column local viewer."""
import json
from pathlib import Path
import shutil


def main():
    old = Path('runtime/roi-fusion-video3-gid3')
    root = Path('runtime/roi-fusion-video3-gid3-v2')
    geometry = json.loads((root/'geometry/summary.json').read_text())['methods']
    texture = json.loads((root/'texture-trunc8/summary.json').read_text())
    views = {
        'geometry-view': [
            (f'TSDF 截断 {k}×', root/f'geometry/trunc{k}/tsdf.glb', geometry[f'trunc{k}'])
            for k in (4, 8, 12)],
        'texture-view': [
            ('8× TSDF · 原分块贴图', root/'geometry/trunc8/tsdf.glb', geometry['trunc8']),
            ('8× TSDF · 邻接选源', root/'texture-trunc8/strict.glb',
             {'texture': texture['strict'], 'geometry_proxy': geometry['trunc8']['geometry_proxy']}),
            ('8× TSDF · 网格遮挡＋邻接选源', root/'texture-trunc8/mesh.glb',
             {'texture': texture['mesh'], 'geometry_proxy': geometry['trunc8']['geometry_proxy']}),
        ],
    }
    for name, variants in views.items():
        out = root/name
        out.mkdir(parents=True, exist_ok=True)
        meta = json.loads((old/'meta.json').read_text())
        meta['title'] = 'Video3 · TSDF 几何对照' if name == 'geometry-view' else 'Video3 · 同一 TSDF 网格的贴图对照'
        meta['notice'] = ('相机、输入深度、体素大小和贴图算法相同；仅改变 TSDF 截断距离。没有补洞或删除碎片。'
                          if name == 'geometry-view' else
                          '三列使用完全相同的网格。第三列同时使用网格首交遮挡检查和有界深度容差；不是单独射线检查的消融。灰色面保留。')
        meta['method_labels'], meta['stats'] = {}, {}
        meta.pop('geometry_metrics', None)
        for slot, (label, asset, stats) in zip(('baseline', 'fused', 'tsdf'), variants):
            shutil.copy2(asset, out/f'{slot}.glb')
            meta['method_labels'][slot] = label
            meta['stats'][slot] = stats
        for asset in ('observations.jpg', 'production-frame0.png', 'context.glb'):
            shutil.copy2(old/asset, out/asset)
        (out/'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
