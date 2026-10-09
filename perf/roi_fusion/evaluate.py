"""Record local surface spread proxies and shared comparison metadata."""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def local_spreads(points, normals, center, axes, cell, reference_cells):
    keep = normals @ axes[:, 2] > np.cos(np.deg2rad(35))
    local = (points[keep] - center) @ axes
    keys = np.floor(local[:, :2] / cell).astype(int)
    groups = {}
    for k, height in zip(map(tuple, keys), local[:, 2]):
        if k in reference_cells:
            groups.setdefault(k, []).append(height)
    spread = {k: float(np.quantile(v, .9)-np.quantile(v, .1)) for k,v in groups.items() if len(v)>=3}
    return groups, spread


def main(root):
    data = dict(np.load(root / 'input.npz', allow_pickle=False))
    geometries = {}
    for name in ['baseline','fused','tsdf']:
        mesh = trimesh.load(root/f'{name}.glb', force='scene', process=False).to_geometry()
        rng = np.random.default_rng(42)
        areas = mesh.area_faces
        selection = rng.choice(len(mesh.faces), size=100000, p=areas/areas.sum())
        triangle = mesh.triangles[selection]
        a = np.sqrt(rng.random((len(selection),1)))
        b = rng.random((len(selection),1))
        samples = (1-a)*triangle[:,0] + a*(1-b)*triangle[:,1] + a*b*triangle[:,2]
        geometries[name] = (samples, mesh.face_normals[selection])
    ref = data['frame']==0
    normal = np.median(data['normals'][ref], axis=0);normal /= np.linalg.norm(normal)
    right = data['E'][0,0,:3].copy();right -= np.dot(right,normal)*normal;right /= np.linalg.norm(right)
    up = np.cross(normal,right)
    axes = np.stack([right,up,normal],axis=1)
    center = np.median(data['points'][ref],axis=0)
    cell = 4*float(data['spacing'])
    ref_points = data['points'][ref & (data['normals']@normal>np.cos(np.deg2rad(35)))]
    reference_cells = set(map(tuple,np.floor(((ref_points-center)@axes)[:,:2]/cell).astype(int)))
    measured = {k:local_spreads(p,n,center,axes,cell,reference_cells) for k,(p,n) in geometries.items()}
    common = set.intersection(*(set(spread) for _,spread in measured.values()))
    result = {'cell_size_m':cell,'reference_cells':len(reference_cells),'common_cells':len(common),'methods':{},
              'surface_samples_per_method':100000,'sampling_seed':42,
              'interpretation':'Internal estimated-geometry proxy: within-cell normal-axis P90-P10 spread. Not physical thickness or ground truth. All three exported surfaces use 100000 area-weighted samples. Check coverage and screenshots together; removing geometry can lower spread.'}
    if not common:
        raise ValueError('No shared front-surface cells to compare')
    for name,(groups,spread) in measured.items():
        values=np.array([spread[k] for k in sorted(common)])
        result['methods'][name]={'front_reference_coverage':len(groups)/len(reference_cells),
                                'normal_spread_median_m':float(np.median(values)),
                                'normal_spread_p90_m':float(np.quantile(values,.9))}
    (root/'geometry-metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    meta_path=root/'meta.json';meta=json.loads(meta_path.read_text());stats={}
    for name in geometries:
        path=root/f'{name}-stats.json'
        stats[name]=json.loads(path.read_text())
        stats[name]['geometry_proxy']=result['methods'][name]
    meta['stats']=stats;meta['geometry_metrics']='geometry-metrics.json'
    meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();main(args.root)
