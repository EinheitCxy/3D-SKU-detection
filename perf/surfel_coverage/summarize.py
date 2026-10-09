"""Assemble matched comparisons and the isolated browser manifest."""
import argparse
import json
import shutil

import numpy as np

from perf.surfel_coverage.common import OLD, OUT, ROOT


def texture_transition():
    with np.load(OUT / 'fused/mesh.npz', allow_pickle=False) as z:
        vertices, faces, before = (z[k] for k in ('vertices', 'faces', 'sources'))
    with np.load(OUT / 'texture/mesh.npz', allow_pickle=False) as z:
        if not np.array_equal(vertices, z['vertices']) or not np.array_equal(faces, z['faces']):
            raise ValueError('Texture comparison must use exactly the same geometry')
        after = z['sources']
    triangles = vertices[faces].astype(np.float64)
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) / 2
    masks = {'newly_textured': (before < 0) & (after >= 0),
             'lost_texture': (before >= 0) & (after < 0),
             'retained_texture_changed_source': (before >= 0) & (after >= 0) & (before != after),
             'retained_same_source': (before >= 0) & (before == after),
             'both_untextured': (before < 0) & (after < 0)}
    return {name: {'faces': int(mask.sum()), 'triangle_area_fraction': float(area[mask].sum()/area.sum())}
            for name, mask in masks.items()}


def main(include_bounded=False):
    results = {}
    methods = ['uniform', 'fused', 'texture', 'adaptive', 'coverage']
    if include_bounded:
        methods.append('coverage_bounded')
    for method in methods:
        results[method] = {kind: json.loads((OUT / method / f'{filename}.json').read_text())
                           for kind, filename in (('stats', 'stats'), ('evaluation', 'evaluation'))}
    trials = {
        'texture': {'label': 'Retained sources + residual patches', 'reference': 'fused',
                    'reference_label': 'Fused Surfel · previous patch textures',
                    'description': 'Identical fused geometry. Valid primary sources stay fixed; only coherent residual regions receive another strictly visible source.'},
        'adaptive': {'label': 'Adaptive edge sampling', 'reference': 'uniform',
                    'reference_label': 'Uniform stride-4 · original textures',
                    'description': 'Same point budget and source texture rule. More samples near boundaries, with smaller edge footprints and sparser flat interiors.'},
        'coverage': {'label': 'Coverage-prioritized point budget', 'reference': 'uniform',
                    'reference_label': 'Uniform stride-4 · original textures',
                    'description': 'Same point budget, source texture rule and footprint scale. Spatial/normal occupancy and observation quality guide selection; occupancy is a coverage proxy.'},
    }
    if include_bounded:
        trials['coverage_bounded'] = {
            'label': 'Coverage budget · bounded exchange', 'reference': 'uniform',
            'reference_label': 'Uniform stride-4 · original textures',
            'description': 'One repair after the first coverage trial lost surfaces. At least 93.75% of baseline observations stay fixed; only a small supported subset is exchanged for novel observed cells. Same point budget and footprint scale.'}
    comparison = {}
    for method, trial in trials.items():
        candidate, reference = results[method], results[trial['reference']]
        comparison[method] = {
            'reference': trial['reference'],
            'delta_percentage_points': {key: 100*(value-reference['evaluation']['pooled'][key])
                for key, value in candidate['evaluation']['pooled'].items()},
            'untextured_area_delta_pp': 100*(candidate['stats']['texture']['untextured_area_fraction']-reference['stats']['texture']['untextured_area_fraction']),
            'point_count_delta': candidate['stats']['points']-reference['stats']['points'],
            'glb_bytes_delta': candidate['stats']['export']['glb_bytes']-reference['stats']['export']['glb_bytes'],
        }
    comparison['texture']['face_source_transition'] = texture_transition()
    (OUT / 'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    old = json.loads((OLD / 'meta.json').read_text())
    focus = json.loads((ROOT / 'runtime/roi-fusion-video3-gid3-v2/geometry-view/meta.json').read_text())
    manifest = {key: old[key] for key in ('center', 'radius', 'up', 'camera_positions')}
    manifest.update(focus={key: focus[key] for key in ('center', 'radius')},
                    results=results, trials=trials, comparisons=comparison)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    shutil.copyfile(OLD / 'observations.jpg', OUT / 'observations.jpg')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-bounded', action='store_true')
    main(parser.parse_args().include_bounded)
