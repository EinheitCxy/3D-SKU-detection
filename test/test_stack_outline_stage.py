import json
from pathlib import Path
import numpy as np
import pytest
from src import stack_outline_stage as candidate
from src import da3_footprint_stage as production
from utils.config import default_sam3_mask_cache_root
from utils.outline_exchange import validate_outline_document
from test.test_da3_footprint_stage import make_metric_fixture


def fixture(tmp_path, monkeypatch):
    dataset, root, _ = make_metric_fixture(tmp_path)
    cache_path = root / dataset.name / 'da3_cache' / 'predictions.npz'
    orientation, _ = production.get_scene_orientation(cache_path)
    transform = np.eye(4); transform[:3,:3] = orientation.rotation
    transform[:3,3] = [2,-3,4]
    manifest = dict(dataset_name=dataset.name, backend='DA3', frame_count=3,
                    world_to_view=transform.ravel().tolist(), display_bounds=[-10,-10,-10,10,10,10],
                    measurement=dict(coordinate_unit='m', horizontal_plane='viewer_xz', orientation_status='fitted'))
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(manifest))
    def forbidden(*args, **kwargs):
        pytest.fail('candidate must not use production OBB/shadow/model inference')
    monkeypatch.setattr(production, 'run_da3_footprint', forbidden)
    monkeypatch.setattr(production, 'fused_carton_footprint', forbidden)
    monkeypatch.setattr(production, 'build_shadow_evidence', forbidden)
    from utils import sam3_utils
    monkeypatch.setattr(sam3_utils, 'sam3_masks_from_bboxes_predict_inst', forbidden)
    return dict(dataset_path=dataset, save_root=root,
                sam3_mask_cache_root=default_sam3_mask_cache_root(root / dataset.name),
                viewer_manifest=path, output=tmp_path / 'candidate.json',
                gap_radius_m=.02, max_hole_area_m2=.04)


def test_candidate_cache_only_and_output(monkeypatch, tmp_path):
    args = fixture(tmp_path, monkeypatch)
    original = production._masked_observations
    def masked(*a, **k):
        assert k['collect_evidence_masks'] is False
        result = original(*a, **k)
        assert not result[3]
        return result
    monkeypatch.setattr(production, '_masked_observations', masked)
    data = candidate.run_stack_outline(**args)
    validate_outline_document(data, 'candidate')
    assert data['value_m2'] is None
    assert data['regions']
    assert args['output'].with_suffix('.png').exists()
    assert not (args['save_root'] / Path(args['dataset_path']).name / 'ground_stack_footprint').exists()
    with pytest.raises(FileExistsError):
        candidate.run_stack_outline(**args)


def test_orientation_mismatch(monkeypatch, tmp_path):
    args = fixture(tmp_path, monkeypatch)
    path = args['viewer_manifest']; data = json.loads(path.read_text())
    data['world_to_view'] = np.eye(4).ravel().tolist(); path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='rotation differs'):
        candidate.run_stack_outline(**args)
    assert not args['output'].exists()


def test_input_failure_and_programming_keyerror(monkeypatch, tmp_path):
    args = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(production, 'prepare_da3_inputs', lambda *a: (_ for _ in ()).throw(KeyError('programming')))
    with pytest.raises(KeyError, match='programming'):
        candidate.run_stack_outline(**args)


def test_partial_and_empty(monkeypatch, tmp_path):
    args = fixture(tmp_path, monkeypatch)
    from utils.stack_outline import OutlineResult
    from shapely.geometry import Polygon
    monkeypatch.setattr(candidate, 'extract_stack_outlines', lambda *a, **k: OutlineResult(
        (Polygon([(0,0),(1,0),(1,1),(0,1)]),), {'complete':False}, ('ambiguous hole',)))
    data = candidate.run_stack_outline(**args)
    assert data['status'] == 'needs_review' and data['total_area_m2'] == 1
    monkeypatch.setattr(candidate, 'extract_stack_outlines', lambda *a, **k: OutlineResult((), {'complete':False}, ('empty',)))
    data = candidate.run_stack_outline(**args, overwrite=True)
    assert not data['complete'] and not data['regions'] and data['value_m2'] is None


def test_missing_cache_and_weak_point_selection(monkeypatch, tmp_path):
    from utils.ground_stack_footprint import elevated_world_points, ObservationPoints, SupportPlane
    plane = SupportPlane(point=np.zeros(3), normal=np.array([0.,1.,0.]),
                         u_axis=np.array([1.,0.,0.]), v_axis=np.array([0.,0.,1.]),
                         inlier_count=0, inlier_fraction=0., p95_residual_m=0.)
    points = np.array([[0.,.02,0.], [1.,.03,0.], [0.,.01,1.]])
    assert len(elevated_world_points([ObservationPoints(8, points)], plane, {8:0.})) == 2
    args = fixture(tmp_path, monkeypatch)
    (args['save_root'] / Path(args['dataset_path']).name / 'da3_cache' / 'predictions.npz').unlink()
    with pytest.raises(FileNotFoundError):
        candidate.run_stack_outline(**args)
