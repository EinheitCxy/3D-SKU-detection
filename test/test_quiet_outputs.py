from pathlib import Path
from types import SimpleNamespace
import ast
import pytest
import torch
import main
from src import inference
from utils.config import SKUMatchingConfig
from utils.sku_matching_system import SKUMatchingSystem
from src.deduplicate_detections import parse_all_matches
from test.test_main_pipeline import _pipeline_fixture, _write_enriched_detections


def test_quiet_pipeline_skips_auxiliary_stages(monkeypatch, tmp_path):
    app,dataset=_pipeline_fixture(monkeypatch,tmp_path)
    app.classifier_enabled=False
    _write_enriched_detections(dataset)
    def forbidden(*args,**kwargs): pytest.fail('auxiliary stage ran')
    for name in ['run_detection_visualization','run_improved_sku_analysis','run_accuracy_evaluation']:
        monkeypatch.setattr(app,name,forbidden)
    seen={}
    monkeypatch.setattr(app,'run_sku_matching',lambda *a,**kw:seen.update(kw) or {'success':True})
    monkeypatch.setattr(app,'run_dedup_sequence',lambda *a,**kw:seen.update(dedup=True) or {'success':True})
    result=app.run_complete_pipeline(str(dataset),'3d',quiet_outputs=True)
    assert seen['quiet_outputs'] and seen['dedup']
    assert result['matching'] and result['dedup'] and result['classification']
    assert 'improved_analysis' not in result


@pytest.mark.parametrize('yaml',[False,True])
def test_quiet_option_reaches_matching_config(monkeypatch,tmp_path,yaml):
    app,dataset=_pipeline_fixture(monkeypatch,tmp_path)
    app.config_path=main.PROJECT_ROOT/'config.yaml' if yaml else None
    seen=[]
    monkeypatch.setattr(inference,'run_3d_mapping',lambda args: seen.append(inference.create_config_from_args(args,'3d_mapping')) or {})
    app.run_sku_matching(str(dataset),'3d',batch_all_refs=False,backend='da3',save_json=True,quiet_outputs=True)
    assert seen[0].quiet_outputs and not seen[0].save_json


def test_quiet_matching_keeps_required_summary_without_images_or_json(monkeypatch,tmp_path):
    import utils.sku_matching_system as module
    system=object.__new__(SKUMatchingSystem)
    system.config=SKUMatchingConfig.for_3d_mapping(device='cpu',quiet_outputs=True,save_json=True,output_dir=str(tmp_path/'0'))
    def forbidden(*args,**kwargs): pytest.fail('optional output ran')
    monkeypatch.setattr(system,'_load_da3_visualization_images',forbidden)
    monkeypatch.setattr(module,'visualize_results',forbidden)
    monkeypatch.setattr(module,'save_correspondences_json',forbidden)
    system._post_process_results({1:[{'object_id':0,'target_obj_id':0,'correspondence_ratio':.9,'matched_points':9,'total_points':10}]},None,torch.empty((2,3,2,2),device='meta'),[],0,[],['0.jpg','2.jpg'])
    assert 'Found 1 matches' in (tmp_path/'0/matching_summary.txt').read_text()
    assert [p.name for p in (tmp_path/'0').iterdir()]==['matching_summary.txt']


def test_actual_docker_wrapper_enables_quiet_outputs(tmp_path):
    path=main.PROJECT_ROOT/'runtime/worktrees/mapping-docker-rebuild/processor.py'
    tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_mapping_request')
    seen={}
    pipeline=SimpleNamespace(run_complete_pipeline=lambda *a,**kw:seen.update(kw) or {'success':True})
    ns={'Path':Path,'SKUDetectionMain':lambda:pipeline,'MAIN_PROJECT_ROOT':main.PROJECT_ROOT,'export_web_viewer_bundle':lambda **kw:{'manifest_path':str(tmp_path/'viewer/manifest.json')}}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    ns['run_mapping_request'](tmp_path/'dataset',tmp_path/'out',tmp_path/'viewer','model')
    assert seen['quiet_outputs'] is True and seen['evaluate_accuracy'] is False
