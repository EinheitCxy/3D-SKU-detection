"""One full in-run detailed core profile; instrumentation overhead unquantified."""
import json
import logging
import os
import sys
import time
import traceback
from detail_common import ROOT, OUT, DATA, record, wrap, memory_snapshot, save_outputs

sys.path.insert(0,str(ROOT))
start=time.perf_counter()
import main
from core_detail import install_core
from utils.profiling import dump_stages
viewer,pvs=install_core()
record('core_imports_and_instrumentation',start,time.perf_counter())
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True)
for name in ('prepare_web_viewer_geometry','export_web_viewer_bundle','_load_da3_cache',
             '_sample_points','_instance_labels_v2','_select_background_voxels','_publish_bundle',
             '_generate_thumbnails','get_scene_orientation','_attach_point_index_ranges'):
    wrap(viewer,name)
wrap(pvs,'load_model','pvs_model_load')
wrap(pvs,'predict','pvs_predict',lambda result:dict(metrics=result[4],box_count=len(result[0])))
for name in ('run_reconstruction','run_dedup_sequence','validate_dataset'):
    wrap(main.SKUDetectionMain,name)
original_matching=main.SKUDetectionMain.run_sku_matching

def matching(self,*args,**kwargs):
    kwargs['enable_profiling']=True
    return original_matching(self,*args,**kwargs)

main.SKUDetectionMain.run_sku_matching=matching
wrap(main.SKUDetectionMain,'run_sku_matching')
wrap(main.SKUDetectionMain,'_run_single_matching','reference_matching')
app=main.SKUDetectionMain()
app.save_root=OUT/'outputs'
app.match_backend='da3'
app.classifier_enabled=False
app.config_path=OUT/'config.yaml'
DATA['profile_scope']='One core pipeline including in-run DA3 child, all references, PVS and Viewer/filters/dedup detail. Delivery is another process of this same run. No independent component replay.'
DATA['status']='running'
memory_snapshot('core.before_pipeline')
start=time.perf_counter()
try:
    summary=app.run_complete_pipeline(str(OUT/'dataset'),algorithm='3d',
        model_path=os.environ['PROFILE_MODEL_PATH'],evaluate_accuracy=False,quiet_outputs=True,
        surfel_output_dir=OUT/'viewer',surfel_texture_edge=1920)
    record('core_pipeline',start,time.perf_counter())
    (OUT/'pipeline-summary.json').write_text(json.dumps(summary,indent=2))
    DATA['status']='success'
    print(json.dumps(summary),flush=True)
except BaseException as error:
    DATA.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    record('core_pipeline',start,time.perf_counter(),error=repr(error))
    raise
finally:
    memory_snapshot('core.after_pipeline')
    dump_stages(OUT/'matching-stages.json')
    save_outputs()
