"""DA3 subprocess instrumentation in the full run; argv starts with runner path."""
import functools
import sys
import time
import traceback
from detail_common import ROOT, DATA, span, record, wrap, memory_snapshot, save_outputs

sys.path[:0]=[str(ROOT),str(ROOT/'Depth-Anything-3/src')]
from core_detail import install_da3,instrument_module,consume_events
start=time.perf_counter()
with span('da3.imports_and_instrumentation'):
    pending=install_da3()
    script,*arguments=sys.argv[1:]
    from pathlib import Path
    runner=instrument_module(Path(script).resolve(),{'main'},'multivideo_da3_runner')
    from depth_anything_3.api import DepthAnything3
    import safetensors.torch
record('da3_imports_and_instrumentation',start,time.perf_counter())
wrap(DepthAnything3,'__init__','da3.network_constructor')
wrap(safetensors.torch,'load_file','da3.safetensors_load_file')
wrap(safetensors.torch,'load_model','da3.safetensors_load_model')
wrap(DepthAnything3,'load_state_dict','da3.state_assignment')
wrap(DepthAnything3,'to','da3.model_to_device')
original_pretrained=DepthAnything3.from_pretrained

def load(cls,*args,**kwargs):
    begin=time.perf_counter()
    with span('da3.load_cpu'):
        result=original_pretrained(*args,**kwargs)
    record('da3_load_cpu',begin,time.perf_counter())
    return result

DepthAnything3.from_pretrained=classmethod(load)
for attr,label in [('_preprocess_inputs','preprocess'),('_prepare_model_inputs','input_enqueue'),
                   ('_run_model_forward','forward_native_sync'),('_convert_to_prediction','prediction_D2H'),
                   ('_add_processed_images','processed_RGB'),('inference','inference_api')]:
    wrap(DepthAnything3,attr,'da3.'+label)
original_forward=DepthAnything3._run_model_forward

def forward(self,*args,**kwargs):
    memory_snapshot('da3.forward.before',self)
    try:
        result=original_forward(self,*args,**kwargs)
        # Original _run_model_forward has completed its own post-forward sync.
        consume_events(pending)
        return result
    finally:
        memory_snapshot('da3.forward.after',self)

DepthAnything3._run_model_forward=forward
original_to=DepthAnything3.to

def to(self,*args,**kwargs):
    result=original_to(self,*args,**kwargs)
    memory_snapshot('da3.model_ready',self)
    return result

DepthAnything3.to=to
wrap(runner,'_depth_to_world_points','da3.CPU_backprojection')
wrap(runner.np,'savez','da3.NPZ_write_excluding_argument_evaluation')
DATA.update(status='running',profile_scope='DA3 child of the same full pipeline; argv unchanged; native synchronizations retained.')
start=time.perf_counter()
try:
    with span('da3.runner_main'):
        runner.main(arguments)
    record('da3_runner_main',start,time.perf_counter())
    DATA['status']='success'
except BaseException as error:
    DATA.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    record('da3_runner_main',start,time.perf_counter(),error=repr(error))
    raise
finally:
    memory_snapshot('da3.process_end')
    save_outputs()
