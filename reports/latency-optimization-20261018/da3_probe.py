"""CPU-only real-checkpoint coverage and baseline/candidate loading comparison.

Run only in a coordinator-approved CPU window. No inference, GPU or checkpoint
copies/hashes. Pair mode runs two processes sequentially and retains receipts.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / 'runtime/profiling-20261018/da3-model/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7'


def single(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError("This CPU probe requires CUDA_VISIBLE_DEVICES=''")
    started = time.perf_counter()
    result = dict(mode=args.mode, status='running', pid=os.getpid(), model_path=str(args.model_path),
                  interpretation='Independent CPU component run; not a full-pipeline improvement. Coverage uses safetensors metadata only. Weight sample equality is not a full tensor equivalence test.')
    try:
        sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'Depth-Anything-3/src')]
        import torch
        import da3_model_loading as loading
        from depth_anything_3.api import DepthAnything3
        result['import_s'] = time.perf_counter() - started
        timings = {}
        original_init = DepthAnything3.__init__
        @functools.wraps(original_init)
        def constructor(self, *a, **kw):
            begin = time.perf_counter()
            original_init(self, *a, **kw)
            timings['constructor_s'] = time.perf_counter()-begin
        DepthAnything3.__init__ = constructor
        audit_report = {}
        original_audit = loading.audit_checkpoint
        def audit(*a, **kw):
            begin = time.perf_counter()
            report = original_audit(*a, **kw)
            timings['metadata_audit_s'] = time.perf_counter()-begin
            audit_report.update(report)
            return report
        loading.audit_checkpoint = audit
        torch.manual_seed(42)
        begin = time.perf_counter()
        if args.mode == 'candidate':
            model = loading.load_da3_model(str(args.model_path))
        else:
            class AuditedBaseline(DepthAnything3):
                @classmethod
                def _load_as_safetensor(cls, model, model_file, map_location, strict):
                    audit(model, model_file)
                    return super()._load_as_safetensor(model, model_file, map_location, True)
            model = AuditedBaseline.from_pretrained(str(args.model_path), strict=True)
        timings['CPU_construct_audit_load_s'] = time.perf_counter()-begin
        eligibility = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.modules.conv._ConvNd)) and module.weight.numel() >= loading.MIN_SKIPPED_WEIGHT_NUMEL:
                eligibility.append(dict(name=name+'.weight', numel=module.weight.numel(),
                                        bytes=module.weight.numel()*module.weight.element_size()))
        buffers = {row['name']: model.get_buffer(row['name']).detach().tolist()
                   for row in audit_report['nonpersistent_buffers'] if model.get_buffer(row['name']).numel() <= 64}
        samples = {}
        parameters = dict(model.named_parameters())
        for row in eligibility[::max(1,len(eligibility)//8)][:8]:
            tensor = parameters[row['name']].detach().reshape(-1)
            samples[row['name']] = [tensor[index].item() for index in (0, tensor.numel()//2, tensor.numel()-1)]
        result.update(status='success', timings=timings, coverage=audit_report,
                      eligible_random_weights=eligibility,
                      eligible_parameter_tensors=len(eligibility), eligible_bytes=sum(row['bytes'] for row in eligibility),
                      total_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                      nonpersistent_buffer_values=buffers, loaded_weight_samples=samples,
                      torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                      cuda_initialized=torch.cuda.is_initialized())
        if result['cuda_initialized']:
            raise RuntimeError('CPU probe unexpectedly initialized CUDA')
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        result.update(elapsed_s=time.perf_counter()-started, rss_hwm_kib=usage.ru_maxrss,
                      cpu_user_s=usage.ru_utime, cpu_system_s=usage.ru_stime)
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/(args.mode+'.json')).write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(mode=args.mode,status=result['status'],timings=result.get('timings'),
                             result=str(args.out/(args.mode+'.json')))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('baseline','candidate','pair'),default='pair')
    parser.add_argument('--model-path',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.mode!='pair':
        if (args.out/(args.mode+'.json')).exists():
            raise FileExistsError('Probe receipt already exists')
        single(args)
        return
    if args.out.exists():
        raise FileExistsError('Use a new probe directory')
    args.out.mkdir(parents=True)
    result=dict(status='running',interpretation='Sequential fresh-process CPU comparison; do not splice into GPU pipeline timing.')
    try:
        for mode in ('baseline','candidate'):
            command=[sys.executable,str(Path(__file__).resolve()),'--mode',mode,
                     '--model-path',str(args.model_path),'--out',str(args.out)]
            with (args.out/(mode+'.log')).open('w') as stream:
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        baseline=json.loads((args.out/'baseline.json').read_text())
        candidate=json.loads((args.out/'candidate.json').read_text())
        result.update(status='success', baseline_timings=baseline['timings'], candidate_timings=candidate['timings'],
                      sampled_loaded_parameters_equal=baseline['loaded_weight_samples']==candidate['loaded_weight_samples'],
                      nonpersistent_buffers_equal=baseline['nonpersistent_buffer_values']==candidate['nonpersistent_buffer_values'],
                      coverage_state_keys=candidate['coverage']['state_keys'],
                      checkpoint_keys=candidate['coverage']['checkpoint_keys'],
                      raw_missing=candidate['coverage']['raw_missing'],
                      covered_aliases=candidate['coverage']['covered_aliases'],
                      uncovered_missing=candidate['coverage']['uncovered_missing'],
                      physical_state_tensors=candidate['coverage']['physical_state_tensors'],
                      physical_parameters=candidate['coverage']['parameter_tensors'],
                      physical_persistent_buffers=candidate['coverage']['physical_persistent_buffers'],
                      physical_nonpersistent_buffers=candidate['coverage']['physical_nonpersistent_buffers'],
                      skipped_parameter_tensors=candidate['coverage']['skipped_parameter_tensors'],
                      skipped_bytes=candidate['coverage']['skipped_bytes'])
        if not result['sampled_loaded_parameters_equal'] or not result['nonpersistent_buffers_equal']:
            raise RuntimeError('Loaded samples or nonpersistent buffers differ')
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        (args.out/'pair.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
