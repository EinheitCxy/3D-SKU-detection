"""Install statement/event instrumentation once into the same production run."""
from __future__ import annotations
import ast
import functools
import importlib
import importlib.util
import sys
from detail_common import ROOT, DATA, current_context, span, memory_snapshot

SOURCES = {}
EVENTS = {}
DATA['source_mapping'] = SOURCES
DATA['cuda_events'] = EVENTS


def load_helper(filename, name):
    path = ROOT / 'reports/profiling-20261018' / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def instrument_module(path, selectors, name, deep=False):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    class Transformer(ast.NodeTransformer):
        def __init__(self):
            self.classes=[]
        def visit_ClassDef(self, node):
            self.classes.append(node.name)
            self.generic_visit(node)
            self.classes.pop()
            return node
        def visit_FunctionDef(self, node):
            qualified='.'.join(self.classes+[node.name])
            if qualified not in selectors:
                return node
            def statements(body):
                result=[]
                for statement in body:
                    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value,str):
                        result.append(statement)
                        continue
                    # PVS has important per-batch work nested in autocast/for;
                    # recursively span those statements without changing execution order.
                    if deep:
                        for attr in ('body','orelse','finalbody'):
                            child=getattr(statement,attr,None)
                            if isinstance(child,list) and child:
                                setattr(statement,attr,statements(child))
                    label=f'{path.relative_to(ROOT)}:{qualified}:L{statement.lineno}'
                    SOURCES[label]=dict(path=str(path.relative_to(ROOT)),function=qualified,line=statement.lineno,
                                        end_line=statement.end_lineno,statement=source.splitlines()[statement.lineno-1].strip()[:220])
                    wrapped=ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id='_detail_span',ctx=ast.Load()),args=[ast.Constant(label)],keywords=[]))],body=[statement])
                    result.append(ast.copy_location(wrapped,statement))
                return result
            node.body=statements(node.body)
            return node
    tree=Transformer().visit(tree)
    ast.fix_missing_locations(tree)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    module._detail_span=span
    sys.modules[name]=module
    exec(compile(tree,str(path),'exec'),module.__dict__)
    return module


def event_wrap(torch, owner, attr, name, pending, context=True):
    original=getattr(owner,attr)
    @functools.wraps(original)
    def call(*args,**kwargs):
        key=(current_context()+' > '+name) if context else name
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        try:
            with span(name+'.host'):
                return original(*args,**kwargs)
        finally:
            end.record()
            pending.append((key,begin,end))
    setattr(owner,attr,call)


def consume_events(pending):
    # Only call after the algorithm's existing CUDA synchronization boundary.
    for name,begin,end in pending:
        seconds=begin.elapsed_time(end)/1000
        row=EVENTS.setdefault(name,dict(event_s=0.,calls=0,max_s=0.))
        row['event_s']+=seconds
        row['calls']+=1
        row['max_s']=max(row['max_s'],seconds)
    pending.clear()


def install_da3():
    # Initialize the parent before canonical replacement to avoid circular import.
    old=importlib.import_module('depth_anything_3.model.da3')
    module=instrument_module(ROOT/'Depth-Anything-3/src/depth_anything_3/model/da3.py',{
        'DepthAnything3Net.forward','DepthAnything3Net._process_depth_head',
        'DepthAnything3Net._process_camera_estimation','DepthAnything3Net._process_ray_pose_estimation',
        'DepthAnything3Net._process_mono_sky_estimation','NestedDepthAnything3Net.forward',
        'NestedDepthAnything3Net._apply_metric_scaling','NestedDepthAnything3Net._apply_depth_alignment',
        'NestedDepthAnything3Net._handle_sky_regions'},'depth_anything_3.model.da3')
    package=importlib.import_module('depth_anything_3.model')
    package.da3=module
    for name in ('DepthAnything3Net','NestedDepthAnything3Net'):
        if getattr(package,name,None) is getattr(old,name):
            setattr(package,name,getattr(module,name))
    package.__export__=[module.NestedDepthAnything3Net,module.DepthAnything3Net]
    import torch
    from depth_anything_3.model.dinov2.dinov2 import DinoV2
    pending=[]
    for owner,attr,label in [
        (module.NestedDepthAnything3Net,'forward','da3.nested_forward'),
        (module.DepthAnything3Net,'forward','da3.branch_forward'),
        (DinoV2,'forward','da3.backbone_forward'),
        (module.NestedDepthAnything3Net,'_apply_metric_scaling','da3.metric_scaling'),
        (module.NestedDepthAnything3Net,'_apply_depth_alignment','da3.depth_alignment'),
        (module.NestedDepthAnything3Net,'_handle_sky_regions','da3.sky_handling')]:
        event_wrap(torch,owner,attr,label,pending)
    return pending


def install_core():
    viewer_diagnostic=load_helper('diagnose_viewer.py','multivideo_viewer_instrumentation')
    viewer_diagnostic.span=span
    viewer=viewer_diagnostic.install()
    DATA['viewer_counts']=viewer_diagnostic.COUNTS
    DATA['viewer_sources']=viewer_diagnostic.SOURCES
    pvs=instrument_module(ROOT/'utils/sam31_pvs.py',{'load_model','predict'},'utils.sam31_pvs',deep=True)
    import utils
    utils.sam31_pvs=pvs
    original_load=pvs.load_model
    original_predict=pvs.predict
    pending=[]
    def load(*args,**kwargs):
        with span('pvs.model_load'):
            model,count=original_load(*args,**kwargs)
        import torch
        event_wrap(torch,model,'forward_image','pvs.encoder',pending,context=False)
        event_wrap(torch,model,'_forward_sam_heads','pvs.decoder',pending,context=False)
        memory_snapshot('pvs.model_ready',model)
        return model,count
    def predict(*args,**kwargs):
        with span('pvs.predict'):
            result=original_predict(*args,**kwargs)
        consume_events(pending)  # native predict final sync already completed
        memory_snapshot('pvs.predict_end',args[0] if args else kwargs['model'])
        return result
    pvs.load_model=load
    pvs.predict=predict
    for attr in ('prepare_web_viewer_geometry','export_web_viewer_bundle'):
        original=getattr(viewer,attr)
        def boundary(*args,_fn=original,_label=attr,**kwargs):
            memory_snapshot(_label+'.before')
            try:
                return _fn(*args,**kwargs)
            finally:
                memory_snapshot(_label+'.after')
        setattr(viewer,attr,boundary)
    return viewer,pvs
