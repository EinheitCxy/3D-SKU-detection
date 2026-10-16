"""Single approved RGB comparison; all checks are outside raw export timers."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['UV_CACHE_DIR'] = '/tmp/autoresearch-uv-cache'
import sys, json, time, resource, signal, traceback, types, threading, copy
from pathlib import Path
from io import BytesIO
TASK = Path(__file__).resolve().parent
REPO = Path('/home/xingyu/3D_Recognization')
ROOT = REPO / 'runtime/rgb-dilation-validation-20261016'
ART = ROOT / '.research/artifacts'
BINDING = ART / 'artifact-86e661cb4c15-98a006db/binding.json'
binding = json.loads(BINDING.read_text())
phase = sys.argv[sys.argv.index('--phase') + 1]
assert phase in ('smoke', 'official')
os.environ.update(binding['environment'])
TMP = Path(os.environ['TMPDIR']); TMP.mkdir(exist_ok=True)
import tempfile
tempfile.tempdir = str(TMP)
sys.path.insert(0, str(REPO))
started = time.monotonic()
result = {'phase': phase, 'status': 'running', 'raw': [], 'failures': [], 'metrics': {'minimum_paired_complete_cpu_export_saving_s': None}, 'binding': str(BINDING)}
sample_stop = threading.Event()
sampled_rss = [0]; child_seen = [False]; sample_count = [0]
def sample():
    while not sample_stop.is_set():
        try:
            rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
            children = Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()
            child_seen[0] |= bool(children)
            for child in children:
                rss += int(Path(f'/proc/{child}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
            sampled_rss[0] = max(sampled_rss[0], rss); sample_count[0] += 1
        except FileNotFoundError:
            pass
        sample_stop.wait(.1)
def save():
    r = resource.getrusage(resource.RUSAGE_SELF); c = resource.getrusage(resource.RUSAGE_CHILDREN)
    result['resources'] = {
        'cpu_process_seconds_including_children': r.ru_utime+r.ru_stime+c.ru_utime+c.ru_stime,
        'aggregate_peak_rss_gib': max(sampled_rss[0], r.ru_maxrss*1024)/1024**3,
        'self_ru_maxrss_gib': r.ru_maxrss/1024**2,
        'rss_sampling_interval_s': .1, 'rss_samples': sample_count[0],
        'children_observed': child_seen[0], 'gpu_seconds': 0,
        'official_wall_seconds' if phase == 'official' else 'smoke_wall_seconds': time.monotonic()-started,
        'enforced_disk_upper_bound_bytes': 603979776,
        'cycle_new_storage_gib': 603979776/1024**3,
        'measured_linked_disk_bytes': sum(p.stat().st_size for p in TASK.rglob('*') if p.is_file()),
        'max_output_size_bytes': max([r.get('output_bytes', 0) for r in result['raw']] or [0]),
        'address_space_limit_bytes': 12*1024**3, 'single_file_limit_bytes': 256*1024**2,
        'cpu_limit_seconds': 80 if phase == 'smoke' else 2000,
    }
    text = json.dumps(result, indent=2)
    assert len(text.encode()) < 64*1024**2
    pending = TASK / (phase + '.pending.json'); pending.write_text(text)
    pending.replace(TASK / (phase + '.json'))
def sigcpu(sig, frame):
    raise RuntimeError('SIGXCPU resource limit')
signal.signal(signal.SIGXCPU, sigcpu)
resource.setrlimit(resource.RLIMIT_CPU, (80 if phase == 'smoke' else 2000, 82 if phase == 'smoke' else 2002))
resource.setrlimit(resource.RLIMIT_AS, (12*1024**3, 12*1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (256*1024**2, 256*1024**2))

def frozen(artifact, filename, name):
    module = types.ModuleType(name); module.__file__ = str(ART / artifact / filename)
    sys.modules[name] = module
    exec(compile(Path(module.__file__).read_text(), module.__file__, 'exec'), module.__dict__)
    return module

def main():
    core = frozen('artifact-aa2a0cfcef26-91af91ca', 'perf__surfel_latency.py', 'frozen_core')
    ids = core.validate_cpu_ids(binding['cpu_ids'], os.sched_getaffinity(0))
    os.sched_setaffinity(0, ids)
    import numpy as np
    from PIL import Image
    import src
    def thread_affinity():
        affinities = {p.name: sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()}
        assert all(set(cpus).issubset(set(ids)) for cpus in affinities.values()), 'thread affinity escaped fixed CPU set'
        return affinities
    result['numerical_import_thread_affinities'] = thread_affinity()
    sampler = threading.Thread(target=sample, daemon=True); sampler.start()
    result['environment'] = {'python': sys.version, 'affinity': sorted(os.sched_getaffinity(0)), 'env': binding['environment'], 'argv': sys.argv, 'pid': os.getpid()}
    sources = [
        ('artifact-709b65780dd7-fd871eb4', 'src__surfel_mesh.py', 'surfel_mesh'),
        ('artifact-62cdb7505bfb-74fea448', 'src__surfel_glb.py', 'surfel_glb'),
        ('artifact-f981f6bd7c3d-87eab877', 'src__surfel_glb_texture.py', 'surfel_glb_texture'),
        ('artifact-3fb4e695cf9b-496d3287', 'src__surfel_glb_compact.py', 'surfel_glb_compact'),
        ('artifact-c3d77a236333-6721728e', 'src__surfel_lod.py', 'surfel_lod'),
    ]
    modules = {}
    for artifact, filename, name in sources:
        assert (REPO/'src'/(name+'.py')).read_text() == (ART/artifact/filename).read_text(), 'live/frozen source text mismatch: '+name
        modules[name] = frozen(artifact, filename, 'src.'+name)
        setattr(src, name, modules[name])
    baseline = modules['surfel_mesh']
    candidate = frozen('artifact-30a7e69979fe-6a03f72c', 'candidate_surfel_mesh.py', 'frozen_candidate')
    glb = modules['surfel_glb']; compact = modules['surfel_glb_compact']; lod = modules['surfel_lod']
    np.random.seed(binding['seed'])
    def duplicate(v):
        if isinstance(v, np.ndarray): return v.copy()
        if isinstance(v, dict): return {k: duplicate(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return type(v)(duplicate(x) for x in v)
        return copy.deepcopy(v)
    def equality(a, b):
        assert type(a) is type(b), 'type mismatch'
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a: equality(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b): equality(x, y)
        else: assert a == b
    def array_bytes(v):
        if isinstance(v, np.ndarray): return v.nbytes
        if isinstance(v, dict): return sum(array_bytes(x) for x in v.values())
        if isinstance(v, (list, tuple)): return sum(array_bytes(x) for x in v)
        return 0
    generation = Path(binding['generation'])
    if phase == 'smoke':
        cases = []
        rng = np.random.default_rng(0)
        for name, shape in [('empty',(0,8,8,4)), ('boundary_tie_multibatch',(7,8,8,4)), ('non_square',(3,3,7,4)), ('single',(2,1,1,4)), ('transparent',(3,8,8,4)), ('opaque',(3,8,8,4))]:
            tile = rng.integers(0,256,shape,dtype=np.uint8)
            tile[...,3] = 0
            if name == 'opaque': tile[...,3] = 255
            elif name not in ('empty','transparent'):
                tile[:,0,0,3] = 255; tile[:,-1,-1,3] = 255
            a=baseline._dilate_rgb(tile.copy()); b=candidate._dilate_rgb(tile.copy()); equality(a,b)
            np.testing.assert_array_equal(a[...,3],tile[...,3]); cases.append(name)
        result['synthetic_cases'] = cases
        tests = frozen('artifact-e796b5f20d6c-084f53c2','test__test_surfel_mesh.py','frozen_mesh_tests')
        names = [n for n in vars(tests) if n.startswith('test_')]
        for module in (baseline,candidate):
            tests.bake_surfel_pages=module.bake_surfel_pages
            # Helpers import src.surfel_mesh; bind it consistently with the baker.
            sys.modules['src.surfel_mesh']=module; src.surfel_mesh=module
            for name in names: getattr(tests,name)()
        result['existing_mesh_tests_both_arms']=names
        with glb._assets(generation) as read:
            _,inputs=glb._load(read,None)
            frame_id=int(inputs['frame_indices'][0]); slots=np.flatnonzero(inputs['frame_indices']==frame_id)[:1024]
            frame=inputs['frames'][frame_id]
            image=np.asarray(Image.open(BytesIO(read(frame['texture']))).convert('RGB'),dtype=np.float32)
            args=[inputs[k][slots].astype(np.float32) for k in ('positions','u','v','scales')]
            args += [inputs['depths'][frame_id].astype(np.float32),frame,image,2.,8]
            equality(baseline._bake_batch(*args),candidate._bake_batch(*args))
            result['real_batch']={'selected':len(slots),'complete_rgba_geometry_equal':True}
        del inputs, args, image
        fixture=frozen('artifact-226ae0e359aa-5dc39c11','test__test_surfel_glb.py','frozen_glb_tests')
        generation=fixture.bundle.__wrapped__(Path(tempfile.mkdtemp(prefix='smoke-', dir=TMP)))
    else:
        manifest=json.loads(Path(binding['manifest']).read_text()); meta=json.loads(Path(binding['surfel_metadata']).read_text())
        assert manifest['frame_count']==29 and meta['point_count']==2402589 and len(meta['frames'])==29
        assert manifest['schema_version']=='3.0.0' and manifest['backend'].lower()=='da3' and meta['version']==3 and meta['grid_size']==[280,504]
        result['input_counts']={'frames':29,'points':2402589}
    original_bakers={module:module.bake_surfel_pages for module in (baseline,candidate)}
    original_dilates={module:module._dilate_rgb for module in (baseline,candidate)}
    original_lod=lod.select_surfel_lod
    state={}; clock_values=[]
    def clock():
        if len(clock_values) % 2 == 0:
            result.setdefault('attempted_labels', []).append(core.ORDER[len(clock_values)//2])
            save()
        value=time.perf_counter(); clock_values.append(value); return value
    def capture(page):
        state['pages'].append(duplicate(page))
    def select(**kwargs):
        # Identical provenance capture and selector for each arm.
        state['lod_inputs']=duplicate({k:kwargs[k] for k in ('object_ids','frame_indices','frames')})
        chosen, report=original_lod(**kwargs)
        state['chosen']=chosen.copy()
        return chosen, report
    lod.select_surfel_lod=select
    def export(label, capture_fn):
        module=baseline if label.startswith('B') else candidate
        state.clear(); state.update(pages=[], dilation_wall_s=0.)
        def dilate(tiles):
            before=time.perf_counter()
            try: return original_dilates[module](tiles)
            finally: state['dilation_wall_s']+=time.perf_counter()-before
        def baker(**kwargs):
            state['baker_inputs']=duplicate({k:kwargs[k] for k in ('frame_indices','frames')})
            for page in original_bakers[module](**kwargs):
                capture_fn(page)
                yield page
        module._dilate_rgb=dilate
        # Exporter imports the exact shared wrapper regardless of chosen arm.
        baseline.bake_surfel_pages=baker
        sys.modules['src.surfel_mesh']=baseline; src.surfel_mesh=baseline
        output=TASK/(label+'.glb')
        state['output']=output
        try:
            state['production_report']=glb.export_scene_glb(generation,output,radius=binding['export']['radius'],tile_size=binding['export']['tile_size'],lod=binding['export']['lod'],palette_colors=binding['export']['palette_colors'])
        finally:
            module._dilate_rgb=original_dilates[module]
            baseline.bake_surfel_pages=original_bakers[baseline]
        return dict(state)
    baseline_artifact=[None]
    def compare(label, current, reference):
        raw={'label':label,'raw_export_wall_s':clock_values[-1]-clock_values[-2], 'export_complete':True,'compare_passed':False, 'dilation_wall_s':current['dilation_wall_s'],'output_bytes':current['output'].stat().st_size,'pages':len(current['pages']), 'capture_array_bytes':array_bytes(current),'production_report':current['production_report']}
        result['raw'].append(raw); save()
        for key in ('pages','lod_inputs','chosen','baker_inputs'):
            equality(current[key],reference[key])
        assert current['pages']
        for page in current['pages']:
            assert set(np.unique(page['image_rgba'][...,3])).issubset({0,255})
        doc,binary,size=compact._read(current['output'])
        if label=='B1': baseline_artifact[0]=(doc,bytes(binary),size)
        rd,rb,rs=baseline_artifact[0]
        assert size==rs; equality(doc,rd)
        for accessor in doc['accessors']:
            view=doc['bufferViews'][accessor['bufferView']]; offset=view.get('byteOffset',0); n=view['byteLength']
            assert binary[offset:offset+n]==rb[offset:offset+n]
        for entry in doc['images']:
            view=doc['bufferViews'][entry['bufferView']]; offset=view.get('byteOffset',0); n=view['byteLength']
            with Image.open(BytesIO(binary[offset:offset+n])) as left, Image.open(BytesIO(rb[offset:offset+n])) as right:
                equality(np.array(left.convert('RGBA')),np.array(right.convert('RGBA')))
        raw.update(compare_passed=True, complete_prequantization_rgba_geometry_equal=True, source_object_lod_input_and_selection_equal=True, final_accessor_payloads_equal=True, final_decoded_rgba_equal=True, full_document_associations_equal=True, output_size_equal=True)
        raw['thread_affinities'] = thread_affinity()
        result['baseline_reference_array_bytes']=array_bytes(reference)
        result['baseline_glb_ram_bytes']=len(rb)
        current['output'].unlink()
        if label!='B1':
            current.clear(); state.clear()
        save(); print(json.dumps({'completed':label,'raw_export_wall_s':raw['raw_export_wall_s']}),flush=True)
        return True
    core_result=core.run_exports(export=export,compare=compare,capture=capture,clock=clock)
    result['core']=core_result
    # Persist failed attempts even when export never returned an artifact.
    known={r['label'] for r in result['raw']}
    for record in core_result['runs']:
        if record['label'] not in known: result['raw'].append(record.copy())
    if core_result['status']!='complete':
        raise RuntimeError('four-arm sequence failed: '+str(core_result['runs'][-1]))
    assert not child_seen[0], 'unexpected child process'
    times={r['label']:r['raw_export_wall_s'] for r in result['raw']}
    differences=[times['B1']-times['C1'], times['B2']-times['C2']]
    result['paired_savings_s']=differences
    result['metrics']={'minimum_paired_complete_cpu_export_saving_s':min(differences),'quality_contract_violation_count':0,'nonimproving_pair_count':sum(x<=0 for x in differences)}
    result['status']='completed'
    result['resource_scope']='No subprocesses created. RSS sampled self and direct children at 0.1s plus self ru_maxrss; sampled aggregate may miss subinterval peaks. Disk is enforced structural bound (one BIN + one output each <=256MiB plus <=64MiB metadata), not measured unlinked-temp peak.'

try:
    main()
except BaseException as exc:
    result['status']='failed'
    result['metrics']['minimum_paired_complete_cpu_export_saving_s']=None
    result['failures'].append({'class':'resource' if isinstance(exc,MemoryError) or 'SIGXCPU' in str(exc) else 'implementation','error':repr(exc),'traceback':traceback.format_exc()})
    save()
    raise
finally:
    sample_stop.set(); save()
