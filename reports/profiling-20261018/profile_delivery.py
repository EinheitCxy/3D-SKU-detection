"""ZIP then current production-default compact GLB; no scene/page capture."""
import json
import sys
import time
from profile_common import OUT, ROOT, record

sys.path.insert(0, str(ROOT))
start = time.perf_counter()
from src.surfel_glb import export_scene_glb
from perf.viewer_clarity.pack import pack_bundle
record("delivery_imports", start, time.perf_counter())
run_id = json.loads((OUT / "viewer/CURRENT").read_text())["run_id"]
generation = OUT / "viewer/runs" / run_id
start = time.perf_counter()
size = pack_bundle(generation, OUT / "viewer_bundle.zip")
record("zip_pack", start, time.perf_counter(), bytes=size)
start = time.perf_counter()
report = export_scene_glb(generation, OUT / "scene.glb")
record("glb_export", start, time.perf_counter())
(OUT / "glb-report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
