"""One actual paired ZIP/GLB delivery; bounded atlas workers are explicit."""
import json
import os
import sys
import time

from detail_common import ROOT, OUT, DATA, record, span, wrap, memory_snapshot, save_outputs

sys.path.insert(0, str(ROOT))
started = time.perf_counter()
from src import surfel_mesh as mesh, surfel_glb as glb, surfel_lod as lod
from src import surfel_glb_texture as texture
from perf.viewer_clarity.pack import pack_bundle
record("delivery_imports", started, time.perf_counter())
for owner, name, label in [
    (mesh, "_bake_batch", "glb.bake_batch"),
    (mesh, "_bilinear", "glb.bilinear"),
    (mesh, "_dilate_rgb", "glb.dilate_rgb"),
    (glb, "_load", "glb.load"),
    (lod, "select_surfel_lod", "glb.LOD"),
    (texture, "encode_atlas", "glb.encode_atlas"),
]:
    wrap(owner, name, label)
workers = int(os.environ.get("PROFILE_ENCODER_WORKERS", "1"))
DATA.update(status="running", code_root=str(ROOT), encoder_workers=workers,
            profile_scope="One actual paired video3 delivery; nested function spans, no AST blocks.")
generation = OUT / "viewer/runs" / json.loads((OUT / "viewer/CURRENT").read_text())["run_id"]
try:
    before = time.perf_counter()
    with span("zip.total"):
        size = pack_bundle(generation, OUT / "viewer_bundle.zip")
    record("zip_pack", before, time.perf_counter(), bytes=size)
    before = time.perf_counter()
    with span("glb.total"):
        report = glb.export_scene_glb(generation, OUT / "scene.glb", encoder_workers=workers)
    record("glb_export", before, time.perf_counter())
    (OUT / "glb-report.json").write_text(json.dumps(report, indent=2))
    DATA["status"] = "success"
    print(json.dumps(report), flush=True)
finally:
    memory_snapshot("delivery.process_end")
    save_outputs()
