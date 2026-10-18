"""Paired video3 core run with function spans and explicit RNG state."""
import ast
import inspect
import json
import logging
import os
import random
import sys
import time

from detail_common import ROOT, OUT, DATA, record, span, wrap, memory_snapshot, save_outputs

sys.path.insert(0, str(ROOT))
started = time.perf_counter()
import numpy as np
import torch
import main
from src import web_viewer_export as viewer
from utils import pointcloud_filter as filters
from utils.profiling import dump_stages
import open3d as o3d

record("core_imports_and_instrumentation", started, time.perf_counter())
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
o3d.utility.random.seed(42)
DATA.update(profile_scope="One core pipeline; paired video3 function-level profiling, fixed input and seed42.",
            code_root=str(ROOT), status="running", seed=42)

# Capture the exact post-SOR/DBSCAN domain in both runs, before ground removal.
# This diagnostic is identical in both arms; copy/write costs are recorded.
captured = {}

def capture_ground(pcd, valid_idx, protected, median_nn, config):
    start = time.perf_counter()
    captured["ground_points"] = np.asarray(pcd.points).copy()
    captured["ground_protected"] = protected[valid_idx].copy()
    captured["ground_meta"] = dict(median_nn=float(median_nn), config=vars(config))
    record("diagnostic_ground_capture", start, time.perf_counter())

source = inspect.getsource(filters.filter_scene_points)
tree = ast.parse(source)
definition = tree.body[0]
for index, statement in enumerate(definition.body):
    if isinstance(statement, ast.If) and "config.remove_ground" in ast.unparse(statement.test):
        call = ast.parse("_capture_ground(pcd, valid_idx, protected, median_nn, config)").body[0]
        definition.body.insert(index, ast.copy_location(call, statement))
        break
else:
    raise RuntimeError("Ground capture boundary was not found")
ast.fix_missing_locations(tree)
namespace = dict(filters.__dict__, _capture_ground=capture_ground)
exec(compile(tree, inspect.getfile(filters.filter_scene_points), "exec"), namespace)
filters.filter_scene_points = namespace["filter_scene_points"]
viewer.filter_scene_points = filters.filter_scene_points

for owner, names in [
    (viewer, ("prepare_web_viewer_geometry", "export_web_viewer_bundle", "_load_da3_cache",
              "_sample_points", "_instance_labels_v2", "_select_background_voxels", "_publish_bundle",
              "_generate_thumbnails", "get_scene_orientation", "_attach_point_index_ranges")),
    (filters, ("prepare_scene_filter", "_cluster_labels")),
    (main.SKUDetectionMain, ("run_reconstruction", "run_sku_matching", "run_dedup_sequence", "validate_dataset")),
]:
    for name in names:
        wrap(owner, name)
wrap(viewer, "filter_scene_points")
if hasattr(filters, "_ground_filter_mask"):
    wrap(filters, "_ground_filter_mask")
for name in ("prepare_scene_filter", "_cluster_labels", "_ground_filter_mask"):
    if hasattr(filters, name):
        namespace[name] = getattr(filters, name)

original_ranges = viewer._attach_point_index_ranges
def ranges(objects, labels, label_keys):
    result = original_ranges(objects, labels, label_keys)
    start = time.perf_counter()
    captured["range_labels"] = labels.copy()
    captured["range_keys"] = label_keys
    record("diagnostic_ranges_capture", start, time.perf_counter())
    return result
viewer._attach_point_index_ranges = ranges

app = main.SKUDetectionMain()
app.save_root = OUT / "outputs"
app.match_backend = "da3"
app.classifier_enabled = False
app.config_path = OUT / "config.yaml"
started = time.perf_counter()
try:
    summary = app.run_complete_pipeline(str(OUT / "dataset"), algorithm="3d",
        model_path=os.environ["PROFILE_MODEL_PATH"], evaluate_accuracy=False, quiet_outputs=True,
        surfel_output_dir=OUT / "viewer", surfel_texture_edge=1920)
    record("core_pipeline", started, time.perf_counter())
    (OUT / "pipeline-summary.json").write_text(json.dumps(summary, indent=2))
    DATA["status"] = "success"
    before = time.perf_counter()
    for name in ("ground_points", "ground_protected", "range_labels"):
        np.save(OUT / (name + ".npy"), captured[name], allow_pickle=False)
    for name in ("ground_meta", "range_keys"):
        (OUT / (name + ".json")).write_text(json.dumps(captured[name], indent=2))
    record("diagnostic_domain_save", before, time.perf_counter())
    print(json.dumps(summary), flush=True)
finally:
    memory_snapshot("core.after_pipeline")
    dump_stages(OUT / "matching-stages.json")
    save_outputs()
