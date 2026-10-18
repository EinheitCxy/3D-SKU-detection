"""Current-default core pipeline, with external classification from video3 JSON."""
import json
import logging
import os
import sys
import time
from profile_common import OUT, ROOT, record, wrap

sys.path.insert(0, str(ROOT))
start = time.perf_counter()
import main
import src.web_viewer_export as viewer
import src.surfel_export as surfel
import utils.pointcloud_filter as filters
import utils.sam31_pvs as pvs
from utils.profiling import dump_stages
record("core_imports", start, time.perf_counter())
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)

for name in ["prepare_web_viewer_geometry", "export_web_viewer_bundle", "_load_da3_cache",
             "_sample_points", "filter_scene_points", "_instance_labels_v2",
             "_select_background_voxels", "_publish_bundle", "_generate_thumbnails",
             "get_scene_orientation", "_attach_point_index_ranges"]:
    wrap(viewer, name)
wrap(viewer, "prepare_scene_filter", "prepare_scene_filter")
for name in ["grid_tangents", "prepare_surfel_textures", "prepare_surfels"]:
    wrap(surfel, name)
wrap(filters, "_median_nn_distance")
wrap(pvs, "load_model", "pvs_model_load")
wrap(pvs, "predict", "pvs_predict", lambda result: {"metrics":result[4], "box_count":len(result[0])})
for name in ["run_reconstruction", "run_dedup_sequence", "validate_dataset"]:
    wrap(main.SKUDetectionMain, name)
original_matching = main.SKUDetectionMain.run_sku_matching


def matching(self, *args, **kwargs):
    kwargs["enable_profiling"] = True
    return original_matching(self, *args, **kwargs)


main.SKUDetectionMain.run_sku_matching = matching
wrap(main.SKUDetectionMain, "run_sku_matching")
wrap(main.SKUDetectionMain, "_run_single_matching", "reference_matching")
app = main.SKUDetectionMain()
app.save_root = OUT / "outputs"
app.match_backend = "da3"
app.classifier_enabled = False
app.config_path = OUT / "config.yaml"
start = time.perf_counter()
summary = app.run_complete_pipeline(
    str(OUT / "dataset"), algorithm="3d", model_path=os.environ["PROFILE_MODEL_PATH"],
    evaluate_accuracy=False, quiet_outputs=True,
    surfel_output_dir=OUT / "viewer", surfel_texture_edge=1920,
)
record("core_pipeline", start, time.perf_counter())
dump_stages(OUT / "matching-stages.json")
(OUT / "pipeline-summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
