"""Public source layout after removing the historical code/ wrapper."""

from __future__ import annotations

import importlib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_3d_core_packages_are_importable_from_root_layout() -> None:
    assert importlib.import_module("src").__file__ is not None
    assert importlib.import_module("utils").__file__ is not None


def test_business_modules_have_clear_source_entrypoints() -> None:
    expected = (
        "modules/sku_detector/bbox_gen.py",
        "modules/personalcare_classifier/source/processor.py",
        "modules/viewer_web/package.json",
        "modules/video_to_dedup/run.sh",
    )
    for relative_path in expected:
        assert (REPOSITORY_ROOT / relative_path).is_file(), relative_path


def test_save_root_defaults_and_relative_paths_are_repository_anchored() -> None:
    import main

    assert main._resolve_save_root(None) == main.DEFAULT_SAVE_ROOT
    assert main._resolve_save_root("Output") == main.DEFAULT_SAVE_ROOT


def test_default_configuration_targets_the_da3_root_layout() -> None:
    from utils import load_yaml_config

    config = load_yaml_config(REPOSITORY_ROOT / "config.yaml")
    assert config["reconstruction"]["backend"] == "da3"
    assert config["inference"]["sam3_checkpoint_path"] == "sam3/checkpoints/sam3.pt"


def test_programmatic_and_direct_cli_defaults_target_the_root_da3_layout() -> None:
    import main

    assert main.SKUDetectionMain().match_backend == "da3"
    inference_source = (REPOSITORY_ROOT / "src" / "inference.py").read_text()
    assert 'default="imdata/floor_display2/images"' in inference_source
    assert 'default="imdata/floor_display2/detections_results"' in inference_source
    assert 'default="Output/floor_display2"' in inference_source


def test_direct_dedup_cli_uses_the_root_da3_output_contract() -> None:
    source = (REPOSITORY_ROOT / "src" / "deduplicate_detections.py").read_text()

    assert "default='imdata/floor_display2'" in source
    assert "--output_root" in source
    assert "default='Output'" in source
    assert "choices=['vggt', 'pi3', 'da3'], default='da3'" in source
    assert "imdata0911" not in source


def test_footprint_is_a_v2_read_only_sam3_cache_consumer() -> None:
    from src import da3_footprint_stage
    from utils import sam3_utils

    assert not hasattr(da3_footprint_stage, "_SAM3_CHECKPOINT")
    assert not hasattr(da3_footprint_stage, "sam3_utils")
    assert da3_footprint_stage.load_complete_frame_masks.__module__ == (
        "utils.sam3_mask_cache"
    )
    assert sam3_utils._ensure_sam3_in_path() == REPOSITORY_ROOT / "sam3"
