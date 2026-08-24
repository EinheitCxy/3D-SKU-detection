"""Project-owned shell entrypoints keep their paths rooted at this checkout."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_batch_operations_use_the_root_layout() -> None:
    batch = (REPOSITORY_ROOT / "scripts/3d/ops/batch.sh").read_text()
    launcher = (REPOSITORY_ROOT / "scripts/3d/ops/k.sh").read_text()

    assert 'PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"' in batch
    assert 'IMAGE_FOLDER="$PROJECT_ROOT/imdata/$FLOOR_DISPLAY/images"' in batch
    assert "utils/process_image_orientation.py" in batch
    assert "src/inference.py" in batch
    assert "../imdata" not in batch
    assert 'bash "$SCRIPT_DIR/batch.sh"' in launcher
    assert "code/" not in batch + launcher


def test_accuracy_all_uses_the_root_annotation_and_runtime_output() -> None:
    script = (
        REPOSITORY_ROOT / "scripts/3d/evaluation/batch_accuracy_evaluation_all.sh"
    ).read_text()

    assert 'ACCURACY_SCRIPT="$PROJECT_ROOT/accuracy_annotation.py"' in script
    assert 'uv run python "$ACCURACY_SCRIPT"' in script
    assert '结果目录: $PROJECT_ROOT/Output/$FD/' in script
    assert '["da3"]="output_3dmapping_da3:accuracy_evaluation_da3:3D Mapping (DA3)"' in script
    assert "for algo_type in pt vggt pi3 da3; do" in script
    assert "code/Output" not in script


def test_accuracy_defaults_and_ignored_local_references_use_current_layout() -> None:
    accuracy = (
        REPOSITORY_ROOT / "scripts/3d/evaluation/accuracy_evaluation.sh"
    ).read_text()
    batch = (
        REPOSITORY_ROOT / "scripts/3d/evaluation/batch_accuracy_evaluation.sh"
    ).read_text()
    ignore = (REPOSITORY_ROOT / ".gitignore").read_text()

    assert 'DATA_ROOT="${SAVE_ROOT:-$PROJECT_ROOT/Output}"' in accuracy
    assert 'DATA_ROOT="${SAVE_ROOT:-$PROJECT_ROOT/Output}"' in batch
    assert 'SUMMARY_DIR="$DATA_ROOT/batch_accuracy_results_${BACKEND}"' in batch
    assert "legacy/" in ignore
    assert "knowledge/" in ignore


def test_video_default_and_viewer_artifacts_are_root_layout_safe() -> None:
    video_script = (REPOSITORY_ROOT / "modules/video_to_dedup/run.sh").read_text()
    viewer_ignore = (REPOSITORY_ROOT / "modules/viewer_web/.gitignore").read_text()

    assert 'VIDEO_ARG="${1:-$REPO_DIR/small_fd_video/video-test/6-1.mp4}"' in video_script
    assert "public/data/CURRENT" in viewer_ignore
