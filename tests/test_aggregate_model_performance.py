"""Regression coverage for runtime-root DA3 accuracy aggregation."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aggregate_model_performance as aggregate


def test_aggregate_writes_a_da3_runtime_report(tmp_path: Path) -> None:
    summary = (
        tmp_path
        / "floor_display2"
        / "accuracy_evaluation_da3"
        / "summary.txt"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text(
        "总体召回率 (Recall): 80.00%\n"
        "VGGT有效率 (Effectiveness): 70.00%\n"
        "Reference ID映射准确率 (Precision): 90.00%\n",
        encoding="utf-8",
    )

    report = aggregate.main(tmp_path)

    assert report == tmp_path / "overall_model_performance.txt"
    assert "3D Mapping (DA3)" in report.read_text(encoding="utf-8")


def test_aggregate_defaults_to_the_root_output_directory() -> None:
    assert aggregate.DEFAULT_OUTPUT_DIR == aggregate.REPOSITORY_ROOT / "Output"
