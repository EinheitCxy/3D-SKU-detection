from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from perf.benchmark import (
    StageReceipt,
    StageExecution,
    StageSpec,
    CaseReceipt,
    build_case_stages,
    execute_stage,
    main,
    receipt_from_stage_observation,
    render_markdown_report,
    run_case,
    summarise_cases,
    parse_nvidia_smi_samples,
)
from perf.stage_entry import dispatch_stage, project_root_for_entry


def test_stage_receipt_requires_duration_exit_and_gpu_peaks() -> None:
    receipt = StageReceipt.from_payload(
        {
            "name": "reconstruction",
            "wall_seconds": 12.5,
            "exit_code": 0,
            "gpu_baseline_mib": 512,
            "gpu_peak_mib": 8192,
            "status": "completed",
        }
    )

    assert receipt.gpu_peak_delta_mib == 7680
    assert receipt.is_complete


def test_stage_receipt_rejects_missing_gpu_peak() -> None:
    payload = {
        "name": "reconstruction",
        "wall_seconds": 12.5,
        "exit_code": 0,
        "gpu_baseline_mib": 512,
        "status": "completed",
    }

    try:
        StageReceipt.from_payload(payload)
    except ValueError as error:
        assert "gpu_peak_mib" in str(error)
    else:
        raise AssertionError("missing GPU peak must be rejected")


def test_parse_nvidia_smi_samples_preserves_memory_peak() -> None:
    samples = parse_nvidia_smi_samples(
        "2026/08/21 18:10:00.000, 2, 512, 0, 12.50\n"
        "2026/08/21 18:10:00.100, 2, 8192, 97, 352.10\n"
    )

    assert [sample.memory_mib for sample in samples] == [512.0, 8192.0]
    assert max(sample.memory_mib for sample in samples) == 8192.0


def test_receipt_uses_highest_driver_sample_and_never_hides_baseline() -> None:
    samples = parse_nvidia_smi_samples(
        "2026/08/21 18:10:00.000, 2, 512, 0, 12.50\n"
        "2026/08/21 18:10:00.100, 2, 8192, 97, 352.10\n"
    )

    receipt = receipt_from_stage_observation(
        name="reconstruction",
        wall_seconds=12.5,
        exit_code=0,
        gpu_baseline_mib=1024,
        samples=samples,
    )

    assert receipt.gpu_peak_mib == 8192
    assert receipt.gpu_peak_delta_mib == 7168
    assert receipt.status == "completed"


def test_receipt_fails_closed_when_gpu_sampler_has_no_valid_sample() -> None:
    try:
        receipt_from_stage_observation(
            name="reconstruction",
            wall_seconds=12.5,
            exit_code=0,
            gpu_baseline_mib=1024,
            samples=[],
        )
    except ValueError as error:
        assert "GPU telemetry" in str(error)
    else:
        raise AssertionError("no GPU samples must not become a zero peak")


def test_cold_and_warm_cases_have_explicit_cache_boundary(tmp_path: Path) -> None:
    cold = build_case_stages(
        repo_root=tmp_path,
        dataset_name="floor_display2",
        case_root=tmp_path / "cold",
        cache_mode="cold",
    )
    warm = build_case_stages(
        repo_root=tmp_path,
        dataset_name="floor_display2",
        case_root=tmp_path / "warm",
        cache_mode="warm",
    )

    assert [stage.name for stage in cold][:2] == ["reconstruction", "matching"]
    assert "reconstruction" not in [stage.name for stage in warm]
    assert [stage.name for stage in warm][:2] == ["matching", "analysis_dedup"]
    assert all("floor_display2" in " ".join(stage.command) for stage in cold)


def test_matching_dispatch_enables_existing_stage_profiling(tmp_path: Path) -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def run_sku_matching(self, *args: object, **kwargs: object) -> dict[str, object]:
            self.calls.append((args, kwargs))
            return {"success": True, "duration_s": 3.0}

    app = FakeApp()
    result = dispatch_stage(
        stage="matching",
        app=app,
        dataset=tmp_path / "floor_display2",
        save_root=tmp_path / "output",
    )

    assert result["success"] is True
    assert app.calls == [
        (
            (str(tmp_path / "floor_display2"), "3d"),
            {"batch_all_refs": True, "backend": "da3", "enable_profiling": True},
        )
    ]


def test_stage_entry_resolves_the_root_core_service() -> None:
    repository_root = Path(__file__).resolve().parents[2]

    assert project_root_for_entry() == repository_root


def test_analysis_dedup_dispatch_stops_after_analysis_failure(tmp_path: Path) -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.dedup_called = False

        def run_improved_sku_analysis(
            self, *_: object, **__: object
        ) -> dict[str, object]:
            return {"success": False, "error": "missing matching output"}

        def run_dedup_sequence(self, *_: object, **__: object) -> dict[str, object]:
            self.dedup_called = True
            return {"success": True}

    app = FakeApp()
    result = dispatch_stage(
        stage="analysis_dedup",
        app=app,
        dataset=tmp_path / "floor_display2",
        save_root=tmp_path / "output",
    )

    assert result["success"] is False
    assert app.dedup_called is False


def test_viewer_export_dispatch_uses_case_specific_output(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def exporter(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"point_count": 42}

    result = dispatch_stage(
        stage="viewer_export",
        app=object(),
        dataset=tmp_path / "floor_display2",
        save_root=tmp_path / "cold" / "output",
        viewer_output=tmp_path / "warm" / "viewer-data",
        exporter=exporter,
    )

    assert result == {"success": True, "point_count": 42}
    assert captured["output_dir"] == tmp_path / "warm" / "viewer-data"
    assert captured["sam3_mask_cache_root"] == (
        tmp_path / "cold" / "output" / "floor_display2" / "sam3_mask_cache" / "v2"
    )


def test_rejected_formal_footprint_with_artifact_remains_viewer_eligible(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "measurement_report.json"
    report_path.write_text("{}", encoding="utf-8")

    result = dispatch_stage(
        stage="footprint",
        app=object(),
        dataset=tmp_path / "floor_display2",
        save_root=tmp_path / "output",
        footprint_runner=lambda *_: {
            "success": False,
            "status": "rejected",
            "report_path": str(report_path),
        },
    )

    assert result["success"] is True
    assert result["formal_status"] == "rejected"
    assert result["formal_success"] is False


def test_execute_stage_persists_stdout_and_driver_peak(tmp_path: Path) -> None:
    class FakeSampler:
        def start(self) -> None:
            return None

        def stop(self) -> list[object]:
            return parse_nvidia_smi_samples(
                "2026/08/21 18:10:00.000, 2, 4096, 93, 300.10\n"
            )

    spec = StageSpec(
        name="smoke",
        command=(sys.executable, "-c", "print('stage stdout')"),
        cwd=tmp_path,
    )
    result = execute_stage(
        spec=spec,
        logs_dir=tmp_path / "logs",
        telemetry_dir=tmp_path / "telemetry",
        gpu_index=2,
        query_gpu_memory_mib=lambda _: 512.0,
        sampler_factory=lambda *_: FakeSampler(),
    )

    assert result.receipt.is_complete
    assert result.receipt.gpu_peak_delta_mib == 3584
    assert result.stdout_path.read_text(encoding="utf-8") == "stage stdout\n"


def test_summary_excludes_incomplete_cases_from_means() -> None:
    def receipt(name: str, seconds: float) -> StageReceipt:
        return StageReceipt(name, seconds, 0, 100, 200, "completed")

    summary = summarise_cases(
        [
            CaseReceipt("fd2", "cold", (receipt("reconstruction", 10), receipt("matching", 20))),
            CaseReceipt("fd3", "cold", (receipt("reconstruction", 30), receipt("matching", 10))),
            CaseReceipt("fd4", "cold", (receipt("reconstruction", 99),)),
        ],
        expected_stages={"cold": ("reconstruction", "matching")},
    )

    assert summary["cold"]["completed_dataset_count"] == 2
    assert summary["cold"]["excluded_datasets"] == ["fd4"]
    assert summary["cold"]["stage_mean_seconds"] == {
        "reconstruction": 20.0,
        "matching": 15.0,
    }
    assert summary["cold"]["stage_mean_driver_peak_mib"] == {
        "reconstruction": 200.0,
        "matching": 200.0,
    }
    assert summary["cold"]["stage_mean_driver_peak_delta_mib"] == {
        "reconstruction": 100.0,
        "matching": 100.0,
    }
    assert summary["cold"]["end_to_end_mean_seconds"] == 35.0


def test_run_case_stops_after_failed_stage_and_writes_receipts(tmp_path: Path) -> None:
    spec = StageSpec("reconstruction", ("unused",), tmp_path)
    failed = StageExecution(
        StageReceipt("reconstruction", 1.0, 1, 100, 200, "failed"),
        tmp_path / "stdout.log",
        tmp_path / "stderr.log",
        tmp_path / "telemetry.csv",
    )

    seen: dict[str, object] = {}

    def executor(**kwargs: object) -> StageExecution:
        seen.update(kwargs)
        return failed

    case = run_case(
        repo_root=tmp_path,
        run_root=tmp_path / "run",
        dataset_name="floor_display2",
        cache_mode="cold",
        gpu_index=2,
        stage_builder=lambda **_: (spec,),
        executor=executor,
    )

    assert case.stages == (failed.receipt,)
    assert seen["environment"] == {
        "CUDA_VISIBLE_DEVICES": "2",
        "PLAYWRIGHT_BROWSERS_PATH": str(tmp_path / "perf" / ".playwright"),
    }
    payload = json.loads((tmp_path / "run" / "fd2" / "cold" / "stages.json").read_text())
    assert payload["stages"][0]["status"] == "failed"


def test_report_marks_missing_modes_without_fabricating_an_average() -> None:
    report = render_markdown_report(
        {
            "cold": {
                "completed_dataset_count": 0,
                "excluded_datasets": ["floor_display2"],
                "stage_mean_seconds": {},
                "end_to_end_mean_seconds": None,
            }
        }
    )

    assert "cold" in report
    assert "N/A (no complete cases)" in report
    assert "floor_display2" in report


def test_cli_help_exits_before_starting_a_benchmark(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert "Benchmark DA3 reconstruction" in capsys.readouterr().out


def test_cli_dry_run_executes_after_all_module_definitions() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repository_root / "perf" / "benchmark.py"), "--dry-run"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "floor_display2: cold -> warm" in result.stdout
