"""Schema primitives for isolated DA3-to-viewer benchmark receipts.

The command runner is deliberately kept separate from existing reconstruction
code: timing must not change the numeric paths that it measures.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class StageReceipt:
    """One completed or failed stage with driver-visible GPU memory evidence."""

    name: str
    wall_seconds: float
    exit_code: int
    gpu_baseline_mib: float
    gpu_peak_mib: float
    status: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StageReceipt":
        required = (
            "name",
            "wall_seconds",
            "exit_code",
            "gpu_baseline_mib",
            "gpu_peak_mib",
            "status",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError("stage receipt missing: " + ", ".join(missing))
        name = payload["name"]
        status = payload["status"]
        if not isinstance(name, str) or not name:
            raise ValueError("stage receipt name must be a nonempty string")
        if status not in {"completed", "failed"}:
            raise ValueError("stage receipt status must be completed or failed")
        wall_seconds = float(payload["wall_seconds"])
        gpu_baseline_mib = float(payload["gpu_baseline_mib"])
        gpu_peak_mib = float(payload["gpu_peak_mib"])
        exit_code = payload["exit_code"]
        if wall_seconds < 0 or gpu_baseline_mib < 0 or gpu_peak_mib < 0:
            raise ValueError("stage receipt metrics must be nonnegative")
        if gpu_peak_mib < gpu_baseline_mib:
            raise ValueError("gpu_peak_mib cannot be lower than gpu_baseline_mib")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise ValueError("stage receipt exit_code must be an integer")
        return cls(
            name=name,
            wall_seconds=wall_seconds,
            exit_code=exit_code,
            gpu_baseline_mib=gpu_baseline_mib,
            gpu_peak_mib=gpu_peak_mib,
            status=status,
        )

    @property
    def gpu_peak_delta_mib(self) -> float:
        return self.gpu_peak_mib - self.gpu_baseline_mib

    @property
    def is_complete(self) -> bool:
        return self.status == "completed" and self.exit_code == 0


@dataclass(frozen=True)
class GpuSample:
    """One `nvidia-smi --loop-ms` row for the benchmark GPU."""

    timestamp: str
    gpu_index: int
    memory_mib: float
    utilization_percent: float
    power_watts: float


@dataclass(frozen=True)
class StageSpec:
    """A command boundary that produces one independent stage receipt."""

    name: str
    command: tuple[str, ...]
    cwd: Path


class GpuSampler(Protocol):
    """A sampler active while exactly one benchmark subprocess runs."""

    def start(self) -> None: ...

    def stop(self) -> list[GpuSample]: ...


@dataclass(frozen=True)
class StageExecution:
    """Stage receipt plus immutable command-output locations."""

    receipt: StageReceipt
    stdout_path: Path
    stderr_path: Path
    telemetry_path: Path


@dataclass(frozen=True)
class CaseReceipt:
    """All observed critical-path stages for one dataset and cache mode."""

    dataset: str
    cache_mode: str
    stages: tuple[StageReceipt, ...]

    def stage_by_name(self) -> dict[str, StageReceipt]:
        return {stage.name: stage for stage in self.stages}


class NvidiaSmiSampler:
    """Persist 100 ms driver telemetry using the host's NVIDIA CLI."""

    _QUERY = "timestamp,index,memory.used,utilization.gpu,power.draw"

    def __init__(self, gpu_index: int, telemetry_path: Path) -> None:
        self._gpu_index = gpu_index
        self._telemetry_path = telemetry_path
        self._process: subprocess.Popen[str] | None = None
        self._stream: Any | None = None

    def start(self) -> None:
        self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self._telemetry_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            [
                "nvidia-smi",
                "--id",
                str(self._gpu_index),
                f"--query-gpu={self._QUERY}",
                "--format=csv,noheader,nounits",
                "--loop-ms=100",
            ],
            stdout=self._stream,
            stderr=subprocess.PIPE,
            text=True,
        )

    def stop(self) -> list[GpuSample]:
        if self._process is None or self._stream is None:
            raise RuntimeError("NVIDIA sampler was not started")
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        self._stream.close()
        return parse_nvidia_smi_samples(
            self._telemetry_path.read_text(encoding="utf-8")
        )


def query_nvidia_memory_mib(gpu_index: int) -> float:
    """Read one driver-visible memory baseline before a stage starts."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu_index),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def parse_nvidia_smi_samples(output: str) -> list[GpuSample]:
    """Parse CSV rows emitted by the fixed sampler query.

    A malformed row is ignored because `nvidia-smi` can emit transient notices
    while a CUDA process is exiting. An empty valid sample set is handled by the
    stage runner as an explicit telemetry failure, never as a zero peak.
    """

    samples: list[GpuSample] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5 or any(value in {"", "N/A"} for value in fields):
            continue
        try:
            samples.append(
                GpuSample(
                    timestamp=fields[0],
                    gpu_index=int(fields[1]),
                    memory_mib=float(fields[2]),
                    utilization_percent=float(fields[3]),
                    power_watts=float(fields[4]),
                )
            )
        except ValueError:
            continue
    return samples


def receipt_from_stage_observation(
    *,
    name: str,
    wall_seconds: float,
    exit_code: int,
    gpu_baseline_mib: float,
    samples: Sequence[GpuSample],
) -> StageReceipt:
    """Create a fail-closed receipt from one process and its GPU samples."""

    if not samples:
        raise ValueError("GPU telemetry has no valid samples")
    peak = max(gpu_baseline_mib, *(sample.memory_mib for sample in samples))
    return StageReceipt(
        name=name,
        wall_seconds=wall_seconds,
        exit_code=exit_code,
        gpu_baseline_mib=gpu_baseline_mib,
        gpu_peak_mib=peak,
        status="completed" if exit_code == 0 else "failed",
    )


def execute_stage(
    *,
    spec: StageSpec,
    logs_dir: Path,
    telemetry_dir: Path,
    gpu_index: int,
    query_gpu_memory_mib: Callable[[int], float] = query_nvidia_memory_mib,
    sampler_factory: Callable[[int, Path], GpuSampler] = NvidiaSmiSampler,
    environment: Mapping[str, str] | None = None,
) -> StageExecution:
    """Run one stage and create an auditable wall-time and GPU receipt."""

    logs_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{spec.name}.stdout.log"
    stderr_path = logs_dir / f"{spec.name}.stderr.log"
    telemetry_path = telemetry_dir / f"{spec.name}.nvidia-smi.csv"
    baseline = query_gpu_memory_mib(gpu_index)
    sampler = sampler_factory(gpu_index, telemetry_path)
    child_environment = dict(os.environ)
    if environment:
        child_environment.update(environment)

    sampler.start()
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            spec.command,
            cwd=spec.cwd,
            env=child_environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
            text=True,
        )
    wall_seconds = time.perf_counter() - started
    samples = sampler.stop()
    receipt = receipt_from_stage_observation(
        name=spec.name,
        wall_seconds=wall_seconds,
        exit_code=completed.returncode,
        gpu_baseline_mib=baseline,
        samples=samples,
    )
    return StageExecution(
        receipt=receipt,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        telemetry_path=telemetry_path,
    )


def summarise_cases(
    cases: Sequence[CaseReceipt], *, expected_stages: Mapping[str, Sequence[str]]
) -> dict[str, dict[str, Any]]:
    """Average only cases with every required successful stage.

    This avoids turning a browser/telemetry failure into an apparent reduction
    in end-to-end latency.
    """

    summaries: dict[str, dict[str, Any]] = {}
    for mode, required_names in expected_stages.items():
        completed: list[CaseReceipt] = []
        excluded: list[str] = []
        for case in (case for case in cases if case.cache_mode == mode):
            by_name = case.stage_by_name()
            if all(
                name in by_name and by_name[name].is_complete for name in required_names
            ):
                completed.append(case)
            else:
                excluded.append(case.dataset)
        stage_means = {
            name: sum(case.stage_by_name()[name].wall_seconds for case in completed)
            / len(completed)
            for name in required_names
        } if completed else {}
        stage_peak_means = {
            name: sum(case.stage_by_name()[name].gpu_peak_mib for case in completed)
            / len(completed)
            for name in required_names
        } if completed else {}
        stage_peak_delta_means = {
            name: sum(
                case.stage_by_name()[name].gpu_peak_delta_mib for case in completed
            )
            / len(completed)
            for name in required_names
        } if completed else {}
        summaries[mode] = {
            "completed_dataset_count": len(completed),
            "excluded_datasets": sorted(excluded),
            "stage_mean_seconds": stage_means,
            "stage_mean_driver_peak_mib": stage_peak_means,
            "stage_mean_driver_peak_delta_mib": stage_peak_delta_means,
            "end_to_end_mean_seconds": sum(stage_means.values()) if stage_means else None,
        }
    return summaries


def _case_directory_name(dataset_name: str) -> str:
    prefix = "floor_display"
    return f"fd{dataset_name[len(prefix):]}" if dataset_name.startswith(prefix) else dataset_name


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_case(
    *,
    repo_root: Path,
    run_root: Path,
    dataset_name: str,
    cache_mode: str,
    gpu_index: int,
    stage_builder: Callable[..., Sequence[StageSpec]] | None = None,
    executor: Callable[..., StageExecution] = execute_stage,
) -> CaseReceipt:
    """Run all dependent stages for one cold or warm dataset case.

    A stage receipt is written immediately so an interrupted GPU job remains
    auditable. Downstream work stops at the first non-complete stage because it
    would otherwise consume stale or missing artifacts.
    """

    case_root = run_root / _case_directory_name(dataset_name) / cache_mode
    actual_stage_builder = stage_builder or build_case_stages
    stage_specs = actual_stage_builder(
        repo_root=repo_root,
        dataset_name=dataset_name,
        case_root=case_root,
        cache_mode=cache_mode,
    )
    completed: list[StageReceipt] = []
    persisted: list[dict[str, Any]] = []
    for spec in stage_specs:
        execution = executor(
            spec=spec,
            logs_dir=case_root / "logs",
            telemetry_dir=case_root / "telemetry",
            gpu_index=gpu_index,
            environment={
                "CUDA_VISIBLE_DEVICES": str(gpu_index),
                "PLAYWRIGHT_BROWSERS_PATH": str(
                    repo_root / "perf" / ".playwright"
                ),
            },
        )
        completed.append(execution.receipt)
        persisted.append(
            {
                **asdict(execution.receipt),
                "stdout_path": str(execution.stdout_path),
                "stderr_path": str(execution.stderr_path),
                "telemetry_path": str(execution.telemetry_path),
            }
        )
        _write_json(
            case_root / "stages.json",
            {
                "dataset": dataset_name,
                "cache_mode": cache_mode,
                "stages": persisted,
            },
        )
        if not execution.receipt.is_complete:
            break
    return CaseReceipt(dataset_name, cache_mode, tuple(completed))


def render_markdown_report(summary: Mapping[str, Mapping[str, Any]]) -> str:
    """Render averages without manufacturing a value for incomplete cases."""

    lines = ["# DA3 到 Web Viewer 性能报告", ""]
    for mode, details in summary.items():
        lines.extend([f"## {mode}", ""])
        completed = details["completed_dataset_count"]
        excluded = details["excluded_datasets"]
        if completed == 0:
            lines.extend(
                [
                    "端到端平均：N/A (no complete cases)",
                    "",
                    f"排除的数据集：{', '.join(excluded) if excluded else 'none'}",
                    "",
                ]
            )
            continue
        total = details["end_to_end_mean_seconds"]
        lines.extend(
            [
                f"完整数据集数：{completed}",
                f"端到端平均：{total:.3f} s",
                "",
                "| 阶段 | 平均耗时 (s) | 占端到端 | 平均 driver peak (MiB) | 平均新增 (MiB) |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, seconds in details["stage_mean_seconds"].items():
            share = 0.0 if total == 0 else seconds / total * 100
            peak = details["stage_mean_driver_peak_mib"][name]
            delta = details["stage_mean_driver_peak_delta_mib"][name]
            lines.append(
                f"| {name} | {seconds:.3f} | {share:.1f}% | {peak:.1f} | {delta:.1f} |"
            )
        lines.extend(
            [
                "",
                f"排除的数据集：{', '.join(excluded) if excluded else 'none'}",
                "",
            ]
        )
    return "\n".join(lines)


def _expected_stage_names(cache_mode: str) -> tuple[str, ...]:
    later = (
        "matching",
        "analysis_dedup",
        "footprint",
        "viewer_export",
        "browser_first_interactive",
    )
    return ("reconstruction", *later) if cache_mode == "cold" else later


def _case_payload(case: CaseReceipt) -> dict[str, Any]:
    return {
        "dataset": case.dataset,
        "cache_mode": case.cache_mode,
        "stages": [asdict(stage) for stage in case.stages],
        "complete": all(
            name in case.stage_by_name() and case.stage_by_name()[name].is_complete
            for name in _expected_stage_names(case.cache_mode)
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark DA3 reconstruction through viewer display")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=("floor_display2", "floor_display3", "floor_display4"),
    )
    parser.add_argument("--gpu-index", type=int, default=2)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the executable scheduling path without running stages",
    )
    args = parser.parse_args(argv)

    repository_root = Path(__file__).resolve().parents[1]
    if args.dry_run:
        for dataset_name in args.datasets:
            print(f"{dataset_name}: cold -> warm")
        return 0
    run_root = args.run_root or (
        repository_root
        / "perf"
        / "runs"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    run_root.mkdir(parents=True, exist_ok=False)
    preflight = repository_root / "perf" / "resources.preflight.json"
    if preflight.exists():
        shutil.copy2(preflight, run_root / "environment.preflight.json")

    cases: list[CaseReceipt] = []
    for dataset_name in args.datasets:
        cold = run_case(
            repo_root=repository_root,
            run_root=run_root,
            dataset_name=dataset_name,
            cache_mode="cold",
            gpu_index=args.gpu_index,
        )
        cases.append(cold)
        if _case_payload(cold)["complete"]:
            cases.append(
                run_case(
                    repo_root=repository_root,
                    run_root=run_root,
                    dataset_name=dataset_name,
                    cache_mode="warm",
                    gpu_index=args.gpu_index,
                )
            )

    expected = {mode: _expected_stage_names(mode) for mode in ("cold", "warm")}
    summary = summarise_cases(cases, expected_stages=expected)
    _write_json(
        run_root / "summary.json",
        {"generated_at": datetime.now(UTC).isoformat(), "cases": [_case_payload(case) for case in cases], "summary": summary},
    )
    (run_root / "report.md").write_text(render_markdown_report(summary), encoding="utf-8")
    print(run_root)
    return 0


def build_case_stages(
    *,
    repo_root: Path,
    dataset_name: str,
    case_root: Path,
    cache_mode: str,
) -> Sequence[StageSpec]:
    """Build the critical DA3-to-viewer path for one isolated case.

    Warm deliberately starts from matching: reconstructing again would measure
    a second DA3 inference rather than cache reuse. Every later stage is
    rerun, so the case still measures the path that creates a fresh bundle and
    opens it in a browser with an empty HTTP cache.
    """

    if cache_mode not in {"cold", "warm"}:
        raise ValueError("cache_mode must be cold or warm")
    dataset = repo_root / "imdata" / dataset_name
    save_root = (
        case_root / "output"
        if cache_mode == "cold"
        else case_root.parent / "cold" / "output"
    )
    receipt_root = case_root / "stage-payloads"
    entry = repo_root / "perf" / "stage_entry.py"
    browser = repo_root / "perf" / "browser-benchmark.mjs"
    code_dir = repo_root

    def stage_command(name: str) -> tuple[str, ...]:
        return (
            "uv",
            "run",
            "--offline",
            "python",
            str(entry),
            "--stage",
            name,
            "--dataset",
            str(dataset),
            "--save-root",
            str(save_root),
            "--viewer-output",
            str(case_root / "viewer-data"),
            "--payload-path",
            str(receipt_root / f"{name}.json"),
        )

    stages: list[StageSpec] = []
    if cache_mode == "cold":
        stages.append(
            StageSpec("reconstruction", stage_command("reconstruction"), code_dir)
        )
    for name in ("matching", "analysis_dedup", "footprint", "viewer_export"):
        stages.append(StageSpec(name, stage_command(name), code_dir))
    stages.append(
        StageSpec(
            "browser_first_interactive",
            (
                "node",
                str(browser),
                "--data-root",
                str(case_root / "viewer-data"),
                "--dataset-name",
                dataset_name,
                "--output",
                str(case_root / "browser.json"),
            ),
            repo_root / "perf",
        )
    )
    return tuple(stages)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
