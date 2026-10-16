"""Injected four-export timing core; callers own artifacts, capture and comparison."""
from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Iterable


ORDER = ("B1", "C1", "C2", "B2")


def validate_cpu_ids(
    requested: Iterable[int], available: Iterable[int], *, required_count: int = 4
) -> tuple[int, ...]:
    """Validate an explicit CPU allocation without choosing or changing CPUs."""
    ids = tuple(requested)
    if type(required_count) is not int or required_count < 1:
        raise ValueError("required_count must be a positive integer")
    if len(ids) != required_count:
        raise ValueError(f"exactly {required_count} explicit CPU IDs required")
    if any(type(cpu) is not int or cpu < 0 for cpu in ids):
        raise ValueError("CPU IDs must be non-negative integers")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate CPU IDs")
    if not set(ids).issubset(set(available)):
        raise ValueError("requested CPU IDs are unavailable")
    return ids


def run_exports(
    *,
    export: Callable[[str, Callable[..., Any]], Any],
    compare: Callable[[str, Any, Any], bool],
    capture: Callable[..., Any],
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Run B1/C1/C2/B2; export chooses its arm from the label.

    The same capture callable runs inside each export timer. Comparison (including
    prequantization and final-output checks) runs after that timer stops and must
    return True. Export returns caller-owned artifacts, including page copies.
    Compare receives (label, artifact, B1 artifact), including a B1 self-check.
    A failed attempt retains its raw duration and stops the sequence. No adjusted
    timing or accepted metric is calculated from a successful subset. Hard process
    termination requires caller-level persistence; this core cannot catch SIGKILL.
    """
    result: dict[str, Any] = {
        "status": "partial", "runs": [], "primary_saving_s": None,
    }
    reference = None
    for label in ORDER:
        run = {"label": label, "export_complete": False, "compare_passed": False}
        result["runs"].append(run)
        started = clock()
        try:
            try:
                artifact = export(label, capture)
                run["export_complete"] = True
            finally:
                run["raw_export_wall_s"] = clock() - started
            if label == "B1":
                reference = artifact
            if compare(label, artifact, reference) is not True:
                raise ValueError("comparison did not pass")
            run["compare_passed"] = True
        except (Exception, SystemExit) as error:
            run["error"] = f"{type(error).__name__}: {error}"
            result["status"] = "failed"
            return result
    times = {run["label"]: run["raw_export_wall_s"] for run in result["runs"]}
    result["primary_saving_s"] = min(times["B1"] - times["C1"], times["B2"] - times["C2"])
    result["status"] = "complete"
    return result
