from __future__ import annotations

import pytest

from perf.surfel_latency import run_exports, validate_cpu_ids


def exercise(*, fail_export=None, fail_compare=None, raise_compare=False):
    events = []
    now = [0.0]
    durations = {"B1": 10, "C1": 6, "C2": 7, "B2": 12}
    hooks = []
    artifacts = {}

    def capture(label):
        events.append(("capture", label))
        now[0] += 2  # Equal page-copy work stays inside every timer.

    def export(label, hook):
        events.append(("export", label))
        hooks.append(hook)
        hook(label)
        now[0] += durations[label]
        if label == fail_export:
            raise RuntimeError("export failed")
        artifacts[label] = object()
        return artifacts[label]

    def compare(label, artifact, reference):
        events.append(("compare", label))
        assert artifact is artifacts[label]
        assert reference is artifacts["B1"]
        now[0] += 1000  # Must never enter raw export time.
        if label == fail_compare and raise_compare:
            raise RuntimeError("compare failed")
        return label != fail_compare

    result = run_exports(export=export, compare=compare, capture=capture, clock=lambda: now[0])
    return result, events, hooks, capture


def test_complete_order_same_capture_and_outside_comparison():
    result, events, hooks, capture = exercise()
    assert result["status"] == "complete"
    assert result["primary_saving_s"] == 4
    assert [run["raw_export_wall_s"] for run in result["runs"]] == [12, 8, 9, 14]
    assert all(run["compare_passed"] for run in result["runs"])
    assert hooks == [capture] * 4
    assert events == [(event, label) for label in ("B1", "C1", "C2", "B2")
                      for event in ("export", "capture", "compare")]


@pytest.mark.parametrize("label,count", [("B1", 1), ("C1", 2), ("C2", 3), ("B2", 4)])
def test_export_failure_retains_attempt_and_stops(label, count):
    result, events, _, _ = exercise(fail_export=label)
    assert result["status"] == "failed"
    assert result["primary_saving_s"] is None
    assert len(result["runs"]) == count
    assert result["runs"][-1]["raw_export_wall_s"] > 0
    assert result["runs"][-1]["export_complete"] is False
    assert ("compare", label) not in events


@pytest.mark.parametrize("raises", [False, True])
def test_last_compare_failure_does_not_accept_successful_subset(raises):
    result, _, _, _ = exercise(fail_compare="B2", raise_compare=raises)
    assert len(result["runs"]) == 4
    assert result["status"] == "failed"
    assert result["primary_saving_s"] is None
    assert result["runs"][-1]["raw_export_wall_s"] == 14
    assert result["runs"][-1]["export_complete"] is True
    assert result["runs"][-1]["compare_passed"] is False


def test_early_compare_failure_stops():
    result, events, _, _ = exercise(fail_compare="C1")
    assert len(result["runs"]) == 2
    assert events[-1] == ("compare", "C1")
    assert result["primary_saving_s"] is None


@pytest.mark.parametrize("requested", [[], [0, 1, 2], [0, 1, 2, 2],
                                        [0, 1, 2, 9], [-1, 0, 1, 2], [False, 1, 2, 3]])
def test_invalid_cpu_selection_is_rejected(requested):
    with pytest.raises(ValueError):
        validate_cpu_ids(requested, range(8))


def test_cpu_order_is_preserved_without_automatic_selection():
    assert validate_cpu_ids([7, 5, 3, 1], range(8)) == (7, 5, 3, 1)
