"""Stage timing instrumentation for the SKU matching pipeline.

Pure-additive: only perf_counter accumulation + json/log output.
DISABLED by default (zero overhead). Enable via set_enabled(True).
Never alters algorithm logic, sampling, scoring, or numeric paths.

Usage:
    from utils.profiling import StageTimer
    with StageTimer("cache_npz_load"):
        data = np.load(...)
    # or manual (for whole-function / single-line scopes):
    t0 = time.perf_counter()
    ...
    StageTimer.record("uniqueness_constraint", time.perf_counter() - t0)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# name -> {"total": float seconds, "calls": int}
_STAGES: Dict[str, Dict[str, float]] = {}
ENABLED: bool = False


def _accumulate(name: str, dt: float) -> None:
    rec = _STAGES.setdefault(name, {"total": 0.0, "calls": 0})
    rec["total"] += dt
    rec["calls"] += 1


class StageTimer:
    """Context-manager stage timer + module-level accumulator.

    When ENABLED is False (default), __enter__/__exit__/record are no-ops
    (zero overhead for production runs).
    """

    def __init__(self, name: str, enabled: Optional[bool] = None):
        self.name = name
        # None => follow the module-level ENABLED flag; explicit bool overrides
        self._force_enabled = enabled
        self.t0: Optional[float] = None
        self._active: bool = False

    def __enter__(self):
        self._active = ENABLED if self._force_enabled is None else self._force_enabled
        if self._active:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if not self._active:
            return False
        dt = time.perf_counter() - self.t0
        _accumulate(self.name, dt)
        return False

    @staticmethod
    def record(name: str, dt: float) -> None:
        """Manually record an externally-measured duration (no-op when disabled)."""
        if not ENABLED:
            return
        _accumulate(name, dt)


def set_enabled(enabled: bool) -> None:
    """Enable/disable instrumentation globally. Default False (no-op)."""
    global ENABLED
    ENABLED = bool(enabled)


def is_enabled() -> bool:
    return ENABLED


def reset_stages() -> None:
    """Clear all accumulated stage timings."""
    _STAGES.clear()


def get_stages() -> Dict[str, Dict[str, float]]:
    """Return a shallow copy of accumulated stage timings."""
    return {k: {"total": v["total"], "calls": v["calls"]} for k, v in _STAGES.items()}


def dump_stages(path) -> None:
    """Write accumulated stage timings to a JSON file (pretty-printed)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(get_stages(), f, indent=2, ensure_ascii=False)


def log_stages_sorted(logger_obj=None) -> None:
    """Log every stage's total/calls sorted by total descending."""
    log = logger_obj or logger
    if not _STAGES:
        log.info("[PROF] no stage timings recorded")
        return
    log.info("[PROF] stage breakdown (sorted by total desc):")
    for name, rec in sorted(
        _STAGES.items(), key=lambda kv: kv[1]["total"], reverse=True
    ):
        log.info(f"[PROF] {name}: total={rec['total']:.3f}s calls={rec['calls']}")
