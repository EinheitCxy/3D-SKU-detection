"""Reuse the established scalar profiler with an explicit experiment source root."""
import os
from pathlib import Path

_project = Path(__file__).resolve().parents[2]
_source = (_project / "reports/profiling-multivideo-20261018/detail_common.py").read_text()
_source = _source.replace(
    "ROOT = Path(__file__).resolve().parents[2]",
    "ROOT = Path(os.environ['PROFILE_CODE_ROOT']).absolute()",
    1,
)
exec(compile(_source, str(_project / "reports/profiling-multivideo-20261018/detail_common.py"), "exec"))
