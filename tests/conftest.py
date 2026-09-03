"""Expose flattened Docker runtime modules to Docker-focused tests."""

from __future__ import annotations

import sys
from pathlib import Path


DOCKER_RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "docker"
if str(DOCKER_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(DOCKER_RUNTIME_ROOT))
