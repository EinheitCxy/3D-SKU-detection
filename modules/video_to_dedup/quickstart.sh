#!/bin/bash
# Quick entrypoint for the maintained video -> detection -> DA3 dedup workflow.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/run.sh" "$@"
