#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/.tools/python_packages${PYTHONPATH:+:$PYTHONPATH}"
exec python3 momentum_backtest/desktop_app.py
