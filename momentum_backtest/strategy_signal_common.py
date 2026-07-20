#!/usr/bin/env python3
"""围绕 signal-driven 历史研究稳定暴露的公共 helper。"""

from __future__ import annotations

try:
    from .compare_strategy_refinements import run_signal_strategy
except ImportError:
    from compare_strategy_refinements import run_signal_strategy

