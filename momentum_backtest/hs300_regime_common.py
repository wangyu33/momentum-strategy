#!/usr/bin/env python3
"""围绕 hs300 regime 研究链稳定暴露的公共 helper。"""

from __future__ import annotations

try:
    from .compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
except ImportError:
    from compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
