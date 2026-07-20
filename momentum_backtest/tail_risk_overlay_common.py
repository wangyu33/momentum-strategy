#!/usr/bin/env python3
"""围绕 tail-risk overlay 研究链稳定暴露的公共 helper。"""

from __future__ import annotations

try:
    from .compare_tail_risk_bond_overlay import build_base_target_weights, load_market_proxy_best_context
except ImportError:
    from compare_tail_risk_bond_overlay import build_base_target_weights, load_market_proxy_best_context

