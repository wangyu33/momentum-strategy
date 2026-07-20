#!/usr/bin/env python3
"""围绕 market proxy 研究链稳定暴露的公共 helper。"""

from __future__ import annotations

try:
    from .compare_market_proxy_variants import build_proxy_catalog, build_risk_proxy_features, evaluate_strategy
except ImportError:
    from compare_market_proxy_variants import build_proxy_catalog, build_risk_proxy_features, evaluate_strategy

