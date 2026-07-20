#!/usr/bin/env python3
"""围绕 goal optimization 研究链稳定暴露的公共 helper。"""

from __future__ import annotations

try:
    from .compare_goal_optimizations import (
        DEFAULT_REGIME_MIX_VOLUME_GUARD_RELIEF_BUFFER,
        DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN,
        GOAL_OUTPUT_DIR,
        MARKET_VOLUME_CACHE_PATH,
        METRIC_TOLERANCE,
        build_dynamic_core_target_weights,
        build_parametrized_hs300_trend_filter,
        load_market_volume_proxy,
        parse_regime_mix_strategy_name,
        send_improvement_notification,
    )
    from .daily_monitor import DEFAULT_FEISHU_WEBHOOK
except ImportError:
    from compare_goal_optimizations import (
        DEFAULT_REGIME_MIX_VOLUME_GUARD_RELIEF_BUFFER,
        DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN,
        GOAL_OUTPUT_DIR,
        MARKET_VOLUME_CACHE_PATH,
        METRIC_TOLERANCE,
        build_dynamic_core_target_weights,
        build_parametrized_hs300_trend_filter,
        load_market_volume_proxy,
        parse_regime_mix_strategy_name,
        send_improvement_notification,
    )
    from daily_monitor import DEFAULT_FEISHU_WEBHOOK
