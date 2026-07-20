#!/usr/bin/env python3
"""stress-bond 历史搜索链共用的策略实现辅助函数。

调用方默认来自 stress-bond 历史搜索链，而不是正式 28.2691 官方基线；
helper 只负责复用策略实现，不应隐式引入官方净值锚定。
"""

from __future__ import annotations

import pandas as pd

from overlay_strategy_helpers import apply_risk_cap, apply_treasury_cap

try:
    from ...goal_optimization_common import build_dynamic_core_target_weights
    from ...hs300_regime_common import run_target_weights_strategy
    from .archive_strategy_common import (
        DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
        DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        RISK_CODES,
    )
except ImportError:
    from goal_optimization_common import build_dynamic_core_target_weights
    from hs300_regime_common import run_target_weights_strategy
    from archive_strategy_common import (
        DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
        DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        RISK_CODES,
    )


def build_stressbond_target_weights(
    prices: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    vg_cap: float,
    overheat_cap: float,
    overheat_hi_cap: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """构造 stress-bond 链共用的基础目标权重。"""
    aggressive_weights, aggressive_momentum, _, _, aggressive_trend = build_dynamic_core_target_weights(
        prices,
        lookback=25,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        core_weight=float(params["aggressive_core_weight"]),
        hs300_mom60_cut=0.05,
        hs300_mom120_cut=0.10,
        hs300_ma_window=90,
    )
    conservative_weights, conservative_momentum, _, _, _ = build_dynamic_core_target_weights(
        prices,
        lookback=25,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        core_weight=float(params["conservative_core_weight"]),
        hs300_mom60_cut=0.04,
        hs300_mom120_cut=0.10,
        hs300_ma_window=120,
    )
    aggressive_mask = aggressive_trend & (aggressive_momentum >= float(params["regime_momentum_cut"]))
    mixed_target_weights = conservative_weights.copy()
    mixed_target_weights.loc[aggressive_mask] = aggressive_weights.loc[aggressive_mask]
    mixed_momentum = conservative_momentum.copy()
    mixed_momentum.loc[aggressive_mask] = aggressive_momentum.loc[aggressive_mask]
    mixed_momentum = mixed_momentum.rename("current_momentum")

    proxy = proxy.reindex(prices.index).ffill()
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    row_risk_weight = mixed_target_weights[risk_codes].sum(axis=1)
    volume_weak_mask = (
        (proxy["market_amount_ratio_20_60"] < float(params["volume_ratio_cut"]))
        & (proxy["market_amount_ratio_5_20"] < float(params["volume_short_ratio_cut"]))
        & (proxy["market_breadth_proxy"] < float(params["volume_breadth_cut"]))
        & (mixed_momentum <= float(params["volume_guard_momentum_ceiling"]))
    ).fillna(False)

    mixed_target_weights = apply_risk_cap(
        mixed_target_weights,
        risk_budget_codes=risk_codes,
        trigger_mask=volume_weak_mask,
        risk_cap=vg_cap,
    )

    base_result, _ = run_target_weights_strategy(
        prices,
        pd.DataFrame({"code": list(prices.columns), "theme": list(prices.columns), "name": list(prices.columns)}),
        mixed_target_weights,
        mixed_momentum,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
    )
    post_guard_risk = mixed_target_weights[risk_codes].sum(axis=1)
    reduce_mask = (
        (post_guard_risk > overheat_cap)
        & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
        & (mixed_momentum >= float(params["overheat_momentum_cut"]))
    )
    mixed_target_weights = apply_risk_cap(
        mixed_target_weights,
        risk_budget_codes=risk_codes,
        trigger_mask=reduce_mask,
        risk_cap=overheat_cap,
    )

    post_overheat_risk = mixed_target_weights[risk_codes].sum(axis=1)
    high_reduce_mask = (
        (post_overheat_risk > overheat_hi_cap)
        & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
        & (mixed_momentum >= float(params["overheat_high_momentum_cut"]))
    )
    mixed_target_weights = apply_risk_cap(
        mixed_target_weights,
        risk_budget_codes=risk_codes,
        trigger_mask=high_reduce_mask,
        risk_cap=overheat_hi_cap,
    )

    return mixed_target_weights, mixed_momentum


def apply_stressbond_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    treasury_code: str,
    risk_cap: float,
    ratio_cut: float,
    breadth_cut: float,
    vg_cap: float,
    overheat_cap: float,
    overheat_hi_cap: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在基础 regime-mix / 过热控制上叠加 stress-bond 切债规则。"""
    base_weights, mixed_momentum = build_stressbond_target_weights(
        prices=prices,
        proxy=proxy,
        params=params,
        vg_cap=vg_cap,
        overheat_cap=overheat_cap,
        overheat_hi_cap=overheat_hi_cap,
    )
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    proxy = proxy.reindex(prices.index).ffill()

    trigger_mask = (
        (proxy["market_amount_ratio_20_60"] < ratio_cut)
        & (proxy["market_breadth_proxy"] < breadth_cut)
    ).fillna(False)
    overlaid_weights = apply_treasury_cap(
        base_weights,
        risk_budget_codes=risk_codes,
        treasury_code=treasury_code,
        trigger_mask=trigger_mask,
        risk_cap=risk_cap,
    )

    return run_target_weights_strategy(prices, selected, overlaid_weights, mixed_momentum, fee_rate, slippage_rate)
