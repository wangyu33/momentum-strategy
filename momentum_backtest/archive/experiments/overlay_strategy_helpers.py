#!/usr/bin/env python3
"""历史防守覆盖层搜索脚本共用的策略实现辅助函数。"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

try:
    from ...hs300_regime_common import run_target_weights_strategy
    from ...tail_risk_overlay_common import build_base_target_weights
    from ...run_backtest import resolve_strategy_universe
except ImportError:
    from hs300_regime_common import run_target_weights_strategy
    from tail_risk_overlay_common import build_base_target_weights
    from run_backtest import resolve_strategy_universe


def apply_treasury_cap(
    base_weights: pd.DataFrame,
    *,
    risk_budget_codes: list[str],
    treasury_code: str,
    trigger_mask: pd.Series,
    risk_cap: float,
) -> pd.DataFrame:
    """按触发条件把风险仓压到上限，并把释放仓位切到国债。"""
    overlaid_weights, scale_mask, moved_weight = apply_risk_cap_with_moved_weight(
        base_weights,
        risk_budget_codes=risk_budget_codes,
        trigger_mask=trigger_mask,
        risk_cap=risk_cap,
    )
    if not scale_mask.any():
        return overlaid_weights

    overlaid_weights.loc[scale_mask, treasury_code] = overlaid_weights.loc[scale_mask, treasury_code].add(
        moved_weight.loc[scale_mask],
        fill_value=0.0,
    )
    return overlaid_weights


def apply_risk_cap(
    base_weights: pd.DataFrame,
    *,
    risk_budget_codes: list[str],
    trigger_mask: pd.Series,
    risk_cap: float | pd.Series,
) -> pd.DataFrame:
    """按触发条件压缩风险仓，但不处理释放出来的剩余仓位。"""
    overlaid_weights, _, _ = apply_risk_cap_with_moved_weight(
        base_weights,
        risk_budget_codes=risk_budget_codes,
        trigger_mask=trigger_mask,
        risk_cap=risk_cap,
    )
    return overlaid_weights


def apply_risk_cap_with_moved_weight(
    base_weights: pd.DataFrame,
    *,
    risk_budget_codes: list[str],
    trigger_mask: pd.Series,
    risk_cap: float | pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """按触发条件压缩风险仓，并返回命中掩码和释放出的仓位。"""
    overlaid_weights = base_weights.copy()
    row_risk_weight = base_weights[risk_budget_codes].sum(axis=1)
    risk_cap_series = (
        risk_cap.reindex(base_weights.index).astype("float64")
        if isinstance(risk_cap, pd.Series)
        else pd.Series(risk_cap, index=base_weights.index, dtype="float64")
    )
    scale_mask = trigger_mask.fillna(False) & (row_risk_weight > risk_cap_series)
    moved_weight = pd.Series(0.0, index=base_weights.index, dtype="float64")
    if not scale_mask.any():
        return overlaid_weights, scale_mask, moved_weight

    scale = pd.Series(1.0, index=base_weights.index, dtype="float64")
    scale.loc[scale_mask] = risk_cap_series.loc[scale_mask] / row_risk_weight.loc[scale_mask]
    overlaid_weights.loc[scale_mask, risk_budget_codes] = overlaid_weights.loc[scale_mask, risk_budget_codes].mul(
        scale.loc[scale_mask], axis=0
    )
    moved_weight.loc[scale_mask] = row_risk_weight.loc[scale_mask] - risk_cap_series.loc[scale_mask]
    return overlaid_weights, scale_mask, moved_weight


def apply_split_defensive_cap(
    base_weights: pd.DataFrame,
    *,
    risk_budget_codes: list[str],
    primary_code: str,
    secondary_code: str,
    secondary_share: float,
    trigger_mask: pd.Series,
    risk_cap: float,
) -> pd.DataFrame:
    """按触发条件压缩风险仓，并把释放仓位按比例分配到两个防守资产。"""
    overlaid_weights, scale_mask, moved_weight = apply_risk_cap_with_moved_weight(
        base_weights,
        risk_budget_codes=risk_budget_codes,
        trigger_mask=trigger_mask,
        risk_cap=risk_cap,
    )
    if not scale_mask.any():
        return overlaid_weights

    overlaid_weights.loc[scale_mask, primary_code] = overlaid_weights.loc[scale_mask, primary_code].add(
        moved_weight.loc[scale_mask] * (1.0 - secondary_share),
        fill_value=0.0,
    )
    overlaid_weights.loc[scale_mask, secondary_code] = overlaid_weights.loc[scale_mask, secondary_code].add(
        moved_weight.loc[scale_mask] * secondary_share,
        fill_value=0.0,
    )
    return overlaid_weights


def apply_two_tier_treasury_cap(
    base_weights: pd.DataFrame,
    *,
    risk_budget_codes: list[str],
    treasury_code: str,
    proxy: pd.DataFrame,
    weak_ratio_cut: float,
    weak_breadth_cut: float,
    weak_cap: float,
    extreme_ratio_cut: float,
    extreme_breadth_cut: float,
    extreme_cap: float,
    extreme_short_ratio_cut: float | None,
) -> pd.DataFrame:
    """按弱市/极弱市两档阈值压缩风险仓，并把释放仓位切到国债。"""
    weak_mask = build_market_stress_trigger(
        proxy,
        index=base_weights.index,
        ratio_cut=weak_ratio_cut,
        breadth_cut=weak_breadth_cut,
    )
    extreme_mask = build_market_stress_trigger(
        proxy,
        index=base_weights.index,
        ratio_cut=extreme_ratio_cut,
        breadth_cut=extreme_breadth_cut,
        short_ratio_cut=extreme_short_ratio_cut,
    )

    overlaid_weights = apply_treasury_cap(
        base_weights,
        risk_budget_codes=risk_budget_codes,
        treasury_code=treasury_code,
        trigger_mask=weak_mask & ~extreme_mask,
        risk_cap=weak_cap,
    )
    return apply_treasury_cap(
        overlaid_weights,
        risk_budget_codes=risk_budget_codes,
        treasury_code=treasury_code,
        trigger_mask=extreme_mask,
        risk_cap=extreme_cap,
    )


def apply_persistent_treasury_cap(
    base_weights: pd.DataFrame,
    *,
    risk_budget_codes: list[str],
    treasury_code: str,
    raw_trigger: pd.Series,
    enter_days: int,
    exit_days: int,
    risk_cap: float,
    persistence_builder: Callable[..., pd.Series],
) -> pd.DataFrame:
    """按持续触发状态机压缩风险仓，并把释放仓位切到国债。"""
    trigger_mask = persistence_builder(raw_trigger.fillna(False), enter_days=enter_days, exit_days=exit_days)
    return apply_treasury_cap(
        base_weights,
        risk_budget_codes=risk_budget_codes,
        treasury_code=treasury_code,
        trigger_mask=trigger_mask,
        risk_cap=risk_cap,
    )


def apply_persistent_market_stress_treasury_cap(
    base_weights: pd.DataFrame,
    proxy: pd.DataFrame,
    *,
    index: pd.Index,
    risk_budget_codes: list[str],
    treasury_code: str,
    ratio_cut: float,
    breadth_cut: float,
    enter_days: int,
    exit_days: int,
    risk_cap: float,
    persistence_builder: Callable[..., pd.Series],
    short_ratio_cut: float | None = None,
) -> pd.DataFrame:
    """基于市场弱势触发和持续状态机压缩风险仓，并把释放仓位切到国债。"""
    raw_trigger = build_market_stress_trigger(
        proxy,
        index=index,
        ratio_cut=ratio_cut,
        breadth_cut=breadth_cut,
        short_ratio_cut=short_ratio_cut,
    )
    return apply_persistent_treasury_cap(
        base_weights,
        risk_budget_codes=risk_budget_codes,
        treasury_code=treasury_code,
        raw_trigger=raw_trigger,
        enter_days=enter_days,
        exit_days=exit_days,
        risk_cap=risk_cap,
        persistence_builder=persistence_builder,
    )


def build_overlay_trigger(
    *,
    mode: str,
    prices_index: pd.Index,
    proxy: pd.DataFrame,
    base_result: pd.DataFrame,
    drawdown_cut: float | None,
    ratio_cut: float | None,
    breadth_cut: float | None,
) -> pd.Series:
    """按 overlay 模式统一构造触发掩码。"""
    if mode == "drawdown_only":
        if drawdown_cut is None:
            raise ValueError("drawdown_only mode requires drawdown_cut")
        return (base_result["drawdown"] <= drawdown_cut).fillna(False)
    if mode == "market_stress_only":
        if ratio_cut is None or breadth_cut is None:
            raise ValueError("market_stress_only mode requires ratio_cut and breadth_cut")
        return build_market_stress_trigger(
            proxy,
            index=prices_index,
            ratio_cut=ratio_cut,
            breadth_cut=breadth_cut,
        )
    raise ValueError(f"unsupported mode: {mode}")


def apply_simple_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    treasury_code: str,
    mode: str,
    risk_cap: float,
    drawdown_cut: float | None,
    ratio_cut: float | None,
    breadth_cut: float | None,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在当前基线上叠加简单切债规则。"""
    base_weights, mixed_momentum, base_result = build_base_target_weights(prices, proxy, params)
    configured_risk_codes = params.get("risk_codes")
    configured_defensive_codes = params.get("defensive_codes")
    active_risk_codes, _ = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in configured_risk_codes] if configured_risk_codes is not None else None,
        defensive_codes=[str(code) for code in configured_defensive_codes] if configured_defensive_codes is not None else None,
    )
    risk_budget_codes = list(active_risk_codes)
    if "510300" in prices.columns and "510300" not in risk_budget_codes:
        risk_budget_codes.append("510300")

    trigger_mask = build_overlay_trigger(
        mode=mode,
        prices_index=prices.index,
        proxy=proxy,
        base_result=base_result,
        drawdown_cut=drawdown_cut,
        ratio_cut=ratio_cut,
        breadth_cut=breadth_cut,
    )

    overlaid_weights = apply_treasury_cap(
        base_weights,
        risk_budget_codes=risk_budget_codes,
        treasury_code=treasury_code,
        trigger_mask=trigger_mask,
        risk_cap=risk_cap,
    )

    return run_target_weights_strategy(prices, selected, overlaid_weights, mixed_momentum, fee_rate, slippage_rate)


def apply_split_defensive_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    primary_code: str,
    secondary_code: str,
    secondary_share: float,
    mode: str,
    risk_cap: float,
    drawdown_cut: float | None,
    ratio_cut: float | None,
    breadth_cut: float | None,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在基础 target weights 上叠加双防守桶覆盖层。"""
    base_weights, mixed_momentum, base_result = build_base_target_weights(prices, proxy, params)
    configured_risk_codes = params.get("risk_codes")
    configured_defensive_codes = params.get("defensive_codes")
    active_risk_codes, _ = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in configured_risk_codes] if configured_risk_codes is not None else None,
        defensive_codes=[str(code) for code in configured_defensive_codes] if configured_defensive_codes is not None else None,
    )
    risk_budget_codes = list(active_risk_codes)
    if "510300" in prices.columns and "510300" not in risk_budget_codes:
        risk_budget_codes.append("510300")

    trigger_mask = build_overlay_trigger(
        mode=mode,
        prices_index=prices.index,
        proxy=proxy,
        base_result=base_result,
        drawdown_cut=drawdown_cut,
        ratio_cut=ratio_cut,
        breadth_cut=breadth_cut,
    )

    overlaid_weights = apply_split_defensive_cap(
        base_weights,
        risk_budget_codes=risk_budget_codes,
        primary_code=primary_code,
        secondary_code=secondary_code,
        secondary_share=secondary_share,
        trigger_mask=trigger_mask,
        risk_cap=risk_cap,
    )
    return run_target_weights_strategy(prices, selected, overlaid_weights, mixed_momentum, fee_rate, slippage_rate)


def build_market_stress_trigger(
    proxy: pd.DataFrame,
    *,
    index: pd.Index,
    ratio_cut: float,
    breadth_cut: float,
    short_ratio_cut: float | None = None,
) -> pd.Series:
    """构造基于量能/广度的弱市触发掩码。"""
    aligned_proxy = proxy.reindex(index).ffill()
    trigger_mask = (
        (aligned_proxy["market_amount_ratio_20_60"] < ratio_cut)
        & (aligned_proxy["market_breadth_proxy"] < breadth_cut)
    ).fillna(False)
    if short_ratio_cut is not None:
        trigger_mask = trigger_mask & (aligned_proxy["market_amount_ratio_5_20"] < short_ratio_cut).fillna(False)
    return trigger_mask


def apply_market_stress_treasury_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    *,
    treasury_code: str,
    risk_cap: float,
    ratio_cut: float,
    breadth_cut: float,
    fee_rate: float,
    slippage_rate: float,
    short_ratio_cut: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在基础 target weights 上叠加市场弱势切国债规则。"""
    base_weights, mixed_momentum, _ = build_base_target_weights(prices, proxy, params)
    configured_risk_codes = params.get("risk_codes")
    configured_defensive_codes = params.get("defensive_codes")
    active_risk_codes, _ = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in configured_risk_codes] if configured_risk_codes is not None else None,
        defensive_codes=[str(code) for code in configured_defensive_codes] if configured_defensive_codes is not None else None,
    )
    risk_budget_codes = list(active_risk_codes)
    if "510300" in prices.columns and "510300" not in risk_budget_codes:
        risk_budget_codes.append("510300")

    trigger_mask = build_market_stress_trigger(
        proxy,
        index=prices.index,
        ratio_cut=ratio_cut,
        breadth_cut=breadth_cut,
        short_ratio_cut=short_ratio_cut,
    )
    overlaid_weights = apply_treasury_cap(
        base_weights,
        risk_budget_codes=risk_budget_codes,
        treasury_code=treasury_code,
        trigger_mask=trigger_mask,
        risk_cap=risk_cap,
    )
    return run_target_weights_strategy(prices, selected, overlaid_weights, mixed_momentum, fee_rate, slippage_rate)


def apply_two_tier_market_stress_treasury_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    *,
    treasury_code: str,
    weak_ratio_cut: float,
    weak_breadth_cut: float,
    weak_cap: float,
    extreme_ratio_cut: float,
    extreme_breadth_cut: float,
    extreme_cap: float,
    fee_rate: float,
    slippage_rate: float,
    extreme_short_ratio_cut: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在基础 target weights 上叠加 two-tier 市场弱势切国债规则。"""
    base_weights, mixed_momentum, _ = build_base_target_weights(prices, proxy, params)
    configured_risk_codes = params.get("risk_codes")
    configured_defensive_codes = params.get("defensive_codes")
    active_risk_codes, _ = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in configured_risk_codes] if configured_risk_codes is not None else None,
        defensive_codes=[str(code) for code in configured_defensive_codes] if configured_defensive_codes is not None else None,
    )
    risk_budget_codes = list(active_risk_codes)
    if "510300" in prices.columns and "510300" not in risk_budget_codes:
        risk_budget_codes.append("510300")

    overlaid_weights = apply_two_tier_treasury_cap(
        base_weights,
        risk_budget_codes=risk_budget_codes,
        treasury_code=treasury_code,
        proxy=proxy,
        weak_ratio_cut=weak_ratio_cut,
        weak_breadth_cut=weak_breadth_cut,
        weak_cap=weak_cap,
        extreme_ratio_cut=extreme_ratio_cut,
        extreme_breadth_cut=extreme_breadth_cut,
        extreme_cap=extreme_cap,
        extreme_short_ratio_cut=extreme_short_ratio_cut,
    )
    return run_target_weights_strategy(prices, selected, overlaid_weights, mixed_momentum, fee_rate, slippage_rate)
