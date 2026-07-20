#!/usr/bin/env python3
"""当前正式 stress-bond 细化链共用的默认上下文。"""

from __future__ import annotations

try:
    from .overlay_candidate_catalog import lookup_overlay_asset
except ImportError:
    from overlay_candidate_catalog import lookup_overlay_asset

try:
    from .archive_data_loaders import load_named_market_proxy, load_recent_selected_prices
except ImportError:
    from archive_data_loaders import load_named_market_proxy, load_recent_selected_prices

try:
    from .hs300_regime_common import summarize
except ImportError:
    from hs300_regime_common import summarize

try:
    from .hs300_regime_common import run_target_weights_strategy
except ImportError:
    from hs300_regime_common import run_target_weights_strategy

try:
    from .overlay_strategy_helpers import apply_market_stress_treasury_overlay, apply_persistent_market_stress_treasury_cap
except ImportError:
    from overlay_strategy_helpers import apply_market_stress_treasury_overlay, apply_persistent_market_stress_treasury_cap

try:
    from .tail_risk_overlay_common import build_base_target_weights
except ImportError:
    from tail_risk_overlay_common import build_base_target_weights

try:
    from .variant_compare_helpers import append_variant_result
except ImportError:
    from variant_compare_helpers import append_variant_result

try:
    from ...official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

try:
    from .archive_strategy_common import (
        DEFAULT_REGIME_MIX_AGGRESSIVE_CORE_WEIGHT,
        DEFAULT_REGIME_MIX_CONSERVATIVE_CORE_WEIGHT,
        DEFAULT_REGIME_MIX_MOMENTUM_CUT,
        DEFAULT_REGIME_MIX_OVERHEAT_DRAWDOWN_CUT,
        DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MAX_EXPOSURE,
        DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT,
        DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE,
        DEFAULT_REGIME_MIX_OVERHEAT_MOMENTUM_CUT,
        DEFAULT_REGIME_MIX_PROXY_KIND,
        DEFAULT_REGIME_MIX_VOLUME_BREADTH_CUT,
        DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP,
        DEFAULT_REGIME_MIX_VOLUME_GUARD_MOMENTUM_CEILING,
        DEFAULT_REGIME_MIX_VOLUME_RATIO_CUT,
        DEFAULT_REGIME_MIX_VOLUME_SHORT_RATIO_CUT,
        DEFAULT_STRESS_BOND_BREADTH_CUT,
        DEFAULT_STRESS_BOND_CODE,
        DEFAULT_STRESS_BOND_RATIO_CUT,
        DEFAULT_STRESS_BOND_RISK_CAP,
        DEFAULT_STRATEGY_NAME,
        build_persistent_trigger_mask,
        load_default_strategy_backtest_pool,
    )
except ImportError:
    from archive_strategy_common import (
        DEFAULT_REGIME_MIX_AGGRESSIVE_CORE_WEIGHT,
        DEFAULT_REGIME_MIX_CONSERVATIVE_CORE_WEIGHT,
        DEFAULT_REGIME_MIX_MOMENTUM_CUT,
        DEFAULT_REGIME_MIX_OVERHEAT_DRAWDOWN_CUT,
        DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MAX_EXPOSURE,
        DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT,
        DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE,
        DEFAULT_REGIME_MIX_OVERHEAT_MOMENTUM_CUT,
        DEFAULT_REGIME_MIX_PROXY_KIND,
        DEFAULT_REGIME_MIX_VOLUME_BREADTH_CUT,
        DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP,
        DEFAULT_REGIME_MIX_VOLUME_GUARD_MOMENTUM_CEILING,
        DEFAULT_REGIME_MIX_VOLUME_RATIO_CUT,
        DEFAULT_REGIME_MIX_VOLUME_SHORT_RATIO_CUT,
        DEFAULT_STRESS_BOND_BREADTH_CUT,
        DEFAULT_STRESS_BOND_CODE,
        DEFAULT_STRESS_BOND_RATIO_CUT,
        DEFAULT_STRESS_BOND_RISK_CAP,
        DEFAULT_STRATEGY_NAME,
        build_persistent_trigger_mask,
        load_default_strategy_backtest_pool,
    )


CURRENT_STRESSBOND_RISK_CODES = ["510300", "159949", "159954", "159941", "513650", "513880"]


def build_current_stressbond_params() -> dict[str, float | list[str]]:
    return {
        "drop_codes": ["511580", "513650"],
        "aggressive_core_weight": DEFAULT_REGIME_MIX_AGGRESSIVE_CORE_WEIGHT,
        "conservative_core_weight": DEFAULT_REGIME_MIX_CONSERVATIVE_CORE_WEIGHT,
        "regime_momentum_cut": DEFAULT_REGIME_MIX_MOMENTUM_CUT,
        "volume_ratio_cut": DEFAULT_REGIME_MIX_VOLUME_RATIO_CUT,
        "volume_short_ratio_cut": DEFAULT_REGIME_MIX_VOLUME_SHORT_RATIO_CUT,
        "volume_breadth_cut": DEFAULT_REGIME_MIX_VOLUME_BREADTH_CUT,
        "volume_guard_cap": DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP,
        "volume_guard_momentum_ceiling": DEFAULT_REGIME_MIX_VOLUME_GUARD_MOMENTUM_CEILING,
        "overheat_drawdown_cut": DEFAULT_REGIME_MIX_OVERHEAT_DRAWDOWN_CUT,
        "overheat_momentum_cut": DEFAULT_REGIME_MIX_OVERHEAT_MOMENTUM_CUT,
        "overheat_max_exposure": DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE,
        "overheat_high_momentum_cut": DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT,
        "overheat_high_max_exposure": DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MAX_EXPOSURE,
    }


def load_current_stressbond_context() -> dict[str, object]:
    treasury_row = lookup_overlay_asset(DEFAULT_STRESS_BOND_CODE)
    return {
        "strategy": DEFAULT_STRATEGY_NAME,
        "description": (
            "当前正式基线：hybrid breadth proxy + regime mix + 弱量能限仓 + 过热双层降仓 + "
            f"{treasury_row['name']}弱市防守。"
        ),
        "treasury_code": DEFAULT_STRESS_BOND_CODE,
        "risk_cap": DEFAULT_STRESS_BOND_RISK_CAP,
        "ratio_cut": DEFAULT_STRESS_BOND_RATIO_CUT,
        "breadth_cut": DEFAULT_STRESS_BOND_BREADTH_CUT,
        "vg_cap": DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP,
        "overheat_cap": DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE,
        "overheat_hi_cap": DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MAX_EXPOSURE,
    }


def load_current_stressbond_runtime(
    years: int,
    *,
    proxy_kind: str = DEFAULT_REGIME_MIX_PROXY_KIND,
) -> dict[str, object]:
    """统一装配当前 stress-bond 细化链的运行时上下文。"""
    current_context = load_current_stressbond_context()
    selected = load_default_strategy_backtest_pool()
    prices = load_recent_selected_prices(selected, years)
    proxy = load_named_market_proxy(prices, years=years, proxy_kind=proxy_kind)
    return {
        "current_context": current_context,
        "treasury_row": lookup_overlay_asset(str(current_context["treasury_code"])),
        "params": build_current_stressbond_params(),
        "selected": selected,
        "prices": prices,
        "proxy": proxy,
    }


def run_current_stressbond_overlay(
    prices: "pd.DataFrame",
    selected: "pd.DataFrame",
    proxy: "pd.DataFrame",
    params: dict[str, float | list[str]],
    treasury_code: str,
    risk_cap: float,
    ratio_cut: float,
    breadth_cut: float,
    fee_rate: float,
    slippage_rate: float,
    short_ratio_cut: float | None = None,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """运行当前 stress-bond 历史搜索链的单层弱市切债变体。"""
    return apply_market_stress_treasury_overlay(
        prices,
        selected,
        proxy,
        params,
        treasury_code=treasury_code,
        risk_cap=risk_cap,
        ratio_cut=ratio_cut,
        breadth_cut=breadth_cut,
        short_ratio_cut=short_ratio_cut,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )


def run_current_stressbond_persistence_overlay(
    prices: "pd.DataFrame",
    selected: "pd.DataFrame",
    proxy: "pd.DataFrame",
    params: dict[str, float | list[str]],
    treasury_code: str,
    risk_cap: float,
    ratio_cut: float,
    breadth_cut: float,
    enter_days: int,
    exit_days: int,
    fee_rate: float,
    slippage_rate: float,
    short_ratio_cut: float | None = None,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """运行当前 stress-bond 历史搜索链的持续状态机切债变体。"""
    base_weights, mixed_momentum, _ = build_base_target_weights(prices, proxy, params)
    risk_codes = [code for code in CURRENT_STRESSBOND_RISK_CODES if code in prices.columns]
    overlaid_weights = apply_persistent_market_stress_treasury_cap(
        base_weights,
        proxy,
        index=prices.index,
        risk_budget_codes=risk_codes,
        treasury_code=treasury_code,
        ratio_cut=ratio_cut,
        breadth_cut=breadth_cut,
        enter_days=enter_days,
        exit_days=exit_days,
        risk_cap=risk_cap,
        persistence_builder=build_persistent_trigger_mask,
        short_ratio_cut=short_ratio_cut,
    )
    return run_target_weights_strategy(prices, selected, overlaid_weights, mixed_momentum, fee_rate, slippage_rate)


def append_current_stressbond_baseline(
    rows: list[dict[str, object]],
    descriptions: dict[str, str],
    nav_compare: "pd.DataFrame",
    *,
    baseline_result: "pd.DataFrame",
    baseline_trades: "pd.DataFrame",
    selected: "pd.DataFrame",
    current_context: dict[str, object],
) -> tuple["pd.DataFrame", dict[str, object]]:
    """统一追加当前 stress-bond 正式链的 baseline 行。"""
    baseline_result = apply_official_baseline_nav_anchor(baseline_result)
    baseline_summary = append_variant_result(
        rows,
        nav_compare,
        descriptions,
        strategy=str(current_context["strategy"]),
        result=baseline_result,
        summary=summarize(baseline_result, baseline_trades, selected),
        description=str(current_context["description"]),
    )
    return baseline_result, baseline_summary
