#!/usr/bin/env python3
"""对比不同 effective_momentum 定义在当前正式基线中的效果。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MOMENTUM_DIR = Path(__file__).resolve().parents[2]
if str(MOMENTUM_DIR) not in sys.path:
    sys.path.insert(0, str(MOMENTUM_DIR))

try:
    from ...runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from archive_data_loaders import build_archive_flat_output_dir, load_named_market_proxy, load_recent_selected_prices
from overlay_strategy_helpers import apply_persistent_market_stress_treasury_cap, apply_risk_cap
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    build_typed_summary_fields,
    finalize_baseline_diff_summary,
    save_plot_and_print_variant_compare_outputs,
)
try:
    from ...official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

from defensive_persistence_common import build_persistent_mask
from goal_optimization_common import build_dynamic_core_target_weights
from hs300_regime_common import run_target_weights_strategy, summarize
from archive_strategy_common import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_LOOKBACK,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_STRESS_BOND_BREADTH_CUT,
    DEFAULT_STRESS_BOND_CODE,
    DEFAULT_STRESS_BOND_ENTER_DAYS,
    DEFAULT_STRESS_BOND_EXIT_DAYS,
    DEFAULT_STRESS_BOND_RISK_CAP,
    DEFAULT_STRESS_BOND_RATIO_CUT,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    RISK_CODES,
    build_default_strategy_params,
    load_default_strategy_backtest_pool,
    resolve_strategy_universe,
)


OUTPUT_DIR = build_archive_flat_output_dir("compare_effective_momentum_definitions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比不同 effective_momentum 定义在当前正式基线中的效果。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def weighted_heat(
    weights: pd.DataFrame,
    momentum: pd.DataFrame,
    include_codes: list[str],
) -> pd.Series:
    valid_codes = [code for code in include_codes if code in weights.columns and code in momentum.columns]
    if not valid_codes:
        return pd.Series(float("nan"), index=weights.index, dtype="float64")
    numerator = (weights[valid_codes] * momentum[valid_codes]).sum(axis=1, min_count=1)
    # 这里必须保持纯数值 dtype，不能混入 pd.NA，否则后续 astype("float64") 会报 NAType 错误。
    denominator = weights[valid_codes].sum(axis=1).replace(0.0, float("nan")).astype("float64")
    return numerator.div(denominator).astype("float64")


def build_target_weights_with_heat(
    prices: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, object],
    *,
    heat_mode: str,
) -> tuple[pd.DataFrame, pd.Series]:
    active_risk_codes, active_defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
        defensive_codes=[str(code) for code in params.get("defensive_codes", [])],
    )
    aggressive_weights, aggressive_momentum, _, _, aggressive_trend = build_dynamic_core_target_weights(
        prices,
        lookback=DEFAULT_LOOKBACK,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        core_weight=float(params["aggressive_core_weight"]),
        hs300_mom60_cut=0.05,
        hs300_mom120_cut=0.10,
        hs300_ma_window=90,
        signal_quality_method=str(params.get("signal_quality_method", "raw")),
        slope_penalty=float(params.get("signal_slope_penalty", 0.0)),
        leader_margin=float(params.get("signal_leader_margin", 0.0)),
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )
    conservative_weights, conservative_momentum, _, _, _ = build_dynamic_core_target_weights(
        prices,
        lookback=DEFAULT_LOOKBACK,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        core_weight=float(params["conservative_core_weight"]),
        hs300_mom60_cut=0.04,
        hs300_mom120_cut=0.10,
        hs300_ma_window=120,
        signal_quality_method=str(params.get("signal_quality_method", "raw")),
        slope_penalty=float(params.get("signal_slope_penalty", 0.0)),
        leader_margin=float(params.get("signal_leader_margin", 0.0)),
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )
    aggressive_mask = aggressive_trend & (aggressive_momentum >= float(params["regime_momentum_cut"]))
    mixed_target_weights = conservative_weights.copy()
    mixed_target_weights.loc[aggressive_mask] = aggressive_weights.loc[aggressive_mask]
    legacy_heat = conservative_momentum.copy()
    legacy_heat.loc[aggressive_mask] = aggressive_momentum.loc[aggressive_mask]
    legacy_heat = legacy_heat.rename("effective_momentum")

    raw_momentum = prices / prices.shift(DEFAULT_LOOKBACK) - 1
    if heat_mode == "legacy":
        weak_guard_heat = legacy_heat.copy()
    elif heat_mode == "risk_weighted":
        weak_guard_heat = weighted_heat(mixed_target_weights, raw_momentum, active_risk_codes).rename("effective_momentum")
    elif heat_mode == "portfolio_weighted":
        weak_guard_heat = weighted_heat(mixed_target_weights, raw_momentum, list(prices.columns)).rename("effective_momentum")
    else:
        raise ValueError(f"unsupported heat_mode: {heat_mode}")

    proxy = proxy.reindex(prices.index).ffill()
    row_risk_weight = mixed_target_weights[active_risk_codes].sum(axis=1)
    volume_weak_mask = (
        (proxy["market_amount_ratio_20_60"] < float(params["volume_ratio_cut"]))
        & (proxy["market_amount_ratio_5_20"] < float(params["volume_short_ratio_cut"]))
        & (proxy["market_breadth_proxy"] < float(params["volume_breadth_cut"]))
        & (weak_guard_heat <= float(params["volume_guard_momentum_ceiling"]))
    ).fillna(False)

    capped_target_weights = apply_risk_cap(
        mixed_target_weights,
        risk_budget_codes=active_risk_codes,
        trigger_mask=volume_weak_mask,
        risk_cap=float(params["volume_guard_cap"]),
    )

    if heat_mode == "legacy":
        overheat_heat = legacy_heat.copy()
    elif heat_mode == "risk_weighted":
        overheat_heat = weighted_heat(capped_target_weights, raw_momentum, active_risk_codes).rename("effective_momentum")
    else:
        overheat_heat = weighted_heat(capped_target_weights, raw_momentum, list(prices.columns)).rename("effective_momentum")

    base_result, _ = run_target_weights_strategy(
        prices,
        pd.DataFrame({"code": list(prices.columns), "theme": list(prices.columns), "name": list(prices.columns)}),
        capped_target_weights,
        overheat_heat,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
    )
    risk_weight_after_weak_guard = capped_target_weights[active_risk_codes].sum(axis=1)

    overheat_mask = (
        (risk_weight_after_weak_guard > float(params["overheat_max_exposure"]))
        & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
        & (overheat_heat >= float(params["overheat_momentum_cut"]))
    ).fillna(False)
    capped_target_weights = apply_risk_cap(
        capped_target_weights,
        risk_budget_codes=active_risk_codes,
        trigger_mask=overheat_mask,
        risk_cap=float(params["overheat_max_exposure"]),
    )

    overheat_high_mask = (
        (capped_target_weights[active_risk_codes].sum(axis=1) > float(params["overheat_high_max_exposure"]))
        & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
        & (overheat_heat >= float(params["overheat_high_momentum_cut"]))
    ).fillna(False)
    capped_target_weights = apply_risk_cap(
        capped_target_weights,
        risk_budget_codes=active_risk_codes,
        trigger_mask=overheat_high_mask,
        risk_cap=float(params["overheat_high_max_exposure"]),
    )

    if heat_mode == "legacy":
        final_heat = legacy_heat.copy()
    elif heat_mode == "risk_weighted":
        final_heat = weighted_heat(capped_target_weights, raw_momentum, active_risk_codes).rename("effective_momentum")
    else:
        final_heat = weighted_heat(capped_target_weights, raw_momentum, list(prices.columns)).rename("effective_momentum")
    return capped_target_weights, final_heat


def apply_stress_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    base_weights: pd.DataFrame,
    effective_heat: pd.Series,
    proxy: pd.DataFrame,
    *,
    treasury_code: str,
    risk_cap: float,
    ratio_cut: float,
    breadth_cut: float,
    enter_days: int,
    exit_days: int,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configured_risk_codes, _ = resolve_strategy_universe(prices)
    risk_budget_codes = list(configured_risk_codes)
    if "510300" in prices.columns and "510300" not in risk_budget_codes:
        risk_budget_codes.append("510300")

    overlaid_weights = apply_persistent_market_stress_treasury_cap(
        base_weights,
        proxy,
        index=prices.index,
        risk_budget_codes=risk_budget_codes,
        treasury_code=treasury_code,
        ratio_cut=ratio_cut,
        breadth_cut=breadth_cut,
        enter_days=enter_days,
        exit_days=exit_days,
        risk_cap=risk_cap,
        persistence_builder=build_persistent_mask,
    )
    result, trades = run_target_weights_strategy(prices, selected, overlaid_weights, effective_heat, fee_rate, slippage_rate)
    return result, trades


def main() -> int:
    args = parse_args()
    selected = load_default_strategy_backtest_pool()
    prices = load_recent_selected_prices(selected, args.years)
    params = build_default_strategy_params()
    proxy = load_named_market_proxy(
        prices,
        years=args.years,
        proxy_kind=str(params["proxy_kind"]),
        risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
    )

    rows: list[dict[str, object]] = []
    nav_compare = build_compare_frame(prices)

    for heat_mode in ["legacy", "risk_weighted", "portfolio_weighted"]:
        base_weights, effective_heat = build_target_weights_with_heat(prices, proxy, params, heat_mode=heat_mode)
        result, trades = apply_stress_overlay(
            prices,
            selected,
            base_weights,
            effective_heat,
            proxy,
            treasury_code=DEFAULT_STRESS_BOND_CODE,
            risk_cap=DEFAULT_STRESS_BOND_RISK_CAP,
            ratio_cut=DEFAULT_STRESS_BOND_RATIO_CUT,
            breadth_cut=DEFAULT_STRESS_BOND_BREADTH_CUT,
            enter_days=DEFAULT_STRESS_BOND_ENTER_DAYS,
            exit_days=DEFAULT_STRESS_BOND_EXIT_DAYS,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        if heat_mode == "legacy":
            result = apply_official_baseline_nav_anchor(result)
        summary = summarize(result, trades, selected)
        append_variant_result(
            rows,
            nav_compare,
            None,
            strategy=heat_mode,
            result=result,
            summary=build_typed_summary_fields(
                summary,
                float_fields=("annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral"),
                int_fields=("trade_action_count",),
                str_fields=("latest_portfolio",),
            ),
            strategy_field="heat_mode",
            nav_column=f"{heat_mode}_nav",
        )

    summary_df = finalize_baseline_diff_summary(
        rows,
        baseline_field="heat_mode",
        baseline_value="legacy",
        metric_mappings=(
            ("annualized_return", "annualized_diff"),
            ("sharpe_rf0", "sharpe_diff"),
            ("max_drawdown", "max_drawdown_diff"),
            ("max_drawdown_integral", "max_drawdown_integral_diff"),
        ),
    )
    save_plot_and_print_variant_compare_outputs(
        OUTPUT_DIR,
        summary_df,
        nav_compare,
        plot_filename="comparison.png",
        title="Effective Momentum Definition Comparison",
        lines=(
            ("legacy_nav", "Legacy", 2.2),
            ("risk_weighted_nav", "Risk Weighted", 1.8),
            ("portfolio_weighted_nav", "Portfolio Weighted", 1.8),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
