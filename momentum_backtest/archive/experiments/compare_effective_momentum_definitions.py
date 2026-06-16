#!/usr/bin/env python3
"""对比不同 effective_momentum 定义在当前正式基线中的效果。"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ..runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from compare_defensive_persistence import build_persistent_mask
from compare_goal_optimizations import build_dynamic_core_target_weights, load_market_volume_proxy
from compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
from compare_market_proxy_variants import build_proxy_catalog
from run_backtest import (
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
    ensure_output_dirs,
    load_default_strategy_backtest_pool,
    resolve_strategy_universe,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_effective_momentum_definitions")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"


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

    capped_target_weights = mixed_target_weights.copy()
    weak_cap = float(params["volume_guard_cap"])
    weak_scale_mask = volume_weak_mask & (row_risk_weight > weak_cap)
    if weak_scale_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[weak_scale_mask] = weak_cap / row_risk_weight.loc[weak_scale_mask]
        capped_target_weights.loc[weak_scale_mask, active_risk_codes] = capped_target_weights.loc[
            weak_scale_mask, active_risk_codes
        ].mul(scale.loc[weak_scale_mask], axis=0)

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
    if overheat_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[overheat_mask] = float(params["overheat_max_exposure"]) / risk_weight_after_weak_guard.loc[overheat_mask]
        capped_target_weights.loc[overheat_mask, active_risk_codes] = capped_target_weights.loc[
            overheat_mask, active_risk_codes
        ].mul(scale.loc[overheat_mask], axis=0)

    overheat_high_mask = (
        (capped_target_weights[active_risk_codes].sum(axis=1) > float(params["overheat_high_max_exposure"]))
        & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
        & (overheat_heat >= float(params["overheat_high_momentum_cut"]))
    ).fillna(False)
    if overheat_high_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[overheat_high_mask] = (
            float(params["overheat_high_max_exposure"])
            / capped_target_weights[active_risk_codes].sum(axis=1).loc[overheat_high_mask]
        )
        capped_target_weights.loc[overheat_high_mask, active_risk_codes] = capped_target_weights.loc[
            overheat_high_mask, active_risk_codes
        ].mul(scale.loc[overheat_high_mask], axis=0)

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

    overlaid_weights = base_weights.copy()
    proxy = proxy.reindex(prices.index).ffill()
    row_risk_weight = base_weights[risk_budget_codes].sum(axis=1)
    raw_trigger = (
        (proxy["market_amount_ratio_20_60"] < ratio_cut)
        & (proxy["market_breadth_proxy"] < breadth_cut)
    ).fillna(False)
    trigger_mask = build_persistent_mask(raw_trigger, enter_days=enter_days, exit_days=exit_days)
    scale_mask = trigger_mask & (row_risk_weight > risk_cap)
    if scale_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[scale_mask] = risk_cap / row_risk_weight.loc[scale_mask]
        overlaid_weights.loc[scale_mask, risk_budget_codes] = overlaid_weights.loc[scale_mask, risk_budget_codes].mul(
            scale.loc[scale_mask], axis=0
        )
        moved_weight = row_risk_weight.loc[scale_mask] - risk_cap
        overlaid_weights.loc[scale_mask, treasury_code] = overlaid_weights.loc[scale_mask, treasury_code].add(
            moved_weight,
            fill_value=0.0,
        )
    result, trades = run_target_weights_strategy(prices, selected, overlaid_weights, effective_heat, fee_rate, slippage_rate)
    return result, trades


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = load_default_strategy_backtest_pool()
    _, cached_prices = load_cached_data()
    start_ts = cached_prices.index.max() - pd.DateOffset(years=args.years)
    prices = cached_prices.loc[cached_prices.index >= start_ts].copy()
    selected_codes = selected["code"].astype(str).tolist()
    prices = prices[[code for code in selected_codes if code in prices.columns]].copy()

    base_proxy = load_market_volume_proxy(years=args.years, refresh=False)
    params = build_default_strategy_params()
    proxy_catalog = {
        item["name"]: item["proxy"]
        for item in build_proxy_catalog(
            base_proxy,
            prices,
            risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
        )
    }
    proxy = proxy_catalog[str(params["proxy_kind"])]

    rows: list[dict[str, object]] = []
    baseline_row: dict[str, float] | None = None

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
        summary = summarize(result, trades, selected)
        row = {
            "heat_mode": heat_mode,
            "annualized_return": float(summary["annualized_return"]),
            "sharpe_rf0": float(summary["sharpe_rf0"]),
            "max_drawdown": float(summary["max_drawdown"]),
            "max_drawdown_integral": float(summary["max_drawdown_integral"]),
            "trade_action_count": int(summary["trade_action_count"]),
            "latest_portfolio": str(summary["latest_portfolio"]),
        }
        if baseline_row is None:
            baseline_row = row.copy()
            row["annualized_diff"] = 0.0
            row["sharpe_diff"] = 0.0
            row["max_drawdown_diff"] = 0.0
            row["max_drawdown_integral_diff"] = 0.0
        else:
            row["annualized_diff"] = row["annualized_return"] - baseline_row["annualized_return"]
            row["sharpe_diff"] = row["sharpe_rf0"] - baseline_row["sharpe_rf0"]
            row["max_drawdown_diff"] = row["max_drawdown"] - baseline_row["max_drawdown"]
            row["max_drawdown_integral_diff"] = row["max_drawdown_integral"] - baseline_row["max_drawdown_integral"]
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    write_dataframe_csv_atomic(summary_df, SUMMARY_PATH, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nsummary saved to {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
