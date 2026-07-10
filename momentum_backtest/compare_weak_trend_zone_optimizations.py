#!/usr/bin/env python3
"""专项比较弱趋势区间的阈值/平滑过渡方案。"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import pandas as pd

import official_strategy_core as official_strategy_core_module
import run_backtest as run_backtest_module

from run_backtest import (
    CORE_OUTPUT_DIR,
    DEFAULT_HISTORY_START,
    DEFAULT_YEARS,
    RESEARCH_OUTPUT_DIR,
    build_default_strategy_params,
    build_strategy_summary,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    run_default_strategy_with_params,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "weak_trend_zone_optimizations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较弱趋势 0~5% 区间的若干优化方案。")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--start-date", type=str, default=str(DEFAULT_HISTORY_START.date()))
    return parser.parse_args()


def build_variants() -> list[dict[str, object]]:
    return [
        {"name": "baseline_step_05", "absolute_threshold": 0.05, "overrides": {}},
        {"name": "step_04", "absolute_threshold": 0.04, "overrides": {"regime_transition_end_cut": 0.04}},
        {"name": "step_06", "absolute_threshold": 0.06, "overrides": {"regime_transition_end_cut": 0.06}},
        {"name": "step_07", "absolute_threshold": 0.07, "overrides": {"regime_transition_end_cut": 0.07}},
        {
            "name": "continuous_00_05",
            "absolute_threshold": 0.05,
            "overrides": {
                "regime_transition_mode": "continuous",
                "regime_transition_start_cut": 0.00,
                "regime_transition_end_cut": 0.05,
            },
        },
        {
            "name": "continuous_01_05",
            "absolute_threshold": 0.05,
            "overrides": {
                "regime_transition_mode": "continuous",
                "regime_transition_start_cut": 0.01,
                "regime_transition_end_cut": 0.05,
            },
        },
        {
            "name": "continuous_00_06",
            "absolute_threshold": 0.06,
            "overrides": {
                "regime_transition_mode": "continuous",
                "regime_transition_start_cut": 0.00,
                "regime_transition_end_cut": 0.06,
            },
        },
        {
            "name": "continuous_01_06",
            "absolute_threshold": 0.06,
            "overrides": {
                "regime_transition_mode": "continuous",
                "regime_transition_start_cut": 0.01,
                "regime_transition_end_cut": 0.06,
            },
        },
        {
            "name": "continuous_02_06",
            "absolute_threshold": 0.06,
            "overrides": {
                "regime_transition_mode": "continuous",
                "regime_transition_start_cut": 0.02,
                "regime_transition_end_cut": 0.06,
            },
        },
    ]


def summarize_zone(nav: pd.Series, daily_ret: pd.Series) -> dict[str, float]:
    if daily_ret.empty:
        return {
            "zone_total_return": 0.0,
            "zone_avg_daily_return": 0.0,
            "zone_win_rate": 0.0,
            "zone_max_drawdown": 0.0,
            "zone_days": 0,
        }
    zone_nav = (1 + daily_ret.fillna(0.0)).cumprod()
    zone_nav.iloc[0] = 1.0
    zone_drawdown = zone_nav / zone_nav.cummax() - 1
    return {
        "zone_total_return": float(zone_nav.iloc[-1] - 1),
        "zone_avg_daily_return": float(daily_ret.mean()),
        "zone_win_rate": float((daily_ret > 0).mean()),
        "zone_max_drawdown": float(zone_drawdown.min()),
        "zone_days": int(len(daily_ret)),
    }


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = load_fixed_etf_pool()
    local_prices_file = CORE_OUTPUT_DIR / "prices.csv"
    if local_prices_file.exists():
        prices = pd.read_csv(local_prices_file, parse_dates=["date"]).set_index("date").sort_index()
        prices.columns = [str(col) for col in prices.columns]
    else:
        prices = fetch_histories(selected, years=args.years)
    prices = prices.loc[prices.index >= pd.Timestamp(args.start_date)].copy()

    variants = build_variants()
    results: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    baseline_params = build_default_strategy_params()
    baseline_result, baseline_trades = run_default_strategy_with_params(prices, selected, baseline_params)
    baseline_zone_mask = baseline_result["effective_momentum"].gt(0) & baseline_result["effective_momentum"].le(0.05)

    for variant in variants:
        params = build_default_strategy_params()
        params.update(variant["overrides"])
        absolute_threshold = float(variant.get("absolute_threshold", 0.05))
        previous_run_threshold = run_backtest_module.DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD
        previous_core_threshold = official_strategy_core_module.DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD
        run_backtest_module.DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD = absolute_threshold
        official_strategy_core_module.DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD = absolute_threshold
        try:
            result, trades = run_default_strategy_with_params(prices, selected, params)
        finally:
            run_backtest_module.DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD = previous_run_threshold
            official_strategy_core_module.DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD = previous_core_threshold
        results[str(variant["name"])] = result
        summary = build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)

        common_zone = result.loc[baseline_zone_mask.reindex(result.index).fillna(False), "strategy_return"].dropna()
        zone_summary = summarize_zone(result["nav"], common_zone)
        own_zone_mask = result["effective_momentum"].gt(0) & result["effective_momentum"].le(float(params["regime_transition_end_cut"]))
        own_zone = result.loc[own_zone_mask, "strategy_return"].dropna()
        own_zone_summary = summarize_zone(result["nav"], own_zone)

        rows.append(
            {
                "variant": variant["name"],
                "transition_mode": params["regime_transition_mode"],
                "absolute_threshold": absolute_threshold,
                "transition_start_cut": float(params["regime_transition_start_cut"]),
                "transition_end_cut": float(params["regime_transition_end_cut"]),
                "annualized_return": float(summary["annualized_return"]),
                "sharpe_rf0": float(summary["sharpe_rf0"]),
                "max_drawdown": float(summary["max_drawdown"]),
                "max_drawdown_integral": float(summary["max_drawdown_integral"]),
                "trade_count": int(summary["trade_count"]),
                "baseline_zone_days": zone_summary["zone_days"],
                "baseline_zone_total_return": zone_summary["zone_total_return"],
                "baseline_zone_avg_daily_return": zone_summary["zone_avg_daily_return"],
                "baseline_zone_win_rate": zone_summary["zone_win_rate"],
                "baseline_zone_max_drawdown": zone_summary["zone_max_drawdown"],
                "own_zone_days": own_zone_summary["zone_days"],
                "own_zone_total_return": own_zone_summary["zone_total_return"],
                "own_zone_avg_daily_return": own_zone_summary["zone_avg_daily_return"],
                "own_zone_win_rate": own_zone_summary["zone_win_rate"],
                "own_zone_max_drawdown": own_zone_summary["zone_max_drawdown"],
            }
        )

    summary_df = pd.DataFrame(rows)
    baseline_row = summary_df.loc[summary_df["variant"] == "baseline_step_05"].iloc[0]
    for metric in [
        "annualized_return",
        "sharpe_rf0",
        "max_drawdown",
        "max_drawdown_integral",
        "trade_count",
        "baseline_zone_total_return",
        "baseline_zone_max_drawdown",
        "own_zone_total_return",
        "own_zone_max_drawdown",
    ]:
        summary_df[f"{metric}_diff_vs_base"] = summary_df[metric] - float(baseline_row[metric])

    summary_df = summary_df.sort_values(
        ["baseline_zone_total_return_diff_vs_base", "annualized_return_diff_vs_base", "sharpe_rf0_diff_vs_base"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    write_dataframe_csv_atomic(summary_df, OUTPUT_DIR / "summary.csv", index=False)

    preview_cols = [
        "variant",
        "transition_mode",
        "transition_start_cut",
        "transition_end_cut",
        "annualized_return",
        "sharpe_rf0",
        "max_drawdown",
        "baseline_zone_total_return",
        "baseline_zone_max_drawdown",
        "annualized_return_diff_vs_base",
        "baseline_zone_total_return_diff_vs_base",
    ]
    print(summary_df[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
