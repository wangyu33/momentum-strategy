#!/usr/bin/env python3
"""比较若干动态动量阈值方案。"""

from __future__ import annotations

import argparse

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import pandas as pd

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


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "dynamic_threshold_optimizations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较动态动量阈值方案。")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--start-date", type=str, default=str(DEFAULT_HISTORY_START.date()))
    return parser.parse_args()


def build_variants() -> list[dict[str, object]]:
    return [
        {"name": "baseline_fixed_05", "overrides": {}},
        {
            "name": "weak_step_06_relaxed",
            "overrides": {
                "dynamic_threshold_mode": "weak_proxy_step",
                "dynamic_threshold_amount_20_60_cut": 1.00,
                "dynamic_threshold_amount_5_20_cut": 1.00,
                "dynamic_threshold_breadth_cut": 0.00,
                "dynamic_threshold_weak_value": 0.06,
            },
        },
        {
            "name": "weak_step_06_strict",
            "overrides": {
                "dynamic_threshold_mode": "weak_proxy_step",
                "dynamic_threshold_amount_20_60_cut": 0.98,
                "dynamic_threshold_amount_5_20_cut": 0.98,
                "dynamic_threshold_breadth_cut": -0.01,
                "dynamic_threshold_weak_value": 0.06,
            },
        },
        {
            "name": "weak_tiered_065",
            "overrides": {
                "dynamic_threshold_mode": "weak_proxy_tiered",
                "dynamic_threshold_amount_20_60_cut": 1.00,
                "dynamic_threshold_amount_5_20_cut": 1.00,
                "dynamic_threshold_breadth_cut": 0.00,
                "dynamic_threshold_tier_step": 0.005,
                "dynamic_threshold_max": 0.065,
            },
        },
        {
            "name": "weak_tiered_060",
            "overrides": {
                "dynamic_threshold_mode": "weak_proxy_tiered",
                "dynamic_threshold_amount_20_60_cut": 1.00,
                "dynamic_threshold_amount_5_20_cut": 1.00,
                "dynamic_threshold_breadth_cut": 0.00,
                "dynamic_threshold_tier_step": 0.005,
                "dynamic_threshold_max": 0.060,
            },
        },
        {
            "name": "asym_045_06",
            "overrides": {
                "dynamic_threshold_mode": "asymmetric_band",
                "dynamic_threshold_amount_20_60_cut": 1.00,
                "dynamic_threshold_breadth_cut": 0.00,
                "dynamic_threshold_weak_value": 0.06,
                "dynamic_threshold_strong_amount_20_60_cut": 1.05,
                "dynamic_threshold_strong_breadth_cut": 0.02,
                "dynamic_threshold_strong_value": 0.045,
            },
        },
    ]


def summarize_zone(daily_ret: pd.Series) -> dict[str, float]:
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

    baseline_params = build_default_strategy_params()
    baseline_result, baseline_trades = run_default_strategy_with_params(prices, selected, baseline_params)
    baseline_zone_mask = baseline_result["effective_momentum"].gt(0) & baseline_result["effective_momentum"].le(0.05)

    rows: list[dict[str, object]] = []
    variant_frames: dict[str, pd.DataFrame] = {"baseline_fixed_05": baseline_result}
    variant_trade_counts: dict[str, int] = {"baseline_fixed_05": len(baseline_trades)}

    for variant in build_variants():
        name = str(variant["name"])
        if name == "baseline_fixed_05":
            result = baseline_result
            trades = baseline_trades
        else:
            params = build_default_strategy_params()
            params.update(variant["overrides"])
            result, trades = run_default_strategy_with_params(prices, selected, params)
            variant_frames[name] = result
            variant_trade_counts[name] = len(trades)

        summary = build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)
        common_zone = result.loc[baseline_zone_mask.reindex(result.index).fillna(False), "strategy_return"].dropna()
        zone_summary = summarize_zone(common_zone)
        threshold_series = result.get("absolute_momentum_threshold", pd.Series(0.05, index=result.index, dtype="float64"))

        rows.append(
            {
                "variant": name,
                "dynamic_threshold_mode": variant.get("overrides", {}).get("dynamic_threshold_mode", "fixed"),
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
                "threshold_mean": float(threshold_series.mean()),
                "threshold_min": float(threshold_series.min()),
                "threshold_max": float(threshold_series.max()),
                "threshold_gt_05_days": int((threshold_series > 0.05).sum()),
                "threshold_lt_05_days": int((threshold_series < 0.05).sum()),
                "latest_threshold": float(threshold_series.iloc[-1]),
            }
        )

    summary_df = pd.DataFrame(rows)
    baseline_row = summary_df.loc[summary_df["variant"] == "baseline_fixed_05"].iloc[0]
    for metric in [
        "annualized_return",
        "sharpe_rf0",
        "max_drawdown",
        "max_drawdown_integral",
        "trade_count",
        "baseline_zone_total_return",
        "baseline_zone_max_drawdown",
    ]:
        summary_df[f"{metric}_diff_vs_base"] = summary_df[metric] - float(baseline_row[metric])

    summary_df = summary_df.sort_values(
        ["baseline_zone_total_return_diff_vs_base", "annualized_return_diff_vs_base", "sharpe_rf0_diff_vs_base"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    write_dataframe_csv_atomic(summary_df, OUTPUT_DIR / "summary.csv", index=False)

    for name, frame in variant_frames.items():
        export_cols = [
            "nav",
            "drawdown",
            "signal",
            "holding",
            "exposure",
            "current_momentum",
            "effective_momentum",
            "absolute_momentum_threshold",
        ]
        existing_cols = [col for col in export_cols if col in frame.columns]
        export_df = frame[existing_cols].copy()
        export_df.index.name = "date"
        write_dataframe_csv_atomic(export_df, OUTPUT_DIR / f"{name}_nav.csv", index=True)

    preview_cols = [
        "variant",
        "dynamic_threshold_mode",
        "annualized_return",
        "sharpe_rf0",
        "max_drawdown",
        "baseline_zone_total_return",
        "baseline_zone_max_drawdown",
        "threshold_mean",
        "threshold_max",
        "threshold_gt_05_days",
        "annualized_return_diff_vs_base",
        "baseline_zone_total_return_diff_vs_base",
    ]
    print(summary_df[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
