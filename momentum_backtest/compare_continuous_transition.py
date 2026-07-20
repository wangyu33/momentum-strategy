#!/usr/bin/env python3
"""对比正式基线与连续进攻/防守过渡方案。"""

from __future__ import annotations

import argparse

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

try:
    from .official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_LOOKBACK,
    DEFAULT_SLIPPAGE_RATE,
    RESEARCH_OUTPUT_DIR,
    build_benchmark_nav,
    build_default_strategy_params,
    build_strategy_summary,
    build_yearly_return_rows,
    ensure_output_dirs,
    load_core_selected_and_prices,
    run_default_strategy_with_params,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "continuous_transition"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
NAV_COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
YEARLY_PATH = OUTPUT_DIR / "yearly_returns.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比正式基线与连续进攻/防守过渡方案。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--transition-start", type=float, default=0.0, help="连续过渡起点动量阈值。")
    parser.add_argument("--transition-end", type=float, default=0.05, help="连续过渡终点动量阈值。")
    return parser.parse_args()


def summarize(name: str, result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float | int | str]:
    return {"strategy": name, **build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)}


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected, prices = load_core_selected_and_prices()
    benchmark_nav = build_benchmark_nav(prices, benchmark_code="510300")

    baseline_params = build_default_strategy_params()
    continuous_params = build_default_strategy_params()
    continuous_params.update(
        {
            "regime_transition_mode": "continuous",
            "regime_transition_start_cut": float(args.transition_start),
            "regime_transition_end_cut": float(args.transition_end),
        }
    )

    baseline_result, baseline_trades = run_default_strategy_with_params(
        prices,
        selected,
        params=baseline_params,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_result = apply_official_baseline_nav_anchor(baseline_result)
    continuous_result, continuous_trades = run_default_strategy_with_params(
        prices,
        selected,
        params=continuous_params,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )

    summary = pd.DataFrame(
        [
            summarize("baseline_step", baseline_result, baseline_trades, selected),
            summarize("continuous_transition", continuous_result, continuous_trades, selected),
        ]
    )
    baseline_row = summary[summary["strategy"] == "baseline_step"].iloc[0]
    for col in ["total_return", "annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral"]:
        summary[f"{col}_diff_vs_baseline"] = summary[col] - float(baseline_row[col])
    summary["trade_count_diff_vs_baseline"] = summary["trade_count"] - int(baseline_row["trade_count"])
    summary["trade_action_count_diff_vs_baseline"] = summary["trade_action_count"] - int(baseline_row["trade_action_count"])

    compare_df = pd.DataFrame(index=prices.index)
    compare_df["baseline_step_nav"] = baseline_result["nav"]
    compare_df["continuous_transition_nav"] = continuous_result["nav"]
    compare_df["hs300_benchmark_nav"] = benchmark_nav.reindex(prices.index)
    compare_df.index.name = "date"

    yearly_rows = []
    for strategy_name, result in [
        ("baseline_step", baseline_result),
        ("continuous_transition", continuous_result),
    ]:
        compare = pd.concat([result["nav"], benchmark_nav.reindex(result.index)], axis=1)
        compare.columns = ["strategy_nav", "benchmark_nav"]
        for row in build_yearly_return_rows(
            compare,
            strategy_col="strategy_nav",
            benchmark_col="benchmark_nav",
            benchmark_return_col="hs300_return",
        ):
            yearly_rows.append({"strategy": strategy_name, **row})

    write_dataframe_csv_atomic(summary, SUMMARY_PATH, index=False)
    write_dataframe_csv_atomic(compare_df, NAV_COMPARE_PATH)
    write_dataframe_csv_atomic(pd.DataFrame(yearly_rows), YEARLY_PATH, index=False)
    write_dataframe_csv_atomic(baseline_trades, OUTPUT_DIR / "baseline_trades.csv", index=False)
    write_dataframe_csv_atomic(continuous_trades, OUTPUT_DIR / "continuous_trades.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["baseline_step_nav"], label="Baseline Step", linewidth=2.0)
    ax.plot(compare_df.index, compare_df["continuous_transition_nav"], label="Continuous Transition", linewidth=2.0)
    ax.plot(compare_df.index, compare_df["hs300_benchmark_nav"], label="HS300 Benchmark", linewidth=1.6, alpha=0.8)
    ax.set_title("Baseline vs Continuous Transition", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "nav_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
