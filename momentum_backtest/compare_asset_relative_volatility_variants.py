#!/usr/bin/env python3
"""对比基于“单标的自身历史波动水位”的 3 类动量优化方案。"""

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

from compare_goal_optimizations import MARKET_VOLUME_CACHE_PATH
from run_backtest import (
    DEFAULT_FEE_RATE,
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


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "asset_relative_volatility_variants"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比按单标的自身波动水位进行筛选/降仓的动量策略。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def summarize(name: str, result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float | int | str]:
    latest = result.iloc[-1]
    return {
        "strategy": name,
        **build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True),
        "latest_signal": latest.get("signal"),
        "latest_holding": latest.get("holding"),
        "latest_exposure": latest.get("exposure"),
        "latest_selected_vol_rank": latest.get("selected_volatility_rank"),
    }


def load_cached_market_proxy() -> pd.DataFrame:
    proxy = pd.read_csv(MARKET_VOLUME_CACHE_PATH, parse_dates=["date"])
    return proxy.set_index("date")


def build_variants() -> list[tuple[str, dict[str, object]]]:
    baseline = build_default_strategy_params()

    vol_pct_penalty = build_default_strategy_params()
    vol_pct_penalty.update(
        {
            "signal_volatility_state_lookback": 252,
            "signal_volatility_percentile_penalty": 0.04,
        }
    )

    vol_pct_divisor = build_default_strategy_params()
    vol_pct_divisor.update(
        {
            "signal_volatility_state_lookback": 252,
            "signal_volatility_percentile_divisor": 1.20,
        }
    )

    vol_pct_cap = build_default_strategy_params()
    vol_pct_cap.update(
        {
            "signal_volatility_state_lookback": 252,
            "signal_selected_volatility_cap_start": 0.85,
            "signal_selected_volatility_cap_end": 0.98,
            "signal_selected_volatility_cap_floor": 0.80,
        }
    )

    return [
        ("baseline", baseline),
        ("own_vol_pct_penalty", vol_pct_penalty),
        ("own_vol_pct_divisor", vol_pct_divisor),
        ("own_vol_pct_cap", vol_pct_cap),
    ]


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected, prices = load_core_selected_and_prices()
    market_proxy = load_cached_market_proxy()
    benchmark_nav = build_benchmark_nav(prices, benchmark_code="510300")

    variants = build_variants()
    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    compare_df = pd.DataFrame(index=prices.index)
    compare_df["hs300_benchmark_nav"] = benchmark_nav.reindex(prices.index)

    for strategy_name, params in variants:
        result, trades = run_default_strategy_with_params(
            prices,
            selected,
            params=params,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            market_proxy=market_proxy,
        )
        summary_rows.append(summarize(strategy_name, result, trades, selected))
        compare_df[f"{strategy_name}_nav"] = result["nav"]

        compare = pd.concat([result["nav"], benchmark_nav.reindex(result.index)], axis=1)
        compare.columns = ["strategy_nav", "benchmark_nav"]
        for row in build_yearly_return_rows(
            compare,
            strategy_col="strategy_nav",
            benchmark_col="benchmark_nav",
            benchmark_return_col="hs300_return",
        ):
            yearly_rows.append({"strategy": strategy_name, **row})

        write_dataframe_csv_atomic(trades, OUTPUT_DIR / f"{strategy_name}_trades.csv", index=False)
        write_dataframe_csv_atomic(result.reset_index(), OUTPUT_DIR / f"{strategy_name}_nav_detail.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    baseline_row = summary.loc[summary["strategy"] == "baseline"].iloc[0]
    for col in ["total_return", "annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral"]:
        summary[f"{col}_diff_vs_baseline"] = summary[col] - float(baseline_row[col])
    summary["trade_count_diff_vs_baseline"] = summary["trade_count"] - int(baseline_row["trade_count"])
    summary["trade_action_count_diff_vs_baseline"] = summary["trade_action_count"] - int(baseline_row["trade_action_count"])
    summary["is_soft_improvement"] = (
        (summary["annualized_return"] >= float(baseline_row["annualized_return"]))
        & (summary["max_drawdown_integral"] <= float(baseline_row["max_drawdown_integral"]))
    )
    summary = summary.sort_values(
        ["is_soft_improvement", "annualized_return", "max_drawdown_integral", "max_drawdown"],
        ascending=[False, False, True, True],
    )

    write_dataframe_csv_atomic(summary, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, OUTPUT_DIR / "nav_compare.csv")
    write_dataframe_csv_atomic(pd.DataFrame(yearly_rows), OUTPUT_DIR / "yearly_returns.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    for col in [c for c in compare_df.columns if c.endswith("_nav") and c != "hs300_benchmark_nav"]:
        ax.plot(compare_df.index, compare_df[col], label=col.replace("_nav", ""), linewidth=1.6)
    ax.plot(compare_df.index, compare_df["hs300_benchmark_nav"], label="hs300_benchmark", linewidth=1.3, alpha=0.8, linestyle="--")
    ax.set_title("Asset-Relative Volatility Variants", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "nav_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
