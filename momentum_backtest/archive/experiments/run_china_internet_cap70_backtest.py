#!/usr/bin/env python3
"""运行中概互联 70% 仓位上限分支，并输出完整结果。"""

from __future__ import annotations

import argparse
import math
try:
    from ..runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

from compare_candidate_pool_additions import ETF_513050
from compare_china_internet_guards import run_variant, summarize
from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RESEARCH_OUTPUT_DIR,
    annualized_return,
    build_benchmark_nav,
    build_yearly_return_rows,
    ensure_output_dirs,
    load_fixed_etf_pool,
    fetch_histories,
    max_drawdown,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "archive_flat" / "run_china_internet_cap70_backtest"
STRATEGY_NAME = "china_internet_cap70"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行中概互联 70% 仓位上限分支。")
    parser.add_argument("--years", type=int, default=6, help="向前抓取多少年历史数据，再对齐公共区间。")
    parser.add_argument("--lookback", type=int, default=25, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def ensure_output_dir() -> None:
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_yearly_returns(result: pd.DataFrame, benchmark_nav: pd.Series) -> None:
    compare = pd.concat([result["nav"], benchmark_nav], axis=1)
    compare.columns = ["strategy_nav", "benchmark_nav"]
    rows = build_yearly_return_rows(
        compare,
        strategy_col="strategy_nav",
        benchmark_col="benchmark_nav",
        benchmark_return_col="hs300_etf_return",
    )
    write_dataframe_csv_atomic(pd.DataFrame(rows), OUTPUT_DIR / "yearly_returns.csv", index=False)


def save_outputs(selected: pd.DataFrame, prices: pd.DataFrame, result: pd.DataFrame, trades: pd.DataFrame, lookback: int) -> None:
    ensure_output_dir()
    prices_to_save = prices.copy()
    prices_to_save.index.name = "date"
    result_to_save = result.copy()
    result_to_save.index.name = "date"
    write_dataframe_csv_atomic(selected, OUTPUT_DIR / "selected_etfs.csv", index=False)
    write_dataframe_csv_atomic(prices_to_save, OUTPUT_DIR / "prices.csv")
    write_dataframe_csv_atomic(result_to_save, OUTPUT_DIR / "backtest_nav.csv")
    write_dataframe_csv_atomic(trades, OUTPUT_DIR / "trades.csv", index=False)
    write_dataframe_csv_atomic(result_to_save[["nav"]].rename(columns={"nav": "historical_nav"}), OUTPUT_DIR / "historical_nav.csv")

    benchmark_nav = build_benchmark_nav(prices, benchmark_code="510300")
    compare_df = pd.concat([result["nav"], benchmark_nav], axis=1)
    write_dataframe_csv_atomic(compare_df, OUTPUT_DIR / "strategy_vs_hs300.csv")
    save_yearly_returns(result, benchmark_nav)

    buy_trades = trades[trades["action"] == "BUY"]
    sell_trades = trades[trades["action"] == "SELL"]

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(result.index, result["nav"], color="#2b6cb0", linewidth=2.2, label="Strategy NAV")
    if not buy_trades.empty:
        ax.scatter(buy_trades["date"], buy_trades["nav"], color="#1f9d55", marker="^", s=55, label="Buy", zorder=3)
    if not sell_trades.empty:
        ax.scatter(sell_trades["date"], sell_trades["nav"], color="#d64545", marker="v", s=55, label="Sell", zorder=3)
    ax.set_title(f"China Internet Cap70 NAV (lookback={lookback})", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "nav_with_trades.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result.index, result["current_momentum"], color="#dd6b20", linewidth=2.0)
    ax.axhline(0.0, color="#4a5568", linestyle="--", linewidth=1.0)
    ax.set_title(f"Signal Momentum (trailing {lookback} trading days)", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Momentum")
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "momentum_series.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    drawdown = result["drawdown"]
    max_dd_date = drawdown.idxmin()
    max_dd_value = float(drawdown.loc[max_dd_date])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(drawdown.index, drawdown, color="#d64545", linewidth=1.8)
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#f3b0b0", alpha=0.6)
    ax.scatter([max_dd_date], [max_dd_value], color="#9b2c2c", s=70, zorder=3)
    ax.annotate(
        f"Max DD {max_dd_value:.2%}\n{max_dd_date.date()}",
        xy=(max_dd_date, max_dd_value),
        xytext=(12, -8),
        textcoords="offset points",
        fontsize=10,
        color="#742a2a",
        ha="left",
        va="top",
    )
    ax.axhline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_title("Historical Drawdown vs Running Peak", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "max_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["nav"], color="#2b6cb0", linewidth=2.2, label="Strategy NAV")
    ax.plot(compare_df.index, compare_df["benchmark_nav"], color="#d64545", linewidth=2.0, label="HS300 ETF")
    ax.set_title("Strategy vs HS300 ETF", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "strategy_vs_hs300.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_summary(result: pd.DataFrame, trades: pd.DataFrame) -> None:
    summary = summarize(result, trades)
    summary.update(
        {
            "strategy": STRATEGY_NAME,
            "annualized_return_full": annualized_return(result["nav"]),
            "max_drawdown_full": max_drawdown(result["nav"]),
        }
    )
    for key, value in summary.items():
        if isinstance(value, float) and math.isfinite(value):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def main() -> int:
    args = parse_args()
    base_pool = load_fixed_etf_pool()
    selected = pd.concat([base_pool, pd.DataFrame([ETF_513050])], ignore_index=True)
    prices = fetch_histories(selected, years=args.years).dropna(how="any")
    result, trades = run_variant(
        prices=prices,
        selected=selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        china_max_exposure=0.70,
    )
    save_outputs(selected, prices, result, trades, lookback=args.lookback)
    print_summary(result, trades)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
