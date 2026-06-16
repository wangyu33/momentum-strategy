#!/usr/bin/env python3
"""导出指定 leader_margin 的研究图表，格式尽量贴近 core 输出。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MOMENTUM_DIR = Path(__file__).resolve().parents[2]
if str(MOMENTUM_DIR) not in sys.path:
    sys.path.insert(0, str(MOMENTUM_DIR))

from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from run_backtest import (
    RESEARCH_OUTPUT_DIR,
    build_benchmark_nav,
    build_contribution_summary,
    build_default_strategy_params,
    build_strategy_summary,
    build_yearly_return_rows,
    configure_matplotlib,
    ensure_output_dirs,
    load_core_selected_and_prices,
    load_default_strategy_backtest_pool,
    save_figure_atomic,
    write_dataframe_csv_atomic,
)
from compare_goal_optimizations import load_market_volume_proxy
from run_backtest import run_default_strategy_with_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出指定 leader_margin 的研究图表。")
    parser.add_argument("--leader-margin", type=float, default=0.01, help="最小领先优势阈值。默认 1%%。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    return parser.parse_args()


def build_output_dir(leader_margin: float) -> Path:
    suffix = f"leader_margin_{int(round(leader_margin * 1000)):03d}"
    return RESEARCH_OUTPUT_DIR / "archive_flat" / f"export_{suffix}"


def save_nav_with_trades(result: pd.DataFrame, trades: pd.DataFrame, output_dir: Path, title_suffix: str) -> None:
    buy_trades = trades[trades["action"] == "BUY"]
    sell_trades = trades[trades["action"] == "SELL"]

    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(result.index, result["nav"], color="#2b6cb0", linewidth=2.2, label="Strategy NAV")
    if not buy_trades.empty:
        ax.scatter(buy_trades["date"], buy_trades["nav"], color="#1f9d55", marker="^", s=55, label="Buy", zorder=3)
    if not sell_trades.empty:
        ax.scatter(sell_trades["date"], sell_trades["nav"], color="#d64545", marker="v", s=55, label="Sell", zorder=3)
    ax.set_title(f"ETF Strategy NAV ({title_suffix})", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, output_dir / "nav_with_trades.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_drawdown_chart(result: pd.DataFrame, output_dir: Path) -> None:
    drawdown = result["drawdown"]
    max_dd_date = drawdown.idxmin()
    max_dd_value = float(drawdown.loc[max_dd_date])

    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()
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
    save_figure_atomic(fig, output_dir / "max_drawdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_compare_chart(prices: pd.DataFrame, result: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    benchmark_nav = build_benchmark_nav(prices, benchmark_code="510300")
    compare_df = pd.concat([result["nav"], benchmark_nav], axis=1)
    write_dataframe_csv_atomic(compare_df, output_dir / "strategy_vs_hs300.csv")

    rows = build_yearly_return_rows(
        compare_df,
        strategy_col="nav",
        benchmark_col="benchmark_nav",
        benchmark_return_col="hs300_etf_return",
    )
    write_dataframe_csv_atomic(pd.DataFrame(rows), output_dir / "yearly_returns.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["nav"], color="#2b6cb0", linewidth=2.2, label="Strategy NAV")
    ax.plot(compare_df.index, compare_df["benchmark_nav"], color="#d64545", linewidth=2.0, label="HS300 Benchmark")
    ax.set_title("Strategy vs HS300 Benchmark", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, output_dir / "strategy_vs_hs300.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return compare_df


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    output_dir = build_output_dir(args.leader_margin)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = load_default_strategy_backtest_pool().copy()
    _, prices = load_core_selected_and_prices()
    start_ts = prices.index.max() - pd.DateOffset(years=args.years)
    prices = prices.loc[prices.index >= start_ts].copy()
    market_proxy = load_market_volume_proxy(years=args.years, refresh=False)

    params = build_default_strategy_params()
    params["signal_leader_margin"] = float(args.leader_margin)
    result, trades = run_default_strategy_with_params(
        prices,
        selected,
        params=params,
        market_proxy=market_proxy,
    )

    result_to_save = result.copy()
    result_to_save.index.name = "date"
    prices_to_save = prices.copy()
    prices_to_save.index.name = "date"

    write_dataframe_csv_atomic(selected, output_dir / "selected_etfs.csv", index=False)
    write_dataframe_csv_atomic(prices_to_save, output_dir / "prices.csv")
    write_dataframe_csv_atomic(result_to_save, output_dir / "backtest_nav.csv")
    write_dataframe_csv_atomic(trades, output_dir / "trades.csv", index=False)
    write_dataframe_csv_atomic(
        result_to_save[["nav"]].rename(columns={"nav": "historical_nav"}),
        output_dir / "historical_nav.csv",
    )

    summary = build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)
    summary_df = pd.DataFrame(
        [
            {
                "leader_margin": float(args.leader_margin),
                **summary,
            }
        ]
    )
    write_dataframe_csv_atomic(summary_df, output_dir / "summary.csv", index=False)

    title_suffix = f"leader_margin={args.leader_margin:.2%}"
    save_nav_with_trades(result, trades, output_dir, title_suffix)
    save_drawdown_chart(result, output_dir)
    compare_df = save_compare_chart(prices, result, output_dir)

    contribution_df = build_contribution_summary(prices, selected, result)
    write_dataframe_csv_atomic(contribution_df, output_dir / "contribution_summary.csv", index=False)

    print(f"output_dir={output_dir}")
    print(summary_df.to_string(index=False))
    print(f"max_drawdown_date={result['drawdown'].idxmin().date()}")
    print(f"latest_nav={float(result['nav'].iloc[-1]):.4f}")
    print(f"benchmark_latest_nav={float(compare_df['benchmark_nav'].iloc[-1]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
