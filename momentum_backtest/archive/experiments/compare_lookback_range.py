#!/usr/bin/env python3
"""Compare threshold dual momentum performance across a continuous lookback range."""

from __future__ import annotations

import argparse

try:
    from ..runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

from run_backtest import (
    CORE_OUTPUT_DIR,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RESEARCH_OUTPUT_DIR,
    annualized_return,
    build_benchmark_nav,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    max_drawdown,
    run_threshold_dual_strategy,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "archive_flat" / "compare_lookback_range"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare threshold dual momentum across lookback range.")
    parser.add_argument("--years", type=int, default=10, help="Backtest years.")
    parser.add_argument("--lookback-start", type=int, default=20, help="Lookback start.")
    parser.add_argument("--lookback-end", type=int, default=30, help="Lookback end.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Fetch fresh histories instead of using local cached output.")
    return parser.parse_args()


def summarize(result: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int]:
    daily_ret = result["strategy_return"].fillna(0.0)
    ann_ret = annualized_return(result["nav"])
    ann_vol = float(daily_ret.std(ddof=0) * (252 ** 0.5))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    return {
        "total_return": float(result["nav"].iloc[-1] - 1),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_drawdown(result["nav"]),
        "trade_count": int(len(trades)),
    }


def main() -> int:
    args = parse_args()
    selected = load_fixed_etf_pool()
    if args.refresh:
        prices = fetch_histories(selected, years=args.years)
    else:
        prices = pd.read_csv(CORE_OUTPUT_DIR / "prices.csv", parse_dates=["date"]).set_index("date")

    rows = []
    compare_df = pd.DataFrame(index=prices.index)
    compare_df["hs300_benchmark"] = build_benchmark_nav(prices, benchmark_code="510300")

    for lookback in range(args.lookback_start, args.lookback_end + 1):
        result, trades = run_threshold_dual_strategy(
            prices,
            selected,
            lookback=lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        row = {"lookback": lookback, **summarize(result, trades)}
        rows.append(row)
        compare_df[f"k_{lookback}_nav"] = result["nav"]

    summary = pd.DataFrame(rows).sort_values(["sharpe_rf0", "annualized_return"], ascending=[False, False])
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    compare_df.to_csv(OUTPUT_DIR / "nav_compare.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    top_rows = summary.head(4)
    for _, row in top_rows.iterrows():
        k = int(row["lookback"])
        ax.plot(compare_df.index, compare_df[f"k_{k}_nav"], linewidth=2.0, label=f"k={k}")
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.6, linestyle="--", label="HS300")
    ax.set_title("Threshold Dual Momentum Lookback Range", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
