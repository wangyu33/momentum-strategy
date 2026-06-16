#!/usr/bin/env python3
"""Compare overheat exposure cap variants against the baseline strategy."""

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
    run_threshold_dual_with_overheat_cap_strategy,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "archive_flat" / "compare_overheat_exposure_cap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare overheat exposure cap variants.")
    parser.add_argument("--years", type=int, default=15, help="Backtest years.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Fetch fresh histories instead of using local cached output.")
    return parser.parse_args()


def summarize(result: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int]:
    daily_ret = result["strategy_return"].fillna(0.0)
    ann_ret = annualized_return(result["nav"])
    ann_vol = float(daily_ret.std(ddof=0) * (252 ** 0.5))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    peak_date = pd.Timestamp("2024-10-08")
    trough_date = pd.Timestamp("2024-10-17")
    episode_ret = float(result.loc[trough_date, "nav"] / result.loc[peak_date, "nav"] - 1)
    return {
        "total_return": float(result["nav"].iloc[-1] - 1),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_rf0": sharpe,
        "max_drawdown": max_drawdown(result["nav"]),
        "trade_count": int(len(trades)),
        "episode_peak_to_trough_20241008_20241017": episode_ret,
    }


def main() -> int:
    args = parse_args()
    selected = load_fixed_etf_pool()
    if args.refresh:
        prices = fetch_histories(selected, years=args.years)
    else:
        prices = pd.read_csv(CORE_OUTPUT_DIR / "prices.csv", parse_dates=["date"]).set_index("date")

    base_result, base_trades = run_threshold_dual_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    cap70_result, cap70_trades = run_threshold_dual_with_overheat_cap_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        overheat_drawdown_cut=-0.03,
        overheat_momentum_cut=0.25,
        overheat_max_exposure=0.70,
    )
    cap40_result, cap40_trades = run_threshold_dual_with_overheat_cap_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        overheat_drawdown_cut=-0.03,
        overheat_momentum_cut=0.25,
        overheat_max_exposure=0.40,
    )
    cap50_result, cap50_trades = run_threshold_dual_with_overheat_cap_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        overheat_drawdown_cut=-0.03,
        overheat_momentum_cut=0.25,
        overheat_max_exposure=0.50,
    )

    rows = [
        {"strategy": "baseline_threshold_dual", **summarize(base_result, base_trades)},
        {"strategy": "overheat_cap_70", **summarize(cap70_result, cap70_trades)},
        {"strategy": "overheat_cap_40", **summarize(cap40_result, cap40_trades)},
        {"strategy": "overheat_cap_50", **summarize(cap50_result, cap50_trades)},
    ]
    summary = pd.DataFrame(rows)
    baseline = summary[summary["strategy"] == "baseline_threshold_dual"].iloc[0]
    summary["annualized_diff"] = summary["annualized_return"] - float(baseline["annualized_return"])
    summary["mdd_diff"] = summary["max_drawdown"] - float(baseline["max_drawdown"])
    summary["episode_diff"] = (
        summary["episode_peak_to_trough_20241008_20241017"]
        - float(baseline["episode_peak_to_trough_20241008_20241017"])
    )

    compare_df = pd.DataFrame(index=prices.index)
    compare_df["baseline_threshold_dual_nav"] = base_result["nav"]
    compare_df["overheat_cap_70_nav"] = cap70_result["nav"]
    compare_df["overheat_cap_40_nav"] = cap40_result["nav"]
    compare_df["overheat_cap_50_nav"] = cap50_result["nav"]
    compare_df["hs300_benchmark"] = build_benchmark_nav(prices, benchmark_code="510300")

    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    compare_df.to_csv(OUTPUT_DIR / "nav_compare.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["baseline_threshold_dual_nav"], linewidth=2.0, label="Baseline")
    ax.plot(compare_df.index, compare_df["overheat_cap_70_nav"], linewidth=1.8, label="Overheat Cap 70%")
    ax.plot(compare_df.index, compare_df["overheat_cap_40_nav"], linewidth=1.8, label="Overheat Cap 40%")
    ax.plot(compare_df.index, compare_df["overheat_cap_50_nav"], linewidth=1.8, label="Overheat Cap 50%")
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.5, linestyle="--", label="HS300")
    ax.set_title("Overheat Exposure Cap Comparison", loc="left", fontsize=16, fontweight="bold")
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
