#!/usr/bin/env python3
"""Grid-search threshold dual momentum parameters."""

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

from compare_dual_momentum_variants import (
    DEFENSIVE_CODES,
    RISK_CODES,
    load_cached_data,
    run_exposure_strategy,
    summarize,
)
from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RESEARCH_OUTPUT_DIR,
    build_benchmark_nav,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    run_strategy,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "archive_flat" / "tune_threshold_dual_momentum"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune threshold dual momentum parameters.")
    parser.add_argument("--years", type=int, default=10, help="Backtest years.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--thresholds", default="0.03,0.05,0.08", help="Comma-separated absolute momentum thresholds.")
    parser.add_argument("--defensive-weights", default="0.3,0.5,0.7", help="Comma-separated defensive weights in weak-trend regime.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Fetch fresh histories instead of using local cached output.")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def run_threshold_variant(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    threshold: float,
    defensive_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum = prices / prices.shift(lookback) - 1
    risk_codes = [c for c in RISK_CODES if c in prices.columns]
    defensive_codes = [c for c in DEFENSIVE_CODES if c in prices.columns]
    risk_mom = momentum[risk_codes]
    def_mom = momentum[defensive_codes]

    risk_winner = risk_mom.idxmax(axis=1, skipna=True)
    risk_best = risk_mom.max(axis=1, skipna=True)
    def_winner = def_mom.idxmax(axis=1, skipna=True)
    def_best = def_mom.max(axis=1, skipna=True)

    signal = pd.Series(index=prices.index, dtype="object")
    exposure = pd.Series(index=prices.index, dtype="float64")
    for dt_idx in prices.index:
        r_asset = risk_winner.loc[dt_idx]
        r_score = risk_best.loc[dt_idx]
        d_asset = def_winner.loc[dt_idx]
        d_score = def_best.loc[dt_idx]

        if pd.isna(r_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            exposure.loc[dt_idx] = 0.0
        elif pd.notna(r_score) and r_score > threshold:
            signal.loc[dt_idx] = r_asset
            exposure.loc[dt_idx] = 1.0
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            exposure.loc[dt_idx] = defensive_weight
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
        else:
            signal.loc[dt_idx] = pd.NA
            exposure.loc[dt_idx] = 0.0

    return run_exposure_strategy(prices, selected, signal, exposure, momentum, fee_rate, slippage_rate)


def main() -> int:
    args = parse_args()
    thresholds = parse_float_list(args.thresholds)
    defensive_weights = parse_float_list(args.defensive_weights)

    if args.refresh:
        selected = load_fixed_etf_pool()
        prices = fetch_histories(selected, years=args.years)
    else:
        selected, prices = load_cached_data()

    base_result, base_trades = run_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    rows = [{"strategy": "single_momentum", **summarize(base_result, base_trades, selected)}]
    compare_df = pd.DataFrame(index=prices.index)
    compare_df["single_momentum_nav"] = base_result["nav"]
    compare_df["hs300_benchmark"] = build_benchmark_nav(prices, benchmark_code="510300")

    best_key = None
    best_sharpe = -1e9
    best_result = None
    best_trades = None
    for threshold in thresholds:
        for defensive_weight in defensive_weights:
            result, trades = run_threshold_variant(
                prices,
                selected,
                lookback=args.lookback,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
                threshold=threshold,
                defensive_weight=defensive_weight,
            )
            key = f"thr_{threshold:.0%}_def_{defensive_weight:.0%}"
            row = {
                "strategy": key,
                "threshold": threshold,
                "weak_trend_def_weight": defensive_weight,
                **summarize(result, trades, selected),
            }
            rows.append(row)
            compare_df[f"{key}_nav"] = result["nav"]
            if row["sharpe_rf0"] > best_sharpe:
                best_sharpe = row["sharpe_rf0"]
                best_key = key
                best_result = result
                best_trades = trades

    summary = pd.DataFrame(rows)
    base = summary.iloc[0]
    summary["return_diff"] = summary["total_return"] - base["total_return"]
    summary["annualized_diff"] = summary["annualized_return"] - base["annualized_return"]
    summary["sharpe_diff"] = summary["sharpe_rf0"] - base["sharpe_rf0"]
    summary["mdd_diff"] = summary["max_drawdown"] - base["max_drawdown"]
    summary["trade_diff"] = summary["trade_count"] - base["trade_count"]
    summary = summary.sort_values(["sharpe_rf0", "annualized_return"], ascending=[False, False])

    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    compare_df.to_csv(OUTPUT_DIR / "nav_compare.csv")
    if best_result is not None and best_trades is not None and best_key is not None:
        best_result.to_csv(OUTPUT_DIR / "best_nav.csv")
        best_trades.to_csv(OUTPUT_DIR / "best_trades.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["single_momentum_nav"], linewidth=2.2, label="Single Momentum")
    top_rows = summary[summary["strategy"] != "single_momentum"].head(3)
    for _, row in top_rows.iterrows():
        key = row["strategy"]
        ax.plot(compare_df.index, compare_df[f"{key}_nav"], linewidth=1.8, label=key)
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.6, linestyle="--", label="HS300")
    ax.set_title("Threshold Dual Momentum Grid Search", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_csv(index=False))
    if best_key:
        print(f"best_by_sharpe={best_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
