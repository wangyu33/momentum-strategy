#!/usr/bin/env python3
"""Focused comparison for China Internet and Germany ETF pool variants."""

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

from compare_candidate_pool_additions import (
    BASE_DEFENSIVE_CODES,
    BASE_RISK_CODES,
    ETF_513030,
    ETF_513050,
    run_custom_threshold_dual_with_overheat,
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
    save_figure_atomic,
    write_dataframe_csv_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "archive_flat" / "compare_china_internet_germany_variants"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare China Internet and Germany ETF variants.")
    parser.add_argument("--years", type=int, default=6, help="History years to request before overlap alignment.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_pool = load_fixed_etf_pool()
    candidates = [
        ("base_pool", [], BASE_RISK_CODES, BASE_DEFENSIVE_CODES),
        ("china_internet_risk", [ETF_513050], BASE_RISK_CODES + ["513050"], BASE_DEFENSIVE_CODES),
        ("germany_risk", [ETF_513030], BASE_RISK_CODES + ["513030"], BASE_DEFENSIVE_CODES),
        ("germany_defensive", [ETF_513030], BASE_RISK_CODES, BASE_DEFENSIVE_CODES + ["513030"]),
        (
            "china_internet_plus_germany_risk",
            [ETF_513050, ETF_513030],
            BASE_RISK_CODES + ["513050", "513030"],
            BASE_DEFENSIVE_CODES,
        ),
        (
            "china_internet_plus_germany_defensive",
            [ETF_513050, ETF_513030],
            BASE_RISK_CODES + ["513050"],
            BASE_DEFENSIVE_CODES + ["513030"],
        ),
    ]

    rows: list[dict[str, object]] = []
    compare_df = None
    base_metrics: dict[str, float | int] = {}

    for name, additions, risk_codes, defensive_codes in candidates:
        selected = pd.concat([base_pool, pd.DataFrame(additions)], ignore_index=True) if additions else base_pool.copy()
        prices = fetch_histories(selected, years=args.years)
        prices = prices.dropna(how="any")
        result, trades = run_custom_threshold_dual_with_overheat(
            prices=prices,
            selected=selected,
            risk_codes=risk_codes,
            defensive_codes=defensive_codes,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        if compare_df is None:
            compare_df = pd.DataFrame(index=result.index)
            compare_df["hs300_benchmark"] = build_benchmark_nav(prices, benchmark_code="510300")
        compare_df[name] = result["nav"].reindex(compare_df.index)

        row = {
            "variant": name,
            "risk_codes": ",".join([code for code in risk_codes if code in prices.columns]),
            "defensive_codes": ",".join([code for code in defensive_codes if code in prices.columns]),
            **summarize(result, trades),
        }
        rows.append(row)
        if name == "base_pool":
            base_metrics = {
                "total_return": float(row["total_return"]),
                "annualized_return": float(row["annualized_return"]),
                "sharpe_rf0": float(row["sharpe_rf0"]),
                "max_drawdown": float(row["max_drawdown"]),
                "trade_count": int(row["trade_count"]),
            }

    summary_df = pd.DataFrame(rows)
    summary_df["return_diff"] = summary_df["total_return"] - float(base_metrics["total_return"])
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(base_metrics["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(base_metrics["sharpe_rf0"])
    summary_df["mdd_diff"] = summary_df["max_drawdown"] - float(base_metrics["max_drawdown"])
    summary_df["trade_diff"] = summary_df["trade_count"] - int(base_metrics["trade_count"])

    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_dataframe_csv_atomic(summary_df, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(compare_df, OUTPUT_DIR / "nav_compare.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(compare_df.index, compare_df["base_pool"], linewidth=2.3, label="Base Pool")
    ax.plot(compare_df.index, compare_df["china_internet_risk"], linewidth=1.8, label="+ China Internet")
    ax.plot(compare_df.index, compare_df["germany_risk"], linewidth=1.8, label="+ Germany Risk")
    ax.plot(compare_df.index, compare_df["germany_defensive"], linewidth=1.8, label="+ Germany Defensive")
    ax.plot(compare_df.index, compare_df["china_internet_plus_germany_risk"], linewidth=1.8, label="+ Both as Risk")
    ax.plot(compare_df.index, compare_df["china_internet_plus_germany_defensive"], linewidth=1.8, label="+ CI Risk + Germany Def")
    ax.plot(compare_df.index, compare_df["hs300_benchmark"], linewidth=1.5, linestyle="--", label="HS300 ETF")
    ax.set_title("China Internet / Germany ETF Focused Variants", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.legend()
    fig.tight_layout()
    save_figure_atomic(fig, OUTPUT_DIR / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
