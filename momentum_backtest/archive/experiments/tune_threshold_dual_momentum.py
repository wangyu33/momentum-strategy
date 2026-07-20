#!/usr/bin/env python3
"""Grid-search threshold dual momentum parameters.

This script compares historical single/dual-momentum variants rather than the official
28.2691 formal baseline, so it should keep its own research-chain semantics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MOMENTUM_DIR = Path(__file__).resolve().parents[2]
if str(MOMENTUM_DIR) not in sys.path:
    sys.path.insert(0, str(MOMENTUM_DIR))

try:
    from ...runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

from archive_data_loaders import build_archive_flat_output_dir, load_core_selected_and_prices
from dual_momentum_helpers import (
    DEFENSIVE_CODES,
    RISK_CODES,
    run_exposure_strategy,
    summarize,
)
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    finalize_baseline_diff_summary,
    save_plot_and_print_variant_compare_outputs,
    save_named_nav_and_trades,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    fetch_histories,
    load_fixed_etf_pool,
    run_strategy,
)

matplotlib.use("Agg")


OUTPUT_DIR = build_archive_flat_output_dir("tune_threshold_dual_momentum")
INTENTIONALLY_UNANCHORED_BASELINE = True


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

    selected, prices = load_core_selected_and_prices(
        load_fixed_etf_pool(),
        years=args.years,
        refresh=args.refresh,
        fetch_fn=fetch_histories,
    )

    base_result, base_trades = run_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    rows: list[dict[str, object]] = []
    compare_df = build_compare_frame(prices)
    append_variant_result(
        rows,
        compare_df,
        None,
        strategy="single_momentum",
        result=base_result,
        summary=summarize(base_result, base_trades, selected),
        nav_column="single_momentum_nav",
    )

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
            row = summarize(result, trades, selected)
            append_variant_result(
                rows,
                compare_df,
                None,
                strategy=key,
                result=result,
                summary=row,
                nav_column=f"{key}_nav",
                extra_fields={
                    "threshold": threshold,
                    "weak_trend_def_weight": defensive_weight,
                },
            )
            if row["sharpe_rf0"] > best_sharpe:
                best_sharpe = row["sharpe_rf0"]
                best_key = key
                best_result = result
                best_trades = trades

    summary = finalize_baseline_diff_summary(
        rows,
        baseline_field="strategy",
        baseline_value="single_momentum",
        metric_mappings=[
            ("total_return", "return_diff"),
            ("annualized_return", "annualized_diff"),
            ("sharpe_rf0", "sharpe_diff"),
            ("max_drawdown", "mdd_diff"),
            ("trade_count", "trade_diff"),
        ],
    )
    summary = summary.sort_values(["sharpe_rf0", "annualized_return"], ascending=[False, False])

    if best_result is not None and best_trades is not None and best_key is not None:
        save_named_nav_and_trades(OUTPUT_DIR, "best", best_result, best_trades)

    top_rows = summary[summary["strategy"] != "single_momentum"].head(3)
    save_plot_and_print_variant_compare_outputs(
        OUTPUT_DIR,
        summary,
        compare_df,
        plot_filename="comparison.png",
        title="Threshold Dual Momentum Grid Search",
        lines=[("single_momentum_nav", "Single Momentum", 2.2)]
        + [(f"{row['strategy']}_nav", str(row["strategy"]), 1.8) for _, row in top_rows.iterrows()],
    )
    if best_key:
        print(f"best_by_sharpe={best_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
