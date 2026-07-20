#!/usr/bin/env python3
"""Compare threshold/breadth dual momentum variants against single momentum."""

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
    save_variant_compare_artifacts,
    save_text_output,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    fetch_histories,
    load_fixed_etf_pool,
    run_strategy,
)

matplotlib.use("Agg")

OUTPUT_DIR = build_archive_flat_output_dir("compare_dual_momentum_variants")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dual momentum variants.")
    parser.add_argument("--years", type=int, default=10, help="Backtest years.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Fetch fresh histories instead of using local cached output.")
    return parser.parse_args()


def run_threshold_dual_momentum(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
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
            continue
        if pd.notna(r_score) and r_score > 0.05:
            signal.loc[dt_idx] = r_asset
            exposure.loc[dt_idx] = 1.0
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            # Weak trend: still use defensive winner and half exposure.
            signal.loc[dt_idx] = d_asset
            exposure.loc[dt_idx] = 0.5
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
        else:
            signal.loc[dt_idx] = pd.NA
            exposure.loc[dt_idx] = 0.0

    return run_exposure_strategy(prices, selected, signal, exposure, momentum, fee_rate, slippage_rate)


def run_breadth_dual_momentum(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum = prices / prices.shift(lookback) - 1

    risk_codes = [c for c in RISK_CODES if c in prices.columns]
    defensive_codes = [c for c in DEFENSIVE_CODES if c in prices.columns]
    risk_mom = momentum[risk_codes]
    def_mom = momentum[defensive_codes]

    risk_winner = risk_mom.idxmax(axis=1, skipna=True)
    def_winner = def_mom.idxmax(axis=1, skipna=True)
    negative_count = (risk_mom <= 0).sum(axis=1)

    signal = pd.Series(index=prices.index, dtype="object")
    exposure = pd.Series(index=prices.index, dtype="float64")
    for dt_idx in prices.index:
        neg = int(negative_count.loc[dt_idx]) if pd.notna(negative_count.loc[dt_idx]) else len(risk_codes)
        r_asset = risk_winner.loc[dt_idx]
        d_asset = def_winner.loc[dt_idx]
        if neg >= 3 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            exposure.loc[dt_idx] = 1.0
        elif neg >= 2 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            exposure.loc[dt_idx] = 0.5
        elif pd.notna(r_asset):
            signal.loc[dt_idx] = r_asset
            exposure.loc[dt_idx] = 1.0
        else:
            signal.loc[dt_idx] = pd.NA
            exposure.loc[dt_idx] = 0.0

    return run_exposure_strategy(prices, selected, signal, exposure, momentum, fee_rate, slippage_rate)
def main() -> int:
    args = parse_args()
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
    threshold_result, threshold_trades = run_threshold_dual_momentum(
        prices, selected, args.lookback, args.fee_rate, args.slippage_rate
    )
    breadth_result, breadth_trades = run_breadth_dual_momentum(
        prices, selected, args.lookback, args.fee_rate, args.slippage_rate
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
    append_variant_result(
        rows,
        compare_df,
        None,
        strategy="threshold_dual_momentum",
        result=threshold_result,
        summary=summarize(threshold_result, threshold_trades, selected),
        nav_column="threshold_dual_momentum_nav",
    )
    append_variant_result(
        rows,
        compare_df,
        None,
        strategy="breadth_dual_momentum",
        result=breadth_result,
        summary=summarize(breadth_result, breadth_trades, selected),
        nav_column="breadth_dual_momentum_nav",
    )
    summary = finalize_baseline_diff_summary(
        rows,
        baseline_field="strategy",
        baseline_value="single_momentum",
        metric_mappings=(
            ("total_return", "return_diff"),
            ("annualized_return", "annualized_diff"),
            ("sharpe_rf0", "sharpe_diff"),
            ("max_drawdown", "mdd_diff"),
            ("trade_count", "trade_diff"),
        ),
    )
    named_outputs = {
        "threshold_dual_momentum": (threshold_result, threshold_trades),
        "breadth_dual_momentum": (breadth_result, breadth_trades),
    }
    save_text_output(
        OUTPUT_DIR,
        "rules.txt",
        """Threshold dual momentum:
- risk winner momentum > 5%: hold risk winner at 100%
- 0% < risk winner momentum <= 5%: hold defensive winner at 50%
- risk winner momentum <= 0%: hold defensive winner at 100% if its momentum > 0, else cash

Breadth dual momentum:
- if >=3 risk assets have non-positive momentum: hold defensive winner at 100%
- if 2 risk assets have non-positive momentum: hold defensive winner at 50%
- otherwise: hold risk winner at 100%
""",
    )

    save_variant_compare_artifacts(
        OUTPUT_DIR,
        summary,
        compare_df,
        named_outputs=named_outputs,
        plot_filename="comparison.png",
        title="Dual Momentum Variants Comparison",
        lines=(
            ("single_momentum_nav", "Single Momentum", 2.2),
            ("threshold_dual_momentum_nav", "Threshold Dual", 1.8),
            ("breadth_dual_momentum_nav", "Breadth Dual", 1.8),
        ),
        benchmark_label="HS300",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
