#!/usr/bin/env python3
"""Compare threshold dual momentum performance across a continuous lookback range.

This script compares historical threshold-dual variants rather than the official 28.2691
formal baseline, so it should keep its own research-chain semantics.
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
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    build_summary_frame,
    finalize_baseline_diff_summary,
    save_plot_and_print_variant_compare_outputs,
    summarize_variant_result,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    fetch_histories,
    load_fixed_etf_pool,
    run_threshold_dual_strategy,
)

matplotlib.use("Agg")


OUTPUT_DIR = build_archive_flat_output_dir("compare_lookback_range")
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare threshold dual momentum across lookback range.")
    parser.add_argument("--years", type=int, default=10, help="Backtest years.")
    parser.add_argument("--lookback-start", type=int, default=20, help="Lookback start.")
    parser.add_argument("--lookback-end", type=int, default=30, help="Lookback end.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Fetch fresh histories instead of using local cached output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected, prices = load_core_selected_and_prices(
        load_fixed_etf_pool(),
        years=args.years,
        refresh=args.refresh,
        fetch_fn=fetch_histories,
    )

    rows = []
    compare_df = build_compare_frame(prices)

    for lookback in range(args.lookback_start, args.lookback_end + 1):
        result, trades = run_threshold_dual_strategy(
            prices,
            selected,
            lookback=lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        append_variant_result(
            rows,
            compare_df,
            None,
            strategy=f"k_{lookback}",
            result=result,
            summary=summarize_variant_result(result, trades),
            strategy_field="strategy",
            nav_column=f"k_{lookback}_nav",
            extra_fields={"lookback": lookback},
        )

    summary = build_summary_frame(rows, sort_by=["sharpe_rf0", "annualized_return"], ascending=[False, False])
    best_lookback = int(summary.iloc[0]["lookback"])
    summary = finalize_baseline_diff_summary(
        summary,
        baseline_field="lookback",
        baseline_value=args.lookback_start,
        metric_mappings=[
            ("total_return", "return_diff_vs_first"),
            ("annualized_return", "annualized_diff_vs_first"),
            ("sharpe_rf0", "sharpe_diff_vs_first"),
            ("max_drawdown", "mdd_diff_vs_first"),
            ("trade_count", "trade_diff_vs_first"),
        ],
    )
    top_rows = summary.head(4)
    save_plot_and_print_variant_compare_outputs(
        OUTPUT_DIR,
        summary,
        compare_df,
        plot_filename="comparison.png",
        title=f"Threshold Dual Momentum Lookback Range (best={best_lookback})",
        lines=[(f"k_{int(row['lookback'])}_nav", f"k={int(row['lookback'])}", 2.0) for _, row in top_rows.iterrows()],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
