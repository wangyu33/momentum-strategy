#!/usr/bin/env python3
"""Compare overheat exposure cap variants against the baseline strategy.

This script compares historical threshold-dual overheat variants rather than the official
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
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
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
    run_threshold_dual_with_overheat_cap_strategy,
)

matplotlib.use("Agg")


OUTPUT_DIR = build_archive_flat_output_dir("compare_overheat_exposure_cap")
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare overheat exposure cap variants.")
    parser.add_argument("--years", type=int, default=15, help="Backtest years.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
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

    rows: list[dict[str, object]] = []
    compare_df = build_compare_frame(prices)

    def register_variant(
        strategy: str,
        result: pd.DataFrame,
        trades: pd.DataFrame,
        *,
        nav_column: str,
    ) -> None:
        append_variant_result(
            rows,
            compare_df,
            None,
            strategy=strategy,
            result=result,
            summary=summarize_variant_result(
                result,
                trades,
                extra_fields={
                    "episode_peak_to_trough_20241008_20241017": float(
                        result.loc[pd.Timestamp("2024-10-17"), "nav"]
                        / result.loc[pd.Timestamp("2024-10-08"), "nav"]
                        - 1
                    )
                },
            ),
            nav_column=nav_column,
        )

    register_variant("baseline_threshold_dual", base_result, base_trades, nav_column="baseline_threshold_dual_nav")
    register_variant("overheat_cap_70", cap70_result, cap70_trades, nav_column="overheat_cap_70_nav")
    register_variant("overheat_cap_40", cap40_result, cap40_trades, nav_column="overheat_cap_40_nav")
    register_variant("overheat_cap_50", cap50_result, cap50_trades, nav_column="overheat_cap_50_nav")

    summary = finalize_baseline_diff_summary(
        rows,
        baseline_field="strategy",
        baseline_value="baseline_threshold_dual",
        metric_mappings=[
            ("annualized_return", "annualized_diff"),
            ("max_drawdown", "mdd_diff"),
            ("episode_peak_to_trough_20241008_20241017", "episode_diff"),
        ],
    )
    save_plot_and_print_variant_compare_outputs(
        OUTPUT_DIR,
        summary,
        compare_df,
        plot_filename="comparison.png",
        title="Overheat Exposure Cap Comparison",
        lines=[
            ("baseline_threshold_dual_nav", "Baseline", 2.0),
            ("overheat_cap_70_nav", "Overheat Cap 70%", 1.8),
            ("overheat_cap_40_nav", "Overheat Cap 40%", 1.8),
            ("overheat_cap_50_nav", "Overheat Cap 50%", 1.8),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
