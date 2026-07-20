#!/usr/bin/env python3
"""Compare multi-horizon composite momentum signals against the current baseline.

This script compares historical threshold-dual / multi-horizon variants rather than the
official 28.2691 formal baseline, so it should keep its own research-chain semantics.
"""

from __future__ import annotations

import argparse
import json
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
from dual_momentum_helpers import run_exposure_strategy, summarize
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    build_summary_frame,
    finalize_baseline_diff_summary,
    save_plot_and_print_variant_compare_outputs,
)
from archive_strategy_common import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    DEFENSIVE_CODES,
    RISK_CODES,
    fetch_histories,
    load_fixed_etf_pool,
    run_threshold_dual_strategy,
)

matplotlib.use("Agg")


OUTPUT_DIR = build_archive_flat_output_dir("compare_multihorizon_combo")
INTENTIONALLY_UNANCHORED_BASELINE = True


MULTIHORIZON_SPECS: list[dict[str, object]] = [
    {"name": "baseline_k25", "kind": "baseline", "lookback": 25},
    {"name": "mh_20_25_30", "kind": "combo", "windows": [20, 25, 30], "weights": [0.3, 0.4, 0.3]},
    {"name": "mh_15_25_60", "kind": "combo", "windows": [15, 25, 60], "weights": [0.25, 0.5, 0.25]},
    {"name": "mh_20_60_120", "kind": "combo", "windows": [20, 60, 120], "weights": [0.4, 0.35, 0.25]},
    {"name": "mh_10_20_25", "kind": "combo", "windows": [10, 20, 25], "weights": [0.2, 0.4, 0.4]},
    {"name": "mh_25_60", "kind": "combo", "windows": [25, 60], "weights": [0.6, 0.4]},
    {"name": "mh_30_60_120", "kind": "combo", "windows": [30, 60, 120], "weights": [0.4, 0.35, 0.25]},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multi-horizon momentum combinations.")
    parser.add_argument("--years", type=int, default=10, help="Backtest years.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--absolute-threshold", type=float, default=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD, help="Risk absolute momentum threshold.")
    parser.add_argument("--thresholds", default="", help="Optional comma-separated thresholds for multi-horizon combos.")
    parser.add_argument("--weak-trend-defensive-weight", type=float, default=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT, help="Defensive exposure when risk trend is weak but positive.")
    parser.add_argument("--refresh", action="store_true", help="Fetch fresh histories instead of using local cached output.")
    return parser.parse_args()


def build_composite_score(prices: pd.DataFrame, windows: list[int], weights: list[float]) -> pd.DataFrame:
    score = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    valid = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for window, weight in zip(windows, weights):
        momentum = prices / prices.shift(window) - 1
        usable = momentum.notna().astype(float)
        score = score.add(momentum.fillna(0.0) * weight, fill_value=0.0)
        valid = valid.add(usable * weight, fill_value=0.0)
    score = score.divide(valid.where(valid > 0))
    return score


def run_multihorizon_threshold_dual_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    windows: list[int],
    weights: list[float],
    fee_rate: float,
    slippage_rate: float,
    absolute_threshold: float,
    weak_trend_defensive_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    composite = build_composite_score(prices, windows, weights)
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    defensive_codes = [code for code in DEFENSIVE_CODES if code in prices.columns]
    risk_score = composite[risk_codes]
    defensive_score = composite[defensive_codes]
    risk_winner = risk_score.idxmax(axis=1, skipna=True)
    risk_best = risk_score.max(axis=1, skipna=True)
    defensive_winner = defensive_score.idxmax(axis=1, skipna=True)
    defensive_best = defensive_score.max(axis=1, skipna=True)

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    signal_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    for dt_idx in prices.index:
        r_asset = risk_winner.loc[dt_idx]
        r_score = risk_best.loc[dt_idx]
        d_asset = defensive_winner.loc[dt_idx]
        d_score = defensive_best.loc[dt_idx]

        if pd.isna(r_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            signal_momentum.loc[dt_idx] = float("nan")
        elif pd.notna(r_score) and r_score > absolute_threshold:
            signal.loc[dt_idx] = r_asset
            target_exposure.loc[dt_idx] = 1.0
            signal_momentum.loc[dt_idx] = float(r_score)
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = weak_trend_defensive_weight
            signal_momentum.loc[dt_idx] = float(r_score)
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            signal_momentum.loc[dt_idx] = float(d_score) if pd.notna(d_score) else float("nan")
        else:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            signal_momentum.loc[dt_idx] = float("nan")

    return run_exposure_strategy(
        prices,
        selected,
        signal,
        target_exposure,
        None,
        fee_rate,
        slippage_rate,
        current_momentum=signal_momentum,
    )


def main() -> int:
    args = parse_args()
    selected, prices = load_core_selected_and_prices(
        load_fixed_etf_pool(),
        years=args.years,
        refresh=args.refresh,
        fetch_fn=fetch_histories,
    )

    thresholds = [args.absolute_threshold]
    if args.thresholds.strip():
        thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    rows: list[dict[str, object]] = []
    compare_df = build_compare_frame(prices)

    for spec in MULTIHORIZON_SPECS:
        active_thresholds = [args.absolute_threshold] if spec["kind"] == "baseline" else thresholds
        for threshold in active_thresholds:
            if spec["kind"] == "baseline":
                result, trades = run_threshold_dual_strategy(
                    prices,
                    selected,
                    lookback=int(spec["lookback"]),
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                    absolute_threshold=threshold,
                    weak_trend_defensive_weight=args.weak_trend_defensive_weight,
                )
                strategy_name = str(spec["name"])
            else:
                result, trades = run_multihorizon_threshold_dual_strategy(
                    prices,
                    selected,
                    windows=list(spec["windows"]),
                    weights=list(spec["weights"]),
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                    absolute_threshold=threshold,
                    weak_trend_defensive_weight=args.weak_trend_defensive_weight,
                )
                strategy_name = f"{spec['name']}_thr{int(round(threshold * 100)):02d}"

            append_variant_result(
                rows,
                compare_df,
                None,
                strategy=strategy_name,
                result=result,
                summary=summarize(result, trades, selected),
                extra_fields={
                    "base_strategy": str(spec["name"]),
                    "threshold": threshold,
                    "windows": json.dumps(spec.get("windows", [spec.get("lookback")]), ensure_ascii=False),
                    "weights": json.dumps(spec.get("weights", [1.0]), ensure_ascii=False),
                },
                nav_column=f"{strategy_name}_nav",
            )

    summary = finalize_baseline_diff_summary(
        build_summary_frame(
            rows,
            sort_by=["sharpe_rf0", "annualized_return"],
            ascending=[False, False],
            reset_index=True,
        ),
        baseline_field="strategy",
        baseline_value="baseline_k25",
        metric_mappings=(
            ("total_return", "return_diff"),
            ("annualized_return", "annualized_diff"),
            ("sharpe_rf0", "sharpe_diff"),
            ("max_drawdown", "mdd_diff"),
            ("trade_count", "trade_diff"),
        ),
    )

    top_rows = summary.head(4)
    save_plot_and_print_variant_compare_outputs(
        OUTPUT_DIR,
        summary,
        compare_df,
        plot_filename="comparison.png",
        title="Multi-Horizon Momentum Comparison",
        lines=[(f"{str(row['strategy'])}_nav", str(row["strategy"]), 2.0) for _, row in top_rows.iterrows()],
        benchmark_label="HS300",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
