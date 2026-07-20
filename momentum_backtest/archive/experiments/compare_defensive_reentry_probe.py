#!/usr/bin/env python3
"""Compare defensive-to-risk probe reentry variants against the current default baseline.

This baseline studies historical threshold-dual overheat behavior rather than the official
28.2691 formal baseline, so this script must keep its own research-chain semantics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MOMENTUM_DIR = Path(__file__).resolve().parents[2]
if str(MOMENTUM_DIR) not in sys.path:
    sys.path.insert(0, str(MOMENTUM_DIR))

try:
    from ...runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from archive_data_loaders import build_archive_flat_output_dir
from dual_momentum_helpers import load_cached_data, run_exposure_strategy, summarize_variant_result as summarize_exposure_variant_result
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    filter_available_plot_lines,
    finalize_baseline_diff_summary,
    save_plot_and_print_variant_compare_outputs,
)
from archive_strategy_common import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_OVERHEAT_DRAWDOWN_CUT,
    DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
    DEFAULT_OVERHEAT_MAX_EXPOSURE,
    DEFAULT_OVERHEAT_MOMENTUM_CUT,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    DEFENSIVE_CODES,
    RISK_CODES,
    run_threshold_dual_with_overheat_cap_strategy,
)


OUTPUT_DIR = build_archive_flat_output_dir("compare_defensive_reentry_probe")
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare defensive-to-risk probe reentry variants.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--probe-exposures", default="0.5,0.6,0.7", help="Comma-separated exposures for first probe days.")
    parser.add_argument("--probe-days", default="1,2,3", help="Comma-separated day counts to keep probe exposure.")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def run_defensive_reentry_probe_strategy(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    probe_exposure: float,
    probe_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    base_result, _ = run_threshold_dual_with_overheat_cap_strategy(
        prices,
        selected,
        lookback=lookback,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        overheat_drawdown_cut=DEFAULT_OVERHEAT_DRAWDOWN_CUT,
        overheat_momentum_cut=DEFAULT_OVERHEAT_MOMENTUM_CUT,
        overheat_max_exposure=DEFAULT_OVERHEAT_MAX_EXPOSURE,
        overheat_high_momentum_cut=DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
        overheat_high_max_exposure=DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
    )
    signal = base_result["signal"].copy()
    target_exposure = base_result["target_exposure"].copy()
    current_momentum = base_result["current_momentum"].copy()
    risk_codes = {code for code in RISK_CODES if code in prices.columns}
    defensive_codes = {code for code in DEFENSIVE_CODES if code in prices.columns}

    probe_days_remaining = 0
    probe_asset: str | None = None
    reentry_count = 0
    for dt_idx in prices.index:
        asset = signal.loc[dt_idx]
        prev_asset = signal.shift(1).loc[dt_idx]

        if asset in risk_codes and prev_asset in defensive_codes:
            probe_days_remaining = probe_days
            probe_asset = asset
            reentry_count += 1
        elif asset != probe_asset:
            probe_days_remaining = 0
            probe_asset = None

        if asset in risk_codes and probe_days_remaining > 0 and asset == probe_asset:
            target_exposure.loc[dt_idx] = min(float(target_exposure.loc[dt_idx]), probe_exposure)
            probe_days_remaining -= 1
            if probe_days_remaining == 0:
                probe_asset = None

    result, trades = run_exposure_strategy(
        prices,
        selected,
        signal,
        target_exposure,
        None,
        fee_rate,
        slippage_rate,
        current_momentum=current_momentum,
        normalize_signal_series=True,
    )
    return result, trades, reentry_count


def window_return(result: pd.DataFrame, start: str, end: str) -> float:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return float(result.loc[end_ts, "nav"] / result.loc[start_ts, "nav"] - 1)


def main() -> int:
    args = parse_args()
    selected, prices = load_cached_data()

    baseline_result, baseline_trades = run_threshold_dual_with_overheat_cap_strategy(
        prices,
        selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        overheat_drawdown_cut=DEFAULT_OVERHEAT_DRAWDOWN_CUT,
        overheat_momentum_cut=DEFAULT_OVERHEAT_MOMENTUM_CUT,
        overheat_max_exposure=DEFAULT_OVERHEAT_MAX_EXPOSURE,
        overheat_high_momentum_cut=DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT,
        overheat_high_max_exposure=DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE,
    )
    rows: list[dict[str, object]] = []
    compare_df = build_compare_frame(prices)
    append_variant_result(
        rows,
        compare_df,
        None,
        strategy="current_default",
        result=baseline_result,
        summary=summarize_exposure_variant_result(
            baseline_result,
            baseline_trades,
            selected,
            extra_fields={
                "reentry_count": 0,
                "episode_2018_full": window_return(baseline_result, "2017-09-06", "2018-11-28"),
                "episode_2018_nov": window_return(baseline_result, "2018-11-01", "2018-11-28"),
                "episode_2024_oct": window_return(baseline_result, "2024-10-08", "2024-10-17"),
            },
        ),
        nav_column="current_default_nav",
    )

    for probe_exposure in parse_float_list(args.probe_exposures):
        for probe_days in parse_int_list(args.probe_days):
            result, trades, reentry_count = run_defensive_reentry_probe_strategy(
                prices,
                selected,
                lookback=args.lookback,
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
                probe_exposure=probe_exposure,
                probe_days=probe_days,
            )
            name = f"probe_{int(round(probe_exposure * 100)):02d}_d{probe_days}"
            append_variant_result(
                rows,
                compare_df,
                None,
                strategy=name,
                result=result,
                summary=summarize_exposure_variant_result(
                    result,
                    trades,
                    selected,
                    extra_fields={
                        "reentry_count": reentry_count,
                        "episode_2018_full": window_return(result, "2017-09-06", "2018-11-28"),
                        "episode_2018_nov": window_return(result, "2018-11-01", "2018-11-28"),
                        "episode_2024_oct": window_return(result, "2024-10-08", "2024-10-17"),
                    },
                ),
                nav_column=f"{name}_nav",
            )

    summary = finalize_baseline_diff_summary(
        rows,
        baseline_field="strategy",
        baseline_value="current_default",
        metric_mappings=(
            ("annualized_return", "annualized_return_diff"),
            ("max_drawdown", "max_drawdown_diff"),
            ("episode_2018_full", "episode_2018_full_diff"),
            ("episode_2018_nov", "episode_2018_nov_diff"),
            ("episode_2024_oct", "episode_2024_oct_diff"),
            ("sharpe_rf0", "sharpe_rf0_diff"),
            ("trade_count", "trade_count_diff"),
        ),
    )

    plot_lines = [
        ("current_default_nav", "Current Default", 2.2),
        ("probe_50_d1_nav", "Probe 50% D1", 1.8),
        ("probe_60_d2_nav", "Probe 60% D2", 1.8),
        ("probe_70_d3_nav", "Probe 70% D3", 1.8),
    ]
    available_plot_lines = filter_available_plot_lines(compare_df, plot_lines)
    save_plot_and_print_variant_compare_outputs(
        OUTPUT_DIR,
        summary,
        compare_df,
        plot_filename="comparison.png",
        title="Defensive Reentry Probe Comparison",
        lines=available_plot_lines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
