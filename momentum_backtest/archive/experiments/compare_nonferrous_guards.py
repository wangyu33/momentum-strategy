#!/usr/bin/env python3
"""Test guard rules for adding Nonferrous Metals ETF into the risk pool."""

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

try:
    from ...official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

from candidate_pool_common import BASE_DEFENSIVE_CODES, BASE_RISK_CODES
from archive_data_loaders import (
    build_archive_flat_output_dir,
    dedupe_selected_pool,
    load_required_candidate_prices,
    load_recent_selected_prices,
)
from dual_momentum_helpers import summarize_variant_result as summarize_exposure_variant_result
from strategy_signal_common import run_signal_strategy
from variant_compare_helpers import (
    append_variant_result,
    ensure_compare_frame,
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
    fetch_histories,
    load_fixed_etf_pool,
)

matplotlib.use("Agg")

NONFERROUS_ETF = {"theme": "有色金属", "code": "512400", "name": "有色金属ETF南方", "sina_symbol": "sh512400"}
NONFERROUS_CODE = "512400"
OUTPUT_DIR = build_archive_flat_output_dir("compare_nonferrous_guards")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Nonferrous Metals ETF guard variants.")
    parser.add_argument("--years", type=int, default=15, help="History years to request before overlap alignment.")
    parser.add_argument("--start-date", type=str, default="2012-01-01", help="Analysis start date.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Fetch fresh histories instead of preferring cached core prices.")
    parser.add_argument("--base-only", action="store_true", help="Only run the cached base_pool variant.")
    return parser.parse_args()


def load_variant_prices(
    base_pool: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    years: int,
    refresh: bool,
    include_nonferrous: bool,
) -> pd.DataFrame:
    selected = dedupe_selected_pool(selected)
    if not refresh and not include_nonferrous:
        return load_recent_selected_prices(selected, years)
    if not include_nonferrous:
        return fetch_histories(selected, years=years).dropna(how="any")

    _, prices = load_required_candidate_prices(
        base_pool,
        [NONFERROUS_ETF],
        years=years,
        refresh=refresh,
        fetch_fn=fetch_histories,
        label="nonferrous candidates",
        failure_prefix="failed to load nonferrous ETF history",
    )
    return prices.dropna(how="any")


def build_base_components(prices: pd.DataFrame, lookback: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    momentum = prices / prices.shift(lookback) - 1
    risk_codes = [code for code in BASE_RISK_CODES + [NONFERROUS_CODE] if code in prices.columns]
    defensive_codes = [code for code in BASE_DEFENSIVE_CODES if code in prices.columns]
    return momentum, momentum[risk_codes], momentum[defensive_codes]


def choose_signal(
    risk_scores: pd.Series,
    defensive_scores: pd.Series,
    nonferrous_abs_threshold: float | None = None,
    nonferrous_margin_threshold: float | None = None,
) -> tuple[object, float, float]:
    valid_risk = risk_scores.dropna().sort_values(ascending=False)
    valid_def = defensive_scores.dropna().sort_values(ascending=False)
    r_asset = valid_risk.index[0] if not valid_risk.empty else pd.NA
    r_score = float(valid_risk.iloc[0]) if not valid_risk.empty else float("nan")
    d_asset = valid_def.index[0] if not valid_def.empty else pd.NA
    d_score = float(valid_def.iloc[0]) if not valid_def.empty else float("nan")

    candidate_asset = r_asset
    candidate_score = r_score
    if pd.notna(r_asset) and str(r_asset) == NONFERROUS_CODE:
        second_score = float(valid_risk.iloc[1]) if len(valid_risk) >= 2 else float("nan")
        second_asset = valid_risk.index[1] if len(valid_risk) >= 2 else pd.NA
        if nonferrous_abs_threshold is not None and pd.notna(r_score) and r_score <= nonferrous_abs_threshold:
            candidate_asset = second_asset
            candidate_score = second_score
        elif (
            nonferrous_margin_threshold is not None
            and pd.notna(second_score)
            and (r_score - second_score) < nonferrous_margin_threshold
        ):
            candidate_asset = second_asset
            candidate_score = second_score

    if pd.notna(candidate_score) and candidate_score > DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD:
        return candidate_asset, 1.0, float(candidate_score)
    if pd.notna(candidate_score) and candidate_score > 0 and pd.notna(d_asset):
        return d_asset, DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT, float(candidate_score)
    if pd.notna(d_asset):
        return d_asset, (1.0 if pd.notna(d_score) and d_score > 0 else 0.0), d_score
    return pd.NA, 0.0, float("nan")


def run_variant(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
    nonferrous_abs_threshold: float | None = None,
    nonferrous_margin_threshold: float | None = None,
    nonferrous_max_exposure: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, risk_mom, defensive_mom = build_base_components(prices, lookback)
    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    for dt_idx in prices.index:
        asset, exposure, momentum = choose_signal(
            risk_scores=risk_mom.loc[dt_idx],
            defensive_scores=defensive_mom.loc[dt_idx],
            nonferrous_abs_threshold=nonferrous_abs_threshold,
            nonferrous_margin_threshold=nonferrous_margin_threshold,
        )
        signal.loc[dt_idx] = asset
        target_exposure.loc[dt_idx] = exposure
        current_momentum.loc[dt_idx] = momentum

    base_result, _ = run_signal_strategy(prices, selected, signal, target_exposure, current_momentum, fee_rate, slippage_rate)
    adjusted_exposure = target_exposure.copy()
    if nonferrous_max_exposure is not None:
        nonferrous_mask = signal == NONFERROUS_CODE
        adjusted_exposure.loc[nonferrous_mask] = adjusted_exposure.loc[nonferrous_mask].clip(upper=nonferrous_max_exposure)

    risk_codes = [code for code in risk_mom.columns if code in prices.columns]
    overheat_mask = (
        signal.isin(risk_codes)
        & (adjusted_exposure > DEFAULT_OVERHEAT_MAX_EXPOSURE)
        & (base_result["drawdown"] >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (current_momentum >= DEFAULT_OVERHEAT_MOMENTUM_CUT)
    )
    adjusted_exposure.loc[overheat_mask] = DEFAULT_OVERHEAT_MAX_EXPOSURE
    extreme_mask = (
        signal.isin(risk_codes)
        & (adjusted_exposure > DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE)
        & (base_result["drawdown"] >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (current_momentum >= DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT)
    )
    adjusted_exposure.loc[extreme_mask] = DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE

    return run_signal_strategy(prices, selected, signal, adjusted_exposure, current_momentum, fee_rate, slippage_rate)


def main() -> int:
    args = parse_args()
    base_pool = load_fixed_etf_pool()
    selected = pd.concat([base_pool, pd.DataFrame([NONFERROUS_ETF])], ignore_index=True)
    selected = dedupe_selected_pool(selected)

    variants = [
        ("base_pool", None, None, None, False),
        ("nonferrous_risk", None, None, None, True),
        ("nonferrous_cap_70", None, None, 0.70, True),
        ("nonferrous_cap_60", None, None, 0.60, True),
        ("nonferrous_abs_08", 0.08, None, None, True),
        ("nonferrous_abs_10", 0.10, None, None, True),
        ("nonferrous_margin_03", None, 0.03, None, True),
        ("nonferrous_margin_05", None, 0.05, None, True),
        ("nonferrous_margin_03_cap_70", None, 0.03, 0.70, True),
    ]
    if args.base_only:
        variants = variants[:1]

    rows: list[dict[str, object]] = []
    compare_df = None
    base_prices = load_variant_prices(base_pool, base_pool, years=args.years, refresh=args.refresh, include_nonferrous=False)
    base_prices = base_prices.loc[base_prices.index >= pd.Timestamp(args.start_date)]
    all_prices: pd.DataFrame | None = None

    for name, abs_thr, margin_thr, max_exposure, include_nonferrous in variants:
        current_selected = selected if include_nonferrous else base_pool.copy()
        if include_nonferrous:
            if all_prices is None:
                all_prices = load_variant_prices(
                    base_pool,
                    selected,
                    years=args.years,
                    refresh=args.refresh,
                    include_nonferrous=True,
                )
                all_prices = all_prices.loc[all_prices.index >= pd.Timestamp(args.start_date)]
            current_prices = all_prices
        else:
            current_prices = base_prices
        result, trades = run_variant(
            current_prices,
            current_selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            nonferrous_abs_threshold=abs_thr if include_nonferrous else None,
            nonferrous_margin_threshold=margin_thr if include_nonferrous else None,
            nonferrous_max_exposure=max_exposure if include_nonferrous else None,
        )
        if name == "base_pool":
            result = apply_official_baseline_nav_anchor(result)
        compare_df = ensure_compare_frame(compare_df, current_prices, index=result.index)
        row = summarize_exposure_variant_result(
            result,
            trades,
            current_selected,
            include_window_dates=True,
            extra_fields={
                "nonferrous_holding_share": float((result["holding"] == NONFERROUS_CODE).fillna(False).mean())
            },
        )
        append_variant_result(
            rows,
            compare_df,
            None,
            strategy=name,
            result=result,
            summary=row,
            strategy_field="variant",
            nav_column=name,
        )
    summary = finalize_baseline_diff_summary(
        rows,
        baseline_field="variant",
        baseline_value="base_pool",
        metric_mappings=(
            ("total_return", "return_diff"),
            ("annualized_return", "annualized_diff"),
            ("sharpe_rf0", "sharpe_diff"),
            ("max_drawdown", "mdd_diff"),
            ("trade_count", "trade_diff"),
        ),
    )

    plot_lines = [
        ("base_pool", "base_pool", 2.3),
        ("nonferrous_risk", "nonferrous_risk", 1.8),
        ("nonferrous_cap_70", "nonferrous_cap_70", 1.8),
        ("nonferrous_abs_08", "nonferrous_abs_08", 1.8),
        ("nonferrous_margin_03", "nonferrous_margin_03", 1.8),
        ("nonferrous_margin_03_cap_70", "nonferrous_margin_03_cap_70", 1.8),
    ]
    save_plot_and_print_variant_compare_outputs(
        OUTPUT_DIR,
        summary,
        compare_df,
        plot_filename="comparison.png",
        title="Nonferrous Metals ETF Guard Variants",
        lines=[line for line in plot_lines if line[0] in compare_df.columns],
        benchmark_label="HS300 ETF",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
