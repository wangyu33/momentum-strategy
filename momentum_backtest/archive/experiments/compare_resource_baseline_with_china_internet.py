#!/usr/bin/env python3
"""Compare China Internet variants using resource_abs08 as the baseline.

This script intentionally preserves the historical resource_abs08 research baseline and
must not be anchored to the official 28.2691 formal baseline chain.
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

from archive_data_loaders import build_archive_flat_output_dir, load_workspace_selected_prices
from candidate_pool_common import BASE_DEFENSIVE_CODES, BASE_RISK_CODES, ETF_513050
from dual_momentum_helpers import summarize_variant_result as summarize_exposure_variant_result
from resource_guard_common import RESOURCE_CODE
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
    RESEARCH_OUTPUT_DIR,
    fetch_histories,
    load_fixed_etf_pool,
)

matplotlib.use("Agg")

CHINA_INTERNET_CODE = "513050"
OUTPUT_DIR = build_archive_flat_output_dir("compare_resource_baseline_with_china_internet")
RESOURCE_ABS08_OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "resource_abs08"
CHINA_INTERNET_CAP70_OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "china_internet_cap70"
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare China Internet variants on top of resource_abs08 baseline.")
    parser.add_argument("--years", type=int, default=6, help="History years to request before overlap alignment.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="重新抓取历史数据，而不是优先复用本地 research/cache 价格。")
    return parser.parse_args()


def load_resource_abs08_selected() -> pd.DataFrame:
    """优先复用历史 resource_abs08 的候选池，确保比较口径不漂移。"""
    selected_path = RESOURCE_ABS08_OUTPUT_DIR / "selected_etfs.csv"
    if selected_path.exists():
        return pd.read_csv(selected_path, dtype={"code": str})
    return pd.concat([load_fixed_etf_pool(), pd.DataFrame([ETF_513050])], ignore_index=True)


def load_prices(selected: pd.DataFrame, *, years: int, refresh: bool) -> pd.DataFrame:
    prices = load_workspace_selected_prices(
        selected,
        years=years,
        refresh=refresh,
        fetch_fn=fetch_histories,
        missing_label="required resource/china histories",
    )
    return prices.dropna(how="any")


def choose_signal(
    risk_scores: pd.Series,
    defensive_scores: pd.Series,
    china_mode: str,
) -> tuple[object, float, float]:
    valid_risk = risk_scores.dropna().sort_values(ascending=False)
    valid_def = defensive_scores.dropna().sort_values(ascending=False)
    if valid_risk.empty:
        r_asset = pd.NA
        r_score = float("nan")
    else:
        r_asset = valid_risk.index[0]
        r_score = float(valid_risk.iloc[0])
    d_asset = valid_def.index[0] if not valid_def.empty else pd.NA
    d_score = float(valid_def.iloc[0]) if not valid_def.empty else float("nan")

    candidate_asset = r_asset
    candidate_score = r_score
    if pd.notna(candidate_asset) and str(candidate_asset) == RESOURCE_CODE and pd.notna(candidate_score) and candidate_score <= 0.08:
        if len(valid_risk) >= 2:
            candidate_asset = valid_risk.index[1]
            candidate_score = float(valid_risk.iloc[1])

    if pd.notna(candidate_asset) and str(candidate_asset) == CHINA_INTERNET_CODE:
        if china_mode == "exclude_to_second":
            if len(valid_risk) >= 2:
                candidate_asset = valid_risk.index[1]
                candidate_score = float(valid_risk.iloc[1])
            else:
                candidate_asset = pd.NA
                candidate_score = float("nan")
        elif china_mode == "reverse_to_defensive":
            candidate_asset = d_asset
            candidate_score = float(d_score) if pd.notna(d_score) else float("nan")

    if pd.notna(candidate_score) and candidate_score > DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD:
        exposure = 1.0
        if pd.notna(candidate_asset) and str(candidate_asset) == CHINA_INTERNET_CODE and china_mode == "cap70":
            exposure = 0.70
        return candidate_asset, exposure, float(candidate_score)
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
    risk_codes: list[str],
    china_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum = prices / prices.shift(lookback) - 1
    risk_scores = momentum[[code for code in risk_codes if code in prices.columns]]
    defensive_scores = momentum[[code for code in BASE_DEFENSIVE_CODES if code in prices.columns]]

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")
    for dt_idx in prices.index:
        asset, exposure, score = choose_signal(
            risk_scores=risk_scores.loc[dt_idx],
            defensive_scores=defensive_scores.loc[dt_idx],
            china_mode=china_mode,
        )
        signal.loc[dt_idx] = asset
        target_exposure.loc[dt_idx] = exposure
        current_momentum.loc[dt_idx] = score

    base_result, _ = run_signal_strategy(prices, selected, signal, target_exposure, current_momentum, fee_rate, slippage_rate)
    adjusted_exposure = target_exposure.copy()
    risk_code_set = [code for code in risk_scores.columns if code in prices.columns]
    overheat_mask = (
        signal.isin(risk_code_set)
        & (adjusted_exposure > DEFAULT_OVERHEAT_MAX_EXPOSURE)
        & (base_result["drawdown"] >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (current_momentum >= DEFAULT_OVERHEAT_MOMENTUM_CUT)
    )
    adjusted_exposure.loc[overheat_mask] = DEFAULT_OVERHEAT_MAX_EXPOSURE
    extreme_mask = (
        signal.isin(risk_code_set)
        & (adjusted_exposure > DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE)
        & (base_result["drawdown"] >= DEFAULT_OVERHEAT_DRAWDOWN_CUT)
        & (current_momentum >= DEFAULT_OVERHEAT_HIGH_MOMENTUM_CUT)
    )
    adjusted_exposure.loc[extreme_mask] = DEFAULT_OVERHEAT_HIGH_MAX_EXPOSURE
    return run_signal_strategy(prices, selected, signal, adjusted_exposure, current_momentum, fee_rate, slippage_rate)


def main() -> int:
    args = parse_args()
    base_pool = load_resource_abs08_selected()
    if CHINA_INTERNET_CODE not in base_pool["code"].astype(str).tolist():
        selected = pd.concat([base_pool, pd.DataFrame([ETF_513050])], ignore_index=True)
    else:
        selected = base_pool.copy()
    prices = load_prices(selected, years=args.years, refresh=args.refresh)

    resource_risk_codes = BASE_RISK_CODES + [RESOURCE_CODE]
    resource_ci_risk_codes = BASE_RISK_CODES + [RESOURCE_CODE, CHINA_INTERNET_CODE]
    variants = [
        ("resource_abs08_baseline", resource_risk_codes, "normal"),
        ("resource_abs08_plus_ci", resource_ci_risk_codes, "normal"),
        ("resource_abs08_plus_ci_cap70", resource_ci_risk_codes, "cap70"),
        ("resource_abs08_ci_exclude_to_second", resource_ci_risk_codes, "exclude_to_second"),
        ("resource_abs08_ci_reverse_to_defensive", resource_ci_risk_codes, "reverse_to_defensive"),
    ]

    rows: list[dict[str, object]] = []
    compare_df = None
    for name, risk_codes, china_mode in variants:
        result, trades = run_variant(
            prices=prices,
            selected=selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            risk_codes=risk_codes,
            china_mode=china_mode,
        )
        row = summarize_exposure_variant_result(
            result,
            trades,
            selected,
            include_window_dates=True,
            extra_fields={
                "china_internet_holding_share": float((result["holding"] == CHINA_INTERNET_CODE).fillna(False).mean()),
                "resource_holding_share": float((result["holding"] == RESOURCE_CODE).fillna(False).mean()),
            },
        )
        compare_df = ensure_compare_frame(compare_df, prices, index=result.index)
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
        baseline_value="resource_abs08_baseline",
        metric_mappings=(
            ("total_return", "return_diff"),
            ("annualized_return", "annualized_diff"),
            ("sharpe_rf0", "sharpe_diff"),
            ("max_drawdown", "mdd_diff"),
            ("trade_count", "trade_diff"),
        ),
    )

    save_plot_and_print_variant_compare_outputs(
        OUTPUT_DIR,
        summary,
        compare_df,
        plot_filename="comparison.png",
        title="Resource Abs08 Baseline vs China Internet Variants",
        lines=(
            ("resource_abs08_baseline", "resource_abs08_baseline", 2.0),
            ("resource_abs08_plus_ci", "resource_abs08_plus_ci", 1.8),
            ("resource_abs08_plus_ci_cap70", "resource_abs08_plus_ci_cap70", 1.8),
            ("resource_abs08_ci_exclude_to_second", "resource_abs08_ci_exclude_to_second", 1.8),
            ("resource_abs08_ci_reverse_to_defensive", "resource_abs08_ci_reverse_to_defensive", 1.8),
        ),
        benchmark_label="HS300 ETF",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
