#!/usr/bin/env python3
"""Compare adding dividend-style defensive ETFs to the current threshold dual momentum pool."""

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

try:
    from ...official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

from archive_data_loaders import (
    build_archive_flat_output_dir,
    dedupe_selected_pool,
    load_required_candidate_prices,
)
from dual_momentum_helpers import run_exposure_strategy, summarize
from variant_compare_helpers import (
    append_variant_result,
    ensure_compare_frame,
    filter_available_plot_lines,
    finalize_baseline_diff_summary,
    save_variant_compare_artifacts,
)
from archive_strategy_common import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    RISK_CODES,
    fetch_histories,
    load_fixed_etf_pool,
)

matplotlib.use("Agg")


OUTPUT_DIR = build_archive_flat_output_dir("compare_dividend_defensive_additions")


ETF_510880 = {"theme": "红利ETF", "code": "510880", "name": "红利ETF华泰柏瑞", "sina_symbol": "sh510880"}
ETF_520550 = {
    "theme": "港股红利低波ETF",
    "code": "520550",
    "name": "招商恒生港股通高股息低波动ETF",
    "sina_symbol": "sh520550",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare adding 510880 / 520550 into the defensive bucket.")
    parser.add_argument("--years", type=int, default=10, help="Backtest years.")
    parser.add_argument("--lookback", type=int, default=25, help="Momentum lookback window.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Fetch fresh histories instead of preferring cached core prices.")
    parser.add_argument("--base-only", action="store_true", help="Only run the cached base_pool variant.")
    return parser.parse_args()


def run_custom_threshold_dual(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    defensive_codes: list[str],
    lookback: int,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum = prices / prices.shift(lookback) - 1
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    defensive_codes = [code for code in defensive_codes if code in prices.columns]

    risk_mom = momentum[risk_codes]
    defensive_mom = momentum[defensive_codes]
    risk_winner = risk_mom.idxmax(axis=1, skipna=True)
    risk_best = risk_mom.max(axis=1, skipna=True)
    defensive_winner = defensive_mom.idxmax(axis=1, skipna=True)
    defensive_best = defensive_mom.max(axis=1, skipna=True)

    signal = pd.Series(index=prices.index, dtype="object", name="signal")
    target_exposure = pd.Series(index=prices.index, dtype="float64", name="target_exposure")
    current_momentum = pd.Series(index=prices.index, dtype="float64", name="current_momentum")

    for dt_idx in prices.index:
        r_asset = risk_winner.loc[dt_idx]
        r_score = risk_best.loc[dt_idx]
        d_asset = defensive_winner.loc[dt_idx]
        d_score = defensive_best.loc[dt_idx]

        if pd.isna(r_score) and pd.isna(d_score):
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")
        elif pd.notna(r_score) and r_score > DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD:
            signal.loc[dt_idx] = r_asset
            target_exposure.loc[dt_idx] = 1.0
            current_momentum.loc[dt_idx] = float(r_score)
        elif pd.notna(r_score) and r_score > 0 and pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT
            current_momentum.loc[dt_idx] = float(r_score)
        elif pd.notna(d_asset):
            signal.loc[dt_idx] = d_asset
            target_exposure.loc[dt_idx] = 1.0 if pd.notna(d_score) and d_score > 0 else 0.0
            current_momentum.loc[dt_idx] = float(d_score) if pd.notna(d_score) else float("nan")
        else:
            signal.loc[dt_idx] = pd.NA
            target_exposure.loc[dt_idx] = 0.0
            current_momentum.loc[dt_idx] = float("nan")

    return run_exposure_strategy(
        prices,
        selected,
        signal,
        target_exposure,
        momentum,
        fee_rate,
        slippage_rate,
        current_momentum=current_momentum,
    )


def load_candidate_prices(
    selected: pd.DataFrame,
    *,
    base_pool: pd.DataFrame,
    years: int,
    refresh: bool,
    additions: list[dict[str, str]],
) -> pd.DataFrame:
    selected = dedupe_selected_pool(selected)
    _, prices = load_required_candidate_prices(
        base_pool,
        additions,
        years=years,
        refresh=refresh,
        fetch_fn=fetch_histories,
        label="dividend defensive candidates",
        failure_prefix="failed to load added dividend ETF history",
    )
    selected_codes = selected["code"].astype(str).tolist()
    return prices[[code for code in selected_codes if code in prices.columns]].copy()


def main() -> int:
    args = parse_args()

    base_pool = load_fixed_etf_pool()
    base_defensive = ["511580", "518880", "512890"]

    candidates = [
        ("base_pool", [], base_defensive),
        ("plus_510880", [ETF_510880], base_defensive + ["510880"]),
        ("plus_520550", [ETF_520550], base_defensive + ["520550"]),
        ("plus_both", [ETF_510880, ETF_520550], base_defensive + ["510880", "520550"]),
    ]
    if args.base_only:
        candidates = candidates[:1]

    rows: list[dict[str, float | int | str]] = []
    compare_df = None
    named_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, additions, defensive_codes in candidates:
        selected = pd.concat([base_pool, pd.DataFrame(additions)], ignore_index=True) if additions else base_pool.copy()
        selected = dedupe_selected_pool(selected)
        prices = load_candidate_prices(
            selected,
            base_pool=base_pool,
            years=args.years,
            refresh=args.refresh,
            additions=additions,
        )
        result, trades = run_custom_threshold_dual(
            prices,
            selected,
            defensive_codes=defensive_codes,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        if name == "base_pool":
            result = apply_official_baseline_nav_anchor(result)
        compare_df = ensure_compare_frame(compare_df, prices, index=result.index)
        append_variant_result(
            rows,
            compare_df,
            None,
            strategy=name,
            result=result,
            summary=summarize(result, trades, selected),
            strategy_field="pool",
            nav_column=f"{name}_nav",
            extra_fields={"defensive_codes": ",".join(defensive_codes)},
        )
        named_outputs[name] = (result, trades)

    summary = finalize_baseline_diff_summary(
        rows,
        baseline_field="pool",
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
        ("base_pool_nav", "Base Pool", 2.2),
        ("plus_510880_nav", "+510880", 1.8),
        ("plus_520550_nav", "+520550", 1.8),
        ("plus_both_nav", "+Both", 1.8),
    ]
    available_plot_lines = filter_available_plot_lines(compare_df, plot_lines)

    save_variant_compare_artifacts(
        OUTPUT_DIR,
        summary,
        compare_df,
        named_outputs=named_outputs,
        plot_filename="comparison.png",
        title="Dividend Defensive Additions Comparison",
        lines=available_plot_lines,
        benchmark_label="HS300",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
