#!/usr/bin/env python3
"""基于当前正式基线，对比资源/能源补位及其叠加极端现金停车。"""

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

from archive_data_loaders import (
    build_archive_flat_output_dir,
    build_cash_tail_mask,
    build_risk_defensive_momentum_frames,
    extract_result_target_weights,
    filter_selected_price_columns,
    load_required_candidate_prices,
)

try:
    from ...official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    filter_available_plot_lines,
    finalize_baseline_diff_summary,
    save_plot_and_print_variant_compare_outputs,
)
from goal_optimization_common import load_market_volume_proxy
from hs300_regime_common import run_target_weights_strategy, summarize
from candidate_pool_common import ETF_159930, ETF_510410
from pool_change_helpers import apply_pool_change
from archive_strategy_common import (
    DEFAULT_BASELINE_DROP_CODES,
    DEFAULT_FEE_RATE,
    DEFAULT_LOOKBACK,
    DEFAULT_SLIPPAGE_RATE,
    build_default_strategy_params,
    fetch_histories,
    load_default_strategy_backtest_pool,
    run_default_strategy_with_params,
)


OUTPUT_DIR = build_archive_flat_output_dir("compare_resource_energy_common_sense")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于当前正式基线，对比资源/能源补位及其叠加极端现金停车。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取缺失历史，而不是优先复用本地缓存。")
    parser.add_argument("--base-only", action="store_true", help="只运行当前正式基线，不补抓资源/能源历史。")
    return parser.parse_args()


def load_combined_prices(selected: pd.DataFrame, *, years: int, refresh: bool) -> pd.DataFrame:
    _, prices = load_required_candidate_prices(
        selected.iloc[0:0].copy(),
        selected,
        years=years,
        refresh=refresh,
        fetch_fn=fetch_histories,
        label="resource/energy candidates",
        failure_prefix="failed to prepare resource/energy candidate histories",
    )
    prices = filter_selected_price_columns(prices, selected)
    prices = prices.loc[prices.index >= pd.Timestamp("2012-01-01")].copy()
    prices = prices.dropna(how="any")
    if prices.empty:
        raise RuntimeError("resource/energy comparison has no overlapping non-null price window")
    return prices

def evaluate_strategy(
    name: str,
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    market_proxy: pd.DataFrame,
    risk_codes: list[str],
    defensive_codes: list[str],
    effective_drop_codes: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    result, trades = run_default_strategy_with_params(
        prices=prices,
        selected=selected,
        params=build_default_strategy_params(
            drop_codes=effective_drop_codes,
            risk_codes=risk_codes,
            defensive_codes=defensive_codes,
        ),
        fee_rate=DEFAULT_FEE_RATE,
        slippage_rate=DEFAULT_SLIPPAGE_RATE,
        market_proxy=market_proxy,
    )
    row = summarize(result, trades, selected)
    if name == "base_pool":
        result = apply_official_baseline_nav_anchor(result)
        row = summarize(result, trades, selected)
    row["cash_days"] = int((result["exposure"] <= 1e-12).sum())
    return row, result


def evaluate_cash_tail_variant(
    name: str,
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    base_result: pd.DataFrame,
    risk_codes: list[str],
    defensive_codes: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    target_weights = extract_result_target_weights(base_result, prices)
    risk_momentum, defensive_momentum = build_risk_defensive_momentum_frames(
        prices,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
        lookback=DEFAULT_LOOKBACK,
    )
    cash_mask = build_cash_tail_mask(risk_momentum, defensive_momentum)
    target_weights.loc[cash_mask, :] = 0.0
    result, trades = run_target_weights_strategy(
        prices=prices,
        selected=selected,
        target_weights=target_weights,
        signal_momentum=base_result["current_momentum"],
        fee_rate=DEFAULT_FEE_RATE,
        slippage_rate=DEFAULT_SLIPPAGE_RATE,
    )
    row = summarize(result, trades, selected)
    row["cash_days"] = int((result["exposure"] <= 1e-12).sum())
    row["cash_days_triggered"] = int(cash_mask.sum())
    return row, result


def build_combined_selected(base_selected: pd.DataFrame) -> pd.DataFrame:
    """一次性补齐本脚本涉及的额外候选，避免每个变体重复拉历史。"""
    extra_selected = pd.DataFrame([ETF_159930, ETF_510410])
    combined = pd.concat([base_selected, extra_selected], ignore_index=True)
    return combined.drop_duplicates(subset="code", keep="first").reset_index(drop=True)


def main() -> int:
    args = parse_args()

    base_selected = load_default_strategy_backtest_pool().copy()
    market_proxy = load_market_volume_proxy(years=args.years, refresh=False)
    changes = [
        {"pool": "base_pool", "kind": "base"},
        {"pool": "plus_resource_510410", "kind": "add", "candidate_kind": "risk", "candidate": ETF_510410},
        {"pool": "plus_energy_159930", "kind": "add", "candidate_kind": "risk", "candidate": ETF_159930},
    ]
    if args.base_only:
        changes = changes[:1]

    combined_selected = base_selected if args.base_only else build_combined_selected(base_selected)
    combined_prices = load_combined_prices(combined_selected, years=args.years, refresh=args.refresh)

    rows: list[dict[str, object]] = []
    nav_compare = build_compare_frame(combined_prices)

    for change in changes:
        selected, risk_codes, defensive_codes, effective_drop_codes = apply_pool_change(
            base_selected,
            list(DEFAULT_BASELINE_DROP_CODES),
            change,
        )
        selected_codes = [code for code in selected["code"].astype(str) if code in combined_prices.columns]
        prices = combined_prices[selected_codes].copy()

        row, result = evaluate_strategy(
            str(change["pool"]),
            selected,
            prices,
            market_proxy,
            risk_codes,
            defensive_codes,
            effective_drop_codes,
        )
        append_variant_result(
            rows,
            nav_compare,
            None,
            strategy=str(change["pool"]),
            result=result,
            summary=row,
            nav_column=str(change["pool"]),
        )

        if str(change["pool"]) != "base_pool":
            cash_row, cash_result = evaluate_cash_tail_variant(
                f"{change['pool']}__cash_tail",
                selected,
                prices,
                result,
                risk_codes,
                defensive_codes,
            )
            append_variant_result(
                rows,
                nav_compare,
                None,
                strategy=f"{change['pool']}__cash_tail",
                result=cash_result,
                summary=cash_row,
                nav_column=f"{change['pool']}__cash_tail",
            )

    summary_df = finalize_baseline_diff_summary(
        rows,
        baseline_field="strategy",
        baseline_value="base_pool",
        metric_mappings=(
            ("annualized_return", "annualized_diff"),
            ("sharpe_rf0", "sharpe_diff"),
            ("max_drawdown", "max_drawdown_diff"),
            ("max_drawdown_integral", "max_drawdown_integral_diff"),
            ("trade_count", "trade_diff"),
            ("cash_days", "cash_days_diff"),
        ),
    )

    plot_lines = [
        ("base_pool", "Base Pool", 2.2),
        ("plus_resource_510410", "+510410", 1.8),
        ("plus_resource_510410__cash_tail", "+510410 Cash Tail", 1.8),
        ("plus_energy_159930", "+159930", 1.8),
        ("plus_energy_159930__cash_tail", "+159930 Cash Tail", 1.8),
    ]
    available_plot_lines = filter_available_plot_lines(nav_compare, plot_lines)
    save_plot_and_print_variant_compare_outputs(
        OUTPUT_DIR,
        summary_df,
        nav_compare,
        plot_filename="comparison.png",
        title="Resource Energy Common Sense Comparison",
        lines=available_plot_lines,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
