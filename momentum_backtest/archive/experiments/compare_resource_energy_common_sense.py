#!/usr/bin/env python3
"""基于当前正式基线，对比资源/能源补位及其叠加极端现金停车。"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ..runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from compare_current_best_pool_additions import apply_pool_change
from compare_goal_optimizations import load_market_volume_proxy
from compare_hs300_regime_fixes import run_target_weights_strategy, summarize
from run_backtest import (
    DEFAULT_BASELINE_DROP_CODES,
    DEFAULT_FEE_RATE,
    DEFAULT_LOOKBACK,
    DEFAULT_SLIPPAGE_RATE,
    build_default_strategy_params,
    fetch_histories,
    load_default_strategy_backtest_pool,
    run_default_strategy_with_params,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_resource_energy_common_sense")
ETF_159930 = {"theme": "能源", "code": "159930", "name": "能源ETF", "sina_symbol": "sz159930"}
ETF_510410 = {"theme": "资源", "code": "510410", "name": "资源ETF", "sina_symbol": "sh510410"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于当前正式基线，对比资源/能源补位及其叠加极端现金停车。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    return parser.parse_args()


def extract_target_weights(result: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    weight_cols = [col for col in result.columns if col.startswith("target_weight_")]
    target_weights = result[weight_cols].copy()
    target_weights.columns = [col.removeprefix("target_weight_") for col in weight_cols]
    return target_weights.reindex(index=prices.index, columns=prices.columns, fill_value=0.0).fillna(0.0)


def build_momentum_frames(prices: pd.DataFrame, risk_codes: list[str], defensive_codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum = prices / prices.shift(DEFAULT_LOOKBACK) - 1
    risk_momentum = momentum[[code for code in risk_codes if code in momentum.columns]].copy()
    defensive_momentum = momentum[[code for code in defensive_codes if code in momentum.columns]].copy()
    return risk_momentum, defensive_momentum


def build_cash_tail_mask(risk_momentum: pd.DataFrame, defensive_momentum: pd.DataFrame) -> pd.Series:
    valid_risk_count = risk_momentum.notna().sum(axis=1)
    risk_positive_count = (risk_momentum > 0).sum(axis=1)
    risk_max = risk_momentum.max(axis=1, skipna=True)
    defensive_max = defensive_momentum.max(axis=1, skipna=True)
    return (
        (valid_risk_count >= max(len(risk_momentum.columns) - 1, 1))
        & (risk_positive_count <= 1)
        & (risk_max <= 0.03)
        & (defensive_max <= -0.005)
    ).fillna(False)


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
    row["strategy"] = name
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
    target_weights = extract_target_weights(base_result, prices)
    risk_momentum, defensive_momentum = build_momentum_frames(prices, risk_codes, defensive_codes)
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
    row["strategy"] = name
    row["cash_days"] = int((result["exposure"] <= 1e-12).sum())
    row["cash_days_triggered"] = int(cash_mask.sum())
    return row, result


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_selected = load_default_strategy_backtest_pool().copy()
    market_proxy = load_market_volume_proxy(years=args.years, refresh=False)
    changes = [
        {"pool": "base_pool", "kind": "base"},
        {"pool": "plus_resource_510410", "kind": "add", "candidate_kind": "risk", "candidate": ETF_510410},
        {"pool": "plus_energy_159930", "kind": "add", "candidate_kind": "risk", "candidate": ETF_159930},
    ]

    rows: list[dict[str, object]] = []
    nav_compare = pd.DataFrame()

    for change in changes:
        selected, risk_codes, defensive_codes, effective_drop_codes = apply_pool_change(
            base_selected,
            list(DEFAULT_BASELINE_DROP_CODES),
            change,
        )
        prices = fetch_histories(selected, years=args.years)
        prices = prices.loc[prices.index >= pd.Timestamp("2012-01-01")].copy()

        row, result = evaluate_strategy(
            str(change["pool"]),
            selected,
            prices,
            market_proxy,
            risk_codes,
            defensive_codes,
            effective_drop_codes,
        )
        rows.append(row)
        nav_compare[str(change["pool"])] = result["nav"]

        if str(change["pool"]) != "base_pool":
            cash_row, cash_result = evaluate_cash_tail_variant(
                f"{change['pool']}__cash_tail",
                selected,
                prices,
                result,
                risk_codes,
                defensive_codes,
            )
            rows.append(cash_row)
            nav_compare[f"{change['pool']}__cash_tail"] = cash_result["nav"]

    summary_df = pd.DataFrame(rows)
    base = summary_df[summary_df["strategy"] == "base_pool"].iloc[0]
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(base["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(base["sharpe_rf0"])
    summary_df["max_drawdown_diff"] = summary_df["max_drawdown"] - float(base["max_drawdown"])
    summary_df["max_drawdown_integral_diff"] = summary_df["max_drawdown_integral"] - float(base["max_drawdown_integral"])
    summary_df["trade_diff"] = summary_df["trade_count"] - int(base["trade_count"])
    summary_df["cash_days_diff"] = summary_df["cash_days"] - int(base["cash_days"])

    write_dataframe_csv_atomic(summary_df, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(nav_compare, OUTPUT_DIR / "nav_compare.csv")

    print(summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
