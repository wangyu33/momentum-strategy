#!/usr/bin/env python3
"""基于当前正式基线，评估几类常识性增强。"""

from __future__ import annotations

from pathlib import Path

try:
    from ..runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from compare_hs300_regime_fixes import run_target_weights_strategy, summarize
from run_backtest import (
    CORE_OUTPUT_DIR,
    DEFAULT_FEE_RATE,
    DEFAULT_LOOKBACK,
    DEFAULT_SLIPPAGE_RATE,
    DEFENSIVE_CODES,
    RISK_CODES,
    build_default_strategy_params,
    load_default_strategy_backtest_pool,
    run_default_strategy_with_params,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_common_sense_enhancements")


def load_core_prices() -> pd.DataFrame:
    prices = pd.read_csv(CORE_OUTPUT_DIR / "prices.csv", parse_dates=["date"]).set_index("date")
    return prices.sort_index()


def extract_target_weights(result: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    weight_cols = [col for col in result.columns if col.startswith("target_weight_")]
    target_weights = result[weight_cols].copy()
    target_weights.columns = [col.removeprefix("target_weight_") for col in weight_cols]
    return target_weights.reindex(index=prices.index, columns=prices.columns, fill_value=0.0).fillna(0.0)


def build_risk_defensive_momentum(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum = prices / prices.shift(DEFAULT_LOOKBACK) - 1
    risk_momentum = momentum[[code for code in RISK_CODES if code in momentum.columns]].copy()
    defensive_momentum = momentum[[code for code in DEFENSIVE_CODES if code in momentum.columns]].copy()
    return risk_momentum, defensive_momentum


def build_cash_overlay_mask(
    risk_momentum: pd.DataFrame,
    defensive_momentum: pd.DataFrame,
    *,
    risk_pos_limit: int,
    risk_max_cut: float,
    defensive_max_cut: float,
) -> pd.Series:
    valid_risk_count = risk_momentum.notna().sum(axis=1)
    risk_positive_count = (risk_momentum > 0).sum(axis=1)
    risk_max = risk_momentum.max(axis=1, skipna=True)
    defensive_max = defensive_momentum.max(axis=1, skipna=True)
    return (
        (valid_risk_count >= max(len(risk_momentum.columns) - 1, 1))
        & (risk_positive_count <= risk_pos_limit)
        & (risk_max <= risk_max_cut)
        & (defensive_max <= defensive_max_cut)
    ).fillna(False)


def evaluate_variant(
    name: str,
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    params: dict[str, object] | None = None,
    target_weights: pd.DataFrame | None = None,
    signal_momentum: pd.Series | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    if target_weights is None:
        if params is None:
            raise ValueError("params and target_weights cannot both be None")
        result, trades = run_default_strategy_with_params(
            prices=prices,
            selected=selected,
            params=params,
            fee_rate=DEFAULT_FEE_RATE,
            slippage_rate=DEFAULT_SLIPPAGE_RATE,
        )
    else:
        if signal_momentum is None:
            raise ValueError("signal_momentum is required when target_weights is provided")
        result, trades = run_target_weights_strategy(
            prices=prices,
            selected=selected,
            target_weights=target_weights,
            signal_momentum=signal_momentum,
            fee_rate=DEFAULT_FEE_RATE,
            slippage_rate=DEFAULT_SLIPPAGE_RATE,
        )

    row = summarize(result, trades, selected)
    row["strategy"] = name
    row["cash_days"] = int((result["exposure"] <= 1e-12).sum())
    row["latest_portfolio"] = row.get("latest_portfolio", "")
    return row, result


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = load_default_strategy_backtest_pool().copy()
    prices = load_core_prices()

    base_params = build_default_strategy_params()
    baseline_row, baseline_result = evaluate_variant(
        "baseline",
        prices,
        selected,
        params=base_params,
    )

    rows: list[dict[str, object]] = [baseline_row]
    nav_compare = pd.DataFrame(index=baseline_result.index)
    nav_compare["baseline"] = baseline_result["nav"]

    # 轻防抖：只在领先幅度足够大时才切主信号，避免震荡市来回切换。
    for leader_margin in (0.005, 0.010, 0.015):
        params = build_default_strategy_params()
        params["signal_leader_margin"] = leader_margin
        name = f"leader_margin_{int(round(leader_margin * 1000)):03d}"
        row, result = evaluate_variant(name, prices, selected, params=params)
        row["leader_margin"] = leader_margin
        rows.append(row)
        nav_compare[name] = result["nav"]

    baseline_target_weights = extract_target_weights(baseline_result, prices)
    baseline_signal_momentum = baseline_result["current_momentum"].copy()
    risk_momentum, defensive_momentum = build_risk_defensive_momentum(prices)

    # 极端现金停车：只有在风险池整体走弱且当前防守资产也不强时，才退到现金。
    cash_specs = [
        ("cash_tail_soft", 1, 0.05, 0.00),
        ("cash_tail_mid", 1, 0.03, -0.005),
        ("cash_tail_strict", 0, 0.00, 0.00),
    ]
    for name, risk_pos_limit, risk_max_cut, defensive_max_cut in cash_specs:
        mask = build_cash_overlay_mask(
            risk_momentum,
            defensive_momentum,
            risk_pos_limit=risk_pos_limit,
            risk_max_cut=risk_max_cut,
            defensive_max_cut=defensive_max_cut,
        )
        target_weights = baseline_target_weights.copy()
        target_weights.loc[mask, :] = 0.0
        row, result = evaluate_variant(
            name,
            prices,
            selected,
            target_weights=target_weights,
            signal_momentum=baseline_signal_momentum,
        )
        row["risk_pos_limit"] = risk_pos_limit
        row["risk_max_cut"] = risk_max_cut
        row["defensive_max_cut"] = defensive_max_cut
        row["cash_days_triggered"] = int(mask.sum())
        rows.append(row)
        nav_compare[name] = result["nav"]

    summary_df = pd.DataFrame(rows)
    base = summary_df[summary_df["strategy"] == "baseline"].iloc[0]
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
