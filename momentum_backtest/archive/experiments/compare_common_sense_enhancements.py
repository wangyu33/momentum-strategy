#!/usr/bin/env python3
"""基于当前正式基线，评估几类常识性增强。"""

from __future__ import annotations

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
)

try:
    from ...official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

try:
    from ...run_backtest import build_official_baseline_summary
except ImportError:
    from run_backtest import build_official_baseline_summary

from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    finalize_baseline_diff_summary,
    save_plot_and_print_variant_compare_outputs,
)
from hs300_regime_common import run_target_weights_strategy, summarize
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_LOOKBACK,
    DEFAULT_SLIPPAGE_RATE,
    DEFENSIVE_CODES,
    RISK_CODES,
    build_default_strategy_params,
    load_core_selected_and_prices,
    load_default_strategy_backtest_pool,
    run_default_strategy_with_params,
)


OUTPUT_DIR = build_archive_flat_output_dir("compare_common_sense_enhancements")


def evaluate_variant(
    name: str,
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    params: dict[str, object] | None = None,
    target_weights: pd.DataFrame | None = None,
    signal_momentum: pd.Series | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
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
    row["cash_days"] = int((result["exposure"] <= 1e-12).sum())
    row["latest_portfolio"] = row.get("latest_portfolio", "")
    return row, result, trades


def main() -> int:
    selected = load_default_strategy_backtest_pool().copy()
    _, prices = load_core_selected_and_prices()

    base_params = build_default_strategy_params()
    baseline_row, baseline_result, baseline_trades = evaluate_variant(
        "baseline",
        prices,
        selected,
        params=base_params,
    )
    baseline_result = apply_official_baseline_nav_anchor(baseline_result)
    baseline_row = build_official_baseline_summary(
        baseline_result,
        baseline_trades,
        selected=selected,
        include_max_drawdown_integral=True,
    )
    baseline_row["cash_days"] = int((baseline_result["exposure"] <= 1e-12).sum())

    rows: list[dict[str, object]] = []
    nav_compare = build_compare_frame(prices)
    append_variant_result(
        rows,
        nav_compare,
        None,
        strategy="baseline",
        result=baseline_result,
        summary=baseline_row,
        nav_column="baseline",
    )

    # 轻防抖：只在领先幅度足够大时才切主信号，避免震荡市来回切换。
    for leader_margin in (0.005, 0.010, 0.015):
        params = build_default_strategy_params()
        params["signal_leader_margin"] = leader_margin
        name = f"leader_margin_{int(round(leader_margin * 1000)):03d}"
        row, result, _ = evaluate_variant(name, prices, selected, params=params)
        append_variant_result(
            rows,
            nav_compare,
            None,
            strategy=name,
            result=result,
            summary=row,
            nav_column=name,
            extra_fields={"leader_margin": leader_margin},
        )

    baseline_target_weights = extract_result_target_weights(baseline_result, prices)
    baseline_signal_momentum = baseline_result["current_momentum"].copy()
    risk_momentum, defensive_momentum = build_risk_defensive_momentum_frames(
        prices,
        risk_codes=RISK_CODES,
        defensive_codes=DEFENSIVE_CODES,
        lookback=DEFAULT_LOOKBACK,
    )

    # 极端现金停车：只有在风险池整体走弱且当前防守资产也不强时，才退到现金。
    cash_specs = [
        ("cash_tail_soft", 1, 0.05, 0.00),
        ("cash_tail_mid", 1, 0.03, -0.005),
        ("cash_tail_strict", 0, 0.00, 0.00),
    ]
    for name, risk_pos_limit, risk_max_cut, defensive_max_cut in cash_specs:
        mask = build_cash_tail_mask(
            risk_momentum,
            defensive_momentum,
            risk_pos_limit=risk_pos_limit,
            risk_max_cut=risk_max_cut,
            defensive_max_cut=defensive_max_cut,
        )
        target_weights = baseline_target_weights.copy()
        target_weights.loc[mask, :] = 0.0
        row, result, _ = evaluate_variant(
            name,
            prices,
            selected,
            target_weights=target_weights,
            signal_momentum=baseline_signal_momentum,
        )
        append_variant_result(
            rows,
            nav_compare,
            None,
            strategy=name,
            result=result,
            summary=row,
            nav_column=name,
            extra_fields={
                "risk_pos_limit": risk_pos_limit,
                "risk_max_cut": risk_max_cut,
                "defensive_max_cut": defensive_max_cut,
                "cash_days_triggered": int(mask.sum()),
            },
        )

    summary_df = finalize_baseline_diff_summary(
        rows,
        baseline_field="strategy",
        baseline_value="baseline",
        metric_mappings=(
            ("annualized_return", "annualized_diff"),
            ("sharpe_rf0", "sharpe_diff"),
            ("max_drawdown", "max_drawdown_diff"),
            ("max_drawdown_integral", "max_drawdown_integral_diff"),
            ("trade_count", "trade_diff"),
            ("cash_days", "cash_days_diff"),
        ),
    )

    save_plot_and_print_variant_compare_outputs(
        OUTPUT_DIR,
        summary_df,
        nav_compare,
        plot_filename="comparison.png",
        title="Common Sense Enhancements Comparison",
        lines=(
            ("baseline", "Baseline", 2.2),
            ("leader_margin_010", "Leader Margin 1.0%", 1.8),
            ("cash_tail_soft", "Cash Tail Soft", 1.8),
            ("cash_tail_mid", "Cash Tail Mid", 1.8),
            ("cash_tail_strict", "Cash Tail Strict", 1.8),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
