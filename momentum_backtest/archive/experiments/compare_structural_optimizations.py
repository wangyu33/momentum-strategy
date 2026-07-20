#!/usr/bin/env python3
"""对比当前正式基线与两步结构优化的效果。"""

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

from archive_data_loaders import build_archive_flat_output_dir, load_selected_prices
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    finalize_baseline_diff_summary,
    save_plot_and_print_variant_compare_outputs,
)
try:
    from ...official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

from goal_optimization_common import load_market_volume_proxy
from hs300_regime_common import summarize
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    build_default_strategy_params,
    fetch_histories,
    load_default_strategy_backtest_pool,
    run_default_strategy_with_params,
)


OUTPUT_DIR = build_archive_flat_output_dir("compare_structural_optimizations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比当前正式基线与两步结构优化的效果。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取价格和市场代理数据。")
    return parser.parse_args()


def build_variant_rows(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    market_proxy: pd.DataFrame,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_params = build_default_strategy_params()
    split_params = dict(base_params)
    split_params["volume_guard_cap"] = float(base_params["aggressive_core_weight"]) + float(base_params["conservative_core_weight"])

    smooth_params = dict(split_params)
    smooth_params["overheat_cap_mode"] = "continuous"

    variants = [
        (
            "baseline",
            "当前正式基线：弱量能直接清零风险仓，弱市触发后再切债，高位过热使用两档台阶式降仓。",
            base_params,
            False,
        ),
        (
            "step1_split_weak_and_stress",
            "第一步：弱量能先把风险仓压到核心仓总和 55%，只负责减风险；更弱市时再把剩余现金统一停泊到十年国债。",
            split_params,
            True,
        ),
        (
            "step2_split_and_smooth_overheat",
            "第二步：保留第一步的弱市拆层，同时把过热降仓从两档台阶改成 25%~32% 动量区间内线性连续收缩。",
            smooth_params,
            True,
        ),
    ]

    summary_rows: list[dict[str, object]] = []
    nav_compare = build_compare_frame(prices)
    for name, description, params, fill_residual_cash in variants:
        result, trades = run_default_strategy_with_params(
            prices=prices,
            selected=selected,
            params=params,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            market_proxy=market_proxy,
            fill_residual_cash_to_treasury=fill_residual_cash,
        )
        if name == "baseline":
            result = apply_official_baseline_nav_anchor(result)
        append_variant_result(
            summary_rows,
            nav_compare,
            None,
            strategy=name,
            result=result,
            summary=summarize(result, trades, selected),
            strategy_field="strategy",
            nav_column=f"{name}_nav",
            extra_fields={
                "description": description,
                "fill_residual_cash_to_treasury": fill_residual_cash,
                "volume_guard_cap": float(params["volume_guard_cap"]),
                "overheat_cap_mode": str(params.get("overheat_cap_mode", "step")),
            },
        )

    return (
        finalize_baseline_diff_summary(
            summary_rows,
            baseline_field="strategy",
            baseline_value="baseline",
            metric_mappings=(
                ("annualized_return", "annualized_diff"),
                ("sharpe_rf0", "sharpe_diff"),
                ("max_drawdown", "max_drawdown_diff"),
                ("max_drawdown_integral", "max_drawdown_integral_diff"),
            ),
        ),
        nav_compare,
    )


def main() -> int:
    args = parse_args()

    selected = load_default_strategy_backtest_pool()
    prices = load_selected_prices(
        selected,
        years=args.years,
        refresh=args.refresh,
        fetch_fn=fetch_histories,
    )
    market_proxy = load_market_volume_proxy(years=args.years, refresh=args.refresh)

    summary_df, nav_compare = build_variant_rows(
        selected=selected,
        prices=prices,
        market_proxy=market_proxy,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    save_plot_and_print_variant_compare_outputs(
        OUTPUT_DIR,
        summary_df,
        nav_compare,
        plot_filename="comparison.png",
        title="Structural Optimizations Comparison",
        lines=(
            ("baseline_nav", "Baseline", 2.2),
            ("step1_split_weak_and_stress_nav", "Step 1", 1.8),
            ("step2_split_and_smooth_overheat_nav", "Step 2", 1.8),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
