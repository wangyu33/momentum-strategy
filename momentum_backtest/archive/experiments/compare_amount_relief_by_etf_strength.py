#!/usr/bin/env python3
"""测试当 A 股量能转弱但 ETF 横截面仍强时，是否放松弱量能闸门。

这个 baseline 研究的是默认 regime/core 量能保护层的历史变体，不是正式 28.2691
官方基线，因此本脚本应保持历史语义，不做官方净值锚定。
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

from archive_data_loaders import build_archive_flat_output_dir, build_named_market_proxy, load_recent_selected_prices
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    build_typed_summary_fields,
    finalize_baseline_diff_summary,
    save_plot_and_print_variant_compare_outputs,
)
from goal_optimization_common import build_dynamic_core_target_weights, load_market_volume_proxy
from hs300_regime_common import run_target_weights_strategy, summarize
from market_proxy_common import build_risk_proxy_features
from overlay_strategy_helpers import apply_risk_cap
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RISK_CODES,
    build_default_strategy_params,
    load_default_strategy_backtest_pool,
    resolve_strategy_universe,
)
OUTPUT_DIR = build_archive_flat_output_dir("compare_amount_relief_by_etf_strength")
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试当 A 股量能转弱但 ETF 横截面仍强时，是否放松弱量能闸门。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def build_relief_strategy_result(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    market_proxy: pd.DataFrame,
    params: dict[str, object],
    *,
    breadth_blend_floor: float | None = None,
    above_ma20_floor: float | None = None,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    proxy = build_named_market_proxy(
        prices,
        base_market_proxy=market_proxy,
        proxy_kind=str(params["proxy_kind"]),
        risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
    ).reindex(prices.index).ffill()
    features = build_risk_proxy_features(
        prices,
        risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
    )

    active_risk_codes, active_defensive_codes = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
        defensive_codes=[str(code) for code in params.get("defensive_codes", [])],
    )

    aggressive_weights, aggressive_momentum, _, _, aggressive_trend = build_dynamic_core_target_weights(
        prices,
        lookback=25,
        absolute_threshold=0.05,
        weak_trend_defensive_weight=0.8,
        core_weight=float(params["aggressive_core_weight"]),
        hs300_mom60_cut=0.05,
        hs300_mom120_cut=0.10,
        hs300_ma_window=90,
        signal_quality_method=str(params.get("signal_quality_method", "raw")),
        slope_penalty=float(params.get("signal_slope_penalty", 0.0)),
        leader_margin=float(params.get("signal_leader_margin", 0.0)),
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )
    conservative_weights, conservative_momentum, _, _, _ = build_dynamic_core_target_weights(
        prices,
        lookback=25,
        absolute_threshold=0.05,
        weak_trend_defensive_weight=0.8,
        core_weight=float(params["conservative_core_weight"]),
        hs300_mom60_cut=0.04,
        hs300_mom120_cut=0.10,
        hs300_ma_window=120,
        signal_quality_method=str(params.get("signal_quality_method", "raw")),
        slope_penalty=float(params.get("signal_slope_penalty", 0.0)),
        leader_margin=float(params.get("signal_leader_margin", 0.0)),
        risk_codes=active_risk_codes,
        defensive_codes=active_defensive_codes,
    )

    aggressive_mask = aggressive_trend & (aggressive_momentum >= float(params["regime_momentum_cut"]))
    mixed_target_weights = conservative_weights.copy()
    mixed_target_weights.loc[aggressive_mask] = aggressive_weights.loc[aggressive_mask]
    mixed_momentum = conservative_momentum.copy()
    mixed_momentum.loc[aggressive_mask] = aggressive_momentum.loc[aggressive_mask]
    mixed_momentum = mixed_momentum.rename("current_momentum")

    row_risk_weight = mixed_target_weights[active_risk_codes].sum(axis=1)
    volume_weak_mask = (
        (proxy["market_amount_ratio_20_60"] < float(params["volume_ratio_cut"]))
        & (proxy["market_amount_ratio_5_20"] < float(params["volume_short_ratio_cut"]))
        & (proxy["market_breadth_proxy"] < float(params["volume_breadth_cut"]))
        & (mixed_momentum <= float(params["volume_guard_momentum_ceiling"]))
    ).fillna(False)

    relief_mask = pd.Series(False, index=prices.index, dtype=bool)
    if breadth_blend_floor is not None:
        relief_mask = relief_mask | (features["breadth_blend"] >= breadth_blend_floor).fillna(False)
    if above_ma20_floor is not None:
        relief_mask = relief_mask | (features["above_ma20_frac"] >= above_ma20_floor).fillna(False)

    guarded_weights = apply_risk_cap(
        mixed_target_weights,
        risk_budget_codes=active_risk_codes,
        trigger_mask=volume_weak_mask & (~relief_mask),
        risk_cap=float(params["volume_guard_cap"]),
    )

    result, trades = run_target_weights_strategy(
        prices,
        selected,
        guarded_weights,
        mixed_momentum,
        fee_rate,
        slippage_rate,
    )
    return result, trades


def main() -> int:
    args = parse_args()

    selected = load_default_strategy_backtest_pool()
    prices = load_recent_selected_prices(selected, args.years)
    market_proxy = load_market_volume_proxy(years=args.years, refresh=False)

    base_params = build_default_strategy_params()
    rows: list[dict[str, object]] = []
    nav_compare = build_compare_frame(prices)

    variants = [
        ("baseline", None, None),
        ("relief_breadth_blend_ge_m2", -0.02, None),
        ("relief_breadth_blend_ge_m1", -0.01, None),
        ("relief_breadth_blend_ge_0", 0.0, None),
        ("relief_above_ma20_ge_55", None, 0.55),
        ("relief_above_ma20_ge_60", None, 0.60),
        ("relief_above_ma20_ge_65", None, 0.65),
        ("relief_combo_m1_or_ma60", -0.01, 0.60),
    ]

    for name, breadth_floor, ma20_floor in variants:
        result, trades = build_relief_strategy_result(
            prices=prices,
            selected=selected,
            market_proxy=market_proxy,
            params=base_params,
            breadth_blend_floor=breadth_floor,
            above_ma20_floor=ma20_floor,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        summary = summarize(result, trades, selected)
        append_variant_result(
            rows,
            nav_compare,
            None,
            strategy=name,
            result=result,
            strategy_field="name",
            nav_column=f"{name}_nav",
            summary=build_typed_summary_fields(
                summary,
                float_fields=("annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral"),
                int_fields=("trade_action_count",),
            ),
            extra_fields={
                "breadth_blend_floor": breadth_floor,
                "above_ma20_floor": ma20_floor,
            },
        )

    summary_df = finalize_baseline_diff_summary(
        rows,
        baseline_field="name",
        baseline_value="baseline",
        metric_mappings=(
            ("annualized_return", "annualized_diff"),
            ("sharpe_rf0", "sharpe_diff"),
            ("max_drawdown", "max_drawdown_diff"),
            ("max_drawdown_integral", "max_drawdown_integral_diff"),
        ),
    )
    save_plot_and_print_variant_compare_outputs(
        OUTPUT_DIR,
        summary_df,
        nav_compare,
        plot_filename="comparison.png",
        title="Amount Relief By ETF Strength Comparison",
        lines=(
            ("baseline_nav", "Baseline", 2.2),
            ("relief_breadth_blend_ge_m1_nav", "Breadth Blend >= -1%", 1.8),
            ("relief_above_ma20_ge_60_nav", "Above MA20 >= 60%", 1.8),
            ("relief_combo_m1_or_ma60_nav", "Combo Relief", 1.8),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
