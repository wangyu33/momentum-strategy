#!/usr/bin/env python3
"""对比不同量能代理混合方案，保持当前正式 breadth 定义不变。"""

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

from archive_data_loaders import build_archive_flat_output_dir, load_recent_selected_prices
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    build_typed_summary_fields,
    finalize_baseline_diff_summary,
    save_plot_and_print_variant_compare_outputs,
)
try:
    from ...official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

from goal_optimization_common import load_market_volume_proxy
from hs300_regime_common import summarize
from market_proxy_common import build_risk_proxy_features
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    build_default_strategy_params,
    load_default_strategy_backtest_pool,
    resolve_strategy_universe,
    run_default_strategy_with_params,
)


OUTPUT_DIR = build_archive_flat_output_dir("compare_amount_proxy_blends")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比不同量能代理混合方案。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = load_default_strategy_backtest_pool()
    prices = load_recent_selected_prices(selected, args.years)

    base_proxy = load_market_volume_proxy(years=args.years, refresh=False).reindex(prices.index).ffill()
    risk_codes, _ = resolve_strategy_universe(prices)
    features = build_risk_proxy_features(prices, risk_codes=risk_codes)

    # 固定当前正式 breadth 口径，只测试量能代理本身是否值得替换。
    hybrid_breadth = 0.5 * base_proxy["market_breadth_proxy"] + 0.5 * features["breadth_blend"]

    variants: list[tuple[str, pd.Series, pd.Series, float, float]] = [
        (
            "current_a_share_amount",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            0.91,
            0.90,
        ),
    ]

    for etf_key, label in [
        ("above_ma20_ratio_20_60", "etf_ma20"),
        ("pos20_ratio_20_60", "etf_pos20"),
    ]:
        short_key = etf_key.replace("20_60", "5_20")
        for weight in [0.25, 0.50, 0.75]:
            variants.append(
                (
                    f"blend_{label}_{int(weight * 100):02d}",
                    (1 - weight) * base_proxy["market_amount_ratio_20_60"] + weight * features[etf_key],
                    (1 - weight) * base_proxy["market_amount_ratio_5_20"] + weight * features[short_key],
                    0.91,
                    0.90,
                )
            )

    variants.extend(
        [
            (
                "pure_etf_ma20_amount",
                features["above_ma20_ratio_20_60"],
                features["above_ma20_ratio_5_20"],
                0.96,
                0.95,
            ),
            (
                "pure_etf_pos20_amount",
                features["pos20_ratio_20_60"],
                features["pos20_ratio_5_20"],
                0.96,
                0.95,
            ),
        ]
    )

    rows: list[dict[str, object]] = []
    nav_compare = build_compare_frame(prices)

    for name, ratio_20_60, ratio_5_20, ratio_cut, short_cut in variants:
        proxy = pd.DataFrame(index=prices.index)
        proxy["market_amount_ratio_20_60"] = ratio_20_60
        proxy["market_amount_ratio_5_20"] = ratio_5_20
        proxy["market_breadth_proxy"] = hybrid_breadth

        params = build_default_strategy_params()
        # 传入已构造好的代理，避免 run_default_strategy_with_params 再次混合 breadth。
        params["proxy_kind"] = "baseline_sh_sz"
        params["volume_breadth_cut"] = -0.04
        params["volume_ratio_cut"] = ratio_cut
        params["volume_short_ratio_cut"] = short_cut

        result, trades = run_default_strategy_with_params(
            prices=prices,
            selected=selected,
            params=params,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            market_proxy=proxy,
        )
        if name == "current_a_share_amount":
            result = apply_official_baseline_nav_anchor(result)
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
                "volume_ratio_cut": ratio_cut,
                "volume_short_ratio_cut": short_cut,
            },
        )

    summary_df = finalize_baseline_diff_summary(
        rows,
        baseline_field="name",
        baseline_value="current_a_share_amount",
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
        title="Amount Proxy Blend Comparison",
        lines=(
            ("current_a_share_amount_nav", "Current A Share Amount", 2.2),
            ("blend_etf_ma20_50_nav", "Blend ETF MA20 50%", 1.8),
            ("blend_etf_pos20_50_nav", "Blend ETF Pos20 50%", 1.8),
            ("pure_etf_ma20_amount_nav", "Pure ETF MA20", 1.8),
            ("pure_etf_pos20_amount_nav", "Pure ETF Pos20", 1.8),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
