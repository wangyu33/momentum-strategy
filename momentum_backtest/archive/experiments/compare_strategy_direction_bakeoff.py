#!/usr/bin/env python3
"""统一对比几类值得优先尝试的正式策略优化方向。"""

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

from goal_optimization_common import MARKET_VOLUME_CACHE_PATH
from archive_data_loaders import build_archive_flat_output_dir
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    build_summary_frame,
    get_summary_row,
    build_yearly_returns_df,
    finalize_baseline_diff_summary,
    save_variant_compare_artifacts,
    summarize_variant_result,
)
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    build_default_strategy_params,
    load_core_selected_and_prices,
    run_default_strategy_with_params,
)

matplotlib.use("Agg")


OUTPUT_DIR = build_archive_flat_output_dir("compare_strategy_direction_bakeoff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一对比几类值得优先尝试的正式策略优化方向。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def load_cached_market_proxy() -> pd.DataFrame:
    proxy = pd.read_csv(MARKET_VOLUME_CACHE_PATH, parse_dates=["date"])
    return proxy.set_index("date")


def build_variants() -> list[tuple[str, dict[str, object]]]:
    baseline = build_default_strategy_params()

    confirm60_top2 = build_default_strategy_params()
    confirm60_top2.update({"signal_confirmation_lookback": 60, "signal_confirmation_top_n": 2})

    confirm60_top3 = build_default_strategy_params()
    confirm60_top3.update({"signal_confirmation_lookback": 60, "signal_confirmation_top_n": 3})

    slope_r2_020 = build_default_strategy_params()
    slope_r2_020.update({"signal_r2_penalty": 0.020})

    tie_r2_gap005 = build_default_strategy_params()
    tie_r2_gap005.update({"signal_secondary_stability_method": "r2", "signal_secondary_stability_gap": 0.005})

    leader_margin005 = build_default_strategy_params()
    leader_margin005.update({"signal_leader_margin": 0.005})

    continuous_transition = build_default_strategy_params()
    continuous_transition.update(
        {
            "regime_transition_mode": "continuous",
            "regime_transition_start_cut": 0.0,
            "regime_transition_end_cut": 0.05,
        }
    )

    extreme_volcap = build_default_strategy_params()
    extreme_volcap.update(
        {
            "signal_selected_volatility_cap_start": 0.95,
            "signal_selected_volatility_cap_end": 0.99,
            "signal_selected_volatility_cap_floor": 0.90,
        }
    )

    combo_confirm_tie_volcap = build_default_strategy_params()
    combo_confirm_tie_volcap.update(
        {
            "signal_confirmation_lookback": 60,
            "signal_confirmation_top_n": 3,
            "signal_secondary_stability_method": "r2",
            "signal_secondary_stability_gap": 0.003,
            "signal_selected_volatility_cap_start": 0.95,
            "signal_selected_volatility_cap_end": 0.99,
            "signal_selected_volatility_cap_floor": 0.90,
        }
    )

    return [
        ("baseline", baseline),
        ("confirm60_top2", confirm60_top2),
        ("confirm60_top3", confirm60_top3),
        ("slope_r2_020", slope_r2_020),
        ("tie_r2_gap005", tie_r2_gap005),
        ("leader_margin005", leader_margin005),
        ("continuous_transition", continuous_transition),
        ("extreme_volcap", extreme_volcap),
        ("combo_confirm_tie_volcap", combo_confirm_tie_volcap),
    ]


def main() -> int:
    args = parse_args()
    selected, prices = load_core_selected_and_prices()
    market_proxy = load_cached_market_proxy()

    variants = build_variants()
    summary_rows: list[dict[str, object]] = []
    yearly_frames: list[pd.DataFrame] = []
    named_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    compare_df = build_compare_frame(prices, benchmark_column="hs300_benchmark_nav")
    benchmark_nav = compare_df["hs300_benchmark_nav"]

    for strategy_name, params in variants:
        result, trades = run_default_strategy_with_params(
            prices,
            selected,
            params=params,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            market_proxy=market_proxy,
        )
        if strategy_name == "baseline":
            result = apply_official_baseline_nav_anchor(result)
        append_variant_result(
            summary_rows,
            compare_df,
            None,
            strategy=strategy_name,
            result=result,
            summary=summarize_variant_result(
                result,
                trades,
                selected=selected,
                include_max_drawdown_integral=True,
                include_latest_signal_holding=True,
            ),
            nav_column=f"{strategy_name}_nav",
        )

        yearly_frames.append(
            build_yearly_returns_df(
                result["nav"],
                benchmark_nav,
                benchmark_return_col="hs300_return",
                strategy=strategy_name,
            )
        )

        named_outputs[strategy_name] = (result, trades)

    summary = finalize_baseline_diff_summary(
        summary_rows,
        baseline_field="strategy",
        baseline_value="baseline",
        metric_mappings=(
            ("total_return", "total_return_diff_vs_baseline"),
            ("annualized_return", "annualized_return_diff_vs_baseline"),
            ("sharpe_rf0", "sharpe_rf0_diff_vs_baseline"),
            ("max_drawdown", "max_drawdown_diff_vs_baseline"),
            ("max_drawdown_integral", "max_drawdown_integral_diff_vs_baseline"),
            ("trade_count", "trade_count_diff_vs_baseline"),
            ("trade_action_count", "trade_action_count_diff_vs_baseline"),
        ),
    )
    baseline_row = get_summary_row(summary, field="strategy", value="baseline")
    summary["is_soft_improvement"] = (
        (summary["annualized_return"] >= float(baseline_row["annualized_return"]))
        & (summary["max_drawdown_integral"] <= float(baseline_row["max_drawdown_integral"]))
    )
    summary = build_summary_frame(
        summary,
        sort_by=["is_soft_improvement", "annualized_return", "max_drawdown_integral", "max_drawdown"],
        ascending=[False, False, True, True],
    )

    save_variant_compare_artifacts(
        OUTPUT_DIR,
        summary,
        compare_df,
        yearly_returns_df=pd.concat(yearly_frames, ignore_index=True),
        named_outputs=named_outputs,
        plot_filename="nav_compare.png",
        title="Strategy Direction Bakeoff",
        lines=[
            (col, col.replace("_nav", ""), 1.6)
            for col in compare_df.columns
            if col.endswith("_nav") and col != "hs300_benchmark_nav"
        ],
        benchmark_label="hs300_benchmark",
        benchmark_column="hs300_benchmark_nav",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
