#!/usr/bin/env python3
"""比较若干动态动量阈值方案。"""

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

import pandas as pd

try:
    from ...official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

from archive_data_loaders import build_archive_flat_output_dir, load_core_price_panel_with_fallback
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    build_typed_summary_fields,
    filter_available_plot_lines,
    finalize_baseline_diff_summary,
    save_named_nav_outputs,
    save_plot_and_print_summary_preview,
    summarize_variant_result,
)
from archive_strategy_common import (
    DEFAULT_HISTORY_START,
    DEFAULT_YEARS,
    build_default_strategy_params,
    fetch_histories,
    load_fixed_etf_pool,
    run_default_strategy_with_params,
)


OUTPUT_DIR = build_archive_flat_output_dir("compare_dynamic_thresholds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较动态动量阈值方案。")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--start-date", type=str, default=str(DEFAULT_HISTORY_START.date()))
    return parser.parse_args()


def build_variants() -> list[dict[str, object]]:
    return [
        {"name": "baseline_fixed_05", "overrides": {}},
        {
            "name": "weak_step_06_relaxed",
            "overrides": {
                "dynamic_threshold_mode": "weak_proxy_step",
                "dynamic_threshold_amount_20_60_cut": 1.00,
                "dynamic_threshold_amount_5_20_cut": 1.00,
                "dynamic_threshold_breadth_cut": 0.00,
                "dynamic_threshold_weak_value": 0.06,
            },
        },
        {
            "name": "weak_step_06_strict",
            "overrides": {
                "dynamic_threshold_mode": "weak_proxy_step",
                "dynamic_threshold_amount_20_60_cut": 0.98,
                "dynamic_threshold_amount_5_20_cut": 0.98,
                "dynamic_threshold_breadth_cut": -0.01,
                "dynamic_threshold_weak_value": 0.06,
            },
        },
        {
            "name": "weak_tiered_065",
            "overrides": {
                "dynamic_threshold_mode": "weak_proxy_tiered",
                "dynamic_threshold_amount_20_60_cut": 1.00,
                "dynamic_threshold_amount_5_20_cut": 1.00,
                "dynamic_threshold_breadth_cut": 0.00,
                "dynamic_threshold_tier_step": 0.005,
                "dynamic_threshold_max": 0.065,
            },
        },
        {
            "name": "weak_tiered_060",
            "overrides": {
                "dynamic_threshold_mode": "weak_proxy_tiered",
                "dynamic_threshold_amount_20_60_cut": 1.00,
                "dynamic_threshold_amount_5_20_cut": 1.00,
                "dynamic_threshold_breadth_cut": 0.00,
                "dynamic_threshold_tier_step": 0.005,
                "dynamic_threshold_max": 0.060,
            },
        },
        {
            "name": "asym_045_06",
            "overrides": {
                "dynamic_threshold_mode": "asymmetric_band",
                "dynamic_threshold_amount_20_60_cut": 1.00,
                "dynamic_threshold_breadth_cut": 0.00,
                "dynamic_threshold_weak_value": 0.06,
                "dynamic_threshold_strong_amount_20_60_cut": 1.05,
                "dynamic_threshold_strong_breadth_cut": 0.02,
                "dynamic_threshold_strong_value": 0.045,
            },
        },
    ]


def summarize_zone(daily_ret: pd.Series) -> dict[str, float]:
    if daily_ret.empty:
        return {
            "zone_total_return": 0.0,
            "zone_avg_daily_return": 0.0,
            "zone_win_rate": 0.0,
            "zone_max_drawdown": 0.0,
            "zone_days": 0,
        }
    zone_nav = (1 + daily_ret.fillna(0.0)).cumprod()
    zone_nav.iloc[0] = 1.0
    zone_drawdown = zone_nav / zone_nav.cummax() - 1
    return {
        "zone_total_return": float(zone_nav.iloc[-1] - 1),
        "zone_avg_daily_return": float(daily_ret.mean()),
        "zone_win_rate": float((daily_ret > 0).mean()),
        "zone_max_drawdown": float(zone_drawdown.min()),
        "zone_days": int(len(daily_ret)),
    }


def main() -> None:
    args = parse_args()
    selected = load_fixed_etf_pool()
    prices = load_core_price_panel_with_fallback(
        selected,
        years=args.years,
        fetch_fn=fetch_histories,
    )
    prices = prices.loc[prices.index >= pd.Timestamp(args.start_date)].copy()

    baseline_params = build_default_strategy_params()
    baseline_result, baseline_trades = run_default_strategy_with_params(prices, selected, baseline_params)
    baseline_result = apply_official_baseline_nav_anchor(baseline_result)
    baseline_zone_mask = baseline_result["effective_momentum"].gt(0) & baseline_result["effective_momentum"].le(0.05)

    rows: list[dict[str, object]] = []
    variant_frames: dict[str, pd.DataFrame] = {"baseline_fixed_05": baseline_result}
    variant_trade_counts: dict[str, int] = {"baseline_fixed_05": len(baseline_trades)}

    for variant in build_variants():
        name = str(variant["name"])
        if name == "baseline_fixed_05":
            result = baseline_result
            trades = baseline_trades
        else:
            params = build_default_strategy_params()
            params.update(variant["overrides"])
            result, trades = run_default_strategy_with_params(prices, selected, params)
            variant_frames[name] = result
            variant_trade_counts[name] = len(trades)

        summary = summarize_variant_result(
            result,
            trades,
            selected=selected,
            include_max_drawdown_integral=True,
        )
        common_zone = result.loc[baseline_zone_mask.reindex(result.index).fillna(False), "strategy_return"].dropna()
        zone_summary = summarize_zone(common_zone)
        threshold_series = result.get("absolute_momentum_threshold", pd.Series(0.05, index=result.index, dtype="float64"))

        append_variant_result(
            rows,
            None,
            None,
            strategy=name,
            result=None,
            summary=build_typed_summary_fields(
                summary,
                float_fields=("annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral"),
                int_fields=("trade_count",),
                str_fields=(),
                extra_fields={
                    "dynamic_threshold_mode": variant.get("overrides", {}).get("dynamic_threshold_mode", "fixed"),
                    "baseline_zone_days": zone_summary["zone_days"],
                    "baseline_zone_total_return": zone_summary["zone_total_return"],
                    "baseline_zone_avg_daily_return": zone_summary["zone_avg_daily_return"],
                    "baseline_zone_win_rate": zone_summary["zone_win_rate"],
                    "baseline_zone_max_drawdown": zone_summary["zone_max_drawdown"],
                    "threshold_mean": float(threshold_series.mean()),
                    "threshold_min": float(threshold_series.min()),
                    "threshold_max": float(threshold_series.max()),
                    "threshold_gt_05_days": int((threshold_series > 0.05).sum()),
                    "threshold_lt_05_days": int((threshold_series < 0.05).sum()),
                    "latest_threshold": float(threshold_series.iloc[-1]),
                },
            ),
            strategy_field="variant",
        )

    summary_df = finalize_baseline_diff_summary(
        rows,
        baseline_field="variant",
        baseline_value="baseline_fixed_05",
        metric_mappings=(
            ("annualized_return", "annualized_return_diff_vs_base"),
            ("sharpe_rf0", "sharpe_rf0_diff_vs_base"),
            ("max_drawdown", "max_drawdown_diff_vs_base"),
            ("max_drawdown_integral", "max_drawdown_integral_diff_vs_base"),
            ("trade_count", "trade_count_diff_vs_base"),
            ("baseline_zone_total_return", "baseline_zone_total_return_diff_vs_base"),
            ("baseline_zone_max_drawdown", "baseline_zone_max_drawdown_diff_vs_base"),
        ),
        sort_by=["baseline_zone_total_return_diff_vs_base", "annualized_return_diff_vs_base", "sharpe_rf0_diff_vs_base"],
        ascending=[False, False, False],
        reset_index=True,
    )

    compare_df = build_compare_frame(prices)
    for name, frame in variant_frames.items():
        append_variant_result(
            [],
            compare_df,
            None,
            strategy=name,
            result=frame.reindex(compare_df.index),
            summary={},
            nav_column=f"{name}_nav",
        )

    export_cols = [
        "nav",
        "drawdown",
        "signal",
        "holding",
        "exposure",
        "current_momentum",
        "effective_momentum",
        "absolute_momentum_threshold",
    ]
    save_named_nav_outputs(
        OUTPUT_DIR,
        variant_frames,
        columns=export_cols,
        index=True,
    )

    preview_cols = [
        "variant",
        "dynamic_threshold_mode",
        "annualized_return",
        "sharpe_rf0",
        "max_drawdown",
        "baseline_zone_total_return",
        "baseline_zone_max_drawdown",
        "threshold_mean",
        "threshold_max",
        "threshold_gt_05_days",
        "annualized_return_diff_vs_base",
        "baseline_zone_total_return_diff_vs_base",
    ]
    plot_lines = [
        ("baseline_fixed_05_nav", "Baseline Fixed 5%", 2.2),
        ("weak_step_06_relaxed_nav", "Weak Step 6% Relaxed", 1.8),
        ("weak_step_06_strict_nav", "Weak Step 6% Strict", 1.8),
        ("weak_tiered_060_nav", "Weak Tiered 6.0%", 1.8),
        ("asym_045_06_nav", "Asymmetric 4.5/6.0%", 1.8),
    ]
    available_plot_lines = filter_available_plot_lines(compare_df, plot_lines)
    save_plot_and_print_summary_preview(
        OUTPUT_DIR,
        summary_df,
        compare_df,
        preview_cols,
        plot_filename="comparison.png",
        title="Dynamic Threshold Comparison",
        lines=available_plot_lines,
    )


if __name__ == "__main__":
    main()
