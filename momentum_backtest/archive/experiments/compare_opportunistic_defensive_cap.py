#!/usr/bin/env python3
"""对比黄金/豆粕在弱市高波动时的临时降权方案。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MOMENTUM_DIR = Path(__file__).resolve().parents[2]
if str(MOMENTUM_DIR) not in sys.path:
    sys.path.insert(0, str(MOMENTUM_DIR))

from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

try:
    from ...official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

from goal_optimization_common import load_market_volume_proxy
from hs300_regime_common import run_target_weights_strategy
from archive_data_loaders import (
    build_archive_flat_output_dir,
    build_named_market_proxy,
    extract_result_target_weights,
    load_selected_prices,
)
from overlay_candidate_catalog import TREASURY_10Y
from overlay_strategy_helpers import apply_risk_cap_with_moved_weight
from variant_compare_helpers import (
    append_variant_result,
    build_compare_frame,
    build_summary_frame,
    build_preview_columns,
    filter_available_plot_lines,
    finalize_baseline_diff_summary,
    get_summary_mapping,
    save_plot_and_print_baseline_preview,
    summarize_variant_result,
)

from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    build_default_strategy_params,
    fetch_histories,
    load_default_strategy_backtest_pool,
    run_default_strategy_with_params,
)


OUTPUT_DIR = build_archive_flat_output_dir("compare_opportunistic_defensive_cap")
OPPORTUNISTIC_CODES = ["518880", "159985"]
TREASURY_CODE = TREASURY_10Y["code"]
OPPORTUNISTIC_SCOPES = {
    "both": ["518880", "159985"],
    "gold_only": ["518880"],
    "soymeal_only": ["159985"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比黄金/豆粕在弱市高波动时的临时降权方案。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取历史数据，而不是复用 output/core 缓存。")
    return parser.parse_args()


def apply_opportunistic_cap(
    base_target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    proxy: pd.DataFrame,
    *,
    opp_codes: list[str],
    cap: float,
    vol20_cut: float,
    breadth_cut: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    adjusted = base_target_weights.copy()
    opp_codes = [code for code in opp_codes if code in adjusted.columns]
    if not opp_codes:
        return adjusted, pd.Series(False, index=adjusted.index), pd.Series(0.0, index=adjusted.index), pd.Series(0.0, index=adjusted.index)

    returns = prices[opp_codes].pct_change()
    vol20 = returns.rolling(20).std()
    opp_weight = adjusted[opp_codes].sum(axis=1)

    # 用“当前持有的机会型防守仓”的加权波动率来判断它是否开始表现得更像风险资产。
    weighted_vol20 = (vol20.mul(adjusted[opp_codes], axis=0).sum(axis=1) / opp_weight.replace(0.0, pd.NA)).fillna(0.0)
    trigger_mask = (
        (opp_weight > cap)
        & (weighted_vol20 >= vol20_cut)
        & (proxy["market_breadth_proxy"] <= breadth_cut)
    ).fillna(False)

    adjusted, _, moved_weight = apply_risk_cap_with_moved_weight(
        adjusted,
        risk_budget_codes=opp_codes,
        trigger_mask=trigger_mask,
        risk_cap=cap,
    )
    if trigger_mask.any():
        adjusted.loc[trigger_mask, TREASURY_CODE] = adjusted.loc[trigger_mask, TREASURY_CODE].add(
            moved_weight.loc[trigger_mask],
            fill_value=0.0,
        )

    return adjusted, trigger_mask, weighted_vol20, moved_weight


def main() -> int:
    args = parse_args()

    selected = load_default_strategy_backtest_pool().copy()
    prices = load_selected_prices(
        selected,
        years=args.years,
        refresh=args.refresh,
        fetch_fn=fetch_histories,
    )
    params = build_default_strategy_params()
    market_proxy = load_market_volume_proxy(years=args.years, refresh=args.refresh)
    proxy = build_named_market_proxy(
        prices,
        base_market_proxy=market_proxy,
        proxy_kind=str(params["proxy_kind"]),
        risk_codes=[str(code) for code in params.get("risk_codes", [])],
    ).reindex(prices.index).ffill()

    baseline_result, baseline_trades = run_default_strategy_with_params(
        prices,
        selected,
        params=params,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        market_proxy=market_proxy,
    )
    baseline_result = apply_official_baseline_nav_anchor(baseline_result)
    base_target_weights = extract_result_target_weights(baseline_result, prices)
    signal_momentum = baseline_result["current_momentum"].copy()

    rows: list[dict[str, object]] = []
    nav_compare = build_compare_frame(prices)

    append_variant_result(
        rows,
        nav_compare,
        None,
        strategy="baseline",
        result=baseline_result,
        summary=summarize_variant_result(
            baseline_result,
            baseline_trades,
            selected=selected,
            include_max_drawdown_integral=True,
        ),
        nav_column="baseline_nav",
        extra_fields={
            "trigger_days": 0,
            "avg_moved_weight": 0.0,
            "max_moved_weight": 0.0,
            "avg_weighted_vol20": 0.0,
            "scope": "baseline",
        },
    )
    baseline_summary = rows[-1]

    for scope_name, scope_codes in OPPORTUNISTIC_SCOPES.items():
        for cap in (0.50, 0.65, 0.80):
            for vol20_cut in (0.018, 0.020, 0.022, 0.024):
                for breadth_cut in (-0.010, -0.015, -0.020):
                    adjusted_weights, trigger_mask, weighted_vol20, moved_weight = apply_opportunistic_cap(
                        base_target_weights,
                        prices,
                        proxy,
                        opp_codes=scope_codes,
                        cap=cap,
                        vol20_cut=vol20_cut,
                        breadth_cut=breadth_cut,
                    )
                    result, trades = run_target_weights_strategy(
                        prices,
                        selected,
                        adjusted_weights,
                        signal_momentum,
                        args.fee_rate,
                        args.slippage_rate,
                    )
                    strategy = (
                        f"{scope_name}_cap{int(round(cap * 100)):02d}"
                        f"_vol{int(round(vol20_cut * 1000)):03d}"
                        f"_br{int(round(abs(breadth_cut) * 1000)):03d}"
                    )
                    append_variant_result(
                        rows,
                        nav_compare,
                        None,
                        strategy=strategy,
                        result=result,
                        summary=summarize_variant_result(
                            result,
                            trades,
                            selected=selected,
                            include_max_drawdown_integral=True,
                        ),
                        extra_fields={
                            "scope": scope_name,
                            "cap": cap,
                            "vol20_cut": vol20_cut,
                            "breadth_cut": breadth_cut,
                            "trigger_days": int(trigger_mask.sum()),
                            "avg_moved_weight": float(moved_weight.loc[trigger_mask].mean()) if trigger_mask.any() else 0.0,
                            "max_moved_weight": float(moved_weight.max()),
                            "avg_weighted_vol20": float(weighted_vol20.loc[trigger_mask].mean()) if trigger_mask.any() else 0.0,
                        },
                    )

    summary_df = finalize_baseline_diff_summary(
        rows,
        baseline_field="strategy",
        baseline_value="baseline",
        metric_mappings=(
            ("annualized_return", "annualized_return_diff"),
            ("sharpe_rf0", "sharpe_rf0_diff"),
            ("max_drawdown", "max_drawdown_diff"),
            ("max_drawdown_integral", "max_drawdown_integral_diff"),
        ),
    )
    summary_df["is_integral_valid_change"] = (
        (summary_df["strategy"] != "baseline")
        & (summary_df["annualized_return"] >= float(baseline_summary["annualized_return"]) - 0.002)
        & (summary_df["sharpe_rf0"] >= float(baseline_summary["sharpe_rf0"]) - 0.02)
        & (summary_df["max_drawdown_integral"] <= float(baseline_summary["max_drawdown_integral"]) + 0.2)
        & (
            (summary_df["annualized_return"] > float(baseline_summary["annualized_return"]) + 0.001)
            | (summary_df["sharpe_rf0"] > float(baseline_summary["sharpe_rf0"]) + 0.01)
            | (summary_df["max_drawdown_integral"] < float(baseline_summary["max_drawdown_integral"]) - 0.1)
        )
    )
    summary_df["is_strict_drawdown_safe"] = (
        (summary_df["strategy"] != "baseline")
        & (summary_df["annualized_return"] >= float(baseline_summary["annualized_return"]) - 0.002)
        & (summary_df["sharpe_rf0"] >= float(baseline_summary["sharpe_rf0"]) - 0.02)
        & (summary_df["max_drawdown"] >= float(baseline_summary["max_drawdown"]) - 1e-12)
        & (summary_df["max_drawdown_integral"] <= float(baseline_summary["max_drawdown_integral"]) + 0.2)
        & (
            (summary_df["annualized_return"] > float(baseline_summary["annualized_return"]) + 0.001)
            | (summary_df["sharpe_rf0"] > float(baseline_summary["sharpe_rf0"]) + 0.01)
            | (summary_df["max_drawdown_integral"] < float(baseline_summary["max_drawdown_integral"]) - 0.1)
        )
    )
    summary_df = build_summary_frame(
        summary_df,
        sort_by=["is_integral_valid_change", "is_strict_drawdown_safe", "annualized_return", "sharpe_rf0", "max_drawdown_integral"],
        ascending=[False, False, False, False, True],
    )
    baseline_preview = get_summary_mapping(
        summary_df,
        field="strategy",
        value="baseline",
        columns=["annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral", "trade_count"],
    )
    plot_lines = [
        ("baseline_nav", "Baseline", 2.2),
        ("both_cap65_vol020_br015_nav", "Both Cap65 Vol20 Br15", 1.8),
        ("gold_only_cap65_vol020_br015_nav", "Gold Only Cap65", 1.8),
        ("soymeal_only_cap65_vol020_br015_nav", "Soymeal Only Cap65", 1.8),
    ]
    available_plot_lines = filter_available_plot_lines(nav_compare, plot_lines)
    save_plot_and_print_baseline_preview(
        OUTPUT_DIR,
        summary_df,
        nav_compare,
        baseline_summary=baseline_preview,
        columns=build_preview_columns(
            ["strategy", "scope"],
            metric_columns=["annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral"],
            diff_columns=["annualized_return_diff", "sharpe_rf0_diff", "max_drawdown_integral_diff"],
            suffix_columns=["trigger_days", "avg_moved_weight", "is_integral_valid_change", "is_strict_drawdown_safe"],
        ),
        plot_filename="comparison.png",
        title="Opportunistic Defensive Cap Comparison",
        lines=available_plot_lines,
        head=15,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
