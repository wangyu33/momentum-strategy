#!/usr/bin/env python3
"""Refine stress-bond persistence state machine around the active default baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ..runtime_env import prepare_local_imports, write_json_atomic
except ImportError:
    from runtime_env import prepare_local_imports, write_json_atomic

prepare_local_imports(__file__)

import pandas as pd

from compare_goal_optimizations import DEFAULT_FEISHU_WEBHOOK, METRIC_TOLERANCE, send_improvement_notification
from compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
from compare_market_proxy_variants import build_proxy_catalog
from compare_tail_risk_bond_overlay import build_base_target_weights
from search_utils import load_notify_state, rank_valid_improvements, save_notify_state
from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_REGIME_MIX_AGGRESSIVE_CORE_WEIGHT,
    DEFAULT_REGIME_MIX_CONSERVATIVE_CORE_WEIGHT,
    DEFAULT_REGIME_MIX_MOMENTUM_CUT,
    DEFAULT_REGIME_MIX_OVERHEAT_DRAWDOWN_CUT,
    DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MAX_EXPOSURE,
    DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT,
    DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE,
    DEFAULT_REGIME_MIX_OVERHEAT_MOMENTUM_CUT,
    DEFAULT_REGIME_MIX_PROXY_KIND,
    DEFAULT_REGIME_MIX_VOLUME_BREADTH_CUT,
    DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP,
    DEFAULT_REGIME_MIX_VOLUME_GUARD_MOMENTUM_CEILING,
    DEFAULT_REGIME_MIX_VOLUME_RATIO_CUT,
    DEFAULT_REGIME_MIX_VOLUME_SHORT_RATIO_CUT,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_STRESS_BOND_BREADTH_CUT,
    DEFAULT_STRESS_BOND_CODE,
    DEFAULT_STRESS_BOND_RATIO_CUT,
    DEFAULT_STRESS_BOND_RISK_CAP,
    DEFAULT_STRATEGY_NAME,
    build_persistent_trigger_mask,
    ensure_output_dirs,
    load_default_strategy_backtest_pool,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_stressbond_persistence_current")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine stress-bond persistence around the active baseline.")
    parser.add_argument("--years", type=int, default=15, help="Backtest years.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--notify", action="store_true", help="Send webhook notification if a better strategy is found.")
    parser.add_argument("--webhook-url", type=str, default="", help="Webhook URL override.")
    return parser.parse_args()


def apply_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    treasury_code: str,
    risk_cap: float,
    ratio_cut: float,
    breadth_cut: float,
    enter_days: int,
    exit_days: int,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_weights, mixed_momentum, _ = build_base_target_weights(prices, proxy, params)
    risk_codes = [code for code in ["510300", "159949", "159954", "159941", "513650", "513880"] if code in prices.columns]
    overlaid_weights = base_weights.copy()
    proxy = proxy.reindex(prices.index).ffill()
    row_risk_weight = base_weights[risk_codes].sum(axis=1)

    raw_trigger = (
        (proxy["market_amount_ratio_20_60"] < ratio_cut)
        & (proxy["market_breadth_proxy"] < breadth_cut)
    ).fillna(False)
    trigger_mask = build_persistent_trigger_mask(raw_trigger, enter_days=enter_days, exit_days=exit_days)

    scale_mask = trigger_mask & (row_risk_weight > risk_cap)
    if scale_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[scale_mask] = risk_cap / row_risk_weight.loc[scale_mask]
        overlaid_weights.loc[scale_mask, risk_codes] = overlaid_weights.loc[scale_mask, risk_codes].mul(
            scale.loc[scale_mask], axis=0
        )
        moved_weight = row_risk_weight.loc[scale_mask] - risk_cap
        overlaid_weights.loc[scale_mask, treasury_code] = overlaid_weights.loc[scale_mask, treasury_code].add(
            moved_weight,
            fill_value=0.0,
        )

    return run_target_weights_strategy(prices, selected, overlaid_weights, mixed_momentum, fee_rate, slippage_rate)
def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    params = {
        "drop_codes": ["511580", "513650"],
        "aggressive_core_weight": DEFAULT_REGIME_MIX_AGGRESSIVE_CORE_WEIGHT,
        "conservative_core_weight": DEFAULT_REGIME_MIX_CONSERVATIVE_CORE_WEIGHT,
        "regime_momentum_cut": DEFAULT_REGIME_MIX_MOMENTUM_CUT,
        "volume_ratio_cut": DEFAULT_REGIME_MIX_VOLUME_RATIO_CUT,
        "volume_short_ratio_cut": DEFAULT_REGIME_MIX_VOLUME_SHORT_RATIO_CUT,
        "volume_breadth_cut": DEFAULT_REGIME_MIX_VOLUME_BREADTH_CUT,
        "volume_guard_cap": DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP,
        "volume_guard_momentum_ceiling": DEFAULT_REGIME_MIX_VOLUME_GUARD_MOMENTUM_CEILING,
        "overheat_drawdown_cut": DEFAULT_REGIME_MIX_OVERHEAT_DRAWDOWN_CUT,
        "overheat_momentum_cut": DEFAULT_REGIME_MIX_OVERHEAT_MOMENTUM_CUT,
        "overheat_max_exposure": DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE,
        "overheat_high_momentum_cut": DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT,
        "overheat_high_max_exposure": DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MAX_EXPOSURE,
    }
    selected = load_default_strategy_backtest_pool()

    _, prices = load_cached_data()
    start_ts = prices.index.max() - pd.DateOffset(years=args.years)
    prices = prices.loc[prices.index >= start_ts].copy()
    prices = prices[[code for code in selected["code"].astype(str) if code in prices.columns]].copy()

    from compare_goal_optimizations import load_market_volume_proxy

    base_market_proxy = load_market_volume_proxy(years=args.years, refresh=False)
    proxy_catalog = {item["name"]: item["proxy"] for item in build_proxy_catalog(base_market_proxy, prices)}
    proxy = proxy_catalog[DEFAULT_REGIME_MIX_PROXY_KIND]

    rows: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}
    nav_compare = pd.DataFrame(index=prices.index)

    baseline_result, baseline_trades = apply_overlay(
        prices=prices,
        selected=selected,
        proxy=proxy,
        params=params,
        treasury_code=DEFAULT_STRESS_BOND_CODE,
        risk_cap=DEFAULT_STRESS_BOND_RISK_CAP,
        ratio_cut=DEFAULT_STRESS_BOND_RATIO_CUT,
        breadth_cut=DEFAULT_STRESS_BOND_BREADTH_CUT,
        enter_days=1,
        exit_days=1,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_summary = summarize(baseline_result, baseline_trades, selected)
    baseline_summary["strategy"] = DEFAULT_STRATEGY_NAME
    rows.append(baseline_summary)
    nav_compare[f"{DEFAULT_STRATEGY_NAME}_nav"] = baseline_result["nav"]
    descriptions[DEFAULT_STRATEGY_NAME] = "当前正式基线：hybrid breadth proxy + regime mix + 弱量能限仓 + 过热双层降仓 + 十年国债弱市防守。"

    enter_days_candidates = [1, 2, 3]
    exit_days_candidates = [1, 2, 3, 5]
    risk_cap_candidates = [0.184, 0.186, 0.188]
    total_runs = len(enter_days_candidates) * len(exit_days_candidates) * len(risk_cap_candidates)
    run_idx = 0

    for enter_days in enter_days_candidates:
        for exit_days in exit_days_candidates:
            for risk_cap in risk_cap_candidates:
                run_idx += 1
                print(
                    f"[stressbond_persistence_current] {run_idx}/{total_runs} "
                    f"enter={enter_days} exit={exit_days} cap={risk_cap:.3f}",
                    flush=True,
                )
                result, trades = apply_overlay(
                    prices=prices,
                    selected=selected,
                    proxy=proxy,
                    params=params,
                    treasury_code=DEFAULT_STRESS_BOND_CODE,
                    risk_cap=risk_cap,
                    ratio_cut=DEFAULT_STRESS_BOND_RATIO_CUT,
                    breadth_cut=DEFAULT_STRESS_BOND_BREADTH_CUT,
                    enter_days=enter_days,
                    exit_days=exit_days,
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                )
                strategy = (
                    f"{DEFAULT_STRATEGY_NAME}__prs"
                    f"_rc{int(round(risk_cap * 1000)):03d}"
                    f"_en{enter_days:02d}_ex{exit_days:02d}"
                )
                summary = summarize(result, trades, selected)
                summary["strategy"] = strategy
                summary["risk_cap"] = risk_cap
                summary["enter_days"] = enter_days
                summary["exit_days"] = exit_days
                rows.append(summary)
                nav_compare[f"{strategy}_nav"] = result["nav"]
                descriptions[strategy] = (
                    "围绕当前正式基线细化弱市切债状态机："
                    f"风险仓上限 {risk_cap:.1%}，触发连续 {enter_days} 天后进入防守，"
                    f"信号消失连续 {exit_days} 天后退出防守。"
                )

    summary_df = pd.DataFrame(rows)
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(baseline_summary["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(baseline_summary["sharpe_rf0"])
    summary_df["max_drawdown_integral_diff"] = summary_df["max_drawdown_integral"] - float(
        baseline_summary["max_drawdown_integral"]
    )
    summary_df["is_valid_change"] = (
        (summary_df["strategy"] != DEFAULT_STRATEGY_NAME)
        & (summary_df["annualized_return"] >= float(baseline_summary["annualized_return"]) - METRIC_TOLERANCE)
        & (summary_df["sharpe_rf0"] >= float(baseline_summary["sharpe_rf0"]) - METRIC_TOLERANCE)
        & (summary_df["max_drawdown_integral"] <= float(baseline_summary["max_drawdown_integral"]) + METRIC_TOLERANCE)
        & (
            (summary_df["annualized_return"] > float(baseline_summary["annualized_return"]) + METRIC_TOLERANCE)
            | (summary_df["sharpe_rf0"] > float(baseline_summary["sharpe_rf0"]) + METRIC_TOLERANCE)
            | (summary_df["max_drawdown_integral"] < float(baseline_summary["max_drawdown_integral"]) - METRIC_TOLERANCE)
        )
    )
    summary_df = summary_df.sort_values(
        ["is_valid_change", "annualized_return", "sharpe_rf0", "max_drawdown_integral"],
        ascending=[False, False, False, True],
    )
    write_dataframe_csv_atomic(summary_df, SUMMARY_PATH, index=False)
    write_dataframe_csv_atomic(nav_compare, COMPARE_PATH)

    valid_df = summary_df[summary_df["is_valid_change"]].copy()
    better_ranked_df = rank_valid_improvements(valid_df, pd.Series(baseline_summary))
    write_json_atomic(
        BEST_PATH,
        {
            "baseline": baseline_summary,
            "valid_improvements": better_ranked_df.to_dict(orient="records"),
            "descriptions": descriptions,
        },
    )

    print("Baseline:")
    print(pd.Series(baseline_summary).to_string())
    print("\nTop candidates:")
    print(
        summary_df[
            [
                "strategy",
                "risk_cap",
                "enter_days",
                "exit_days",
                "annualized_return",
                "sharpe_rf0",
                "max_drawdown_integral",
                "annualized_diff",
                "sharpe_diff",
                "max_drawdown_integral_diff",
                "is_valid_change",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    if not args.notify:
        return 0

    if better_ranked_df.empty:
        print("No valid improvements to notify.")
        return 0

    best = better_ranked_df.iloc[0]
    state = load_notify_state(NOTIFY_STATE_PATH)
    if state is not None and state.get("strategy") == str(best["strategy"]):
        print(f"Already notified for {best['strategy']}, skip.")
        return 0

    webhook_url = args.webhook_url or DEFAULT_FEISHU_WEBHOOK
    description = descriptions[str(best["strategy"])]
    send_improvement_notification(baseline_summary, str(best["strategy"]), best.to_dict(), description, webhook_url)
    save_notify_state(NOTIFY_STATE_PATH, str(best["strategy"]), best.to_dict(), description)
    print(f"Notified improvement: {best['strategy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
