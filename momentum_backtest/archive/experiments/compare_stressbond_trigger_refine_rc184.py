#!/usr/bin/env python3
"""Refine stress-bond trigger thresholds around the active rc184 baseline."""

from __future__ import annotations

import argparse
import json
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
    ensure_output_dirs,
    load_default_strategy_backtest_pool,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_stressbond_trigger_refine_rc184")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine current stress-bond trigger thresholds around the rc184 baseline.")
    parser.add_argument("--years", type=int, default=15, help="Backtest years.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--notify", action="store_true", help="Send webhook notification if a better strategy is found.")
    parser.add_argument("--webhook-url", type=str, default="", help="Webhook URL override.")
    return parser.parse_args()


def load_notify_state() -> dict[str, object] | None:
    if NOTIFY_STATE_PATH.exists():
        return json.loads(NOTIFY_STATE_PATH.read_text(encoding="utf-8"))
    return None


def save_notify_state(strategy_name: str, summary: dict[str, object], description: str) -> None:
    write_json_atomic(NOTIFY_STATE_PATH, {"strategy": strategy_name, "summary": summary, "description": description})


def apply_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    treasury_code: str,
    risk_cap: float,
    ratio_cut: float,
    breadth_cut: float,
    short_ratio_cut: float | None,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_weights, mixed_momentum, _ = build_base_target_weights(prices, proxy, params)
    risk_codes = [code for code in ["510300", "159949", "159954", "159941", "513650", "513880"] if code in prices.columns]
    overlaid_weights = base_weights.copy()
    proxy = proxy.reindex(prices.index).ffill()
    row_risk_weight = base_weights[risk_codes].sum(axis=1)

    trigger_mask = (
        (proxy["market_amount_ratio_20_60"] < ratio_cut)
        & (proxy["market_breadth_proxy"] < breadth_cut)
    ).fillna(False)
    if short_ratio_cut is not None:
        trigger_mask = trigger_mask & (proxy["market_amount_ratio_5_20"] < short_ratio_cut).fillna(False)

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


def rank_valid_improvements(frame: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.copy()
    ranked["annualized_return_gain"] = ranked["annualized_return"] - float(baseline["annualized_return"])
    ranked["sharpe_gain"] = ranked["sharpe_rf0"] - float(baseline["sharpe_rf0"])
    ranked["drawdown_integral_improvement"] = float(baseline["max_drawdown_integral"]) - ranked["max_drawdown_integral"]
    ranked["composite_improvement_score"] = (
        ranked["annualized_return_gain"] * 100
        + ranked["sharpe_gain"] * 10
        + ranked["drawdown_integral_improvement"] / 10
    )
    return ranked.sort_values(
        [
            "composite_improvement_score",
            "annualized_return_gain",
            "sharpe_gain",
            "drawdown_integral_improvement",
            "annualized_return",
            "sharpe_rf0",
            "max_drawdown_integral",
        ],
        ascending=[False, False, False, False, False, False, True],
    )


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
        short_ratio_cut=None,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_summary = summarize(baseline_result, baseline_trades, selected)
    baseline_summary["strategy"] = DEFAULT_STRATEGY_NAME
    rows.append(baseline_summary)
    nav_compare[f"{DEFAULT_STRATEGY_NAME}_nav"] = baseline_result["nav"]
    descriptions[DEFAULT_STRATEGY_NAME] = "当前正式基线：hybrid breadth proxy + regime mix + 弱量能限仓 + 过热双层降仓 + 十年国债弱市防守。"

    ratio_candidates = [0.895, 0.900, 0.905]
    breadth_candidates = [-0.035, -0.030, -0.025]
    risk_cap_candidates = [0.182, 0.184, 0.186]
    short_ratio_candidates: list[float | None] = [None, 0.92]
    total_runs = len(ratio_candidates) * len(breadth_candidates) * len(risk_cap_candidates) * len(short_ratio_candidates)
    run_idx = 0

    for ratio_cut in ratio_candidates:
        for breadth_cut in breadth_candidates:
            for risk_cap in risk_cap_candidates:
                for short_ratio_cut in short_ratio_candidates:
                    run_idx += 1
                    print(
                        f"[stressbond_trigger_refine_rc184] {run_idx}/{total_runs} "
                        f"ratio={ratio_cut:.3f} breadth={breadth_cut:.3f} cap={risk_cap:.3f} "
                        f"short={short_ratio_cut if short_ratio_cut is not None else 'na'}",
                        flush=True,
                    )
                    result, trades = apply_overlay(
                        prices=prices,
                        selected=selected,
                        proxy=proxy,
                        params=params,
                        treasury_code=DEFAULT_STRESS_BOND_CODE,
                        risk_cap=risk_cap,
                        ratio_cut=ratio_cut,
                        breadth_cut=breadth_cut,
                        short_ratio_cut=short_ratio_cut,
                        fee_rate=args.fee_rate,
                        slippage_rate=args.slippage_rate,
                    )
                    short_tag = "na" if short_ratio_cut is None else f"{int(round(short_ratio_cut * 1000)):03d}"
                    strategy = (
                        f"{DEFAULT_STRATEGY_NAME}__trf2"
                        f"_vr{int(round(ratio_cut * 1000)):03d}"
                        f"_vb{int(round((breadth_cut + 0.10) * 1000)):03d}"
                        f"_rc{int(round(risk_cap * 1000)):03d}"
                        f"_sr{short_tag}"
                    )
                    summary = summarize(result, trades, selected)
                    summary["strategy"] = strategy
                    summary["ratio_cut"] = ratio_cut
                    summary["breadth_cut"] = breadth_cut
                    summary["risk_cap"] = risk_cap
                    summary["short_ratio_cut"] = short_ratio_cut
                    rows.append(summary)
                    nav_compare[f"{strategy}_nav"] = result["nav"]
                    descriptions[strategy] = (
                        "围绕当前 rc184 正式基线细化弱市切债触发："
                        f"20/60 量能阈值 {ratio_cut:.1%}，广度阈值 {breadth_cut:.1%}，"
                        f"风险仓上限 {risk_cap:.1%}"
                        f"{'' if short_ratio_cut is None else f'，并要求 5/20 量能<{short_ratio_cut:.1%}'}。"
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
                "ratio_cut",
                "breadth_cut",
                "risk_cap",
                "short_ratio_cut",
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
    state = load_notify_state()
    if state is not None and state.get("strategy") == str(best["strategy"]):
        print(f"Already notified for {best['strategy']}, skip.")
        return 0

    webhook_url = args.webhook_url or DEFAULT_FEISHU_WEBHOOK
    description = descriptions[str(best["strategy"])]
    send_improvement_notification(baseline_summary, str(best["strategy"]), best.to_dict(), description, webhook_url)
    save_notify_state(str(best["strategy"]), best.to_dict(), description)
    print(f"Notified improvement: {best['strategy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
