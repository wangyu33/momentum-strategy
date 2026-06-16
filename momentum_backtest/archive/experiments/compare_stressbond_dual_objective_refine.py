#!/usr/bin/env python3
"""Dual-objective local refine between the current winner and the lower-drawdown near neighbor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from ..runtime_env import prepare_local_imports, write_json_atomic
except ImportError:
    from runtime_env import prepare_local_imports, write_json_atomic

prepare_local_imports(__file__)

import pandas as pd

from compare_goal_optimizations import DEFAULT_FEISHU_WEBHOOK, METRIC_TOLERANCE, send_improvement_notification
from compare_hs300_regime_fixes import load_cached_data, summarize
from compare_market_proxy_variants import build_proxy_catalog
from compare_stressbond_nearmiss_repair import apply_overlay
from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_REGIME_MIX_AGGRESSIVE_CORE_WEIGHT,
    DEFAULT_REGIME_MIX_CONSERVATIVE_CORE_WEIGHT,
    DEFAULT_REGIME_MIX_MOMENTUM_CUT,
    DEFAULT_REGIME_MIX_OVERHEAT_DRAWDOWN_CUT,
    DEFAULT_REGIME_MIX_OVERHEAT_MOMENTUM_CUT,
    DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MAX_EXPOSURE,
    DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MOMENTUM_CUT,
    DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE,
    DEFAULT_REGIME_MIX_PROXY_KIND,
    DEFAULT_REGIME_MIX_VOLUME_BREADTH_CUT,
    DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP,
    DEFAULT_REGIME_MIX_VOLUME_GUARD_MOMENTUM_CEILING,
    DEFAULT_REGIME_MIX_VOLUME_RATIO_CUT,
    DEFAULT_REGIME_MIX_VOLUME_SHORT_RATIO_CUT,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_STRESS_BOND_BREADTH_CUT,
    DEFAULT_STRESS_BOND_CODE,
    DEFAULT_STRESS_BOND_RISK_CAP,
    DEFAULT_STRESS_BOND_RATIO_CUT,
    DEFAULT_STRATEGY_NAME,
    ensure_output_dirs,
    load_default_strategy_backtest_pool,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_stressbond_dual_objective_refine")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-objective refine between current best and nearby lower-drawdown candidate.")
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


def load_current_context() -> dict[str, object]:
    return {
        "strategy": DEFAULT_STRATEGY_NAME,
        "description": (
            "当前正式基线：hybrid breadth proxy + regime mix + 弱量能限仓 + 过热双层降仓 + 十年国债弱市防守。"
        ),
        "risk_cap": DEFAULT_STRESS_BOND_RISK_CAP,
        "vg_cap": DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP,
        "overheat_cap": DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE,
        "overheat_hi_cap": DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MAX_EXPOSURE,
    }


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

    current_context = load_current_context()
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
        risk_cap=current_context["risk_cap"],
        ratio_cut=DEFAULT_STRESS_BOND_RATIO_CUT,
        breadth_cut=DEFAULT_STRESS_BOND_BREADTH_CUT,
        vg_cap=current_context["vg_cap"],
        overheat_cap=current_context["overheat_cap"],
        overheat_hi_cap=current_context["overheat_hi_cap"],
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_summary = summarize(baseline_result, baseline_trades, selected)
    baseline_summary["strategy"] = current_context["strategy"]
    rows.append(baseline_summary)
    nav_compare[f"{current_context['strategy']}_nav"] = baseline_result["nav"]
    descriptions[str(current_context["strategy"])] = current_context["description"]

    risk_cap_candidates = [0.188, 0.189, 0.190, 0.191, 0.192]
    vg_cap_candidates = [0.284, 0.285, 0.286]
    overheat_cap_candidates = [0.067, 0.068, 0.069]
    overheat_hi_cap_candidates = [0.027, 0.028, 0.029]
    total_runs = len(risk_cap_candidates) * len(vg_cap_candidates) * len(overheat_cap_candidates) * len(overheat_hi_cap_candidates)
    run_idx = 0

    for risk_cap in risk_cap_candidates:
        for vg_cap in vg_cap_candidates:
            for overheat_cap in overheat_cap_candidates:
                for overheat_hi_cap in overheat_hi_cap_candidates:
                    if overheat_hi_cap > overheat_cap:
                        continue
                    run_idx += 1
                    print(
                        f"[stressbond_dual_objective_refine] {run_idx}/{total_runs} "
                        f"risk_cap={risk_cap:.3f} vg_cap={vg_cap:.3f} "
                        f"overheat={overheat_cap:.3f}/{overheat_hi_cap:.3f}",
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
                        vg_cap=vg_cap,
                        overheat_cap=overheat_cap,
                        overheat_hi_cap=overheat_hi_cap,
                        fee_rate=args.fee_rate,
                        slippage_rate=args.slippage_rate,
                    )
                    strategy = (
                        f"{current_context['strategy']}__dual"
                        f"_rc{int(round(risk_cap * 1000)):03d}"
                        f"_vg{int(round(vg_cap * 1000)):03d}"
                        f"_oc{int(round(overheat_cap * 1000)):03d}"
                        f"_oh{int(round(overheat_hi_cap * 1000)):03d}"
                        f"__stressbond_511260_vr90_vb-1"
                    )
                    summary = summarize(result, trades, selected)
                    summary["strategy"] = strategy
                    summary["risk_cap"] = risk_cap
                    summary["vg_cap"] = vg_cap
                    summary["overheat_cap"] = overheat_cap
                    summary["overheat_hi_cap"] = overheat_hi_cap
                    rows.append(summary)
                    nav_compare[f"{strategy}_nav"] = result["nav"]
                    descriptions[strategy] = (
                        "在当前 rc188 收益/夏普冠军 与 rc190 回撤积分更优候选之间做插值微调："
                        f"风险仓上限 {risk_cap:.1%}，弱量能上限 {vg_cap:.1%}，"
                        f"过热/极热上限 {overheat_cap:.1%}/{overheat_hi_cap:.1%}。"
                    )

    summary_df = pd.DataFrame(rows)
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(baseline_summary["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(baseline_summary["sharpe_rf0"])
    summary_df["max_drawdown_integral_diff"] = summary_df["max_drawdown_integral"] - float(baseline_summary["max_drawdown_integral"])
    summary_df["is_valid_change"] = (
        (summary_df["strategy"] != str(current_context["strategy"]))
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
                "vg_cap",
                "overheat_cap",
                "overheat_hi_cap",
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

    notify_state = load_notify_state()
    notify_df = better_ranked_df.copy()
    if notify_state is not None:
        previous_summary = notify_state.get("summary", {})
        notify_df = notify_df[
            (notify_df["annualized_return"] >= float(previous_summary["annualized_return"]) - METRIC_TOLERANCE)
            & (notify_df["sharpe_rf0"] >= float(previous_summary["sharpe_rf0"]) - METRIC_TOLERANCE)
            & (notify_df["max_drawdown_integral"] <= float(previous_summary["max_drawdown_integral"]) + METRIC_TOLERANCE)
            & (
                (notify_df["annualized_return"] > float(previous_summary["annualized_return"]) + METRIC_TOLERANCE)
                | (notify_df["sharpe_rf0"] > float(previous_summary["sharpe_rf0"]) + METRIC_TOLERANCE)
                | (notify_df["max_drawdown_integral"] < float(previous_summary["max_drawdown_integral"]) - METRIC_TOLERANCE)
            )
        ].copy()
        notify_df = rank_valid_improvements(notify_df, pd.Series(previous_summary))

    if args.notify and not notify_df.empty:
        best = notify_df.iloc[0]
        webhook_url = args.webhook_url or os.getenv("DAILY_MONITOR_WEBHOOK_URL", DEFAULT_FEISHU_WEBHOOK)
        description = descriptions[str(best["strategy"])]
        send_improvement_notification(baseline_summary, str(best["strategy"]), best.to_dict(), description, webhook_url)
        save_notify_state(str(best["strategy"]), best.to_dict(), description)
        print(f"\nWebhook notified for {best['strategy']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
