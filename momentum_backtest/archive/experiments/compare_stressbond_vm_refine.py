#!/usr/bin/env python3
"""Local refine around current baseline with focus on volume-guard momentum ceiling."""

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
from compare_hs300_regime_fixes import load_cached_data, summarize
from compare_market_proxy_variants import build_proxy_catalog
from compare_stressbond_nearmiss_repair import apply_overlay
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
    DEFAULT_STRESS_BOND_RISK_CAP,
    DEFAULT_STRESS_BOND_RATIO_CUT,
    DEFAULT_STRATEGY_NAME,
    ensure_output_dirs,
    load_default_strategy_backtest_pool,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_stressbond_vm_refine")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine baseline around vm/vg/overheat thresholds.")
    parser.add_argument("--years", type=int, default=15, help="Backtest years.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--notify", action="store_true", help="Send webhook notification if a better strategy is found.")
    parser.add_argument("--webhook-url", type=str, default="", help="Webhook URL override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_params = {
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
        params=base_params,
        treasury_code=DEFAULT_STRESS_BOND_CODE,
        risk_cap=DEFAULT_STRESS_BOND_RISK_CAP,
        ratio_cut=DEFAULT_STRESS_BOND_RATIO_CUT,
        breadth_cut=DEFAULT_STRESS_BOND_BREADTH_CUT,
        vg_cap=DEFAULT_REGIME_MIX_VOLUME_GUARD_CAP,
        overheat_cap=DEFAULT_REGIME_MIX_OVERHEAT_MAX_EXPOSURE,
        overheat_hi_cap=DEFAULT_REGIME_MIX_OVERHEAT_HIGH_MAX_EXPOSURE,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_summary = summarize(baseline_result, baseline_trades, selected)
    baseline_summary["strategy"] = DEFAULT_STRATEGY_NAME
    rows.append(baseline_summary)
    nav_compare[f"{DEFAULT_STRATEGY_NAME}_nav"] = baseline_result["nav"]
    descriptions[DEFAULT_STRATEGY_NAME] = "当前正式基线：hybrid breadth proxy + regime mix + 弱量能限仓 + 过热双层降仓 + 十年国债弱市防守。"

    risk_cap_candidates = [DEFAULT_STRESS_BOND_RISK_CAP]
    vg_cap_candidates = [0.283, 0.284, 0.285]
    vm_candidates = [0.14, 0.15, 0.16, 0.17]
    overheat_cap_candidates = [0.066, 0.067, 0.068]
    overheat_hi_cap_candidates = [0.028, 0.029, 0.030]
    total_runs = (
        len(risk_cap_candidates)
        * len(vg_cap_candidates)
        * len(vm_candidates)
        * len(overheat_cap_candidates)
        * len(overheat_hi_cap_candidates)
    )
    run_idx = 0

    for risk_cap in risk_cap_candidates:
        for vg_cap in vg_cap_candidates:
            for vm_cap in vm_candidates:
                for overheat_cap in overheat_cap_candidates:
                    for overheat_hi_cap in overheat_hi_cap_candidates:
                        if overheat_hi_cap > overheat_cap:
                            continue
                        run_idx += 1
                        print(
                            f"[stressbond_vm_refine] {run_idx}/{total_runs} "
                            f"risk_cap={risk_cap:.3f} vg_cap={vg_cap:.3f} vm_cap={vm_cap:.3f} "
                            f"overheat={overheat_cap:.3f}/{overheat_hi_cap:.3f}",
                            flush=True,
                        )
                        params = dict(base_params)
                        params["volume_guard_cap"] = vg_cap
                        params["volume_guard_momentum_ceiling"] = vm_cap
                        params["overheat_max_exposure"] = overheat_cap
                        params["overheat_high_max_exposure"] = overheat_hi_cap

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
                            f"{DEFAULT_STRATEGY_NAME}__vmrf"
                            f"_rc{int(round(risk_cap * 1000)):03d}"
                            f"_vg{int(round(vg_cap * 1000)):03d}"
                            f"_vm{int(round(vm_cap * 1000)):03d}"
                            f"_oc{int(round(overheat_cap * 1000)):03d}"
                            f"_oh{int(round(overheat_hi_cap * 1000)):03d}"
                        )
                        summary = summarize(result, trades, selected)
                        summary["strategy"] = strategy
                        summary["risk_cap"] = risk_cap
                        summary["vg_cap"] = vg_cap
                        summary["vm_cap"] = vm_cap
                        summary["overheat_cap"] = overheat_cap
                        summary["overheat_hi_cap"] = overheat_hi_cap
                        rows.append(summary)
                        nav_compare[f"{strategy}_nav"] = result["nav"]
                        descriptions[strategy] = (
                            "围绕当前正式基线做阈值微调："
                            f"风险仓上限 {risk_cap:.1%} 固定，"
                            f"弱量能上限 {vg_cap:.1%}，弱量能触发动量上限 {vm_cap:.1%}，"
                            f"过热/极热上限 {overheat_cap:.1%}/{overheat_hi_cap:.1%}。"
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
                "vg_cap",
                "vm_cap",
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
