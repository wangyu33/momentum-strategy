#!/usr/bin/env python3
"""Fine-grid search around the strongest stress-bond near-miss candidate."""

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
from compare_stressbond_nearmiss_repair import (
    apply_overlay,
    load_overlay_context,
)
from compare_tail_risk_bond_overlay import load_market_proxy_best_context
from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    ensure_output_dirs,
    load_fixed_etf_pool,
    write_dataframe_csv_atomic,
)
from compare_candidate_pool_additions import ETF_511260


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_stressbond_edge_refine")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-grid refine the strongest stress-bond near-miss candidate.")
    parser.add_argument("--years", type=int, default=15, help="Backtest years.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--notify", action="store_true", help="Send webhook notification if a better strategy is found.")
    parser.add_argument("--webhook-url", type=str, default="", help="Webhook URL override.")
    return parser.parse_args()


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


def load_notify_state() -> dict[str, object] | None:
    if NOTIFY_STATE_PATH.exists():
        return json.loads(NOTIFY_STATE_PATH.read_text(encoding="utf-8"))
    return None


def save_notify_state(strategy_name: str, summary: dict[str, object], description: str) -> None:
    write_json_atomic(NOTIFY_STATE_PATH, {"strategy": strategy_name, "summary": summary, "description": description})


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    market_context = load_market_proxy_best_context()
    overlay_context = load_overlay_context()
    params = dict(market_context["params"])
    params["volume_ratio_cut"] = float(market_context["volume_ratio_cut"])
    params["volume_short_ratio_cut"] = float(market_context["volume_short_ratio_cut"])
    params["volume_breadth_cut"] = float(market_context["volume_breadth_cut"])
    drop_codes = [str(code) for code in params["drop_codes"]]

    base_selected = load_fixed_etf_pool()
    base_selected = base_selected[~base_selected["code"].astype(str).isin(drop_codes)].reset_index(drop=True)
    selected = pd.concat([base_selected, pd.DataFrame([ETF_511260])], ignore_index=True)

    _, prices = load_cached_data()
    start_ts = prices.index.max() - pd.DateOffset(years=args.years)
    prices = prices.loc[prices.index >= start_ts].copy()
    prices = prices[[code for code in selected["code"].astype(str) if code in prices.columns]].copy()

    from compare_goal_optimizations import load_market_volume_proxy

    base_market_proxy = load_market_volume_proxy(years=args.years, refresh=False)
    proxy_catalog = {item["name"]: item["proxy"] for item in build_proxy_catalog(base_market_proxy, prices)}
    proxy = proxy_catalog[str(market_context["proxy_kind"])]

    rows: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}
    nav_compare = pd.DataFrame(index=prices.index)

    baseline_result, baseline_trades = apply_overlay(
        prices=prices,
        selected=selected,
        proxy=proxy,
        params=params,
        treasury_code=overlay_context["treasury_code"],
        risk_cap=overlay_context["risk_cap"],
        ratio_cut=overlay_context["ratio_cut"],
        breadth_cut=overlay_context["breadth_cut"],
        vg_cap=float(params["volume_guard_cap"]),
        overheat_cap=float(params["overheat_max_exposure"]),
        overheat_hi_cap=float(params["overheat_high_max_exposure"]),
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_summary = summarize(baseline_result, baseline_trades, selected)
    baseline_summary["strategy"] = overlay_context["strategy"]
    rows.append(baseline_summary)
    nav_compare[f"{overlay_context['strategy']}_nav"] = baseline_result["nav"]
    descriptions[str(overlay_context["strategy"])] = overlay_context["description"]

    risk_cap_candidates = [0.178, 0.180, 0.182, 0.185, 0.188, 0.190, 0.192]
    vg_cap_candidates = [0.285, 0.290, 0.295]
    overheat_cap_candidates = [0.068, 0.070, 0.072]
    overheat_hi_cap_candidates = [0.028, 0.030, 0.032]
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
                        f"[stressbond_edge_refine] {run_idx}/{total_runs} "
                        f"risk_cap={risk_cap:.3f} vg_cap={vg_cap:.3f} "
                        f"overheat={overheat_cap:.3f}/{overheat_hi_cap:.3f}",
                        flush=True,
                    )
                    result, trades = apply_overlay(
                        prices=prices,
                        selected=selected,
                        proxy=proxy,
                        params=params,
                        treasury_code=overlay_context["treasury_code"],
                        risk_cap=risk_cap,
                        ratio_cut=overlay_context["ratio_cut"],
                        breadth_cut=overlay_context["breadth_cut"],
                        vg_cap=vg_cap,
                        overheat_cap=overheat_cap,
                        overheat_hi_cap=overheat_hi_cap,
                        fee_rate=args.fee_rate,
                        slippage_rate=args.slippage_rate,
                    )
                    strategy = (
                        f"{market_context['strategy']}__edge"
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
                        "围绕当前最接近有效的新候选继续细化："
                        f"风险仓上限 {risk_cap:.1%}，弱量能上限 {vg_cap:.1%}，"
                        f"过热/极热上限 {overheat_cap:.1%}/{overheat_hi_cap:.1%}。"
                    )

    summary_df = pd.DataFrame(rows)
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(baseline_summary["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(baseline_summary["sharpe_rf0"])
    summary_df["max_drawdown_integral_diff"] = summary_df["max_drawdown_integral"] - float(baseline_summary["max_drawdown_integral"])
    summary_df["is_valid_change"] = (
        (summary_df["strategy"] != str(overlay_context["strategy"]))
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
