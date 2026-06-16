#!/usr/bin/env python3
"""对比现金类防守资产替代十年国债覆盖层的历史实验脚本。"""

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
from compare_simple_bond_overlay import apply_simple_overlay
from compare_tail_risk_bond_overlay import load_market_proxy_best_context
from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_cash_overlay_candidates")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"
SIMPLE_OVERLAY_STATE_PATH = Path("momentum_backtest/output/research/archive_flat/compare_simple_bond_overlay/notify_state.json")

CASH_CANDIDATES = [
    {"theme": "货币ETF", "code": "511990", "name": "华宝添益", "sina_symbol": "sh511990"},
    {"theme": "银华日利", "code": "511880", "name": "银华日利ETF", "sina_symbol": "sh511880"},
    {"theme": "保证金", "code": "159001", "name": "保证金ETF", "sina_symbol": "sz159001"},
    {"theme": "货币ETF", "code": "511850", "name": "货币ETF", "sina_symbol": "sh511850"},
    {"theme": "理财金", "code": "511810", "name": "理财金ETF", "sina_symbol": "sh511810"},
]
TREASURY_10Y = {"theme": "10年国债", "code": "511260", "name": "十年国债ETF", "sina_symbol": "sh511260"}
TREASURY_30Y = {"theme": "30年国债", "code": "511090", "name": "30年国债ETF", "sina_symbol": "sh511090"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare cash-like overlays against the current notified treasury overlay.")
    parser.add_argument("--years", type=int, default=15, help="Backtest years.")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Single-side fee rate.")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Single-side slippage rate.")
    parser.add_argument("--refresh", action="store_true", help="Refresh ETF histories instead of using cached core data.")
    parser.add_argument("--notify", action="store_true", help="Send webhook notification if a better strategy is found.")
    parser.add_argument("--webhook-url", type=str, default="", help="Webhook URL override.")
    return parser.parse_args()


def load_notify_state() -> dict[str, object] | None:
    if NOTIFY_STATE_PATH.exists():
        return json.loads(NOTIFY_STATE_PATH.read_text(encoding="utf-8"))
    return None


def save_notify_state(strategy_name: str, summary: dict[str, object], description: str) -> None:
    payload = {"strategy": strategy_name, "summary": summary, "description": description}
    write_json_atomic(NOTIFY_STATE_PATH, payload)


def load_simple_overlay_context() -> dict[str, object]:
    if not SIMPLE_OVERLAY_STATE_PATH.exists():
        raise RuntimeError("missing simple bond overlay notify state")
    payload = json.loads(SIMPLE_OVERLAY_STATE_PATH.read_text(encoding="utf-8"))
    summary = dict(payload["summary"])
    return {
        "strategy": str(payload["strategy"]),
        "summary": summary,
        "description": str(payload["description"]),
        "mode": str(summary["mode"]),
        "risk_cap": float(summary["risk_cap"]),
        "drawdown_cut": None if summary.get("drawdown_cut") is None else float(summary["drawdown_cut"]),
        "ratio_cut": None if summary.get("ratio_cut") is None else float(summary["ratio_cut"]),
        "breadth_cut": None if summary.get("breadth_cut") is None else float(summary["breadth_cut"]),
        "treasury_code": str(summary["treasury_code"]),
    }


def lookup_candidate(code: str) -> dict[str, str]:
    for row in CASH_CANDIDATES + [TREASURY_10Y, TREASURY_30Y]:
        if row["code"] == code:
            return row
    raise KeyError(code)


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    market_context = load_market_proxy_best_context()
    overlay_context = load_simple_overlay_context()
    params = dict(market_context["params"])
    params["volume_ratio_cut"] = float(market_context["volume_ratio_cut"])
    params["volume_short_ratio_cut"] = float(market_context["volume_short_ratio_cut"])
    params["volume_breadth_cut"] = float(market_context["volume_breadth_cut"])
    drop_codes = [str(code) for code in params["drop_codes"]]

    base_selected = load_fixed_etf_pool()
    base_selected = base_selected[~base_selected["code"].astype(str).isin(drop_codes)].reset_index(drop=True)
    candidate_rows = [lookup_candidate(overlay_context["treasury_code"]), TREASURY_10Y, TREASURY_30Y, *CASH_CANDIDATES]
    unique_candidates: list[dict[str, str]] = []
    seen_codes: set[str] = set()
    for row in candidate_rows:
        if row["code"] in seen_codes:
            continue
        seen_codes.add(row["code"])
        unique_candidates.append(row)

    if args.refresh:
        selected_for_prices = pd.concat([base_selected, pd.DataFrame(unique_candidates)], ignore_index=True)
        prices = fetch_histories(selected_for_prices, years=args.years)
    else:
        _, prices = load_cached_data()
        start_ts = prices.index.max() - pd.DateOffset(years=args.years)
        prices = prices.loc[prices.index >= start_ts].copy()
        missing_codes = [row["code"] for row in unique_candidates if row["code"] not in prices.columns]
        if missing_codes:
            extra_rows = [row for row in unique_candidates if row["code"] in missing_codes]
            extra = fetch_histories(pd.DataFrame(extra_rows), years=args.years)
            prices = prices.join(extra, how="outer")
        selected_for_prices = pd.concat([base_selected, pd.DataFrame(unique_candidates)], ignore_index=True)
        prices = prices[[code for code in selected_for_prices["code"].astype(str) if code in prices.columns]].copy()

    from compare_goal_optimizations import load_market_volume_proxy

    base_market_proxy = load_market_volume_proxy(years=args.years, refresh=args.refresh)
    proxy_catalog = {item["name"]: item["proxy"] for item in build_proxy_catalog(base_market_proxy, prices)}
    proxy = proxy_catalog[str(market_context["proxy_kind"])]

    rows: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}
    nav_compare = pd.DataFrame(index=prices.index)

    baseline_candidate = lookup_candidate(overlay_context["treasury_code"])
    baseline_selected = pd.concat([base_selected, pd.DataFrame([baseline_candidate])], ignore_index=True)
    baseline_prices = prices[[code for code in baseline_selected["code"].astype(str) if code in prices.columns]].copy()
    baseline_result, baseline_trades = apply_simple_overlay(
        prices=baseline_prices,
        selected=baseline_selected,
        proxy=proxy,
        params=params,
        treasury_code=overlay_context["treasury_code"],
        mode=overlay_context["mode"],
        risk_cap=overlay_context["risk_cap"],
        drawdown_cut=overlay_context["drawdown_cut"],
        ratio_cut=overlay_context["ratio_cut"],
        breadth_cut=overlay_context["breadth_cut"],
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_summary = summarize(baseline_result, baseline_trades, baseline_selected)
    baseline_summary["strategy"] = overlay_context["strategy"]
    rows.append(baseline_summary)
    nav_compare[f"{overlay_context['strategy']}_nav"] = baseline_result["nav"]
    descriptions[str(overlay_context["strategy"])] = overlay_context["description"]

    for candidate in unique_candidates:
        selected = pd.concat([base_selected, pd.DataFrame([candidate])], ignore_index=True)
        candidate_prices = prices[[code for code in selected["code"].astype(str) if code in prices.columns]].copy()
        risk_caps = [overlay_context["risk_cap"]] if candidate["code"] == overlay_context["treasury_code"] else [
            overlay_context["risk_cap"],
            min(float(overlay_context["risk_cap"]) + 0.10, 0.40),
        ]
        for risk_cap in risk_caps:
            result, trades = apply_simple_overlay(
                prices=candidate_prices,
                selected=selected,
                proxy=proxy,
                params=params,
                treasury_code=str(candidate["code"]),
                mode=overlay_context["mode"],
                risk_cap=float(risk_cap),
                drawdown_cut=overlay_context["drawdown_cut"],
                ratio_cut=overlay_context["ratio_cut"],
                breadth_cut=overlay_context["breadth_cut"],
                fee_rate=args.fee_rate,
                slippage_rate=args.slippage_rate,
            )
            if candidate["code"] == overlay_context["treasury_code"] and abs(float(risk_cap) - float(overlay_context["risk_cap"])) < 1e-12:
                strategy = str(overlay_context["strategy"])
            else:
                strategy = (
                    f"{market_context['strategy']}__cashguard_{candidate['code']}"
                    f"_vr{int(round(float(overlay_context['ratio_cut']) * 100)):02d}"
                    f"_vb{int(round((float(overlay_context['breadth_cut']) + 0.02) * 100)):02d}"
                    f"_cap{int(round(float(risk_cap) * 100)):02d}"
                )
            summary = summarize(result, trades, selected)
            summary["strategy"] = strategy
            summary["defensive_code"] = str(candidate["code"])
            summary["risk_cap"] = float(risk_cap)
            if strategy != str(overlay_context["strategy"]):
                rows.append(summary)
                nav_compare[f"{strategy}_nav"] = result["nav"]
                descriptions[strategy] = (
                    f"保持当前 hybrid market proxy + 弱市切防守框架不变，仅把弱市防守资产替换为 {candidate['name']}。"
                    f"触发条件仍为市场量能20/60<{float(overlay_context['ratio_cut']):.0%} 且广度<{float(overlay_context['breadth_cut']):.1%}，"
                    f"风险资产上限压到 {float(risk_cap):.0%}。"
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
    write_json_atomic(
        BEST_PATH,
        {
            "baseline": baseline_summary,
            "valid_improvements": valid_df.to_dict(orient="records"),
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
                "defensive_code",
                "risk_cap",
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
    notify_df = valid_df.copy()
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

    if args.notify and not notify_df.empty:
        best = notify_df.sort_values(
            ["annualized_return", "sharpe_rf0", "max_drawdown_integral", "max_drawdown"],
            ascending=[False, False, True, False],
        ).iloc[0]
        webhook_url = args.webhook_url or os.getenv("DAILY_MONITOR_WEBHOOK_URL", DEFAULT_FEISHU_WEBHOOK)
        description = descriptions[str(best["strategy"])]
        send_improvement_notification(baseline_summary, str(best["strategy"]), best.to_dict(), description, webhook_url)
        save_notify_state(str(best["strategy"]), best.to_dict(), description)
        print(f"\nWebhook notified for {best['strategy']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
