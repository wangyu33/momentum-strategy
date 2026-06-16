#!/usr/bin/env python3
"""Compact two-tier defensive overlay search around the current 10Y treasury winner."""

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

from compare_candidate_pool_additions import ETF_511260
from compare_goal_optimizations import DEFAULT_FEISHU_WEBHOOK, METRIC_TOLERANCE, send_improvement_notification
from compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
from compare_market_proxy_variants import build_proxy_catalog
from compare_simple_bond_overlay import load_notify_state as load_simple_notify_state
from compare_tail_risk_bond_overlay import build_base_target_weights, load_market_proxy_best_context
from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_two_tier_defense_compact")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact two-tier defensive overlay search.")
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


def load_overlay_context() -> dict[str, object]:
    payload = load_simple_notify_state()
    if payload is None:
        raise RuntimeError("missing simple bond overlay notify state")
    summary = dict(payload["summary"])
    return {
        "strategy": str(payload["strategy"]),
        "summary": summary,
        "description": str(payload["description"]),
        "risk_cap": float(summary["risk_cap"]),
        "ratio_cut": float(summary["ratio_cut"]),
        "breadth_cut": float(summary["breadth_cut"]),
        "treasury_code": str(summary["treasury_code"]),
    }


def apply_two_tier_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    treasury_code: str,
    weak_ratio_cut: float,
    weak_breadth_cut: float,
    weak_cap: float,
    extreme_ratio_cut: float,
    extreme_breadth_cut: float,
    extreme_cap: float,
    extreme_short_ratio_cut: float | None,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_weights, mixed_momentum, _ = build_base_target_weights(prices, proxy, params)
    risk_codes = [code for code in ["510300", "159949", "159954", "159941", "513650", "513880"] if code in prices.columns]
    overlaid_weights = base_weights.copy()
    proxy = proxy.reindex(prices.index).ffill()
    row_risk_weight = base_weights[risk_codes].sum(axis=1)

    weak_mask = (
        (proxy["market_amount_ratio_20_60"] < weak_ratio_cut)
        & (proxy["market_breadth_proxy"] < weak_breadth_cut)
    ).fillna(False)
    extreme_mask = (
        (proxy["market_amount_ratio_20_60"] < extreme_ratio_cut)
        & (proxy["market_breadth_proxy"] < extreme_breadth_cut)
    ).fillna(False)
    if extreme_short_ratio_cut is not None:
        extreme_mask = extreme_mask & (proxy["market_amount_ratio_5_20"] < extreme_short_ratio_cut).fillna(False)

    weak_only_mask = weak_mask & ~extreme_mask

    if weak_only_mask.any():
        weak_scale_mask = weak_only_mask & (row_risk_weight > weak_cap)
        if weak_scale_mask.any():
            scale = pd.Series(1.0, index=prices.index, dtype="float64")
            scale.loc[weak_scale_mask] = weak_cap / row_risk_weight.loc[weak_scale_mask]
            overlaid_weights.loc[weak_scale_mask, risk_codes] = overlaid_weights.loc[weak_scale_mask, risk_codes].mul(
                scale.loc[weak_scale_mask], axis=0
            )
            moved_weight = row_risk_weight.loc[weak_scale_mask] - weak_cap
            overlaid_weights.loc[weak_scale_mask, treasury_code] = overlaid_weights.loc[weak_scale_mask, treasury_code].add(
                moved_weight,
                fill_value=0.0,
            )

    if extreme_mask.any():
        extreme_scale_mask = extreme_mask & (row_risk_weight > extreme_cap)
        if extreme_scale_mask.any():
            scale = pd.Series(1.0, index=prices.index, dtype="float64")
            scale.loc[extreme_scale_mask] = extreme_cap / row_risk_weight.loc[extreme_scale_mask]
            overlaid_weights.loc[extreme_scale_mask, risk_codes] = overlaid_weights.loc[extreme_scale_mask, risk_codes].mul(
                scale.loc[extreme_scale_mask], axis=0
            )
            moved_weight = row_risk_weight.loc[extreme_scale_mask] - extreme_cap
            overlaid_weights.loc[extreme_scale_mask, treasury_code] = overlaid_weights.loc[extreme_scale_mask, treasury_code].add(
                moved_weight,
                fill_value=0.0,
            )

    return run_target_weights_strategy(prices, selected, overlaid_weights, mixed_momentum, fee_rate, slippage_rate)


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
    selected_for_prices = pd.concat([base_selected, pd.DataFrame([ETF_511260])], ignore_index=True)
    if args.refresh:
        prices = fetch_histories(selected_for_prices, years=args.years)
    else:
        _, prices = load_cached_data()
        start_ts = prices.index.max() - pd.DateOffset(years=args.years)
        prices = prices.loc[prices.index >= start_ts].copy()
        if ETF_511260["code"] not in prices.columns:
            extra = fetch_histories(pd.DataFrame([ETF_511260]), years=args.years)
            prices = prices.join(extra, how="outer")
        prices = prices[[code for code in selected_for_prices["code"].astype(str) if code in prices.columns]].copy()

    from compare_goal_optimizations import load_market_volume_proxy

    base_market_proxy = load_market_volume_proxy(years=args.years, refresh=args.refresh)
    proxy_catalog = {item["name"]: item["proxy"] for item in build_proxy_catalog(base_market_proxy, prices)}
    proxy = proxy_catalog[str(market_context["proxy_kind"])]

    rows: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}

    baseline_selected = pd.concat([base_selected, pd.DataFrame([ETF_511260])], ignore_index=True)
    baseline_prices = prices[[code for code in baseline_selected["code"].astype(str) if code in prices.columns]].copy()
    from compare_simple_bond_overlay import apply_simple_overlay

    baseline_result, baseline_trades = apply_simple_overlay(
        prices=baseline_prices,
        selected=baseline_selected,
        proxy=proxy,
        params=params,
        treasury_code=overlay_context["treasury_code"],
        mode="market_stress_only",
        risk_cap=overlay_context["risk_cap"],
        drawdown_cut=None,
        ratio_cut=overlay_context["ratio_cut"],
        breadth_cut=overlay_context["breadth_cut"],
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_summary = summarize(baseline_result, baseline_trades, baseline_selected)
    baseline_summary["strategy"] = overlay_context["strategy"]
    rows.append(baseline_summary)
    descriptions[str(overlay_context["strategy"])] = overlay_context["description"]

    candidate_specs = [
        {"weak_cap": 0.25, "weak_ratio": 0.90, "weak_breadth": -0.03, "extreme_cap": 0.15, "extreme_ratio": 0.89, "extreme_breadth": -0.035, "extreme_short": None},
        {"weak_cap": 0.25, "weak_ratio": 0.90, "weak_breadth": -0.03, "extreme_cap": 0.10, "extreme_ratio": 0.89, "extreme_breadth": -0.035, "extreme_short": None},
        {"weak_cap": 0.25, "weak_ratio": 0.90, "weak_breadth": -0.03, "extreme_cap": 0.15, "extreme_ratio": 0.885, "extreme_breadth": -0.04, "extreme_short": None},
        {"weak_cap": 0.25, "weak_ratio": 0.90, "weak_breadth": -0.03, "extreme_cap": 0.10, "extreme_ratio": 0.885, "extreme_breadth": -0.04, "extreme_short": None},
        {"weak_cap": 0.30, "weak_ratio": 0.90, "weak_breadth": -0.03, "extreme_cap": 0.15, "extreme_ratio": 0.89, "extreme_breadth": -0.035, "extreme_short": None},
        {"weak_cap": 0.30, "weak_ratio": 0.90, "weak_breadth": -0.03, "extreme_cap": 0.10, "extreme_ratio": 0.89, "extreme_breadth": -0.035, "extreme_short": None},
        {"weak_cap": 0.25, "weak_ratio": 0.905, "weak_breadth": -0.025, "extreme_cap": 0.15, "extreme_ratio": 0.89, "extreme_breadth": -0.035, "extreme_short": None},
        {"weak_cap": 0.25, "weak_ratio": 0.905, "weak_breadth": -0.025, "extreme_cap": 0.10, "extreme_ratio": 0.89, "extreme_breadth": -0.035, "extreme_short": None},
        {"weak_cap": 0.25, "weak_ratio": 0.90, "weak_breadth": -0.03, "extreme_cap": 0.15, "extreme_ratio": 0.89, "extreme_breadth": -0.035, "extreme_short": 0.95},
        {"weak_cap": 0.30, "weak_ratio": 0.90, "weak_breadth": -0.03, "extreme_cap": 0.15, "extreme_ratio": 0.89, "extreme_breadth": -0.035, "extreme_short": 0.95},
        {"weak_cap": 0.25, "weak_ratio": 0.90, "weak_breadth": -0.03, "extreme_cap": 0.15, "extreme_ratio": 0.895, "extreme_breadth": -0.035, "extreme_short": None},
        {"weak_cap": 0.25, "weak_ratio": 0.90, "weak_breadth": -0.03, "extreme_cap": 0.10, "extreme_ratio": 0.895, "extreme_breadth": -0.035, "extreme_short": None},
    ]

    for spec in candidate_specs:
        result, trades = apply_two_tier_overlay(
            prices=baseline_prices,
            selected=baseline_selected,
            proxy=proxy,
            params=params,
            treasury_code=overlay_context["treasury_code"],
            weak_ratio_cut=float(spec["weak_ratio"]),
            weak_breadth_cut=float(spec["weak_breadth"]),
            weak_cap=float(spec["weak_cap"]),
            extreme_ratio_cut=float(spec["extreme_ratio"]),
            extreme_breadth_cut=float(spec["extreme_breadth"]),
            extreme_cap=float(spec["extreme_cap"]),
            extreme_short_ratio_cut=spec["extreme_short"],
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        short_tag = "na" if spec["extreme_short"] is None else f"{int(round(float(spec['extreme_short']) * 100)):02d}"
        strategy = (
            f"{market_context['strategy']}__twotier_511260"
            f"_wcap{int(round(float(spec['weak_cap']) * 100)):02d}"
            f"_ecap{int(round(float(spec['extreme_cap']) * 100)):02d}"
            f"_evr{int(round(float(spec['extreme_ratio']) * 1000)):03d}"
            f"_evb{int(round((float(spec['extreme_breadth']) + 0.02) * 1000)):03d}"
            f"_evs{short_tag}"
        )
        summary = summarize(result, trades, baseline_selected)
        summary["strategy"] = strategy
        summary["weak_ratio_cut"] = float(spec["weak_ratio"])
        summary["weak_breadth_cut"] = float(spec["weak_breadth"])
        summary["weak_cap"] = float(spec["weak_cap"])
        summary["extreme_ratio_cut"] = float(spec["extreme_ratio"])
        summary["extreme_breadth_cut"] = float(spec["extreme_breadth"])
        summary["extreme_cap"] = float(spec["extreme_cap"])
        summary["extreme_short_ratio_cut"] = spec["extreme_short"]
        rows.append(summary)
        short_desc = (
            "不加短期量能过滤"
            if spec["extreme_short"] is None
            else f"且极弱市要求5/20<{float(spec['extreme_short']):.0%}"
        )
        descriptions[strategy] = (
            f"保持当前 hybrid market proxy + 十年国债防守结构不变，改成双层防守："
            f"弱市在量能20/60<{float(spec['weak_ratio']):.0%} 且广度<{float(spec['weak_breadth']):.1%} 时，把风险仓压到 {float(spec['weak_cap']):.0%}；"
            f"极弱市在量能20/60<{float(spec['extreme_ratio']):.1%} 且广度<{float(spec['extreme_breadth']):.1%} 时，把风险仓进一步压到 {float(spec['extreme_cap']):.0%}；"
            f"{short_desc}。"
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
                "weak_cap",
                "extreme_cap",
                "extreme_ratio_cut",
                "extreme_breadth_cut",
                "extreme_short_ratio_cut",
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
