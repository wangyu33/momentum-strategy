#!/usr/bin/env python3
"""围绕十年国债防守覆盖层做触发阈值微调的历史实验脚本。"""

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


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_defensive_trigger_refinements")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine nearby trigger parameters around current 10Y treasury overlay.")
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


def apply_refined_overlay(
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
    nav_compare = pd.DataFrame(index=prices.index)

    baseline_selected = pd.concat([base_selected, pd.DataFrame([ETF_511260])], ignore_index=True)
    baseline_prices = prices[[code for code in baseline_selected["code"].astype(str) if code in prices.columns]].copy()
    baseline_result, baseline_trades = apply_refined_overlay(
        prices=baseline_prices,
        selected=baseline_selected,
        proxy=proxy,
        params=params,
        treasury_code=overlay_context["treasury_code"],
        risk_cap=overlay_context["risk_cap"],
        ratio_cut=overlay_context["ratio_cut"],
        breadth_cut=overlay_context["breadth_cut"],
        short_ratio_cut=None,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_summary = summarize(baseline_result, baseline_trades, baseline_selected)
    baseline_summary["strategy"] = overlay_context["strategy"]
    rows.append(baseline_summary)
    nav_compare[f"{overlay_context['strategy']}_nav"] = baseline_result["nav"]
    descriptions[str(overlay_context["strategy"])] = overlay_context["description"]

    ratio_candidates = sorted({overlay_context["ratio_cut"] - 0.01, overlay_context["ratio_cut"] - 0.005, overlay_context["ratio_cut"], overlay_context["ratio_cut"] + 0.005})
    breadth_candidates = sorted({overlay_context["breadth_cut"] - 0.01, overlay_context["breadth_cut"] - 0.005, overlay_context["breadth_cut"], overlay_context["breadth_cut"] + 0.005})
    risk_cap_candidates = sorted({overlay_context["risk_cap"] - 0.05, overlay_context["risk_cap"], overlay_context["risk_cap"] + 0.05})
    short_ratio_candidates: list[float | None] = [None, 0.95, 0.90]

    ratio_candidates = [x for x in ratio_candidates if 0.85 <= x <= 0.93]
    breadth_candidates = [x for x in breadth_candidates if -0.05 <= x <= -0.015]
    risk_cap_candidates = [x for x in risk_cap_candidates if 0.10 <= x <= 0.30]

    for ratio_cut in ratio_candidates:
        for breadth_cut in breadth_candidates:
            for risk_cap in risk_cap_candidates:
                for short_ratio_cut in short_ratio_candidates:
                    result, trades = apply_refined_overlay(
                        prices=baseline_prices,
                        selected=baseline_selected,
                        proxy=proxy,
                        params=params,
                        treasury_code=overlay_context["treasury_code"],
                        risk_cap=float(risk_cap),
                        ratio_cut=float(ratio_cut),
                        breadth_cut=float(breadth_cut),
                        short_ratio_cut=short_ratio_cut,
                        fee_rate=args.fee_rate,
                        slippage_rate=args.slippage_rate,
                    )
                    short_tag = "na" if short_ratio_cut is None else f"{int(round(short_ratio_cut * 100)):02d}"
                    strategy = (
                        f"{market_context['strategy']}__stressbond_511260"
                        f"_vr{int(round(ratio_cut * 1000)):03d}"
                        f"_vb{int(round((breadth_cut + 0.02) * 1000)):03d}"
                        f"_cap{int(round(risk_cap * 100)):02d}"
                        f"_vs{short_tag}"
                    )
                    summary = summarize(result, trades, baseline_selected)
                    summary["strategy"] = strategy
                    summary["ratio_cut"] = float(ratio_cut)
                    summary["breadth_cut"] = float(breadth_cut)
                    summary["risk_cap"] = float(risk_cap)
                    summary["short_ratio_cut"] = short_ratio_cut
                    rows.append(summary)
                    nav_compare[f"{strategy}_nav"] = result["nav"]
                    short_desc = (
                        "不加短期量能过滤"
                        if short_ratio_cut is None
                        else f"且短期量能5/20<{float(short_ratio_cut):.0%}"
                    )
                    descriptions[strategy] = (
                        f"保持当前 hybrid market proxy + 十年国债防守结构不变，仅微调弱市触发阈值："
                        f"市场量能20/60<{float(ratio_cut):.1%}、广度<{float(breadth_cut):.1%}、{short_desc}，"
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
