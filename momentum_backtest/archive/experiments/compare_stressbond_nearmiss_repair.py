#!/usr/bin/env python3
"""Repair the current best stress-bond near-miss by tightening upper-layer caps around lower risk_cap variants."""

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
from compare_goal_optimizations import (
    DEFAULT_FEISHU_WEBHOOK,
    DEFAULT_REGIME_MIX_VOLUME_GUARD_RELIEF_BUFFER,
    DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN,
    METRIC_TOLERANCE,
    build_dynamic_core_target_weights,
    send_improvement_notification,
)
from compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
from compare_market_proxy_variants import build_proxy_catalog
from compare_simple_bond_overlay import load_notify_state as load_simple_notify_state
from compare_tail_risk_bond_overlay import load_market_proxy_best_context
from run_backtest import (
    DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
    RISK_CODES,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_stressbond_nearmiss_repair")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair the current lower-risk-cap stress-bond near-miss by slightly tightening upper-layer caps."
    )
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
    write_json_atomic(NOTIFY_STATE_PATH, {"strategy": strategy_name, "summary": summary, "description": description})


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


def build_target_weights(
    prices: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    vg_cap: float,
    overheat_cap: float,
    overheat_hi_cap: float,
) -> tuple[pd.DataFrame, pd.Series]:
    aggressive_weights, aggressive_momentum, _, _, aggressive_trend = build_dynamic_core_target_weights(
        prices,
        lookback=25,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        core_weight=float(params["aggressive_core_weight"]),
        hs300_mom60_cut=0.05,
        hs300_mom120_cut=0.10,
        hs300_ma_window=90,
    )
    conservative_weights, conservative_momentum, _, _, _ = build_dynamic_core_target_weights(
        prices,
        lookback=25,
        absolute_threshold=DEFAULT_ABSOLUTE_MOMENTUM_THRESHOLD,
        weak_trend_defensive_weight=DEFAULT_WEAK_TREND_DEFENSIVE_WEIGHT,
        core_weight=float(params["conservative_core_weight"]),
        hs300_mom60_cut=0.04,
        hs300_mom120_cut=0.10,
        hs300_ma_window=120,
    )
    aggressive_mask = aggressive_trend & (aggressive_momentum >= float(params["regime_momentum_cut"]))
    mixed_target_weights = conservative_weights.copy()
    mixed_target_weights.loc[aggressive_mask] = aggressive_weights.loc[aggressive_mask]
    mixed_momentum = conservative_momentum.copy()
    mixed_momentum.loc[aggressive_mask] = aggressive_momentum.loc[aggressive_mask]
    mixed_momentum = mixed_momentum.rename("current_momentum")

    proxy = proxy.reindex(prices.index).ffill()
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    row_risk_weight = mixed_target_weights[risk_codes].sum(axis=1)
    volume_weak_mask = (
        (proxy["market_amount_ratio_20_60"] < float(params["volume_ratio_cut"]))
        & (proxy["market_amount_ratio_5_20"] < float(params["volume_short_ratio_cut"]))
        & (proxy["market_breadth_proxy"] < float(params["volume_breadth_cut"]))
        & (mixed_momentum <= float(params["volume_guard_momentum_ceiling"]))
    ).fillna(False)

    weak_mask = volume_weak_mask & (row_risk_weight > vg_cap)
    if weak_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[weak_mask] = vg_cap / row_risk_weight.loc[weak_mask]
        mixed_target_weights.loc[weak_mask, risk_codes] = mixed_target_weights.loc[weak_mask, risk_codes].mul(
            scale.loc[weak_mask], axis=0
        )

    base_result, _ = run_target_weights_strategy(
        prices,
        pd.DataFrame({"code": list(prices.columns), "theme": list(prices.columns), "name": list(prices.columns)}),
        mixed_target_weights,
        mixed_momentum,
        DEFAULT_FEE_RATE,
        DEFAULT_SLIPPAGE_RATE,
    )
    post_guard_risk = mixed_target_weights[risk_codes].sum(axis=1)
    reduce_mask = (
        (post_guard_risk > overheat_cap)
        & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
        & (mixed_momentum >= float(params["overheat_momentum_cut"]))
    )
    if reduce_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[reduce_mask] = overheat_cap / post_guard_risk.loc[reduce_mask]
        mixed_target_weights.loc[reduce_mask, risk_codes] = mixed_target_weights.loc[reduce_mask, risk_codes].mul(
            scale.loc[reduce_mask], axis=0
        )

    post_overheat_risk = mixed_target_weights[risk_codes].sum(axis=1)
    high_reduce_mask = (
        (post_overheat_risk > overheat_hi_cap)
        & (base_result["drawdown"] >= float(params["overheat_drawdown_cut"]))
        & (mixed_momentum >= float(params["overheat_high_momentum_cut"]))
    )
    if high_reduce_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[high_reduce_mask] = overheat_hi_cap / post_overheat_risk.loc[high_reduce_mask]
        mixed_target_weights.loc[high_reduce_mask, risk_codes] = mixed_target_weights.loc[high_reduce_mask, risk_codes].mul(
            scale.loc[high_reduce_mask], axis=0
        )

    return mixed_target_weights, mixed_momentum


def apply_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    treasury_code: str,
    risk_cap: float,
    ratio_cut: float,
    breadth_cut: float,
    vg_cap: float,
    overheat_cap: float,
    overheat_hi_cap: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_weights, mixed_momentum = build_target_weights(
        prices=prices,
        proxy=proxy,
        params=params,
        vg_cap=vg_cap,
        overheat_cap=overheat_cap,
        overheat_hi_cap=overheat_hi_cap,
    )
    risk_codes = [code for code in RISK_CODES if code in prices.columns]
    overlaid_weights = base_weights.copy()
    proxy = proxy.reindex(prices.index).ffill()
    row_risk_weight = base_weights[risk_codes].sum(axis=1)

    trigger_mask = (
        (proxy["market_amount_ratio_20_60"] < ratio_cut)
        & (proxy["market_breadth_proxy"] < breadth_cut)
    ).fillna(False)
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

    market_context = load_market_proxy_best_context()
    overlay_context = load_overlay_context()
    params = dict(market_context["params"])
    params["volume_ratio_cut"] = float(market_context["volume_ratio_cut"])
    params["volume_short_ratio_cut"] = float(market_context["volume_short_ratio_cut"])
    params["volume_breadth_cut"] = float(market_context["volume_breadth_cut"])
    params["volume_guard_relief_buffer"] = float(
        params.get("volume_guard_relief_buffer", DEFAULT_REGIME_MIX_VOLUME_GUARD_RELIEF_BUFFER)
    )
    params["volume_guard_soft_span"] = float(params.get("volume_guard_soft_span", DEFAULT_REGIME_MIX_VOLUME_GUARD_SOFT_SPAN))
    drop_codes = [str(code) for code in params["drop_codes"]]

    base_selected = load_fixed_etf_pool()
    base_selected = base_selected[~base_selected["code"].astype(str).isin(drop_codes)].reset_index(drop=True)
    selected = pd.concat([base_selected, pd.DataFrame([ETF_511260])], ignore_index=True)
    if args.refresh:
        prices = fetch_histories(selected, years=args.years)
    else:
        _, prices = load_cached_data()
        start_ts = prices.index.max() - pd.DateOffset(years=args.years)
        prices = prices.loc[prices.index >= start_ts].copy()
        if ETF_511260["code"] not in prices.columns:
            extra = fetch_histories(pd.DataFrame([ETF_511260]), years=args.years)
            prices = prices.join(extra, how="outer")
        prices = prices[[code for code in selected["code"].astype(str) if code in prices.columns]].copy()

    from compare_goal_optimizations import load_market_volume_proxy

    base_market_proxy = load_market_volume_proxy(years=args.years, refresh=args.refresh)
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

    risk_cap_candidates = [0.15, 0.16, 0.17, 0.18]
    vg_cap_candidates = [0.29, 0.30]
    overheat_cap_candidates = [0.07, 0.08]
    overheat_hi_cap_candidates = [0.03, 0.04]
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
                        f"[stressbond_nearmiss_repair] {run_idx}/{total_runs} "
                        f"risk_cap={risk_cap:.2f} vg_cap={vg_cap:.2f} "
                        f"overheat={overheat_cap:.2f}/{overheat_hi_cap:.2f}",
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
                        f"{market_context['strategy']}__repair"
                        f"_rc{int(round(risk_cap * 100)):02d}"
                        f"_vg{int(round(vg_cap * 100)):02d}"
                        f"_oc{int(round(overheat_cap * 100)):02d}"
                        f"_oh{int(round(overheat_hi_cap * 100)):02d}"
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
                        "保持当前 hybrid market proxy + 十年国债弱市切债框架不变，"
                        f"仅围绕 lower risk cap near-miss 做上层收紧：风险仓上限 {risk_cap:.0%}，"
                        f"弱量能上限 {vg_cap:.0%}，过热上限 {overheat_cap:.0%}，极热上限 {overheat_hi_cap:.0%}。"
                    )

    summary_df = pd.DataFrame(rows)
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(baseline_summary["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(baseline_summary["sharpe_rf0"])
    summary_df["max_drawdown_integral_diff"] = summary_df["max_drawdown_integral"] - float(
        baseline_summary["max_drawdown_integral"]
    )
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
