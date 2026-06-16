#!/usr/bin/env python3
"""对比当前市场代理最佳策略之上的简单国债覆盖层规则。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from ..runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from compare_candidate_pool_additions import ETF_511090, ETF_511260
from compare_goal_optimizations import DEFAULT_FEISHU_WEBHOOK, METRIC_TOLERANCE, send_improvement_notification
from compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
from compare_market_proxy_variants import build_proxy_catalog, evaluate_strategy
from search_utils import (
    extract_valid_previous_summary,
    load_notify_state as load_notify_state_file,
    save_notify_state as save_notify_state_file,
    sort_notify_candidates,
    try_join_missing_candidate_histories,
    write_json_atomic,
)
from compare_tail_risk_bond_overlay import build_base_target_weights, load_market_proxy_best_context
from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    resolve_strategy_universe,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_simple_bond_overlay")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比当前市场代理最佳策略之上的简单国债覆盖层规则。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取 ETF 历史数据，而不是复用本地缓存。")
    parser.add_argument("--notify", action="store_true", help="兼容旧参数；当前默认已开启找到更优策略时的 webhook 通知。")
    parser.add_argument("--disable-notify", action="store_true", help="关闭找到更优策略时的 webhook 通知。")
    parser.add_argument("--webhook-url", type=str, default="", help="临时指定 webhook。")
    return parser.parse_args()


def load_notify_state() -> dict[str, object] | None:
    return load_notify_state_file(NOTIFY_STATE_PATH)


def save_notify_state(strategy_name: str, summary: dict[str, object], description: str) -> None:
    save_notify_state_file(NOTIFY_STATE_PATH, strategy_name, summary, description)


def apply_simple_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    treasury_code: str,
    mode: str,
    risk_cap: float,
    drawdown_cut: float | None,
    ratio_cut: float | None,
    breadth_cut: float | None,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_weights, mixed_momentum, base_result = build_base_target_weights(prices, proxy, params)
    configured_risk_codes = params.get("risk_codes")
    configured_defensive_codes = params.get("defensive_codes")
    active_risk_codes, _ = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in configured_risk_codes] if configured_risk_codes is not None else None,
        defensive_codes=[str(code) for code in configured_defensive_codes] if configured_defensive_codes is not None else None,
    )
    risk_budget_codes = list(active_risk_codes)
    if "510300" in prices.columns and "510300" not in risk_budget_codes:
        risk_budget_codes.append("510300")
    overlaid_weights = base_weights.copy()
    proxy = proxy.reindex(prices.index).ffill()
    row_risk_weight = base_weights[risk_budget_codes].sum(axis=1)

    if mode == "drawdown_only":
        assert drawdown_cut is not None
        trigger_mask = (base_result["drawdown"] <= drawdown_cut).fillna(False)
    elif mode == "market_stress_only":
        assert ratio_cut is not None and breadth_cut is not None
        trigger_mask = (
            (proxy["market_amount_ratio_20_60"] < ratio_cut)
            & (proxy["market_breadth_proxy"] < breadth_cut)
        ).fillna(False)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    scale_mask = trigger_mask & (row_risk_weight > risk_cap)
    if scale_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[scale_mask] = risk_cap / row_risk_weight.loc[scale_mask]
        overlaid_weights.loc[scale_mask, risk_budget_codes] = overlaid_weights.loc[scale_mask, risk_budget_codes].mul(
            scale.loc[scale_mask], axis=0
        )
        moved_weight = row_risk_weight.loc[scale_mask] - risk_cap
        overlaid_weights.loc[scale_mask, treasury_code] = overlaid_weights.loc[scale_mask, treasury_code].add(moved_weight, fill_value=0.0)

    return run_target_weights_strategy(prices, selected, overlaid_weights, mixed_momentum, fee_rate, slippage_rate)


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    context = load_market_proxy_best_context()
    params = dict(context["params"])
    params["volume_ratio_cut"] = float(context["volume_ratio_cut"])
    params["volume_short_ratio_cut"] = float(context["volume_short_ratio_cut"])
    params["volume_breadth_cut"] = float(context["volume_breadth_cut"])
    drop_codes = [str(code) for code in params["drop_codes"]]

    base_selected = load_fixed_etf_pool()
    base_selected = base_selected[~base_selected["code"].astype(str).isin(drop_codes)].reset_index(drop=True)
    treasury_candidates = [ETF_511260, ETF_511090]
    treasury_candidate_df = pd.DataFrame(treasury_candidates)
    if args.refresh:
        selected_for_prices = pd.concat([base_selected, treasury_candidate_df], ignore_index=True)
        prices = fetch_histories(selected_for_prices, years=args.years)
        skipped_treasury_codes: list[str] = []
    else:
        _, prices = load_cached_data()
        start_ts = prices.index.max() - pd.DateOffset(years=args.years)
        prices = prices.loc[prices.index >= start_ts].copy()
        selected_for_prices = pd.concat([base_selected, treasury_candidate_df], ignore_index=True)
        prices, skipped_treasury_codes, fetch_error = try_join_missing_candidate_histories(
            prices=prices,
            candidates=treasury_candidate_df,
            years=args.years,
            fetch_fn=fetch_histories,
        )
        if skipped_treasury_codes:
            skipped_text = ", ".join(skipped_treasury_codes)
            if fetch_error:
                print(f"[warn] skip treasury candidates without local history: {skipped_text}; fetch failed: {fetch_error}")
            else:
                print(f"[warn] skip treasury candidates without usable history: {skipped_text}")
        prices = prices[[code for code in selected_for_prices["code"].astype(str) if code in prices.columns]].copy()
    available_treasury_candidates = [row for row in treasury_candidates if str(row["code"]) in prices.columns]
    if not available_treasury_candidates:
        print("[warn] no treasury candidates available; only baseline result will be generated")
    if str(ETF_511260["code"]) not in prices.columns:
        raise RuntimeError(f"missing required baseline treasury history: {ETF_511260['code']} {ETF_511260['name']}")

    from compare_goal_optimizations import load_market_volume_proxy

    base_market_proxy = load_market_volume_proxy(years=args.years, refresh=args.refresh)
    proxy_catalog = {
        item["name"]: item["proxy"]
        for item in build_proxy_catalog(
            base_market_proxy,
            prices,
            risk_codes=[str(code) for code in params.get("risk_codes", [])] if params.get("risk_codes") is not None else None,
        )
    }
    proxy = proxy_catalog[str(context["proxy_kind"])]

    rows: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}
    nav_compare = pd.DataFrame(index=prices.index)

    baseline_selected = pd.concat([base_selected, pd.DataFrame([ETF_511260])], ignore_index=True)
    baseline_selected = baseline_selected.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)
    baseline_prices = prices[[code for code in baseline_selected["code"].astype(str) if code in prices.columns]].copy()
    baseline_result, baseline_trades = evaluate_strategy(
        selected=baseline_selected,
        prices=baseline_prices,
        market_proxy=proxy,
        params=params,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_summary = summarize(baseline_result, baseline_trades, baseline_selected)
    baseline_summary["strategy"] = str(context["strategy"])
    rows.append(baseline_summary)
    nav_compare[f"{context['strategy']}_nav"] = baseline_result["nav"]
    descriptions[str(context["strategy"])] = str(context["description"])

    for treasury in available_treasury_candidates:
        selected = pd.concat([base_selected, pd.DataFrame([treasury])], ignore_index=True)
        candidate_prices = prices[[code for code in selected["code"].astype(str) if code in prices.columns]].copy()
        for risk_cap in (0.20, 0.30, 0.40, 0.50, 0.60):
            for drawdown_cut in (-0.03, -0.05, -0.08, -0.10):
                result, trades = apply_simple_overlay(
                    prices=candidate_prices,
                    selected=selected,
                    proxy=proxy,
                    params=params,
                    treasury_code=str(treasury["code"]),
                    mode="drawdown_only",
                    risk_cap=risk_cap,
                    drawdown_cut=drawdown_cut,
                    ratio_cut=None,
                    breadth_cut=None,
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                )
                strategy = (
                    f"{context['strategy']}__drawbond_{treasury['code']}"
                    f"_dd{int(round(abs(drawdown_cut) * 100)):02d}_cap{int(round(risk_cap * 100)):02d}"
                )
                summary = summarize(result, trades, selected)
                summary["strategy"] = strategy
                summary["treasury_code"] = str(treasury["code"])
                summary["mode"] = "drawdown_only"
                summary["risk_cap"] = risk_cap
                summary["drawdown_cut"] = drawdown_cut
                rows.append(summary)
                nav_compare[f"{strategy}_nav"] = result["nav"]
                descriptions[strategy] = (
                    f"保持最新 hybrid market proxy 策略不变，仅加简单回撤切债：当策略回撤<={drawdown_cut:.0%} 时，"
                    f"把风险资产上限压到 {risk_cap:.0%}，削减部分切到 {treasury['name']}。"
                )
            for ratio_cut in (0.88, 0.90, 0.92):
                for breadth_cut in (-0.03, -0.02, -0.01):
                    result, trades = apply_simple_overlay(
                        prices=candidate_prices,
                        selected=selected,
                        proxy=proxy,
                        params=params,
                        treasury_code=str(treasury["code"]),
                        mode="market_stress_only",
                        risk_cap=risk_cap,
                        drawdown_cut=None,
                        ratio_cut=ratio_cut,
                        breadth_cut=breadth_cut,
                        fee_rate=args.fee_rate,
                        slippage_rate=args.slippage_rate,
                    )
                    strategy = (
                        f"{context['strategy']}__stressbond_{treasury['code']}"
                        f"_vr{int(round(ratio_cut * 100)):02d}_vb{int(round((breadth_cut + 0.02) * 100)):02d}"
                        f"_cap{int(round(risk_cap * 100)):02d}"
                    )
                    summary = summarize(result, trades, selected)
                    summary["strategy"] = strategy
                    summary["treasury_code"] = str(treasury["code"])
                    summary["mode"] = "market_stress_only"
                    summary["risk_cap"] = risk_cap
                    summary["ratio_cut"] = ratio_cut
                    summary["breadth_cut"] = breadth_cut
                    rows.append(summary)
                    nav_compare[f"{strategy}_nav"] = result["nav"]
                    descriptions[strategy] = (
                        f"保持最新 hybrid market proxy 策略不变，仅加简单市场压力切债：当市场量能20/60<{ratio_cut:.0%} 且广度<{breadth_cut:.1%} 时，"
                        f"把风险资产上限压到 {risk_cap:.0%}，削减部分切到 {treasury['name']}。"
                    )

    summary_df = pd.DataFrame(rows)
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(baseline_summary["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(baseline_summary["sharpe_rf0"])
    summary_df["max_drawdown_integral_diff"] = summary_df["max_drawdown_integral"] - float(baseline_summary["max_drawdown_integral"])
    summary_df["is_valid_change"] = (
        (summary_df["strategy"] != str(context["strategy"]))
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

    valid_df = sort_notify_candidates(summary_df[summary_df["is_valid_change"]].copy())
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
    print(summary_df[["strategy", "mode", "treasury_code", "annualized_return", "sharpe_rf0", "max_drawdown_integral", "annualized_diff", "sharpe_diff", "max_drawdown_integral_diff", "is_valid_change"]].head(20).to_string(index=False))

    notify_state = load_notify_state()
    notify_df = valid_df.copy()
    previous_summary = extract_valid_previous_summary(notify_state)
    if previous_summary is not None:
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

    should_notify = args.notify or not args.disable_notify
    if should_notify and not notify_df.empty:
        best = sort_notify_candidates(notify_df).iloc[0]
        webhook_url = args.webhook_url or os.getenv("DAILY_MONITOR_WEBHOOK_URL", DEFAULT_FEISHU_WEBHOOK)
        description = descriptions[str(best["strategy"])]
        send_improvement_notification(baseline_summary, str(best["strategy"]), best.to_dict(), description, webhook_url)
        save_notify_state(str(best["strategy"]), best.to_dict(), description)
        print(f"\nWebhook notified for {best['strategy']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
