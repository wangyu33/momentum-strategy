#!/usr/bin/env python3
"""对比当前市场代理最佳策略之上的尾部风险国债覆盖层。"""

from __future__ import annotations

import argparse
import re

try:
    from .runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

try:
    from .compare_candidate_pool_additions import ETF_511090, ETF_511260
    from .compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
    from .compare_goal_optimizations import (
        DEFAULT_FEISHU_WEBHOOK,
        GOAL_OUTPUT_DIR,
        METRIC_TOLERANCE,
        parse_regime_mix_strategy_name,
        send_improvement_notification,
    )
    from .compare_market_proxy_variants import build_proxy_catalog
    from .official_strategy_core import build_official_target_weights
    from .search_utils import (
        add_notify_cli_args,
        load_incremental_notify_candidates,
        load_preferred_strategy_payload,
        notify_best_candidate,
        sort_notify_candidates,
        try_join_missing_candidate_histories,
        write_json_atomic,
    )
except ImportError:
    from compare_candidate_pool_additions import ETF_511090, ETF_511260
    from compare_hs300_regime_fixes import load_cached_data, run_target_weights_strategy, summarize
    from compare_goal_optimizations import (
        DEFAULT_FEISHU_WEBHOOK,
        GOAL_OUTPUT_DIR,
        METRIC_TOLERANCE,
        parse_regime_mix_strategy_name,
        send_improvement_notification,
    )
    from compare_market_proxy_variants import build_proxy_catalog
    from official_strategy_core import build_official_target_weights
    from search_utils import (
        add_notify_cli_args,
        load_incremental_notify_candidates,
        load_preferred_strategy_payload,
        notify_best_candidate,
        sort_notify_candidates,
        try_join_missing_candidate_histories,
        write_json_atomic,
    )

from run_backtest import (
    DEFAULT_BASELINE_DROP_CODES,
    DEFENSIVE_CODES,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RESEARCH_OUTPUT_DIR,
    RISK_CODES,
    build_default_strategy_params,
    resolve_strategy_universe,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "tail_risk_bond_overlay"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"
MARKET_PROXY_NOTIFY_STATE_PATH = GOAL_OUTPUT_DIR / "market_proxy_variants" / "notify_state.json"
MARKET_PROXY_BEST_PATH = GOAL_OUTPUT_DIR / "market_proxy_variants" / "best.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比当前市场代理最佳策略之上的尾部风险国债覆盖层。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取 ETF 历史数据，而不是复用本地缓存。")
    add_notify_cli_args(parser)
    return parser.parse_args()


def load_market_proxy_best_context() -> dict[str, object]:
    payload = load_preferred_strategy_payload(MARKET_PROXY_NOTIFY_STATE_PATH, MARKET_PROXY_BEST_PATH)
    if payload is None:
        raise RuntimeError("missing market proxy context")
    strategy_name = str(payload["strategy"])
    summary = dict(payload.get("summary", {}))
    params = parse_regime_mix_strategy_name(strategy_name)
    if params is None:
        required_keys = {"proxy_kind", "volume_ratio_cut", "volume_short_ratio_cut", "volume_breadth_cut"}
        if not required_keys.issubset(summary):
            raise RuntimeError(f"unsupported market proxy strategy: {strategy_name}")
        params = build_default_strategy_params(drop_codes=list(DEFAULT_BASELINE_DROP_CODES))
        proxy_kind = str(summary["proxy_kind"])
        volume_ratio_cut = float(summary["volume_ratio_cut"])
        volume_short_ratio_cut = float(summary["volume_short_ratio_cut"])
        volume_breadth_cut = float(summary["volume_breadth_cut"])
    else:
        match = re.search(r"__proxy_(.+)_vr(\d+)_vs(\d+)_vb(-?\d+)$", strategy_name)
        if match is None:
            raise RuntimeError(f"failed to parse proxy suffix from {strategy_name}")
        proxy_kind = str(match.group(1))
        volume_ratio_cut = int(match.group(2)) / 100
        volume_short_ratio_cut = int(match.group(3)) / 100
        volume_breadth_cut = int(match.group(4)) / 100 - 0.02
    return {
        "strategy": strategy_name,
        "summary": summary,
        "description": payload.get("description", ""),
        "params": params,
        "proxy_kind": proxy_kind,
        "volume_ratio_cut": volume_ratio_cut,
        "volume_short_ratio_cut": volume_short_ratio_cut,
        "volume_breadth_cut": volume_breadth_cut,
    }


def build_base_target_weights(
    prices: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """兼容旧研究脚本的薄包装。

    正式主链实现已收口到 official_strategy_core.build_official_target_weights。
    这里保留同名函数，避免历史研究脚本全面改动。
    """
    return build_official_target_weights(prices, proxy, params)


def apply_tail_bond_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    treasury_code: str,
    tail_ratio_cut: float,
    tail_short_ratio_cut: float,
    tail_breadth_cut: float,
    tail_momentum_ceiling: float,
    tail_drawdown_cut: float,
    tail_risk_cap: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_weights, mixed_momentum, base_result = build_base_target_weights(prices, proxy, params)
    active_risk_codes, _ = resolve_strategy_universe(
        prices,
        risk_codes=[str(code) for code in params.get("risk_codes", RISK_CODES)],
        defensive_codes=[str(code) for code in params.get("defensive_codes", DEFENSIVE_CODES)],
    )
    risk_budget_codes = list(active_risk_codes)
    if "510300" in prices.columns and "510300" not in risk_budget_codes:
        risk_budget_codes.append("510300")
    overlaid_weights = base_weights.copy()
    proxy = proxy.reindex(prices.index).ffill()
    row_risk_weight = base_weights[risk_budget_codes].sum(axis=1)

    tail_mask = (
        (proxy["market_amount_ratio_20_60"] < tail_ratio_cut)
        & (proxy["market_amount_ratio_5_20"] < tail_short_ratio_cut)
        & (proxy["market_breadth_proxy"] < tail_breadth_cut)
        & (mixed_momentum <= tail_momentum_ceiling)
        & (base_result["drawdown"] <= tail_drawdown_cut)
    ).fillna(False)
    scale_mask = tail_mask & (row_risk_weight > tail_risk_cap)
    if scale_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[scale_mask] = tail_risk_cap / row_risk_weight.loc[scale_mask]
        overlaid_weights.loc[scale_mask, risk_budget_codes] = overlaid_weights.loc[scale_mask, risk_budget_codes].mul(
            scale.loc[scale_mask], axis=0
        )
        moved_weight = row_risk_weight.loc[scale_mask] - tail_risk_cap
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
    proxy_catalog = {item["name"]: item["proxy"] for item in build_proxy_catalog(base_market_proxy, prices)}
    if str(context["proxy_kind"]) not in proxy_catalog:
        raise RuntimeError(f"missing proxy kind {context['proxy_kind']}")
    proxy = proxy_catalog[str(context["proxy_kind"])]

    rows: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}
    nav_compare = pd.DataFrame(index=prices.index)

    baseline_selected = pd.concat([base_selected, pd.DataFrame([ETF_511260])], ignore_index=True)
    baseline_selected = baseline_selected.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)
    baseline_prices = prices[[code for code in baseline_selected["code"].astype(str) if code in prices.columns]].copy()
    from compare_market_proxy_variants import evaluate_strategy

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
        for tail_ratio_cut in (0.88, 0.90):
            for tail_short_ratio_cut in (0.88, 0.90):
                for tail_breadth_cut in (-0.03, -0.02):
                    for tail_momentum_ceiling in (0.06, 0.08):
                        for tail_drawdown_cut in (-0.06, -0.08):
                            for tail_risk_cap in (0.30, 0.50, 0.70):
                                result, trades = apply_tail_bond_overlay(
                                    prices=candidate_prices,
                                    selected=selected,
                                    proxy=proxy,
                                    params=params,
                                    treasury_code=str(treasury["code"]),
                                    tail_ratio_cut=tail_ratio_cut,
                                    tail_short_ratio_cut=tail_short_ratio_cut,
                                    tail_breadth_cut=tail_breadth_cut,
                                    tail_momentum_ceiling=tail_momentum_ceiling,
                                    tail_drawdown_cut=tail_drawdown_cut,
                                    tail_risk_cap=tail_risk_cap,
                                    fee_rate=args.fee_rate,
                                    slippage_rate=args.slippage_rate,
                                )
                                strategy = (
                                    f"{context['strategy']}__tailbond_{treasury['code']}"
                                    f"_vr{int(round(tail_ratio_cut * 100)):02d}"
                                    f"_vs{int(round(tail_short_ratio_cut * 100)):02d}"
                                    f"_vb{int(round((tail_breadth_cut + 0.02) * 100)):02d}"
                                    f"_tm{int(round(tail_momentum_ceiling * 100)):02d}"
                                    f"_dd{int(round(abs(tail_drawdown_cut) * 100)):02d}"
                                    f"_cap{int(round(tail_risk_cap * 100)):02d}"
                                )
                                summary = summarize(result, trades, selected)
                                summary["strategy"] = strategy
                                summary["treasury_code"] = str(treasury["code"])
                                summary["tail_ratio_cut"] = tail_ratio_cut
                                summary["tail_short_ratio_cut"] = tail_short_ratio_cut
                                summary["tail_breadth_cut"] = tail_breadth_cut
                                summary["tail_momentum_ceiling"] = tail_momentum_ceiling
                                summary["tail_drawdown_cut"] = tail_drawdown_cut
                                summary["tail_risk_cap"] = tail_risk_cap
                                rows.append(summary)
                                nav_compare[f"{strategy}_nav"] = result["nav"]
                                descriptions[strategy] = (
                                    f"保持最新 hybrid market proxy 策略不变，仅在极端弱市时加入 {treasury['name']} 覆盖层："
                                    f"当市场量能 20/60<{tail_ratio_cut:.0%}、5/20<{tail_short_ratio_cut:.0%}、广度<{tail_breadth_cut:.1%}、"
                                    f"信号动量<={tail_momentum_ceiling:.0%}、策略回撤<={tail_drawdown_cut:.0%} 时，"
                                    f"把风险资产上限压到 {tail_risk_cap:.0%}，削减部分切到 {treasury['theme']}。"
                                )

    summary_df = pd.DataFrame(rows)
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(baseline_summary["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(baseline_summary["sharpe_rf0"])
    summary_df["max_drawdown_integral_diff"] = (
        summary_df["max_drawdown_integral"] - float(baseline_summary["max_drawdown_integral"])
    )
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
    print(
        summary_df[
            [
                "strategy",
                "treasury_code",
                "annualized_return",
                "sharpe_rf0",
                "max_drawdown_integral",
                "annualized_diff",
                "sharpe_diff",
                "max_drawdown_integral_diff",
                "is_valid_change",
            ]
        ].head(20).to_string(index=False)
    )

    notify_df = load_incremental_notify_candidates(
        NOTIFY_STATE_PATH,
        valid_df.copy(),
        metric_tolerance=METRIC_TOLERANCE,
    )

    notified, detail = notify_best_candidate(
        args,
        notify_df,
        descriptions=descriptions,
        baseline_summary=baseline_summary,
        default_webhook=DEFAULT_FEISHU_WEBHOOK,
        notify_state_path=NOTIFY_STATE_PATH,
        send_fn=send_improvement_notification,
    )
    if notified and detail is not None:
        print(f"\nWebhook notified for {detail}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
