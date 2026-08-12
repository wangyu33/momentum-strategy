#!/usr/bin/env python3
"""围绕十年国债防守覆盖层搜索持续性与滞后性参数。"""

from __future__ import annotations

import argparse

try:
    from .runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

try:
    from .candidate_pool_common import ETF_511260
    from .goal_optimization_common import DEFAULT_FEISHU_WEBHOOK, METRIC_TOLERANCE, load_market_volume_proxy, send_improvement_notification
    from .hs300_regime_common import load_cached_data, run_target_weights_strategy, summarize
    from .market_proxy_common import build_proxy_catalog
    from .official_strategy_core import build_official_target_weights
    from .compare_tail_risk_bond_overlay import load_market_proxy_best_context
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
    from candidate_pool_common import ETF_511260
    from goal_optimization_common import DEFAULT_FEISHU_WEBHOOK, METRIC_TOLERANCE, load_market_volume_proxy, send_improvement_notification
    from hs300_regime_common import load_cached_data, run_target_weights_strategy, summarize
    from market_proxy_common import build_proxy_catalog
    from official_strategy_core import build_official_target_weights
    from compare_tail_risk_bond_overlay import load_market_proxy_best_context
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
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RESEARCH_OUTPUT_DIR,
    build_official_baseline_summary,
    ensure_output_dirs,
    fetch_histories,
    load_fixed_etf_pool,
    resolve_strategy_universe,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "defensive_persistence"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"
SIMPLE_BOND_NOTIFY_STATE_PATH = RESEARCH_OUTPUT_DIR / "simple_bond_overlay" / "notify_state.json"
SIMPLE_BOND_BEST_PATH = RESEARCH_OUTPUT_DIR / "simple_bond_overlay" / "best.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="围绕十年国债防守覆盖层搜索持续性与滞后性参数。"
    )
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取 ETF 历史数据，而不是复用本地缓存。")
    add_notify_cli_args(parser)
    return parser.parse_args()


def load_overlay_context() -> dict[str, object]:
    payload = load_preferred_strategy_payload(SIMPLE_BOND_NOTIFY_STATE_PATH, SIMPLE_BOND_BEST_PATH)
    if payload is None:
        raise RuntimeError("missing simple bond overlay context")
    summary = dict(payload["summary"])
    required_keys = {"risk_cap", "ratio_cut", "breadth_cut", "treasury_code"}
    if not required_keys.issubset(summary):
        raise RuntimeError("simple bond overlay context missing required keys")
    return {
        "strategy": str(payload["strategy"]),
        "summary": summary,
        "description": str(payload["description"]),
        "risk_cap": float(summary["risk_cap"]),
        "ratio_cut": float(summary["ratio_cut"]),
        "breadth_cut": float(summary["breadth_cut"]),
        "treasury_code": str(summary["treasury_code"]),
    }


def build_persistent_mask(raw_trigger: pd.Series, enter_days: int, exit_days: int) -> pd.Series:
    active = False
    enter_streak = 0
    exit_streak = 0
    result: list[bool] = []

    for is_triggered in raw_trigger.fillna(False).astype(bool).tolist():
        if is_triggered:
            enter_streak += 1
            exit_streak = 0
        else:
            exit_streak += 1
            enter_streak = 0

        if not active and enter_streak >= enter_days:
            active = True
        elif active and exit_streak >= exit_days:
            active = False

        result.append(active)

    return pd.Series(result, index=raw_trigger.index, dtype=bool)


def apply_persistent_overlay(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    proxy: pd.DataFrame,
    params: dict[str, float | list[str]],
    treasury_code: str,
    risk_cap: float,
    ratio_cut: float,
    breadth_cut: float,
    enter_days: int,
    exit_days: int,
    fee_rate: float,
    slippage_rate: float,
    fill_residual_cash_to_treasury: bool = False,
    return_target_weights: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_weights, mixed_momentum, _ = build_official_target_weights(prices, proxy, params)
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

    raw_trigger = (
        (proxy["market_amount_ratio_20_60"] < ratio_cut)
        & (proxy["market_breadth_proxy"] < breadth_cut)
    ).fillna(False)
    trigger_mask = build_persistent_mask(raw_trigger, enter_days=enter_days, exit_days=exit_days)

    scale_mask = trigger_mask & (row_risk_weight > risk_cap)
    if scale_mask.any():
        scale = pd.Series(1.0, index=prices.index, dtype="float64")
        scale.loc[scale_mask] = risk_cap / row_risk_weight.loc[scale_mask]
        overlaid_weights.loc[scale_mask, risk_budget_codes] = overlaid_weights.loc[scale_mask, risk_budget_codes].mul(
            scale.loc[scale_mask], axis=0
        )
        moved_weight = row_risk_weight.loc[scale_mask] - risk_cap
        overlaid_weights.loc[scale_mask, treasury_code] = overlaid_weights.loc[scale_mask, treasury_code].add(
            moved_weight,
            fill_value=0.0,
        )

    if fill_residual_cash_to_treasury:
        residual_cash = (1.0 - overlaid_weights.sum(axis=1)).clip(lower=0.0)
        residual_mask = trigger_mask & (residual_cash > 1e-12)
        if residual_mask.any():
            overlaid_weights.loc[residual_mask, treasury_code] = overlaid_weights.loc[residual_mask, treasury_code].add(
                residual_cash.loc[residual_mask],
                fill_value=0.0,
            )

    result, trades = run_target_weights_strategy(prices, selected, overlaid_weights, mixed_momentum, fee_rate, slippage_rate)
    if return_target_weights:
        return result, trades, overlaid_weights
    return result, trades


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
    treasury_candidate_df = pd.DataFrame([ETF_511260])
    selected_for_prices = pd.concat([base_selected, treasury_candidate_df], ignore_index=True)
    if args.refresh:
        prices = fetch_histories(selected_for_prices, years=args.years)
    else:
        _, prices = load_cached_data()
        start_ts = prices.index.max() - pd.DateOffset(years=args.years)
        prices = prices.loc[prices.index >= start_ts].copy()
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
    if str(ETF_511260["code"]) not in prices.columns:
        raise RuntimeError(f"missing required baseline treasury history: {ETF_511260['code']} {ETF_511260['name']}")

    base_market_proxy = load_market_volume_proxy(years=args.years, refresh=args.refresh)
    proxy_catalog = {
        item["name"]: item["proxy"]
        for item in build_proxy_catalog(
            base_market_proxy,
            prices,
            risk_codes=[str(code) for code in params.get("risk_codes", [])] if params.get("risk_codes") is not None else None,
        )
    }
    proxy = proxy_catalog[str(market_context["proxy_kind"])]

    rows: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}
    nav_compare = pd.DataFrame(index=prices.index)

    baseline_selected = pd.concat([base_selected, pd.DataFrame([ETF_511260])], ignore_index=True)
    baseline_selected = baseline_selected.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)
    baseline_prices = prices[[code for code in baseline_selected["code"].astype(str) if code in prices.columns]].copy()
    baseline_result, baseline_trades = apply_persistent_overlay(
        prices=baseline_prices,
        selected=baseline_selected,
        proxy=proxy,
        params=params,
        treasury_code=overlay_context["treasury_code"],
        risk_cap=overlay_context["risk_cap"],
        ratio_cut=overlay_context["ratio_cut"],
        breadth_cut=overlay_context["breadth_cut"],
        enter_days=1,
        exit_days=1,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_summary = build_official_baseline_summary(
        baseline_result,
        baseline_trades,
        selected=baseline_selected,
        include_max_drawdown_integral=True,
    )
    baseline_summary["strategy"] = overlay_context["strategy"]
    rows.append(baseline_summary)
    nav_compare[f"{overlay_context['strategy']}_nav"] = baseline_result["nav"]
    descriptions[str(overlay_context["strategy"])] = overlay_context["description"]

    ratio_candidates = [0.895, 0.900]
    breadth_candidates = [-0.035, -0.030]
    risk_cap_candidates = [0.15, 0.20]
    enter_days_candidates = [1, 2, 3]
    exit_days_candidates = [1, 2, 3, 5]

    for ratio_cut in ratio_candidates:
        for breadth_cut in breadth_candidates:
            for risk_cap in risk_cap_candidates:
                for enter_days in enter_days_candidates:
                    for exit_days in exit_days_candidates:
                        result, trades = apply_persistent_overlay(
                            prices=baseline_prices,
                            selected=baseline_selected,
                            proxy=proxy,
                            params=params,
                            treasury_code=overlay_context["treasury_code"],
                            risk_cap=float(risk_cap),
                            ratio_cut=float(ratio_cut),
                            breadth_cut=float(breadth_cut),
                            enter_days=enter_days,
                            exit_days=exit_days,
                            fee_rate=args.fee_rate,
                            slippage_rate=args.slippage_rate,
                        )
                        strategy = (
                            f"{market_context['strategy']}__stressbond_511260"
                            f"_vr{int(round(ratio_cut * 1000)):03d}"
                            f"_vb{int(round((breadth_cut + 0.02) * 1000)):03d}"
                            f"_cap{int(round(risk_cap * 100)):02d}"
                            f"_en{enter_days:02d}_ex{exit_days:02d}"
                        )
                        summary = summarize(result, trades, baseline_selected)
                        summary["strategy"] = strategy
                        summary["ratio_cut"] = float(ratio_cut)
                        summary["breadth_cut"] = float(breadth_cut)
                        summary["risk_cap"] = float(risk_cap)
                        summary["enter_days"] = int(enter_days)
                        summary["exit_days"] = int(exit_days)
                        rows.append(summary)
                        nav_compare[f"{strategy}_nav"] = result["nav"]
                        descriptions[strategy] = (
                            "保持当前 hybrid market proxy + 十年国债防守结构不变，"
                            f"仅对弱市切债增加状态机控制：市场量能20/60<{float(ratio_cut):.1%}、"
                            f"广度<{float(breadth_cut):.1%} 连续 {enter_days} 天后进入防守，"
                            f"信号消失连续 {exit_days} 天后退出防守，风险资产上限压到 {float(risk_cap):.0%}。"
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
                "annualized_return",
                "sharpe_rf0",
                "max_drawdown",
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
