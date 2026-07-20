#!/usr/bin/env python3
"""对比当前正式策略下的大盘量能与广度代理变体。"""

from __future__ import annotations

import argparse

try:
    from .runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

try:
    from .compare_hs300_regime_fixes import load_cached_data, summarize
    from .compare_goal_optimizations import (
        DEFAULT_FEISHU_WEBHOOK,
        GOAL_OUTPUT_DIR,
        METRIC_TOLERANCE,
        load_market_volume_proxy,
        send_improvement_notification,
    )
    from .search_utils import (
        add_notify_cli_args,
        load_incremental_notify_candidates,
        notify_best_candidate,
        raise_if_missing_required_histories,
        sort_notify_candidates,
        try_join_missing_candidate_histories,
        write_json_atomic,
    )
    from .official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from compare_hs300_regime_fixes import load_cached_data, summarize
    from compare_goal_optimizations import (
        DEFAULT_FEISHU_WEBHOOK,
        GOAL_OUTPUT_DIR,
        METRIC_TOLERANCE,
        load_market_volume_proxy,
        send_improvement_notification,
    )
    from search_utils import (
        add_notify_cli_args,
        load_incremental_notify_candidates,
        notify_best_candidate,
        raise_if_missing_required_histories,
        sort_notify_candidates,
        try_join_missing_candidate_histories,
        write_json_atomic,
    )
    from official_baseline import apply_official_baseline_nav_anchor

from run_backtest import (
    DEFAULT_BASELINE_DROP_CODES,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RISK_CODES,
    build_default_strategy_params,
    ensure_output_dirs,
    fetch_histories,
    load_default_strategy_backtest_pool,
    resolve_strategy_universe,
    run_default_strategy_with_params,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = GOAL_OUTPUT_DIR / "market_proxy_variants"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
BEST_PATH = OUTPUT_DIR / "best.json"
NOTIFY_STATE_PATH = OUTPUT_DIR / "notify_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比当前正式策略下的大盘量能与广度代理变体。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取 ETF 与市场代理数据。")
    add_notify_cli_args(parser)
    return parser.parse_args()


def build_risk_proxy_features(prices: pd.DataFrame, risk_codes: list[str] | None = None) -> pd.DataFrame:
    active_risk_codes, _ = resolve_strategy_universe(prices, risk_codes=risk_codes, defensive_codes=[])
    risk_codes = active_risk_codes
    if not risk_codes:
        raise RuntimeError("missing risk ETF prices for proxy construction")
    risk_prices = prices[risk_codes].copy()
    ret20 = risk_prices / risk_prices.shift(20) - 1
    above_ma20 = (risk_prices > risk_prices.rolling(20).mean()).astype(float)
    above_ma60 = (risk_prices > risk_prices.rolling(60).mean()).astype(float)
    pos20 = (ret20 > 0).astype(float)

    features = pd.DataFrame(index=prices.index)
    features["eq20"] = ret20.mean(axis=1)
    features["median20"] = ret20.median(axis=1)
    features["pos20_frac"] = pos20.mean(axis=1)
    features["above_ma20_frac"] = above_ma20.mean(axis=1)
    features["above_ma60_frac"] = above_ma60.mean(axis=1)
    features["pos20_ratio_20_60"] = features["pos20_frac"].rolling(20).mean() / features["pos20_frac"].rolling(60).mean()
    features["pos20_ratio_5_20"] = features["pos20_frac"].rolling(5).mean() / features["pos20_frac"].rolling(20).mean()
    features["above_ma20_ratio_20_60"] = (
        features["above_ma20_frac"].rolling(20).mean() / features["above_ma20_frac"].rolling(60).mean()
    )
    features["above_ma20_ratio_5_20"] = (
        features["above_ma20_frac"].rolling(5).mean() / features["above_ma20_frac"].rolling(20).mean()
    )
    features["breadth_blend"] = 0.5 * features["eq20"] + 0.5 * (features["above_ma20_frac"] - 0.5)
    return features


def build_proxy_catalog(
    base_proxy: pd.DataFrame,
    prices: pd.DataFrame,
    risk_codes: list[str] | None = None,
) -> list[dict[str, object]]:
    features = build_risk_proxy_features(prices, risk_codes=risk_codes)
    base_proxy = base_proxy.reindex(prices.index).ffill().copy()

    def make_proxy(
        name: str,
        ratio_20_60: pd.Series,
        ratio_5_20: pd.Series,
        breadth: pd.Series,
        description: str,
        ratio_cuts: tuple[float, ...],
        short_ratio_cuts: tuple[float, ...],
        breadth_cuts: tuple[float, ...],
    ) -> dict[str, object]:
        proxy = pd.DataFrame(index=prices.index)
        proxy["market_amount_ratio_20_60"] = ratio_20_60
        proxy["market_amount_ratio_5_20"] = ratio_5_20
        proxy["market_breadth_proxy"] = breadth
        return {
            "name": name,
            "proxy": proxy,
            "description": description,
            "ratio_cuts": ratio_cuts,
            "short_ratio_cuts": short_ratio_cuts,
            "breadth_cuts": breadth_cuts,
        }

    return [
        make_proxy(
            "baseline_sh_sz",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            base_proxy["market_breadth_proxy"],
            "维持当前上证+深证综合成交额代理与指数广度代理。",
            (0.91,),
            (0.90,),
            (-0.01,),
        ),
        make_proxy(
            "hybrid_eq20",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            0.5 * base_proxy["market_breadth_proxy"] + 0.5 * features["eq20"],
            "保留当前成交额代理，仅把广度替换为 指数广度 与 风险ETF等权20日收益 的混合。",
            (0.91,),
            (0.90,),
            (-0.03, -0.02, -0.01, 0.00, 0.01),
        ),
        make_proxy(
            "hybrid_median20",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            0.5 * base_proxy["market_breadth_proxy"] + 0.5 * features["median20"],
            "保留当前成交额代理，仅把广度替换为 指数广度 与 风险ETF中位数20日收益 的混合。",
            (0.91,),
            (0.90,),
            (-0.03, -0.02, -0.01, 0.00, 0.01),
        ),
        make_proxy(
            "hybrid_breadth_blend",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            0.5 * base_proxy["market_breadth_proxy"] + 0.5 * features["breadth_blend"],
            "保留当前成交额代理，仅把广度替换为 指数广度 与 风险ETF站上20日均线比例/20日收益混合 的混合。",
            (0.91,),
            (0.90,),
            (-0.04, -0.03, -0.02, -0.01, 0.00, 0.01),
        ),
        make_proxy(
            "hybrid_breadth_blend_25_75",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            0.25 * base_proxy["market_breadth_proxy"] + 0.75 * features["breadth_blend"],
            "保留当前成交额代理，仅把广度替换为 25%指数广度 + 75%风险ETF站上20日均线比例/20日收益混合。",
            (0.91,),
            (0.90,),
            (-0.04, -0.03, -0.02, -0.01, 0.00),
        ),
        make_proxy(
            "hybrid_breadth_blend_75_25",
            base_proxy["market_amount_ratio_20_60"],
            base_proxy["market_amount_ratio_5_20"],
            0.75 * base_proxy["market_breadth_proxy"] + 0.25 * features["breadth_blend"],
            "保留当前成交额代理，仅把广度替换为 75%指数广度 + 25%风险ETF站上20日均线比例/20日收益混合。",
            (0.91,),
            (0.90,),
            (-0.04, -0.03, -0.02, -0.01, 0.00),
        ),
        make_proxy(
            "etf_pos20_regime",
            features["pos20_ratio_20_60"],
            features["pos20_ratio_5_20"],
            features["eq20"],
            "把量能代理改为 风险ETF横截面20日正收益占比 的强弱变化，广度使用风险ETF等权20日收益。",
            (0.92, 0.96, 1.00),
            (0.90, 0.95, 1.00),
            (-0.03, -0.02, -0.01, 0.00),
        ),
        make_proxy(
            "etf_above_ma20_regime",
            features["above_ma20_ratio_20_60"],
            features["above_ma20_ratio_5_20"],
            features["median20"],
            "把量能代理改为 风险ETF站上20日均线比例 的强弱变化，广度使用风险ETF中位数20日收益。",
            (0.92, 0.96, 1.00),
            (0.90, 0.95, 1.00),
            (-0.03, -0.02, -0.01, 0.00),
        ),
        make_proxy(
            "etf_above_ma20_blend",
            features["above_ma20_ratio_20_60"],
            features["above_ma20_ratio_5_20"],
            features["breadth_blend"],
            "把量能代理改为 风险ETF站上20日均线比例 的强弱变化，广度使用站上20日均线比例与20日收益的混合。",
            (0.92, 0.96, 1.00),
            (0.90, 0.95, 1.00),
            (-0.03, -0.02, -0.01, 0.00),
        ),
    ]


def evaluate_strategy(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    market_proxy: pd.DataFrame,
    params: dict[str, object],
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_default_strategy_with_params(
        prices=prices,
        selected=selected,
        params=params,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        market_proxy=market_proxy,
    )


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    drop_codes = list(DEFAULT_BASELINE_DROP_CODES)
    current_params = build_default_strategy_params(drop_codes=drop_codes)
    selected = load_default_strategy_backtest_pool()
    if args.refresh:
        prices = fetch_histories(selected, years=args.years)
    else:
        _, prices = load_cached_data()
        start_ts = prices.index.max() - pd.DateOffset(years=args.years)
        prices = prices.loc[prices.index >= start_ts].copy()
        required_treasury = selected[selected["code"].astype(str) == "511260"].copy()
        prices, skipped_codes, fetch_error = try_join_missing_candidate_histories(
            prices=prices,
            candidates=required_treasury,
            years=args.years,
            fetch_fn=fetch_histories,
        )
        if skipped_codes:
            skipped_text = ", ".join(skipped_codes)
            if fetch_error:
                print(f"[warn] missing required treasury cache: {skipped_text}; fetch failed: {fetch_error}")
            else:
                print(f"[warn] missing required treasury cache: {skipped_text}")
        keep_codes = [code for code in selected["code"].astype(str) if code in prices.columns]
        prices = prices[keep_codes].copy()
    raise_if_missing_required_histories(
        prices,
        selected[selected["code"].astype(str) == "511260"].copy(),
        context="baseline treasury",
    )
    base_proxy = load_market_volume_proxy(years=args.years, refresh=args.refresh)

    rows: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}
    nav_compare = pd.DataFrame(index=prices.index)

    baseline_name = "current_default_strategy"
    baseline_result, baseline_trades = evaluate_strategy(
        selected=selected,
        prices=prices,
        market_proxy=base_proxy,
        params=current_params,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    baseline_result = apply_official_baseline_nav_anchor(baseline_result)
    baseline_summary = summarize(baseline_result, baseline_trades, selected)
    baseline_summary["strategy"] = baseline_name
    baseline_summary["proxy_kind"] = str(current_params["proxy_kind"])
    baseline_summary["volume_ratio_cut"] = current_params["volume_ratio_cut"]
    baseline_summary["volume_short_ratio_cut"] = current_params["volume_short_ratio_cut"]
    baseline_summary["volume_breadth_cut"] = current_params["volume_breadth_cut"]
    rows.append(baseline_summary)
    nav_compare[f"{baseline_name}_nav"] = baseline_result["nav"]
    descriptions[baseline_name] = "当前正式默认策略：默认池子 + hybrid breadth proxy + A股量能防守 + 过热双层降仓 + 十年国债弱市防守。"

    for proxy_spec in build_proxy_catalog(
        base_proxy,
        prices,
        risk_codes=[str(code) for code in current_params.get("risk_codes", RISK_CODES)],
    ):
        proxy_name = str(proxy_spec["name"])
        proxy = proxy_spec["proxy"]
        proxy_description = str(proxy_spec["description"])
        for volume_ratio_cut in proxy_spec["ratio_cuts"]:
            for volume_short_ratio_cut in proxy_spec["short_ratio_cuts"]:
                for volume_breadth_cut in proxy_spec["breadth_cuts"]:
                    if (
                        proxy_name == str(current_params["proxy_kind"])
                        and abs(volume_ratio_cut - current_params["volume_ratio_cut"]) < 1e-12
                        and abs(volume_short_ratio_cut - current_params["volume_short_ratio_cut"]) < 1e-12
                        and abs(volume_breadth_cut - current_params["volume_breadth_cut"]) < 1e-12
                    ):
                        continue
                    params = {
                        **current_params,
                        "proxy_kind": proxy_name,
                        "volume_ratio_cut": float(volume_ratio_cut),
                        "volume_short_ratio_cut": float(volume_short_ratio_cut),
                        "volume_breadth_cut": float(volume_breadth_cut),
                    }
                    result, trades = evaluate_strategy(
                        selected=selected,
                        prices=prices,
                        market_proxy=base_proxy,
                        params=params,
                        fee_rate=args.fee_rate,
                        slippage_rate=args.slippage_rate,
                    )
                    strategy = (
                        f"{baseline_name}__proxy_{proxy_name}"
                        f"_vr{int(round(float(volume_ratio_cut) * 100)):02d}"
                        f"_vs{int(round(float(volume_short_ratio_cut) * 100)):02d}"
                        f"_vb{int(round((float(volume_breadth_cut) + 0.02) * 100)):02d}"
                    )
                    summary = summarize(result, trades, selected)
                    summary["strategy"] = strategy
                    summary["proxy_kind"] = proxy_name
                    summary["volume_ratio_cut"] = float(volume_ratio_cut)
                    summary["volume_short_ratio_cut"] = float(volume_short_ratio_cut)
                    summary["volume_breadth_cut"] = float(volume_breadth_cut)
                    rows.append(summary)
                    nav_compare[f"{strategy}_nav"] = result["nav"]
                    descriptions[strategy] = (
                        f"保持当前 regime/core/过热参数不变，仅替换市场代理。{proxy_description}"
                        f" 弱市场触发阈值改为 20/60={float(volume_ratio_cut):.0%}、"
                        f"5/20={float(volume_short_ratio_cut):.0%}、广度<{float(volume_breadth_cut):.1%}。"
                    )

    summary_df = pd.DataFrame(rows)
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(baseline_summary["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(baseline_summary["sharpe_rf0"])
    summary_df["max_drawdown_integral_diff"] = (
        summary_df["max_drawdown_integral"] - float(baseline_summary["max_drawdown_integral"])
    )
    summary_df["is_valid_change"] = (
        (summary_df["strategy"] != baseline_name)
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
    payload = {"baseline": baseline_summary, "valid_improvements": valid_df.to_dict(orient="records"), "descriptions": descriptions}
    write_json_atomic(BEST_PATH, payload)

    print("Baseline:")
    print(pd.Series(baseline_summary).to_string())
    print("\nTop candidates:")
    print(
        summary_df[
            [
                "strategy",
                "proxy_kind",
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
