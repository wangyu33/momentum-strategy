#!/usr/bin/env python3
"""对比黄金/豆粕在弱市高波动时的临时降权方案。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MOMENTUM_DIR = Path(__file__).resolve().parents[2]
if str(MOMENTUM_DIR) not in sys.path:
    sys.path.insert(0, str(MOMENTUM_DIR))

from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from compare_goal_optimizations import load_market_volume_proxy
from compare_hs300_regime_fixes import run_target_weights_strategy
from compare_market_proxy_variants import build_proxy_catalog

from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    RESEARCH_OUTPUT_DIR,
    build_default_strategy_params,
    build_strategy_summary,
    ensure_output_dirs,
    fetch_histories,
    load_core_selected_and_prices,
    load_default_strategy_backtest_pool,
    run_default_strategy_with_params,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "archive_flat" / "compare_opportunistic_defensive_cap"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"
OPPORTUNISTIC_CODES = ["518880", "159985"]
TREASURY_CODE = "511260"
OPPORTUNISTIC_SCOPES = {
    "both": ["518880", "159985"],
    "gold_only": ["518880"],
    "soymeal_only": ["159985"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比黄金/豆粕在弱市高波动时的临时降权方案。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取历史数据，而不是复用 output/core 缓存。")
    return parser.parse_args()


def load_prices(selected: pd.DataFrame, years: int, refresh: bool) -> pd.DataFrame:
    if refresh:
        return fetch_histories(selected, years=years)

    _, cached_prices = load_core_selected_and_prices()
    start_ts = cached_prices.index.max() - pd.DateOffset(years=years)
    prices = cached_prices.loc[cached_prices.index >= start_ts].copy()
    wanted_codes = [code for code in selected["code"].astype(str) if code in prices.columns]
    return prices[wanted_codes].copy()


def apply_opportunistic_cap(
    base_target_weights: pd.DataFrame,
    prices: pd.DataFrame,
    proxy: pd.DataFrame,
    *,
    opp_codes: list[str],
    cap: float,
    vol20_cut: float,
    breadth_cut: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    adjusted = base_target_weights.copy()
    opp_codes = [code for code in opp_codes if code in adjusted.columns]
    if not opp_codes:
        return adjusted, pd.Series(False, index=adjusted.index), pd.Series(0.0, index=adjusted.index), pd.Series(0.0, index=adjusted.index)

    returns = prices[opp_codes].pct_change()
    vol20 = returns.rolling(20).std()
    opp_weight = adjusted[opp_codes].sum(axis=1)

    # 用“当前持有的机会型防守仓”的加权波动率来判断它是否开始表现得更像风险资产。
    weighted_vol20 = (vol20.mul(adjusted[opp_codes], axis=0).sum(axis=1) / opp_weight.replace(0.0, pd.NA)).fillna(0.0)
    trigger_mask = (
        (opp_weight > cap)
        & (weighted_vol20 >= vol20_cut)
        & (proxy["market_breadth_proxy"] <= breadth_cut)
    ).fillna(False)

    moved_weight = pd.Series(0.0, index=adjusted.index, dtype="float64")
    if trigger_mask.any():
        scale = pd.Series(1.0, index=adjusted.index, dtype="float64")
        scale.loc[trigger_mask] = cap / opp_weight.loc[trigger_mask]
        moved_weight.loc[trigger_mask] = opp_weight.loc[trigger_mask] - cap
        adjusted.loc[trigger_mask, opp_codes] = adjusted.loc[trigger_mask, opp_codes].mul(scale.loc[trigger_mask], axis=0)
        adjusted.loc[trigger_mask, TREASURY_CODE] = adjusted.loc[trigger_mask, TREASURY_CODE].add(
            moved_weight.loc[trigger_mask],
            fill_value=0.0,
        )

    return adjusted, trigger_mask, weighted_vol20, moved_weight


def summarize(result: pd.DataFrame, trades: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float | int | str]:
    return build_strategy_summary(result, trades, selected=selected, include_max_drawdown_integral=True)


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = load_default_strategy_backtest_pool().copy()
    prices = load_prices(selected, years=args.years, refresh=args.refresh)
    params = build_default_strategy_params()
    market_proxy = load_market_volume_proxy(years=args.years, refresh=args.refresh)
    proxy_catalog = {
        item["name"]: item["proxy"]
        for item in build_proxy_catalog(
            market_proxy,
            prices,
            risk_codes=[str(code) for code in params.get("risk_codes", [])],
        )
    }
    proxy = proxy_catalog[str(params["proxy_kind"])].reindex(prices.index).ffill()

    baseline_result, baseline_trades = run_default_strategy_with_params(
        prices,
        selected,
        params=params,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        market_proxy=market_proxy,
    )
    base_target_weights = (
        baseline_result.filter(regex=r"^target_weight_")
        .rename(columns=lambda col: col.replace("target_weight_", "", 1))
        .reindex(columns=prices.columns, fill_value=0.0)
    )
    signal_momentum = baseline_result["current_momentum"].copy()

    rows: list[dict[str, object]] = []
    nav_compare = pd.DataFrame(index=prices.index)

    baseline_summary = summarize(baseline_result, baseline_trades, selected)
    baseline_summary["strategy"] = "baseline"
    baseline_summary["trigger_days"] = 0
    baseline_summary["avg_moved_weight"] = 0.0
    baseline_summary["max_moved_weight"] = 0.0
    baseline_summary["avg_weighted_vol20"] = 0.0
    baseline_summary["scope"] = "baseline"
    rows.append(baseline_summary)
    nav_compare["baseline_nav"] = baseline_result["nav"]

    for scope_name, scope_codes in OPPORTUNISTIC_SCOPES.items():
        for cap in (0.50, 0.65, 0.80):
            for vol20_cut in (0.018, 0.020, 0.022, 0.024):
                for breadth_cut in (-0.010, -0.015, -0.020):
                    adjusted_weights, trigger_mask, weighted_vol20, moved_weight = apply_opportunistic_cap(
                        base_target_weights,
                        prices,
                        proxy,
                        opp_codes=scope_codes,
                        cap=cap,
                        vol20_cut=vol20_cut,
                        breadth_cut=breadth_cut,
                    )
                    result, trades = run_target_weights_strategy(
                        prices,
                        selected,
                        adjusted_weights,
                        signal_momentum,
                        args.fee_rate,
                        args.slippage_rate,
                    )
                    strategy = (
                        f"{scope_name}_cap{int(round(cap * 100)):02d}"
                        f"_vol{int(round(vol20_cut * 1000)):03d}"
                        f"_br{int(round(abs(breadth_cut) * 1000)):03d}"
                    )
                    summary = summarize(result, trades, selected)
                    summary["strategy"] = strategy
                    summary["scope"] = scope_name
                    summary["cap"] = cap
                    summary["vol20_cut"] = vol20_cut
                    summary["breadth_cut"] = breadth_cut
                    summary["trigger_days"] = int(trigger_mask.sum())
                    summary["avg_moved_weight"] = float(moved_weight.loc[trigger_mask].mean()) if trigger_mask.any() else 0.0
                    summary["max_moved_weight"] = float(moved_weight.max())
                    summary["avg_weighted_vol20"] = (
                        float(weighted_vol20.loc[trigger_mask].mean()) if trigger_mask.any() else 0.0
                    )
                    rows.append(summary)
                    nav_compare[f"{strategy}_nav"] = result["nav"]

    summary_df = pd.DataFrame(rows)
    for metric in ("annualized_return", "sharpe_rf0", "max_drawdown", "max_drawdown_integral"):
        summary_df[f"{metric}_diff"] = summary_df[metric] - float(baseline_summary[metric])
    summary_df["is_integral_valid_change"] = (
        (summary_df["strategy"] != "baseline")
        & (summary_df["annualized_return"] >= float(baseline_summary["annualized_return"]) - 0.002)
        & (summary_df["sharpe_rf0"] >= float(baseline_summary["sharpe_rf0"]) - 0.02)
        & (summary_df["max_drawdown_integral"] <= float(baseline_summary["max_drawdown_integral"]) + 0.2)
        & (
            (summary_df["annualized_return"] > float(baseline_summary["annualized_return"]) + 0.001)
            | (summary_df["sharpe_rf0"] > float(baseline_summary["sharpe_rf0"]) + 0.01)
            | (summary_df["max_drawdown_integral"] < float(baseline_summary["max_drawdown_integral"]) - 0.1)
        )
    )
    summary_df["is_strict_drawdown_safe"] = (
        (summary_df["strategy"] != "baseline")
        & (summary_df["annualized_return"] >= float(baseline_summary["annualized_return"]) - 0.002)
        & (summary_df["sharpe_rf0"] >= float(baseline_summary["sharpe_rf0"]) - 0.02)
        & (summary_df["max_drawdown"] >= float(baseline_summary["max_drawdown"]) - 1e-12)
        & (summary_df["max_drawdown_integral"] <= float(baseline_summary["max_drawdown_integral"]) + 0.2)
        & (
            (summary_df["annualized_return"] > float(baseline_summary["annualized_return"]) + 0.001)
            | (summary_df["sharpe_rf0"] > float(baseline_summary["sharpe_rf0"]) + 0.01)
            | (summary_df["max_drawdown_integral"] < float(baseline_summary["max_drawdown_integral"]) - 0.1)
        )
    )
    summary_df = summary_df.sort_values(
        ["is_integral_valid_change", "is_strict_drawdown_safe", "annualized_return", "sharpe_rf0", "max_drawdown_integral"],
        ascending=[False, False, False, False, True],
    )
    write_dataframe_csv_atomic(summary_df, SUMMARY_PATH, index=False)
    write_dataframe_csv_atomic(nav_compare, COMPARE_PATH)

    print("Baseline:")
    print(
        summary_df.loc[summary_df["strategy"] == "baseline", [
            "annualized_return",
            "sharpe_rf0",
            "max_drawdown",
            "max_drawdown_integral",
            "trade_count",
        ]].iloc[0].to_string()
    )
    print("\nTop candidates:")
    print(
        summary_df[
            [
                "strategy",
                "scope",
                "annualized_return",
                "sharpe_rf0",
                "max_drawdown",
                "max_drawdown_integral",
                "annualized_return_diff",
                "sharpe_rf0_diff",
                "max_drawdown_integral_diff",
                "trigger_days",
                "avg_moved_weight",
                "is_integral_valid_change",
                "is_strict_drawdown_safe",
            ]
        ].head(15).to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
