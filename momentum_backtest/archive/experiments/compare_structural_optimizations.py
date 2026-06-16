#!/usr/bin/env python3
"""对比当前正式基线与两步结构优化的效果。"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ..runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from compare_goal_optimizations import load_market_volume_proxy
from compare_hs300_regime_fixes import load_cached_data, summarize
from run_backtest import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    build_default_strategy_params,
    ensure_output_dirs,
    fetch_histories,
    load_default_strategy_backtest_pool,
    run_default_strategy_with_params,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = Path("momentum_backtest/output/research/archive_flat/compare_structural_optimizations")
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
NAV_COMPARE_PATH = OUTPUT_DIR / "nav_compare.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比当前正式基线与两步结构优化的效果。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--refresh", action="store_true", help="重新抓取价格和市场代理数据。")
    return parser.parse_args()


def load_prices(selected: pd.DataFrame, years: int, refresh: bool) -> pd.DataFrame:
    if refresh:
        return fetch_histories(selected, years=years)

    _, cached_prices = load_cached_data()
    start_ts = cached_prices.index.max() - pd.DateOffset(years=years)
    prices = cached_prices.loc[cached_prices.index >= start_ts].copy()
    selected_codes = selected["code"].astype(str).tolist()
    prices = prices[[code for code in selected_codes if code in prices.columns]].copy()
    return prices


def build_variant_rows(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    market_proxy: pd.DataFrame,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_params = build_default_strategy_params()
    split_params = dict(base_params)
    split_params["volume_guard_cap"] = float(base_params["aggressive_core_weight"]) + float(base_params["conservative_core_weight"])

    smooth_params = dict(split_params)
    smooth_params["overheat_cap_mode"] = "continuous"

    variants = [
        (
            "baseline",
            "当前正式基线：弱量能直接清零风险仓，弱市触发后再切债，高位过热使用两档台阶式降仓。",
            base_params,
            False,
        ),
        (
            "step1_split_weak_and_stress",
            "第一步：弱量能先把风险仓压到核心仓总和 55%，只负责减风险；更弱市时再把剩余现金统一停泊到十年国债。",
            split_params,
            True,
        ),
        (
            "step2_split_and_smooth_overheat",
            "第二步：保留第一步的弱市拆层，同时把过热降仓从两档台阶改成 25%~32% 动量区间内线性连续收缩。",
            smooth_params,
            True,
        ),
    ]

    summary_rows: list[dict[str, object]] = []
    nav_compare = pd.DataFrame(index=prices.index)
    baseline_metrics: dict[str, float] | None = None

    for name, description, params, fill_residual_cash in variants:
        result, trades = run_default_strategy_with_params(
            prices=prices,
            selected=selected,
            params=params,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            market_proxy=market_proxy,
            fill_residual_cash_to_treasury=fill_residual_cash,
        )
        summary = summarize(result, trades, selected)
        summary["strategy"] = name
        summary["description"] = description
        summary["fill_residual_cash_to_treasury"] = fill_residual_cash
        summary["volume_guard_cap"] = float(params["volume_guard_cap"])
        summary["overheat_cap_mode"] = str(params.get("overheat_cap_mode", "step"))

        if baseline_metrics is None:
            baseline_metrics = {
                "annualized_return": float(summary["annualized_return"]),
                "sharpe_rf0": float(summary["sharpe_rf0"]),
                "max_drawdown": float(summary["max_drawdown"]),
                "max_drawdown_integral": float(summary["max_drawdown_integral"]),
            }
            summary["annualized_diff"] = 0.0
            summary["sharpe_diff"] = 0.0
            summary["max_drawdown_diff"] = 0.0
            summary["max_drawdown_integral_diff"] = 0.0
        else:
            summary["annualized_diff"] = float(summary["annualized_return"]) - baseline_metrics["annualized_return"]
            summary["sharpe_diff"] = float(summary["sharpe_rf0"]) - baseline_metrics["sharpe_rf0"]
            summary["max_drawdown_diff"] = float(summary["max_drawdown"]) - baseline_metrics["max_drawdown"]
            summary["max_drawdown_integral_diff"] = (
                float(summary["max_drawdown_integral"]) - baseline_metrics["max_drawdown_integral"]
            )

        nav_compare[f"{name}_nav"] = result["nav"]
        summary_rows.append(summary)

    return pd.DataFrame(summary_rows), nav_compare


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = load_default_strategy_backtest_pool()
    prices = load_prices(selected, years=args.years, refresh=args.refresh)
    market_proxy = load_market_volume_proxy(years=args.years, refresh=args.refresh)

    summary_df, nav_compare = build_variant_rows(
        selected=selected,
        prices=prices,
        market_proxy=market_proxy,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    write_dataframe_csv_atomic(summary_df, SUMMARY_PATH, index=False)
    write_dataframe_csv_atomic(nav_compare, NAV_COMPARE_PATH)

    print(summary_df.to_string(index=False))
    print(f"\nsummary saved to {SUMMARY_PATH}")
    print(f"nav compare saved to {NAV_COMPARE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
