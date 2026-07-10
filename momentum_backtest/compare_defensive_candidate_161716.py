#!/usr/bin/env python3
"""评估把 161716 加入当前正式基线防守池的效果。"""

from __future__ import annotations

import argparse

try:
    from .runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__, include_module_dir=False)

import pandas as pd

try:
    from .compare_hs300_regime_fixes import summarize
    from .compare_goal_optimizations import load_market_volume_proxy
except ImportError:
    from compare_hs300_regime_fixes import summarize
    from compare_goal_optimizations import load_market_volume_proxy

from run_backtest import (
    DEFAULT_BASELINE_DROP_CODES,
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    DEFENSIVE_CODES,
    RESEARCH_OUTPUT_DIR,
    RISK_CODES,
    build_default_strategy_params,
    fetch_histories,
    load_default_strategy_backtest_pool,
    run_default_strategy_with_params,
    write_dataframe_csv_atomic,
)


OUTPUT_DIR = RESEARCH_OUTPUT_DIR / "defensive_candidate_161716"
ETF_161716 = {
    "theme": "双债增强",
    "code": "161716",
    "name": "招商双债增强LOF",
    "sina_symbol": "sz161716",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估把 161716 加入当前正式基线防守池的效果。")
    parser.add_argument("--years", type=int, default=15, help="向前抓取多少年历史数据。")
    parser.add_argument("--start-date", type=str, default="2012-01-01", help="分析起始日期。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def evaluate_pool(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    market_proxy: pd.DataFrame,
    drop_codes: list[str],
    risk_codes: list[str],
    defensive_codes: list[str],
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_default_strategy_with_params(
        prices=prices,
        selected=selected,
        params=build_default_strategy_params(
            drop_codes=drop_codes,
            risk_codes=risk_codes,
            defensive_codes=defensive_codes,
        ),
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        market_proxy=market_proxy,
    )


def build_variant_pool(add_161716: bool) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    selected = load_default_strategy_backtest_pool().copy()
    drop_codes = list(DEFAULT_BASELINE_DROP_CODES)
    risk_codes = [code for code in RISK_CODES if code not in drop_codes]
    defensive_codes = [code for code in DEFENSIVE_CODES if code not in drop_codes]
    if add_161716:
        selected = pd.concat([selected, pd.DataFrame([ETF_161716])], ignore_index=True)
        defensive_codes = list(dict.fromkeys(defensive_codes + [ETF_161716["code"]]))
    selected = selected.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)
    return selected, risk_codes, defensive_codes, drop_codes


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    market_proxy = load_market_volume_proxy(years=args.years, refresh=False)

    rows: list[dict[str, object]] = []
    nav_compare = pd.DataFrame()

    for variant_name, add_161716 in (("baseline", False), ("plus_161716_defensive", True)):
        selected, risk_codes, defensive_codes, drop_codes = build_variant_pool(add_161716=add_161716)
        prices = fetch_histories(selected, years=args.years)
        prices = prices.loc[prices.index >= pd.Timestamp(args.start_date)].copy()
        result, trades = evaluate_pool(
            selected=selected,
            prices=prices,
            market_proxy=market_proxy,
            drop_codes=drop_codes,
            risk_codes=risk_codes,
            defensive_codes=defensive_codes,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        row = summarize(result, trades, selected)
        row["variant"] = variant_name
        row["risk_codes"] = ",".join(code for code in risk_codes if code in prices.columns)
        row["defensive_codes"] = ",".join(code for code in defensive_codes if code in prices.columns)
        rows.append(row)
        nav_compare[variant_name] = result["nav"]
        write_dataframe_csv_atomic(trades, OUTPUT_DIR / f"{variant_name}_trades.csv", index=False)

    summary_df = pd.DataFrame(rows)
    base_row = summary_df[summary_df["variant"] == "baseline"].iloc[0]
    summary_df["annualized_diff"] = summary_df["annualized_return"] - float(base_row["annualized_return"])
    summary_df["sharpe_diff"] = summary_df["sharpe_rf0"] - float(base_row["sharpe_rf0"])
    summary_df["max_drawdown_integral_diff"] = summary_df["max_drawdown_integral"] - float(base_row["max_drawdown_integral"])
    summary_df["mdd_diff"] = summary_df["max_drawdown"] - float(base_row["max_drawdown"])
    summary_df["trade_diff"] = summary_df["trade_count"] - int(base_row["trade_count"])

    write_dataframe_csv_atomic(summary_df, OUTPUT_DIR / "summary.csv", index=False)
    write_dataframe_csv_atomic(nav_compare, OUTPUT_DIR / "nav_compare.csv")
    print(summary_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
