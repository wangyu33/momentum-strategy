#!/usr/bin/env python3
"""分析 ETF 动量策略里各标的的历史收益贡献。"""

from __future__ import annotations

import argparse
try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import pandas as pd

from run_backtest import (
    ANALYSIS_OUTPUT_DIR,
    CORE_OUTPUT_DIR,
    DEFAULT_FEE_RATE,
    DEFAULT_LOOKBACK,
    DEFAULT_SLIPPAGE_RATE,
    build_contribution_overview,
    build_contribution_summary,
    configure_matplotlib,
    ensure_output_dirs,
    fetch_histories,
    filter_indexed_frame_to_confirmed_closes,
    load_core_selected_and_prices,
    load_default_strategy_backtest_pool,
    normalize_code,
    run_default_strategy,
    save_contribution_chart,
    write_dataframe_csv_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 ETF 动量策略里各标的的历史收益贡献。")
    parser.add_argument("--years", type=int, default=10, help="回测最近多少年。")
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help="兼容旧参数；当前正式基线刷新路径固定使用默认策略配置，不单独覆盖 lookback。",
    )
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--start-date", type=str, default=None, help="可选的分析起始日期，例如 2016-01-01。")
    parser.add_argument("--end-date", type=str, default=None, help="可选的分析结束日期，例如 2021-12-31。")
    parser.add_argument("--refresh", action="store_true", help="强制重新抓价并重跑回测，而不是复用本地输出缓存。")
    return parser.parse_args()


def compute_contribution(prices: pd.DataFrame, selected: pd.DataFrame, result: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    contribution_df = build_contribution_summary(prices, selected, result)
    summary = build_contribution_overview(result, contribution_df).iloc[0].to_dict()
    return contribution_df, summary


def save_chart(contribution_df: pd.DataFrame) -> None:
    configure_matplotlib()
    save_contribution_chart(contribution_df, ANALYSIS_OUTPUT_DIR / "contribution_rate.png")


def load_cached_result() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected, prices = load_core_selected_and_prices()
    result = pd.read_csv(CORE_OUTPUT_DIR / "backtest_nav.csv", parse_dates=["date"]).set_index("date")
    result = filter_indexed_frame_to_confirmed_closes(result)
    for column in ["signal", "holding"]:
        if column in result.columns:
            result[column] = result[column].map(normalize_code)
    return selected, prices, result


def main() -> int:
    args = parse_args()
    if args.refresh:
        # 刷新口径必须和当前正式基线一致，不能再回退到旧的 fixed pool + run_strategy，
        # 否则同一份贡献分析会因为是否加 --refresh 而得到两套不同策略结果。
        selected = load_default_strategy_backtest_pool()
        prices = fetch_histories(selected, years=args.years)
        result, _ = run_default_strategy(
            prices,
            selected,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
    else:
        selected, prices, result = load_cached_result()

    if args.start_date:
        start_ts = pd.Timestamp(args.start_date)
        prices = prices.loc[prices.index >= start_ts]
        result = result.loc[result.index >= start_ts]
    if args.end_date:
        end_ts = pd.Timestamp(args.end_date)
        prices = prices.loc[prices.index <= end_ts]
        result = result.loc[result.index <= end_ts]

    if result.empty:
        raise RuntimeError("empty result after date filtering")

    contribution_df, summary = compute_contribution(prices, selected, result)

    ensure_output_dirs()
    write_dataframe_csv_atomic(contribution_df, ANALYSIS_OUTPUT_DIR / "contribution_summary.csv", index=False)
    write_dataframe_csv_atomic(pd.DataFrame([summary]), ANALYSIS_OUTPUT_DIR / "contribution_overview.csv", index=False)
    save_chart(contribution_df)

    print("Contribution overview:")
    print(pd.DataFrame([summary]).to_string(index=False))
    print("\nContribution summary:")
    print(contribution_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
