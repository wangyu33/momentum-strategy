#!/usr/bin/env python3
"""分析缓存回测结果中的主要回撤区间。"""

from __future__ import annotations

import argparse
try:
    from .runtime_env import prepare_local_imports
except ImportError:
    from runtime_env import prepare_local_imports

prepare_local_imports(__file__)

import pandas as pd

from run_backtest import (
    ANALYSIS_OUTPUT_DIR,
    CORE_OUTPUT_DIR,
    build_drawdown_episode_report,
    ensure_output_dirs,
    filter_indexed_frame_to_confirmed_closes,
    load_core_selected_and_prices,
    write_dataframe_csv_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析缓存回测结果中的主要回撤区间。")
    parser.add_argument("--head", type=int, default=10, help="打印前多少条回撤区间记录。")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    nav = pd.read_csv(CORE_OUTPUT_DIR / "backtest_nav.csv", parse_dates=["date"]).set_index("date")
    nav = filter_indexed_frame_to_confirmed_closes(nav)
    trades = pd.read_csv(CORE_OUTPUT_DIR / "trades.csv", parse_dates=["date"])
    selected, prices = load_core_selected_and_prices()
    compare = pd.read_csv(CORE_OUTPUT_DIR / "strategy_vs_hs300.csv", parse_dates=["date"]).set_index("date")
    compare = filter_indexed_frame_to_confirmed_closes(compare)
    report = build_drawdown_episode_report(nav, prices, trades, selected, compare)
    ensure_output_dirs()
    write_dataframe_csv_atomic(report, ANALYSIS_OUTPUT_DIR / "drawdown_episodes.csv", index=False)
    print(report.head(args.head).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
