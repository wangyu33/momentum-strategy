#!/usr/bin/env python3
"""运行中概互联 70% 仓位上限分支，并输出完整结果。

这是历史研究分支导出脚本，不应被锚定到正式 28.2691 官方基线净值链。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MOMENTUM_DIR = Path(__file__).resolve().parents[2]
if str(MOMENTUM_DIR) not in sys.path:
    sys.path.insert(0, str(MOMENTUM_DIR))

try:
    from ...runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import matplotlib
import pandas as pd

from archive_data_loaders import build_archive_flat_output_dir, load_selected_prices
from candidate_pool_common import ETF_513050
from single_variant_report_helpers import save_and_print_single_variant_report
from china_internet_guard_common import run_china_internet_guard_variant, summarize_china_internet_guard_result
from archive_strategy_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    load_fixed_etf_pool,
    fetch_histories,
)

matplotlib.use("Agg")


OUTPUT_DIR = build_archive_flat_output_dir("run_china_internet_cap70_backtest")
STRATEGY_NAME = "china_internet_cap70"
CHART_PREFIX = "China Internet Cap70"
INTENTIONALLY_UNANCHORED_BASELINE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行中概互联 70% 仓位上限分支。")
    parser.add_argument("--years", type=int, default=6, help="向前抓取多少年历史数据，再对齐公共区间。")
    parser.add_argument("--lookback", type=int, default=25, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_pool = load_fixed_etf_pool()
    selected = pd.concat([base_pool, pd.DataFrame([ETF_513050])], ignore_index=True)
    prices = load_selected_prices(
        selected,
        years=args.years,
        refresh=True,
        fetch_fn=fetch_histories,
    ).dropna(how="any")
    result, trades = run_china_internet_guard_variant(
        prices=prices,
        selected=selected,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        china_max_exposure=0.70,
    )
    save_and_print_single_variant_report(
        OUTPUT_DIR,
        CHART_PREFIX,
        selected,
        prices,
        result,
        trades,
        lookback=args.lookback,
        strategy_name=STRATEGY_NAME,
        summarize_fn=summarize_china_internet_guard_result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
