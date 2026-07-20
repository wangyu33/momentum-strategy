#!/usr/bin/env python3
"""导出指定 leader_margin 的研究图表，格式尽量贴近 core 输出。"""

from __future__ import annotations

import argparse
from functools import partial
import sys
from pathlib import Path

MOMENTUM_DIR = Path(__file__).resolve().parents[2]
if str(MOMENTUM_DIR) not in sys.path:
    sys.path.insert(0, str(MOMENTUM_DIR))

from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import pandas as pd

from archive_data_loaders import build_archive_flat_output_dir
from single_variant_report_helpers import save_and_print_single_variant_report
from variant_compare_helpers import summarize_variant_result

from archive_strategy_common import (
    DEFAULT_LOOKBACK,
    build_default_strategy_params,
    load_core_selected_and_prices,
    load_default_strategy_backtest_pool,
)
from goal_optimization_common import load_market_volume_proxy
try:
    from ...official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor
from archive_strategy_common import run_default_strategy_with_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出指定 leader_margin 的研究图表。")
    parser.add_argument("--leader-margin", type=float, default=0.01, help="最小领先优势阈值。默认 1%%。")
    parser.add_argument("--years", type=int, default=15, help="回测最近多少年。")
    return parser.parse_args()


def build_output_dir(leader_margin: float) -> Path:
    suffix = f"leader_margin_{int(round(leader_margin * 1000)):03d}"
    return build_archive_flat_output_dir(f"export_{suffix}")


def main() -> int:
    args = parse_args()
    output_dir = build_output_dir(args.leader_margin)

    selected = load_default_strategy_backtest_pool().copy()
    _, prices = load_core_selected_and_prices()
    start_ts = prices.index.max() - pd.DateOffset(years=args.years)
    prices = prices.loc[prices.index >= start_ts].copy()
    market_proxy = load_market_volume_proxy(years=args.years, refresh=False)

    params = build_default_strategy_params()
    params["signal_leader_margin"] = float(args.leader_margin)
    result, trades = run_default_strategy_with_params(
        prices,
        selected,
        params=params,
        market_proxy=market_proxy,
    )
    result = apply_official_baseline_nav_anchor(result)

    benchmark_nav, _ = save_and_print_single_variant_report(
        output_dir,
        "ETF Strategy",
        selected,
        prices,
        result,
        trades,
        lookback=DEFAULT_LOOKBACK,
        strategy_name=f"leader_margin_{args.leader_margin:.3f}",
        summarize_fn=partial(
            summarize_variant_result,
            selected=selected,
            include_max_drawdown_integral=True,
        ),
        summary_extra_fields={"leader_margin": float(args.leader_margin)},
        nav_title=f"ETF Strategy NAV (leader_margin={args.leader_margin:.2%})",
        compare_title="Strategy vs HS300 Benchmark",
        include_momentum_series=False,
    )
    print(f"max_drawdown_date={result['drawdown'].idxmin().date()}")
    print(f"latest_nav={float(result['nav'].iloc[-1]):.4f}")
    print(f"benchmark_latest_nav={float(benchmark_nav.iloc[-1]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
