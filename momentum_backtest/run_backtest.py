#!/usr/bin/env python3
"""运行 ETF 动量正式回测。"""

from __future__ import annotations

import argparse
import warnings

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
    from .core.backtest import run_threshold_dual_strategy
    from .core.config import (
        DEFAULT_CASH_THRESHOLD,
        DEFAULT_FEE_RATE,
        DEFAULT_LOOKBACK,
        DEFAULT_SLIPPAGE_RATE,
        DEFAULT_STRATEGY_NAME,
        DEFAULT_YEARS,
        load_default_strategy_backtest_pool,
    )
    from .core.data import fetch_histories
    from .core.reporting import print_summary, save_outputs
    from .core.strategy import run_default_strategy
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports
    from core.backtest import run_threshold_dual_strategy
    from core.config import (
        DEFAULT_CASH_THRESHOLD,
        DEFAULT_FEE_RATE,
        DEFAULT_LOOKBACK,
        DEFAULT_SLIPPAGE_RATE,
        DEFAULT_STRATEGY_NAME,
        DEFAULT_YEARS,
        load_default_strategy_backtest_pool,
    )
    from core.data import fetch_histories
    from core.reporting import print_summary, save_outputs
    from core.strategy import run_default_strategy


prepare_local_imports(__file__)
configure_matplotlib_env()
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 ETF 动量轮动正式回测。")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="动量回看窗口，单位为交易日。")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="回测最近多少年。")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="单边手续费率。")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="单边滑点率。")
    parser.add_argument("--cash-threshold", type=float, default=DEFAULT_CASH_THRESHOLD, help="兼容旧参数；正式基线不使用。")
    parser.add_argument("--disable-overheat-cap", action="store_true", help="兼容旧参数；关闭正式基线里的 Boll 过热压仓，只保留 raw 25 日 dual momentum。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = load_default_strategy_backtest_pool()
    prices = fetch_histories(selected, years=args.years)
    if args.disable_overheat_cap:
        result, trades = run_threshold_dual_strategy(
            prices,
            selected,
            lookback=args.lookback,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        strategy_name = "threshold_dual_momentum"
    else:
        result, trades = run_default_strategy(
            prices,
            selected,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
        strategy_name = DEFAULT_STRATEGY_NAME
    save_outputs(selected, prices, result, trades, lookback=args.lookback, strategy_name=strategy_name)
    print_summary(selected, result, trades, lookback=args.lookback, strategy_name=strategy_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
