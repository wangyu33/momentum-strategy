#!/usr/bin/env python3
"""计算两个候选标的价格与收益的相关性。"""

from __future__ import annotations

import argparse

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

prepare_local_imports(__file__)
configure_matplotlib_env()

import pandas as pd

from compare_candidate_pool_additions import ETF_513400
from run_backtest import fetch_histories


PAIR_CANDIDATES = {
    "nasdaq": {"theme": "纳指ETF", "code": "159941", "name": "纳指ETF广发", "sina_symbol": "sz159941"},
    "dow": ETF_513400,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析两个标的的相关性。")
    parser.add_argument("--left", choices=sorted(PAIR_CANDIDATES.keys()), default="dow")
    parser.add_argument("--right", choices=sorted(PAIR_CANDIDATES.keys()), default="nasdaq")
    parser.add_argument("--years", type=int, default=5)
    return parser.parse_args()


def safe_corr(frame: pd.DataFrame, left_code: str, right_code: str) -> float:
    if len(frame) < 2:
        return float("nan")
    return float(frame[left_code].corr(frame[right_code]))


def main() -> int:
    args = parse_args()
    left = PAIR_CANDIDATES[args.left]
    right = PAIR_CANDIDATES[args.right]
    selected = pd.DataFrame([left, right])
    prices = fetch_histories(selected, years=args.years).dropna(how="any")
    returns = prices.pct_change().dropna(how="any")

    left_code = str(left["code"])
    right_code = str(right["code"])
    rolling_60 = returns[left_code].rolling(60).corr(returns[right_code]).dropna()

    result = {
        "left": left_code,
        "right": right_code,
        "sample_start": prices.index[0].date().isoformat(),
        "sample_end": prices.index[-1].date().isoformat(),
        "price_corr": safe_corr(prices, left_code, right_code),
        "daily_return_corr": safe_corr(returns, left_code, right_code),
        "daily_return_corr_250d": safe_corr(returns.tail(250), left_code, right_code),
        "daily_return_corr_120d": safe_corr(returns.tail(120), left_code, right_code),
        "daily_return_corr_60d": safe_corr(returns.tail(60), left_code, right_code),
        "rolling_60_latest": float(rolling_60.iloc[-1]) if not rolling_60.empty else float("nan"),
        "rolling_60_min": float(rolling_60.min()) if not rolling_60.empty else float("nan"),
        "rolling_60_max": float(rolling_60.max()) if not rolling_60.empty else float("nan"),
    }
    print(pd.Series(result).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
