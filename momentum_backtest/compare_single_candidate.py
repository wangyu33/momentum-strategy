#!/usr/bin/env python3
"""只对比 base_pool 与单个候选加入后的结果。"""

from __future__ import annotations

import argparse

try:
    from .runtime_env import configure_matplotlib_env, prepare_local_imports
except ImportError:
    from runtime_env import configure_matplotlib_env, prepare_local_imports

try:
    from .official_baseline import apply_official_baseline_nav_anchor
except ImportError:
    from official_baseline import apply_official_baseline_nav_anchor

prepare_local_imports(__file__)
configure_matplotlib_env()

import pandas as pd

try:
    from .candidate_pool_common import (
        BASE_DEFENSIVE_CODES,
        BASE_RISK_CODES,
        ETF_164824,
        ETF_513400,
        run_custom_threshold_dual_with_overheat,
        summarize,
    )
except ImportError:
    from candidate_pool_common import (
        BASE_DEFENSIVE_CODES,
        BASE_RISK_CODES,
        ETF_164824,
        ETF_513400,
        run_custom_threshold_dual_with_overheat,
        summarize,
    )
from run_backtest import fetch_histories, load_fixed_etf_pool


CANDIDATES = {
    "india_lof": ETF_164824,
    "dow": ETF_513400,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比 base_pool 与单个候选加入后的结果。")
    parser.add_argument("--candidate", choices=sorted(CANDIDATES.keys()), required=True)
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument("--lookback", type=int, default=25)
    parser.add_argument("--start-date", type=str, default="2012-01-01")
    parser.add_argument("--fee-rate", type=float, default=0.0003)
    parser.add_argument("--slippage-rate", type=float, default=0.0002)
    return parser.parse_args()


def run_case(name: str, selected: pd.DataFrame, risk_codes: list[str], defensive_codes: list[str], args: argparse.Namespace) -> dict[str, object]:
    prices = fetch_histories(selected, years=args.years)
    prices = prices.dropna(how="any")
    prices = prices.loc[prices.index >= pd.Timestamp(args.start_date)]
    result, trades = run_custom_threshold_dual_with_overheat(
        prices=prices,
        selected=selected,
        risk_codes=risk_codes,
        defensive_codes=defensive_codes,
        lookback=args.lookback,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    if name == "base_pool":
        result = apply_official_baseline_nav_anchor(result)
    summary = summarize(result, trades)
    summary["pool"] = name
    summary["latest_holding"] = result["holding"].iloc[-1]
    summary["latest_signal"] = result["signal"].iloc[-1]
    summary["latest_exposure"] = result["exposure"].iloc[-1]
    return summary


def main() -> int:
    args = parse_args()
    candidate = CANDIDATES[args.candidate]
    candidate_code = str(candidate["code"])

    base_pool = load_fixed_etf_pool()
    candidate_pool = pd.concat([base_pool, pd.DataFrame([candidate])], ignore_index=True)

    rows = [
        run_case("base_pool", base_pool, BASE_RISK_CODES, BASE_DEFENSIVE_CODES, args),
        run_case(f"plus_{args.candidate}", candidate_pool, BASE_RISK_CODES + [candidate_code], BASE_DEFENSIVE_CODES, args),
    ]
    summary = pd.DataFrame(rows)
    base = summary.loc[summary["pool"] == "base_pool"].iloc[0]
    challenger = summary.loc[summary["pool"] == f"plus_{args.candidate}"].iloc[0]
    diff = pd.DataFrame(
        [
            {
                "metric": "total_return",
                "base_pool": base["total_return"],
                "candidate_pool": challenger["total_return"],
                "diff": challenger["total_return"] - base["total_return"],
            },
            {
                "metric": "annualized_return",
                "base_pool": base["annualized_return"],
                "candidate_pool": challenger["annualized_return"],
                "diff": challenger["annualized_return"] - base["annualized_return"],
            },
            {
                "metric": "sharpe_rf0",
                "base_pool": base["sharpe_rf0"],
                "candidate_pool": challenger["sharpe_rf0"],
                "diff": challenger["sharpe_rf0"] - base["sharpe_rf0"],
            },
            {
                "metric": "max_drawdown",
                "base_pool": base["max_drawdown"],
                "candidate_pool": challenger["max_drawdown"],
                "diff": challenger["max_drawdown"] - base["max_drawdown"],
            },
            {
                "metric": "trade_count",
                "base_pool": base["trade_count"],
                "candidate_pool": challenger["trade_count"],
                "diff": challenger["trade_count"] - base["trade_count"],
            },
        ]
    )
    print("=== summary ===")
    print(summary.to_csv(index=False))
    print("=== diff ===")
    print(diff.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
